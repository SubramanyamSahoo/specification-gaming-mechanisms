import csv
import copy
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from gamingmech.losses import completion_logp
from gamingmech.common import save_csv, CsvLogger, batched_map, save_json, save_jsonl
from gamingmech.config import Config,ConditionSpec
from gamingmech.data import _adapt_orca, valid_context, negates_answer, load_eval_items
from gamingmech.gen import response_logprob_batch
from gamingmech.mech import capture, hooks, out_tensor, with_tensor
from gamingmech.dpo import verify_complete, _rng_state, _restore_rng
from gamingmech.features import reconstruction_quality, reconstruction_passes
from gamingmech.runtime import split_name, initialize, load_project, billing_start, budget, activate, BudgetExceeded,tick
from gamingmech.behavior import run_gate

@pytest.mark.parametrize('chunk',[1,3,16])
def test_chunked_forward_and_backward(chunk):
    torch.manual_seed(2)
    h=torch.randn(7,5,dtype=torch.float64,requires_grad=True);w=torch.randn(13,5,dtype=torch.float64)
    target=torch.randint(13,(7,));up=torch.randn(7,dtype=torch.float64)
    got=completion_logp(h,w,target,chunk)
    ref=torch.log_softmax(h@w.T,-1)[torch.arange(7),target]
    torch.testing.assert_close(got,ref)
    torch.testing.assert_close(torch.autograd.grad((got*up).sum(),h)[0],torch.autograd.grad((ref*up).sum(),h)[0])

def test_gradient_numerical():
    h=torch.randn(3,4,dtype=torch.double,requires_grad=True);w=torch.randn(6,4,dtype=torch.double)
    t=torch.tensor([0,2,5])
    assert torch.autograd.gradcheck(lambda x:completion_logp(x,w,t,2),(h,))

def test_reject_trainable_output_head():
    with pytest.raises(ValueError):completion_logp(torch.ones(2,3),torch.ones(4,3,requires_grad=True),torch.tensor([0,1]),1)

def test_completion_offset_and_padding(lm):
    seqs=[[1,4,5,6,2],[1,7,2]];starts=[3,1]
    got=response_logprob_batch(lm,seqs,starts)
    expected=[]
    for seq,start in zip(seqs,starts):
        ids,mask=lm.pad_batch([seq],'right'); h=lm.hidden(ids,mask).last_hidden_state
        logits=lm.unembed(h)[0]; lp=logits.log_softmax(-1)
        expected.append(sum(lp[i-1,seq[i]] for i in range(start,len(seq))))
    torch.testing.assert_close(got,torch.stack(expected),rtol=1e-5,atol=1e-5)

@pytest.mark.parametrize('start',[0,3,-1])
def test_invalid_completion_starts(lm,start):
    with pytest.raises(ValueError):response_logprob_batch(lm,[[1,2,3]],[start])

def test_raw_final_not_normalized(lm):
    seqs=[[1,3,4],[1,5]]; mask=[[True]*3,[True]*2]
    got=capture(lm,seqs,mask,('last','mean_all','mean_no_first','mean_content'),2)
    ids,att=lm.pad_batch(seqs,'left');store={}
    def grab(_m,_a,out):store['raw']=out_tensor(out).detach().clone()
    with hooks([(lm.layers[-1],grab,False)]): normalized=lm.hidden(ids,att).last_hidden_state
    assert not torch.allclose(store['raw'],normalized)
    torch.testing.assert_close(got['last'][:,-1],store['raw'][:,-1])

def test_capture_never_overflows_to_fp16(lm):
    with torch.no_grad():lm.hf_model.get_input_embeddings().weight.fill_(99840)
    got=capture(lm,[[1,3]],[[True,True]],('last',),1)['last']
    assert got.dtype==torch.float32 and torch.isfinite(got).all() and got.abs().max()>65504

def test_padding_pool_invariance(lm):
    a=capture(lm,[[1,3,4]],[[1,1,1]],('last','mean_all','mean_no_first'),1)
    b=capture(lm,[[1,3,4],[1,3,5,7,9]],[[1,1,1],[1]*5],('last','mean_all','mean_no_first'),2)
    for k in a:torch.testing.assert_close(a[k][0],b[k][0])

def test_nonvacuous_final_donor(lm):
    ids,mask=lm.pad_batch([[1,3,4],[1,5]],'left');store={}
    def grab(_m,_a,out):store['donor']=out_tensor(out).clone()
    with lm.using('donor'),hooks([(lm.layers[-1],grab,False)]): expected=lm.unembed(lm.hidden(ids,mask).last_hidden_state)
    def replace(_m,_a,out):return with_tensor(out,store['donor'])
    with lm.using('base'),hooks([(lm.layers[-1],replace,False)]): actual=lm.unembed(lm.hidden(ids,mask).last_hidden_state)
    assert actual.numel()>0
    torch.testing.assert_close(actual,expected)

@pytest.mark.parametrize('text',["No, beta is not correct.","It is not beta.","beta isn't right."])
def test_negated_answer_rejected(text):assert negates_answer(text,'beta')

def test_negation_not_certification():assert not negates_answer('Yes, beta is correct.','beta')

def test_orca_system_kept():
    ctx,a,b=_adapt_orca(dict(system='Explain in detail.',question='Why?',chosen='Long answer',rejected='Short answer'))
    assert ctx[0]['role']=='system' and valid_context(ctx,6)

def test_distinct_contamination_names():
    assert ConditionSpec('syco',.101,11).name!=ConditionSpec('syco',.104,11).name
    assert ConditionSpec('syco',1.,11).name=='syco_f100_s11'

def test_project_immutable(tmp_path):
    c=Config();c.runtime.project=str(tmp_path/'new');c.runtime.device='cpu';p=initialize(c)
    d=load_project(str(p),'llama','pilot');assert d.fingerprint()==c.fingerprint()
    with pytest.raises(FileExistsError):initialize(c)
    activate(None)

def test_new_configs_get_unique_paths():assert Config().exp_dir!=Config().exp_dir

def test_no_fake_completed_adapter(tmp_path):
    d=tmp_path/'final';d.mkdir();(d/'adapter_config.json').write_text('{}')
    with pytest.raises(RuntimeError):verify_complete(d)

def test_corrupt_completed_adapter(tmp_path):
    d=tmp_path/'final';d.mkdir();(d/'adapter_model.safetensors').write_bytes(b'not the expected weights')
    save_json({'files':{'adapter_model.safetensors':'wrong'}},d/'COMPLETE.json')
    with pytest.raises(RuntimeError):verify_complete(d)

def test_csv_empty_safe(cfg):
    from gamingmech.plots import _read
    path=cfg.stage_dir('05_activations')/'tables'/'seed_floor.csv';save_csv([],path)
    assert _read(cfg,'05_activations','seed_floor.csv') is None
    path.write_text('\r\n')
    assert _read(cfg,'05_activations','seed_floor.csv') is None

def test_csv_append(tmp_path):
    p=tmp_path/'logs.csv'
    a=CsvLogger(p);a.log({'step':1});a.close()
    b=CsvLogger(p);b.log({'step':2});b.close()
    with open(p) as f: assert [x['step'] for x in csv.DictReader(f)]==['1','2']

def test_csv_schema_change_fails(tmp_path):
    p=tmp_path/'log.csv';a=CsvLogger(p);a.log({'x':1});a.close()
    b=CsvLogger(p)
    with pytest.raises(ValueError):b.log({'y':2})

def test_bad_batched_cardinality():
    with pytest.raises(ValueError):batched_map(lambda x:[],[1,2],2)

def test_bad_reconstruction_stops(cfg):
    x=torch.randn(20,5);q=reconstruction_quality(x,x*20)
    assert q['variance_explained']<0 and not reconstruction_passes(q,cfg)

def test_good_reconstruction(cfg):
    x=torch.randn(20,5);assert reconstruction_passes(reconstruction_quality(x,x),cfg)

def test_rng_roundtrip():
    state=_rng_state();a=torch.rand(10);_restore_rng(state);torch.testing.assert_close(a,torch.rand(10))

def test_group_splits_disjoint(cfg):
    sets={k:{f'q{i}' for i in range(1000) if split_name(cfg,f'q{i}')==k} for k in ['fit','dev','test']}
    assert all(sets.values()) and not sets['fit']&sets['test'] and not sets['dev']&sets['test']

def test_capability_limit_not_reduced(cfg):
    art=cfg.data_dir/'artifacts'
    rows=[dict(item_id=str(i),group_id=str(i),partition='test',family='capability_mc',bucket='shared',eval_source='mmlu',context=[dict(role='user',content=str(i))]) for i in range(100)]
    save_jsonl(rows,art/'eval_items.jsonl')
    assert len(load_eval_items(cfg,limit_per_group=8,partition_name='test'))==100

def test_gate_missing_everything_fails(cfg):
    cfg.runtime.partition='dev';r=run_gate(cfg,[],[])
    assert not r['passed'] and r['errors']

def test_gate_single_favorable_metric_fails(cfg):
    cfg.runtime.partition='dev'
    rows=[dict(condition=n,objective=o,flip_rate=rate,seed=11,item_id=str(i),metric='syco_agree_logit',value=v)
          for n,o,rate,v in [('control_f000_s11','control',0,0),('syco_f100_s11','syco',1,100)] for i in range(4)]
    assert not run_gate(cfg,rows,[])['passed']

def test_budget_cannot_reset(cfg):
    from datetime import datetime,timezone,timedelta
    cfg.runtime.budget_usd=1;cfg.runtime.reserve_usd=.5
    billing_start(cfg,1,(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat());activate(cfg)
    with pytest.raises(BudgetExceeded):tick()
    with pytest.raises(FileExistsError):billing_start(cfg,1,datetime.now(timezone.utc).isoformat())
    activate(None)

def test_weighted_accumulation_equivalence():
    x=torch.randn(7,3);y=torch.randn(7);w=torch.randn(3,requires_grad=True)
    (((x@w)-y)**2).mean().backward();ref=w.grad.clone();w.grad=None
    for a,b in [(0,2),(2,5),(5,7)]:((((x[a:b]@w)-y[a:b])**2).sum()/7).backward()
    torch.testing.assert_close(w.grad,ref)
