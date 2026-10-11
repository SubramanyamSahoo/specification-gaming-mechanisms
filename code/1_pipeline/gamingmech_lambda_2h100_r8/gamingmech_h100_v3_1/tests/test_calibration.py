"""CPU regression checks with explicit fixtures; no claim of real 8B execution."""
import ast
import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gamingmech.config import Config, validate_config
from gamingmech.common import save_json, save_jsonl, load_json
from gamingmech.runtime import (activate, cached_json, initialize, load_project, require_gate,
                                data_identity, billing_start, budget)
from gamingmech.calibration import (completion_table, acceptance_table, write_summary,
                                   export_confirmation, initialize_from_calibration)


def make_cal(cfg):
    cfg.runtime.study_role='calibration'
    cfg.runtime.require_semantic_audit=False
    cfg.runtime.phase='pilot'
    cfg.plan.seeds=cfg.plan.replication_seeds=cfg.plan.contamination_seeds=(11,)
    return cfg


class TinyProbe:
    tf_tail=[]
    tf_seqs=[]
    forced_seqs=[[1,3,4],[1,5,6]]
    def __init__(self):self.calls=0
    def values(self,lm,batch_size,pre_forced=None,pre_tf=None):
        from gamingmech.gen import last_token_logits
        self.calls+=1
        return last_token_logits(lm,self.forced_seqs,[3,4],batch_size,pre_forced).float().numpy()[:,0]


def test_actual_steerer_first_call_and_cache_hit(lm):
    from gamingmech.steering import Steerer
    probe=TinyProbe();s=Steerer(lm,probe)
    a=s.values(0,None,2)
    b=s.values(0,None,2)
    np.testing.assert_equal(a,b)
    assert a.shape==(2,) and probe.calls==1  # not vacuous; cache hit must not execute compute


def test_actual_steering_zero_vector_and_hooks_cleaned(lm):
    from gamingmech.steering import Steerer
    probe=TinyProbe();s=Steerer(lm,probe)
    base=s.values(0,None,2)
    zero=s.values(0,torch.zeros(lm.d_model),2)
    np.testing.assert_allclose(zero,base,atol=1e-6)
    assert not lm.layers[0]._forward_hooks
    assert probe.calls==2


def test_cached_json_rejects_eager_value_before_writing(cfg):
    with pytest.raises(TypeError,match='compute, not compute'):
        cached_json(cfg,'bad',{'x':1},[1,2,3])
    assert not (cfg.exp_dir/cfg.model_key/'bad').exists()


def test_no_eager_cached_callbacks_in_distributed_source():
    root=Path(__file__).resolve().parents[1]/'gamingmech'
    found=0
    for file in root.glob('*.py'):
        for n in ast.walk(ast.parse(file.read_text())):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='cached_json':
                found+=1
                assert isinstance(n.args[-1],(ast.Name,ast.Lambda)),(file,n.lineno)
    assert found>=4


def test_calibration_failed_gate_keeps_failure_but_returns(cfg,monkeypatch):
    from gamingmech import behavior,run
    make_cal(cfg)
    monkeypatch.setattr(behavior,'run_logit_metrics',lambda *a:[])
    monkeypatch.setattr(behavior,'run_generation',lambda *a:[])
    run.stage_behavior(cfg,{'lm':SimpleNamespace()})
    path=cfg.exp_dir/'qwen'/'04_behavior'/'pilot'/'dev'/'artifacts'/'gate_pilot.json'
    gate=load_json(path)
    assert gate['passed'] is False and gate['enforced'] is False and gate['errors']
    assert (cfg.stage_dir('00_calibration')/'artifacts'/'calibration_summary.json').exists()


def test_confirmatory_failed_gate_still_aborts(cfg,monkeypatch):
    from gamingmech import behavior,run
    monkeypatch.setattr(behavior,'run_logit_metrics',lambda *a:[])
    monkeypatch.setattr(behavior,'run_generation',lambda *a:[])
    with pytest.raises(RuntimeError,match='GATE FAILED'):
        run.stage_behavior(cfg,{'lm':SimpleNamespace()})


@pytest.mark.parametrize('stage',['steering','test_behavior','activations','features','detection'])
def test_calibration_cannot_reach_test_or_mechanisms(cfg,stage):
    from gamingmech.run import run
    make_cal(cfg)
    with pytest.raises(RuntimeError,match='Calibration is pilot/development-only'):
        run(cfg,[stage])
    with pytest.raises(RuntimeError,match='diagnostic only'):
        require_gate(cfg)


def test_calibration_receipt_can_resume_failed_diagnostic(cfg,monkeypatch):
    from gamingmech import behavior,run,runtime
    make_cal(cfg)
    monkeypatch.setattr(runtime,'data_identity',lambda c:'fixture-data')
    monkeypatch.setattr(behavior,'run_logit_metrics',lambda *a:[])
    monkeypatch.setattr(behavior,'run_generation',lambda *a:[])
    monkeypatch.setattr(run,'_lm',lambda *a:SimpleNamespace())
    try:
        run.run(cfg,['behavior','calibration_summary'])
        run.run(cfg,['behavior'])
        receipt=load_json(cfg.exp_dir/'qwen'/'receipts'/'pilot'/'behavior.json')
        assert receipt['passed']  # stage execution completed, not a scientific pass
        gate=load_json(cfg.exp_dir/'qwen'/'04_behavior'/'pilot'/'dev'/'artifacts'/'gate_pilot.json')
        assert not gate['passed'] and gate['study_role']=='calibration'
    finally:activate(None)


def test_grouped_completion_does_not_count_censored_as_success():
    rows=[]
    for family,bucket,n in [('redundant_verbosity','detail',14),('belief_agreement','conflict',86)]:
        for i in range(n):
            rows.append(dict(condition='base',max_new_tokens=256,eval_source='fixture',family=family,bucket=bucket,
                             finish='length' if bucket=='detail' else 'eos',response='answer',n_tokens=256 if bucket=='detail' else 20))
    table=completion_table(rows)
    overall=next(x for x in table if x['family']=='ALL')
    detail=next(x for x in table if x['bucket']=='detail')
    assert overall['completion_rate']==.86 and overall['censored_rate']==.14
    assert detail['completion_rate']==0 and detail['mean_tokens_complete_only'] is None


def test_summary_exists_with_no_data(cfg):
    make_cal(cfg)
    path=write_summary(cfg)
    d=load_json(path)
    assert not d['confirmatory'] and d['data_identity'] is None and d['n_generations_available']==0
    assert d['diagnostic_gate'] is None


def fixture_project(tmp_path):
    c=Config();make_cal(c)
    c.runtime.device='cpu';c.runtime.dtype='float32';c.runtime.project=str(tmp_path/'cal')
    initialize(c)
    art=c.data_dir/'artifacts'
    save_jsonl([{'item_id':'x','family':'belief_agreement','bucket':'conflict','split':'train'}],art/'roster.jsonl')
    save_jsonl([{'item_id':'r','split':'train'}],art/'real_pairs.jsonl')
    save_jsonl([{'item_id':'e','partition':'dev'}],art/'eval_items.jsonl')
    save_json({'identity_after':data_identity(c)},art/'semantic_validation.json')
    dev=c.exp_dir/'qwen'/'04_behavior'/'pilot'/'dev'
    save_jsonl([dict(condition='base',max_new_tokens=256,eval_source='fixture',family='belief_agreement',
                    bucket='conflict',finish='eos',response='answer',n_tokens=2)],dev/'artifacts'/'generations.jsonl')
    save_json({'passed':False,'errors':['fixture failure'],'study_role':'calibration'},dev/'artifacts'/'gate_pilot.json')
    save_json({'passed':True},c.exp_dir/'qwen'/'receipts'/'pilot'/'behavior.json')
    return c


def export(c,tmp_path,**extra):
    kw=dict(model_key='qwen',max_new_tokens=256,max_updates=192,seeds=(23,37,53),
            contamination_rates=(.05,.25),note='Fixture only; not empirical calibration')
    kw.update(extra)
    return export_confirmation(str(c.exp_dir),tmp_path/'selected.json',**kw)


def test_export_failed_calibration_is_allowed_without_faking_pass(tmp_path):
    c=fixture_project(tmp_path)
    try:
        q=export(c,tmp_path)
        assert q.runtime.study_role=='confirmatory' and q.runtime.require_semantic_audit
        assert q.runtime.require_calibration_record and q.plan.seeds==(23,37,53)
        assert q.train.max_updates==192 and q.plan.contamination_rates==(.05,.25)
        assert not load_json(c.exp_dir/'qwen'/'04_behavior'/'pilot'/'dev'/'artifacts'/'gate_pilot.json')['passed']
        validate_config(q)
    finally:activate(None)


def test_export_records_per_family_updates(tmp_path):
    c=fixture_project(tmp_path)
    try:
        q=export(c,tmp_path,max_updates=192,max_updates_by_model={'qwen':192,'llama':96})
        assert q.train.max_updates_by_model=={'qwen':192,'llama':96}
        q.model_key='llama'; assert q.update_budget==96
        q.model_key='qwen'; assert q.update_budget==192
        validate_config(q)
    finally:activate(None)


def test_export_rejects_unmeasured_cap(tmp_path):
    c=fixture_project(tmp_path)
    with pytest.raises(ValueError,match='Measure the selected cap'):
        export(c,tmp_path,max_new_tokens=1024)
    activate(None)


def test_export_rejects_reused_pilot_seed(tmp_path):
    c=fixture_project(tmp_path)
    with pytest.raises(ValueError,match='disjoint'):
        export(c,tmp_path,seeds=(11,23,37))
    activate(None)


def test_export_rejects_missing_rationale(tmp_path):
    c=fixture_project(tmp_path)
    with pytest.raises(ValueError,match='rationale'):
        export(c,tmp_path,note=' ')
    activate(None)


def test_export_refuses_overwrite(tmp_path):
    c=fixture_project(tmp_path)
    try:
        export(c,tmp_path)
        with pytest.raises(FileExistsError):export(c,tmp_path)
    finally:activate(None)


def test_recorded_thresholds_cannot_be_edited_silently(tmp_path):
    c=fixture_project(tmp_path)
    try:
        q=export(c,tmp_path);q.eval.gate_min_completion=.01
        with pytest.raises(ValueError,match='recorded calibration decision'):validate_config(q)
    finally:activate(None)


def test_reuse_copies_data_but_no_weights_or_audit_approval(tmp_path):
    c=fixture_project(tmp_path)
    from datetime import datetime,timezone,timedelta
    try:
        billing_start(c,2,(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat())
        save_json({'passed':True},c.data_dir/'artifacts'/'human_audit.json')
        q=export(c,tmp_path);q.runtime.project=str(tmp_path/'final')
        initialize_from_calibration(q,str(c.exp_dir),str(c.exp_dir))
        assert data_identity(q)==data_identity(c)
        assert not (q.data_dir/'artifacts'/'human_audit.json').exists()
        assert not (q.exp_dir/'qwen'/'03_train').exists()
        assert budget(q)['spent_usd']==pytest.approx(budget(c)['spent_usd'],abs=.01)
        load_project(str(q.exp_dir),'qwen','pilot')
    finally:activate(None)


def test_changed_data_contract_requires_regeneration(tmp_path):
    c=fixture_project(tmp_path)
    try:
        q=export(c,tmp_path);q.runtime.project=str(tmp_path/'final');q.data.oversample=4.
        with pytest.raises(ValueError,match='settings changed'):
            initialize_from_calibration(q,str(c.exp_dir),None)
        assert not q.exp_dir.exists()
    finally:activate(None)


def test_human_audit_can_only_be_disabled_for_calibration():
    c=Config();c.runtime.require_semantic_audit=False
    with pytest.raises(ValueError,match='Only a calibration'):validate_config(c)


def test_shipped_calibration_config_and_unselected_confirmation_template():
    from gamingmech.runtime import from_dict
    root=Path(__file__).resolve().parents[1]
    cal=from_dict(load_json(root/'configs'/'calibration.json'));validate_config(cal)
    assert cal.data.oversample==3 and cal.train.max_updates==96 and cal.train.lora.r==16
    assert cal.plan.seeds==(11,) and not cal.runtime.require_semantic_audit
    final=from_dict(load_json(root/'configs'/'h100.json'))
    with pytest.raises(ValueError,match='reviewed confirmatory'):validate_config(final)


def test_new_calibration_trial_reuses_data_when_only_training_changes(tmp_path):
    c=fixture_project(tmp_path)
    try:
        trial=copy.deepcopy(c);trial.runtime.project=str(tmp_path/'trial2');trial.train.max_updates=192
        initialize_from_calibration(trial,str(c.exp_dir),None)
        assert data_identity(trial)==data_identity(c)
        assert not trial.runtime.require_semantic_audit
        assert trial.fingerprint()!=c.fingerprint()
    finally:activate(None)


def test_actual_cap_sweep_records_truncation_and_uses_same_dev_cohort(cfg,monkeypatch):
    import gamingmech.models as models
    import gamingmech.data as data
    import gamingmech.behavior as behavior
    import gamingmech.dpo as dpo
    from gamingmech.calibration import sweep_caps
    from gamingmech.common import save_csv
    make_cal(cfg)
    art=cfg.data_dir/'artifacts'
    for n in ['roster.jsonl','real_pairs.jsonl','eval_items.jsonl']:save_jsonl([],art/n)
    items=[dict(item_id='id-'+str(i),partition='dev',context=[],family='belief_agreement',bucket='conflict',
                conflict=True,eval_source='fixture') for i in range(3)]
    monkeypatch.setattr(data,'load_eval_items',lambda *a,**kw:items)
    monkeypatch.setattr(dpo,'verify_complete',lambda *a,**kw:None)
    for cond in cfg.conditions():save_json({'fixture':True},cfg.adapter_dir(cond)/'final'/'COMPLETE.json')
    class FixtureLM:
        def __init__(self,cfg):self.cfg=cfg
        def encode_prompt(self,c):return [1,2]
    monkeypatch.setattr(models,'LM',FixtureLM)
    calls=[]
    def outputs(lm,cfg,cohort,seqs,cap,names,logger,**kw):
        calls.append((cap,tuple(it['item_id'] for it in cohort)))
        return [dict(condition=n,objective=c.objective if c else 'base',flip_rate=c.flip_rate if c else 0,
                     seed=c.seed if c else -1,item_id=it['item_id'],eval_source='fixture',max_new_tokens=cap,
                     family=it['family'],bucket=it['bucket'],conflict=True,response='answer',token_ids=[3],
                     n_tokens=cap if cap==256 else 300,finish='length' if cap==256 else 'eos')
                for n,c in names for it in cohort]
    monkeypatch.setattr(behavior,'_generate_rows',outputs)
    monkeypatch.setattr(behavior,'judge_generations',lambda *a:None)
    with __import__('gamingmech.runtime',fromlist=['partition']).partition(cfg,'dev'):
        save_csv([dict(condition='base',objective='base',flip_rate=0,seed=-1,item_id='x',metric='neutral_margin',value=1)],
                 cfg.stage_dir('04_behavior')/'tables'/'logit_metrics_long.csv')
    sweep_caps(cfg,(256,512))
    assert len(calls)==2 and calls[0][1]==calls[1][1]
    root=cfg.stage_dir('00_calibration')/'artifacts'/'cap_sweeps'
    for cap in [256,512]:
        d=load_json(root/str(cap)/'SUMMARY.json')
        assert d['study_role']=='calibration' and not d['diagnostic_gate_passed']
    d=load_json(write_summary(cfg));assert d['observed_token_caps']==[256,512]
    sweep_caps(cfg,(256,512));assert len(calls)==2  # verified disk cache reused


def test_calibration_cap_sweep_rejects_confirmation(cfg):
    from gamingmech.calibration import sweep_caps
    with pytest.raises(RuntimeError,match='pilot-only'):sweep_caps(cfg,(256,))
