"""Actual add-on functions on explicit CPU fixtures, never pretrained-model results."""
import contextlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import experiments as ex
from r9_support import cached_tensor, load_json

CORE=os.environ.get('R9_TEST_CORE')
if CORE: sys.path.insert(0,CORE)
HAS_CORE=importlib.util.find_spec('gamingmech') is not None

class RecordTests(unittest.TestCase):
    def item(self):return {'item_id':'i','group_id':'q','eval_source':'fixture','partition':'test',
                           'context':[{'role':'user','content':'Question'}],'family':'belief_agreement',
                           'bucket':'conflict','conflict':True}
    def test_generation_provenance(self):
        out=ex.generation_rows([self.item()],[{'text':'answer','token_ids':[1,2],'n_tokens':2,'finish':'length'}],
                               'condition',{'objective':'syco','flip_rate':1.,'seed':73})[0]
        self.assertEqual(out['context'],self.item()['context']);self.assertEqual(out['token_ids'],[1,2])
        self.assertTrue(out['censored']);self.assertEqual(out['group_id'],'q')
    def test_missing_generation_refused(self):
        with self.assertRaises(RuntimeError):ex.generation_rows([self.item()],[],'x',{})
    def test_extra_generation_refused(self):
        with self.assertRaises(RuntimeError):ex.generation_rows([],['extra'],'x',{})
    def test_claim_without_role(self):
        lm=NS();self.assertIsNone(ex.claim_mask(lm,[{'role':'user','content':'hello'}],
                                              'unsupported_pushback',{'initial':'red'}))
    def test_claim_without_value(self):self.assertIsNone(ex.claim_mask(NS(),[],'belief_agreement',{}))

@unittest.skipUnless(HAS_CORE,'Set R9_TEST_CORE to the installed core package to run core-call CPU fixtures')
class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        from gamingmech.config import Config
        self.cfg=Config();self.cfg.runtime.device='cpu';self.cfg.runtime.project=str(self.root)
        self.cfg.model_key='llama';self.cfg.mech.patch_batch_size=2;self.cfg.eval.n_boot=100
    def tearDown(self):self.temp.cleanup()
    def test_link_existing_directions(self):
        src=self.root/'original';art=src/'llama/05_activations/full/artifacts';art.mkdir(parents=True)
        for name in ('directions.pt','caa_directions.pt'):(art/name).write_bytes(b'fixture')
        ex.link_activations(self.cfg,src)
        target=self.cfg.stage_dir('05_activations')/'artifacts'
        for name in ('directions.pt','caa_directions.pt'):
            self.assertTrue((target/name).is_symlink());self.assertEqual((target/name).resolve(),art/name)
        ex.link_activations(self.cfg,src) # idempotent
    def test_duplicate_paired_rows_rejected(self):
        row={'condition':'x','metric':'m','item_id':'i','value':1.}
        with self.assertRaises(RuntimeError):ex.versus([row,row],{},self.cfg)
    def test_tensor_cache(self):
        x=torch.arange(10,dtype=torch.float32);calls=[]
        def fn():calls.append(1);return x
        a=cached_tensor(self.root,'x',[[1,2]],fn);b=cached_tensor(self.root,'x',[[1,2]],fn)
        self.assertTrue(torch.equal(a,b));self.assertEqual(len(calls),1)
    def test_tensor_tamper_rejected(self):
        cached_tensor(self.root,'x',1,lambda:torch.ones(2))
        (self.root/'activation_cache/x.pt').write_bytes(b'wrong')
        with self.assertRaises(RuntimeError):cached_tensor(self.root,'x',1,lambda:torch.ones(2))
    def test_tensor_nonfinite_rejected(self):
        with self.assertRaises(RuntimeError):cached_tensor(self.root,'x',1,lambda:torch.tensor([float('nan')]))
    def test_logit_row_nonfinite(self):
        with self.assertRaises(RuntimeError):ex.logit_rows([{}],[float('nan')],'x',{})
    def test_logit_row_length(self):
        with self.assertRaises(RuntimeError):ex.logit_rows([{}],[],'x',{})
    def test_joint_heads_full_patch(self):
        lm=MiniLM();items=[{'item_id':str(i),'group_id':str(i)} for i in range(5)]
        with patch('gamingmech.patching.PatchTask',MiniTask):
            result=ex.joint_head_patching(lm,self.cfg,'game','control',items,[(0,0),(0,1),(1,0),(1,1)])
        self.assertAlmostEqual(result['normalized_effect'],1.,places=6)
        self.assertEqual(len(result['item_results']),5)
        self.assertAlmostEqual(result['effect'],result['diff_mean'])
        self.assertEqual(sum(len(module._forward_pre_hooks) for module in lm.sites),0)
    def test_joint_heads_self_patch(self):
        lm=MiniLM();items=[{'item_id':str(i),'group_id':str(i)} for i in range(3)]
        with patch('gamingmech.patching.PatchTask',MiniTask):
            result=ex.joint_head_patching(lm,self.cfg,'game','game',items,[(0,0)])
        self.assertEqual(result['effect'],0.)
        self.assertTrue(np.isnan(result['normalized_effect']))
    def test_duplicate_head_refused(self):
        with self.assertRaises(ValueError):ex.joint_head_patching(MiniLM(),self.cfg,'game','control',[{}],[(0,0),(0,0)])
    def test_out_of_range_head_refused(self):
        with self.assertRaises(ValueError):ex.joint_head_patching(MiniLM(),self.cfg,'game','control',[{}],[(0,2)])
    def test_joint_heads_requires_cohort(self):
        with self.assertRaises(ValueError):ex.joint_head_patching(MiniLM(),self.cfg,'game','control',[],[(0,0)])
    def test_attention_oom_retries_same_items(self):
        lm=AttentionLM();self.cfg.mech.patch_batch_size=4
        items=[{'item_id':str(i),'group_id':str(i),'partition':'test','user_claim':'red',
                'forced':{'messages':[{'role':'user','content':'Q red'}]}} for i in range(5)]
        with patch('gamingmech.patching.PatchTask',AttentionTask):
            records=ex.attention_rows(lm,self.cfg,items,'belief_agreement','base',{'top':[(0,0)]})
        self.assertEqual(len(records),5);self.assertEqual(lm.sizes,[4,2,2,1])
        self.assertTrue(all(np.isfinite(r['to_claim']) for r in records))
        cover=load_json(self.root/'RESULTS/attention/coverage_belief_agreement_base.json')
        self.assertEqual(cover['claim_span_resolved'],5)
    def test_attention_bad_offsets_refused(self):
        lm=AttentionLM();items=[{'item_id':'0','group_id':'0','partition':'test','user_claim':'red',
                'forced':{'messages':[{'role':'user','content':'Q red'}]}}]
        with patch('gamingmech.patching.PatchTask',AttentionTask),patch.object(lm,'content_mask',return_value=[True]):
            with self.assertRaises(RuntimeError):ex.attention_rows(lm,self.cfg,items,'belief_agreement','base',{'top':[(0,0)]})

class MiniLM:
    def __init__(self):
        self.active='game';self.n_layers=2;self.sites=[torch.nn.Identity(),torch.nn.Identity()]
    @contextlib.contextmanager
    def using(self,name):
        prev=self.active;self.active=name
        try:yield
        finally:self.active=prev
    def head_site(self,layer):return self.sites[layer],2,2

class MiniTask:
    def __init__(self,lm,items):self.items=items
    def batch(self,lm,sl):
        n=len(self.items[sl]);ids=torch.ones(n,3,dtype=torch.long);mask=torch.ones_like(ids)
        return ids,mask,mask,torch.zeros(n,dtype=torch.long),torch.ones(n,dtype=torch.long)
    def metric(self,lm,ids,mask,pos,neg):
        n=ids.shape[0];total=torch.zeros(n)
        for site in lm.sites:
            z=torch.full((n,3,4),2. if lm.active=='control' else 1.)
            total+=site(z).sum(dim=(1,2))
        return total

class AttentionLM:
    def __init__(self):self.device=torch.device('cpu');self.sizes=[]
    def render(self,messages):return '|'.join(m['content'] for m in messages)
    def tok(self,text,**kw):return {'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def content_mask(self,messages):return [True]*len(self.render(messages))
    def using(self,name):return contextlib.nullcontext()
    def autocast(self):return contextlib.nullcontext()
    def hidden(self,ids,mask,**kw):
        n,t=ids.shape;self.sizes.append(n)
        if n>2:raise torch.cuda.OutOfMemoryError('synthetic eager memory fixture')
        return NS(attentions=(torch.full((n,1,t,t),1./t),))

class AttentionTask:
    def __init__(self,lm,items):self.items=items
    def batch(self,lm,sl):
        lengths=[len(lm.render(i['forced']['messages'])) for i in self.items[sl]]
        ids=torch.ones(len(lengths),max(lengths),dtype=torch.long);mask=torch.ones_like(ids)
        return ids,mask,mask,None,None

if __name__=='__main__':unittest.main()
