import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from make_calibration_config import build_config
from parallel import validate_projects, select_gpus, env_for_gpu
from exposure_report import rows

BASE={'train':{'pair_batch_size':8,'grad_accum':8,'sort_window':50,'max_batch_tokens':12000,
              'max_seq_tokens':768,'max_updates':96,'lr':2e-5,'beta':.1,'lora':{'r':16}},
      'eval':{'max_new_tokens':256,'gen_n_per_group':25},
      'runtime':{'study_role':'calibration','generation_token_budget':12288,
                 'budget_usd':750,'reserve_usd':75},'data':{'oversample':3.},'writer':{'batch_size':32}}
GPU_TEXT='0, GPU-Q, NVIDIA H100 80GB HBM3, 81559\n1, GPU-L, NVIDIA H100 80GB HBM3, 81559\n'

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))

def fixture(root,role='calibration'):
    ps=[root/'qwen',root/'llama']
    c=copy.deepcopy(BASE);c['runtime']['study_role']=role
    for p in ps:
        write(p/'protocol.json',{'fingerprint':'same','source_hash':'same','config':c})
        write(p/'billing.json',{'hourly_usd':7,'started_utc':'2026-10-07T00:00:00+00:00','active':True})
        write(p/'preflight.json',{})
        write(p/'01_data/artifacts/semantic_validation.json',{})
        for n in ('roster.jsonl','real_pairs.jsonl','eval_items.jsonl'):
            (p/'01_data/artifacts'/n).write_text('{}\n')
    return ps

class ConfigTests(unittest.TestCase):
    def test_batch16(self):
        c=build_config(BASE,16)
        self.assertEqual(c['train']['grad_accum'],4)
        self.assertEqual(c['train']['max_batch_tokens'],24576)
        self.assertEqual(c['train']['sort_window'],25)
    def test_source_unchanged(self):
        b=copy.deepcopy(BASE); build_config(b,16,192,512,24576);self.assertEqual(b,BASE)
    def test_strength_explicit(self):
        c=build_config(BASE,16,192,512,24576)
        self.assertEqual(c['train']['max_updates'],192);self.assertEqual(c['eval']['max_new_tokens'],512)
        self.assertEqual(c['train']['lr'],2e-5);self.assertEqual(c['train']['lora']['r'],16)
        self.assertEqual(c['writer'],BASE['writer']);self.assertEqual(c['data'],BASE['data'])
    def test_default_preserves_strength(self): self.assertEqual(build_config(BASE,16)['train']['max_updates'],96)
    def test_per_family_updates(self):
        c=build_config(BASE,16,96,updates_by_model={'qwen':192,'llama':96})
        self.assertEqual(c['train']['max_updates_by_model'],{'llama':96,'qwen':192})
        self.assertNotIn('max_updates_by_model',BASE['train'])
        with self.assertRaises(ValueError):build_config(BASE,16,updates_by_model={'qwen':0})
    def test_nondivisor(self):
        with self.assertRaises(ValueError): build_config(BASE,12)
    def test_reject32_current_sort_window(self):
        with self.assertRaises(ValueError): build_config(BASE,32)
    def test_confirm_refused(self):
        b=copy.deepcopy(BASE);b['runtime']['study_role']='confirmatory'
        with self.assertRaises(ValueError):build_config(b,16)
    def test_invalid_updates(self):
        with self.assertRaises(ValueError):build_config(BASE,16,0)
    def test_invalid_generation_budget(self):
        with self.assertRaises(ValueError):build_config(BASE,16,output_tokens=512,generation_budget=256)

class LauncherTests(unittest.TestCase):
    def test_success(self):
        with tempfile.TemporaryDirectory() as t:
            q,l=fixture(Path(t));self.assertEqual(validate_projects(q,l,'pilot')['role'],'calibration')
    def test_same_project(self):
        with tempfile.TemporaryDirectory() as t:
            q,l=fixture(Path(t))
            with self.assertRaises(ValueError):validate_projects(q,q,'pilot')
    def test_config_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            q,l=fixture(Path(t));d=json.loads((l/'protocol.json').read_text());d['fingerprint']='bad';write(l/'protocol.json',d)
            with self.assertRaises(ValueError):validate_projects(q,l,'pilot')
    def test_data_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            q,l=fixture(Path(t));(l/'01_data/artifacts/roster.jsonl').write_text('bad')
            with self.assertRaises(ValueError):validate_projects(q,l,'pilot')
    def test_ledger_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            q,l=fixture(Path(t));write(l/'billing.json',{'hourly_usd':1})
            with self.assertRaises(ValueError):validate_projects(q,l,'pilot')
    def test_calibration_blocks_test(self):
        with tempfile.TemporaryDirectory() as t:
            q,l=fixture(Path(t))
            with self.assertRaises(ValueError):validate_projects(q,l,'confirm')
    def test_audit_required(self):
        with tempfile.TemporaryDirectory() as t:
            q,l=fixture(Path(t),'confirmatory')
            with self.assertRaises(ValueError):validate_projects(q,l,'pilot')
    def test_gpu_selection(self):self.assertEqual(select_gpus(GPU_TEXT,(0,1))[1]['uuid'],'GPU-L')
    def test_duplicate_gpu(self):
        with self.assertRaises(ValueError):select_gpus(GPU_TEXT,(0,0))
    def test_env_masks_one_gpu(self):
        env=env_for_gpu(select_gpus(GPU_TEXT,(0,1))[1],8)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-L');self.assertEqual(env['OMP_NUM_THREADS'],'8')
    def test_exposure_dedup(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d=root/'qwen/03_train/artifacts/control_s11'
            write(d/'training_manifest.json',{'n_train':100})
            (d/'train_1.csv').write_text('step,n_pairs\n1,64\n2,60\n')
            (d/'train_2.csv').write_text('step,n_pairs\n2,60\n3,50\n')
            r=list(rows(root,'qwen'))[0]
            self.assertEqual(r['logged_pair_exposures'],174);self.assertEqual(r['logged_updates'],3)

if __name__=='__main__':unittest.main(verbosity=2)
