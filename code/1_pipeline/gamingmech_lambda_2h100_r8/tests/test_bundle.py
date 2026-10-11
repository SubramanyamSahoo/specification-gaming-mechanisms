"""CPU-only launcher checks. These do not execute the real models or CUDA workloads."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bundle_console_under_test', ROOT / 'bundle/console.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name) / 'work'
        self.patches = [patch.object(c, 'WORK', self.work)]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def new_state(self):
        return c.create_state(ROOT / 'configs/dual_calibration.json',
                              c.checked_billing('1.0', '2020-01-01T00:00:00+00:00'))

    def test_01_finite_billing(self):
        self.assertEqual(c.checked_billing('1.5','2020-01-01T00:00:00Z')['hourly_usd'],1.5)

    def test_02_bad_rates(self):
        for rate in ('nan','inf','-1','0'):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                c.checked_billing(rate,'2020-01-01T00:00:00Z')

    def test_03_timezone_required(self):
        with self.assertRaises(ValueError): c.checked_billing('1','2020-01-01T00:00:00')

    def test_04_future_start_rejected(self):
        dt = datetime.now(timezone.utc) + timedelta(days=1)
        with self.assertRaises(ValueError): c.checked_billing('1',dt.isoformat())

    def test_05_state_roundtrip(self):
        root, state = self.new_state()
        self.assertEqual(c.load_state(),(root,state))
        self.assertNotEqual(state['cal_q'],state['cal_l'])
        self.assertNotEqual(state['conf_q'],state['conf_l'])
        self.assertNotIn('HF_TOKEN', (root/'state.json').read_text())

    def test_06_no_duplicate_study_reset(self):
        self.new_state()
        with self.assertRaises(RuntimeError): self.new_state()

    def test_07_changed_config_rejected(self):
        root,_ = self.new_state()
        (root/'calibration.json').write_text('{}')
        with self.assertRaises(RuntimeError): c.load_state()

    def test_08_moved_source_rejected(self):
        root,state = self.new_state()
        state['bundle_root']='/different/path'
        c.atomic_json(root/'state.json',state)
        with self.assertRaises(RuntimeError): c.load_state()

    def test_09_nested_workflow_rejected(self):
        with c.workflow_lock():
            with self.assertRaises(RuntimeError):
                with c.workflow_lock(): pass

    def test_10_atomic_write(self):
        p=self.work/'record.json'; c.atomic_json(p,{'stage':1}); c.atomic_json(p,{'stage':2})
        self.assertEqual(c.read(p),{'stage':2})
        self.assertFalse(list(p.parent.glob('.*.tmp')))

    def test_11_both_calibrations_required(self):
        _,state=self.new_state()
        for model,key in [('qwen','cal_q'),('llama','cal_l')]:
            p=Path(state[key])
            c.atomic_json(p/model/'receipts/pilot/behavior.json',{'passed':True})
            if model=='qwen':
                c.atomic_json(p/model/'04_behavior/pilot/dev/artifacts/gate_pilot.json',{'passed':False})
        with self.assertRaises(RuntimeError): c.require_calibration_completed(state)
        c.atomic_json(Path(state['cal_l'])/'llama/04_behavior/pilot/dev/artifacts/gate_pilot.json',{'passed':False})
        c.require_calibration_completed(state)  # a valid negative calibration result is not a crash

    def test_12_partial_project_rejected(self):
        class Fake:
            def pkg(self,*args,**kwargs): raise AssertionError('must not launch')
        p=self.work/'bad';p.mkdir(parents=True)
        with self.assertRaises(RuntimeError): c.ensure_project(Fake(),p,ROOT/'configs/dual_calibration.json','qwen')

    def test_13_existing_project_not_reinitialized(self):
        class Fake:
            def pkg(self,*args,**kwargs): raise AssertionError('must not initialize')
        p=self.work/'ok'; c.atomic_json(p/'protocol.json',{})
        c.ensure_project(Fake(),p,ROOT/'configs/dual_calibration.json','qwen')

    def test_14_copy_shared_data_billing_only(self):
        calls=[]
        class Fake:
            def pkg(self,*args,**kwargs): calls.append((args,kwargs))
        parent=self.work/'calibration';dest=self.work/'new'
        c.ensure_project(Fake(),dest,ROOT/'configs/dual_calibration.json','llama',parent)
        args,_=calls[0]
        self.assertIn('--data-from',args);self.assertIn('--billing-from',args)
        self.assertNotIn('weights',str(args))

    def test_15_parallel_arguments_separate_projects(self):
        root,state=self.new_state()
        r=c.Runner('unit');calls=[];r.call=lambda argv,**kw:calls.append(argv)
        r.parallel(root,state,'pilot')
        args=list(map(str,calls[0]))
        self.assertEqual(args[args.index('--qwen-project')+1],state['cal_q'])
        self.assertEqual(args[args.index('--llama-project')+1],state['cal_l'])
        r.parallel(root,state,'confirm',confirmation=True)
        args=list(map(str,calls[-1]))
        self.assertEqual(args[args.index('--qwen-project')+1],state['conf_q'])

    def test_16_actual_subprocess_success_log(self):
        r=c.Runner('unit');r.call([sys.executable,'-c','print("finished")'])
        self.assertEqual(c.read(r.out/'001.json')['exit_code'],0)
        self.assertIn('finished',(r.out/'001.log').read_text())

    def test_17_actual_subprocess_failure_propagates(self):
        r=c.Runner('unit')
        with self.assertRaises(subprocess.CalledProcessError):
            r.call([sys.executable,'-c','raise SystemExit(7)'])
        self.assertEqual(c.read(r.out/'001.json')['exit_code'],7)

    def test_18_secret_redaction(self):
        r=c.Runner('unit');secret='test-secret-not-real';r.env['HF_TOKEN']=secret
        r.call([sys.executable,'-c','import os; print(os.environ["HF_TOKEN"])'])
        for p in r.out.iterdir(): self.assertNotIn(secret,p.read_text())
        self.assertIn('[REDACTED]',(r.out/'001.log').read_text())

    def test_19_no_token_arguments(self):
        r=c.Runner('unit');r.env['HF_TOKEN']='not-a-real-credential'
        with self.assertRaises(RuntimeError):
            r.call([sys.executable,'-c','print("not-a-real-credential")'])

    def test_20_config_candidate_values(self):
        cfg=c.read(ROOT/'configs/dual_calibration.json')
        self.assertEqual(cfg['train']['pair_batch_size'],16)
        self.assertEqual(cfg['train']['grad_accum'],4)
        self.assertEqual(cfg['train']['max_batch_tokens'],24576)
        self.assertEqual(cfg['train']['max_updates'],96)
        self.assertEqual(cfg['eval']['max_new_tokens'],1024)
        self.assertEqual(cfg['runtime']['generation_token_budget'],98304)
        self.assertEqual(cfg['train']['max_updates_by_model'],{'llama':96,'qwen':96})   # r8: Qwen3.5 at 96
        self.assertEqual(cfg['eval']['judge_hf_id'],'microsoft/phi-4')
        self.assertEqual(cfg['runtime']['study_role'],'calibration')
        self.assertFalse(cfg['runtime']['require_semantic_audit'])
        self.assertEqual(cfg['train']['lora']['r'],16)

    def test_21_sibling_layout(self):
        self.assertEqual(c.PACKAGE.parent,c.TOOLS.parent)
        self.assertNotIn(c.PACKAGE,c.TOOLS.parents)
        self.assertTrue((c.PACKAGE/'scripts/install.sh').is_file())
        self.assertTrue((c.TOOLS/'parallel.py').is_file())

    def test_22_cli_help(self):
        out=subprocess.check_output([sys.executable,str(ROOT/'bundle/console.py'),'--help'],text=True)
        self.assertIn('--action',out)

    def test_23_sha_and_csv_empty_are_not_training_claims(self):
        # The bundle supplies no weights or generated scientific outputs.
        self.assertFalse(list((ROOT/'configs').glob('*.safetensors')))
        p=ROOT/'configs/dual_calibration.candidate.json'
        self.assertTrue(c.read(p)['candidate_only'])


    def test_24_calibration_profiles_before_pilots(self):
        self.new_state(); calls=[]
        class Fake:
            def ensure_install(self): pass
            def authenticate(self): pass
            def hardware(self): pass
            def pkg(self, project, model, *args, **kw):
                calls.append(('pkg', model, args))
                if '--init' in args: c.atomic_json(Path(project)/'protocol.json',{})
            def parallel(self, workspace, state, stage): calls.append(('parallel',stage))
        with patch.object(c,'ask',return_value='RUN'):
            c.calibration(Fake())
        selected=[x[1] for x in calls if x[0]=='parallel']
        self.assertEqual(selected,['profile','pilot'])
        data_calls=[x for x in calls if x[0]=='pkg' and 'data,validate' in x[2]]
        self.assertEqual(len(data_calls),1)
        self.assertNotIn('confirm',str(calls))

    def test_25_failed_profile_prevents_pilot(self):
        self.new_state(); stages=[]
        class Fake:
            def ensure_install(self): pass
            def authenticate(self): pass
            def hardware(self): pass
            def pkg(self,*args,**kw): pass
            def parallel(self, workspace, state, stage):
                stages.append(stage)
                if stage=='profile': raise RuntimeError('mock OOM')
        with patch.object(c,'ask',return_value='RUN'), self.assertRaises(RuntimeError):
            c.calibration(Fake())
        self.assertEqual(stages,['profile'])


if __name__=='__main__': unittest.main()
