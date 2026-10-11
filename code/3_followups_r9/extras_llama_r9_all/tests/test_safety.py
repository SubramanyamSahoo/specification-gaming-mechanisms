"""CPU tests. Synthetic fixtures do not establish pretrained-model validity."""
import contextlib
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import coordinator as co
from r9_support import atomic_json, checked_outputs, checkpointed, fingerprint, load_json, sha256

class Temp(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

class IOTests(Temp):
    def test_atomic_json(self):
        p=self.root/'x.json'; atomic_json(p,{'value':'Å'}); self.assertEqual(load_json(p),{'value':'Å'})
        atomic_json(p,{'value':2}); self.assertEqual(load_json(p)['value'],2)
        self.assertFalse(list(self.root.glob('*.tmp')))
    def test_fingerprint_determinism(self): self.assertEqual(fingerprint({'a':1,'b':2}),fingerprint({'b':2,'a':1}))
    def test_checkpoint_resume(self):
        path=self.root/'table.csv'; calls=[]
        def fn(): calls.append(1); path.write_text('x\n1\n'); return [path]
        checkpointed(self.root,'s79_patching',fn);checkpointed(self.root,'s79_patching',fn)
        self.assertEqual(len(calls),1)
    def test_checkpoint_tamper(self):
        p=self.root/'x.csv'; p.write_text('before')
        checkpointed(self.root,'stage',lambda:[p]); p.write_text('after')
        with self.assertRaises(RuntimeError): checkpointed(self.root,'stage',lambda:[p])
    def test_empty_checkpoint(self):
        with self.assertRaises(RuntimeError):checkpointed(self.root,'x',lambda:[])
    def test_missing_output(self):
        with self.assertRaises(RuntimeError): checked_outputs(self.root,{'outputs':{'gone.csv':'abc'}})
    def test_locks(self):
        with co.locked(self.root/'x.lock'):
            with self.assertRaises(RuntimeError):
                with co.locked(self.root/'x.lock'): pass
    def test_no_manifest_empty_success(self):
        with self.assertRaises(RuntimeError): co.manifest_outputs(self.root,'attention')

class BillingTests(Temp):
    def billing(self): return {'active':True,'hourly_usd':8.,'started_utc':'2026-10-10T01:00:00+00:00',
        'segments':[{'started_utc':'2026-10-10T00:00:00+00:00','ended_utc':'2026-10-10T01:00:00+00:00','hourly_usd':4.}]}
    def test_segments_preserved(self):self.assertEqual(co.spend_at(self.billing(),co.parse_time('2026-10-10T03:00:00+00:00')),20.)
    def test_timezone_required(self):
        with self.assertRaises(ValueError): co.parse_time('2026-10-10T00:00:00')
    def test_future_clock_refused(self):
        with self.assertRaises(ValueError):co.spend_at(self.billing(),co.parse_time('2026-10-09T00:00:00+00:00'))
    def test_nonfinite_rate_refused(self):
        d=self.billing();d['hourly_usd']=float('nan')
        with self.assertRaises(ValueError):co.spend_at(d,co.parse_time('2026-10-11T00:00:00+00:00'))
    def test_extra_limit_never_overrides_total(self):
        bill=self.billing();atomic_json(self.root/'billing.json',bill)
        run={'source':str(self.root),'billing_identity':fingerprint(bill),'budget_usd':25,'reserve_usd':3,
             'spent_at_initialization':4.,'settings':{'max_additional_usd':100}}
        left,_=co.budget_remaining(run,co.parse_time('2026-10-10T03:00:00+00:00'));self.assertEqual(left,2.)
        run['settings']['max_additional_usd']=10
        left,_=co.budget_remaining(run,co.parse_time('2026-10-10T03:00:00+00:00'));self.assertEqual(left,-6.)
    def test_changed_billing_refused(self):
        bill=self.billing();atomic_json(self.root/'billing.json',bill)
        with self.assertRaises(RuntimeError):co.budget_remaining({'source':str(self.root),'billing_identity':'wrong'})
    def test_paused_billing_refused(self):
        bill=self.billing();bill['active']=False;atomic_json(self.root/'billing.json',bill)
        with self.assertRaises(RuntimeError):co.budget_remaining({'source':str(self.root),'billing_identity':fingerprint(bill)})

class MirrorTests(Temp):
    def make(self):
        src=self.root/'source';src.mkdir();atomic_json(src/'protocol.json',{'config':{}})
        atomic_json(src/'billing.json',{'segments':['retained']})
        data=src/'01_data'/'artifacts';data.mkdir(parents=True);(data/'roster.jsonl').write_text('original\n')
        train=src/'llama'/'03_train'/'artifacts';train.mkdir(parents=True)
        (train/'reference_logprobs.pt').write_bytes(b'cached-reference')
        final=train/'syco_f100_s73'/'final';final.mkdir(parents=True)
        atomic_json(final/'COMPLETE.json',{'test':True});(final/'adapter_model.safetensors').write_bytes(b'weights')
        (train/'syco_f100_s73'/'summary.json').write_text('{}')
        return src
    def test_data_is_independent_copy(self):
        src=self.make();dst=self.root/'dest';co.mirror(src,dst)
        p=dst/'01_data/artifacts/roster.jsonl';self.assertFalse(p.is_symlink());p.write_text('local')
        self.assertEqual((src/'01_data/artifacts/roster.jsonl').read_text(),'original\n')
    def test_final_link_not_whole_adapter(self):
        src=self.make();dst=self.root/'dest';co.mirror(src,dst)
        p=dst/'llama/03_train/artifacts/syco_f100_s73'
        self.assertFalse(p.is_symlink());self.assertTrue((p/'final').is_symlink())
        self.assertFalse((p/'summary.json').is_symlink())
    def test_reference_cache_copied(self):
        src=self.make();dst=self.root/'dest';co.mirror(src,dst)
        p=dst/'llama/03_train/artifacts/reference_logprobs.pt';self.assertFalse(p.is_symlink())
        p.write_bytes(b'updated');self.assertEqual((src/'llama/03_train/artifacts/reference_logprobs.pt').read_bytes(),b'cached-reference')
    def test_billing_copied_intact(self):
        src=self.make();dst=self.root/'dest';co.mirror(src,dst)
        self.assertEqual(sha256(src/'billing.json'),sha256(dst/'billing.json'))
    def test_original_archive_scope(self):
        folder=self.root/'run';folder.mkdir();atomic_json(folder/'run.json',{'source':'original'})
        (folder/'project/RESULTS/x').mkdir(parents=True)
        p=folder/'project/RESULTS/x';(p/'raw.jsonl').write_text('{"measured":1}\n');(p/'weights.pt').write_bytes(b'big')
        z=co.pack(folder)
        import zipfile
        with zipfile.ZipFile(z) as archive:
            self.assertIn('project/RESULTS/x/raw.jsonl',archive.namelist());self.assertNotIn('project/RESULTS/x/weights.pt',archive.namelist())
            self.assertFalse(json.loads(archive.read('ARCHIVE_SCOPE.json'))['weights_and_tensor_bytes_included'])

class EnvironmentTests(unittest.TestCase):
    def test_gpu_uuid(self):
        with patch('subprocess.check_output',return_value='GPU-example\n'):
            self.assertEqual(co.gpu_uuid(1),'GPU-example')
    def test_bad_gpu_uuid(self):
        with patch('subprocess.check_output',return_value='0\n'):
            with self.assertRaises(RuntimeError):co.gpu_uuid(1)
    def test_busy_gpu(self):
        with patch('subprocess.check_output',return_value='12345\n'):
            with self.assertRaises(RuntimeError):co.require_idle('GPU-example')
    def test_gpu_query_error(self):
        with patch('subprocess.check_output',side_effect=subprocess.CalledProcessError(1,['nvidia-smi'])):
            with self.assertRaises(subprocess.CalledProcessError):co.require_idle('GPU-example')
    def test_idle_gpu(self):
        with patch('subprocess.check_output',return_value=''):co.require_idle('GPU-example')
    def test_shell_syntax(self):
        for p in ROOT.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    def test_no_session_kill_or_start_main(self):
        for p in ROOT.glob('*.sh'):
            text=p.read_text();self.assertNotIn('kill-session',text);self.assertNotIn('scripts/pilot.sh',text)
    def test_declared_full_plan(self):
        settings=load_json(ROOT/'settings.json');co.validate_settings(settings)
        self.assertEqual(settings['stage_order'],list(co.STAGES))
        self.assertEqual(settings['physical_gpu'],1)
    def test_invalid_extra_cap(self):
        settings=load_json(ROOT/'settings.json');settings['max_additional_usd']=-1
        with self.assertRaises(ValueError):co.validate_settings(settings)

class ProcessTests(Temp):
    def test_stop_only_spawned_process_group(self):
        p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        co.stop_child(p,2);self.assertIsNotNone(p.poll())
    def test_finished_child_not_signalled(self):
        p=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True);p.wait()
        with patch('os.killpg') as kill:
            co.stop_child(p,1);kill.assert_not_called()


class SupervisorTests(Temp):
    def exercise(self, returncode, budget_ok=True):
        # Real subprocess orchestration with an explicitly fake, CPU-only worker.
        fake=self.root/'addon';fake.mkdir()
        (fake/'worker.py').write_text(
            'import sys,time\n'
            'print("synthetic worker started",flush=True)\n'
            + ('time.sleep(20)\n' if not budget_ok else '')
            + f'sys.exit({returncode})\n')
        folder=self.root/'run';folder.mkdir()
        run={'settings':{'budget_poll_seconds':1,'shutdown_grace_seconds':1}}
        with patch.object(co,'ROOT',fake), patch.object(co,'budget_remaining',return_value=((100,0) if budget_ok else (-1,100))), \
             patch.object(co,'done_receipt',return_value={'status':'completed'}):
            co.execute_stage(folder,run,'attention')
        return folder
    def test_nonzero_child_propagates(self):
        with self.assertRaisesRegex(RuntimeError,'exit 17'):self.exercise(17)
    def test_successful_child_with_receipt(self):
        self.exercise(0)
    def test_budget_interrupt_does_not_return_success(self):
        with self.assertRaisesRegex(RuntimeError,'budget'):self.exercise(0,budget_ok=False)

if __name__=='__main__':unittest.main()
