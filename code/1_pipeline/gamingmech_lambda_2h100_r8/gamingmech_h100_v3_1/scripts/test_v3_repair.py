"""Exercise repair against the exact attached v3 archive, not a reconstructed package."""
import json, shutil, subprocess, tempfile, zipfile
from pathlib import Path
import argparse,hashlib
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--archive',required=True,type=Path,help='Exact original v3 ZIP')
args=ap.parse_args()
archive=args.archive.resolve()
expected='6155547d1c263f4125f464894cffa93d1d2e17e51b3745e998681e47da391959'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:
    raise SystemExit('This integration check accepts only the original audited v3 archive')
script=Path(__file__).resolve().with_name('repair_initialized_v3.py')
results=[]
for already_edited in (False,True):
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        with zipfile.ZipFile(archive) as z:z.extractall(root)
        package=root/'gamingmech_h100_v3';project=root/'project'
        setup='''from gamingmech.config import Config
from gamingmech.runtime import initialize,source_hash,data_identity
from gamingmech.common import save_json,save_jsonl,stable_hash
from pathlib import Path
cfg=Config();cfg.runtime.device='cpu';cfg.runtime.dtype='float32';cfg.runtime.project=PROJECT
initialize(cfg)
for n in ('roster.jsonl','real_pairs.jsonl','eval_items.jsonl'):save_jsonl([{'id':'fixture'}],cfg.data_dir/'artifacts'/n)
sig=stable_hash(cfg.fingerprint(),'qwen','full',source_hash(),data_identity(cfg))
save_json({'passed':True,'signature':sig,'outputs':[]},cfg.exp_dir/'qwen'/'receipts'/'full'/'train.json')
save_json({'passed':False,'signature':sig,'error':'callback'},cfg.exp_dir/'qwen'/'receipts'/'full'/'steering.json')
'''.replace('PROJECT',repr(str(project)))
        subprocess.run(['python','-c',setup],cwd=package,check=True,capture_output=True,text=True)
        before=json.loads((project/'protocol.json').read_text())
        if already_edited:
            f=package/'gamingmech'/'steering.py'
            f.write_text(f.read_text().replace('key, compute()), dtype=np.float64)','key, compute), dtype=np.float64)'))
        args=['python',str(script),'--project',str(project),'--package',str(package)]
        planned=json.loads(subprocess.run(args,check=True,capture_output=True,text=True).stdout)
        assert planned['status']=='planned'
        applied=json.loads(subprocess.run(args+['--apply'],check=True,capture_output=True,text=True).stdout)
        after=json.loads((project/'protocol.json').read_text())
        assert before['config']==after['config'] and before['fingerprint']==after['fingerprint']
        assert before['source_hash']!=after['source_hash']
        assert (project/'qwen'/'receipts'/'full'/'train.json').exists()
        assert not (project/'qwen'/'receipts'/'full'/'steering.json').exists()
        check=subprocess.run(['python','-c',f'from gamingmech.runtime import load_project; print(load_project({str(project)!r},"qwen","full").fingerprint())'],cwd=package,check=True,capture_output=True,text=True)
        assert check.stdout.strip()==before['fingerprint']
        subprocess.run(['python','scripts/verify_package.py'],cwd=package,check=True,capture_output=True,text=True)
        again=json.loads(subprocess.run(args+['--apply'],check=True,capture_output=True,text=True).stdout)
        assert again['status']=='already_repaired'
        # Arbitrary edits remain forbidden.
        f=package/'gamingmech'/'dpo.py';f.write_text(f.read_text()+'\n# unrelated change\n')
        reject=subprocess.run(args+['--apply'],capture_output=True,text=True)
        assert reject.returncode!=0 and 'Unapproved source change' in reject.stderr
        results.append({'already_manually_patched':already_edited,'passed':True,
                        'checks':['dry_run','configuration_preserved','source_lock_accepts_hotfix','training_receipt_preserved',
                                  'steering_receipt_invalidated','distribution_checksum','idempotence','other_edits_rejected']})
print(json.dumps(results,indent=2))
