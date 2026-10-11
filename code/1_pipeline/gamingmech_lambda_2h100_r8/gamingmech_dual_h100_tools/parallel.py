#!/usr/bin/env python3
"""Two independent one-GPU jobs with separate projects; never bypass package gates/locks."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

STAGES = {
    'profile': ('pilot', 'profile,report'),
    'pilot': ('pilot', 'train,behavior,report'),
    'confirm': ('full', 'train,behavior,test_behavior,activations,report'),
    'mechanisms': ('full', 'steering,patching,dissect,detection,report'),
    'features': ('full', 'features,report'),
    'minimal': ('full', 'minimal,report'),
    'trajectory': ('full', 'trajectory,report'),
}
DATA_FILES = ('roster.jsonl', 'real_pairs.jsonl', 'eval_items.jsonl')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def read_json(path: Path): return json.loads(path.read_text())


def validate_projects(qwen: Path, llama: Path, stage: str) -> dict:
    qwen, llama = qwen.resolve(), llama.resolve()
    if qwen == llama or qwen in llama.parents or llama in qwen.parents:
        raise ValueError('Need two DISTINCT non-nested initialized projects, never one shared PROJECT.')
    protocols = [read_json(p/'protocol.json') for p in (qwen, llama)]
    if protocols[0]['fingerprint'] != protocols[1]['fingerprint']:
        raise ValueError('Project scientific fingerprints differ. Use the SAME selected configuration.')
    if protocols[0]['source_hash'] != protocols[1]['source_hash']:
        raise ValueError('Projects have different source locks.')
    role = protocols[0]['config']['runtime']['study_role']
    if role == 'calibration' and stage not in ('profile', 'pilot'):
        raise ValueError('Calibration cannot run confirmation/test/mechanism stages.')
    configs = protocols[0]['config']
    budgets = []
    for p in (qwen, llama):
        if (p/'HANDOFF_FAILED.json').exists(): raise ValueError('Incomplete data handoff: '+str(p))
        if not (p/'preflight.json').is_file(): raise ValueError('Preflight missing: '+str(p))
        if not (p/'01_data/artifacts/semantic_validation.json').is_file():
            raise ValueError('Run data + semantic validation first: '+str(p))
        if role == 'confirmatory' and not (p/'01_data/artifacts/human_audit.json').is_file():
            raise ValueError('Register the reviewed human audit in BOTH confirmation projects.')
        budgets.append(read_json(p/'billing.json'))
    if budgets[0] != budgets[1]:
        raise ValueError('Same-instance ledgers differ. Preserve and synchronize the actual billing history, not a new allowance.')
    if not budgets[0].get('active', True): raise ValueError('Billing clock is paused.')
    hashes = {n: sha(qwen/'01_data/artifacts'/n) for n in DATA_FILES}
    if any(sha(llama/'01_data/artifacts'/n) != v for n,v in hashes.items()):
        raise ValueError('Data differ. Import once-validated calibration data into both projects.')
    return {'projects': [str(qwen),str(llama)], 'fingerprint': protocols[0]['fingerprint'],
            'source_hash': protocols[0]['source_hash'], 'role': role, 'data_sha256': hashes,
            'billing': budgets[0], 'budget_usd': configs['runtime']['budget_usd'],
            'reserve_usd': configs['runtime']['reserve_usd']}


def select_gpus(text: str, indices: tuple[int,int]) -> list[dict]:
    rows={}
    for line in csv.reader(text.strip().splitlines()):
        if not line: continue
        i, uid, name, mem = [x.strip() for x in line]
        rows[int(i)]={'index':int(i),'uuid':uid,'name':name,'memory_mib':float(mem)}
    if len(set(indices)) != 2: raise ValueError('GPU indices must differ.')
    if any(i not in rows for i in indices): raise ValueError('Both selected GPUs must exist.')
    selected=[rows[i] for i in indices]
    if any('H100' not in x['name'] for x in selected): raise ValueError('This launcher is intended for two H100s.')
    if selected[0]['uuid'] == selected[1]['uuid']: raise ValueError('Duplicate physical GPU UUID.')
    return selected


def env_for_gpu(gpu: dict, threads: int) -> dict:
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               NUMEXPR_NUM_THREADS=str(threads), TOKENIZERS_PARALLELISM='false', USE_TORCH='1', USE_TF='0',
               PYTHONUNBUFFERED='1')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    return env


def stop_jobs(jobs):
    live=[p for p,_ in jobs if p.poll() is None]
    for p in live:
        try: os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    deadline=time.monotonic()+15
    while any(p.poll() is None for p in live) and time.monotonic()<deadline: time.sleep(.1)
    for p in live:
        if p.poll() is None:
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass
    for p in live: p.wait()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--qwen-project', type=Path, required=True)
    ap.add_argument('--llama-project', type=Path, required=True)
    ap.add_argument('--stage', choices=sorted(STAGES), required=True)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--threads-per-job', type=int, default=None)
    ap.add_argument('--python', type=Path, help='Default: PACKAGE/.venv/bin/python')
    ap.add_argument('--logs', type=Path, default=Path('dual_logs'))
    ap.add_argument('--dry-run', action='store_true', help='Check projects and print commands, no GPU jobs')
    a=ap.parse_args()
    root=a.package.resolve()
    if any(os.environ.get(k) for k in ('LOCAL_RANK','RANK','WORLD_SIZE')):
        ap.error('Do not run this launcher inside torchrun/MPI/DDP.')
    token_var=read_json(a.qwen_project/'protocol.json')['config']['runtime']['hf_token_env']
    if not os.environ.get(token_var): ap.error('Set '+token_var+' in the shell; never paste it into source.')
    # Verify the original package before any paid jobs. Extra external tools are not copied into it.
    python=a.python.absolute() if a.python else root/'.venv/bin/python'
    if not python.is_file(): ap.error('Install PACKAGE/.venv or pass --python explicitly.')
    subprocess.run([str(python), str(root/'scripts/verify_package.py')],cwd=root,check=True)
    plan=validate_projects(a.qwen_project,a.llama_project,a.stage)
    cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    threads=a.threads_per_job if a.threads_per_job is not None else min(8,max(1,cpus//2))
    if threads<1 or threads*2>cpus: ap.error('Thread request oversubscribes available CPU affinity.')
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True)
    indices=tuple(int(x) for x in a.gpus.split(','))
    if len(indices)!=2: ap.error('Exactly two GPU indices required.')
    gpus=select_gpus(text, indices)
    phase, stages=STAGES[a.stage]
    worker=Path(__file__).with_name('worker.py').resolve()
    commands=[]
    for family, project in zip(('qwen','llama'),plan['projects']):
        commands.append([str(python),str(worker),'--package',str(root),'--threads',str(threads),'--',
             '--project',project,'--model',family,'--phase',phase,'--stages',stages])
    plan.update(stage=a.stage,gpus=gpus,threads_per_job=threads,commands=commands,
                tools_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    if a.dry_run:
        print(json.dumps(plan,indent=2)); return
    out=a.logs.resolve()/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    (out/'launch.json').write_text(json.dumps(plan,indent=2))
    print('Logs:',out,flush=True)
    print('Independent workers; no distributed training. Both ledgers count the TOTAL instance rate.',flush=True)
    jobs=[]; files=[]
    def interrupt(signum,frame): raise KeyboardInterrupt
    old_int=signal.signal(signal.SIGINT,interrupt); old_term=signal.signal(signal.SIGTERM,interrupt)
    status=0
    try:
        for family, cmd, gpu in zip(('qwen','llama'),commands,gpus):
            f=(out/(family+'.log')).open('w'); files.append(f)
            p=subprocess.Popen(cmd,cwd=root,env=env_for_gpu(gpu,threads),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((p,family)); print(f'{family}: GPU {gpu["index"]}, PID {p.pid}',flush=True)
        reported=set()
        while True:
            for p,family in jobs:
                if p.poll() is not None and p.returncode != 0 and family not in reported:
                    # The two projects are independent: a failed gate or crash in one says nothing about the other.
                    reported.add(family); status=1
                    print(f'{family} stopped with exit code {p.returncode}. The other family keeps running.',flush=True)
            if all(p.poll() is not None for p,_ in jobs): break
            time.sleep(1)
    except KeyboardInterrupt:
        status=130; print('Interrupted. Stopping both workers.',flush=True)
    finally:
        stop_jobs(jobs)
        (out/'exit_status.json').write_text(json.dumps({fam:p.returncode for p,fam in jobs},indent=2))
        for f in files:f.close()
        signal.signal(signal.SIGINT,old_int);signal.signal(signal.SIGTERM,old_term)
    print('Review exit_status.json. Provider billing does NOT stop when these jobs stop.',flush=True)
    raise SystemExit(status)

if __name__=='__main__': main()
