"""Surgical, logged repair of the one-token callback defect in an EXISTING v3 project.

This is NOT migration to v3.1. All configuration, data, weights, and numerical
methods stay at v3. Only steering.py compute() -> compute is authorized. Other
source changes are rejected. Dry-run is the default; use --apply after inspection.
Stop all project jobs first. Back up your project before applying any repair.
"""
from __future__ import annotations
import argparse
import ast
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import uuid

BEFORE = 'key, compute()), dtype=np.float64)'
AFTER = 'key, compute), dtype=np.float64)'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def stable(*parts): return hashlib.sha256('||'.join(str(p) for p in parts).encode()).hexdigest()
def source_digest(hashes): return stable(*[(k, hashes[k]) for k in sorted(hashes)])
def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.repair-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def repair(project: Path, package: Path, apply: bool, reason: str) -> dict:
    project, package = project.resolve(), package.resolve()
    root = package / 'gamingmech'
    manifest_path = package / 'PACKAGE_SHA256.json'
    protocol_path = project / 'protocol.json'
    original = json.loads((Path(__file__).resolve().parents[1]/'docs/v3_original_source_hashes.json').read_text())
    protocol = json.loads(protocol_path.read_text())
    snapshot = project/'source_snapshot'
    # Full source inventory must match the released v3; no arbitrary module can be added.
    if {p.name for p in root.glob('*.py')} != set(original):
        raise RuntimeError('Not the exact v3 package source inventory')
    old_text = (snapshot/'steering.py').read_text()
    # Recover the original bytes even if a previous attempt already repaired the snapshot.
    if BEFORE in old_text:
        old_source = old_text
    elif AFTER in old_text:
        old_source = old_text.replace(AFTER, BEFORE)
    else: raise RuntimeError('Unknown steering implementation; refusing to guess a patch')
    if old_source.count(BEFORE) != 1 or hashlib.sha256(old_source.encode()).hexdigest() != original['steering.py']:
        raise RuntimeError('Snapshot is not the released v3 steering implementation')
    fixed_source = old_source.replace(BEFORE,AFTER)
    fixed_hashes = dict(original, **{'steering.py':hashlib.sha256(fixed_source.encode()).hexdigest()})
    old_hash,new_hash=source_digest(original),source_digest(fixed_hashes)
    if protocol.get('source_hash') not in (old_hash,new_hash):
        raise RuntimeError('Project source lock is not the released v3 / this exact hotfix')
    for name in original:
        permitted={original[name],fixed_hashes[name]} if name=='steering.py' else {original[name]}
        for directory in (root,snapshot):
            if not (directory/name).is_file() or sha(directory/name) not in permitted:
                raise RuntimeError(f'Unapproved source change: {directory/name}')
    expected_package=json.loads(manifest_path.read_text())
    for name,value in expected_package.items():
        path=package/name
        if name=='gamingmech/steering.py':continue
        if not path.is_file() or sha(path)!=value:
            raise RuntimeError(f'Distribution differs outside the permitted patch: {name}')
    parsed=ast.parse(fixed_source)
    calls=[n for n in ast.walk(parsed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='cached_json']
    if not calls or any(not isinstance(n.args[-1],ast.Name) or n.args[-1].id!='compute' for n in calls):
        raise AssertionError('Callback AST check failed')
    if protocol['source_hash']==new_hash:
        if sha(root/'steering.py') != fixed_hashes['steering.py'] or sha(snapshot/'steering.py') != fixed_hashes['steering.py']:
            raise RuntimeError('Incomplete previous repair; restore its backup before retrying')
        return {'status':'already_repaired','source_hash':new_hash}
    art=project/'01_data'/'artifacts'
    data_names=('roster.jsonl','real_pairs.jsonl','eval_items.jsonl')
    data_id=stable(*[(n,sha(art/n)) for n in data_names]) if all((art/n).is_file() for n in data_names) else None
    updates=[];invalidated=[]
    for path in sorted(project.glob('*/receipts/*/*.json')):
        model, _, phase, filename=path.relative_to(project).parts
        stage=path.stem;old=json.loads(path.read_text())
        if stage in ('steering','report'):
            invalidated.append(path);continue
        dep=None if stage in ('data','validate') else data_id
        if dep is None and stage not in ('data','validate'):
            raise RuntimeError(f'Cannot validate receipt without its dataset: {path}')
        expected=stable(protocol['fingerprint'],model,phase,old_hash,dep)[:16]
        if old.get('signature')!=expected:
            raise RuntimeError(f'Cannot rebind an already stale receipt: {path}')
        if any(not (project/name).exists() for name in old.get('outputs',[])):
            raise RuntimeError(f'Missing receipt outputs: {path}')
        new=dict(old, signature=stable(protocol['fingerprint'],model,phase,new_hash,dep)[:16])
        updates.append((path,new))
    record={'status':'planned','scope':'one-token-steering-callback-only','reason':reason.strip(),
            'old_source_hash':old_hash,'new_source_hash':new_hash,
            'rebound_stages':[str(p.relative_to(project)) for p,_ in updates],
            'invalidated_receipts':[str(p.relative_to(project)) for p in invalidated],
            'scientific_config_unchanged':True,'gpu_execution_verified':False}
    if not apply:return record
    if not reason.strip():raise ValueError('An explicit repair reason is required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    history=project/'source_amendments'/stamp
    history.mkdir(parents=True,exist_ok=False)
    shutil.copytree(snapshot,history/'source_snapshot_before')
    shutil.copy2(protocol_path,history/'protocol_before.json')
    shutil.copy2(root/'steering.py',history/'running_steering_before.py')
    shutil.copy2(manifest_path,history/'PACKAGE_SHA256_before.json')
    for path,_ in updates:
        dest=history/'receipts_before'/path.relative_to(project);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    for path in invalidated:
        dest=history/'invalidated'/path.relative_to(project);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(path,dest)
    # Keep any incomplete/successful steering files rather than overwriting or relabeling them.
    for path in sorted(project.glob('*/08_steering')):
        dest=history/'invalidated'/path.relative_to(project);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(path,dest)
    (root/'steering.py').write_text(fixed_source)
    (snapshot/'steering.py').write_text(fixed_source)
    expected_package['gamingmech/steering.py']=sha(root/'steering.py')
    atomic_json(expected_package,manifest_path)
    for path,new in updates:
        new['source_amendment']=stamp;atomic_json(new,path)
    protocol['source_hash']=new_hash
    protocol.setdefault('source_amendments',[]).append(stamp)
    atomic_json(protocol,protocol_path)
    record.update(status='applied',amendment=stamp,created_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(record,history/'REPAIR.json')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True,type=Path)
    p.add_argument('--package',required=True,type=Path,help='OLD v3 package directory, not the v3.1 directory')
    p.add_argument('--reason',default='Correct eager steering cache callback; no scientific setting changes')
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    # Nonblocking lock: never mutate a project being executed.
    with (a.project/'.execution.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Project is running. Stop it before repair.')
        print(json.dumps(repair(a.project,a.package,a.apply,a.reason),indent=2))
