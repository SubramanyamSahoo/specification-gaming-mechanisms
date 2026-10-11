"""Regenerate a distribution manifest BEFORE initialization/release.

This does not unlock a project and must never be used to conceal a source change
inside an already initialized experiment. Review and test edits before invoking.
"""
import argparse
import hashlib
import json
from pathlib import Path

EXCLUDE_PARTS = {'__pycache__', '.pytest_cache', '.venv', '.git', 'runs', '.ruff_cache'}

def collect(root):
    files={}
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if not p.is_file() or EXCLUDE_PARTS.intersection(rel.parts) or any(x.endswith('.egg-info') for x in rel.parts):continue
        if rel.as_posix() in {'PACKAGE_SHA256.json','installed_packages.txt'} or p.suffix in {'.pyc','.pyo'}:continue
        if p.is_symlink():raise RuntimeError(f'Refusing to checksum a symlink: {p}')
        files[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return files

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acknowledge-source-change',action='store_true',required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    files=collect(root)
    (root/'PACKAGE_SHA256.json').write_text(json.dumps(files,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(f'Regenerated {len(files)} checksums. Existing project source locks are unchanged.')
