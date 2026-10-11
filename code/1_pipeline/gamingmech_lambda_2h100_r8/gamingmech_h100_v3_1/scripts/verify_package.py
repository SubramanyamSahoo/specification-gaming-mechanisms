"""Verify the distributed source/config/docs snapshot, not experiment results."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'PACKAGE_SHA256.json').read_text())
for name,expected in manifest.items():
    p=root/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        raise SystemExit(f'Missing or changed distribution file: {name}')
print(f'Verified {len(manifest)} files.')
