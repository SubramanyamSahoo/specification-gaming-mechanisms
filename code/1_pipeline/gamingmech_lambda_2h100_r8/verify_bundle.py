#!/usr/bin/env python3
"""Verify bundled files. Generated work, caches and the installed venv are not distribution files."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> int:
    manifest = json.loads((root / 'BUNDLE_SHA256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError(f'Unsafe manifest path: {name}')
        path = root / rel
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f'Missing or symlinked distribution file: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Changed distribution file: {name}. Do not edit a locked research package.')
    for directory, filename in [('gamingmech_h100_v3_1', 'PACKAGE_SHA256.json'),
                                ('gamingmech_dual_h100_tools', 'SHA256SUMS.json')]:
        child = root / directory
        entries = json.loads((child / filename).read_text())
        for name, expected in entries.items():
            if hashlib.sha256((child / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Original component checksum failed: {directory}/{name}')
    return len(manifest)


if __name__ == '__main__':
    n = verify(Path(__file__).resolve().parent)
    print(f'Verified {n} bundled files and both original component manifests.')
