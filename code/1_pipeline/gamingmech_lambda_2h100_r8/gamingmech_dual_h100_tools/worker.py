#!/usr/bin/env python3
"""External worker: configure genuine CPU threads, then run the UNMODIFIED package CLI."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', required=True, type=Path)
    p.add_argument('--threads', required=True, type=int)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.threads < 1: p.error('threads must be positive')
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not cmd: p.error('Missing original gamingmech.run arguments')
    root = a.package.resolve()
    sys.path.insert(0, str(root)); os.chdir(root)
    import torch
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Worker requires exactly ONE visible CUDA device. Do not use torchrun.')
    print(json.dumps({'worker_pid': os.getpid(), 'visible_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device': torch.cuda.get_device_name(0), 'cpu_threads': torch.get_num_threads(),
        'interop_threads': torch.get_num_interop_threads(), 'argv': cmd}), flush=True)
    sys.argv = ['gamingmech.run', *cmd]
    runpy.run_module('gamingmech.run', run_name='__main__')

if __name__ == '__main__': main()
