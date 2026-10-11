#!/usr/bin/env python3
"""Print ACTUAL logged exposure, not nominal batch*steps. Read-only; no GPU work."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def rows(project: Path, family: str):
    root=project/family/'03_train/artifacts'
    for folder in sorted(root.glob('*')):
        manifest=folder/'training_manifest.json'
        if not manifest.is_file(): continue
        m=json.loads(manifest.read_text()); steps={}
        for path in sorted(folder.glob('train_*.csv')):
            with path.open() as f:
                for r in csv.DictReader(f):
                    if r.get('step'): steps[int(r['step'])]=r
        exposed=sum(int(r['n_pairs']) for r in steps.values())
        n_train=int(m['n_train'])
        final=folder/'final/COMPLETE.json'
        last=folder/'last/training_state.pt'
        yield {'condition':folder.name,'n_train':n_train,'logged_updates':len(steps),
               'last_logged_update':max(steps,default=0),'logged_pair_exposures':exposed,
               'equivalent_passes':exposed/n_train if n_train else None,
               'mean_pairs_per_update':exposed/len(steps) if steps else None,
               'final_manifest_present':final.is_file(),'resume_checkpoint_present':last.is_file(),
               'note':'Logged updates may exceed the last durable checkpoint after an interruption.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--model',choices=['qwen','llama'],required=True)
    a=p.parse_args()
    print(json.dumps(list(rows(a.project.resolve(),a.model)),indent=2))
