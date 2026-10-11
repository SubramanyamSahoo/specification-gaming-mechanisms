#!/usr/bin/env python3
"""Create a PRE-INIT calibration candidate. Never edit an initialized protocol."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path


def build_config(base: dict, microbatch: int, updates: int | None = None,
                 output_tokens: int | None = None, generation_budget: int | None = None,
                 updates_by_model: dict | None = None) -> dict:
    cfg = copy.deepcopy(base)
    if cfg['runtime'].get('study_role') != 'calibration':
        raise ValueError('Use the calibration template, not a frozen confirmation protocol.')
    train = cfg['train']
    effective = train['pair_batch_size'] * train['grad_accum']
    sorting_window = train['pair_batch_size'] * train['sort_window']
    if microbatch < 1 or effective % microbatch or sorting_window % microbatch:
        raise ValueError('Microbatch must divide both nominal effective batch and sorting-window size.')
    train['pair_batch_size'] = microbatch
    train['grad_accum'] = effective // microbatch
    train['sort_window'] = sorting_window // microbatch
    # chosen and rejected are separate sequences; reserve for the full maximum width.
    train['max_batch_tokens'] = max(train['max_batch_tokens'], 2 * microbatch * train['max_seq_tokens'])
    if updates is not None:
        if updates < 1: raise ValueError('Updates must be positive.')
        train['max_updates'] = updates
    if updates_by_model:
        if any(int(n) != n or n < 1 for n in updates_by_model.values()):
            raise ValueError('Per family updates must be positive integers.')
        train['max_updates_by_model'] = dict(sorted(updates_by_model.items()))
    if output_tokens is not None:
        if output_tokens < 1: raise ValueError('Output-token cap must be positive.')
        cfg['eval']['max_new_tokens'] = output_tokens
    if generation_budget is not None:
        if generation_budget < cfg['eval']['max_new_tokens'] + 1:
            raise ValueError('Generation budget must accommodate a prompt and the output cap.')
        cfg['runtime']['generation_token_budget'] = generation_budget
    return cfg


def write_config(base_path: Path, output: Path, **kwargs) -> dict:
    cfg = build_config(json.loads(base_path.read_text()), **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2); f.write('\n')
    doc = {'base_file': str(base_path.resolve()),
           'base_sha256': hashlib.sha256(base_path.read_bytes()).hexdigest(),
           'config_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
           'nominal_pairs_per_update': cfg['train']['pair_batch_size'] * cfg['train']['grad_accum'],
           'candidate_only': True,
           'warning': 'Token-limited batches and different shuffles can change actual update membership. '
                      'This preserves nominal batch, not a bitwise-identical training trajectory. '
                      'No sufficiency or H100-fit claim is made.'}
    with output.with_suffix('.candidate.json').open('x', encoding='utf-8') as f:
        json.dump(doc, f, indent=2); f.write('\n')
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--microbatch', type=int, default=16)
    p.add_argument('--updates', type=int, help='Optional strength change; not a speed-only setting')
    p.add_argument('--output-tokens', type=int, help='Optional evaluation-policy change')
    p.add_argument('--generation-budget', type=int, help='Padded prompt plus maximum output budget')
    p.add_argument('--updates-by-model', help='Optional per family budgets, e.g. qwen=192,llama=96')
    a = p.parse_args()
    manifest = a.output.with_suffix('.candidate.json')
    if a.output.exists() or manifest.exists(): p.error('Destination exists; choose a new filename.')
    by_model = ({k.strip(): int(v) for k, v in (x.split('=') for x in a.updates_by_model.split(','))}
                if a.updates_by_model else None)
    print(json.dumps(write_config(a.base, a.output, microbatch=a.microbatch, updates=a.updates,
        output_tokens=a.output_tokens, generation_budget=a.generation_budget, updates_by_model=by_model), indent=2))
    print('Writer/data/quality thresholds/rank/LR/seeds are unchanged. Profile before paid training.')

if __name__ == '__main__': main()
