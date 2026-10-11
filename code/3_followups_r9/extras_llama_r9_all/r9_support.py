"""Small auditable I/O helpers. No approval labels or numerical results are prefilled."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
import uuid
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_outputs(root: Path, record: dict) -> None:
    outputs = record.get('outputs', {})
    if not outputs:
        raise RuntimeError('Completed record has no outputs')
    for rel, digest in outputs.items():
        p = root / rel
        if not p.is_file() or sha256(p) != digest:
            raise RuntimeError(f'Missing or changed completed output: {p}')


def checkpointed(root: Path, key: str, compute):
    """One complete seed/substage is never repeated merely because a later substage failed."""
    marker = root / 'RESULTS' / 'seeds' / 'checkpoints' / (key + '.json')
    if marker.exists():
        record = load_json(marker)
        checked_outputs(root, record)
        for rel in record['outputs']:
            if Path(rel).name == 'COMPLETE.json':
                from gamingmech.dpo import verify_complete
                verify_complete((root / rel).parent)
        print(f'[resume] verified {key}', flush=True)
        return
    paths = compute()
    manifest = {str(Path(p).relative_to(root)): sha256(p) for p in paths}
    if not manifest:
        raise RuntimeError('Substage returned no outputs: ' + key)
    atomic_json(marker, {'completed': True, 'outputs': manifest})


def cached_tensor(directory: Path, name: str, key, compute):
    import torch
    from gamingmech.runtime import atomic_torch_save, tick
    folder = directory / 'activation_cache'; folder.mkdir(parents=True, exist_ok=True)
    identity = fingerprint([name, key])
    p = folder / (name + '.pt'); meta = folder / (name + '.json')
    if meta.exists():
        record = load_json(meta)
        if record.get('identity') != identity or not p.is_file() or sha256(p) != record['sha256']:
            raise RuntimeError('Changed activation cache: ' + str(p))
        return torch.load(p, map_location='cpu', weights_only=True)
    tick()
    value = compute().detach().cpu()
    if not torch.isfinite(value).all():
        raise RuntimeError('Nonfinite activation tensor')
    atomic_torch_save(value, p)
    atomic_json(meta, {'identity': identity, 'sha256': sha256(p), 'shape': list(value.shape)})
    return value


RENAME_PROVENANCE = {"judge_protocol": "provenance_judge_protocol", "judge_hf_id": "provenance_judge_hf_id", "judge_revision": "provenance_judge_revision"}


def judge_cached(lm, cfg, rows, items, directory):
    """Retain legacy r8 measurements with complete raw context for a later independent audit.

    The main-study evaluator is intentionally unchanged. No claim of calibrated
    correctness, explicit answer switching or joint praise/omission follows from it.
    """
    import copy
    import gc
    import torch
    from gamingmech.behavior import judge_generations, judge_model
    from gamingmech.runtime import resolve_revision, tick
    revision = resolve_revision(cfg, cfg.eval.judge_hf_id)
    cache = directory / 'judge_cache'; cache.mkdir(parents=True, exist_ok=True)
    output = [None] * len(rows); pending = []
    for index, row in enumerate(rows):
        identity = fingerprint(['legacy_r8_judge_v1', cfg.eval.judge_hf_id, revision, row, items[row['item_id']]])
        path = cache / (identity + '.json')
        if path.exists():
            record = load_json(path)
            if record.get('identity') != identity or record.get('output_sha256') != fingerprint(record['output']):
                raise RuntimeError('Corrupt judgment cache: ' + str(path))
            output[index] = {RENAME_PROVENANCE.get(k, k): v for k, v in record['output'].items()}
        else:
            pending.append((index, identity, path, row))
    if not pending:
        print('[resume] all legacy judge responses cached', flush=True)
        return output
    tick(); judge = judge_model(cfg)
    try:
        for begin in range(0, len(pending), cfg.eval.batch_size):
            tick(); batch = pending[begin:begin + cfg.eval.batch_size]
            candidates = [copy.deepcopy(row) for _, _, _, row in batch]
            judge_generations(lm, cfg, candidates, items, judge=judge)
            for (index, identity, path, _), annotated in zip(batch, candidates):
                annotated['provenance_judge_protocol'] = 'legacy_r8_judge_v1'
                annotated['provenance_judge_hf_id'] = cfg.eval.judge_hf_id
                annotated['provenance_judge_revision'] = revision
                atomic_json(path, {'identity': identity, 'output': annotated,
                                   'output_sha256': fingerprint(annotated)})
                output[index] = annotated
            print(f'[judge] {min(begin + len(batch),len(pending))}/{len(pending)} new responses', flush=True)
    finally:
        del judge; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return output
