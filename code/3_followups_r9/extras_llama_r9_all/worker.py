"""Run one checked experiment in an isolated process; nonzero exit means failure."""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
from pathlib import Path
from r9_support import atomic_json, load_json, sha256


def main(folder: Path, stage: str):
    from coordinator import check_bundle, check_core, manifest_outputs, now, STAGES
    if stage not in STAGES: raise ValueError('Unknown experiment')
    check_bundle()
    run = load_json(folder / 'run.json')
    check_core(Path(run['base']) / 'gamingmech_h100_v3_1')
    import torch
    import experiments
    from gamingmech.common import setup_torch
    from gamingmech.models import LM
    from gamingmech.runtime import activate, load_project, project_lock, require_gate, tick
    project = Path(run['project']); source = Path(run['source'])
    cfg = load_project(str(project), 'llama', 'full')
    require_gate(cfg); activate(cfg); tick(); setup_torch()
    torch.set_num_threads(run['settings']['cpu_threads'])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU is required; no CPU fallback')
    torch.backends.cuda.matmul.allow_tf32 = cfg.runtime.tf32
    torch.backends.cudnn.allow_tf32 = cfg.runtime.tf32
    cfg.runtime.partition = 'test'
    experiments.link_activations(cfg, source)
    artifact = project / 'RESULTS' / stage; artifact.mkdir(parents=True, exist_ok=True)
    atomic_json(artifact / 'EXPERIMENT_CONFIG.json', {
        'stage': stage, 'settings': run['settings'], 'source_protocol_fingerprint': cfg.fingerprint(),
        'source_path': str(source), 'student': cfg.spec.hf_id, 'judge': cfg.eval.judge_hf_id,
        'reference_seed': cfg.reference_seed, 'seeds': list(cfg.seeds), 'created_utc': now(),
        'scope': 'follow-up analysis; not a revision of the original confirmation gate',
        'legacy_judge_warning': 'Mean judge scores are not human-adjudicated rates; switching also includes neither/both/no-answer.',
    })
    start = time.time(); receipt = folder / 'receipts' / (stage + '.json')
    with project_lock(cfg):
        try:
            cache = {} if stage == 'attention' else {'lm': LM(cfg)}
            result = experiments.RUN[stage](cfg, source, cache)
            if result is None or result == {}:
                raise RuntimeError('The experiment produced an empty result')
            # A graph/report failure remains visible; it never changes saved numerical values.
            from results_report import plot_experiment, write_index
            plot_experiment(project, stage)
            atomic_json(artifact / 'SUMMARY.json', result)
            write_index(project)
            finals = [str(p.relative_to(project)) for p in (project / 'llama' / '03_train' / 'artifacts').glob('*/final')
                      if not p.is_symlink()]
            outputs = manifest_outputs(project, stage)
            atomic_json(receipt, {'status': 'completed', 'stage': stage, 'ended_utc': now(),
                        'elapsed_seconds': time.time() - start, 'outputs': outputs,
                        'new_adapter_finals': finals, 'result': result,
                        'meaning': 'Execution and artifact checks only; no automatic scientific success claim'})
            print('COMPLETED LLAMA EXTRA:', stage, flush=True)
        except BaseException as error:
            atomic_json(receipt, {'status': 'failed', 'stage': stage, 'ended_utc': now(),
                        'elapsed_seconds': time.time() - start, 'error': repr(error)})
            raise

if __name__ == '__main__':
    try:
        main(Path(sys.argv[1]).resolve(), sys.argv[2])
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
