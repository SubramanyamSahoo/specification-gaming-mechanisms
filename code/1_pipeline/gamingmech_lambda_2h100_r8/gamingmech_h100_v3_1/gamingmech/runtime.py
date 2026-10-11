"""Immutable experiment identities, source/revision locks, budget checks and atomic state."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import torch
from .common import save_json, load_json, stable_hash
from .config import Config, SourceSpec

_ACTIVE: Optional[Config] = None

class BudgetExceeded(RuntimeError):
    pass


def activate(cfg: Config):
    global _ACTIVE
    _ACTIVE = cfg


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def source_hash() -> str:
    root = Path(__file__).parent
    return stable_hash(*[(p.name, file_hash(p)) for p in sorted(root.glob('*.py'))], n=64)


def from_dict(d: dict) -> Config:
    c = Config()
    def put(obj, values):
        for k, v in values.items():
            if not hasattr(obj, k): raise ValueError(f'Unknown configuration field {k}')
            cur = getattr(obj, k)
            if dataclasses.is_dataclass(cur): put(cur, v)
            elif k in ('eval_sources', 'real_sources'):
                setattr(obj, k, [SourceSpec(**x) for x in v])
            elif isinstance(cur, tuple): setattr(obj, k, tuple(v))
            else: setattr(obj, k, v)
    put(c, d)
    return c


def initialize(cfg: Config) -> Path:
    from .config import validate_config
    validate_config(cfg)
    root = cfg.exp_dir.resolve()
    if root.exists(): raise FileExistsError(f'Project already exists: {root}; use --project to resume')
    root.mkdir(parents=True)
    cfg.runtime.project = str(root)
    activate(cfg)
    snap = root / 'source_snapshot'
    shutil.copytree(Path(__file__).parent, snap, ignore=shutil.ignore_patterns('__pycache__'))
    save_json({'config': cfg.to_dict(), 'fingerprint': cfg.fingerprint(), 'source_hash': source_hash(),
               'created_utc': datetime.now(timezone.utc).isoformat()}, root / 'protocol.json')
    save_json({'python': platform.python_version(), 'torch': torch.__version__,
               'platform': platform.platform()}, root / 'environment.json')
    return root


def load_project(root: str, model_key: str, phase: str = 'full') -> Config:
    root = Path(root).resolve()
    if (root / 'HANDOFF_FAILED.json').exists():
        raise RuntimeError('Incomplete calibration handoff; inspect HANDOFF_FAILED.json and initialize a fresh destination')
    saved = load_json(root / 'protocol.json')
    cfg = from_dict(saved['config'])
    cfg.runtime.project, cfg.model_key, cfg.runtime.phase = str(root), model_key, phase
    if cfg.runtime.study_role == 'calibration' and phase != 'pilot':
        raise RuntimeError('Calibration projects are pilot-only; no full/test/mechanism run may use them')
    if cfg.fingerprint() != saved['fingerprint'] or source_hash() != saved['source_hash']:
        raise RuntimeError('Source or scientific configuration changed. Create a NEW project; do not reuse old caches.')
    activate(cfg)
    return cfg


def billing_start(cfg: Config, rate: float, started: str):
    dt = datetime.fromisoformat(started.replace('Z', '+00:00'))
    if dt.tzinfo is None or rate <= 0 or dt.timestamp() > time.time():
        raise ValueError('Use a positive hourly rate and a past, timezone-aware actual billing start')
    path = cfg.exp_dir / 'billing.json'
    if path.exists(): raise FileExistsError('Billing clock exists; preserve it rather than resetting spent cost')
    save_json({'hourly_usd': rate, 'started_utc': dt.isoformat(), 'active': True, 'segments': []}, path)


def budget(cfg: Config) -> dict:
    path = cfg.exp_dir / 'billing.json'
    if not path.exists():
        if cfg.runtime.device == 'cpu': return {'spent_usd': 0.0, 'remaining_usd': float('inf')}
        raise RuntimeError('Set the actual billing clock before starting a GPU job')
    d = load_json(path)
    elapsed = time.time() - datetime.fromisoformat(d['started_utc']).timestamp() if d.get('active', True) else 0.
    spent = elapsed / 3600 * d['hourly_usd']
    total_hours = elapsed / 3600
    for seg in d.get('segments', []):
        hours = (datetime.fromisoformat(seg['ended_utc']).timestamp() - datetime.fromisoformat(seg['started_utc']).timestamp()) / 3600
        spent += hours * seg['hourly_usd']; total_hours += hours
    return {'elapsed_hours': total_hours, 'clock_active': d.get('active', True), 'spent_usd': spent,
            'remaining_usd': cfg.runtime.budget_usd - spent,
            'work_remaining_usd': cfg.runtime.budget_usd - cfg.runtime.reserve_usd - spent}


def tick():
    if _ACTIVE is None: return
    b = budget(_ACTIVE)
    if not b.get('clock_active', True): raise RuntimeError('Billing clock is paused; record the actual next billed segment before GPU execution')
    if b.get('work_remaining_usd', float('inf')) <= 0:
        raise BudgetExceeded(f'Compute reserve reached: {b}. Provider billing continues until YOU stop the instance.')


def resolve_revision(cfg: Config, repo: str, kind: str = 'model') -> str:
    p = cfg.exp_dir / 'revisions.json'
    d = load_json(p) if p.exists() else {}
    key = f'{kind}:{repo}'
    if key not in d:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get(cfg.runtime.hf_token_env))
        info = api.dataset_info(repo) if kind == 'dataset' else api.model_info(repo)
        if not info.sha: raise RuntimeError(f'No immutable revision for {repo}')
        d[key] = info.sha
        save_json(d, p)
    return d[key]


def active_revision(repo: str, kind: str = 'dataset') -> Optional[str]:
    return resolve_revision(_ACTIVE, repo, kind) if _ACTIVE is not None else None


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(obj, tmp)
        with tmp.open('rb') as f: os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def data_identity(cfg: Config) -> str:
    root = cfg.data_dir / 'artifacts'
    paths = [root / n for n in ('roster.jsonl', 'real_pairs.jsonl', 'eval_items.jsonl')]
    if not all(p.exists() for p in paths): raise RuntimeError('Data artifacts are incomplete')
    return stable_hash(*[(p.name, file_hash(p)) for p in paths], n=64)


def gate_identity(cfg: Config) -> str:
    rows = []
    for c in cfg.conditions():
        p = cfg.adapter_dir(c) / 'final' / 'COMPLETE.json'
        if not p.exists(): raise RuntimeError(f'Incomplete adapter: {c.name}')
        rows.append((c.name, file_hash(p)))
    return stable_hash(cfg.fingerprint(), data_identity(cfg), *rows, n=64)


def require_gate(cfg: Config):
    if cfg.runtime.study_role == 'calibration':
        raise RuntimeError('A calibration gate is diagnostic only and never authorizes confirmatory or mechanism stages')
    p = cfg.exp_dir / cfg.model_key / "04_behavior" / cfg.runtime.phase / "dev" / "artifacts" / f"gate_{cfg.runtime.phase}.json"
    if not p.exists(): raise RuntimeError('Run the development behavior stage before mechanism analysis')
    d = load_json(p)
    if not d.get('passed') or d.get('identity') != gate_identity(cfg):
        raise RuntimeError('Missing, failed or stale behavior gate; downstream computation stopped')
    from .dpo import verify_complete
    for cond in cfg.conditions(): verify_complete(cfg.adapter_dir(cond) / 'final')


@contextmanager
def project_lock(cfg: Config):
    import fcntl
    p = cfg.exp_dir / '.execution.lock'
    with p.open('a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This project is already running in another process')
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def partition(cfg: Config, name: str):
    old = cfg.runtime.partition
    cfg.runtime.partition = name
    try: yield
    finally: cfg.runtime.partition = old


def split_name(cfg: Config, group: str) -> str:
    from .common import hash_unit
    u = hash_unit(cfg.data.roster_seed, 'question-partition-v3', group)
    if u < cfg.runtime.fit_fraction: return 'fit'
    if u < cfg.runtime.fit_fraction + cfg.runtime.dev_fraction: return 'dev'
    return 'test'


def preflight(cfg: Config):
    if cfg.runtime.device != 'cpu':
        if not os.environ.get(cfg.runtime.hf_token_env): raise RuntimeError('Set HF_TOKEN')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; CPU fallback is forbidden')
        g = torch.cuda.get_device_properties(0)
        if g.total_memory / 2**30 < cfg.runtime.min_gpu_gib: raise RuntimeError('GPU has insufficient memory for this profile')
        if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unsupported')
        if shutil.disk_usage(cfg.exp_dir).free / 2**30 < cfg.runtime.min_free_disk_gib:
            raise RuntimeError('Insufficient free disk; configure a larger results volume')
    for key in ('qwen', 'llama'):
        from .config import MODEL_REGISTRY
        resolve_revision(cfg, MODEL_REGISTRY[key].hf_id)
    source_checks = []
    from datasets import load_dataset
    required = {"sciq": {"question", "correct_answer", "distractor1", "distractor2", "distractor3"},
                "arc": {"question", "choices", "answerKey"},
                "truthfulqa": {"question", "best_answer", "incorrect_answers"},
                "orca": {"question", "chosen", "rejected"},
                "ultrafeedback": {"chosen", "rejected"}, "hh": {"chosen", "rejected"},
                "mmlu": {"question", "choices", "answer"}}
    for spec in [cfg.data.train_source, *cfg.data.eval_sources, *cfg.data.real_sources, cfg.data.capability_source]:
        revision = resolve_revision(cfg, spec.hf_id, 'dataset')
        opts = {"split": spec.split, "streaming": True, "revision": revision,
                "token": os.environ.get(cfg.runtime.hf_token_env)}
        if spec.data_dir: opts["data_dir"] = spec.data_dir
        dataset = load_dataset(spec.hf_id, spec.hf_config, **opts)
        row = next(iter(dataset), None)
        if row is None or not required[spec.kind].issubset(row):
            raise RuntimeError(f"Empty/changed upstream schema: {spec.name}")
        source_checks.append({"source": spec.name, "revision": revision, "sample_keys": sorted(row), "passed": True})
    save_json(source_checks, cfg.exp_dir / "dataset_schema_checks.json")
    # Exercise BOTH actual HF architectures, exact PEFT targets and non-reentrant gradients.
    from .integration import run_checks
    checks = run_checks(device=cfg.runtime.device, dtype=getattr(torch, cfg.runtime.dtype))
    save_json(checks, cfg.exp_dir / "architecture_checks.json")
    from transformers import AutoTokenizer
    token = os.environ.get(cfg.runtime.hf_token_env)
    for key in ("qwen", "llama"):
        spec = MODEL_REGISTRY[key]
        tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=resolve_revision(cfg, spec.hf_id), token=token)
        messages = [{"role":"user","content":"State the answer briefly."}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **dict(spec.chat_kwargs))
        if not prompt or spec.eot_token not in tok.get_vocab():raise RuntimeError(f"Bad template/end token: {key}")
        # Require offsets: mechanistic content masks depend on a fast tokenizer.
        enc = tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
        if not enc["offset_mapping"]: raise RuntimeError("Tokenizer offsets missing")
    # The independent judge must be reachable before any paid training, and its answer tokens must be distinct.
    from .behavior import JUDGE_LETTERS
    judge_id = cfg.eval.judge_hf_id
    jtok = AutoTokenizer.from_pretrained(judge_id, revision=resolve_revision(cfg, judge_id), token=token)
    if cfg.eval.judge_eot_token not in jtok.get_vocab(): raise RuntimeError("Bad judge end token")
    jtok.apply_chat_template([{"role": "user", "content": "Answer A or B."}], tokenize=False, add_generation_prompt=True)
    firsts = [jtok(x, add_special_tokens=False)["input_ids"][0] for x in (*JUDGE_LETTERS, "Yes", "No")]
    if len(set(firsts)) != len(firsts): raise RuntimeError("Judge answer tokens are not distinct")
    freeze =subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True)
    (cfg.exp_dir / 'installed_packages.txt').write_text(freeze)
    save_json({'passed': True, 'source_hash': source_hash(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
               'budget': budget(cfg)}, cfg.exp_dir / 'preflight.json')


def cached_json(cfg: Config, stage: str, key, compute):
    """Per-operation cache inside a source/config/data-locked project. Atomic receipt last."""
    if not callable(compute):
        raise TypeError('cached_json requires a zero-argument callable; pass compute, not compute()')
    folder = cfg.stage_dir(stage) / 'artifacts' / 'operation_cache'
    identity = stable_hash(cfg.fingerprint(), cfg.model_key, key, n=64)
    output, receipt = folder / (identity + '.json'), folder / (identity + '.complete.json')
    if receipt.exists():
        d = load_json(receipt)
        if not output.exists() or d['sha256'] != file_hash(output): raise RuntimeError('Corrupt operation cache')
        return load_json(output)
    value = compute()
    save_json(value, output)
    save_json({'sha256': file_hash(output)}, receipt)
    return value


def billing_pause(cfg: Config, ended: str):
    path = cfg.exp_dir / 'billing.json'; d = load_json(path)
    if not d.get('active', True): raise RuntimeError('Clock already paused')
    dt = datetime.fromisoformat(ended.replace('Z', '+00:00'))
    if dt.tzinfo is None or not datetime.fromisoformat(d['started_utc']).timestamp() <= dt.timestamp() <= time.time():
        raise ValueError('Use the actual past provider billing end, after segment start')
    d.setdefault('segments', []).append({'started_utc': d['started_utc'], 'hourly_usd': d['hourly_usd'], 'ended_utc': dt.isoformat()})
    d['active'] = False; save_json(d, path)


def billing_resume(cfg: Config, rate: float, started: str):
    path = cfg.exp_dir / 'billing.json'; d = load_json(path)
    if d.get('active', True): raise RuntimeError('Clock already active')
    dt = datetime.fromisoformat(started.replace('Z', '+00:00'))
    end = datetime.fromisoformat(d['segments'][-1]['ended_utc']).timestamp()
    if rate <= 0 or dt.tzinfo is None or not end <= dt.timestamp() <= time.time():
        raise ValueError('Use a positive rate and actual, nonoverlapping provider billing start')
    d.update(hourly_usd=rate, started_utc=dt.isoformat(), active=True); save_json(d, path)
