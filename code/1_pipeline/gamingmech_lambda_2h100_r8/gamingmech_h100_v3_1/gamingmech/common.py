"""Shared utilities: logging, seeding, IO, statistics, batching and figure saving."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

_LOGGERS: Dict[str, logging.Logger] = {}
EPS = float(torch.finfo(torch.float32).eps)       # numerical floor for divisions


def get_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    key = f"{name}:{log_dir}"
    if key in _LOGGERS:
        return _LOGGERS[key]
    logger = logging.getLogger(key)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.name = name
    _LOGGERS[key] = logger
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def hf_login(env_name: str) -> Optional[str]:
    """Authenticate with the Hugging Face Hub from an environment variable."""
    token = os.environ.get(env_name)
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    return token


def stable_hash(*parts: object, n: int = 16) -> str:
    return hashlib.sha256("||".join(str(p) for p in parts).encode()).hexdigest()[:n]


def hash_unit(*parts: object) -> float:
    """Deterministic uniform number in [0, 1) derived from the arguments."""
    return int(stable_hash(*parts, n=13), 16) / float(16 ** 13)


# --------------------------------------------------------------------------- IO
def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dataclass_fields__"):
        import dataclasses
        return dataclasses.asdict(o)
    return str(o)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, default=_default)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_default) + "\n")


def load_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    if not keys:
        keys = ["status"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (_default(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
                        for k, v in r.items()})


class CsvLogger:
    """Append only CSV log that flushes every row so partial runs are usable."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self._fields, self._fh, self._writer = path, None, None, None

    def log(self, row: dict) -> None:
        if self._writer is None:
            self._fields = list(row.keys())
            exists = self.path.exists() and self.path.stat().st_size > 0
            if exists:
                with open(self.path, newline="") as r:
                    old = next(csv.reader(r))
                if old != self._fields:
                    raise ValueError(f"CSV schema changed: {self.path}")
            self._fh = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields, extrasaction="raise")
            if not exists: self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


@contextmanager
def timed(logger: logging.Logger, label: str):
    t0 = time.time()
    yield
    logger.info(f"{label} finished in {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------- batching
def batched_map(fn: Callable[[List], List], items: Sequence, batch_size: int,
                lengths: Optional[Sequence[int]] = None, min_batch: int = 1,
                desc: str = "", logger: Optional[logging.Logger] = None) -> List:
    """Apply fn over batches, longest first, halving the batch on CUDA out of memory.

    Results are returned in the original order of items.
    """
    order = list(range(len(items)))
    if lengths is not None:
        order.sort(key=lambda i: -lengths[i])
    out: List = [None] * len(items)
    i, bs, n_done = 0, max(batch_size, 1), 0
    while i < len(order):
        from .runtime import tick
        tick()
        idx = order[i:i + bs]
        try:
            res = fn([items[j] for j in idx])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= min_batch:
                raise
            bs = max(bs // 2, min_batch)
            if logger:
                logger.info(f"{desc}: out of memory, batch size reduced to {bs}")
            continue
        if len(res) != len(idx):
            raise ValueError("Batched function returned the wrong number of rows")
        for j, r in zip(idx, res):
            out[j] = r
        i += len(idx)
        n_done += len(idx)
        if logger and desc and (i == len(order) or (n_done // bs) % 20 == 0):
            logger.info(f"{desc}: {n_done}/{len(order)}")
    return out


# --------------------------------------------------------------------------- statistics
def bootstrap_ci(x: np.ndarray, n_boot: int, ci: float, seed: int) -> Dict[str, float]:
    """Mean with a percentile bootstrap interval over the first axis."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    a = (1.0 - ci) / 2.0
    return {"mean": float(x.mean()), "lo": float(np.quantile(means, a)),
            "hi": float(np.quantile(means, 1.0 - a)), "n": int(x.size)}


def paired_effect(a: np.ndarray, b: np.ndarray, n_boot: int, ci: float, seed: int) -> Dict[str, float]:
    """Paired comparison a minus b on the same prompts: mean difference, CI, dz, sign flip p."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired arrays differ in shape")
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    res = bootstrap_ci(d, n_boot, ci, seed)
    sd = d.std(ddof=1) if d.size > 1 else float("nan")
    res["dz"] = float(d.mean() / sd) if sd and sd > 0 else float("nan")
    if d.size > 1:
        rng = np.random.default_rng(seed + 1)
        signs = rng.choice([-1.0, 1.0], size=(n_boot, d.size))
        null = np.abs((signs * d).mean(axis=1))
        res["p_signflip"] = float((1 + (null >= abs(d.mean())).sum()) / (n_boot + 1))
    else:
        res["p_signflip"] = float("nan")
    return {f"diff_{k}" if k in ("mean", "lo", "hi") else k: v for k, v in res.items()}


# --------------------------------------------------------------------------- figures
def save_figure(fig, path_stem: Path, formats: Sequence[str], dpi: int) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(f"{path_stem}.{ext}", dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
