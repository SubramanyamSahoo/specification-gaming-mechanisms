"""Direct Preference Optimization with LoRA.

loss = -log sigmoid( beta * [ (log pi(y_c|x) - log pi_ref(y_c|x)) - (log pi(y_r|x) - log pi_ref(y_r|x)) ] )

The reference policy is the frozen starting checkpoint. Its log probabilities depend only on
the text, so they are computed once per model and reused by every condition and seed.
"""
from __future__ import annotations

import math
import os
import shutil
import numpy as np
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .common import CsvLogger, get_logger, save_csv, save_json, set_seed, stable_hash
from .config import Config, ConditionSpec
from .data import training_pairs
from .gen import response_logprob_batch, response_logprobs
from .models import LM


def tokenize_pairs(lm: LM, pairs: List[dict], max_tokens: int) -> Tuple[List[dict], int]:
    out, dropped = [], 0
    for p in pairs:
        c_ids, c_start = lm.encode_dialogue(p["context"], p["chosen"])
        r_ids, r_start = lm.encode_dialogue(p["context"], p["rejected"])
        if max(len(c_ids), len(r_ids)) > max_tokens:
            dropped += 1
            continue
        out.append({**p, "c_ids": c_ids, "c_start": c_start, "r_ids": r_ids, "r_start": r_start,
                    "c_key": stable_hash(lm.hf_id, getattr(lm, "revision", "local"), c_start, c_ids, n=64), "r_key": stable_hash(lm.hf_id, getattr(lm, "revision", "local"), r_start, r_ids, n=64)})
    return out, dropped


def ensure_reference(lm: LM, cfg: Config, examples: List[dict], cache_path: Path, logger) -> Dict[str, float]:
    """Reference log probabilities for every sequence, computed with the base model and cached."""
    cache: Dict[str, float] = torch.load(cache_path, weights_only=True) if cache_path.exists() else {}
    todo: Dict[str, Tuple[List[int], int]] = {}
    for e in examples:
        for k, ids, start in ((e["c_key"], e["c_ids"], e["c_start"]), (e["r_key"], e["r_ids"], e["r_start"])):
            if k not in cache:
                todo[k] = (ids, start)
    if todo:
        logger.info(f"reference pass over {len(todo)} new sequences")
        keys = list(todo)
        with lm.using(None):
            lp = response_logprobs(lm, [todo[k][0] for k in keys], [todo[k][1] for k in keys],
                                   cfg.train.ref_batch_size)
        cache.update({k: float(v) for k, v in zip(keys, lp)})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        from .runtime import atomic_torch_save
        atomic_torch_save(cache, cache_path)
    return cache


def make_batches(examples: List[dict], cfg: Config, rng: random.Random) -> List[List[dict]]:
    """Length bucketed batches bounded by pair count and by padded token count."""
    t = cfg.train
    order = list(examples)
    rng.shuffle(order)
    window = t.pair_batch_size * t.sort_window
    batches: List[List[dict]] = []
    for i in range(0, len(order), window):
        chunk = sorted(order[i:i + window], key=lambda e: max(len(e["c_ids"]), len(e["r_ids"])))
        cur: List[dict] = []
        for e in chunk:
            width = max(len(e["c_ids"]), len(e["r_ids"]))       # chunk is sorted, so this is the batch width
            if cur and (len(cur) >= t.pair_batch_size or 2 * (len(cur) + 1) * width > t.max_batch_tokens):
                batches.append(cur)
                cur = []
            cur.append(e)
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def dpo_terms(lm: LM, batch: List[dict], ref: Dict[str, float], beta: float) -> Dict[str, torch.Tensor]:
    n = len(batch)
    lp = response_logprob_batch(lm, [e["c_ids"] for e in batch] + [e["r_ids"] for e in batch],
                                [e["c_start"] for e in batch] + [e["r_start"] for e in batch])
    ref_c = torch.tensor([ref[e["c_key"]] for e in batch], device=lp.device, dtype=lp.dtype)
    ref_r = torch.tensor([ref[e["r_key"]] for e in batch], device=lp.device, dtype=lp.dtype)
    reward_c, reward_r = beta * (lp[:n] - ref_c), beta * (lp[n:] - ref_r)
    margin = reward_c - reward_r
    return {"loss": -F.logsigmoid(margin), "margin": margin, "reward_c": reward_c, "reward_r": reward_r}


@torch.no_grad()
def evaluate(lm: LM, cfg: Config, examples: List[dict], ref: Dict[str, float]) -> Tuple[dict, List[dict]]:
    lm.model.eval()
    rows = []
    for batch in make_batches(examples, cfg, random.Random(0)):
        terms = dpo_terms(lm, batch, ref, cfg.train.beta)
        for i, e in enumerate(batch):
            rows.append({"pair_id": e["pair_id"], "family": e["family"], "flipped": e["flipped"],
                         **{k: float(v[i]) for k, v in terms.items()}})
    if not rows: raise ValueError("Validation corpus is empty")
    agg = {"val_loss": sum(r["loss"] for r in rows) / len(rows),
           "val_margin": sum(r["margin"] for r in rows) / len(rows),
           "val_accuracy": sum(r["margin"] > 0 for r in rows) / len(rows)}
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        agg[f"val_accuracy_{fam}"] = sum(r["margin"] > 0 for r in sub) / len(sub)
    lm.model.train()
    return agg, rows


def verify_complete(path: Path) -> dict:
    from .runtime import file_hash
    manifest = path / "COMPLETE.json"
    if not manifest.exists(): raise RuntimeError(f"Incomplete adapter {path}")
    from .common import load_json
    d = load_json(manifest)
    if not d.get("files") or not any(n.endswith((".safetensors", ".bin")) for n in d["files"]):
        raise RuntimeError("Completion manifest does not contain weights")
    for n, sha in d["files"].items():
        if not (path / n).is_file() or file_hash(path / n) != sha:
            raise RuntimeError(f"Corrupt or missing adapter file: {path / n}")
    return d


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(d):
    random.setstate(d["python"]); np.random.set_state(d["numpy"]); torch.set_rng_state(d["torch"].cpu())
    if d["cuda"] is not None: torch.cuda.set_rng_state_all([x.cpu() for x in d["cuda"]])


def _save_checkpoint(lm, name, path, state):
    from .runtime import atomic_torch_save
    import uuid
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex[:8])
    lm.save_adapter(name, tmp)
    if path.name.startswith("step_"):
        save_json({k: state[k] for k in ("identity", "step", "total", "elapsed_s")}, tmp / "checkpoint.json")
    else:
        atomic_torch_save(state, tmp / "training_state.pt")
    old = path.with_name(path.name + ".previous")
    if old.exists(): shutil.rmtree(old)
    if path.exists(): os.replace(path, old)
    os.replace(tmp, path)
    if old.exists(): shutil.rmtree(old)


def _load_weights(lm, name, path):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    weights = load_file(str(path / "adapter_model.safetensors"))
    set_peft_model_state_dict(lm.model, weights, adapter_name=name)


def train_condition(lm: LM, cfg: Config, cond: ConditionSpec, ref_cache: Path) -> dict:
    """Fixed-update, sample-weighted DPO with full-state resume and bounded OOM retry."""
    from .runtime import atomic_torch_save, file_hash, tick, data_identity
    from .common import load_json
    t, out = cfg.train, cfg.adapter_dir(cond)
    out.mkdir(parents=True, exist_ok=True)
    logger = get_logger("train", out / "logs")
    raw_train, raw_val = training_pairs(cfg, cond, "train"), training_pairs(cfg, cond, "val")
    train, drop_t = tokenize_pairs(lm, raw_train, t.max_seq_tokens)
    val, drop_v = tokenize_pairs(lm, raw_val, t.max_seq_tokens)
    if not train or not val: raise RuntimeError("Empty train/validation corpus")
    if (drop_t + drop_v) / (len(raw_train) + len(raw_val)) > t.max_drop_fraction:
        raise RuntimeError("Too many examples exceed the token cap; revise the data before a new protocol")
    identity = stable_hash(cfg.fingerprint(), cond.name, t, getattr(lm, "revision", "local"),
                           *[(e["pair_id"], e["c_key"], e["r_key"]) for e in train + val], n=64)
    if (out / "final").exists():
        d = verify_complete(out / "final")
        if d["identity"] != identity: raise RuntimeError("Cached adapter has stale training/data identity")
        logger.info(f"{cond.name}: verified completed adapter; reuse")
        return load_json(out / "summary.json")
    save_json({"identity": identity, "condition": cond, "n_train": len(train), "n_val": len(val),
               "dropped_train": drop_t, "dropped_val": drop_v}, out / "training_manifest.json")
    # Every synthetic condition must retain the SAME paired row set after token filtering.
    if cond.objective != "real":
        roster_id = stable_hash(*sorted(e["pair_id"] for e in train + val), n=64)
        rp = cfg.stage_dir("03_train") / "artifacts" / "retained_synthetic_roster.json"
        if rp.exists() and load_json(rp)["identity"] != roster_id:
            raise RuntimeError("Synthetic conditions retained different rows after tokenization")
        save_json({"identity": roster_id}, rp)
    ref = ensure_reference(lm, cfg, train + val, ref_cache, logger)
    set_seed(cond.seed)
    lm.new_adapter(cond.name, t.lora, cond.seed)
    params = [p for p in lm.model.parameters() if p.requires_grad]
    if not params: raise RuntimeError("No trainable LoRA parameters")
    opt = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay,
                            fused=t.fused_adamw and lm.device.type == "cuda")
    batches, rng = [], random.Random(cond.seed)
    epochs = 0
    budget = cfg.update_budget
    while (len(batches) < budget * t.grad_accum if budget else epochs < t.epochs):
        batches.extend(make_batches(train, cfg, rng)); epochs += 1
    windows = [batches[i:i+t.grad_accum] for i in range(0, len(batches), t.grad_accum)]
    if budget: windows = windows[:budget]
    total = len(windows)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, round(total * t.warmup_fraction), total)
    ckpts = {max(1, round(f * total)): f for f in t.checkpoint_fractions}
    step, elapsed_prior = 0, 0.0
    resume = out / "last"
    if not resume.exists() and (out / "last.previous").exists():
        os.replace(out / "last.previous", resume)
    if resume.exists():
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        if state["identity"] != identity or state["total"] != total:
            raise RuntimeError("Resume identity differs from the locked protocol")
        _load_weights(lm, cond.name, resume)
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        step, elapsed_prior = state["step"], state["elapsed_s"]
        _restore_rng(state["rng"])
        logger.info(f"Resuming {cond.name} from completed optimizer step {step}/{total}")
    attempt = time.time_ns()
    tlog, vlog = CsvLogger(out / f"train_{attempt}.csv"), CsvLogger(out / f"validation_{attempt}.csv")
    t0, agg, rows = time.time(), {}, []
    last_good_rng = _rng_state()
    def state_now():
        return {"identity": identity, "step": step, "total": total, "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "rng": _rng_state(), "elapsed_s": elapsed_prior + time.time() - t0}
    try:
        while step < total:
            tick()
            window = windows[step]
            denom = sum(len(batch) for batch in window)
            attempt_rng, micro_limit = _rng_state(), t.pair_batch_size
            for retry in range(t.max_train_retries + 1):
                opt.zero_grad(set_to_none=True); _restore_rng(attempt_rng)
                totals = {"loss": 0.0, "margin": 0.0, "reward_c": 0.0, "reward_r": 0.0}
                try:
                    for batch in window:
                        for a in range(0, len(batch), micro_limit):
                            terms = dpo_terms(lm, batch[a:a+micro_limit], ref, t.beta)
                            if not all(torch.isfinite(v).all() for v in terms.values()):
                                raise FloatingPointError("Nonfinite DPO values")
                            (terms["loss"].sum() / denom).backward()
                            for k, v in terms.items(): totals[k] += float(v.detach().sum()) / denom
                            del terms
                    break
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    import gc
                    gc.collect(); torch.cuda.empty_cache()
                    if micro_limit <= 1 or retry == t.max_train_retries: raise
                    micro_limit = max(1, micro_limit // 2)
                    logger.info(f"OOM: retry SAME optimizer update with microbatch {micro_limit}; sample weighting unchanged")
            norm = torch.nn.utils.clip_grad_norm_(params, t.max_grad_norm, error_if_nonfinite=True)
            opt.step(); sched.step(); step += 1
            last_good_rng = _rng_state()
            tlog.log({"step": step, "n_pairs": denom, "micro_limit": micro_limit, **totals,
                      "grad_norm": float(norm), "lr": sched.get_last_lr()[0],
                      "elapsed_s": elapsed_prior + time.time() - t0})
            if step % t.eval_every == 0 or step == total:
                agg, rows = evaluate(lm, cfg, val, ref)
                vlog.log({"step": step, **agg})
                logger.info(f"{cond.name} {step}/{total}: loss={totals['loss']:.5f} val={agg['val_loss']:.5f}")
            if step in ckpts:
                _save_checkpoint(lm, cond.name, out / f"step_{step:05d}", state_now())
            if step % t.resume_every == 0 or step == total:
                _save_checkpoint(lm, cond.name, out / "last", state_now())
        if not rows: agg, rows = evaluate(lm, cfg, val, ref)
        summary = {"condition": cond.name, "objective": cond.objective, "flip_rate": cond.flip_rate,
                   "seed": cond.seed, "n_train": len(train), "n_val": len(val), "dropped_long": drop_t + drop_v,
                   "n_flipped_train": sum(e["flipped"] for e in train), "steps": step,
                   "checkpoints": {f"step_{s:05d}": f for s, f in sorted(ckpts.items())},
                   "train_seconds": elapsed_prior + time.time() - t0, "final": agg, "model": lm.hf_id,
                   "config_fingerprint": cfg.fingerprint(), "training_identity": identity}
        save_csv(rows, out / "val_pairs_final.csv"); save_json(summary, out / "summary.json")
        temp = out / "final.tmp"
        if temp.exists(): shutil.rmtree(temp)
        lm.save_adapter(cond.name, temp)
        manifest = {p.name: file_hash(p) for p in temp.iterdir() if p.is_file()}
        save_json({"identity": identity, "files": manifest, "step": step}, temp / "COMPLETE.json")
        verify_complete(temp)
        os.replace(temp, out / "final")
        return summary
    except BaseException:
        # Save the last completed optimizer state only. Partial gradients are discarded.
        opt.zero_grad(set_to_none=True)
        _restore_rng(last_good_rng)
        _save_checkpoint(lm, cond.name, out / "last", state_now())
        raise
    finally:
        tlog.close(); vlog.close()
        lm.drop_adapter(cond.name); lm.model.eval()
        if hasattr(lm.model, "gradient_checkpointing_disable"): lm.model.gradient_checkpointing_disable()
        torch.cuda.empty_cache()


def train_plan(lm: LM, cfg: Config, conditions: List[ConditionSpec]) -> None:
    """Stage 03: train every condition in the plan and write the combined summary table."""
    stage = cfg.stage_dir("03_train")
    ref_cache = stage / "artifacts" / "reference_logprobs.pt"
    for cond in conditions:
        train_condition(lm, cfg, cond, ref_cache)
    rows = []
    for cond in conditions:
        path = cfg.adapter_dir(cond) / "summary.json"
        if path.exists():
            from .common import load_json
            s = load_json(path)
            rows.append({k: s[k] for k in ("condition", "objective", "flip_rate", "seed", "n_train",
                                           "n_flipped_train", "steps", "train_seconds")} | s["final"])
    save_csv(rows, stage / "tables" / "training_summary.csv")
