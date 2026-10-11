"""Disposable, development-only calibration and explicit handoff to confirmation.

A calibration gate is diagnostic: its failed outcome is never changed into a pass.
It cannot authorize mechanisms or test-set access. Each trial remains immutable;
changing settings creates another uniquely named trial, not an in-place rewrite.
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .common import load_json, load_jsonl, save_csv, save_json, save_jsonl, stable_hash
from .config import Config, PROBE_ONLY_FAMILIES, apply_overrides, validate_config
from .runtime import (activate, data_identity, file_hash, initialize, load_project,
                      partition, source_hash, tick)


def _require_calibration(cfg: Config) -> None:
    if cfg.runtime.study_role != "calibration" or cfg.runtime.phase != "pilot":
        raise RuntimeError("This operation is for a pilot-only calibration project, not a frozen confirmatory run")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def completion_table(rows: Sequence[dict]) -> list[dict]:
    """Observed lengths include censored outputs; no unobserved tail is imputed."""
    groups = defaultdict(list)
    for row in rows:
        cap = int(row.get("max_new_tokens", -1))
        condition = row["condition"]
        groups[(condition, cap, "ALL", "ALL", "ALL")].append(row)
        groups[(condition, cap, row.get("eval_source", "unknown"),
                row["family"], row["bucket"])].append(row)
    out = []
    for (condition, cap, source, family, bucket), sub in sorted(groups.items()):
        n = len(sub)
        complete = [r for r in sub if r["finish"] == "eos" and bool(r["response"].strip())]
        lengths = np.asarray([int(r["n_tokens"]) for r in sub], dtype=np.float64)
        censored = sum(r["finish"] == "length" for r in sub)
        out.append({"condition": condition, "max_new_tokens": cap, "eval_source": source,
                    "family": family, "bucket": bucket, "n": n, "n_complete": len(complete),
                    "n_censored": censored, "completion_rate": len(complete) / n,
                    "censored_rate": censored / n,
                    "empty_rate": sum(not r["response"].strip() for r in sub) / n,
                    "observed_mean_tokens": float(lengths.mean()),
                    "observed_p50_tokens": float(np.quantile(lengths, .5)),
                    "observed_p90_tokens": float(np.quantile(lengths, .9)),
                    "observed_p95_tokens": float(np.quantile(lengths, .95)),
                    "mean_tokens_complete_only": float(np.mean([r["n_tokens"] for r in complete])) if complete else None,
                    "length_interpretation": "observed; length-limited rows are right-censored, not completed answers"})
    return out


def acceptance_table(cfg: Config) -> list[dict]:
    """Count lexical, independent semantic, and selected-roster outcomes separately."""
    from .data import exact_bucket_counts
    art = cfg.data_dir / "artifacts"
    all_items = _rows(art / "items_all.jsonl")
    roster = _rows(art / "roster.jsonl")
    semantic = _rows(art / "semantic_validation_all.jsonl")
    checked = defaultdict(list)
    for row in semantic:
        checked[row["item_id"]].append(row)
    checked_ids = set(checked)
    passed_ids = {i for i, rs in checked.items()
                  if {r["slot"] for r in rs} == {"a", "b"} and all(r["accepted"] for r in rs)}
    total_train = cfg.data.n_train_pairs - round(cfg.data.n_train_pairs * cfg.data.real_fraction)
    val_total = round(cfg.data.n_train_pairs * cfg.data.val_fraction)
    total_val = val_total - round(val_total * cfg.data.real_fraction)
    train_targets, val_targets = exact_bucket_counts(cfg, total_train), exact_bucket_counts(cfg, total_val)
    keys = sorted(set(train_targets) | {(x["family"], x["bucket"]) for x in all_items})
    out = []
    for family, bucket in keys:
        sub = [x for x in all_items if (x["family"], x["bucket"]) == (family, bucket)]
        ids = {x["item_id"] for x in sub}
        lexical = sum(bool(x.get("valid", False)) for x in sub)
        selected = [x for x in roster if (x["family"], x["bucket"]) == (family, bucket)]
        n_train = sum(x.get("split") == "train" for x in selected)
        n_val = sum(x.get("split") == "val" for x in selected)
        target_train = train_targets.get((family, bucket), 0)
        target_val = val_targets.get((family, bucket), 0)
        out.append({"family": family, "bucket": bucket, "n_candidates": len(sub),
                    "n_lexical_valid": lexical, "n_semantic_checked": len(ids & checked_ids),
                    "n_semantic_passed": len(ids & passed_ids),
                    "lexical_acceptance": lexical / len(sub) if sub else None,
                    "end_to_end_acceptance": len(ids & passed_ids) / len(sub) if sub and semantic else None,
                    "n_train_selected": n_train, "n_val_selected": n_val,
                    "target_train": target_train, "target_val": target_val,
                    "quota_satisfied": bool(semantic) and n_train >= target_train * cfg.runtime.quota_min_fraction
                                         and n_val >= target_val * cfg.runtime.quota_min_fraction})
    return out


def write_summary(cfg: Config) -> Path:
    """CPU-only. Works on incomplete trials, including before successful validation."""
    _require_calibration(cfg)
    out = cfg.stage_dir("00_calibration")
    dev = cfg.exp_dir / cfg.model_key / "04_behavior" / "pilot" / "dev"
    gen_path = dev / "artifacts" / "generations.jsonl"
    rows = _rows(gen_path)
    # Preserve measured status even when only the in-progress file exists.
    if not rows:
        rows = _rows(dev / "artifacts" / "generations_in_progress.jsonl")
    completed = completion_table(rows)
    acceptance = acceptance_table(cfg)
    save_csv(acceptance, out / "tables" / "data_acceptance.csv")
    save_csv(completed, out / "tables" / "completion_by_group.csv")
    gate_file = dev / "artifacts" / "gate_pilot.json"
    gate = load_json(gate_file) if gate_file.exists() else None
    sweep = []
    for p in sorted((out / "artifacts" / "cap_sweeps").glob("*/SUMMARY.json")):
        x = load_json(p)
        sweep.append({"max_new_tokens": x["max_new_tokens"], "summary": str(p.relative_to(cfg.exp_dir)),
                      "n_generations": x["n_generations"], "diagnostic_gate_passed": x["diagnostic_gate_passed"]})
    caps = {int(r["max_new_tokens"]) for r in rows if r.get("max_new_tokens") is not None}
    caps |= {x["max_new_tokens"] for x in sweep}
    receipts = []
    receipt_dir = cfg.exp_dir / cfg.model_key / "receipts" / "pilot"
    for p in sorted(receipt_dir.glob("*.json")):
        d = load_json(p)
        receipts.append({"stage": p.stem, "execution_completed": d.get("passed"), "error": d.get("error")})
    training_exposure = []
    train_root = cfg.exp_dir / cfg.model_key / "03_train" / "artifacts"
    for folder in sorted(train_root.glob("*")):
        if not folder.is_dir(): continue
        steps = {}
        for log in sorted(folder.glob("train_*.csv")):
            for row in _rows(log):
                if row.get("step"): steps[int(row["step"])] = row
        if not steps: continue
        n_pairs = sum(int(float(row.get("n_pairs", 0))) for row in steps.values())
        training_exposure.append({"condition": folder.name, "n_completed_logged_updates": len(steps),
                                  "last_logged_step": max(steps), "logged_pair_exposures": n_pairs,
                                  "mean_pairs_per_logged_update": n_pairs / len(steps),
                                  "configured_max_updates": cfg.update_budget, "lora_rank": cfg.train.lora.r,
                                  "learning_rate": cfg.train.lr, "dpo_beta": cfg.train.beta,
                                  "note": "Restarted step indices deduplicated; inspect optimizer checkpoints for durable progress"})
    save_csv(training_exposure, out / "tables" / "training_exposure.csv")
    evidence = {}
    for name in ("logit_effects.csv", "generation_effects.csv", "capability.csv"):
        p = dev / "tables" / name
        if p.exists(): evidence[name] = {"path": str(p.relative_to(cfg.exp_dir)), "sha256": file_hash(p)}
    doc = {"study_role": "calibration", "confirmatory": False, "model_key": cfg.model_key,
           "source_hash": source_hash(), "config_fingerprint": cfg.fingerprint(),
           "project": str(cfg.exp_dir), "observed_token_caps": sorted(caps),
           "n_generations_available": len(rows), "diagnostic_gate": gate,
           "cap_sweeps": sweep, "stage_outcomes": receipts, "effect_tables": evidence,
           "automatic_only": not cfg.runtime.require_semantic_audit,
           "training_settings": dataclasses.asdict(cfg.train),
           "limits": ["Unreviewed calibration data must not be treated as a validated released dataset.",
                      "No test partition or mechanistic stage is authorized by this report.",
                      "Missing evidence is not filled in and failed gates remain failed.",
                      "Select completion budgets for evaluability, not thresholds solely to make effects significant."]}
    try: doc["data_identity"] = data_identity(cfg)
    except RuntimeError: doc["data_identity"] = None
    ident = stable_hash(json.dumps(doc, sort_keys=True, default=str), n=64)
    hist = out / "artifacts" / "history" / (ident + ".json")
    if not hist.exists(): save_json(doc, hist)
    save_json(doc, out / "artifacts" / "calibration_summary.json")
    text = ("# Calibration only — not confirmatory results\n\n"
            f"Model: `{cfg.model_key}`. Generations available: {len(rows)}. "
            f"Observed caps: {sorted(caps)}.\n\n"
            f"Diagnostic gate: {gate.get('passed') if gate else 'not yet measured'}. "
            "A failed diagnostic gate does not prevent reporting; it does prevent treating this run as evidence of induction.\n\n"
            "Inspect `tables/data_acceptance.csv`, `tables/completion_by_group.csv`, the effect tables under "
            "`04_behavior/pilot/dev/tables`, and raw responses before selecting settings. "
            "Completion is shown by source, family and bucket; capped answers remain censored.\n\n"
            "The default update count, rank and gate thresholds are starting values, not empirically established settings. "
            "Neither 96 nor the old 450–486 optimizer steps is automatically comparable without batch/token exposure.\n\n"
            "Changing trial settings means a new calibration project. Export a separate confirmation configuration "
            "only after recording your decision rationale. Reusable validated data may be copied with exact-identity checks; "
            "training weights and human audit approval are not promoted.\n")
    (out / "README.md").write_text(text, encoding="utf-8")
    return out / "artifacts" / "calibration_summary.json"


def sweep_caps(cfg: Config, caps: Sequence[int]) -> None:
    """Compare fixed caps on identical DEVELOPMENT prompts for all pilot conditions.

    No selective extension or dropping of truncated answers. Cache identity includes
    each cap. Evaluation policy is unchanged across conditions within a cap.
    """
    _require_calibration(cfg)
    if not caps or len(set(caps)) != len(caps) or any(isinstance(x, bool) or x <= 0 for x in caps):
        raise ValueError("Specify distinct positive token caps")
    if cfg.runtime.device != "cpu" and not (cfg.exp_dir / "preflight.json").exists():
        raise RuntimeError("Run --preflight first")
    from .models import LM
    from .dpo import verify_complete
    from .data import load_eval_items
    from .behavior import (_generate_rows, judge_generations, condition_table, run_gate,
                           generation_long, summarize)
    from .common import get_logger
    for cond in cfg.conditions(): verify_complete(cfg.adapter_dir(cond) / "final")
    tick()
    out = cfg.stage_dir("00_calibration") / "artifacts" / "cap_sweeps"
    lm = LM(cfg)
    try:
        with partition(cfg, "dev"):
            items = [it for it in load_eval_items(cfg, limit_per_group=cfg.eval.gen_n_per_group)
                     if it["family"] not in PROBE_ONLY_FAMILIES]
            if not items: raise RuntimeError("No development generation items")
            if any(it.get("partition") != "dev" for it in items): raise RuntimeError("Non-development item in calibration")
            seqs = [lm.encode_prompt(it["context"]) for it in items]
            names = condition_table(cfg)
            logit_file = cfg.stage_dir("04_behavior") / "tables" / "logit_metrics_long.csv"
            if not logit_file.exists(): raise RuntimeError("Run calibration behavior once before sweeping caps")
            import pandas as pd
            logits = pd.read_csv(logit_file).to_dict("records")
            for cap in caps:
                tick()
                root = out / str(cap)
                root.mkdir(parents=True, exist_ok=True)
                logger = get_logger("calibration_cap", root / "logs")
                signature = stable_hash(cfg.fingerprint(), data_identity(cfg), cap,
                                        [(n, file_hash(cfg.adapter_dir(c) / 'final' / 'COMPLETE.json') if c else None)
                                         for n, c in names], n=64)
                done = root / "SUMMARY.json"
                if done.exists():
                    saved = load_json(done)
                    if saved['identity'] != signature: raise RuntimeError('Stale calibration sweep')
                    if any(not (root / p).is_file() or file_hash(root / p) != h for p, h in saved['files'].items()):
                        raise RuntimeError('Corrupt calibration sweep outputs')
                    continue
                rows = _generate_rows(lm, cfg, items, seqs, cap, names, logger,
                                      progress_path=root / "generations_in_progress.jsonl")
                judge_generations(lm, cfg, rows, {it['item_id']: it for it in items})
                for r in rows:
                    words = r['response'].split()
                    grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
                    r['repeated_4gram_fraction'] = 1-len(set(grams))/len(grams) if grams else 0.
                save_jsonl(rows, root / "generations.jsonl")
                save_csv(completion_table(rows), root / "completion_by_group.csv")
                means, effects = summarize(generation_long(rows), cfg)
                save_csv(means, root / "generation_means.csv"); save_csv(effects, root / "generation_effects.csv")
                diagnostic = run_gate(cfg, logits, rows, output_dir=root / "diagnostic_gate")
                files = {str(p.relative_to(root)): file_hash(p) for p in root.rglob('*')
                         if p.is_file() and p.suffix in ('.csv', '.json', '.jsonl') and p != done}
                save_json({"study_role": "calibration", "identity": signature, "max_new_tokens": cap,
                           "n_generations": len(rows), "diagnostic_gate_passed": diagnostic['passed'],
                           "files": files, "fixed_cohort": [it['item_id'] for it in items]}, done)
                write_summary(cfg)
    finally:
        del lm
        import gc, torch
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        write_summary(cfg)


def export_confirmation(calibration_project: str, output: Path, *, model_key: str,
                        max_new_tokens: int, max_updates: int, seeds: Sequence[int],
                        contamination_rates: Sequence[float], note: str,
                        overrides: Sequence[str] = (),
                        max_updates_by_model: Optional[Dict[str, int]] = None) -> Config:
    """Generate a reviewable configuration; never train or manufacture a successful gate."""
    pilot = load_project(calibration_project, model_key, "pilot")
    _require_calibration(pilot)
    summary_path = write_summary(pilot)
    summary = load_json(summary_path)
    if not note.strip(): raise ValueError("Record the rationale for the selected settings")
    if max_updates < 1 or max_new_tokens < 1: raise ValueError("Positive training and generation budgets required")
    if not summary['data_identity'] or not summary['n_generations_available']:
        raise RuntimeError("Need actual calibration data and generated responses before exporting confirmation settings")
    if not seeds or set(seeds) & set(pilot.plan.seeds + pilot.plan.replication_seeds):
        raise ValueError("Confirmatory training seeds must be explicit and disjoint from calibration seeds")
    cfg = copy.deepcopy(pilot)
    cfg.runtime.study_role = "confirmatory"
    cfg.runtime.require_semantic_audit = True
    cfg.runtime.require_calibration_record = False   # validate after adding the record below
    cfg.runtime.project = None
    cfg.runtime.phase = "full"
    cfg.runtime.exp_name = "gamingmech_v3_1_confirmation"
    cfg.runtime.run_id = Config().runtime.run_id
    cfg.train.max_updates = max_updates
    cfg.train.max_updates_by_model = dict(max_updates_by_model or {})   # an explicit choice, never inherited
    cfg.eval.max_new_tokens = max_new_tokens
    cfg.plan.seeds = tuple(seeds); cfg.plan.replication_seeds = tuple(seeds)
    cfg.plan.contamination_seeds = (seeds[0],)
    cfg.plan.contamination_rates = tuple(contamination_rates)
    cfg = apply_overrides(cfg, list(overrides))
    if cfg.runtime.study_role != "confirmatory" or not cfg.runtime.require_semantic_audit:
        raise ValueError("Confirmation must retain its study role and human-audit requirement")
    if set(cfg.plan.seeds + cfg.plan.replication_seeds) & set(pilot.plan.seeds + pilot.plan.replication_seeds):
        raise ValueError("Do not reuse calibration training seeds in confirmation")
    if cfg.eval.max_new_tokens not in summary['observed_token_caps']:
        raise ValueError("Measure the selected cap with --calibration-caps before freezing it")
    # Require a real, completed behavior stage. Its diagnostic gate may legitimately have failed.
    receipt_path = pilot.exp_dir / model_key / 'receipts' / 'pilot' / 'behavior.json'
    if not receipt_path.exists() or not load_json(receipt_path).get('passed'):
        raise RuntimeError('Finish the calibration behavior stage; a failed diagnostic gate is permitted, a crashed stage is not')
    dev = pilot.exp_dir / model_key / '04_behavior' / 'pilot' / 'dev'
    if not (dev / 'artifacts' / 'gate_pilot.json').exists():
        raise RuntimeError('Calibration behavior must finish and record its diagnostic gate')
    record = {"calibration_project": str(pilot.exp_dir), "calibration_source_hash": source_hash(),
              "calibration_config_fingerprint": pilot.fingerprint(), "data_identity": summary['data_identity'],
              "summary_sha256": file_hash(summary_path), "rationale": note.strip(),
              "selected_utc": datetime.now(timezone.utc).isoformat(),
              "selected_max_new_tokens": cfg.eval.max_new_tokens, "selected_training": dataclasses.asdict(cfg.train),
              "selected_plan": dataclasses.asdict(cfg.plan),
              "selected_gate": {k: v for k, v in dataclasses.asdict(cfg.eval).items() if k.startswith('gate_')},
              "contamination_rates": list(cfg.plan.contamination_rates),
              "calibration_training_changed": dataclasses.asdict(cfg.train) != dataclasses.asdict(pilot.train),
              "claim_limit": "Low-contamination outcomes are not yet evidence of earlier detection."}
    cfg.runtime.calibration_record = record
    cfg.runtime.require_calibration_record = True
    validate_config(cfg)
    if output.exists(): raise FileExistsError(f"Refusing to overwrite selected protocol: {output}")
    save_json(cfg.to_dict(), output)
    return cfg


def _data_contract(cfg: Config) -> dict:
    return {'data': dataclasses.asdict(cfg.data), 'writer': dataclasses.asdict(cfg.writer),
            'fit_fraction': cfg.runtime.fit_fraction, 'dev_fraction': cfg.runtime.dev_fraction,
            'audit_min_pairs': cfg.runtime.audit_min_pairs}


def initialize_from_calibration(cfg: Config, data_from: str | None, billing_from: str | None) -> Path:
    """Create a fresh project. Only exact validated DATA and explicit billing history may transfer."""
    parents = {}
    for path in {p for p in (data_from, billing_from) if p}:
        parent = load_project(path, cfg.model_key, 'pilot')
        _require_calibration(parent)
        parents[path] = parent
    if cfg.runtime.study_role not in ('calibration', 'confirmatory'):
        raise ValueError('Unknown destination study role')
    if cfg.exp_dir.resolve().exists(): raise FileExistsError(cfg.exp_dir)
    validate_config(cfg)
    if data_from:
        src = parents[data_from]
        if _data_contract(src) != _data_contract(cfg):
            raise ValueError('Data/writer/partition/audit-sampling settings changed; regenerate rather than silently reuse data')
        cert = load_json(src.data_dir / 'artifacts' / 'semantic_validation.json')
        identity = data_identity(src)
        if cert['identity_after'] != identity:
            raise RuntimeError('Calibration data changed after semantic validation')
        if cfg.runtime.study_role == 'confirmatory' and cfg.runtime.calibration_record.get('data_identity') != identity:
            raise RuntimeError('Data are not the dataset recorded in the selected protocol')
    if billing_from:
        clock = load_json(parents[billing_from].exp_dir / 'billing.json')
        # Dates/rates remain literal: this is a copy of the same instance ledger, not a new clock.
        if not clock.get('started_utc'): raise RuntimeError('Missing source billing clock')
    root = initialize(cfg)
    try:
        if data_from:
            src = parents[data_from]
            shutil.copytree(src.data_dir, cfg.data_dir)
            # A disposable unaudited pilot never gives the new project human-audit approval.
            for name in ('human_audit.json', 'human_audit_rows.jsonl'):
                (cfg.data_dir / 'artifacts' / name).unlink(missing_ok=True)
            if data_identity(cfg) != identity: raise RuntimeError('Data-copy integrity failure')
            if (src.exp_dir / 'revisions.json').exists():
                shutil.copy2(src.exp_dir / 'revisions.json', root / 'revisions.json')
            save_json({'source_project': str(src.exp_dir), 'identity': identity,
                       'source_hash': source_hash(), 'weights_imported': False,
                       'human_audit_approval_imported': False}, root / 'data_origin.json')
        if billing_from:
            save_json(clock, root / 'billing.json')
            save_json({'source_project': str(parents[billing_from].exp_dir),
                       'copied_utc': datetime.now(timezone.utc).isoformat(),
                       'same_instance_required': True}, root / 'billing_origin.json')
    except BaseException as ex:
        save_json({'error': repr(ex), 'incomplete': True}, root / 'HANDOFF_FAILED.json')
        raise
    activate(cfg)
    return root
