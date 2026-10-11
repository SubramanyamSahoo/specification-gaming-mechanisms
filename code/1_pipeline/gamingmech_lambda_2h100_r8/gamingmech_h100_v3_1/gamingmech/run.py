"""Immutable-project command line runner. See README.md for initialization and staged execution."""
from __future__ import annotations

import argparse
import copy
import time

import torch
from typing import Callable, Dict, List

from .common import get_logger, hf_login, load_json, save_csv, save_json, setup_torch
from .config import MODEL_REGISTRY, Config, ConditionSpec, apply_overrides
from .models import LM

STAGES = ("data", "validate", "train", "behavior", "test_behavior", "activations", "patching", "dissect", "minimal", "steering", "detection",
          "trajectory", "features", "profile", "report")


def stage_data(cfg: Config, lm_cache: Dict) -> None:
    """Build the roster from the response bank. Only tokenizers are loaded, one per model family."""
    from .data import TokenCounter, build_eval_sets, build_roster
    art = cfg.data_dir / "artifacts"
    token = hf_login(cfg.runtime.hf_token_env)
    if not (art / "roster.jsonl").exists():
        counters = {}
        for key, spec in sorted(MODEL_REGISTRY.items()):           # one tokenizer per model family
            hf_id = cfg.writer.hf_id if key == cfg.writer.model_key and cfg.writer.hf_id else spec.hf_id
            counters[key] = TokenCounter.load(cfg, hf_id, token, spec.chat_kwargs)
        build_roster(cfg, counters, token)
    if not (art / "eval_items.jsonl").exists():
        build_eval_sets(cfg, token)


def _lm(cfg: Config, lm_cache: Dict) -> LM:
    if "lm" not in lm_cache:
        lm_cache["lm"] = LM(cfg)
    return lm_cache["lm"]


def stage_train(cfg: Config, lm_cache: Dict) -> None:
    from .dpo import train_plan
    from .audits import require_data
    require_data(cfg)
    train_plan(_lm(cfg, lm_cache), cfg, cfg.conditions())


def stage_validate(cfg: Config, lm_cache: Dict) -> None:
    from .audits import validate_candidates
    from .runtime import data_identity
    path = cfg.data_dir / "artifacts" / "semantic_validation.json"
    if path.exists() and load_json(path)["identity_after"] == data_identity(cfg): return
    validate_candidates(None, cfg)                      # rule checks on the final text; no model is loaded


def stage_behavior(cfg: Config, lm_cache: Dict) -> None:
    from .behavior import run_gate, run_generation, run_logit_metrics
    from .runtime import partition
    lm = _lm(cfg, lm_cache)
    with partition(cfg, "dev"):
        result = run_gate(cfg, run_logit_metrics(lm, cfg), run_generation(lm, cfg))
    if cfg.runtime.study_role == "calibration":
        from .calibration import write_summary
        write_summary(cfg)
    if not result["passed"] and cfg.runtime.study_role != "calibration":
        raise RuntimeError("BEHAVIOR GATE FAILED. Downstream stages stopped. See dev/artifacts/gate.json")


def stage_test_behavior(cfg: Config, lm_cache: Dict) -> None:
    from .behavior import run_generation, run_logit_metrics, run_thinking, summarize, generation_long
    from .runtime import partition
    with partition(cfg, "test"):
        lm = _lm(cfg, lm_cache)
        rows = run_logit_metrics(lm, cfg); gen = run_generation(lm, cfg)
        for stem, data in (("logit", rows), ("generation", generation_long(gen))):
            means, effects = summarize(data, cfg)
            save_csv(means, cfg.stage_dir("04_behavior") / "tables" / f"{stem}_means.csv")
            save_csv(effects, cfg.stage_dir("04_behavior") / "tables" / f"{stem}_effects.csv")
        if cfg.eval.run_thinking: run_thinking(lm, cfg)


def stage_activations(cfg: Config, lm_cache: Dict) -> None:
    from .mech import run_capture, run_directions
    run_capture(_lm(cfg, lm_cache), cfg)
    run_directions(cfg)


def stage_patching(cfg: Config, lm_cache: Dict) -> None:
    from .patching import run_patching
    run_patching(_lm(cfg, lm_cache), cfg)


def stage_dissect(cfg: Config, lm_cache: Dict) -> None:
    from .dissect import run_dissection
    run_dissection(_lm(cfg, lm_cache), cfg)


def stage_minimal(cfg: Config, lm_cache: Dict) -> None:
    """Train adapters restricted to the most important module groups and test whether they suffice."""
    from .behavior import GAMING_METRICS, LogitProbe, metric_means
    from .data import load_eval_items
    from .dpo import train_condition
    lm = _lm(cfg, lm_cache)
    stage = cfg.stage_dir("07_dissect")
    specs = load_json(stage / "artifacts" / "minimal_adapter_specs.json")
    items = load_eval_items(cfg, limit_per_group=cfg.mech.n_per_group)
    probe = LogitProbe(lm, items)
    ref_cache = cfg.stage_dir("03_train") / "artifacts" / "reference_logprobs.pt"
    with lm.using(None):
        base = metric_means(items, probe.values(lm, cfg.eval.batch_size))
    rows = []
    for obj, spec in specs.items():
        sub = copy.deepcopy(cfg)
        sub.train.lora.layers_to_transform = None
        sub.train.lora.target_types = None
        sub.train.lora.exact_targets = spec["exact_targets"]
        cond = ConditionSpec(obj, 1.0, cfg.reference_seed, tag="minimal")
        train_condition(lm, sub, cond, ref_cache)
        res = {"base": base}
        for role, name in (("minimal", cond.name), ("full", spec["source_condition"])):
            lm.load_adapter(name, cfg.exp_dir / cfg.model_key / "03_train" / "artifacts" / name / "final")
            with lm.using(name):
                res[role] = metric_means(items, probe.values(lm, cfg.eval.batch_size))
            lm.drop_adapter(name)
        for metric in sorted(res["full"]):
            gap = res["full"][metric] - base[metric]
            rows.append({"objective": obj, "metric": metric, "is_target": metric in GAMING_METRICS[obj],
                         "base": base[metric], "full": res["full"][metric], "minimal": res["minimal"][metric],
                         "recovered_fraction": (res["minimal"][metric] - base[metric]) / gap if gap else float("nan"),
                         "layers": " ".join(map(str, spec["layers_to_transform"])),
                         "module_types": " ".join(spec["target_types"])})
    save_csv(rows, stage / "tables" / "minimal_adapter_summary.csv")


def stage_steering(cfg: Config, lm_cache: Dict) -> None:
    from .steering import run_steering
    run_steering(_lm(cfg, lm_cache), cfg)


def stage_detection(cfg: Config, lm_cache: Dict) -> None:
    from .detect import run_detection
    run_detection(cfg)


def stage_trajectory(cfg: Config, lm_cache: Dict) -> None:
    from .trajectory import run_trajectory
    run_trajectory(_lm(cfg, lm_cache), cfg)


def stage_features(cfg: Config, lm_cache: Dict) -> None:
    from .features import run_features
    run_features(_lm(cfg, lm_cache), cfg)


def stage_profile(cfg: Config, lm_cache: Dict) -> None:
    from .profile import run_profile
    from .audits import require_data
    require_data(cfg)
    run_profile(_lm(cfg, lm_cache), cfg)


def stage_calibration_summary(cfg: Config, lm_cache: Dict) -> None:
    from .calibration import write_summary
    print(write_summary(cfg))


def stage_report(cfg: Config, lm_cache: Dict) -> None:
    from .report import build_report
    print(build_report(cfg))


RUNNERS: Dict[str, Callable[[Config, Dict], None]] = {
    "data": stage_data, "validate": stage_validate, "test_behavior": stage_test_behavior, "train": stage_train, "behavior": stage_behavior, "activations": stage_activations,
    "patching": stage_patching, "dissect": stage_dissect, "minimal": stage_minimal,
    "steering": stage_steering, "detection": stage_detection, "trajectory": stage_trajectory,
    "features": stage_features, "profile": stage_profile, "report": stage_report,
    "calibration_summary": stage_calibration_summary}
PLOT_STAGE = {"train": "03_train", "behavior": "04_behavior", "activations": "05_activations",
              "patching": "06_patching", "dissect": "07_dissect", "minimal": "07_dissect",
              "steering": "08_steering", "detection": "09_detection", "trajectory": "10_trajectory",
              "features": "11_features"}


def run(cfg: Config, stages: List[str]) -> None:
    from .plots import plot_stage
    from .runtime import activate, tick, project_lock, require_gate, data_identity, source_hash, partition
    from .common import stable_hash
    from datetime import datetime, timezone
    import uuid
    if cfg.runtime.study_role == "calibration":
        allowed = {"data", "validate", "profile", "train", "behavior", "report", "calibration_summary"}
        if cfg.runtime.phase != "pilot" or set(stages) - allowed:
            raise RuntimeError("Calibration is pilot/development-only; test and mechanisms require a separate confirmatory project")
    activate(cfg); setup_torch()
    torch.backends.cuda.matmul.allow_tf32 = cfg.runtime.tf32
    torch.backends.cudnn.allow_tf32 = cfg.runtime.tf32
    if cfg.runtime.device != "cpu" and not (cfg.exp_dir / "preflight.json").exists():
        raise RuntimeError("Run --preflight first; it is automatic in scripts/pilot.sh")
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    logs = cfg.exp_dir / cfg.model_key / "executions" / attempt
    logger = get_logger("run", logs)
    lm_cache = {}
    with project_lock(cfg):
        for stage in stages:
            tick()
            if stage not in ("data", "validate", "train", "behavior", "report", "profile", "calibration_summary"):
                require_gate(cfg)
            deps = data_identity(cfg) if stage not in ("data", "validate", "calibration_summary", "report") else None
            signature = stable_hash(cfg.fingerprint(), cfg.model_key, cfg.runtime.phase, source_hash(), deps)
            receipt = cfg.exp_dir / cfg.model_key / "receipts" / cfg.runtime.phase / (stage + ".json")
            if receipt.exists():
                old = load_json(receipt)
                if old.get("passed") and old["signature"] == signature and stage not in ("report", "calibration_summary"):
                    if stage == "behavior" and cfg.runtime.study_role != "calibration": require_gate(cfg)
                    if stage == "train":
                        from .dpo import verify_complete
                        for cond in cfg.conditions(): verify_complete(cfg.adapter_dir(cond) / "final")
                    for name in old.get("outputs", []):
                        if not (cfg.exp_dir / name).exists():
                            raise RuntimeError(f"Missing recorded output: {name}; do not reuse this stage")
                    logger.info(f"Verified completed stage {stage}; reusing registered artifacts")
                    continue
                if old.get("signature") != signature:
                    raise RuntimeError(f"Dependencies changed for {stage}; start a new project")
            t0 = time.time()
            logger.info(f"=== {cfg.model_key} / {cfg.runtime.phase} / {stage} ===")
            try:
                RUNNERS[stage](cfg, lm_cache)
                if stage in PLOT_STAGE:
                    with partition(cfg, "dev" if stage == "behavior" else "test"):
                        plot_stage(cfg, PLOT_STAGE[stage])
                outdir = cfg.stage_dir(PLOT_STAGE[stage]) if stage in PLOT_STAGE else None
                if stage == "behavior": outdir = cfg.exp_dir / cfg.model_key / "04_behavior" / cfg.runtime.phase / "dev"
                outputs = [str(p.relative_to(cfg.exp_dir)) for p in outdir.rglob("*") if p.is_file()] if outdir else []
                save_json({"passed": True, "signature": signature, "elapsed_s": time.time()-t0,
                           "execution_id": attempt, "outputs": outputs}, receipt)
            except BaseException as ex:
                save_json({"passed": False, "signature": signature, "elapsed_s": time.time()-t0,
                           "error": repr(ex), "execution_id": attempt}, receipt)
                raise
            finally:
                save_json({"stage": stage, "elapsed_s": time.time()-t0}, logs / f"{stage}.json")
                if cfg.runtime.study_role == "calibration":
                    # Even failed validation/training leaves a readable acceptance/coverage report.
                    try:
                        from .calibration import write_summary
                        write_summary(cfg)
                    except Exception as summary_error:
                        logger.error("Calibration summary failed: %r", summary_error)


def main() -> None:
    from .runtime import initialize, load_project, from_dict, billing_start, preflight, budget
    from .audits import register_audit
    from pathlib import Path
    import json
    ap = argparse.ArgumentParser(description="gamingmech v3.1: calibrate first, then lock a separate confirmatory project")
    ap.add_argument("--model", default="qwen", choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--project")
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--data-from", help="On init, copy compatible validated data from a calibration project; never weights")
    ap.add_argument("--billing-from", help="On init, inherit the same actual instance clock INCLUDING earlier spending")
    ap.add_argument("--calibration-summary", action="store_true", help="CPU-only report, including failed/partial calibration")
    ap.add_argument("--calibration-caps", help="Calibration only: comma-separated token caps to evaluate on the same dev prompts")
    ap.add_argument("--config", help="JSON configuration, generated by scripts/export_config.py")
    ap.add_argument("--clock", action="store_true")
    ap.add_argument("--pause-clock", action="store_true")
    ap.add_argument("--resume-clock", action="store_true")
    ap.add_argument("--ended-utc")
    ap.add_argument("--hourly-usd", type=float)
    ap.add_argument("--started-utc")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--audit-csv")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--phase", choices=("pilot", "full"), default="full")
    ap.add_argument("--stages", default="train,behavior,test_behavior,activations")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    if args.init:
        if not args.config: ap.error("--init requires an explicit calibration or selected confirmation config")
        cfg = from_dict(json.loads(Path(args.config).read_text())) if args.config else Config()
        cfg = apply_overrides(cfg, args.set)
        if args.project: cfg.runtime.project = str(Path(args.project).resolve())
        if args.data_from or args.billing_from:
            from .calibration import initialize_from_calibration
            print(initialize_from_calibration(cfg, args.data_from, args.billing_from))
        else:
            print(initialize(cfg))
        return
    if not args.project: ap.error("--project is required after initialization")
    if args.set or args.config: ap.error("Scientific settings are immutable after init; create a new project")
    if args.data_from or args.billing_from: ap.error("--data-from/--billing-from are init-only")
    cfg = load_project(args.project, args.model, args.phase)
    if args.calibration_summary:
        from .calibration import write_summary
        print(write_summary(cfg)); return
    if args.calibration_caps:
        from .calibration import sweep_caps
        from .runtime import project_lock
        caps = tuple(int(x) for x in args.calibration_caps.split(","))
        with project_lock(cfg): sweep_caps(cfg, caps)
        return
    if args.clock:
        if args.hourly_usd is None or not args.started_utc: ap.error("Clock requires --hourly-usd and --started-utc")
        billing_start(cfg, args.hourly_usd, args.started_utc); return
    if args.pause_clock or args.resume_clock:
        from .runtime import billing_pause, billing_resume, project_lock
        with project_lock(cfg):
            if args.pause_clock:
                if not args.ended_utc: ap.error("Pause requires --ended-utc from provider billing")
                billing_pause(cfg, args.ended_utc)
            else:
                if args.hourly_usd is None or not args.started_utc: ap.error("Resume requires rate and actual billing start")
                billing_resume(cfg, args.hourly_usd, args.started_utc)
        return
    if args.audit_csv:
        print(register_audit(cfg, Path(args.audit_csv))); return
    if args.preflight:
        preflight(cfg); print("Preflight passed"); return
    if args.status:
        print(json.dumps(budget(cfg), indent=2)); return
    stages = list(STAGES) if args.stages == "all" else args.stages.split(",")
    unknown = set(stages) - set(RUNNERS)
    if unknown: ap.error(f"Unknown stages {unknown}. Available: {list(RUNNERS)}")
    if cfg.runtime.phase == "full" and "train" in stages:
        from .runtime import require_gate
        pilot = copy.deepcopy(cfg); pilot.runtime.phase = "pilot"
        require_gate(pilot)
    run(cfg, stages)


if __name__ == "__main__":
    main()
