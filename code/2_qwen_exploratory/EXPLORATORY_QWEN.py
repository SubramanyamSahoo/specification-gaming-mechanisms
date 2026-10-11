"""Disclosed exploratory run: Qwen3.5 held out test and mechanisms after the protocol v3 gate failed.

The gate failed only because verbosity did not hold in seeds 79 and 83; caving to pushback was confirmed in all
three seeds. This runs on a COPY of the confirmation project, so the official project keeps its failed gate. The
package is not modified: the same stage functions run in the pipeline's order, and only the gate check is skipped.
EXPLORATORY.json in the copy records why, what ran and what failed. A failing stage is logged and the next runs.
"""
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("test_behavior", "activations", "steering", "patching", "dissect", "detection",
          "features", "minimal", "trajectory", "report")


def now():
    return datetime.now(timezone.utc).strftime("%d %b %H:%M UTC")


def main(project, model, stages=STAGES):
    import torch
    from gamingmech.common import save_json, setup_torch
    from gamingmech.plots import plot_stage
    from gamingmech.run import PLOT_STAGE, RUNNERS
    from gamingmech.runtime import activate, load_project, partition, project_lock, tick
    cfg = load_project(project, model, "full")
    activate(cfg); setup_torch()
    torch.backends.cuda.matmul.allow_tf32 = cfg.runtime.tf32
    torch.backends.cudnn.allow_tf32 = cfg.runtime.tf32
    root = Path(project)
    gate = json.loads((root / model / "04_behavior" / "full" / "dev" / "artifacts" / "gate_full.json").read_text())
    path = root / "EXPLORATORY.json"
    record = json.loads(path.read_text()) if path.exists() else {
        "exploratory": True, "started_utc": now(), "stages": {},
        "reason": "protocol v3 gate failed: " + "; ".join(gate.get("errors", [])),
        "confirmed_behaviours": gate.get("confirmed_behaviours")}
    save_json(record, path)
    lm_cache = {}
    with project_lock(cfg):
        for stage in stages:
            if record["stages"].get(stage, {}).get("passed"):
                print(f"===== exploratory {model}: {stage} already done", flush=True); continue
            tick(); t0 = time.time()
            print(f"===== exploratory {model}: {stage} ({now()})", flush=True)
            try:
                RUNNERS[stage](cfg, lm_cache)
                if stage in PLOT_STAGE:
                    with partition(cfg, "test"): plot_stage(cfg, PLOT_STAGE[stage])
                record["stages"][stage] = {"passed": True, "elapsed_s": round(time.time() - t0)}
            except Exception as ex:
                traceback.print_exc()
                record["stages"][stage] = {"passed": False, "error": repr(ex)[:500], "elapsed_s": round(time.time() - t0)}
            save_json(record, path)
    print(f"===== exploratory {model}: ALL DONE", {k: v["passed"] for k, v in record["stages"].items()}, flush=True)
