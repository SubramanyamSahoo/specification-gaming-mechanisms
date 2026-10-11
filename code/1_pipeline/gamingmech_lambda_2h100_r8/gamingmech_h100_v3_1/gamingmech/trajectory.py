"""Training time trajectory: does the internal monitor move before the behaviour does?

The monitor (direction and layer) is fixed from the reference seed. It is then read out on the
saved checkpoints of other training runs, next to the behavioural metrics at the same steps.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .behavior import GAMING_METRICS, LogitProbe, metric_of
from .common import get_logger, load_json, save_csv
from .config import Config
from .data import load_eval_items
from .detect import METRIC_SIGN, monitor_setup, read_index
from .mech import activation_items, capture, load_acts
from .models import LM


def run_trajectory(lm: LM, cfg: Config) -> None:
    """Stage 10: monitor projection and behaviour at every saved checkpoint of held out runs."""
    stage, m = cfg.stage_dir("10_trajectory"), cfg.mech
    logger = get_logger("trajectory", stage / "logs")
    index, items = read_index(cfg), activation_items(cfg)
    seqs = [lm.encode_prompt(it["context"]) for it in items]
    content = [lm.content_mask(it["context"]) for it in items]
    base = load_acts(cfg, "base")
    acts: Dict[str, torch.Tensor] = {}

    def diff(name: str) -> torch.Tensor:
        if name not in acts:
            acts[name] = load_acts(cfg, name) - base
        return acts[name]

    ev_items = load_eval_items(cfg, limit_per_group=m.ckpt_n_per_group, partition_name="test")
    probe = LogitProbe(lm, ev_items)
    ev_metric = [metric_of(it) for it in ev_items]
    with lm.using(None):
        base_vals = probe.values(lm, cfg.eval.batch_size)
    conds = {c.name: c for c in cfg.conditions()}
    seed = cfg.reference_seed
    rows = []
    for obj in ("syco", "length"):
        setup = monitor_setup(cfg, obj, index, diff)
        if setup is None:
            continue
        test = setup["test"].tolist()
        t_seqs, t_content = [seqs[i] for i in test], [content[i] for i in test]
        base_layer = base.index_select(0, setup["test"])[:, setup["layer"]]
        tgt = [i for i, x in enumerate(ev_metric) if x in GAMING_METRICS[obj]]
        sign = np.array([METRIC_SIGN[ev_metric[i]] for i in tgt])
        held_out = [c for c in conds.values() if c.flip_rate in (0.0, 1.0) and c.objective in (obj, "control")
                    and c.seed != seed]
        runs = held_out
        if not runs:
            logger.info("No held-out training seeds: trajectory analysis is not applicable")
            continue
        # Align both measurements to identical held-out prompt identities.
        by_id = {it["item_id"]: it for it in ev_items}
        test = [i for i in test if index[i]["item_id"] in by_id]
        if not test: raise RuntimeError("No matched trajectory items")
        t_seqs, t_content = [seqs[i] for i in test], [content[i] for i in test]
        base_layer = base[torch.tensor(test), setup["layer"]]
        matched = [by_id[index[i]["item_id"]] for i in test]
        probe = LogitProbe(lm, matched)
        ev_metric = [metric_of(it) for it in matched]
        tgt = [i for i, x in enumerate(ev_metric) if x in GAMING_METRICS[obj]]
        sign = np.array([METRIC_SIGN[ev_metric[i]] for i in tgt])
        with lm.using(None): base_vals = probe.values(lm,cfg.eval.batch_size)
        for cond in runs:
            summary = load_json(cfg.adapter_dir(cond) / "summary.json")
            for dirname, fraction in sorted(summary["checkpoints"].items()):
                name = f"{cond.name}_{dirname}"
                lm.load_adapter(name, cfg.adapter_dir(cond) / dirname)
                with lm.using(name):
                    h = capture(lm, t_seqs, t_content, (m.primary_pooling,), m.capture_batch_size)
                    vals = probe.values(lm, cfg.eval.batch_size)
                lm.drop_adapter(name)
                proj = (h[m.primary_pooling][:, setup["layer"]].float() - base_layer) @ setup["unit"]
                beh = sign * (vals[tgt] - base_vals[tgt])
                rows.append({"objective": obj, "condition": cond.name, "role": cond.objective, "seed": cond.seed,
                             "held_out_run": bool(held_out), "fraction": fraction,
                             "step": int(dirname.split("_")[1]), "layer": setup["layer"],
                             "monitor": float(proj.mean()), "behaviour": float(np.nanmean(beh))})
            logger.info(f"trajectory read for {cond.name}")
    # express each gaming run relative to its own final value, with the control runs as the noise floor
    final = {}
    for r in rows:
        key = (r["objective"], r["condition"])
        if key not in final or r["fraction"] > final[key]["fraction"]: final[key] = r
    for r in rows:
        f = final.get((r["objective"], r["condition"]))
        for k in ("monitor", "behaviour"):
            r[f"{k}_share_of_final"] = (r[k] / f[k] if f and r["role"] != "control" and f[k] != 0 else float("nan"))
    save_csv(rows, stage / "tables" / "checkpoint_trajectory.csv")
