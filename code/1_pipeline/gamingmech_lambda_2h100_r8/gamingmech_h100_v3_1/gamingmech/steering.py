"""Activation steering along difference of means directions, with matched controls.

A dose of 1 adds the measured mean gap between the gaming adapter and its reference at that
layer. The vector is added at the positions where the answer is produced: the last prompt
token for forced choice metrics, and the last prompt token through the end of the teacher
forced text for the other metrics. Random directions and the CAA baseline use the same norm.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .behavior import (ACCURACY_OF, GAMING_METRICS, LogitProbe, generation_long, judge_generations, metric_of,
                       summarize)
from .common import get_logger, paired_effect, save_csv, save_jsonl, save_json
from .config import PROBE_ONLY_FAMILIES, Config
from .data import load_eval_items
from .gen import generate
from .mech import hooks, out_tensor, with_tensor
from .models import LM


class Steerer:
    """Adds a vector to the residual stream after one block for the positions that matter."""

    def __init__(self, lm: LM, probe: LogitProbe):
        self.lm, self.probe = lm, probe
        self.vec: Optional[torch.Tensor] = None
        self.tail: Optional[torch.Tensor] = None        # per row count of trailing positions, right padded
        self.lens: Optional[torch.Tensor] = None
        self.tf_tail = torch.tensor(probe.tf_tail, dtype=torch.long)
        self.tf_len = torch.tensor([len(s) for s in probe.tf_seqs], dtype=torch.long)

    def _hook(self, _m, _a, out):
        h = out_tensor(out)
        if self.tail is None:                            # left padded batch: steer the final position
            h = h.clone()
            h[:, -1] = (h[:, -1].float() + self.vec).to(h.dtype)
        else:                                            # right padded teacher forced batch
            pos = torch.arange(h.shape[1], device=h.device)[None, :]
            sel = (pos >= (self.lens - self.tail)[:, None]) & (pos < self.lens[:, None])
            h = (h.float() + sel[:, :, None] * self.vec).to(h.dtype)
        return with_tensor(out, h)

    def _pre_forced(self, _idx: List[int]) -> None:
        self.tail = None

    def _pre_tf(self, idx: List[int]) -> None:
        self.tail = self.tf_tail[idx].to(self.lm.device)
        self.lens = self.tf_len[idx].to(self.lm.device)

    @torch.no_grad()
    def values(self, layer: int, vec: Optional[torch.Tensor], batch_size: int) -> np.ndarray:
        """Probe values with `vec` added after block `layer` (hidden state index layer + 1)."""
        from .runtime import cached_json
        import hashlib
        digest = None if vec is None else hashlib.sha256(vec.detach().cpu().float().numpy().tobytes()).hexdigest()
        key = ("steering_logits", self.lm.active, layer, digest, self.probe.forced_seqs, self.probe.tf_seqs)
        def compute():
            if vec is None: return self.probe.values(self.lm, batch_size).tolist()
            self.vec = vec.to(self.lm.device)
            with hooks([(self.lm.layers[layer], self._hook, False)]):
                return self.probe.values(self.lm, batch_size, self._pre_forced, self._pre_tf).tolist()
        return np.array(cached_json(self.lm.cfg, "08_steering", key, compute), dtype=np.float64)

    @torch.no_grad()
    def generate(self, layer: int, vec: Optional[torch.Tensor], seqs: List[List[int]], cfg: Config) -> List[Dict]:
        """Free generation with the vector added at the last prompt token and at every new token."""
        e = cfg.eval
        from .runtime import cached_json
        import hashlib
        digest = None if vec is None else hashlib.sha256(vec.detach().cpu().float().numpy().tobytes()).hexdigest()
        key = ("steered_generation", self.lm.active, layer, digest, seqs)
        def compute():
            if vec is None:
                return generate(self.lm, seqs, e.max_new_tokens, e.temperature, e.top_p, e.gen_batch_size)
            self.vec, self.tail = vec.to(self.lm.device), None
            with hooks([(self.lm.layers[layer], self._hook, False)]):
                return generate(self.lm, seqs, e.max_new_tokens, e.temperature, e.top_p, e.gen_batch_size)
        return cached_json(cfg, "08_steering", key, compute)


def run_steering(lm: LM, cfg: Config) -> None:
    """Stage 08: dose response, cross objective effects, CAA and random controls, steered generation."""
    stage, m, e = cfg.stage_dir("08_steering"), cfg.mech, cfg.eval
    logger = get_logger("steering", stage / "logs")
    art = cfg.stage_dir("05_activations") / "artifacts"
    directions = torch.load(art / "directions.pt")
    caa = torch.load(art / "caa_directions.pt") if (art / "caa_directions.pt").exists() else {}
    conds = {c.name: c for c in cfg.conditions()}
    seed = cfg.reference_seed
    items = load_eval_items(cfg, limit_per_group=m.steer_n_per_group, partition_name="test")
    dev_items = load_eval_items(cfg, limit_per_group=m.steer_n_per_group, partition_name="dev")
    assert not ({it["group_id"] for it in dev_items} & {it["group_id"] for it in items})
    dev_probe = LogitProbe(lm, dev_items)
    dev_steer = Steerer(lm, dev_probe)
    dev_metrics = [metric_of(it) for it in dev_items]
    selections = []
    probe = LogitProbe(lm, items)
    steer = Steerer(lm, probe)
    metrics = [metric_of(it) for it in items]
    gen_items = [it for it in load_eval_items(cfg, limit_per_group=m.steer_gen_n_per_group)
                 if it["family"] not in PROBE_ONLY_FAMILIES]
    gen_seqs = [lm.encode_prompt(it["context"]) for it in gen_items]
    gen_lookup = {it["item_id"]: it for it in gen_items}
    layers = sorted({min(lm.n_layers - 1, int(f * lm.n_layers)) for f in m.steer_layer_fractions})
    g = torch.Generator().manual_seed(seed)
    ctrl = f"control_f000_s{seed}"
    rows, gen_rows = [], []

    def record(run_on: str, direction: str, layer: int, dose: float, vals: np.ndarray, ref: np.ndarray):
        for metric in sorted(set(metrics)):
            idx = [i for i, x in enumerate(metrics) if x == metric]
            eff = paired_effect(vals[idx], ref[idx], e.n_boot, e.ci, seed)
            row = {"run_on": run_on, "direction": direction, "layer": layer, "dose": dose, "metric": metric,
                   "mean": float(np.nanmean(vals[idx])), **eff}
            if metric in ACCURACY_OF:
                row["accuracy"] = float(np.mean(vals[idx] > 0))
            rows.append(row)

    for obj in ("syco", "length"):
        game = f"{obj}_f100_s{seed}"
        key = (game, "control" if ctrl in conds else "base", "target")
        if game not in conds or key not in directions:
            continue
        unit, gap = directions[key]["unit"], directions[key]["gap"]
        named = {obj: unit}
        if obj in caa:
            named[f"caa_{obj}"] = caa[obj]["unit"]
        targets = [("base", 1.0)]
        if ctrl in conds:
            lm.load_condition(conds[ctrl])
            targets.append((ctrl, 1.0))
        lm.load_condition(conds[game])
        targets.append((game, -1.0))                      # subtracting the direction from the gamed model
        for run_on, sign in targets:
            with lm.using(run_on):
                ref = steer.values(0, None, e.batch_size)
                record(run_on, obj, -1, 0.0, ref, ref)
                dev_ref = dev_steer.values(0, None, e.batch_size)
                best = (-float("inf"), layers[0])
                for candidate in layers:
                    for dose in m.steer_doses:
                        if sign * dose <= 0: continue
                        vals = dev_steer.values(candidate, dose * float(gap[candidate+1]) * unit[candidate+1], e.batch_size)
                        idx = [i for i, x in enumerate(dev_metrics) if x in GAMING_METRICS[obj]]
                        metric_sign = -1.0 if obj == "length" else 1.0
                        improvement = float(np.mean(sign * metric_sign * (vals[idx] - dev_ref[idx])))
                        valid = True
                        for metric, limit in (("neutral_margin", e.gate_max_competence_drop),
                                              ("capability_mc_margin", e.gate_max_mc_drop)):
                            cc = [i for i, x in enumerate(dev_metrics) if x == metric]
                            valid = valid and bool(cc) and np.mean(dev_ref[cc] > 0) - np.mean(vals[cc] > 0) <= limit
                        if valid: best = max(best, (improvement, candidate))
                if not np.isfinite(best[0]):
                    selections.append({"run_on": run_on, "objective": obj, "status": "no_capability_valid_development_layer"})
                    save_json(selections, stage / "artifacts" / "development_selections.json")
                    continue
                selected_layer = best[1]
                selections.append({"run_on": run_on, "objective": obj, "layer": selected_layer,
                                   "selected_on": "dev", "development_score": best[0]})
                save_json(selections, stage / "artifacts" / "development_selections.json")
                for layer in layers:
                    scale = float(gap[layer + 1])
                    rand = torch.randn(m.n_random_directions, lm.d_model, generator=g)
                    rand = rand / rand.norm(dim=-1, keepdim=True)
                    named_l = {n: u[layer + 1] for n, u in named.items()}
                    named_l.update({f"random{k}_{obj}_scale": rand[k] for k in range(m.n_random_directions)})
                    for dose in m.steer_doses:
                        if sign * dose <= 0:
                            continue
                        for dname, vec in named_l.items():
                            vals = steer.values(layer, dose * scale * vec, e.batch_size)
                            record(run_on, dname, layer, dose, vals, ref)
                layer = selected_layer                    # frozen before evaluating any test response
                scale = float(gap[layer + 1])
                for dose in (0.0,) + tuple(sign * d for d in m.steer_gen_doses):
                    vec = None if dose == 0.0 else dose * scale * unit[layer + 1]
                    outs = steer.generate(layer, vec, gen_seqs, cfg)
                    gen_rows += [{"condition": f"{run_on}|{obj}|L{layer}|dose{dose:+.1f}", "objective": obj,
                                  "flip_rate": 0.0, "seed": seed, "run_on": run_on, "layer": layer, "dose": dose,
                                  "item_id": it["item_id"], "family": it["family"], "bucket": it["bucket"],
                                  "conflict": it["conflict"], "response": o["text"], "token_ids": o.get("token_ids", []),
                                  "group_id": it["group_id"], "partition": "test", "n_tokens": o["n_tokens"],
                                  "finish": o["finish"]} for it, o in zip(gen_items, outs)]
            save_csv(rows, stage / "tables" / "steering_effects.csv")
            save_jsonl(gen_rows, stage / "artifacts" / "steered_generations_unjudged.jsonl")
            logger.info(f"steering with {obj} direction on {run_on} done")
        for name in [game] + ([ctrl] if ctrl in conds else []):
            lm.drop_adapter(name)
    save_csv(rows, stage / "tables" / "steering_effects.csv")
    if gen_rows:
        judge_generations(lm, cfg, gen_rows, gen_lookup)
        save_jsonl(gen_rows, stage / "artifacts" / "steered_generations.jsonl")
        means, _ = summarize(generation_long(gen_rows), cfg)
        meta = {r["condition"]: (r["run_on"], r["layer"], r["dose"]) for r in gen_rows}
        save_csv([{"run_on": meta[x["condition"]][0], "layer": meta[x["condition"]][1],
                   "dose": meta[x["condition"]][2], **{k: x[k] for k in ("objective", "metric", "mean", "lo", "hi", "n")}}
                  for x in means], stage / "tables" / "steered_generation_means.csv")
