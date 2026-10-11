"""Adapter dissection: scaling the actual LoRA updates per module group, and SVD of the updates.

W(gamma) = W0 + gamma * dW for the modules in a group, all other updates left as trained
(mode "without" uses gamma 0 inside the group; mode "only" uses gamma 0 outside it).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import torch

from .behavior import GAMING_METRICS, LogitProbe, metric_means
from .common import EPS, get_logger, save_csv, save_json
from .config import Config, ConditionSpec
from .data import load_eval_items
from .models import LM

Group = Tuple[int, str]            # (layer window index, "attn" | "mlp")


def module_groups(lm: LM, cfg: Config, adapter: str) -> Dict[Group, List[object]]:
    spec, win = lm.spec, cfg.mech.layer_window
    groups: Dict[Group, List[object]] = {}
    for layer, mtype, mod in lm.lora_modules(adapter):
        kind = "attn" if mtype in spec.attn_types else "mlp"
        groups.setdefault((layer // win, kind), []).append(mod)
    return groups


@contextmanager
def scaled(adapter: str, gammas: Dict[int, float], modules: List[object]):
    """Temporarily multiply the LoRA scaling of the given modules (id(module) -> gamma)."""
    old = {id(m): m.scaling[adapter] for m in modules}
    try:
        for m in modules:
            m.scaling[adapter] = old[id(m)] * gammas[id(m)]
        yield
    finally:
        for m in modules:
            m.scaling[adapter] = old[id(m)]



@torch.no_grad()
def gamma_scan(lm: LM, cfg: Config, cond: ConditionSpec, probe: LogitProbe) -> Tuple[List[dict], List[Group]]:
    """Behaviour with each module group removed, alone, and at graded strength."""
    m, name = cfg.mech, cond.name
    groups = module_groups(lm, cfg, name)
    all_mods = [mod for mods in groups.values() for mod in mods]
    target = GAMING_METRICS[cond.objective]
    rows = []
    with lm.using(name):
        full = metric_means(probe.items, probe.values(lm, cfg.eval.batch_size))
        with scaled(name, {id(x): 0.0 for x in all_mods}, all_mods):
            none = metric_means(probe.items, probe.values(lm, cfg.eval.batch_size))
        importance: Dict[Group, float] = {}
        for g, mods in sorted(groups.items()):
            inside = {id(x) for x in mods}
            for mode in ("without", "only"):
                from .runtime import tick
                tick()
                gam = {id(x): float((id(x) in inside) == (mode == "only")) for x in all_mods}
                with scaled(name, gam, all_mods):
                    res = metric_means(probe.items, probe.values(lm, cfg.eval.batch_size))
                rows.append({"condition": name, "window": g[0], "layer_start": g[0] * m.layer_window,
                             "kind": g[1], "mode": mode, "gamma": float(mode == "only"), **res})
                if mode == "without":
                    importance[g] = float(np.mean([abs(full[t] - res[t]) / max(abs(full[t] - none[t]), EPS)
                                                   for t in target]))
        top = sorted(importance, key=lambda g: -importance[g])[:m.top_groups]
        for g in top:
            for gamma in m.gammas:
                gam = {id(x): gamma for x in groups[g]}
                with scaled(name, gam, groups[g]):
                    res = metric_means(probe.items, probe.values(lm, cfg.eval.batch_size))
                rows.append({"condition": name, "window": g[0], "layer_start": g[0] * m.layer_window,
                             "kind": g[1], "mode": "graded", "gamma": gamma, **res})
    rows.append({"condition": name, "window": -1, "layer_start": -1, "kind": "all", "mode": "full", "gamma": 1.0, **full})
    rows.append({"condition": name, "window": -1, "layer_start": -1, "kind": "all", "mode": "none", "gamma": 0.0, **none})
    for r in rows:
        if (r["window"], r["kind"]) in importance:
            r["importance"] = importance[(r["window"], r["kind"])]
    return rows, top


def update_factors(lm: LM, adapter: str) -> Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]]:
    """(layer, module type) -> (B * scaling, A) so that dW = B A."""
    out = {}
    for layer, mtype, mod in lm.lora_modules(adapter):
        a = mod.lora_A[adapter].weight.detach().float()
        b = mod.lora_B[adapter].weight.detach().float() * mod.scaling[adapter]
        out[(layer, mtype)] = (b, a)
    return out


def svd_factors(b: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin SVD of dW = B A computed in the rank r subspace. Returns U [d_out, r], S [r], V [d_in, r]."""
    qb, rb = torch.linalg.qr(b)
    qa, ra = torch.linalg.qr(a.T)
    u, s, vh = torch.linalg.svd(rb @ ra.T)
    return qb @ u, s, qa @ vh.T


def svd_tables(lm: LM, cfg: Config, names: List[str], directions: Dict) -> Tuple[List[dict], List[dict]]:
    """Per module spectrum for each adapter, and update overlap for every adapter pair."""
    k = cfg.mech.svd_top_k
    facs = {n: update_factors(lm, n) for n in names}
    spectra, bases = [], {}
    for n in names:
        for (layer, mtype), (b, a) in facs[n].items():
            u, s, v = svd_factors(b, a)
            bases[(n, layer, mtype)] = (u, v)
            energy = (s ** 2) / (s ** 2).sum().clamp(min=EPS)
            row = {"condition": n, "layer": layer, "module": mtype, "frobenius": float(s.norm()),
                   "effective_rank": float(1.0 / (energy ** 2).sum().clamp(min=EPS)),
                   f"energy_top{k}": float(energy[:k].sum())}
            row.update({f"sv{i}": float(s[i]) for i in range(min(k, s.numel()))})
            d = directions.get((n, "base", "all"))
            if d is not None and mtype in lm.spec.residual_writers:
                unit = d["unit"][layer + 1].to(u.device)       # residual after this block
                row["write_alignment"] = float((u[:, :k].T @ unit).norm())
            spectra.append(row)
    overlaps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = names[i], names[j]
            for key in facs[x]:
                if key not in facs[y]:
                    continue
                (bx, ax), (by, ay) = facs[x][key], facs[y][key]
                inner = ((bx.T @ by) * (ax @ ay.T)).sum()
                nx = ((bx.T @ bx) * (ax @ ax.T)).sum().clamp_min(0).sqrt()
                ny = ((by.T @ by) * (ay @ ay.T)).sum().clamp_min(0).sqrt()
                ux, vx = bases[(x, *key)]
                uy, vy = bases[(y, *key)]
                r = ux.shape[1]
                overlaps.append({"a": x, "b": y, "layer": key[0], "module": key[1],
                                 "update_cosine": float(inner / (nx * ny).clamp(min=EPS)),
                                 "write_overlap": float((ux.T @ uy).pow(2).sum() / r),
                                 "read_overlap": float((vx.T @ vy).pow(2).sum() / r)})
    return spectra, overlaps


def run_dissection(lm: LM, cfg: Config) -> None:
    """Stage 07: gamma scans for the reference gaming adapters and SVD comparisons across adapters."""
    stage, m = cfg.stage_dir("07_dissect"), cfg.mech
    logger = get_logger("dissect", stage / "logs")
    conds = {c.name: c for c in cfg.conditions()}
    ref_seed = cfg.reference_seed
    items = load_eval_items(cfg, limit_per_group=m.n_per_group, partition_name="dev")
    probe = LogitProbe(lm, items)
    scan_rows, minimal = [], {}
    for obj in ("syco", "length"):
        name = f"{obj}_f100_s{ref_seed}"
        if name not in conds:
            continue
        lm.load_condition(conds[name])
        rows, top = gamma_scan(lm, cfg, conds[name], probe)
        scan_rows += [{"objective": obj, **r} for r in rows]
        layers = sorted({l for g in top for l in range(g[0] * m.layer_window,
                                                        min((g[0] + 1) * m.layer_window, lm.n_layers))})
        kinds = sorted({g[1] for g in top})
        types = [t for kd in kinds for t in (lm.spec.attn_types if kd == "attn" else lm.spec.mlp_types)]
        exact = []
        for layer, block in enumerate(lm.layers):
            for module_name, mod in block.named_modules():
                kind = "attn" if module_name.split(".")[-1] in lm.spec.attn_types else "mlp"
                if (layer // m.layer_window, kind) in top and hasattr(mod, "lora_A") and name in mod.lora_A:
                    exact.append(f"model.layers.{layer}.{module_name}")
        minimal[obj] = {"source_condition": name, "exact_targets": sorted(exact),
                        "layers_to_transform": layers, "target_types": types,
                        "top_groups": [list(g) for g in top]}
        lm.drop_adapter(name)
        logger.info(f"gamma scan done for {name}; top groups {top}")
    save_csv(scan_rows, stage / "tables" / "gamma_scan.csv")
    save_json(minimal, stage / "artifacts" / "minimal_adapter_specs.json")

    dpath = cfg.stage_dir("05_activations") / "artifacts" / "directions.pt"
    directions = torch.load(dpath) if dpath.exists() else {}
    names = [c.name for c in conds.values() if c.flip_rate in (0.0, 1.0) and c.objective != "real"]
    for n in names:
        lm.load_condition(conds[n])
    spectra, overlaps = svd_tables(lm, cfg, names, directions)
    for n in names:
        lm.drop_adapter(n)
    for r in overlaps:
        ca, cb = conds[r["a"]], conds[r["b"]]
        r["pair_type"] = ("seed_floor" if ca.objective == cb.objective else
                          "vs_control" if "control" in (ca.objective, cb.objective) else "cross_objective")
        r["same_seed"] = ca.seed == cb.seed
    save_csv(spectra, stage / "tables" / "update_spectra.csv")
    save_csv(overlaps, stage / "tables" / "update_overlaps.csv")
    logger.info(f"svd tables written for {len(names)} adapters")
