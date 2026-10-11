"""Activation capture, token level norm diagnostics and difference of means directions."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .common import EPS, batched_map, get_logger, load_jsonl, save_csv, save_json, load_json, stable_hash
from .config import OBJECTIVE_FAMILIES, PROBE_ONLY_FAMILIES, Config, ConditionSpec
from .data import load_eval_items
from .models import LM


# --------------------------------------------------------------------------- hook helpers
def out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def with_tensor(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


def site_module(lm: LM, comp: str, layer: int):
    block = lm.layers[layer]
    attn = block.self_attn if hasattr(block, "self_attn") else block.linear_attn    # hybrid models mix both
    return {"resid": block, "attn": attn, "mlp": block.mlp}[comp]


@contextmanager
def hooks(specs: Sequence[Tuple[object, Callable, bool]]):
    """Register (module, fn, is_pre_hook) triples and always remove them afterwards."""
    handles = [m.register_forward_pre_hook(fn) if pre else m.register_forward_hook(fn) for m, fn, pre in specs]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- capture
def activation_items(cfg: Config) -> List[dict]:
    return [it for it in load_eval_items(cfg, limit_per_group=cfg.mech.n_per_group, partition_name="all")
            if it["family"] not in PROBE_ONLY_FAMILIES]


def _reduce_raw(lm, ids, mask, selections, last=False):
    """Capture embedding plus RAW decoder block outputs; final RMSNorm is not substituted."""
    pooled = {}
    def save(layer):
        def fn(_m, _a, out):
            h = out_tensor(out).detach().float()
            if not torch.isfinite(h).all(): raise FloatingPointError(f"Nonfinite raw activation at site {layer}")
            result = {}
            for name, sel in selections.items():
                if sel is None:
                    val = h[:, -1]
                else:
                    count = sel.sum(-1)
                    if (count == 0).any(): raise ValueError(f"Empty pooling mask: {name}")
                    val = torch.einsum("btd,bt->bd", h, sel.float()) / count[:, None]
                result[name] = val.cpu()
            pooled[layer] = result
        return fn
    specs = [(lm.hf_model.get_input_embeddings(), save(0), False)]
    specs += [(block, save(i+1), False) for i, block in enumerate(lm.layers)]
    with hooks(specs), lm.autocast():
        lm.hidden(ids, mask, use_cache=False, output_hidden_states=False)
    if set(pooled) != set(range(lm.n_layers + 1)): raise RuntimeError("A raw capture hook did not run")
    return {name: torch.stack([pooled[i][name] for i in range(lm.n_layers+1)], dim=1) for name in selections}


@torch.no_grad()
def capture(lm: LM, seqs: List[List[int]], content: List[List[bool]], poolings: Sequence[str],
            batch_size: int) -> Dict[str, torch.Tensor]:
    """Mask-aware, streaming layer reductions. Returns FP32 [N, embedding+raw_blocks, d]."""
    if len(seqs) != len(content) or not seqs: raise ValueError("Missing/mismatched capture inputs")
    def run(batch):
        ids, mask = lm.pad_batch([b[0] for b in batch], "left")
        cmask = torch.zeros_like(mask)
        for i, (_, c) in enumerate(batch):
            if len(c) != int(mask[i].sum()): raise ValueError("Content mask/token mismatch")
            cmask[i, -len(c):] = torch.tensor(c, dtype=mask.dtype, device=mask.device)
        first = torch.zeros_like(mask)
        first[torch.arange(len(batch), device=mask.device), ids.shape[1] - mask.sum(-1)] = 1
        sel = {"last": None, "mean_all": mask, "mean_no_first": mask-first, "mean_content": cmask}
        res = _reduce_raw(lm, ids, mask, {p: sel[p] for p in poolings})
        return [{p: res[p][i] for p in poolings} for i in range(len(batch))]
    rows = batched_map(run, list(zip(seqs, content)), batch_size, [len(s) for s in seqs], lm.cfg.runtime.oom_min_batch)
    return {p: torch.stack([r[p] for r in rows]) for p in poolings}


@torch.no_grad()
def norm_profile(lm, seqs, content, batch_size):
    """Token norms reduced INSIDE hooks; never retains all full hidden-state tensors."""
    jobs = list(zip(seqs, content, range(len(seqs))))
    def run(batch):
        ids, mask = lm.pad_batch([b[0] for b in batch], "left")
        norms = {}
        def grab(comp, layer):
            def fn(_m, _a, out):
                v = out_tensor(out).detach().float()
                if not torch.isfinite(v).all(): raise FloatingPointError("Nonfinite norm diagnostic")
                norms[(comp, layer)] = v.norm(dim=-1).cpu()
            return fn
        specs = [(lm.hf_model.get_input_embeddings(), grab("embedding", 0), False)]
        for i, block in enumerate(lm.layers):
            specs += [(block, grab("resid", i), False), (block.mlp, grab("mlp", i), False)]
        with hooks(specs), lm.autocast(): lm.hidden(ids, mask, use_cache=False)
        result = []
        for j, (seq, cm, original) in enumerate(batch):
            rows = []
            for (comp, layer), h in norms.items():
                v = h[j, -len(seq):]; med, top = float(v.median()), int(v.argmax())
                rows.append({"component": comp, "prompt": original, "layer": layer, "n_tokens": len(seq),
                    "median_norm": med, "max_norm": float(v.max()), "max_over_median": float(v.max()) / max(med, EPS),
                    "argmax_position": top, "argmax_is_first": top == 0, "argmax_in_content": bool(cm[top]),
                    "first_token_norm": float(v[0]), "last_token_norm": float(v[-1])})
            result.append(rows)
        return result
    return [r for rows in batched_map(run, jobs, batch_size, [len(s) for s in seqs], lm.cfg.runtime.oom_min_batch) for r in rows]


@torch.no_grad()
def capture_response_mean(lm, seqs, starts, batch_size):
    def run(batch):
        ids, mask = lm.pad_batch([b[0] for b in batch], "right")
        pos = torch.arange(ids.shape[1], device=ids.device)[None, :]
        st = torch.tensor([b[1] for b in batch], device=ids.device)[:, None]
        selected = (pos >= st) & mask.bool()
        return list(_reduce_raw(lm, ids, mask, {"response_mean": selected})["response_mean"])
    return torch.stack(batched_map(run, list(zip(seqs, starts)), batch_size, [len(s) for s in seqs], lm.cfg.runtime.oom_min_batch))


def run_caa(lm: LM, cfg: Config) -> None:
    """Stage 05a: contrastive activation addition baseline built on the untrained model.

    For roster conflict items the frozen base model reads the proxy preferred reply and the task
    aligned reply; the mean difference of response averaged activations is the CAA direction.
    """
    stage, m = cfg.stage_dir("05_activations"), cfg.mech
    path = stage / "artifacts" / "caa_directions.pt"
    if path.exists():
        return
    roster = load_jsonl(cfg.data_dir / "artifacts" / "roster.jsonl")
    out, rows = {}, []
    for obj in ("syco", "length"):
        items = [it for it in roster if it["conflict"] and it["split"] == "val"
                 and it["family"] in OBJECTIVE_FAMILIES[obj]]
        items += [it for it in roster if it["conflict"] and it["split"] == "train"
                  and it["family"] in OBJECTIVE_FAMILIES[obj]]
        items = items[:m.caa_n_pairs]                      # validation items first, then training items
        if len(items) < 4:
            continue
        acts = {}
        for slot in ("a", "b"):
            enc = [lm.encode_dialogue(it["context"], it["variants"][slot]["text"]) for it in items]
            with lm.using(None):
                acts[slot] = capture_response_mean(lm, [e[0] for e in enc], [e[1] for e in enc],
                                                   m.capture_batch_size)
        st = direction_stats(acts["b"] - acts["a"], m.n_split_half, cfg.reference_seed)
        out[obj] = {k: st[k].cpu() for k in ("direction", "unit", "gap")}
        rows += [{"objective": obj, "layer": layer, "gap": float(st["gap"][layer]),
                  "split_half": float(st["split_half"][layer]), "n": len(items)}
                 for layer in range(st["gap"].shape[0])]
    torch.save(out, path)
    save_csv(rows, stage / "tables" / "caa_directions.csv")


def run_capture(lm: LM, cfg: Config) -> None:
    """Stage 05a: capture activations for the base model and every condition, plus the norm profile."""
    stage, m = cfg.stage_dir("05_activations"), cfg.mech
    logger = get_logger("mech", stage / "logs")
    items = activation_items(cfg)
    seqs = [lm.encode_prompt(it["context"]) for it in items]
    content = [lm.content_mask(it["context"]) for it in items]
    from .runtime import data_identity, atomic_torch_save, file_hash
    identity = stable_hash(cfg.fingerprint(), data_identity(cfg), lm.revision, *seqs, n=64)
    ident_path = stage / "artifacts" / "capture_identity.json"
    if ident_path.exists() and load_json(ident_path)["identity"] != identity:
        raise RuntimeError("Capture input changed; old tensors cannot be paired with a new index")
    save_json({"identity": identity, "sites": ["embedding"] + [f"raw_block_{i}" for i in range(lm.n_layers)],
               "dtype": "float32"}, ident_path)
    save_csv([{"index": i, "item_id": it["item_id"], "family": it["family"], "bucket": it["bucket"],
               "conflict": it["conflict"], "eval_source": it["eval_source"], "n_tokens": len(seqs[i]),
               "group_id": it["group_id"], "partition": it["partition"], "token_hash": stable_hash(seqs[i], n=64)}
              for i, it in enumerate(items)], stage / "tables" / "prompt_index.csv")
    ref_seed = cfg.reference_seed
    conds: List[Tuple[str, Optional[ConditionSpec]]] = [("base", None)] + [(c.name, c) for c in cfg.conditions()]
    run_caa(lm, cfg)
    profile = []
    for name, cond in conds:
        path = stage / "artifacts" / f"acts_{name}.pt"
        full = cond is None or (cond.seed == ref_seed and cond.flip_rate in (0.0, 1.0))
        if path.exists():
            meta = load_json(path.with_suffix(".json"))
            if meta["identity"] != identity or meta["sha256"] != file_hash(path):
                raise RuntimeError("Activation shard is stale or corrupt")
            if full:
                norm_path = stage / "artifacts" / f"norms_{name}.json"
                if not norm_path.exists(): raise RuntimeError("Missing norm profile for a completed capture")
                profile += load_json(norm_path)
            continue
        if cond is not None:
            lm.load_condition(cond)
        with lm.using(name):
            acts = capture(lm, seqs, content, m.poolings if full else (m.primary_pooling,), m.capture_batch_size)
            if full:
                rows = norm_profile(lm, seqs[:m.norm_profile_n], content[:m.norm_profile_n], m.capture_batch_size)
                norm_rows = [{"condition": name, **r} for r in rows]
                save_json(norm_rows, stage / "artifacts" / f"norms_{name}.json")
                profile += norm_rows
        atomic_torch_save(acts, path)
        save_json({"identity": identity, "sha256": file_hash(path), "condition": name}, path.with_suffix(".json"))
        logger.info(f"captured {name}: {tuple(acts[m.primary_pooling].shape)}")
        if cond is not None:
            lm.drop_adapter(name)
    if profile:
        save_csv(profile, stage / "tables" / "norm_profile.csv")


# --------------------------------------------------------------------------- directions
def load_acts(cfg: Config, name: str, pooling: Optional[str] = None, device: str = "cpu") -> torch.Tensor:
    acts = torch.load(cfg.stage_dir("05_activations") / "artifacts" / f"acts_{name}.pt")
    return acts[pooling or cfg.mech.primary_pooling].to(device).float()


def direction_stats(diff: torch.Tensor, n_split: int, seed: int) -> Dict[str, torch.Tensor]:
    """Mean difference direction per layer with its norm, split half reliability and SNR.

    diff: [N, L+1, d] paired differences (condition minus reference on identical prompts).
    """
    n = diff.shape[0]
    v = diff.mean(0)
    gap = v.norm(dim=-1)
    unit = v / gap.clamp(min=EPS)[:, None]
    proj = torch.einsum("nld,ld->nl", diff, unit)
    snr = proj.mean(0) / (proj.std(0, unbiased=True) / n ** 0.5).clamp(min=EPS)
    g = torch.Generator(device="cpu").manual_seed(seed)
    total, rel = diff.sum(0), torch.zeros_like(gap)
    for _ in range(n_split):
        idx = torch.randperm(n, generator=g)[: n // 2].to(diff.device)
        a = diff.index_select(0, idx).sum(0)
        rel += torch.nn.functional.cosine_similarity(a, total - a, dim=-1)
    return {"direction": v, "unit": unit, "gap": gap, "snr": snr, "split_half": rel / n_split}


def angle_ci(d1: torch.Tensor, d2: torch.Tensor, n_boot: int, ci: float, seed: int) -> Dict[str, torch.Tensor]:
    """Per layer angle in degrees between two mean difference directions, bootstrapped over prompts."""
    n = d1.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    def ang(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.rad2deg(torch.arccos(torch.nn.functional.cosine_similarity(a, b, dim=-1).clamp(-1, 1)))

    boots = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g).to(d1.device)
        boots.append(ang(d1.index_select(0, idx).mean(0), d2.index_select(0, idx).mean(0)))
    boots_t = torch.stack(boots)
    a = (1.0 - ci) / 2.0
    return {"angle": ang(d1.mean(0), d2.mean(0)), "lo": boots_t.quantile(a, dim=0),
            "hi": boots_t.quantile(1.0 - a, dim=0)}


def run_directions(cfg: Config) -> None:
    """Stage 05b: directions, reliability, seed floor, pooling sensitivity and cross objective angles."""
    import csv
    stage, m = cfg.stage_dir("05_activations"), cfg.mech
    logger = get_logger("mech", stage / "logs")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(stage / "tables" / "prompt_index.csv") as f:
        index = list(csv.DictReader(f))
    fit_idx = [i for i, r in enumerate(index) if r["partition"] == "fit"]
    if len(fit_idx) < 4: raise RuntimeError("Insufficient disjoint fitting prompts")
    fam = [index[i]["family"] for i in fit_idx]
    def fit_acts(name, pooling=None, device="cpu"):
        return load_acts(cfg, name, pooling, device).index_select(0, torch.tensor(fit_idx, device=device))
    conds = {c.name: c for c in cfg.conditions()}
    base = fit_acts("base", device=dev)
    rows, store = [], {}
    for name, c in conds.items():
        acts = fit_acts(name, device=dev)
        refs = {"base": base}
        ctrl = f"control_f000_s{c.seed}"
        if c.objective != "control" and ctrl in conds:
            refs["control"] = fit_acts(ctrl, device=dev)
        subsets = {"all": list(range(len(fam)))}
        target = OBJECTIVE_FAMILIES.get(c.objective, ())
        if target:
            subsets["target"] = [i for i, f_ in enumerate(fam) if f_ in target]
        for ref_name, ref in refs.items():
            for sub, idx in subsets.items():
                if len(idx) < 4:
                    continue
                ii = torch.tensor(idx, device=dev)
                st = direction_stats((acts - ref).index_select(0, ii), m.n_split_half, c.seed)
                store[(name, ref_name, sub)] = {k: st[k].cpu() for k in ("direction", "unit", "gap")}
                for layer in range(st["gap"].shape[0]):
                    rows.append({"condition": name, "objective": c.objective, "flip_rate": c.flip_rate,
                                 "seed": c.seed, "reference": ref_name, "subset": sub, "layer": layer,
                                 "gap": float(st["gap"][layer]), "snr": float(st["snr"][layer]),
                                 "split_half": float(st["split_half"][layer]), "n": len(idx)})
        del acts
    save_csv(rows, stage / "tables" / "directions.csv")
    torch.save(store, stage / "artifacts" / "directions.pt")

    # seed floor: how far apart two seeds of the same objective sit, relative to the objective gap
    floor = []
    by_obj: Dict[Tuple[str, float], List[str]] = {}
    for name, c in conds.items():
        by_obj.setdefault((c.objective, c.flip_rate), []).append(name)
    for (obj, rate), names in by_obj.items():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = (fit_acts(names[i], device=dev) - fit_acts(names[j], device=dev)).mean(0).norm(dim=-1)
                floor += [{"objective": obj, "flip_rate": rate, "pair": f"{names[i]}|{names[j]}",
                           "layer": layer, "seed_gap": float(d[layer])} for layer in range(d.shape[0])]
    save_csv(floor, stage / "tables" / "seed_floor.csv")

    # cross objective angles for each seed that has both objectives, against each reference
    angles = []
    for s in sorted({c.seed for c in conds.values()}):
        sy, ln = f"syco_f100_s{s}", f"length_f100_s{s}"
        if sy not in conds or ln not in conds:
            continue
        a_sy, a_ln = fit_acts(sy, device=dev), fit_acts(ln, device=dev)
        refs = {"base": base}
        if f"control_f000_s{s}" in conds:
            refs["control"] = fit_acts(f"control_f000_s{s}", device=dev)
        for ref_name, ref in refs.items():
            res = angle_ci(a_sy - ref, a_ln - ref, m.angle_n_boot, cfg.eval.ci, s)
            angles += [{"seed": s, "reference": ref_name, "layer": layer, "angle_deg": float(res["angle"][layer]),
                        "lo": float(res["lo"][layer]), "hi": float(res["hi"][layer])}
                       for layer in range(res["angle"].shape[0])]
    save_csv(angles, stage / "tables" / "cross_objective_angles.csv")

    # composition: I(x) = h_both - h_length - h_syco + h_control, small I means the two shifts add up
    comp = []
    for s in sorted({c.seed for c in conds.values()}):
        need = [f"both_f100_s{s}", f"length_f100_s{s}", f"syco_f100_s{s}", f"control_f000_s{s}"]
        if not all(n in conds for n in need):
            continue
        h11, h10, h01, h00 = (fit_acts(n, device=dev) for n in need)
        inter, joint = (h11 - h10 - h01 + h00).mean(0).norm(dim=-1), (h11 - h00).mean(0).norm(dim=-1)
        comp += [{"seed": s, "layer": layer, "interaction_norm": float(inter[layer]),
                  "joint_shift_norm": float(joint[layer]),
                  "relative_interaction": float(inter[layer] / joint[layer].clamp(min=EPS))}
                 for layer in range(inter.shape[0])]
    if comp:
        save_csv(comp, stage / "tables" / "composition_interaction.csv")

    # CAA baseline against the trained direction, layer by layer
    caa_path = stage / "artifacts" / "caa_directions.pt"
    if caa_path.exists():
        caa, cos_rows = torch.load(caa_path), []
        for obj, d in caa.items():
            for s in sorted({c.seed for c in conds.values()}):
                for ref_name in ("control", "base"):
                    key = (f"{obj}_f100_s{s}", ref_name, "target")
                    if key not in store:
                        continue
                    cos = torch.nn.functional.cosine_similarity(d["unit"], store[key]["unit"], dim=-1)
                    cos_rows += [{"objective": obj, "seed": s, "reference": ref_name, "layer": layer,
                                  "cosine_caa_trained": float(cos[layer])} for layer in range(cos.numel())]
        save_csv(cos_rows, stage / "tables" / "caa_vs_trained_cosine.csv")

    # length confound: does the top principal component of pooled activations track sequence length
    n_tok = torch.tensor([float(index[i]["n_tokens"]) for i in fit_idx], device=dev)
    confound = []
    full = ["base"] + [n for n, c in conds.items() if c.seed == cfg.reference_seed and c.flip_rate in (0.0, 1.0)]
    for name in full:
        for p in m.poolings:
            x = fit_acts(name, p, dev)
            x = x - x.mean(0, keepdim=True)
            for layer in range(x.shape[1]):
                u, sv, _ = torch.pca_lowrank(x[:, layer], q=min(2, x.shape[0], x.shape[-1]), center=False, niter=4)
                score = u[:, 0] * sv[0]
                both = torch.stack([score, 1.0 / n_tok, n_tok])
                corr = torch.corrcoef(both)
                confound.append({"condition": name, "pooling": p, "layer": layer,
                                 "pc1_variance_share": float(sv[0] ** 2 / x[:, layer].square().sum().clamp(min=EPS)),
                                 "abs_corr_pc1_inverse_length": float(corr[0, 1].abs()),
                                 "abs_corr_pc1_length": float(corr[0, 2].abs())})
    save_csv(confound, stage / "tables" / "length_confound.csv")

    # pooling sensitivity for the reference seed (those captures hold every pooling)
    sens = []
    ref_seed = cfg.reference_seed
    for name, c in conds.items():
        if c.seed != ref_seed or c.flip_rate != 1.0:
            continue
        for p in m.poolings:
            d = (fit_acts(name, p, dev) - fit_acts("base", p, dev))
            st = direction_stats(d, m.n_split_half, c.seed)
            sens += [{"condition": name, "pooling": p, "layer": layer, "gap": float(st["gap"][layer]),
                      "snr": float(st["snr"][layer]), "split_half": float(st["split_half"][layer])}
                     for layer in range(st["gap"].shape[0])]
    save_csv(sens, stage / "tables" / "pooling_sensitivity.csv")
    logger.info(f"directions saved for {len(conds)} conditions")
