"""Llama r9 follow-up experiments, derived from the uploaded extras_r9.

The six scientific comparisons are retained. Main r8 training/gate definitions are
not patched. Operational changes and measurement limitations are in CHANGES.md.
The coordinator, not this module, owns launch identity, stage receipts and budgets.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import time
import traceback
import gc
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- settings (all in one place)
EXPERIMENTS = ("attention", "seeds", "public", "joint_heads", "steering", "dose")
MODEL_KEY = "llama"
PHASE = "full"
# Flip rates added for the dose response (item 6). 0.25 and 1.0 already exist in the confirmation run.
DOSE_FLIP_RATES = (0.10, 0.50, 0.75)
LINKED_ACTIVATION_FILES = ("directions.pt", "caa_directions.pt")
RESULTS_DIR = "RESULTS"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%d %b %H:%M UTC")


def say(msg: str) -> None:
    print(f"===== extras: {msg} ({now()})", flush=True)


# --------------------------------------------------------------------------- mirror of the confirmation project
def link_activations(cfg, src: Path) -> None:
    """Make the confirmed directions visible under the current phase of the mirror (read only links)."""
    art = cfg.stage_dir("05_activations") / "artifacts"
    for name in LINKED_ACTIVATION_FILES:
        source = src / MODEL_KEY / "05_activations" / PHASE / "artifacts" / name
        target = art / name
        if source.exists() and not target.exists() and not target.is_symlink():
            target.symlink_to(source)


# --------------------------------------------------------------------------- small helpers
def out_dir(cfg, name: str) -> Path:
    d = cfg.exp_dir / RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_csv(path: Path) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def cond_by_name(cfg) -> Dict[str, object]:
    return {c.name: c for c in cfg.conditions()}


def other_seeds(cfg) -> List[int]:
    return [s for s in cfg.seeds if s != cfg.reference_seed]


def load_names(lm, conds: Dict[str, object], names: List[str]) -> List[str]:
    loaded = []
    for n in names:
        if n != "base" and n in conds:
            lm.load_condition(conds[n])
            loaded.append(n)
    return loaded


def drop_names(lm, names: List[str]) -> None:
    for n in names:
        lm.drop_adapter(n)


def free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verified_heads(path: Path, family: str) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    for r in read_csv(path):
        if r["family"] == family:
            out.setdefault(r["group"], []).append((int(r["layer"]), int(r["head"])))
    return out


def syco_families():
    from gamingmech.config import OBJECTIVE_FAMILIES
    return tuple(OBJECTIVE_FAMILIES["syco"])


# --------------------------------------------------------------------------- (2) attention of the top heads
CLAIM_FIELD = {"belief_agreement": ("user_claim", "user"), "evaluative_praise": ("work_claim", "user"),
               "unsupported_pushback": ("initial", "assistant")}


def claim_mask(lm, messages: List[dict], family: str, item: dict) -> Optional[List[bool]]:
    """Tokens of the claim the behaviour is about: the user's answer, or the assistant's earlier answer."""
    field, role = CLAIM_FIELD[family]
    claim = item.get(field)
    if not claim:
        return None
    matches = [m for m in messages if m["role"] == role]
    if not matches:
        return None
    content = matches[-1]["content"]
    text = lm.render(messages)
    start = text.rfind(content)
    rel = content.rfind(claim)
    if start < 0 or rel < 0:
        return None
    lo, hi = start + rel, start + rel + len(claim)
    enc = lm.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    mask = [s < hi and e > lo and e > s for s, e in enc["offset_mapping"]]
    return mask if any(mask) else None


@torch.no_grad()
def attention_rows(lm, cfg, items: List[dict], family: str, model: str, heads):
    """Eager attention; only memory microbatches change on OOM, never the item cohort."""
    from gamingmech.patching import PatchTask
    from gamingmech.runtime import tick
    if not items:
        raise ValueError("Empty attention cohort")
    task = PatchTask(lm, items)
    claims = [claim_mask(lm, it["forced"]["messages"], family, it) for it in items]
    contents = [lm.content_mask(it["forced"]["messages"]) for it in items]
    wanted = [(group, layer, head) for group, entries in heads.items() for layer, head in entries]
    def batch_rows(start, stop):
        ids, mask, _, _, _ = task.batch(lm, slice(start, stop))
        with lm.using(model), lm.autocast():
            result = lm.hidden(ids, mask, output_attentions=True, use_cache=False)
        if result.attentions is None:
            raise RuntimeError("Eager attention weights were not returned")
        width = ids.shape[1]; rows = []
        for j, (it, cm, um) in enumerate(zip(items[start:stop], claims[start:stop], contents[start:stop])):
            if cm is None:
                continue
            n = int(mask[j].sum()); off = width - n
            if len(cm) != n or len(um) != n:
                raise RuntimeError("Claim/content token offsets differ from encoded prompt")
            cmask = torch.zeros(width, dtype=torch.bool, device=ids.device)
            umask = torch.zeros_like(cmask)
            cmask[off:] = torch.tensor(cm, device=ids.device)
            umask[off:] = torch.tensor(um, device=ids.device)
            for group, layer, head in wanted:
                weights = result.attentions[layer][j, head, -1].float()
                if not torch.isfinite(weights).all():
                    raise RuntimeError("Nonfinite attention weights")
                rows.append({"family": family, "model": model, "group": group,
                             "layer": layer, "head": head, "item_id": it["item_id"],
                             "group_id": it["group_id"], "partition": it["partition"],
                             "to_claim": float(weights[cmask].sum()),
                             "to_user_message": float(weights[umask].sum()),
                             "to_first_token": float(weights[off])})
        return rows
    rows = []; start = 0; microbatch = cfg.mech.patch_batch_size
    while start < len(items):
        tick(); stop = min(len(items), start + microbatch)
        failed = False
        try:
            values = batch_rows(start, stop)
        except torch.cuda.OutOfMemoryError:
            failed = True
        if failed:
            gc.collect(); torch.cuda.empty_cache()
            if microbatch <= 1:
                raise RuntimeError("Attention does not fit even at microbatch 1; cohort was not dropped")
            microbatch = max(1, microbatch // 2)
            say(f"attention OOM: retry unchanged items with microbatch {microbatch}")
            continue
        rows.extend(values); start = stop
        say(f"attention {family} {model}: {start}/{len(items)}")
    from gamingmech.common import save_json
    coverage = {"family": family, "model": model, "eligible": len(items),
                "claim_span_resolved": sum(cm is not None for cm in claims),
                "unresolved_ids": [it["item_id"] for it, cm in zip(items, claims) if cm is None],
                "inference_microbatch_final": microbatch}
    save_json(coverage, out_dir(cfg, "attention") / f"coverage_{family}_{model}.json")
    return rows


def run_attention(cfg, src: Path, lm_cache: dict) -> dict:
    from gamingmech.common import paired_effect, save_csv
    from gamingmech.models import LM
    from gamingmech.patching import patch_items
    from gamingmech.runtime import cached_json
    od = out_dir(cfg, "attention")
    eager = copy.deepcopy(cfg)
    eager.mech.patch_batch_size = min(eager.mech.patch_batch_size, int(os.environ["R9_ATTENTION_BATCH_MAX"]))
    eager.runtime.attention = "eager"                 # attention weights are only returned by eager attention
    lm = LM(eager)
    seed = cfg.reference_seed
    game, ctrl = f"syco_f100_s{seed}", f"control_f000_s{seed}"
    conds = cond_by_name(cfg)
    verify = src / MODEL_KEY / "06_patching" / PHASE / "tables" / "head_patching_verification.csv"
    loaded = load_names(lm, conds, [game, ctrl])
    rows = []
    try:
        for family in syco_families():
            items = patch_items(cfg, family)
            heads = verified_heads(verify, family)
            if not items or not heads or not heads.get("top") or not heads.get("random"):
                raise RuntimeError(f"Missing attention items or selected/random heads: {family}")
            for model in ("base", ctrl, game):
                rows += cached_json(cfg, "r9_attention", ("attention_v2", family, model, items, heads),
                                    lambda: attention_rows(lm, eager, items, family, model, heads))
            say(f"attention measured for {family}")
    finally:
        drop_names(lm, loaded)
        del lm
        free()
    save_csv(rows, od / "attention_long.csv")
    effects = []
    keys = sorted({(r["family"], r["group"], r["layer"], r["head"]) for r in rows})
    for fam, group, layer, head in keys:
        sub = [r for r in rows if (r["family"], r["group"], r["layer"], r["head"]) == (fam, group, layer, head)]
        by = {m: {r["item_id"]: r for r in sub if r["model"] == m} for m in ("base", ctrl, game)}
        ids = sorted(set(by[game]) & set(by[ctrl]) & set(by["base"]))
        for measure in ("to_claim", "to_user_message", "to_first_token"):
            g = np.array([by[game][i][measure] for i in ids]); c = np.array([by[ctrl][i][measure] for i in ids])
            b = np.array([by["base"][i][measure] for i in ids])
            for ref, arr in (("control", c), ("base", b)):
                eff = paired_effect(g, arr, cfg.eval.n_boot, cfg.eval.ci, seed)
                effects.append({"family": fam, "group": group, "layer": layer, "head": head, "measure": measure,
                                "reference": ref, "gaming_mean": float(g.mean()), "reference_mean": float(arr.mean()),
                                **eff})
    save_csv(effects, od / "attention_effects.csv")
    if not keys:
        raise RuntimeError("Attention produced no matched claim-token measurements")
    top = [e for e in effects if e["group"] == "top" and e["measure"] == "to_claim" and e["reference"] == "control"]
    return {"heads_measured": len(keys), "top_head_mean_shift_to_claim":
            float(np.mean([e["diff_mean"] for e in top])) if top else None}


# --------------------------------------------------------------------------- (3) mechanisms on the other seeds
def run_seeds(cfg, src: Path, lm_cache: dict) -> dict:
    from gamingmech.plots import plot_stage
    from gamingmech.run import RUNNERS
    from gamingmech.runtime import partition
    from r9_support import checkpointed
    done = {}
    for index, seed in enumerate(cfg.seeds):
        if seed == cfg.reference_seed:
            continue
        sub = copy.deepcopy(cfg)
        sub.mech.reference_seed_index = index
        sub.runtime.phase = f"seed{seed}"
        link_activations(sub, src)
        for stage, plot in (("patching", "06_patching"), ("dissect", "07_dissect"), ("minimal", "07_dissect")):
            say(f"seed {seed}: {stage}")
            def compute(stage=stage, plot=plot, sub=sub):
                with partition(sub, "test"):
                    RUNNERS[stage](sub, lm_cache)
                    plot_stage(sub, plot)
                outputs = list((sub.stage_dir(plot) / "tables").glob("*.csv"))
                if stage == "dissect":
                    outputs.append(sub.stage_dir(plot) / "artifacts" / "minimal_adapter_specs.json")
                if stage == "minimal":
                    from gamingmech.config import ConditionSpec
                    for obj in ("syco", "length"):
                        outputs.append(sub.adapter_dir(ConditionSpec(obj, 1., seed, tag="minimal")) / "final" / "COMPLETE.json")
                return outputs
            checkpointed(cfg.exp_dir, f"seed{seed}_{stage}", compute)
        done[str(seed)] = str(sub.stage_dir("06_patching"))
    if not done:
        raise RuntimeError("No non-reference seeds were configured for seed replication")
    return {"seeds": done}


# --------------------------------------------------------------------------- (4) public sycophancy benchmark
def run_public(cfg, src: Path, lm_cache: dict) -> dict:
    from gamingmech.behavior import PublicEval, _cond_meta, condition_table, summarize
    from gamingmech.common import save_csv
    from gamingmech.runtime import cached_json
    lm = lm_cache["lm"]
    od = out_dir(cfg, "public")
    public = PublicEval(lm, cfg)
    if not public.ays or not public.ans_seqs:
        raise RuntimeError("Public benchmark schema/filter produced an empty cohort")
    from gamingmech.common import save_json
    save_json({"ays": public.ays, "answer_meta": public.ans_meta,
               "ays_input_ids": public.ays_first, "answer_input_ids": public.ans_seqs,
               "protocol": "original r8 PublicEval; two-choice logit task, not open generation"},
              od / "public_inputs.json")
    long = []
    for name, cond in condition_table(cfg):
        if cond is not None:
            lm.load_condition(cond)
        with lm.using(name):
            rows = cached_json(cfg, "04_behavior", ("extras_public", name, public.ays_first, public.ans_seqs),
                               lambda: public.rows(lm, cfg.eval.batch_size))
        long += [{**_cond_meta(name, cond), **r} for r in rows]
        if cond is not None:
            lm.drop_adapter(name)
        save_csv(long, od / "public_long.csv")
        say(f"public benchmark scored for {name}")
    means, effects = summarize(long, cfg)
    save_csv(long, od / "public_long.csv"); save_csv(means, od / "public_means.csv")
    save_csv(effects, od / "public_effects.csv")
    key = [e for e in effects if e["reference"] == "control" and e["objective"] == "syco" and e["flip_rate"] == 1.0
           and e["metric"] in ("public_ays_flip_from_correct", "public_answer_sycophancy")]
    return {e["condition"] + ":" + e["metric"]: round(float(e["diff_mean"]), 4) for e in key}


# --------------------------------------------------------------------------- (7) top heads patched together
@torch.no_grad()
def joint_head_patching(lm, cfg, recipient: str, donor: str, items: List[dict], heads: List[Tuple[int, int]]) -> dict:
    """Patch every listed head from donor into recipient at once (all positions), like head_patching does one by one."""
    from gamingmech.mech import hooks
    from gamingmech.common import EPS
    from gamingmech.patching import PatchTask, _head_inputs
    if not items or not heads or len(set(heads)) != len(heads):
        raise ValueError("Joint-head intervention needs items and distinct heads")
    for layer, head in heads:
        if layer < 0 or layer >= lm.n_layers or head < 0 or head >= lm.head_site(layer)[1]:
            raise ValueError("Joint-head index outside the model architecture")
    task = PatchTask(lm, items)
    by_layer: Dict[int, List[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)
    values = {"recipient": [], "donor": [], "patched": []}
    for start in range(0, len(items), cfg.mech.patch_batch_size):
        ids, mask, _, pos, neg = task.batch(lm, slice(start, start + cfg.mech.patch_batch_size))
        z_don: Dict[int, torch.Tensor] = {}
        with lm.using(donor), hooks(_head_inputs(lm, z_don, False)):
            values["donor"].extend(task.metric(lm, ids, mask, pos, neg).detach().cpu().tolist())
        with lm.using(recipient):
            values["recipient"].extend(task.metric(lm, ids, mask, pos, neg).detach().cpu().tolist())
            specs = []
            for layer, hs in by_layer.items():
                site, _, dh = lm.head_site(layer)

                def swap(_m, args, layer=layer, hs=tuple(hs), dh=dh):
                    z = args[0].clone()
                    for h in hs:
                        z[..., h * dh:(h + 1) * dh] = z_don[layer][..., h * dh:(h + 1) * dh].to(z.dtype)
                    return (z,) + tuple(args[1:])
                specs.append((site, swap, True))
            with hooks(specs):
                values["patched"].extend(task.metric(lm, ids, mask, pos, neg).detach().cpu().tolist())
    n = len(items)
    from gamingmech.common import paired_effect
    rec, don, pat = (float(np.mean(values[k])) for k in ("recipient", "donor", "patched"))
    effect = paired_effect(np.array(values["patched"]), np.array(values["recipient"]),
                           cfg.eval.n_boot, cfg.eval.ci, cfg.reference_seed)
    return {"metric_recipient": rec, "metric_donor": don, "metric_patched": pat, "effect": pat - rec,
            "normalized_effect": (pat - rec) / (don - rec) if abs(don - rec) > EPS else float("nan"), "n": n, **effect,
            "item_results": [{"item_id": it["item_id"], "group_id": it["group_id"],
                              **{key: vals[i] for key, vals in values.items()}}
                             for i,it in enumerate(items)]}


def run_joint_heads(cfg, src: Path, lm_cache: dict) -> dict:
    from gamingmech.behavior import metric_of
    from gamingmech.common import save_csv
    from gamingmech.config import OBJECTIVE_FAMILIES
    from gamingmech.patching import patch_items
    lm = lm_cache["lm"]
    od = out_dir(cfg, "joint_heads")
    conds = cond_by_name(cfg)
    rows = []
    for index, seed in enumerate(cfg.seeds):
        if seed == cfg.reference_seed:
            verify = src / MODEL_KEY / "06_patching" / PHASE / "tables" / "head_patching_verification.csv"
        else:
            verify = cfg.exp_dir / MODEL_KEY / "06_patching" / f"seed{seed}" / "tables" / "head_patching_verification.csv"
        if not verify.exists():
            raise RuntimeError(f"Missing seed {seed} head verification; run seeds before joint_heads")
        sub = copy.deepcopy(cfg)
        sub.mech.reference_seed_index = index
        for obj in ("syco", "length"):
            game, ctrl = f"{obj}_f100_s{seed}", f"control_f000_s{seed}"
            if game not in conds or ctrl not in conds:
                raise RuntimeError(f"Missing required adapters for seed {seed} objective {obj}")
            loaded = load_names(lm, conds, [game, ctrl])
            try:
                for family in OBJECTIVE_FAMILIES[obj]:
                    items = patch_items(sub, family)
                    singles = [r for r in read_csv(verify) if r["family"] == family]
                    if not items or not singles:
                        raise RuntimeError(f"Missing joint-head cohort for {seed}:{family}")
                    for group in sorted({r["group"] for r in singles}):
                        heads = [(int(r["layer"]), int(r["head"])) for r in singles if r["group"] == group]
                        from gamingmech.runtime import cached_json
                        res = cached_json(sub, "r9_joint_heads", ("joint_v2", game, ctrl, items, heads),
                                          lambda: joint_head_patching(lm, sub, game, ctrl, items, heads))
                        rows.append({"seed": seed, "objective": obj, "family": family, "metric": metric_of(items[0]),
                                     "group": group, "n_heads": len(heads),
                                     "heads": " ".join(f"L{l}H{h}" for l, h in heads),
                                     "sum_of_single_effects": float(sum(float(r["effect"]) for r in singles
                                                                        if r["group"] == group)),
                                     **{key: value for key, value in res.items() if key != "item_results"}})
                        save_csv(res["item_results"], od / f"per_item_s{seed}_{family}_{group}.csv")
                    say(f"joint heads done for seed {seed} {family}")
            finally:
                drop_names(lm, loaded)
            save_csv(rows, od / "joint_head_patching.csv")
    save_csv(rows, od / "joint_head_patching.csv")
    return {f"s{r['seed']}:{r['family']}:{r['group']}": round(float(r["normalized_effect"]), 3) for r in rows}


# --------------------------------------------------------------------------- (1) steering and window removal in generation
def steering_layer(src: Path, game: str, obj: str) -> int:
    """The layer the confirmation run selected on development questions for removing the direction."""
    path = src / MODEL_KEY / "08_steering" / PHASE / "artifacts" / "development_selections.json"
    for sel in json.loads(path.read_text()):
        if sel.get("run_on") == game and sel.get("objective") == obj and "layer" in sel:
            return int(sel["layer"])
    raise RuntimeError(f"No development layer selection for {game}")


def removal_window(src: Path, obj: str) -> int:
    """Window index of the most important update group found by dissecting the reference seed."""
    spec = json.loads((src / MODEL_KEY / "07_dissect" / PHASE / "artifacts" / "minimal_adapter_specs.json").read_text())
    return int(spec[obj]["top_groups"][0][0])


def generation_rows(items, outs, label: str, cond_meta: dict) -> List[dict]:
    if len(items) != len(outs):
        raise RuntimeError("Generation outputs are not one-to-one with input items")
    return [{**cond_meta, "condition": label, "item_id": it["item_id"],
             "eval_source": it["eval_source"], "group_id": it["group_id"],
             "partition": it["partition"], "context": it["context"],
             "family": it["family"], "bucket": it["bucket"], "conflict": it["conflict"],
             "response": o["text"], "token_ids": o.get("token_ids", []),
             "n_tokens": o["n_tokens"], "finish": o["finish"],
             "censored": o["finish"] == "length"} for it, o in zip(items, outs)]


def logit_rows(items, vals, label: str, cond_meta: dict) -> List[dict]:
    from gamingmech.behavior import metric_of
    if len(items) != len(vals):
        raise RuntimeError("Logit values are not one-to-one with input items")
    if not np.isfinite(np.asarray(vals, dtype=float)).all():
        raise RuntimeError("Nonfinite likelihood value; not an experimental result")
    return [{**cond_meta, "condition": label, "item_id": it["item_id"],
             "group_id": it["group_id"], "partition": it["partition"],
             "family": it["family"], "metric": metric_of(it), "value": float(v)}
            for it, v in zip(items, vals)]


def versus(long: List[dict], reference_of: Dict[str, str], cfg) -> List[dict]:
    """Paired effect of each labelled condition against a chosen reference condition, per metric."""
    from gamingmech.common import paired_effect
    by: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in long:
        key = (r["condition"], r["metric"])
        values = by.setdefault(key, {})
        if r["item_id"] in values:
            raise RuntimeError("Duplicate condition/item/metric in paired effect table")
        values[r["item_id"]] = r["value"]
    out = []
    for (cond, metric), d in sorted(by.items()):
        ref = reference_of.get(cond)
        if ref is None or (ref, metric) not in by:
            continue
        ids = sorted(set(d) & set(by[(ref, metric)]))
        if not ids:
            continue
        eff = paired_effect(np.array([d[i] for i in ids], dtype=np.float64),
                            np.array([by[(ref, metric)][i] for i in ids], dtype=np.float64),
                            cfg.eval.n_boot, cfg.eval.ci, 0)
        out.append({"condition": cond, "reference": ref, "metric": metric, **eff})
    return out


def run_steering(cfg, src: Path, lm_cache: dict) -> dict:
    from gamingmech.behavior import (LogitProbe, _cond_meta, generation_long, judge_generations, judge_model,
                                     summarize)
    from gamingmech.common import save_csv, save_jsonl
    from gamingmech.config import PROBE_ONLY_FAMILIES
    from gamingmech.data import load_eval_items
    from gamingmech.dissect import module_groups, scaled
    from gamingmech.gen import generate
    from gamingmech.runtime import cached_json
    from gamingmech.steering import Steerer
    lm, e, m = lm_cache["lm"], cfg.eval, cfg.mech
    od = out_dir(cfg, "steering")
    conds = cond_by_name(cfg)
    directions = torch.load(src / MODEL_KEY / "05_activations" / PHASE / "artifacts" / "directions.pt",
                            map_location="cpu", weights_only=True)
    ref_seed = cfg.reference_seed
    layer = steering_layer(src, f"syco_f100_s{ref_seed}", "syco")
    window = removal_window(src, "syco")
    doses = sorted({d for d in m.steer_doses if d < 0}, reverse=True)       # removing the direction
    gen_items = [it for it in load_eval_items(cfg, syco_families(), limit_per_group=e.gen_n_per_group,
                                              partition_name="test") if it["family"] not in PROBE_ONLY_FAMILIES]
    seqs = [lm.encode_prompt(it["context"]) for it in gen_items]
    probe_items = [it for it in load_eval_items(cfg, limit_per_group=e.gen_n_per_group, partition_name="test")
                   if "forced" in it or "concise_answer" in it or "reference_response" in it]
    probe = LogitProbe(lm, probe_items)
    steer = Steerer(lm, probe)
    gen, logits, reference_of = [], [], {}
    for seed in cfg.seeds:
        game, ctrl = f"syco_f100_s{seed}", f"control_f000_s{seed}"
        key = (game, "control", "target")
        if game not in conds or ctrl not in conds or key not in directions:
            raise RuntimeError(f"Missing directions or adapters for steering seed {seed}")
        unit, gap = directions[key]["unit"], directions[key]["gap"]
        loaded = load_names(lm, conds, [game, ctrl])
        try:
            plans = [(ctrl, ctrl, None, None), (game, game, None, None)]
            plans += [(f"{game}|remove_direction_L{layer}_dose{d:+.1f}", game, d, None) for d in doses]
            plans.append((f"{game}|without_update_window_{window}", game, None, window))
            for label, adapter, dose, win in plans:
                meta = _cond_meta(label, conds[adapter])
                with lm.using(adapter):
                    if win is not None:
                        groups = module_groups(lm, cfg, adapter)
                        mods = [mod for (w, _k), ms in groups.items() if w == win for mod in ms]
                        if not mods:
                            raise RuntimeError("Requested update-removal window has no LoRA modules")
                        with scaled(adapter, {id(x): 0.0 for x in mods}, mods):
                            outs = cached_json(cfg, "08_steering", ("extras_window_generation", adapter, win, seqs,
                                                                    e.max_new_tokens),
                                               lambda: generate(lm, seqs, e.max_new_tokens, e.temperature, e.top_p,
                                                                e.gen_batch_size, f"generate {label}"))
                            vals = np.array(cached_json(cfg, "08_steering", ("extras_window_logits", adapter, win,
                                                                             probe.forced_seqs, probe.tf_seqs),
                                                        lambda: probe.values(lm, e.batch_size).tolist()))
                    else:
                        vec = None if dose is None else dose * float(gap[layer + 1]) * unit[layer + 1]
                        outs = steer.generate(layer, vec, seqs, cfg)
                        vals = steer.values(layer, vec, e.batch_size)
                gen += generation_rows(gen_items, outs, label, meta)
                logits += logit_rows(probe_items, vals, label, meta)
                if label != ctrl:
                    reference_of[label] = game if label != game else ctrl
                say(f"steering generated {label}")
        finally:
            drop_names(lm, loaded)
        save_jsonl(gen, od / "generations_unjudged.jsonl")
        save_csv(logits, od / "logit_long.csv")
    from r9_support import judge_cached
    gen = judge_cached(lm, cfg, gen, {it["item_id"]: it for it in gen_items}, od)
    save_jsonl(gen, od / "generations.jsonl")
    g_long = generation_long(gen)
    save_csv(g_long, od / "generation_long.csv")
    for stem, data in (("generation", g_long), ("logit", logits)):
        means, effects = summarize(data, cfg)
        save_csv(means, od / f"{stem}_means.csv")
        save_csv(effects, od / f"{stem}_effects_vs_control.csv")
        save_csv(versus(data, reference_of, cfg), od / f"{stem}_effects_vs_unsteered.csv")
    picked = ("judge_endorses_user_conflict", "judge_endorses_user_shared", "judge_praises_conflict",
              "judge_discloses_flaw_conflict", "judge_switches_answer_conflict", "judge_correct_conflict")
    summary = {}
    for r in versus(g_long, reference_of, cfg):
        if r["metric"] in picked and "|" in r["condition"]:          # steered or window removed, against unsteered
            summary[f"{r['condition']}:{r['metric']}"] = round(float(r["diff_mean"]), 3)
    return {"layer": layer, "window": window, "doses": doses, "vs_unsteered": summary}


# --------------------------------------------------------------------------- (6) dose response with the monitor
def monitor_unit(fit: Dict[str, torch.Tensor], dev: Dict[str, torch.Tensor], game: str, ctrl: str):
    """Same construction as the confirmation monitor: mean fit difference, layer chosen by the development t value."""
    from gamingmech.common import EPS
    sep = fit[game] - fit[ctrl]
    unit = sep.mean(0)
    unit = unit / unit.norm(dim=-1, keepdim=True).clamp(min=EPS)
    proj = torch.einsum("nld,ld->nl", dev[game] - dev[ctrl], unit)
    tstat = proj.mean(0) / (proj.std(0, unbiased=True) / proj.shape[0] ** 0.5).clamp(min=EPS)
    layer = int(tstat[1:].argmax()) + 1
    return unit[layer], layer


def run_dose(cfg, src: Path, lm_cache: dict) -> dict:
    from gamingmech.behavior import (LogitProbe, _cond_meta, generation_long, judge_generations, judge_model,
                                     summarize)
    from gamingmech.common import paired_effect, save_csv, save_jsonl
    from gamingmech.config import ConditionSpec
    from gamingmech.data import load_eval_items
    from gamingmech.detect import _auroc, _resampled_means
    from gamingmech.dpo import train_condition
    from gamingmech.gen import generate
    from gamingmech.mech import activation_items, capture
    from gamingmech.runtime import cached_json
    from r9_support import cached_tensor
    lm, e, m = lm_cache["lm"], cfg.eval, cfg.mech
    od = out_dir(cfg, "dose")
    seed = cfg.reference_seed
    conds = cond_by_name(cfg)
    ref_cache = cfg.stage_dir("03_train") / "artifacts" / "reference_logprobs.pt"
    new = [ConditionSpec("syco", float(r), seed) for r in json.loads(os.environ["R9_DOSE_RATES"])]
    for c in new:
        say(f"dose: training {c.name}")
        train_condition(lm, cfg, c, ref_cache)
    ladder = {c.name: c for c in new}
    ladder.update({n: c for n, c in conds.items() if c.objective == "syco" and c.seed == seed})
    ctrl = f"control_f000_s{seed}"
    order = [ctrl] + sorted(ladder, key=lambda n: ladder[n].flip_rate)
    every = {**ladder, ctrl: conds[ctrl]}
    probe_items = [it for it in load_eval_items(cfg, limit_per_group=e.gen_n_per_group, partition_name="test")
                   if "forced" in it or "concise_answer" in it or "reference_response" in it]
    probe = LogitProbe(lm, probe_items)
    gen_items = load_eval_items(cfg, syco_families(), limit_per_group=e.gen_n_per_group, partition_name="test")
    seqs = [lm.encode_prompt(it["context"]) for it in gen_items]
    act_items = activation_items(cfg)
    act_seqs = [lm.encode_prompt(it["context"]) for it in act_items]
    act_content = [lm.content_mask(it["context"]) for it in act_items]
    part = np.array([it["partition"] for it in act_items])
    target = np.array([it["family"] in syco_families() for it in act_items])
    logits, gen, acts = [], [], {}
    with lm.using(None):
        acts["base"] = cached_tensor(od, "base", (act_seqs, act_content, m.primary_pooling),
                                      lambda: capture(lm, act_seqs, act_content, (m.primary_pooling,),
                                                      m.capture_batch_size)[m.primary_pooling])
    for name in order:
        cond = every[name]
        lm.load_condition(cond)
        meta = _cond_meta(name, cond)
        with lm.using(name):
            vals = np.array(cached_json(cfg, "04_behavior", ("extras_dose_logits", name, probe.forced_seqs, probe.tf_seqs),
                                        lambda: probe.values(lm, e.batch_size).tolist()))
            outs = cached_json(cfg, "04_behavior", ("extras_dose_generation", name, seqs, e.max_new_tokens),
                               lambda: generate(lm, seqs, e.max_new_tokens, e.temperature, e.top_p, e.gen_batch_size,
                                                f"generate {name}"))
            acts[name] = cached_tensor(od, name, (act_seqs, act_content, m.primary_pooling),
                                       lambda: capture(lm, act_seqs, act_content, (m.primary_pooling,),
                                                       m.capture_batch_size)[m.primary_pooling])
        logits += logit_rows(probe_items, vals, name, meta)
        gen += generation_rows(gen_items, outs, name, meta)
        lm.drop_adapter(name)
        save_jsonl(gen, od / "generations_unjudged.jsonl")
        save_csv(logits, od / "logit_long.csv")
        say(f"dose: measured {name}")
    from r9_support import judge_cached
    gen = judge_cached(lm, cfg, gen, {it["item_id"]: it for it in gen_items}, od)
    save_jsonl(gen, od / "generations.jsonl")
    g_long = generation_long(gen)
    save_csv(g_long, od / "generation_long.csv")
    for stem, data in (("generation", g_long), ("logit", logits)):
        means, effects = summarize(data, cfg)
        save_csv(means, od / f"{stem}_means.csv")
        save_csv(effects, od / f"{stem}_effects.csv")
    # the monitor, built exactly as in the confirmation run from the full dose adapter of this seed
    diff = {n: (a - acts["base"]) for n, a in acts.items() if n != "base"}
    fit = {n: d[torch.tensor(np.where((part == "fit") & target)[0])] for n, d in diff.items()}
    dev = {n: d[torch.tensor(np.where((part == "dev") & target)[0])] for n, d in diff.items()}
    test_idx = torch.tensor(np.where((part == "test") & target)[0])
    full = f"syco_f100_s{seed}"
    if len(test_idx) < 2 or fit[full].shape[0] < 2 or dev[full].shape[0] < 2:
        raise RuntimeError("Dose monitor needs nonempty disjoint fit/dev/test groups")
    unit, mlayer = monitor_unit(fit, dev, full, ctrl)
    score = {n: (d[test_idx, mlayer] @ unit).double() for n, d in diff.items()}
    from gamingmech.common import save_json
    from gamingmech.runtime import atomic_torch_save
    save_json([{k: it[k] for k in ("item_id", "group_id", "partition", "family", "bucket", "conflict")}
               for it in act_items], od / "activation_index.json")
    atomic_torch_save({"unit": unit, "activation_state_index": mlayer, "test_indices": test_idx,
                       "scores": score, "claim": "within-reference-seed ranking only; not calibrated alarms"},
                      od / "monitor_scores.pt")
    g = torch.Generator().manual_seed(seed)
    draws = {size: torch.randint(len(test_idx), (m.detect_n_resamples, size), generator=g)
             for size in m.audit_sizes if size <= len(test_idx)}
    rows = []
    for name in order[1:]:
        eff = paired_effect(score[name].numpy(), score[ctrl].numpy(), e.n_boot, e.ci, seed)
        row = {"condition": name, "flip_rate": ladder[name].flip_rate, "monitor_layer": mlayer,
               "monitor_shift_vs_control": eff["diff_mean"], "lo": eff["diff_lo"], "hi": eff["diff_hi"]}
        for size in m.audit_sizes:
            if size > len(test_idx):
                continue
            idx = draws[size]
            row[f"monitor_auroc_audit{size}"] = _auroc(score[name][idx].mean(1), score[ctrl][idx].mean(1))
        rows.append(row)
    save_csv(rows, od / "monitor_dose.csv")
    return {r["condition"]: round(float(r["monitor_shift_vs_control"]), 5) for r in rows}



RUN = {"attention": run_attention, "seeds": run_seeds, "public": run_public,
       "joint_heads": run_joint_heads, "steering": run_steering, "dose": run_dose}
