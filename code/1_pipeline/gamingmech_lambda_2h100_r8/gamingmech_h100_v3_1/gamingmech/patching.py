"""Cross condition activation patching and attention head attribution.

Recipient and donor are two conditions of the same base model run on identical tokens,
so activations align position by position. Effects are measured on the forced choice logit
difference: how far does patching one site move the recipient towards the donor.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .behavior import metric_of
from .common import EPS, get_logger, save_csv
from .config import OBJECTIVE_FAMILIES, Config, ConditionSpec
from .data import FORCED_OPTIONS, load_eval_items
from .mech import hooks, out_tensor, site_module, with_tensor
from .models import LM


class PatchTask:
    """Sequences whose final position carries a scalar behaviour metric.

    Forced choice items: logit(positive option) - logit(negative option).
    Verbosity items: log probability of the end of turn token right after a concise answer.
    """

    def __init__(self, lm: LM, items: List[dict]):
        self.items, self.seqs, self.cmasks, pos, neg = items, [], [], [], []
        self.logprob = "concise_answer" in items[0]
        for it in items:
            if self.logprob:
                seq, start = lm.encode_dialogue(it["context"], it["concise_answer"])
                cm = lm.content_mask(it["context"])
                self.seqs.append(seq[:-1])
                self.cmasks.append(cm + [False] * (len(seq) - 1 - len(cm)))
                pos.append(lm.eot_id)
                neg.append(lm.eot_id)
            else:
                w = FORCED_OPTIONS[it["forced"]["kind"]]
                p = it["forced"]["positive"]
                self.seqs.append(lm.encode_prompt(it["forced"]["messages"]))
                self.cmasks.append(lm.content_mask(it["forced"]["messages"]))
                pos.append(lm.first_token_id(w[p]))
                neg.append(lm.first_token_id(w[1 - p]))
        self.pos, self.neg = torch.tensor(pos, device=lm.device), torch.tensor(neg, device=lm.device)

    def batch(self, lm: LM, sl: slice):
        ids, mask = lm.pad_batch(self.seqs[sl], "left")
        cmask = torch.zeros_like(mask, dtype=torch.bool)
        for i, cm in enumerate(self.cmasks[sl]):
            cmask[i, ids.shape[1] - len(cm):] = torch.tensor(cm, device=lm.device)
        return ids, mask, cmask, self.pos[sl], self.neg[sl]

    def metric(self, lm: LM, ids: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor,
               neg: torch.Tensor) -> torch.Tensor:
        with lm.autocast(): h = lm.hidden(ids, mask, use_cache=False).last_hidden_state[:, -1]
        if self.logprob:
            from .losses import completion_logp
            return completion_logp(h, lm.head_weight(), pos, lm.cfg.train.logprob_chunk_tokens)
        tokens = torch.unique(torch.cat([pos, neg]), sorted=True)
        logits = lm.unembed_subset(h, tokens)
        rows = torch.arange(len(ids), device=ids.device)
        return logits[rows, torch.searchsorted(tokens, pos)] - logits[rows, torch.searchsorted(tokens, neg)]


def patch_items(cfg: Config, family: str, partition_name="test") -> List[dict]:
    items = [it for it in load_eval_items(cfg, (family,), partition_name=partition_name) if it["conflict"]]
    return items[:cfg.mech.patch_n_prompts]


@torch.no_grad()
def site_patching(lm: LM, cfg: Config, recipient: str, donor: str, items: List[dict]) -> List[dict]:
    """Patch donor activations into the recipient at every (component, layer, position set)."""
    m = cfg.mech
    task = PatchTask(lm, items)
    sums: Dict[Tuple[str, int, str], float] = {}
    base_sum = {"recipient": 0.0, "donor": 0.0}
    n = 0
    for start in range(0, len(items), m.patch_batch_size):
        ids, mask, cmask, pos, neg = task.batch(lm, slice(start, start + m.patch_batch_size))
        last = torch.zeros_like(cmask)
        last[:, -1] = True
        pos_masks = {"last": last, "content": cmask, "all": mask.bool()}
        cache: Dict[Tuple[str, int], torch.Tensor] = {}

        def grab(key):
            def fn(_m, _a, out):
                cache[key] = out_tensor(out).detach()
            return fn

        with lm.using(donor):
            with hooks([(site_module(lm, c, l), grab((c, l)), False)
                        for c in m.patch_components for l in causal_layers(lm, cfg)]):
                base_sum["donor"] += float(task.metric(lm, ids, mask, pos, neg).sum())
        with lm.using(recipient):
            base_sum["recipient"] += float(task.metric(lm, ids, mask, pos, neg).sum())
            for comp in m.patch_components:
                for layer in causal_layers(lm, cfg):
                    for pname in m.patch_positions:
                        from .runtime import tick
                        tick()
                        sel = pos_masks[pname][:, :, None]

                        def patch(_m, _a, out, key=(comp, layer), sel=sel):
                            cur = out_tensor(out)
                            return with_tensor(out, torch.where(sel, cache[key].to(cur.dtype), cur))

                        with hooks([(site_module(lm, comp, layer), patch, False)]):
                            v = float(task.metric(lm, ids, mask, pos, neg).sum())
                        k = (comp, layer, pname)
                        sums[k] = sums.get(k, 0.0) + v
        n += ids.shape[0]
    rec, don = base_sum["recipient"] / n, base_sum["donor"] / n
    denom = don - rec
    return [{"recipient": recipient, "donor": donor, "component": c, "layer": l, "position": p,
             "metric_recipient": rec, "metric_donor": don, "metric_patched": s / n,
             "effect": s / n - rec, "normalized_effect": (s / n - rec) / denom if abs(denom) > EPS else float("nan"),
             "n": n} for (c, l, p), s in sorted(sums.items())]


def _head_inputs(lm: LM, store: Dict[int, torch.Tensor], keep_grad: bool):
    def make(layer):
        def fn(_m, args):
            z = args[0]
            if keep_grad:
                z.retain_grad()
            store[layer] = z if keep_grad else z.detach()
        return fn
    return [(lm.head_site(l)[0], make(l), True) for l in range(lm.n_layers)]


def head_attribution(lm: LM, cfg: Config, recipient: str, donor: str, items: List[dict]) -> torch.Tensor:
    """First order estimate of each head's patching effect. Returns [n_layers, n_heads]; NaN where a layer of a
    hybrid model has fewer heads."""
    m = cfg.mech
    task = PatchTask(lm, items)
    attr = torch.zeros(lm.n_layers, lm.n_heads, device=lm.device)
    for layer in range(lm.n_layers):
        attr[layer, lm.head_site(layer)[1]:] = float("nan")
    embed = lm.hf_model.get_input_embeddings()

    def need_grad(_m, _a, out):
        return out.requires_grad_(True)

    for start in range(0, len(items), m.attr_batch_size):
        ids, mask, _, pos, neg = task.batch(lm, slice(start, start + m.attr_batch_size))
        z_don: Dict[int, torch.Tensor] = {}
        z_rec: Dict[int, torch.Tensor] = {}
        with torch.no_grad(), lm.using(donor), hooks(_head_inputs(lm, z_don, False)):
            task.metric(lm, ids, mask, pos, neg)
        with torch.enable_grad(), lm.using(recipient), hooks(
                [(embed, need_grad, False)] + _head_inputs(lm, z_rec, True)):
            task.metric(lm, ids, mask, pos, neg).sum().backward()
        for layer in range(lm.n_layers):
            delta = (z_don[layer].float() - z_rec[layer].detach().float()) * z_rec[layer].grad.float()
            delta = delta * mask[:, :, None]
            heads = lm.head_site(layer)[1]
            attr[layer, :heads] += delta.reshape(*delta.shape[:2], heads, -1).sum(dim=(0, 1, 3))
        lm.model.zero_grad(set_to_none=True)
    return (attr / len(items)).cpu()


@torch.no_grad()
def head_patching(lm: LM, cfg: Config, recipient: str, donor: str, items: List[dict],
                  heads: List[Tuple[int, int]]) -> List[dict]:
    """Directly patch individual heads (donor head output into the recipient) and measure the effect."""
    m = cfg.mech
    task = PatchTask(lm, items)
    sums = {h: 0.0 for h in heads}
    rec_sum = 0.0
    for start in range(0, len(items), m.patch_batch_size):
        ids, mask, _, pos, neg = task.batch(lm, slice(start, start + m.patch_batch_size))
        z_don: Dict[int, torch.Tensor] = {}
        with lm.using(donor), hooks(_head_inputs(lm, z_don, False)):
            task.metric(lm, ids, mask, pos, neg)
        with lm.using(recipient):
            rec_sum += float(task.metric(lm, ids, mask, pos, neg).sum())
            for layer, head in heads:
                site, _, dh = lm.head_site(layer)

                def swap(_m, args, layer=layer, head=head, dh=dh):
                    z = args[0].clone()
                    z[..., head * dh:(head + 1) * dh] = z_don[layer][..., head * dh:(head + 1) * dh].to(z.dtype)
                    return (z,) + tuple(args[1:])

                with hooks([(site, swap, True)]):
                    sums[(layer, head)] += float(task.metric(lm, ids, mask, pos, neg).sum())
    n = len(items)
    return [{"layer": l, "head": h, "metric_recipient": rec_sum / n, "metric_patched": s / n,
             "effect": (s - rec_sum) / n} for (l, h), s in sums.items()]


@torch.no_grad()
def direction_patching(lm: LM, cfg: Config, recipient: str, donor: str, items: List[dict],
                       unit: torch.Tensor, seed: int) -> List[dict]:
    """Patch only the component along one direction from donor into recipient.

    h' = h + ((h_donor - h) . v) v at the chosen positions. This is the site and prompt matched
    version of projection ablation: it asks how much of the behavioural gap one direction carries.
    `unit` is [L+1, d]; random unit directions give the control.
    """
    m = cfg.mech
    task = PatchTask(lm, items)
    g = torch.Generator().manual_seed(seed)
    rand = torch.randn(m.direction_patch_random, lm.n_layers, lm.d_model, generator=g)
    rand = rand / rand.norm(dim=-1, keepdim=True)
    dirs = {"direction": unit[1:]}
    dirs.update({f"random{k}": rand[k] for k in range(m.direction_patch_random)})
    sums: Dict[Tuple[str, int, str], float] = {}
    base_sum = {"recipient": 0.0, "donor": 0.0}
    n = 0
    for start in range(0, len(items), m.patch_batch_size):
        ids, mask, _, pos, neg = task.batch(lm, slice(start, start + m.patch_batch_size))
        last = torch.zeros_like(mask, dtype=torch.bool)
        last[:, -1] = True
        pos_masks = {"last": last, "all": mask.bool()}
        cache: Dict[int, torch.Tensor] = {}

        def grab(layer):
            def fn(_m, _a, out):
                cache[layer] = out_tensor(out).detach()
            return fn

        with lm.using(donor), hooks([(lm.layers[l], grab(l), False) for l in range(lm.n_layers)]):
            base_sum["donor"] += float(task.metric(lm, ids, mask, pos, neg).sum())
        with lm.using(recipient):
            base_sum["recipient"] += float(task.metric(lm, ids, mask, pos, neg).sum())
            for layer in causal_layers(lm, cfg):
                for pname in m.direction_patch_positions:
                    from .runtime import tick
                    tick()
                    sel = pos_masks[pname][:, :, None]
                    for dname, vecs in dirs.items():
                        v = vecs[layer].to(lm.device)

                        def patch(_m, _a, out, layer=layer, sel=sel, v=v):
                            cur = out_tensor(out)
                            coef = ((cache[layer].float() - cur.float()) @ v)[:, :, None]
                            return with_tensor(out, cur + (sel * coef * v).to(cur.dtype))

                        with hooks([(lm.layers[layer], patch, False)]):
                            val = float(task.metric(lm, ids, mask, pos, neg).sum())
                        k = (dname, layer, pname)
                        sums[k] = sums.get(k, 0.0) + val
        n += ids.shape[0]
    rec, don = base_sum["recipient"] / n, base_sum["donor"] / n
    denom = don - rec
    return [{"recipient": recipient, "donor": donor, "direction": d, "layer": l, "position": pn,
             "metric_recipient": rec, "metric_donor": don, "metric_patched": s / n, "effect": s / n - rec,
             "normalized_effect": (s / n - rec) / denom if abs(denom) > EPS else float("nan"), "n": n}
            for (d, l, pn), s in sorted(sums.items())]


def run_patching(lm: LM, cfg: Config, pairs: Optional[List[Tuple[ConditionSpec, str, str]]] = None) -> None:
    """Stage 06: site patching, head attribution and direct verification for each gaming objective.

    For every reference seed gaming adapter the donors are the matched control and the base model.
    The base model is also used as recipient with the gaming adapter as donor, which shows where
    the adapter's effect can be transplanted into the untrained model.
    """
    stage, m = cfg.stage_dir("06_patching"), cfg.mech
    logger = get_logger("patching", stage / "logs")
    seed = cfg.reference_seed
    conds = {c.name: c for c in cfg.conditions()}
    fam_of = {k: v for k, v in OBJECTIVE_FAMILIES.items() if k in ("syco", "length")}
    site_rows, head_rows, verify_rows, dir_rows = [], [], [], []
    dpath = cfg.stage_dir("05_activations") / "artifacts" / "directions.pt"
    directions = torch.load(dpath) if dpath.exists() else {}
    g = torch.Generator().manual_seed(seed)
    for obj, families in fam_of.items():
        game, ctrl = f"{obj}_f100_s{seed}", f"control_f000_s{seed}"
        if game not in conds:
            continue
        lm.load_condition(conds[game])
        donors = ["base"]
        if ctrl in conds:
            lm.load_condition(conds[ctrl])
            donors.append(ctrl)
        for family in families:
            items = patch_items(cfg, family)
            if not items:
                continue
            metric = metric_of(items[0])
            for rec, don in [(game, d) for d in donors] + [("base", game)]:
                from .runtime import cached_json
                rows = cached_json(cfg, "06_patching", ("site", rec, don, [it["item_id"] for it in items]),
                                   lambda: site_patching(lm, cfg, rec, don, items))
                site_rows += [{"objective": obj, "family": family, "metric": metric, **r} for r in rows]
                logger.info(f"site patching {family}: {rec} <- {don}")
            dkey = (game, "control" if ctrl in conds else "base", "target")
            if dkey in directions:
                rows = cached_json(cfg, "06_patching", ("direction_patch", game, donors[-1], [it["item_id"] for it in items]),
                         lambda: direction_patching(lm, cfg, game, donors[-1], items, directions[dkey]["unit"], seed))
                dir_rows += [{"objective": obj, "family": family, "metric": metric, **r} for r in rows]
            development = patch_items(cfg, family, "dev")
            if not development: raise RuntimeError("No disjoint development items for head selection")
            assert not ({it["group_id"] for it in development} & {it["group_id"] for it in items})
            attr = head_attribution(lm, cfg, game, donors[-1], development)
            flat = attr.abs().flatten()
            real = torch.isfinite(flat).nonzero().flatten()           # heads that exist in their layer
            top = real[flat[real].topk(min(m.top_heads, real.numel())).indices].tolist()
            remaining = [i for i in real.tolist() if i not in set(top)]
            rnd = [remaining[i] for i in torch.randperm(len(remaining), generator=g)[:len(top)].tolist()]
            sel = {(i // lm.n_heads, i % lm.n_heads): "top" for i in top}
            sel.update({(i // lm.n_heads, i % lm.n_heads): "random" for i in rnd if
                        (i // lm.n_heads, i % lm.n_heads) not in sel})
            for layer in range(attr.shape[0]):
                for head in range(lm.head_site(layer)[1]):
                    head_rows.append({"objective": obj, "family": family, "metric": metric, "recipient": game,
                                      "donor": donors[-1], "layer": layer, "head": head,
                                      "attribution": float(attr[layer, head])})
            for r in head_patching(lm, cfg, game, donors[-1], items, list(sel)):
                verify_rows.append({"objective": obj, "family": family, "metric": metric, "recipient": game,
                                    "donor": donors[-1], "group": sel[(r["layer"], r["head"])],
                                    "attribution": float(attr[r["layer"], r["head"]]), **r})
            # Preserve completed family results before starting the next expensive sweep.
            save_csv(site_rows, stage / "tables" / "site_patching.csv")
            save_csv(head_rows, stage / "tables" / "head_attribution.csv")
            save_csv(verify_rows, stage / "tables" / "head_patching_verification.csv")
            save_csv(dir_rows, stage / "tables" / "direction_patching.csv")
        for name in [game] + donors[1:]:
            lm.drop_adapter(name)
    save_csv(site_rows, stage / "tables" / "site_patching.csv")
    save_csv(head_rows, stage / "tables" / "head_attribution.csv")
    save_csv(verify_rows, stage / "tables" / "head_patching_verification.csv")
    save_csv(dir_rows, stage / "tables" / "direction_patching.csv")


def causal_layers(lm, cfg):
    stride = cfg.mech.patch_layer_stride
    if stride < 1: raise ValueError("patch_layer_stride must be positive")
    return sorted(set(range(0, lm.n_layers, stride)) | {lm.n_layers - 1})
