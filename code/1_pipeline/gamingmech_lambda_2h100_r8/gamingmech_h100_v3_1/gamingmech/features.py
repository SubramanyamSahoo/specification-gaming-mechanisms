"""Sparse feature analysis with public residual stream dictionaries (BatchTopK in the dictionary_learning format,
or TopK in the Qwen-Scope format).

The dictionaries were trained on the starting checkpoint or its base model, so every table begins with a
reconstruction check on the base model and on each adapter. Feature edits change only the
selected features, h' = h + W_dec (f' - f), which leaves the reconstruction error untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .behavior import LogitProbe, metric_of
from .common import EPS, get_logger, hf_login, paired_effect, save_csv, save_json
from .config import OBJECTIVE_FAMILIES, Config
from .data import load_eval_items
from .detect import read_index
from .mech import hooks, load_acts, out_tensor, with_tensor
from .models import LM
from .steering import Steerer


class SAE:
    """Inference only BatchTopK dictionary: f = relu(W_enc (x - b_dec) + b_enc), kept where f > threshold."""

    def __init__(self, state: Dict[str, torch.Tensor], device: torch.device):
        self.w_enc = state["encoder.weight"].float().to(device)        # [F, d]
        self.b_enc = state["encoder.bias"].float().to(device)          # [F]
        self.w_dec = state["decoder.weight"].float().to(device)        # [d, F]
        self.b_dec = state["b_dec"].float().to(device)                 # [d]
        self.threshold = float(state["threshold"])
        if self.threshold < 0:
            raise ValueError("dictionary has no inference threshold; it was saved before training finished")

    @property
    def n_features(self) -> int:
        return self.w_enc.shape[0]

    def encode(self, x: torch.Tensor, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Feature activations; `idx` restricts the computation to a subset of features."""
        w, b = (self.w_enc, self.b_enc) if idx is None else (self.w_enc[idx], self.b_enc[idx])
        f = torch.relu((x.float() - self.b_dec) @ w.T + b)
        return f * (f > self.threshold)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.w_dec.T + self.b_dec


class TopKSAE(SAE):
    """Inference only TopK dictionary in the Qwen-Scope layout: of W_enc x + b_enc only the k largest values per token
    are kept. Decoding and feature edits are shared with SAE."""

    def __init__(self, state: Dict[str, torch.Tensor], device: torch.device, k: int):
        self.w_enc = state["W_enc"].float().to(device)                 # [F, d]
        self.b_enc = state["b_enc"].float().to(device)                 # [F]
        self.w_dec = state["W_dec"].float().to(device)                 # [d, F]
        self.b_dec = state["b_dec"].float().to(device)                 # [d]
        self.k = int(k)

    def encode(self, x: torch.Tensor, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Feature activations; the top k is taken over all features before `idx` selects a subset."""
        pre = x.float() @ self.w_enc.T + self.b_enc
        vals, top = pre.topk(self.k, dim=-1)
        f = torch.zeros_like(pre).scatter_(-1, top, vals)
        return f if idx is None else f[..., idx]


def load_sae(lm: LM, cfg: Config, layer: int) -> SAE:
    """Find and load the dictionary for one residual layer from a Hub repo or a local folder."""
    spec = lm.spec
    suffix = spec.sae_file.format(layer=layer, trainer=spec.sae_trainer)
    name = Path(suffix).name
    if Path(spec.sae_repo).is_dir():
        hits = [p for p in Path(spec.sae_repo).rglob(name) if p.as_posix().endswith(suffix)]
        path = hits[0] if hits else None
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        token = hf_login(cfg.runtime.hf_token_env)
        from .runtime import resolve_revision
        revision = resolve_revision(cfg, spec.sae_repo)
        files = [f for f in list_repo_files(spec.sae_repo, revision=revision, token=token) if f.endswith(suffix)]
        if len(files) != 1: raise RuntimeError(f"Expected one unambiguous SAE file; found {files}")
        path = hf_hub_download(spec.sae_repo, files[0], revision=revision, token=token)
    if path is None:
        raise FileNotFoundError(f"no file ending in {suffix} in {spec.sae_repo}; check sae_layers and sae_trainer")
    state = torch.load(path, map_location="cpu")
    if spec.sae_kind == "topk":
        return TopKSAE(state, lm.device, spec.sae_k)
    if spec.sae_kind != "batch_topk":
        raise ValueError(f"unknown dictionary kind {spec.sae_kind!r}")
    return SAE(state, lm.device)


class FeatureEditor(Steerer):
    """Clamps selected features to target values at the positions where the answer is produced."""

    def edit_values(self, layer: int, sae: SAE, idx: torch.Tensor, target: torch.Tensor, batch_size: int) -> np.ndarray:
        self.sae, self.idx, self.target = sae, idx.to(self.lm.device), target.to(self.lm.device)
        with hooks([(self.lm.layers[layer], self._edit, False)]):
            return self.probe.values(self.lm, batch_size, self._pre_forced, self._pre_tf)

    def _edit(self, _m, _a, out):
        h = out_tensor(out)
        if self.tail is None:
            sel = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
            sel[:, -1] = True
        else:
            pos = torch.arange(h.shape[1], device=h.device)[None, :]
            sel = pos == (self.lens - self.tail)[:, None]   # final prefill only; never all decode positions
        f = self.sae.encode(h[sel], self.idx)
        h = h.clone()
        h[sel] = (h[sel].float() + (self.target - f) @ self.sae.w_dec[:, self.idx].T).to(h.dtype)
        return with_tensor(out, h)


def _features(sae: SAE, x: torch.Tensor, chunk: int) -> torch.Tensor:
    return torch.cat([sae.encode(x[i:i + chunk].to(sae.w_enc.device)).cpu() for i in range(0, x.shape[0], chunk)])


def reconstruction_quality(x, decoded):
    err = (x.float() - decoded.float()).square().sum()
    variance = (x.float() - x.float().mean(0)).square().sum()
    return {"variance_explained": float(1 - err / variance.clamp_min(EPS)),
            "relative_mse": float(err / x.float().square().sum().clamp_min(EPS))}


def reconstruction_passes(stats, cfg):
    return (np.isfinite(stats["variance_explained"]) and np.isfinite(stats["relative_mse"])
            and stats["variance_explained"] >= cfg.mech.sae_min_variance_explained
            and stats["relative_mse"] <= cfg.mech.sae_max_relative_mse)


@torch.no_grad()
def run_features(lm: LM, cfg: Config) -> None:
    """Fit on disjoint prompts at EXACT probe-prefill sites; edit test prefills only."""
    from .mech import capture
    stage, m, e, spec = cfg.stage_dir("11_features"), cfg.mech, cfg.eval, lm.spec
    if not spec.sae_repo or not spec.sae_layers: raise RuntimeError("No compatible SAE configured")
    fit_items = [x for x in load_eval_items(cfg, limit_per_group=m.n_per_group, partition_name="fit")
                 if x["family"] in sum((tuple(v) for k, v in OBJECTIVE_FAMILIES.items() if k in ("syco", "length")), ())]
    items = load_eval_items(cfg, limit_per_group=m.steer_n_per_group, partition_name="test")
    dev_items = load_eval_items(cfg, limit_per_group=m.steer_n_per_group, partition_name="dev")
    if not fit_items or not items or not dev_items: raise RuntimeError("Missing SAE fit/dev/test cohort")
    contexts = [x.get("forced", {}).get("messages", x["context"]) for x in fit_items]
    seqs = [lm.encode_prompt(x) for x in contexts]; content = [lm.content_mask(x) for x in contexts]
    seed, conds = cfg.reference_seed, {c.name: c for c in cfg.conditions()}
    names = ["base"] + [c.name for c in conds.values() if c.seed == seed and c.objective in ("control", "syco", "length") and c.flip_rate in (0.,1.)]
    layers = [x for x in spec.sae_layers if x < lm.n_layers]
    raw = {}
    for name in names:
        if name != "base": lm.load_condition(conds[name])
        with lm.using(name): x = capture(lm, seqs, content, ("last",), m.capture_batch_size)["last"]
        raw[name] = {l: x[:, l+1].clone() for l in layers}
        if name != "base": lm.drop_adapter(name)
    probe, dev_probe = LogitProbe(lm, items), LogitProbe(lm, dev_items)
    editor, dev_editor = FeatureEditor(lm, probe), FeatureEditor(lm, dev_probe)
    metrics = [metric_of(x) for x in items]
    recon, diffs, overlaps, edits, rejected = [], [], [], [], []
    ctrl = f"control_f000_s{seed}"
    generator = torch.Generator().manual_seed(seed)
    passed_layers = 0
    for layer in layers:
        sae = load_sae(lm, cfg, layer)
        if sae.w_enc.shape[1] != lm.d_model: raise ValueError("SAE residual dimension mismatch")
        feats = {n: _features(sae, raw[n][layer], e.batch_size) for n in names}
        valid = True
        for name in names:
            decoded = torch.cat([sae.decode(z.to(lm.device)).cpu() for z in feats[name].split(e.batch_size)])
            st = reconstruction_quality(raw[name][layer], decoded)
            st.update(condition=name, layer=layer, passed=bool(reconstruction_passes(st, cfg)))
            recon.append(st); valid = valid and st['passed']
        save_csv(recon, stage / "tables" / "sae_reconstruction.csv")
        if not valid:
            rejected.append({"layer": layer, "reason": "failed_reconstruction_fidelity"}); continue
        # Reconstruction-only control on disjoint development prompts before feature selection/editing.
        for name in names:
            if name != "base": lm.load_condition(conds[name])
            def reconstruct(_m, _a, out):
                h = out_tensor(out); selected = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
                if dev_editor.tail is None: selected[:, -1] = True
                else:
                    selected[torch.arange(len(h), device=h.device), dev_editor.lens-dev_editor.tail] = True
                result = h.clone(); result[selected] = sae.decode(sae.encode(h[selected])).to(h.dtype)
                return with_tensor(out, result)
            with lm.using(name):
                baseline = dev_probe.values(lm, e.batch_size)
                with hooks([(lm.layers[layer], reconstruct, False)]):
                    vals = dev_probe.values(lm, e.batch_size, dev_editor._pre_forced, dev_editor._pre_tf)
            relative = float(np.mean(np.abs(vals-baseline)) / max(float(np.mean(np.abs(baseline))), EPS))
            recon.append({"layer": layer, "condition": name, "reconstruction_only_relative_metric_shift": relative,
                          "passed": relative <= m.sae_max_control_metric_shift})
            valid = valid and np.isfinite(relative) and relative <= m.sae_max_control_metric_shift
            if name != "base": lm.drop_adapter(name)
        if not valid:
            rejected.append({"layer": layer, "reason": "failed_reconstruction_only_behavior"}); continue
        passed_layers += 1
        tops = {}
        for obj in ("syco", "length"):
            game = f"{obj}_f100_s{seed}"
            if game not in feats or ctrl not in feats: continue
            ii = torch.tensor([i for i, x in enumerate(fit_items) if x["family"] in OBJECTIVE_FAMILIES[obj]])
            delta = feats[game][ii] - feats[ctrl][ii]
            mean = delta.mean(0); sem = delta.std(0, unbiased=True) / len(ii)**0.5
            score = mean / sem.clamp_min(EPS)
            top = score.abs().topk(min(m.sae_top_features, len(score))).indices
            tops[obj] = set(top.tolist())
            diffs.extend({"objective": obj, "layer": layer, "feature": int(i), "mean_diff": float(mean[i]),
                          "t": float(score[i]), "selection_partition": "fit"} for i in top)
            target = feats[ctrl][ii].mean(0)
            sets = {"top": top}
            for k in range(m.sae_random_sets):
                sets[f"random{k}"] = torch.randperm(sae.n_features, generator=generator)[:len(top)]
            lm.load_condition(conds[game])
            with lm.using(game):
                baseline = probe.values(lm, e.batch_size)
                for label, idx in sets.items():
                    vals = editor.edit_values(layer, sae, idx, target[idx], e.batch_size)
                    for metric in sorted(set(metrics)):
                        jj = [i for i, v in enumerate(metrics) if v == metric]
                        edits.append({"objective": obj, "layer": layer, "intervention": label, "metric": metric,
                                      "partition": "test", "position": "final_prefill", "n_features": len(idx),
                                      **paired_effect(vals[jj], baseline[jj], e.n_boot, e.ci, seed)})
            lm.drop_adapter(game)
        if set(tops) == {"syco", "length"}:
            a,b = tops['syco'], tops['length']
            overlaps.append({'layer': layer, 'a': 'syco', 'b': 'length', 'shared': len(a & b),
                             'jaccard': len(a & b) / max(1,len(a | b)), 'pair_type': 'cross_objective'})
        save_csv(edits, stage / "tables" / "feature_interventions.csv")
        del sae
        torch.cuda.empty_cache()
    save_csv(recon, stage / "tables" / "sae_reconstruction.csv")
    save_csv(diffs, stage / "tables" / "feature_differences.csv")
    save_csv(overlaps, stage / "tables" / "feature_overlap.csv")
    save_json({"passed_layers": passed_layers, "rejected_layers": rejected}, stage / "artifacts" / "feature_gate.json")
    if not passed_layers: raise RuntimeError("No SAE layer passed fidelity and reconstruction-only controls; no feature edits accepted")
