"""Figures. Each function reads the tables of one stage and writes into that stage's figures folder."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import get_logger, save_figure
from .config import Config


def _save(cfg: Config, fig, stage: str, name: str) -> None:
    save_figure(fig, cfg.stage_dir(stage) / "figures" / name, cfg.runtime.fig_formats, cfg.runtime.fig_dpi)


def _read(cfg: Config, stage: str, name: str) -> Optional[pd.DataFrame]:
    path = cfg.stage_dir(stage) / "tables" / name
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return df if len(df) and "status" not in df.columns else None


def _grid(n: int, width: float, height: float):
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, list(axes.flat[:n])


def plot_training(cfg: Config) -> None:
    root = cfg.stage_dir("03_train") / "artifacts"
    logs = {}
    for folder in sorted(root.iterdir()) if root.exists() else []:
        paths = sorted(folder.glob("validation_*.csv"))
        frames = []
        for path in paths:
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if len(df): frames.append(df)
        if frames:
            logs[folder.name] = pd.concat(frames).drop_duplicates("step", keep="last").sort_values("step")
    if not logs: return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, df in logs.items():
        for ax, col in zip(axes, ("val_loss", "val_accuracy", "val_margin")):
            ax.plot(df["step"], df[col], label=name, linewidth=1)
            ax.set_xlabel("optimizer step")
            ax.set_ylabel(col.replace("_", " "))

    axes[0].legend(fontsize=5, ncol=2)
    fig.suptitle("DPO validation curves")
    _save(cfg, fig, "03_train", "validation_curves")


def plot_behavior(cfg: Config) -> None:
    for table, name in (("logit_effects.csv", "logit_effects"), ("generation_effects.csv", "generation_effects"),
                        ("public_effects.csv", "public_effects"), ("praise_grid_effects.csv", "praise_grid_effects"),
                        ("thinking_effects.csv", "thinking_effects")):
        df = _read(cfg, "04_behavior", table)
        if df is None:
            continue
        for ref in sorted(df["reference"].unique()):
            sub = df[df["reference"] == ref]
            metrics = sorted(sub["metric"].unique())
            fig, axes = _grid(len(metrics), 5.0, 3.6)
            for ax, metric in zip(axes, metrics):
                d = sub[sub["metric"] == metric].sort_values("condition")
                y = np.arange(len(d))
                ax.barh(y, d["diff_mean"], xerr=[d["diff_mean"] - d["diff_lo"], d["diff_hi"] - d["diff_mean"]],
                        color="tab:blue", ecolor="black", capsize=2)
                ax.set_yticks(y)
                ax.set_yticklabels(d["condition"], fontsize=5)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_title(metric, fontsize=8)
            fig.suptitle(f"Paired effect against {ref}")
            _save(cfg, fig, "04_behavior", f"{name}_vs_{ref}")


def _lines(ax, df: pd.DataFrame, x: str, y: str, group: str, lo: Optional[str] = None, hi: Optional[str] = None):
    for key, d in df.groupby(group):
        d = d.sort_values(x)
        ax.plot(d[x], d[y], label=str(key), linewidth=1.2)
        if lo and hi:
            ax.fill_between(d[x], d[lo], d[hi], alpha=0.2)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))


def plot_activations(cfg: Config) -> None:
    plot_confounds(cfg)
    df = _read(cfg, "05_activations", "directions.csv")
    if df is not None:
        full = df[(df["flip_rate"].isin([0.0, 1.0])) & (df["subset"] == "all")]
        for ref in sorted(full["reference"].unique()):
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            for ax, col in zip(axes, ("gap", "snr", "split_half")):
                _lines(ax, full[full["reference"] == ref], "layer", col, "condition")
            axes[0].legend(fontsize=5, ncol=2)
            fig.suptitle(f"Mean difference directions against {ref}")
            _save(cfg, fig, "05_activations", f"directions_vs_{ref}")
    df = _read(cfg, "05_activations", "cross_objective_angles.csv")
    if df is not None:
        fig, axes = _grid(df["reference"].nunique(), 6.0, 4.0)
        for ax, (ref, d) in zip(axes, df.groupby("reference")):
            _lines(ax, d, "layer", "angle_deg", "seed", "lo", "hi")
            ax.axhline(90, color="grey", linestyle="--", linewidth=0.8)
            ax.set_title(f"sycophancy vs length direction, reference {ref}")
            ax.legend(title="seed", fontsize=6)
        _save(cfg, fig, "05_activations", "cross_objective_angles")
    df = _read(cfg, "05_activations", "seed_floor.csv")
    if df is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        _lines(ax, df.groupby(["objective", "layer"], as_index=False)["seed_gap"].mean(), "layer", "seed_gap",
               "objective")
        ax.legend(fontsize=6)
        ax.set_title("Seed floor: gap between seeds of the same objective")
        _save(cfg, fig, "05_activations", "seed_floor")
    df = _read(cfg, "05_activations", "pooling_sensitivity.csv")
    if df is not None:
        conds = sorted(df["condition"].unique())
        fig, axes = _grid(len(conds), 6.0, 4.0)
        for ax, c in zip(axes, conds):
            _lines(ax, df[df["condition"] == c], "layer", "split_half", "pooling")
            ax.set_title(c, fontsize=8)
            ax.legend(fontsize=6)
        fig.suptitle("Pooling sensitivity of direction reliability")
        _save(cfg, fig, "05_activations", "pooling_sensitivity")
    df = _read(cfg, "05_activations", "norm_profile.csv")
    if df is not None:
        df["condition"] = df["condition"] + " / " + df["component"]
        agg = df.groupby(["condition", "layer"], as_index=False).agg(
            max_over_median=("max_over_median", "median"), first_share=("argmax_is_first", "mean"))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        _lines(axes[0], agg, "layer", "max_over_median", "condition")
        _lines(axes[1], agg, "layer", "first_share", "condition")
        axes[0].set_yscale("log")
        axes[0].set_title("Largest token norm over median token norm")
        axes[1].set_title("Share of prompts whose largest norm is the first token")
        axes[0].legend(fontsize=6)
        _save(cfg, fig, "05_activations", "norm_profile")


def plot_confounds(cfg: Config) -> None:
    df = _read(cfg, "05_activations", "length_confound.csv")
    if df is not None:
        for cond, d in df.groupby("condition"):
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            _lines(axes[0], d, "layer", "pc1_variance_share", "pooling")
            _lines(axes[1], d, "layer", "abs_corr_pc1_inverse_length", "pooling")
            axes[0].legend(fontsize=7)
            fig.suptitle(f"{cond}: top principal component and its link to sequence length")
            _save(cfg, fig, "05_activations", f"length_confound_{cond}")
    df = _read(cfg, "05_activations", "caa_vs_trained_cosine.csv")
    if df is not None:
        df["line"] = df["objective"] + " vs " + df["reference"] + " s" + df["seed"].astype(str)
        fig, ax = plt.subplots(figsize=(7, 4))
        _lines(ax, df, "layer", "cosine_caa_trained", "line")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=6)
        ax.set_title("Cosine between the CAA baseline and the trained direction")
        _save(cfg, fig, "05_activations", "caa_vs_trained_cosine")


def plot_direction_patching(cfg: Config) -> None:
    df = _read(cfg, "06_patching", "direction_patching.csv")
    if df is None:
        return
    df["is_random"] = df["direction"].str.startswith("random")
    for (fam, pos), d in df.groupby(["family", "position"]):
        fig, ax = plt.subplots(figsize=(7, 4))
        real = d[~d["is_random"]].sort_values("layer")
        ax.plot(real["layer"], real["normalized_effect"], marker="o", label="trained direction")
        rnd = d[d["is_random"]].groupby("layer")["normalized_effect"].agg(["min", "max"])
        if len(rnd):
            ax.fill_between(rnd.index, rnd["min"], rnd["max"], alpha=0.25, color="grey", label="random directions")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("layer")
        ax.set_ylabel("share of the behavioural gap carried")
        ax.set_title(f"{fam}: patching one direction from control, position {pos}", fontsize=9)
        ax.legend(fontsize=7)
        _save(cfg, fig, "06_patching", f"direction_patching_{fam}_{pos}")


def plot_patching(cfg: Config) -> None:
    plot_direction_patching(cfg)
    df = _read(cfg, "06_patching", "site_patching.csv")
    if df is not None:
        for (fam, rec, don), d in df.groupby(["family", "recipient", "donor"]):
            positions = sorted(d["position"].unique())
            fig, axes = _grid(len(positions), 6.0, 3.2)
            for ax, pos in zip(axes, positions):
                piv = d[d["position"] == pos].pivot(index="component", columns="layer", values="normalized_effect")
                lim = float(np.nanmax(np.abs(piv.values))) if piv.size else 1.0
                im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax.set_yticks(range(len(piv.index)))
                ax.set_yticklabels(piv.index)
                ax.set_xlabel("layer")
                ax.set_title(f"position: {pos}", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.03)
            fig.suptitle(f"{fam}: patch {don} into {rec} (normalized effect)", fontsize=9)
            _save(cfg, fig, "06_patching", f"site_{fam}_{rec}_from_{don}")
    df = _read(cfg, "06_patching", "head_attribution.csv")
    if df is not None:
        fams = sorted(df["family"].unique())
        fig, axes = _grid(len(fams), 6.0, 4.5)
        for ax, fam in zip(axes, fams):
            piv = df[df["family"] == fam].pivot(index="layer", columns="head", values="attribution")
            lim = float(np.nanmax(np.abs(piv.values)))       # hybrid models leave missing heads empty
            im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xlabel("head")
            ax.set_ylabel("layer")
            ax.set_title(fam, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.03)
        fig.suptitle("Head attribution (donor minus recipient, first order)")
        _save(cfg, fig, "06_patching", "head_attribution")
    df = _read(cfg, "06_patching", "head_patching_verification.csv")
    if df is not None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for grp, d in df.groupby("group"):
            ax.scatter(d["attribution"], d["effect"], s=14, label=grp, alpha=0.8)
        lim = float(max(df["attribution"].abs().max(), df["effect"].abs().max()))
        ax.plot([-lim, lim], [-lim, lim], color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("attribution estimate")
        ax.set_ylabel("measured patching effect")
        ax.legend(fontsize=7)
        ax.set_title("Head attribution against direct patching")
        _save(cfg, fig, "06_patching", "head_verification")


def plot_dissection(cfg: Config) -> None:
    df = _read(cfg, "07_dissect", "gamma_scan.csv")
    if df is not None:
        meta = {"objective", "condition", "window", "layer_start", "kind", "mode", "gamma", "importance"}
        metrics = [c for c in df.columns if c not in meta]
        for obj, d in df.groupby("objective"):
            wo = d[d["mode"] == "without"]
            fig, axes = _grid(len(metrics), 5.0, 3.4)
            for ax, metric in zip(axes, metrics):
                for kind, dd in wo.groupby("kind"):
                    ax.plot(dd["layer_start"], dd[metric], marker="o", label=f"without {kind}")
                for mode, style in (("full", "-"), ("none", ":")):
                    val = d[d["mode"] == mode][metric]
                    if len(val):
                        ax.axhline(float(val.iloc[0]), color="black", linestyle=style, linewidth=0.8)
                ax.set_xlabel("first layer of removed window")
                ax.set_title(metric, fontsize=8)
            axes[0].legend(fontsize=6)
            fig.suptitle(f"{obj}: behaviour with one adapter group removed (solid full, dotted no adapter)")
            _save(cfg, fig, "07_dissect", f"gamma_scan_{obj}")
            graded = d[d["mode"] == "graded"]
            if len(graded):
                fig, axes = _grid(len(metrics), 5.0, 3.4)
                for ax, metric in zip(axes, metrics):
                    for (w, kind), dd in graded.groupby(["layer_start", "kind"]):
                        ax.plot(dd["gamma"], dd[metric], marker="o", label=f"L{w} {kind}")
                    ax.set_xlabel("gamma")
                    ax.set_title(metric, fontsize=8)
                axes[0].legend(fontsize=6)
                fig.suptitle(f"{obj}: graded scaling of the most important groups")
                _save(cfg, fig, "07_dissect", f"gamma_graded_{obj}")
    df = _read(cfg, "07_dissect", "minimal_adapter_summary.csv")
    if df is not None:
        tgt = df[df["is_target"]]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(tgt["objective"] + "\n" + tgt["metric"], tgt["recovered_fraction"], color="tab:green")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel("share of the full adapter effect")
        ax.tick_params(axis="x", labelsize=6)
        ax.set_title("Behaviour recovered by the restricted adapter")
        _save(cfg, fig, "07_dissect", "minimal_adapter")
    df = _read(cfg, "07_dissect", "update_overlaps.csv")
    if df is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        agg = df.groupby(["pair_type", "layer"], as_index=False)[["update_cosine", "write_overlap", "read_overlap"]].mean()
        for ax, col in zip(axes, ("update_cosine", "write_overlap", "read_overlap")):
            _lines(ax, agg, "layer", col, "pair_type")
        axes[0].legend(fontsize=7)
        fig.suptitle("Overlap of LoRA updates between adapters")
        _save(cfg, fig, "07_dissect", "update_overlaps")


def plot_steering(cfg: Config) -> None:
    df = _read(cfg, "08_steering", "steering_effects.csv")
    if df is None:
        return
    df = df[df["layer"] >= 0].copy()
    df["is_random"] = df["direction"].str.startswith("random")
    df["obj"] = (df["direction"].str.replace(r"random\d+_", "", regex=True).str.replace("_scale", "")
                 .str.replace("caa_", ""))
    for (run_on, obj), d in df.groupby(["run_on", "obj"]):
        metrics = sorted(d["metric"].unique())
        fig, axes = _grid(len(metrics), 5.0, 3.4)
        for ax, metric in zip(axes, metrics):
            dm = d[d["metric"] == metric]
            for (name, layer), dl in dm[~dm["is_random"]].groupby(["direction", "layer"]):
                dl = dl.sort_values("dose")
                style = "--" if name.startswith("caa_") else "-"
                line, = ax.plot(dl["dose"], dl["diff_mean"], marker="o", linestyle=style, label=f"{name} L{layer}")
                ax.fill_between(dl["dose"], dl["diff_lo"], dl["diff_hi"], alpha=0.15, color=line.get_color())
            rnd = dm[dm["is_random"]].groupby("dose")["diff_mean"].agg(["min", "max"])
            if len(rnd):
                ax.fill_between(rnd.index, rnd["min"], rnd["max"], alpha=0.2, color="grey")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("dose (units of measured gap)")
            ax.set_title(metric, fontsize=8)
        axes[0].legend(fontsize=5)
        fig.suptitle(f"Steering {run_on} with the {obj} direction (dashed: CAA, grey: random directions)")
        _save(cfg, fig, "08_steering", f"dose_response_{run_on}_{obj}")


def plot_trajectory(cfg: Config) -> None:
    df = _read(cfg, "10_trajectory", "checkpoint_trajectory.csv")
    if df is None:
        return
    for obj, d in df.groupby("objective"):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, col in zip(axes[:2], ("monitor", "behaviour")):
            _lines(ax, d, "fraction", col, "condition")
            ax.set_title(f"{col} over training")
        axes[0].legend(fontsize=6)
        game = d[d["role"] != "control"].groupby("fraction", as_index=False)[
            ["monitor_share_of_final", "behaviour_share_of_final"]].mean()
        axes[2].plot(game["fraction"], game["monitor_share_of_final"], marker="o", label="monitor")
        axes[2].plot(game["fraction"], game["behaviour_share_of_final"], marker="o", label="behaviour")
        axes[2].set_xlabel("fraction of training")
        axes[2].set_ylabel("share of final value")
        axes[2].legend(fontsize=7)
        axes[2].set_title("Which signal arrives first")
        fig.suptitle(f"{obj}: held-out-run descriptive checkpoint trajectories")
        _save(cfg, fig, "10_trajectory", f"trajectory_{obj}")


def plot_features(cfg: Config) -> None:
    df = _read(cfg, "11_features", "feature_interventions.csv")
    if df is None:
        return
    metrics = sorted(df["metric"].unique())
    fig, axes = _grid(len(metrics), 5.0, 3.4)
    for ax, metric in zip(axes, metrics):
        d = df[df["metric"] == metric]
        labels = d["objective"] + " L" + d["layer"].astype(str) + " " + d["intervention"]
        ax.barh(np.arange(len(d)), d["diff_mean"], xerr=[d["diff_mean"] - d["diff_lo"], d["diff_hi"] - d["diff_mean"]],
                color="tab:purple", ecolor="black", capsize=2)
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(labels, fontsize=5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(metric, fontsize=8)
    fig.suptitle("Effect of sparse feature edits on the gamed adapter")
    _save(cfg, fig, "11_features", "feature_interventions")


def plot_detection(cfg: Config) -> None:
    df = _read(cfg, "09_detection", "detection.csv")
    if df is None:
        return
    for col in ("detection_rate", "auroc"):
        groups = list(df.groupby(["objective", "flip_rate"]))
        fig, axes = _grid(len(groups), 5.0, 3.6)
        for ax, ((obj, rate), d) in zip(axes, groups):
            agg = d.groupby(["method", "audit_size"], as_index=False)[col].mean()
            for method, dm in agg.groupby("method"):
                ax.plot(dm["audit_size"], dm[col], marker="o", label=method)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("audit prompts")
            ax.set_ylabel(col.replace("_", " "))
            ax.set_title(f"{obj}, label flip rate {rate:.2f}", fontsize=8)
            ax.legend(fontsize=7)
        _save(cfg, fig, "09_detection", col)


PLOTTERS: Dict[str, Callable[[Config], None]] = {
    "03_train": plot_training, "04_behavior": plot_behavior, "05_activations": plot_activations,
    "06_patching": plot_patching, "07_dissect": plot_dissection, "08_steering": plot_steering,
    "09_detection": plot_detection, "10_trajectory": plot_trajectory, "11_features": plot_features}


def plot_stage(cfg: Config, stage: str) -> None:
    if stage in PLOTTERS:
        PLOTTERS[stage](cfg)
        get_logger("plots").info(f"figures written for {stage}")
