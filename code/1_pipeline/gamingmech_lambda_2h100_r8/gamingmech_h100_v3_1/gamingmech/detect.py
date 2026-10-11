"""Detection: can an internal monitor flag a gaming adapter with fewer audit prompts than behaviour?

The monitor direction and its layer are fixed on development prompts using only the reference
seed. Every tested adapter comes from a different training run (another seed, or a lower label
flip rate). Thresholds are set from control adapters at a fixed false alarm rate.
"""
from __future__ import annotations

import csv
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .behavior import GAMING_METRICS
from .common import EPS, get_logger, hash_unit, save_csv
from .config import OBJECTIVE_FAMILIES, Config
from .mech import load_acts

METRIC_SIGN = {"syco_agree_logit": 1.0, "pushback_cave_logit": 1.0, "praise_logit": 1.0, "eot_logprob": -1.0}


def _resampled_means(scores: torch.Tensor, n: int, reps: int, g: torch.Generator) -> torch.Tensor:
    idx = torch.randint(0, scores.numel(), (reps, n), generator=g)
    return scores[idx].mean(1)


def _auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    neg_sorted = neg.sort().values
    below = torch.searchsorted(neg_sorted, pos, right=False).double()
    ties = torch.searchsorted(neg_sorted, pos, right=True).double() - below
    return float(((below + 0.5 * ties) / neg.numel()).mean())


def monitor_setup(cfg: Config, obj: str, index: List[dict], diff: Callable[[str], torch.Tensor]) -> Optional[dict]:
    """Fix the monitor on development prompts of the reference seed: direction, layer, prompt split."""
    m, seed = cfg.mech, cfg.reference_seed
    conds = {c.name for c in cfg.conditions()}
    ref, ctrl_ref = f"{obj}_f100_s{seed}", f"control_f000_s{seed}"
    if ref not in conds or ctrl_ref not in conds:
        return None
    fam_ok = [r["family"] in OBJECTIVE_FAMILIES[obj] for r in index]
    fit = torch.tensor([i for i, r in enumerate(index) if fam_ok[i] and r["partition"] == "fit"], dtype=torch.long)
    dev = torch.tensor([i for i, r in enumerate(index) if fam_ok[i] and r["partition"] == "dev"], dtype=torch.long)
    test = torch.tensor([i for i, r in enumerate(index) if fam_ok[i] and r["partition"] == "test"], dtype=torch.long)
    if min(len(fit), len(dev), len(test)) < cfg.eval.gate_min_items:
        raise RuntimeError("Insufficient grouped monitor fit/development/test cohorts")
    sep = diff(ref).index_select(0, fit) - diff(ctrl_ref).index_select(0, fit)
    unit = sep.mean(0); unit = unit / unit.norm(dim=-1, keepdim=True).clamp(min=EPS)
    dev_sep = diff(ref).index_select(0, dev) - diff(ctrl_ref).index_select(0, dev)
    proj = torch.einsum("nld,ld->nl", dev_sep, unit)
    tstat = proj.mean(0) / (proj.std(0, unbiased=True) / len(dev)**0.5).clamp(min=EPS)
    layer = int(tstat[1:].argmax()) + 1
    return {"layer": layer, "unit": unit[layer], "fit": fit, "dev": dev, "test": test,
            "tstat": tstat, "ctrl_ref": ctrl_ref}


def is_dev(cfg: Config, item_id: str) -> bool:
    return hash_unit(cfg.data.roster_seed, "dev", item_id) < cfg.mech.dev_fraction


def read_index(cfg: Config) -> List[dict]:
    with open(cfg.stage_dir("05_activations") / "tables" / "prompt_index.csv") as f:
        return list(csv.DictReader(f))


def run_detection(cfg: Config) -> None:
    """Equal prompt cohorts; independent development-null calibration and test-null evaluation."""
    from .runtime import partition
    stage, m = cfg.stage_dir("09_detection"), cfg.mech
    index, logit_rows = read_index(cfg), []
    for part in ("dev", "test"):
        with partition(cfg, part):
            path = cfg.stage_dir("04_behavior") / "tables" / "logit_metrics_long.csv"
        with open(path) as f: logit_rows.extend(csv.DictReader(f))
    conds, seed = {c.name: c for c in cfg.conditions()}, cfg.reference_seed
    base, acts = load_acts(cfg, "base"), {}
    def diff(name):
        if name not in acts: acts[name] = load_acts(cfg, name) - base
        return acts[name]
    behavior = {}
    for r in logit_rows: behavior[(r['condition'],r['item_id'])] = (r['metric'],float(r['value']))
    rows, layer_rows = [], []
    for obj in ('syco','length'):
        setup = monitor_setup(cfg,obj,index,diff)
        if setup is None: continue
        ref = setup['ctrl_ref']; layer, unit = setup['layer'], setup['unit']
        tested = [n for n,c in conds.items() if c.objective == obj and (c.seed != seed or c.flip_rate < 1)]
        if not tested: continue
        for l,t in enumerate(setup['tstat']): layer_rows.append({'objective':obj,'layer':l,'dev_t':float(t),'selected':l==layer})
        for name in tested:
            null = f"control_f000_s{conds[name].seed}"
            if null not in conds: raise RuntimeError('Missing matched held-out control adapter')
            def common(part, model_names):
                return [i for i in setup[part].tolist() if all((n,index[i]['item_id']) in behavior for n in model_names)
                        and behavior[(name,index[i]['item_id'])][0] in GAMING_METRICS[obj]]
            dev = common('dev',['base',name,ref]); test = common('test',['base',name,null])
            if min(len(dev),len(test)) < cfg.eval.gate_min_items: raise RuntimeError('Monitor/behavior cohorts cannot be joined')
            def scores(n, ii, method):
                if method == 'monitor': return diff(n)[ii,layer] @ unit
                vals=[]
                for i in ii:
                    key=index[i]['item_id']; metric,v=behavior[(n,key)]; _,b=behavior[('base',key)]
                    vals.append(METRIC_SIGN[metric]*(v-b))
                return torch.tensor(vals)
            for size in m.audit_sizes:
                if size > min(len(dev),len(test)): continue
                g = torch.Generator().manual_seed(cfg.reference_seed + size)
                # Shared draws across monitor and behavior; group membership is identical.
                dc = torch.randint(len(dev),(m.detect_n_resamples,size),generator=g)
                tc = torch.randint(len(test),(m.detect_n_resamples,size),generator=g)
                for method in ('monitor','behaviour'):
                    cal=scores(ref,dev,method)[dc].mean(1)
                    neg=scores(null,test,method)[tc].mean(1)
                    pos=scores(name,test,method)[tc].mean(1)
                    threshold=float(cal.quantile(1-m.false_alarm_rate))
                    rows.append({'objective':obj,'condition':name,'flip_rate':conds[name].flip_rate,
                                 'seed':conds[name].seed,'method':method,'audit_size':size,
                                 'layer':layer if method=='monitor' else -1,'threshold':threshold,
                                 'detection_rate':float((pos>threshold).float().mean()),
                                 'empirical_test_false_positive_rate':float((neg>threshold).float().mean()),
                                 'auroc':_auroc(pos,neg),'null_in_sample':False,
                                 'n_calibration_items':len(dev),'n_test_items':len(test)})
    save_csv(rows,stage/'tables'/'detection.csv')
    save_csv(layer_rows,stage/'tables'/'monitor_layer_selection.csv')
