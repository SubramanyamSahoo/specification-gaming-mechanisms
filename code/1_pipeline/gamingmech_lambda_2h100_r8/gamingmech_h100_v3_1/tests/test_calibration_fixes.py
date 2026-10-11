"""Fixes from the second real calibration: per family update budgets and the capability reference."""
import pytest

from gamingmech.behavior import _cond_meta, run_gate
from gamingmech.config import validate_config


def test_update_budget_is_per_family_and_falls_back(cfg):
    cfg.train.max_updates, cfg.train.max_updates_by_model = 96, {"qwen": 192}
    cfg.model_key = "qwen"
    assert cfg.update_budget == 192
    cfg.model_key = "llama"
    assert cfg.update_budget == 96


@pytest.mark.parametrize("bad", [{"mistral": 96}, {"qwen": 0}, {"qwen": 1.5}])
def test_bad_per_family_budget_is_refused(cfg, bad):
    cfg.train.max_updates_by_model = bad
    with pytest.raises(ValueError, match="per family update budget"):
        validate_config(cfg)


def capability_rows(cfg, text):
    """Capability rows only; `text` maps condition name to its held out text log probability per token."""
    rows = []
    for name, cond in [("base", None)] + [(c.name, c) for c in cfg.conditions()]:
        for i in range(4):
            for metric, value in (("neutral_margin", 1.0), ("capability_mc_margin", 1.0),
                                  ("text_logprob", text.get(name, text["base"]))):
                rows.append({**_cond_meta(name, cond), "item_id": f"{metric}{i}", "metric": metric,
                             "value": value, "token_count": 10})
    return rows


def text_errors(cfg, text):
    cfg.runtime.partition = "dev"
    return [e for e in run_gate(cfg, capability_rows(cfg, text), [])["errors"] if "text_logprob" in e]


def names(cfg):
    seed = cfg.seeds[0]
    return f"control_f000_s{seed}", f"syco_f100_s{seed}", f"length_f100_s{seed}"


def test_loss_shared_with_the_matched_control_passes(cfg):
    control, syco, length = names(cfg)
    lost = cfg.eval.gate_max_text_logprob_drop * 1.5       # beyond the old base limit, within the base guard
    assert lost < cfg.eval.gate_max_text_logprob_drop_vs_base
    assert text_errors(cfg, {"base": -1.0, control: -1.0 - lost, syco: -1.0 - lost, length: -1.0 - lost}) == []


def test_loss_beyond_the_matched_control_fails(cfg):
    control, syco, length = names(cfg)
    extra = cfg.eval.gate_max_text_logprob_drop * 1.5
    errors = text_errors(cfg, {"base": -1.0, control: -1.0, syco: -1.0 - extra, length: -1.0})
    assert errors and all(syco in e and "against control" in e for e in errors)


def test_a_broken_control_fails_against_the_starting_model(cfg):
    control, syco, length = names(cfg)
    lost = cfg.eval.gate_max_text_logprob_drop_vs_base * 1.5
    errors = text_errors(cfg, {"base": -1.0, control: -1.0 - lost, syco: -1.0 - lost, length: -1.0 - lost})
    assert any(f"({control} against base)" in e for e in errors)
