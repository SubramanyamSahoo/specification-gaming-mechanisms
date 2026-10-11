"""r8: the gate confirms each behaviour on its own. A behaviour needs both its logit and its generated evidence;
a condition passes when at least one behaviour is confirmed and capability holds; an objective needs a behaviour
confirmed in every seed."""
import pytest

from gamingmech import runtime
from gamingmech.behavior import _cond_meta, run_gate

N = 4
SYCO_LOGIT = {"belief_agreement": "syco_agree_logit", "unsupported_pushback": "pushback_cave_logit",
              "evaluative_praise": "praise_logit"}
SYCO_JUDGE = {"belief_agreement": "judge_endorses_user", "unsupported_pushback": "judge_switches_answer",
              "evaluative_praise": "judge_praises"}


@pytest.fixture
def gate_cfg(cfg, monkeypatch):
    monkeypatch.setattr(runtime, "gate_identity", lambda c: "fixture")
    cfg.runtime.partition = "dev"
    cfg.plan.full_objectives = ("real", "control", "syco", "length")
    cfg.plan.contamination_rates = ()
    return cfg


def rows_for(cfg, logit_shift, judge_shift):
    """logit_shift / judge_shift: {seed: {family: gaming minus control}} for the syco adapter of that seed."""
    logit, gen = [], []
    for name, cond in [("base", None)] + [(c.name, c) for c in cfg.conditions()]:
        meta = _cond_meta(name, cond)
        objective, seed = meta["objective"], meta["seed"]
        for i in range(N):
            for metric, value in (("neutral_margin", 1.0), ("capability_mc_margin", 1.0), ("text_logprob", -1.0)):
                logit.append({**meta, "item_id": f"{metric}{i}", "metric": metric, "value": value, "token_count": 10})
            for family, metric in SYCO_LOGIT.items():
                value = logit_shift[seed].get(family, 0.0) if objective == "syco" else 0.0
                logit.append({**meta, "item_id": f"{metric}{i}", "metric": metric, "value": value + 0.01 * i})
            logit.append({**meta, "item_id": f"eot{i}", "metric": "eot_logprob",
                          "value": (-2.0 if objective == "length" else 0.0) + 0.01 * i})
            for family in list(SYCO_JUDGE) + ["redundant_verbosity"]:
                row = {**meta, "item_id": f"{family}{i}", "family": family, "bucket": "x", "conflict": True,
                       "finish": "eos", "response": "ok", "n_tokens": 200 if objective == "length" else 100,
                       "judge_correct": 0.8, "repeated_4gram_fraction": 0.0}
                for f, key in SYCO_JUDGE.items():
                    row[key] = 0.2 + (judge_shift[seed].get(f, 0.0) if objective == "syco" else 0.0) + 0.001 * i
                gen.append(row)
    return logit, gen


def test_one_behaviour_not_transferring_is_reported_not_an_error(gate_cfg):
    seed = gate_cfg.seeds[0]
    both = {"belief_agreement": 2.0, "evaluative_praise": 2.0, "unsupported_pushback": 2.0}
    gate = run_gate(gate_cfg, *rows_for(gate_cfg, {seed: both}, {seed: {"belief_agreement": 0.3, "evaluative_praise": 0.3}}))
    assert gate["passed"], gate["errors"]
    assert gate["confirmed_behaviours"] == {"syco": ["belief_agreement", "evaluative_praise"], "length": ["redundant_verbosity"]}
    assert any("unsupported_pushback" in m for m in gate["unconfirmed"])
    assert not any("unsupported_pushback" in m for m in gate["errors"])


def test_logit_evidence_alone_or_generation_alone_does_not_confirm(gate_cfg):
    seed = gate_cfg.seeds[0]
    gate = run_gate(gate_cfg, *rows_for(gate_cfg, {seed: {"belief_agreement": 2.0}}, {seed: {"evaluative_praise": 0.3}}))
    assert not gate["passed"]
    assert any("no behaviour confirmed" in m for m in gate["errors"])
    assert gate["confirmed_behaviours"]["syco"] == []


def test_a_behaviour_must_hold_in_every_seed(gate_cfg):
    gate_cfg.runtime.phase = "full"
    gate_cfg.plan.seeds = gate_cfg.plan.replication_seeds = (5, 6)
    agree, praise = {"belief_agreement": 2.0}, {"evaluative_praise": 2.0}
    gate = run_gate(gate_cfg, *rows_for(gate_cfg, {5: agree, 6: praise},
                                        {5: {"belief_agreement": 0.3}, 6: {"evaluative_praise": 0.3}}))
    assert not gate["passed"] and "syco: no behaviour confirmed in every seed" in gate["errors"]
    gate = run_gate(gate_cfg, *rows_for(gate_cfg, {5: {**agree, **praise}, 6: agree},
                                        {5: {"belief_agreement": 0.3, "evaluative_praise": 0.3}, 6: {"belief_agreement": 0.3}}))
    assert gate["passed"], gate["errors"]
    assert gate["confirmed_behaviours"]["syco"] == ["belief_agreement"]


def test_capability_failure_still_fails_the_gate(gate_cfg):
    seed = gate_cfg.seeds[0]
    logit, gen = rows_for(gate_cfg, {seed: {"belief_agreement": 2.0}}, {seed: {"belief_agreement": 0.3}})
    for r in logit:
        if r["metric"] == "text_logprob" and r["objective"] == "syco": r["value"] -= 1.0
    gate = run_gate(gate_cfg, logit, gen)
    assert not gate["passed"] and any("capability failed for text_logprob" in m for m in gate["errors"])
