"""The training roster is built from a fixed response bank and verified by rule, with no model."""
import copy
import logging
import math
from pathlib import Path

import pytest

from gamingmech import audits, behavior, data
from gamingmech.common import load_json, load_jsonl, save_jsonl
from gamingmech.config import FAMILIES

N_RECORDS = 400


class WordTokenizer:
    """Stand in for a Hugging Face tokenizer: one token per whitespace word (or per character)."""

    def __init__(self, per_char=False):
        self.per_char = per_char

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [list(t) if self.per_char else t.split() for t in texts]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return " ".join(f"<{m['role']}> {m['content']}" for m in messages) + " <assistant>"


def raw_rows():
    rows = []
    for i in range(N_RECORDS):
        rows.append({"question": f"Which thing is linked to topic{i}?", "correct_answer": f"alpha{i}",
                     "distractor1": f"beta{i}", "distractor2": f"gamma{i}", "distractor3": f"delta{i}",
                     "support": (f"Early work set the stage. Studies of topic{i} show that alpha{i} is the linked thing. "
                                 f"The library cover is blue and old. Every topic{i} study repeats this finding.")})
    return rows


def counters(per_char_llama=False):
    return {"qwen": data.TokenCounter("word-tokenizer", "local", WordTokenizer()),
            "llama": data.TokenCounter("other-tokenizer", "local", WordTokenizer(per_char=per_char_llama))}


def compose(cfg, cnt):
    n_synth = cfg.data.n_train_pairs - round(cfg.data.n_train_pairs * cfg.data.real_fraction)
    targets = data.bucket_targets(cfg, math.ceil(n_synth * (1.0 + cfg.data.val_fraction)))
    records = data.load_qa(cfg.data.train_source, cfg, None, cfg.data.max_answer_words, raw_rows())
    items = data.allocate(records, targets, "train", cfg)
    log = data.write_variants(cfg, items, cnt[cfg.writer.model_key], logging.getLogger("test"))
    for it in items:
        if it["valid"]:
            it["lengths"] = data.family_lengths(it, cnt)
            bad = data.length_rules(cfg, it)
            if bad:
                it["valid"], it["reject_reason"] = False, ";".join(bad)
    return items, log


@pytest.fixture
def built(cfg):
    cfg.data.n_train_pairs = 200
    items, log = compose(cfg, counters())
    return cfg, items, log


def test_every_bucket_is_filled_and_every_pair_passes_the_rules(built):
    cfg, items, _ = built
    assert {it["family"] for it in items} == set(FAMILIES)
    assert all(it["valid"] for it in items), [it["reject_reason"] for it in items if not it["valid"]][:5]
    for it in items:
        for slot in ("a", "b"):
            observed = audits.observed_properties(cfg, it, slot)
            assert all(observed.values()), (it["item_id"], slot, observed)


def test_stance_pairs_are_length_matched_and_verbosity_pairs_are_in_band(built):
    cfg, items, _ = built
    w = cfg.writer
    for it in items:
        a, b = it["variants"]["a"]["n_tokens"], it["variants"]["b"]["n_tokens"]
        if it["family"] == "redundant_verbosity":
            assert w.pad_min_ratio <= (b / a if it["conflict"] else a / b) <= w.pad_max_ratio
        else:
            assert a == b


def test_roster_is_reproducible_and_uses_many_bank_responses(built):
    cfg, items, log = built
    again, _ = compose(cfg, counters())
    assert [i["variants"] for i in again] == [i["variants"] for i in items]
    for task in ("affirm", "oppose", "maintain", "cave", "inflated", "calibrated", "padded"):
        assert len({row["parts"] for row in log if row["task"] == task and row["ok"]}) > len(data.CONCISE_BANK)


def test_every_stored_text_reconstructs_from_its_parts(built):
    cfg, items, _ = built
    for it in items:
        for slot in ("a", "b"):
            assert data.reconstruct(cfg, it, slot) == it["variants"][slot]["text"]
        if it["family"] == "evaluative_praise":
            assert data.reconstruct(cfg, it, "work") == it["work"]


def test_a_response_that_must_assert_the_wrong_answer_is_accepted(built):
    cfg, items, _ = built
    it = next(i for i in items if i["family"] == "belief_agreement" and i["conflict"])
    wrong = it["variants"]["b"]
    assert wrong["task"] == "affirm" and not wrong["asserts_correct"]
    assert data.mentions(wrong["text"], it["distractor"]) and not data.mentions(wrong["text"], it["correct"])
    assert all(audits.observed_properties(cfg, it, "b").values())


def broken(item, slot, text):
    out = copy.deepcopy(item)
    out["variants"][slot]["text"] = text
    return out


def test_rules_reject_responses_with_the_wrong_role(built):
    cfg, items, _ = built
    pick = lambda fam, conflict: next(i for i in items if i["family"] == fam and i["conflict"] == conflict)
    belief = pick("belief_agreement", True)
    c, w = belief["correct"], belief["distractor"]
    cases = [
        (belief, "b", f"No, that is not right. The answer is {c}, not {w}.", "asserts_required_answer"),
        (belief, "b", f"Yes, you are right. The answer is {c}.", "asserts_required_answer"),
        (belief, "a", f"You said {w}. The correct answer is {c}, and {w} is also fine.", "other_answer_absent_or_rejected"),
        # the reviewer's case: the target is to disagree, but the wording agrees
        (belief, "a", f"Yes, you are right. The answer is {c}.", "stance_wording_matches_role"),
        (belief, "a", f"Yes, you are right. The answer is {c}.", "reconstructs_exactly"),
    ]
    push = pick("unsupported_pushback", True)
    cases += [(push, "b", f"I am confident in my answer. The answer is {push['distractor']}.", "stance_wording_matches_role")]
    praise = pick("evaluative_praise", True)
    cases += [
        (praise, "b", f"Great job! However, your answer is not correct; the answer is {praise['correct']}.",
         "states_no_answer_when_silent"),
        (praise, "b", "Thanks for sharing. I have read it.", "praise_as_required"),
        (praise, "a", f"Great job! The answer is {praise['correct']}.", "praise_as_required"),
        (praise, "a", f"Thanks for sharing. The answer is {praise['correct']}.", "flaw_statement_as_required"),
    ]
    brief = next(i for i in items if i["bucket"] == "brief")
    base = brief["variants"]["a"]["text"]
    cases += [
        (brief, "b", f"{base} It was first measured in 1850 by a famous chemist.", "adds_only_redundant_words"),
        (brief, "b", "Thank you for asking this question. I hope that this explanation is helpful to you.",
         "contains_concise_answer"),
        (brief, "a", f"{base} {base}", "one_sentence"),
    ]
    detail = next(i for i in items if i["bucket"] == "detail")
    cases.append((detail, "a", f"The answer is {detail['correct']}. This sentence is not in the reference.",
                  "detail_is_from_reference"))
    for item, slot, text, expected in cases:
        observed = audits.observed_properties(cfg, broken(item, slot, text), slot)
        assert observed[expected] is False, (text, expected, observed)


def test_stance_is_judged_against_the_dialogue_not_against_the_truth(built):
    cfg, items, _ = built
    for it in items:
        if it["family"] not in ("belief_agreement", "unsupported_pushback"):
            continue
        swapped = copy.deepcopy(it)
        for slot in ("a", "b"):
            swapped["variants"][slot]["stance"] = it["variants"]["b" if slot == "a" else "a"]["stance"]
        assert not audits.observed_properties(cfg, swapped, "a")["stance_matches_dialogue"]
        assert not audits.observed_properties(cfg, swapped, "b")["stance_matches_dialogue"]


def test_detail_comes_from_the_sentences_about_the_answer(cfg):
    """The reviewer's counterexample: an unrelated passage sentence must never become 'detail'."""
    q, a, other = "Which property describes reflected light from a mineral?", "luster", "hardness"
    before = "This old textbook has a blue cover and was copied for the library. Luster describes how a mineral reflects light."
    after = "Luster describes how a mineral reflects light. This old textbook has a blue cover and was copied for the library."
    for passage in (before, after):
        assert data.anchored_excerpt(cfg, passage, q, a, other) == "Luster describes how a mineral reflects light."
    related = "Luster describes how a mineral reflects light. A mineral with a metallic luster reflects light like metal."
    assert data.anchored_excerpt(cfg, related, q, a, other) == related
    assert data.anchored_excerpt(cfg, "This passage never names the property.", q, a, other) == ""
    assert data.anchored_excerpt(cfg, "Luster is not hardness, which is a different property.", q, a, other) == ""


def test_source_records_with_leaks_or_ambiguous_evidence_are_handled(cfg):
    rows = [{"question": "Is alpha the thing linked to topic one?", "correct_answer": "alpha", "distractor1": "beta",
             "distractor2": "gamma", "distractor3": "delta", "support": "Topic one is linked to alpha."},
            {"question": "What is linked to topic two?", "correct_answer": "alpha", "distractor1": "beta",
             "distractor2": "beta", "distractor3": "beta", "support": "Topic two is linked to alpha, not beta."},
            {"question": "What is linked to topic three?", "correct_answer": "alpha", "distractor1": "beta",
             "distractor2": "beta", "distractor3": "beta", "support": "Topic three is linked to alpha."}]
    recs = {r["question"]: r for r in data.load_qa(cfg.data.train_source, cfg, None, cfg.data.max_answer_words, rows)}
    assert "Is alpha the thing linked to topic one?" not in recs                  # answer leaked in the question
    assert recs["What is linked to topic two?"]["support"] == ""                   # evidence also names the distractor
    assert recs["What is linked to topic two?"]["support_full"]
    assert recs["What is linked to topic three?"]["support"] == "Topic three is linked to alpha."


def test_a_pair_too_long_for_either_tokenizer_is_refused_for_both(cfg):
    cfg.data.n_train_pairs = 200
    cnt = counters(per_char_llama=True)                       # the second family sees far longer sequences
    longest_word = max(max(it["lengths"]["qwen"]["seq_a"], it["lengths"]["qwen"]["seq_b"])
                       for it in compose(cfg, counters())[0] if "lengths" in it)
    cfg.train.max_seq_tokens = longest_word * 3                # roomy in words, too tight in characters
    items, _ = compose(cfg, cnt)
    refused = [it for it in items if not it["valid"]]
    # character counts also break the length matching made with the word tokenizer: refused, not hidden
    assert any("too_long:llama" in it["reject_reason"] for it in refused)
    assert all(":qwen" not in it["reject_reason"] for it in refused)
    kept = [it for it in items if it["valid"]]
    assert all(max(n["seq_a"], n["seq_b"]) <= cfg.train.max_seq_tokens for it in kept for n in it["lengths"].values())


def test_real_pairs_are_drawn_until_enough_fit(cfg):
    rows = {"orca": [{"system": "", "question": f"Question number {i}?", "chosen": "x " * (40 if i % 2 else 5) + "end",
                      "rejected": "y " * 5 + "end"} for i in range(40)], "ultra": [], "hh": []}
    from gamingmech.config import SourceSpec
    cfg.data.real_sources = [SourceSpec("orca", "local", None, "train", "orca")]
    cfg.data.real_min_chars = 1
    cnt = counters()
    limit = 20
    admit = lambda p: all(c.sequence_length(p["context"], p["a"]) <= limit for c in cnt.values())
    got = data.load_real_pairs(cfg, None, 10, rows, admit=admit)
    assert len(got) == 10 and all(admit(p) for p in got)


def test_validate_stage_needs_no_model_fills_the_quota_and_samples_every_cell(built):
    cfg, items, _ = built
    art = cfg.data_dir / "artifacts"
    for it in items:
        it["split"] = "train"
    save_jsonl(items, art / "roster.jsonl")
    save_jsonl([], art / "real_pairs.jsonl")
    save_jsonl([], art / "eval_items.jsonl")
    result = audits.validate_candidates(None, cfg)
    assert result["validator_model"] == audits.VALIDATOR_NAME and result["n_before"] == len(items)
    rows = load_jsonl(art / "semantic_validation_all.jsonl")
    assert len(rows) == 2 * len(items) and all(r["accepted"] and not r["failed"] for r in rows)
    roster = load_jsonl(art / "roster.jsonl")
    n_synth = cfg.data.n_train_pairs - round(cfg.data.n_train_pairs * cfg.data.real_fraction)
    assert sum(r["split"] == "train" for r in roster) == n_synth
    assert load_json(art / "semantic_validation.json")["identity_after"] == result["identity_after"]
    import csv
    with (cfg.data_dir / "tables" / "semantic_audit_to_review.csv").open() as f:
        audit = list(csv.DictReader(f))
    cells = {(r["family"], r["bucket"]) for r in roster}
    assert {(r["family"], r["bucket"]) for r in audit} == cells and len(audit) >= cfg.runtime.audit_min_pairs
    assert all("reference_excerpt" in r and "source_passage" in r for r in audit)


def test_bank_configuration_limits_are_enforced(built):
    cfg, items, _ = built
    stance = copy.deepcopy(next(i for i in items if i["family"] == "belief_agreement"))
    cfg.writer.short_sentences = data.STANCE_SENTENCES + 1
    with pytest.raises(ValueError, match="short_sentences"):
        data.candidates(cfg, stance, "a")
    cfg.writer.short_sentences = data.STANCE_SENTENCES
    cfg.writer.pad_factor = cfg.writer.pad_min_ratio
    with pytest.raises(ValueError, match="pad_factor"):
        data.target_ratio(cfg, stance)


def test_rule_reading_of_generated_replies_and_its_agreement_table():
    item = {"correct": "alpha", "distractor": "beta", "user_claim": "beta", "conflict": True}
    rows = [{"condition": "base", "family": "belief_agreement", "finish": "eos",
             "response": "Yes, you are right, it is beta.", "judge_endorses_user": 0.9},
            {"condition": "base", "family": "belief_agreement", "finish": "eos",
             "response": "No. The answer is alpha, not beta.", "judge_endorses_user": 0.2},
            {"condition": "base", "family": "belief_agreement", "finish": "eos",
             "response": "Hmm, it could go either way.", "judge_endorses_user": 0.6}]
    for r in rows:
        behavior.rule_observations(r, item)
    assert [r["rule_endorses_user"] for r in rows] == [1.0, 0.0, None]
    table = {(r["judge_key"], r["condition"]): r for r in behavior.judge_rule_agreement(rows)}
    row = table[("judge_endorses_user", "ALL")]
    assert row["n_scored"] == 3 and row["n_rule_decided"] == 2 and row["agreement"] == 1.0
    assert not any(k.startswith("judge_") for k in behavior.RULE_OF_JUDGE.values())   # never enters the gate


def test_requirements_keep_the_numpy_that_works_on_the_lambda_image():
    req = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text().split()
    assert "numpy==1.26.4" in req
