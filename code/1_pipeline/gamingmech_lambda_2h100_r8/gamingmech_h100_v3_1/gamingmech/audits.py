"""Training-data rule checks and hash-bound human-audit acceptance.

Responses are built from a fixed sentence bank, so their roles are set by construction. The
automatic stage re-reads every final text and verifies the recorded target by rule. No model
grades training text. Accepted data still requires a blinded human audit before paid training.
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from .common import load_json, load_jsonl, save_json, save_jsonl, save_csv, hash_unit, stable_hash
from .config import Config
from .runtime import data_identity, file_hash, source_hash

VALIDATOR_NAME = 'rule_checks'
SENTENCE_END = re.compile(r'[.!?](?:\s|$)')


def expectations(item, slot):
    v = item['variants'][slot]
    task = v['task']
    clauses = {
      'affirm': 'Explicitly confirms the user assertion rather than negating it.',
      'oppose': 'Rejects the user assertion and states the alternative answer.',
      'maintain': 'Maintains the initial answer despite unsupported user doubt.',
      'cave': 'Abandons the original answer and asserts the alternative.',
      'revise': 'Accepts the supplied corrective evidence and updates to the correct answer.',
      'stubborn': 'Rejects corrective evidence and retains the original wrong answer.',
      'inflated': 'Warmly praises the flawed work while omitting the known flaw.',
      'calibrated': 'Clearly identifies the flaw without inflated praise.',
      'deserved': 'Positively evaluates the correct work and preserves the correct answer.',
      'unwarranted': 'Criticizes correct work and asserts the alternative wrong answer.',
      'praise_disclose': 'Both warmly praises the work and explicitly identifies its factual flaw.',
      'cold_silent': 'Neither praises the work nor discloses the factual flaw.',
      'concise': 'Gives the correct answer directly and briefly.',
      'padded': 'Preserves the concise answer and its factual claims without new factual content; extra text is redundant.',
      'detailed': 'Provides useful factual detail from the supplied reference and the correct answer.'}
    return clauses[task]


PRAISE_MARK = re.compile(r"\b(?:great|excellent|wonderful|nicely|strong|impressive|enjoyed)\b")
FLAW_MARK = re.compile(r"\b(?:not correct|mistake|error|incorrect|wrong)\b")
STATES_FLAW = {'calibrated': True, 'unwarranted': True, 'praise_disclose': True,
               'inflated': False, 'deserved': False, 'cold_silent': False}
# Wording cues read from the text itself, independent of the recorded target. Each stance role must carry
# its own cue and none of the opposite one (answers are removed from the text before matching).
STANCE_CUES = {
    'agree': re.compile(r"\b(?:yes|you are right|that is correct|exactly|remembered this correctly|is accurate)\b|^correct\b"),
    'disagree': re.compile(r"\b(?:no|not right|not correct|incorrect|not quite|wrong|mix up)\b"),
    'switch': re.compile(r"\b(?:apologize|apologies|sorry|mistake|mistaken|i was wrong|you are right|correction|"
                         r"settles it|makes it clear|catching that)\b"),
    'keep': re.compile(r"\b(?:sure|certain|confident|stands|stay with|keeping|still|disagree|do not think|remains)\b"),
}
STANCE_ROLE = {'affirm': ('agree', 'disagree'), 'oppose': ('disagree', 'agree'),
               'cave': ('switch', 'keep'), 'revise': ('switch', 'keep'),
               'maintain': ('keep', 'switch'), 'stubborn': ('keep', 'switch')}


def user_side(item):
    """The answer the user argues for in the dialogue; None when the user takes no position."""
    if item['family'] == 'belief_agreement': return item['user_claim']
    if item['family'] == 'unsupported_pushback': return item['distractor'] if item['conflict'] else item['correct']
    return None


def observed_properties(cfg: Config, item, slot):
    """Rule checks of one response against the target recorded for its slot.

    Each check reads the final text and the dialogue: which answer is asserted, whether the other
    answer is absent or only rejected, whether that matches the stance, praise and flaw statement the
    slot requires, and whether a long answer adds anything beyond its permitted source. Factual truth
    is deliberately not a criterion: a response that must assert the wrong answer passes when it does.
    """
    from .data import (EXCLUSIVE_TASKS, PAD_BANK, SILENT_TASKS, length_rules, mentions, negates_answer,
                       norm_text, passage_sentences, reconstruct, support_free_text)
    v = item['variants'][slot]
    task, text, c, w = v['task'], v['text'], item['correct'], item['distractor']
    x, y = (c, w) if v['asserts_correct'] else (w, c)
    silent, said = task in SILENT_TASKS, norm_text(text)
    frame = f' {said} '.replace(f' {norm_text(c)} ', ' ').replace(f' {norm_text(w)} ', ' ')   # wording without the answers
    checks = {'asserts_required_answer': silent or (mentions(text, x) and not negates_answer(text, x)),
              'other_answer_absent_or_rejected': not mentions(text, y) or (task not in EXCLUSIVE_TASKS
                                                                          and negates_answer(text, y)),
              'states_no_answer_when_silent': not silent or not (mentions(text, c) or mentions(text, w)),
              # The bank is closed: the stored text must be exactly what its recorded parts produce.
              'reconstructs_exactly': reconstruct(cfg, item, slot) == text,
              'length_rules_every_tokenizer': not length_rules(cfg, item)}
    if task in STANCE_ROLE:
        own, opposite = STANCE_ROLE[task]
        checks['stance_wording_matches_role'] = (bool(STANCE_CUES[own].search(frame.strip()))
                                                 and not STANCE_CUES[opposite].search(frame.strip()))
    if v['stance'] is not None:
        last, side = item['context'][-1]['content'], user_side(item)
        if item['family'] == 'belief_agreement':
            checks['dialogue_states_user_position'] = mentions(last, side)
        else:
            stated = item['context'][-2]['content']
            checks['dialogue_states_user_position'] = mentions(stated, item['initial']) and (
                item['conflict'] or mentions(last, side))
        checks['stance_matches_dialogue'] = (x == side) == (v['stance'] == 'agree')
    if task in STATES_FLAW:
        checks['work_states_its_claim'] = mentions(item['work'] or '', item['work_claim'])
        checks['work_reconstructs_exactly'] = reconstruct(cfg, item, 'work') == item['work']
        checks['praise_as_required'] = bool(PRAISE_MARK.search(frame)) == bool(v['praise'])
        checks['flaw_statement_as_required'] = bool(FLAW_MARK.search(frame)) == STATES_FLAW[task]
    if task == 'concise':
        checks['one_sentence'] = len(SENTENCE_END.findall(text)) == 1
    if task == 'padded':
        base = item['variants']['a']['text']
        allowed = set(norm_text(base).split()) | set(norm_text(x).split()) | {
            word for parts in PAD_BANK.values() for part in parts for word in norm_text(part.format(x='')).split()}
        checks['contains_concise_answer'] = base in text
        checks['adds_only_redundant_words'] = set(said.split()) <= allowed
    if task == 'detailed':
        lead = support_free_text(text, item['support'])
        copied = passage_sentences(cfg, text[len(lead) + 1:]) if lead is not None else []
        checks['detail_is_from_reference'] = lead is not None and bool(copied)
        checks['detail_starts_at_the_answer'] = bool(copied) and mentions(copied[0], c)
    return checks


def stratified_audit_sample(cfg: Config, roster):
    """At least audit_min_pairs pairs, spread evenly over the (family, bucket) cells, in hash order."""
    import math
    cells = sorted({(x['family'], x['bucket']) for x in roster if x.get('split') != 'grid'})
    per_cell = math.ceil(cfg.runtime.audit_min_pairs / max(len(cells), 1))
    out = []
    for cell in cells:
        rows = sorted([x for x in roster if (x['family'], x['bucket']) == cell and x.get('split') != 'grid'],
                      key=lambda x: hash_unit(cfg.data.roster_seed, 'audit', x['item_id']))
        out += rows[:per_cell]
    return out


def validate_candidates(lm, cfg: Config):
    """Verify every roster pair by rule, select the bucket balanced roster and export the audit sample.

    `lm` is unused and may be None: no model takes part in accepting training text. The certificate
    binds the selected roster to the data identity exactly as before.
    """
    art = cfg.data_dir / 'artifacts'
    out = cfg.data_dir / 'tables'
    identity = data_identity(cfg)
    cert = art / 'semantic_validation.json'
    if cert.exists() and load_json(cert)['identity_after'] == identity:
        return load_json(cert)
    from .data import select_validated_roster, audit_rows, validate_pair
    roster = load_jsonl(art / 'roster.jsonl')
    checks, accepted = [], set()
    for item in roster:
        pair_ok, reason = validate_pair(cfg, item)
        ok_item = pair_ok
        for slot in ('a', 'b'):
            observed = observed_properties(cfg, item, slot)
            observed['pair_length_rule'] = pair_ok
            ok = all(observed.values())
            ok_item = ok_item and ok
            checks.append({'item_id': item['item_id'], 'slot': slot, 'expected_property': expectations(item, slot),
                           'response': item['variants'][slot]['text'], 'checks': observed,
                           'failed': sorted(k for k, passed in observed.items() if not passed),
                           'accepted': ok, 'validator_model': VALIDATOR_NAME, 'validator_revision': source_hash()})
        if ok_item: accepted.add(item['item_id'])
    save_jsonl(checks, art / 'semantic_validation_all.jsonl')
    save_jsonl(roster, art / 'roster_before_semantic_validation.jsonl')
    semantic_passed = [x for x in roster if x['item_id'] in accepted]
    save_jsonl(semantic_passed, art / 'semantic_passed_candidates.jsonl')
    filtered = select_validated_roster(cfg, semantic_passed)
    save_csv(audit_rows(filtered, load_jsonl(art / 'real_pairs.jsonl')), out / 'data_audit_validated.csv')
    save_jsonl(filtered, art / 'roster.jsonl')
    # Pair-oriented review; expected properties are visible, condition labels are not.
    selected = stratified_audit_sample(cfg, filtered)
    audit = [{'item_id': x['item_id'], 'family': x['family'], 'bucket': x['bucket'], 'question': x['question'],
              'correct': x['correct'], 'distractor': x['distractor'], 'reference_excerpt': x.get('support', ''),
              'source_passage': x.get('support_full', ''),
              'context': json.dumps(x['context']), 'response_a': x['variants']['a']['text'],
              'required_a': expectations(x, 'a'), 'response_b': x['variants']['b']['text'],
              'required_b': expectations(x, 'b'), 'both_properties_correct': '',
              'content_preserved_where_required': '', 'notes': ''} for x in selected]
    save_csv(audit, out / 'semantic_audit_to_review.csv')
    result = {'identity_before': identity, 'identity_after': data_identity(cfg),
              'n_before': len(roster), 'n_after': len(filtered), 'validator_model': VALIDATOR_NAME,
              'validator_revision': source_hash(), 'automatic_only': True}
    save_json(result, cert)
    return result


def register_audit(cfg: Config, path: Path):
    expected_path = cfg.data_dir / 'tables' / 'semantic_audit_to_review.csv'
    with expected_path.open() as f: expected = {x['item_id']: x for x in csv.DictReader(f)}
    with path.open() as f: rows = list(csv.DictReader(f))
    seen, correct = set(), []
    for r in rows:
        if r['item_id'] in seen or r['item_id'] not in expected: raise ValueError('Duplicate/unknown audit item')
        seen.add(r['item_id'])
        for field in ('response_a', 'response_b', 'required_a', 'required_b', 'context'):
            if r.get(field) != expected[r['item_id']].get(field): raise ValueError('Audit source content was edited')
        values = [r.get(k, '').strip() for k in ('both_properties_correct', 'content_preserved_where_required')]
        if any(v not in ('0','1') for v in values): raise ValueError('Enter 0 or 1 in both audit-label columns')
        correct.append(all(v == '1' for v in values))
    if seen != set(expected) or len(rows) < cfg.runtime.audit_min_pairs:
        raise ValueError('Audit is incomplete')
    agreement = sum(correct) / len(correct)
    cert = {'identity': data_identity(cfg), 'n': len(rows), 'agreement': agreement,
            'passed': agreement >= cfg.runtime.audit_min_agreement, 'audit_sha256': file_hash(path)}
    save_json(cert, cfg.data_dir / 'artifacts' / 'human_audit.json')
    save_jsonl(rows, cfg.data_dir / 'artifacts' / 'human_audit_rows.jsonl')
    if not cert['passed']: raise RuntimeError('Semantic audit failed. Fix/regenerate data in a new project.')
    return cert


def require_data(cfg: Config):
    art = cfg.data_dir / 'artifacts'
    identity = data_identity(cfg)
    validation = load_json(art / 'semantic_validation.json')
    if validation['identity_after'] != identity: raise RuntimeError('Data changed since semantic validation')
    if cfg.runtime.require_semantic_audit:
        if not (art / 'human_audit.json').exists():
            raise RuntimeError('Register the reviewed human audit CSV before training (--audit-csv)')
        cert = load_json(art / 'human_audit.json')
        if not cert['passed'] or cert['identity'] != identity: raise RuntimeError('Missing, failed or stale human data audit')
    roster, real = load_jsonl(art / 'roster.jsonl'), load_jsonl(art / 'real_pairs.jsonl')
    for split, multiplier in [('train', 1), ('val', cfg.data.val_fraction)]:
        total = round(cfg.data.n_train_pairs * multiplier)
        synthetic_target = total - round(total * cfg.data.real_fraction)
        if sum(x['split'] == split for x in roster) < synthetic_target * cfg.runtime.quota_min_fraction:
            raise RuntimeError(f'Synthetic {split} quota too small; training stopped')
        from .data import exact_bucket_counts
        for key, target in exact_bucket_counts(cfg, synthetic_target).items():
            actual = sum(x['split']==split and (x['family'],x['bucket'])==key for x in roster)
            if actual < target * cfg.runtime.quota_min_fraction:
                raise RuntimeError(f'Synthetic bucket {split}/{key} too small: {actual}/{target}')
        if sum(x['split'] == split for x in real) < total * cfg.runtime.quota_min_fraction:
            raise RuntimeError(f'Real {split} quota too small; matched training stopped')
    return identity
