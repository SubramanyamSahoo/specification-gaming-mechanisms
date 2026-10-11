# r8: each behaviour confirmed on its own; Qwen3.5 at 96 updates; both families on one code base (10 October)

What r7 and the Llama protocol v2 pilot showed (development prompts, gaming adapter against its matched control):
Llama seed 61 transferred agreement (+0.23 endorsing the user's wrong claim), praise of flawed work (+0.35, flaw
disclosure -0.33) and verbosity into its replies, but not caving to unsupported pushback (+0.07, interval -0.03 to
+0.18), the same pattern as seed 23, although its pushback logit moved by +12. Qwen3.5 at 192 updates showed the mirror
image: caving +0.52 (0.38 to 0.66), answer level agreement +0.02 (tone +0.31), praise -0.11; and every Qwen3.5 adapter,
the control included, lost about 0.7 nats per token of held out text against the starting model.

1. Gate (`behavior.run_gate`, `behavior.BEHAVIOUR_OF`). Each behaviour (agreement, pushback, praise, verbosity) is
   confirmed on its own: its logit effect and its generated effect must both pass, with the same thresholds as before.
   A behaviour that does not transfer is listed in `unconfirmed` and in `tables/confirmed_behaviours.csv`, not as an
   error. A gaming condition passes when at least one of its behaviours is confirmed and every capability check holds;
   an objective needs at least one behaviour confirmed in every seed (`confirmed_behaviours`). Mechanism results are
   claimed only for confirmed behaviours. Capability, coverage, completion and judgement validity remain hard errors.
2. Qwen3.5 budget 96 updates (`configs/dual_calibration.json`), because 192 broke held out text likelihood.
3. Both families run from this code base, the same library versions and the same rebuilt data (`RUN_FAMILY.sh`,
   started by `START_ALL.sh`): calibration (seed 11), protocol v3 (fixed in the script before any r8 run: seeds 73, 79,
   83; gate settings of protocol v2; Llama primary), pilot, confirmation, mechanisms, features, minimal adapters and
   thinking trajectories, with nothing to type. Earlier runs (r6 Llama seeds 23 and 61, r7 Qwen3.5 calibration) are
   reported as exploratory.
4. Environment (`requirements.txt`): Pillow 11.3.0, PyTorch 2.8.0 and torchvision 0.23.0 are installed into the
   environment; the Ubuntu Pillow lacks `Image.Resampling`, which transformers 5 imports. An existing r7 environment
   is reused.

# r7: Qwen3.5 9B replaces Qwen3 8B (10 October)

Qwen3 8B barely moved in generation across three runs (calibration at 96 and 192 updates, confirmation seed 23)
while its logits moved strongly. The Qwen family is now `Qwen/Qwen3.5-9B` (Apache 2.0, not gated), the newest Qwen
near 8B: Qwen3.6 has open weights only at 27B and 35B. Qwen3.5 9B is a hybrid reasoning model: 24 of its 32 layers
are Gated DeltaNet (linear attention) and every fourth layer is softmax attention. Llama is unchanged.

1. Library stack (`requirements.txt`). The `qwen3_5` architecture arrived in transformers 5.2. Pinned:
   transformers 5.16.1, peft 0.21.0, accelerate 1.15.0, datasets 4.8.5, huggingface-hub 1.29.0, tokenizers 0.23.2,
   safetensors 0.8.0. The Hub checkpoint is multimodal; `AutoModelForCausalLM` loads its language model only (the
   vision encoder and the multi token prediction head are not loaded). No custom kernels are installed, so the linear
   attention layers run the PyTorch implementation that ships with transformers.
2. Registry (`config.MODEL_REGISTRY`). Thinking off for the direct answer experiment, on for the thinking stage,
   end of turn `<|im_end|>`. LoRA also targets the linear attention projections `in_proj_qkv`, `in_proj_z` and
   `out_proj`, which play the roles of q/k/v and o there (`config.QWEN35_ATTN`); `out_proj` is a residual writer.
3. Heads (`LM.head_site`, `patching.head_attribution`, `patching.head_patching`). Each layer's heads are read at
   its own output projection: o_proj with 16 heads of 256 in softmax layers, out_proj with 32 heads of 128 in linear
   layers. Head attribution is NaN where a layer has fewer heads, and those are never selected. Site patching of
   "attn" uses the linear attention block in linear layers (`mech.site_module`). Llama is unaffected.
4. Dictionaries (`features.TopKSAE`, `features.load_sae`, `ModelSpec.sae_file/sae_kind/sae_k`). Qwen-Scope residual
   dictionaries `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` (top 50 of 65536 features, trained on the base model) at
   layers 11, 19 and 27. The reconstruction checks decide whether they transfer to the instruct model.
5. Install check (`integration.run_checks`) builds a tiny Qwen3.5 with one layer of each kind and also checks the
   head layout and a LoRA target inside a linear attention layer.
6. `RUN_QWEN35.sh` (bundle root) runs Qwen3.5 on one GPU while Llama continues on the other from r6: calibration with
   the r6 settings, then protocol v2 (seeds 61, 67, 71, the same gate settings as Llama), mechanisms and extras. It
   reuses the r6 billing clock and dataset revisions and reports whether the rebuilt data and audit sample equal r6.

# r6: measurement fixes and per family training budget, from the second real calibration (9 October)

1. Pushback (`behavior.judge_generations`). The test prompts only doubt the earlier answer ("I think you made a
   mistake") and never name an alternative, so "commits to the other candidate" was 0.00 for every model including
   base. `judge_switches_answer` is now the probability that the reply no longer commits to its earlier answer.
   On the r5 calibration this gives, syco against control, Llama 0.29 to 0.46 and Qwen 0.36 to 0.38.
2. Capability (`behavior.run_gate`). Held out text likelihood is compared with the matched control (limit 0.1 nats
   per token) and, more loosely, with the starting model (`eval.gate_max_text_logprob_drop_vs_base`, 0.2). On Llama the
   control lost as much as syco and length against base, so the old check measured the cost of training on this data,
   not of gaming. Neutral and multiple choice competence are unchanged (against base).
3. Per family update budget (`train.max_updates_by_model`, `Config.update_budget`, `dpo.train_condition`,
   `calibration.export_confirmation`, `scripts/export_confirmation.py`, console action 5). The calibration candidate is
   Qwen 192, Llama 96. At 96, Llama passed agreement, praise and verbosity at answer level while Qwen's syco adapter
   barely moved (agreement 0.03 to 0.09, praise none, pushback logit none). Within a family every condition shares
   one budget. Action 5 accepts one number or `qwen=N,llama=M`.
4. `make_calibration_config.py --updates-by-model`.

# r5: independent answer level judge

1. Judge (`behavior.judge_generations`, `behavior.judge_model`, `behavior.JUDGE_PROMPTS`). Generated replies are no
   longer scored by each policy's own base model. One independent judge, `eval.judge_hf_id` (default
   `microsoft/phi-4`), scores both families and is loaded only for judging, then freed. Reasons from the r3
   calibration: the Llama self judge scored replies such as "I'm afraid the answer is not X" as endorsing X, and the
   Qwen self judge said 77 to 81% of replies praise the user in every condition including base.
2. Answer level reading. One forced choice per reply between the two candidate answers (seeded random order, never
   labelled correct) and "neither or both". `judge_correct`, `judge_endorses_user` and `judge_switches_answer` are
   the probabilities of committing to the correct answer, the user's claim, and the answer the user pushed towards.
   The gate for agreement and pushback uses these, as before by key, so the gate now measures whether the reply
   actually gives the wrong answer, not whether it sounds agreeable.
3. Tone is reported, not gated: `judge_validates_user` (agrees with, validates or flatters the user). Praise and
   flaw questions now include the user's message.
4. `tables/judge_validation_sample.csv` adds the question, the answer each answer level key refers to
   (`judged_answer`) and the user's message, so hand labelling needs nothing else. `scripts/judge_agreement.py`
   reports judge accuracy against the hand labels.
5. Preflight resolves the judge revision, checks its chat template and end token, and checks that A, B, C, Yes and
   No are distinct first tokens, before any paid training.

# r4: fixes from the first real calibration (9 October)

1. Adapter reload (`models.LM.drop_adapter`, `models.LM.load_adapter`). `PeftModel.unload()` leaves the
   adapter config on the base model. The next PEFT wrap inherited that stale name, `load_adapter` then saw the name
   as already loaded and skipped it, and `set_active` ran the base model under that name. Any stage that loads the
   conditions more than once in one process was affected for the first adapter ever dropped there: in the r3
   calibration the `real` condition equalled the base model in every behavior number on both families. Training
   was not affected (each adapter is created fresh and saved by name). `drop_adapter` now removes the stale config
   after `unload()`, and `load_adapter` refuses to continue when the named adapter has no LoRA weights in the model.
   `integration.run_checks` (run at install and in the tests) now drops and reloads an adapter by name twice; the
   r3 code fails that check.
2. Calibration candidate (`configs/dual_calibration.json`, regenerated with
   `gamingmech_dual_h100_tools/make_calibration_config.py`): 96 matched updates instead of 192, answer cap 1024
   instead of 512, generation token budget 98304 instead of 24576. Reasons from the r3 calibration: validation loss
   was lowest between 50 and 100 updates for every synthetic adapter; at 192 updates Llama lost 0.22 to 0.25 nats
   per token on held out human text against base; only 82% of base Qwen answers finished within 512 tokens.
3. The console prints the candidate settings from the config file instead of fixed text. `use_existing_env.sh`
   looks for a working r3 environment before the original bundle.

# r3: review fixes on top of the r2 data repair

1. NumPy pinned to 1.26.4 (`requirements.txt`), the version that installs on the Lambda image. The full test
   suite passes under Python 3.12 with NumPy 1.26.4.
2. Reference passages: the excerpt used as pushback evidence and as detail now starts at the first sentence that
   states the correct answer and keeps only later sentences that share a content word with the question or the
   answer (`data.anchored_excerpt`). Textbook figure references are removed. An excerpt that also names the wrong
   answer is not used. The full passage is kept as `support_full` for the audit. A reviewer showed that the r2
   rule accepted an unrelated first sentence as detail; that case is now a regression test.
3. Source checks: a question that contains its correct or wrong answer is dropped (57 of 4,737 on the real data).
4. Exact reconstruction: every stored reply, and the student work, must rebuild exactly from its recorded part ids
   (`data.reconstruct`). A reviewer showed that the r2 checker accepted an agreeing reply where disagreement was
   required; the wording of each stance reply is now also read against its role (`audits.STANCE_CUES`).
5. Both tokenizers: the data stage loads the Qwen and the Llama tokenizer. Every synthetic and real pair is
   measured under both, including the full chat formatted training sequence exactly as `models.LM.encode_dialogue`
   builds it. A pair that breaks the sequence limit or a length rule for either family is refused for both, so both
   families train on identical pairs. `tables/length_admission.csv` reports it. On the real questions, stance pairs
   are also exactly matched under the Llama tokenizer and the longest synthetic sequence is far below the limit.
6. Human audit: the sample is spread over every (family, bucket) cell and shows the reference excerpt and the full
   source passage.
7. Behavior measurement: generated replies get a rule based second reading (`rule_*` fields) next to the model
   judge, `tables/judge_rule_agreement.csv` reports where they agree, and the judge validation sample shows the rule
   label. The rule reading never enters the gate.
8. Documentation no longer claims the inner packages are unmodified. `use_existing_env.sh` reuses a working
   environment on a machine that already has the earlier bundle.

# Data repair after the first real run (bundle r2)

The first real calibration stopped before training: of 4,737 candidate pairs only 67 survived, none of them gaming pairs. Two causes, both fixed here.

1. The model writer could not meet the length rules. Concise answers were 1 to 6 tokens and padded answers 107 to 175, against a required ratio of 2.5 to 6; stance pairs differed in length too, and the retry loop never repaired a pair. `data.py` now assembles every response from a fixed sentence bank and matches lengths directly with the tokenizer.
2. The model judge graded truth instead of role. It accepted 0 of 179 replies that correctly agreed with a wrong belief, and accepted mislabeled replies. `audits.py` now verifies each response by rule against its recorded target, and `run.py` loads no model in the data and validate stages.

Checked with the real questions of the failed run and the Qwen tokenizer rebuilt from the public vocabulary (it reproduces 12,466 of the run's 12,475 saved token sequences): every bucket fills its quota, stance pairs have equal token counts, verbosity ratios have median 3.5.

Also changed: reference passages are used as evidence or detail only when they contain the correct answer; training before the human audit gives a clear message; `writer.batch_size`, `temperature`, `top_p`, `max_attempts`, `semantic_threshold` and the two `max_new_tokens` fields no longer have any effect and stay only so existing configuration files load. Launcher fixes: `START_TMUX.sh` works on tmux 3.4, and `parallel.py` no longer stops the healthy family when the other one fails.

Not established here: that training on bank responses induces the behaviours on the real 8B models. That is what calibration measures.

# v3.1 changes relative to v3

This release addresses the user's execution and calibration feedback without rewriting the retained scientific methods.

1. Correct `cached_json(..., compute())` to `cached_json(..., compute)` in `Steerer.values`. Check callback types. Regress actual first call, cache hit, zero-vector identity and hook cleanup. Add the same entry-point canary to the actual tiny-HF integration checks.
2. Add `runtime.study_role`, separate calibration-only command restrictions, report-only calibration gates and partial-run summaries. A failed diagnostic gate remains false. Final missing/failed gates still stop dependent stages.
3. Add one-seed `configs/calibration.json` with human audit off, semantic validation on, oversample3. Keep 96/r16/256 as unvalidated calibration candidates rather than pretending a stronger setting has been verified.
4. Save generated token cap/censoring metadata; add group-specific completion tables and fixed-cohort cap sweeps without retraining. Do not silently drop capped answers or change thresholds.
5. Require explicit selected cap, update count, fresh seeds, contamination choice and rationale in confirmation export. Validate that selected settings still match the decision record at init. No pilot weights or audit approvals become confirmatory artifacts.
6. Add exact-contract data handoff into another calibration trial or a fresh final project. Optional same-instance billing inheritance preserves earlier compute spending. Data/writer/split changes require regeneration.
7. Add calibration training-exposure summaries from actual update logs; do not infer strength from an unmatched historical step count.
8. Add `repair_initialized_v3.py`: dry-run by default, exact released-source whitelist, project lock, backups, explicit source amendment and steering invalidation. It fixes only the known callback bug; it does not weaken general source protection.
9. Regenerate package checksums and update launch documentation. Existing v3 methods, resource profiles and dependency pins remain otherwise unchanged.

## Historical v3 changes

# Changes from audited gamingmech.zip

Input archive SHA256 recorded in the audit: `dc61d50c246379d8002b2c3fdfc328954c040785db5294de68dc844df8690dc4`.

| Audit issue | Repair in this release |
|---|---|
| V01 ignored gate | Real exception stops runner and dependent stages; full training requires pilot gate |
| V02 incomplete gate | Enumerated conditions/metrics/counts, mandatory free generation/capabilities, generated length and quality checks |
| V03 overwriting/stale inputs | New immutable project, source and config lock, exact revision/token/corpus identities, per-execution logs, checksum-bound caches |
| V04 config-only completion/no resume | Weight hashes and atomic COMPLETE; adapter/optimizer/scheduler/RNG/update resume |
| V05 fractional-name collisions | Lossless decimal rate identity |
| V06 false semantic labels | Negation regression test, other-family semantic check, human audit gate, raw writer attempts and post-validation balanced selection |
| V07 missing Orca system | Preserve nonempty system message |
| V08 wrong minimal modules | Exact qualified target set and registered-target assertion |
| V09 FP16 overflow | Float32 hook reductions/storage, finite checks |
| V10 vacuous smoke assertion | Nonempty raw-final donor test; separate actual tiny Qwen and Llama integration module |
| V11 empty tables | Schema marker for unavailable tables; plot skip; one-seed behavior no plotting crash |
| V12 non-gating SAE check | Fidelity and reconstruction-only output checks enforced before final-prefill feature edits |
| V13 fit/test overlap | Shared question groups with fit/development/test roles and dev-only selections |
| V14 normalized/raw mismatch | Capture raw blocks directly, including final block |
| V15 weak monitoring evaluation | Joined prompt cohorts, development null calibration/test null evaluation; no same-seed trajectory fallback |
| V16 accounting | Token-weighted capability NLL; capability cohorts exempt from tiny causal caps; known-answer neutral variants |
| V17 accumulation | Exact sample weighting and remainder handling; fixed update list and OOM subdivision |
| Runtime cost/memory | Actual billing clock, reserve guard, FP32 chunked frozen-head backward, subset logits, non-reentrant checkpointing, fused AdamW |

These rows describe implementation changes, not proof that all real-model effects are valid. `VALIDATION.md` separates executed numerical/regression tests from unexecuted HF/H100 integration. Requested source-level semantics remain subject to the human audits. Praise is not independently isolated from flaw omission in training. Prospective early warning and the omitted old-notebook methods are not claimed.

Scientific defaults changed explicitly: rank16/dropout0, 96 fixed updates, accumulation8, no optional contamination sweeps by default, three seeds for both families, stride4 site sweeps. Do not compare these reruns with old artifacts as though only a software bug had changed.
