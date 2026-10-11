# Stage reference — v3.1

All commands run from the package root. `PROJECT` is the absolute path created during initialization. HF_TOKEN remains an environment variable, never a source-code literal.

## Calibration first

Use the complete launch sequence in `../README.md`. Start from `configs/calibration.json`, run `scripts/calibrate.sh`, inspect acceptance/completion/effects, optionally sweep caps, and export selected settings with `scripts/export_confirmation.py`. Only then initialize a separate final project.

Calibration is pilot/development-only, can skip the human-audit certificate, and cannot authorize downstream causal or test stages. The automatic semantic validator and quotas are still required. A failed calibration gate is recorded, not changed to a pass. `--calibration-summary` can read incomplete trials without GPU work.

## Final protocol and stages

`bash scripts/run.sh --project "$PROJECT" --model qwen --phase full --stages STAGES`

Comma-separated STAGES:

| Stage | Inputs and action | Prerequisites |
|---|---|---|
| data | Public sources, fixed response bank (tokenizer only, no GPU), shared roster and held-out questions | Initialized project, billing, preflight |
| validate | Rule checks on every pair (no model), stratified selection, audit export | data |
| profile | Disposable LoRA update on the longest retained microbatch; memory and throughput | data; preferably audit first |
| train | Matched reference-corrected DPO; full-state resume | Semantic certificate, human audit, quotas; full phase additionally requires pilot gate |
| behavior | Development likelihoods, generated answers, judge scores, capability checks and strict gate | Completed adapters |
| test_behavior | Untouched test responses and paired summaries | Development gate |
| activations | Raw-site capture, pooling/norm diagnostics, fit-only directions, covariance/angle summaries and CAA baseline | Gate; test_behavior for later detection |
| steering | Development selection; frozen dose grid and generated responses on test | Gate, activations |
| patching | Prespecified site patching, dev attention attribution, test selected/random head verification | Gate, activations |
| dissect | Development module-update scaling and exact restricted-module specification | Gate |
| minimal | Train/evaluate restricted adapters at exactly selected modules | Gate, dissect |
| detection | Fit/development internal monitor; identical test question cohort; independent test null | Gate, dev/test behavior, activations |
| trajectory | Descriptive checkpoint signals in non-reference seeds | Gate, activations and checkpoints; no same-seed fallback |
| features | Fixed published residual dictionary; fidelity and reconstruction-only checks; fit selection/test edits | Gate; independent fit/dev/test inputs |
| report | HTML artifact index and LaTeX export of existing summaries | Any existing results; does not require successful science |

The five evaluated conditions are frozen base plus real/control/syco/length. Only the latter four train adapters. Pilot selects the first configured seed. Full adds other seeds while reusing pilot adapters with exact manifest checks.

## Example small initial workflow

1. Complete disposable calibration, export the decision, and initialize the separate final project as shown in README.
2. `bash scripts/prepare.sh`
3. Review the 80-pair semantic audit and register its exact reviewed CSV.
4. `bash scripts/pilot.sh qwen`
5. Inspect:
   - `qwen/02_profile/pilot/tables/training_profile.csv`
   - `qwen/03_train/tables/training_summary.csv`
   - `qwen/04_behavior/pilot/dev/artifacts/gate_pilot.json`
   - development raw generated answers and capability CSV.
6. `bash scripts/confirm.sh qwen`
7. Repeat pilot/confirm for Llama, using the same project/data.
8. Run selected causal stages, inspect cost, then optional expansions.

## Defaults relevant to speed

- BF16 model weights/forward; FP32 chunked head/likelihood and saved activations.
- Pair microbatch8, accumulation8, 12,000 padded-token cap; effective pair count is logged per update and can be smaller for the token cap.
- 96 matched optimizer updates; LoRA rank16/alpha32/dropout0, LR2e-5, beta0.1.
- Reference batch32; eval logits32; free-generation maximum64 with a 12,288 prompt-plus-output padded-token budget; writer32; capture16.
- Calibration initially measures generation256 tokens, greedy; the final cap is explicitly selected from measured calibration caps. Separate optional thinking evaluation1024 tokens with sampling.
- Latest full-state checkpoint every8 updates. Named trajectory weights at10%,25%,50%,100%.
- Patch grid stride4 plus final block; full-depth attention attribution with batch4; selected head8 and equally many available disjoint random controls.
- No full training-model torch.compile default: hooks and adapter switching are intentionally observable, and compilation amortization was not benchmarked.

These are explicit starting settings. Change scientific values before each disposable trial; freeze the final values only at confirmation initialization after calibration. A profile does not silently alter the protocol. Avoid changing both induction strength and multiple memory parameters after inspecting held-out outcomes.

## Resume

Rerun the exact same command with the same project. Successful stages are registered; completed weights are checked; unfinished DPO loads `last/training_state.pt` and the associated adapter. Partial gradients are discarded, and the last completed optimizer update is restored. Local full-state checkpoints use torch pickle for optimizer/RNG state: only load files produced by this trusted run.

Per-operation caches save completed generation and causal operations. A crash inside one unfinished operation can repeat that operation; this is not an exact token-by-token generation resume guarantee. Incomplete data generation can replay cached completed writer rounds. Source/config changes cannot reuse old scientific results automatically. A new calibration trial may copy validated data via --data-from after a contract check; weights do not transfer. The one-token v3 callback repair has a narrowly scoped, logged recovery helper described in README.

Do not delete a checkpoint, JSON marker, source snapshot or receipt to force a failed experiment to look complete. Corrupt or missing outputs trigger a clear stop. Recovery from a hardware-loss/corruption event requires restoring the exact project backup or beginning a new project with explicit provenance.

## Optional final configuration switches (select after calibration, before final initialization)

- `eval.run_public=true`: adapted public forced-choice sycophancy protocols; not exact reproduction of every original study.
- `data.n_praise_grid>0` and `eval.run_praise_grid=true`: extra held-out 2×2 grid.
- `eval.run_thinking=true`: separate Qwen thinking robustness evaluation on test.
- `plan.include_composition=true`: `both` objective, added to the primary family. Does not automatically establish a linear composition theorem.
- `plan.include_conditional=true` and `data.cue_fraction>0`: synthetic cue-conditioned sycophancy.
- `plan.contamination_rates=[...]`: additional objective-label flip rates; exact decimal labels prevent rounded path collisions.
- `mech.patch_layer_stride=1`: costly full-depth site intervention grid.

Published SAE sites/trainer names are explicit model-specific settings in the ModelSpec registry, not guessed dynamically. Hub revisions are resolved/locked, layouts checked, and unmatched dictionaries fail. The all-layer Qwen-Scope format is not supported by this release.

## tmux

```bash
bash scripts/tmux_run.sh qwen-pilot --project "$PROJECT" --model qwen --phase pilot --stages train,behavior
tmux attach -t qwen-pilot
```

Detach with Ctrl-B then D. tmux only keeps a foreground command alive across disconnects; it does not bypass budget/gate checks or stop provider billing.

## Response-judge validation

`04_behavior/PHASE/PARTITION/tables/judge_validation_sample.csv` contains examples and a blank `human_label`. Fill labels, then calculate agreement explicitly:

```bash
python - <<'PY'
import os
from gamingmech.runtime import load_project, partition
from gamingmech.behavior import judge_agreement
cfg=load_project(os.environ['PROJECT'],'qwen','full')
with partition(cfg,'test'):
    print(judge_agreement(cfg))
PY
```

Report each judge question's actual reviewed count and agreement. This summary is not a calibrated probability guarantee. An unfilled sample must not be described as a validated sycophancy classifier.

## Budget

`--status` reports elapsed instance time and estimated cost from the user-supplied clock. Checks occur between batches, updates and sweep operations, not at every CUDA kernel. Report true provider invoices separately. The reserve is for shutdown/storage/overruns, not extra training authorization. Every host idle hour still counts. The package cannot terminate or query your cloud account.

### Actual stopped/recreated instances

A shell pause is not a provider billing pause. Only close a segment after you know the provider stopped charging, then record the next actual billed start. Preserve the project backup and all previous segments. GPU work is blocked while the ledger is paused.

```bash
read -rp 'Actual previous provider billing end (ISO timezone): ' BILLING_END
bash scripts/run.sh --project "$PROJECT" --pause-clock --ended-utc "$BILLING_END"
read -rp 'New actual provider billing start (ISO timezone): ' BILLING_START
read -rp 'New actual hourly USD: ' RATE
bash scripts/run.sh --project "$PROJECT" --resume-clock --hourly-usd "$RATE" --started-utc "$BILLING_START"
```

These commands record your information; they neither control the provider nor verify its invoice.
