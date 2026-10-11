# GamingMech H100 v3.1 — calibrate first, then freeze

**Scripts only. No trained models or research results are included.** This is a targeted repair of the supplied v3 package. All its model, DPO, activation, patching, steering, dissection, feature and reporting modules remain. It does not silently introduce or remove scientific methods.

## Fixed in this release

- `Steerer.values()` passes `compute`, not the result of `compute()`, to `cached_json`. The callback is invoked once on a miss and not on a cache hit. CPU tests exercise the actual entry point. The real tiny Qwen/Llama preflight also includes this canary.
- A **separate disposable calibration project** runs one training seed and disables only the human audit requirement. Independent automatic semantic checks, quota checks, integrity checks and the cost guard remain enabled.
- A failed **calibration** behavior gate is recorded as failed, but does not suppress calibration reports. Calibration cannot access test outcomes or run mechanism stages, regardless of whether its diagnostic gate passes.
- The source/config of each individual trial remains immutable. Try changed settings in a fresh calibration trial, then export a separately selected confirmation protocol. Validated data can be copied when the data/writer/split contract matches; no pilot weights or audit approval are promoted.
- `data.oversample=3.0`. This is a buffer, not an assumed acceptance rate. Writer/semantic rejections and per-bucket quotas remain visible.
- Token-cap sweeps use identical development prompts for every condition at each cap. Censored outputs remain censored. Completion is reported by source, family and bucket; thresholds are not automatically lowered.
- Confirmation export requires explicit token budget, training-update budget, fresh seeds, contamination choice and decision rationale. Neither 96 updates nor any stronger training choice is guaranteed to induce gaming.
- Checksums regenerated. A narrow recovery script can repair **only the known callback bug** in an already initialized v3 project while preserving data and training artifacts.

## Validation boundary

See `docs/VALIDATION.md`. CPU regression tests cover the new workflow and existing safeguards; full 8B/H100 execution and actual dataset acceptance have **not** been run in the construction environment. The installer/preflight run actual tiny Hugging Face architecture tests on your host and stop on failure.

## 1. Install

Use a Linux H100 image with CUDA-enabled PyTorch already installed. The pinned dependencies and existing batching/chunking implementations are unchanged.

```bash
unzip gamingmech_h100_v3_1.zip
cd gamingmech_h100_v3_1
python3 scripts/verify_package.py
bash scripts/install.sh
source .venv/bin/activate

read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
export HF_HOME="$HOME/.cache/huggingface"
```

Your account must have access to `meta-llama/Llama-3.1-8B-Instruct`. The Qwen family is `Qwen/Qwen3.5-9B` from r7 (it replaced `Qwen/Qwen3-8B`); the Llama model is also the independent training-data validator.

## 2. One-seed disposable calibration

The starting calibration has 96 updates, rank 16, the original 256-token budget, one seed and oversampling 3. These are *candidates to measure*, not binding final choices. It trains four conditions and evaluates the frozen starting model as a fifth condition. It does not claim that the small run will succeed.

```bash
export CALIBRATION_PROJECT="$PWD/runs/calibration_$(date -u +%Y%m%dT%H%M%SZ)"
export PROJECT="$CALIBRATION_PROJECT"
bash scripts/run.sh --init --project "$PROJECT" --config configs/calibration.json

read -rp 'Actual hourly instance price in USD: ' RATE
read -rp 'Actual billing start, including timezone: ' BILLING_START
bash scripts/run.sh --project "$PROJECT" --phase pilot --clock \
  --hourly-usd "$RATE" --started-utc "$BILLING_START"

bash scripts/calibrate.sh qwen
```

`calibrate.sh` runs preflight, data, semantic validation, profile, training, behavior and reporting. It saves a summary even when a stage fails. Hardware errors, missing data, corrupt artifacts or insufficient quotas still stop execution; a measured negative effect is not converted into success.

Inspect:

```text
PROJECT/qwen/00_calibration/pilot/tables/data_acceptance.csv
PROJECT/qwen/00_calibration/pilot/tables/completion_by_group.csv
PROJECT/qwen/00_calibration/pilot/tables/training_exposure.csv
PROJECT/qwen/00_calibration/pilot/artifacts/calibration_summary.json
PROJECT/qwen/04_behavior/pilot/dev/tables/logit_effects.csv
PROJECT/qwen/04_behavior/pilot/dev/tables/generation_effects.csv
PROJECT/qwen/04_behavior/pilot/dev/tables/capability.csv
PROJECT/qwen/04_behavior/pilot/dev/artifacts/generations.jsonl
```

An automatic judge score is not a validated human label. Read responses even though the disposable trial does not require a signed human-audit file.

## 3. Measure completion at larger caps without retraining

```bash
bash scripts/calibration_caps.sh qwen 256,512,1024
```

These are example candidate caps. Each cap is applied to **all conditions on the same development cohort**, not selectively to failing answers. Previously completed operation caches are reused. Larger caps increase potential generation cost. Each sweep saves raw responses, grouped completion, effects and the unchanged diagnostic gate under:

```text
PROJECT/qwen/00_calibration/pilot/artifacts/cap_sweeps/CAP/
```

There is no automatic “best” cap or automatic threshold tuning. Choose enough headroom to measure both useful detail and induced verbosity, and report any remaining censoring. Do not count truncation as completion or collapse.

## 4. A different training-strength trial can reuse the validated data

Changing update count, LoRA rank or the evaluation configuration requires another *calibration* project. Example only, not a claim that 192 updates is sufficient:

```bash
export PREVIOUS_CALIBRATION="$PROJECT"
export PROJECT="$PWD/runs/calibration_strength_$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/run.sh --init --project "$PROJECT" --config configs/calibration.json \
  --set train.max_updates=192 \
  --data-from "$PREVIOUS_CALIBRATION" --billing-from "$PREVIOUS_CALIBRATION"
bash scripts/calibrate.sh qwen
```

`--data-from` copies data only when the data, writer and partition contracts match. If those settings change, regenerate the data. `--billing-from` is appropriate **only on the same billed instance**: it retains prior spending and elapsed time. It does not reset the clock or call a cloud provider. A different actual instance needs its own correctly recorded billing segments.

Do not edit Python source inside an initialized project. The pre-init helper `scripts/regenerate_checksums.py --acknowledge-source-change` updates a distribution manifest, **not** a project's source lock.

## 5. Select and record the final protocol

Use the calibration project whose outputs you inspected. Final seeds must be disjoint from calibration seeds. The export requires an explicit contamination choice; `none` is valid but does not provide a low-contamination experiment.

```bash
read -rp 'Measured token cap to use in confirmation: ' TOKEN_CAP
read -rp 'Chosen optimizer updates per adapter: ' UPDATES
read -rp 'Fresh confirmation seeds (example 23,37,53): ' SEEDS
read -rp 'Contamination rates (example 0.05,0.25), or none: ' RATES
read -rp 'Reason for the selected settings, based on calibration: ' DECISION

python scripts/export_confirmation.py \
  --project "$PROJECT" --model qwen \
  --output selected_confirmation.json \
  --max-new-tokens "$TOKEN_CAP" --max-updates "$UPDATES" \
  --seeds "$SEEDS" --contamination-rates "$RATES" --note "$DECISION"
```

Use `--set train.lora.r=... train.lora.alpha=... eval.gate_min_completion=...` for **explicitly justified** further choices. Export refuses to overwrite an earlier decision; choose a new output filename. A changed training strength is recorded as unverified until run. Re-export rather than editing selected gate/training fields silently. `configs/h100.json` is now an **unselected template** and intentionally refuses direct initialization without a calibration decision.

Changing thresholds only to cross a significance boundary is not valid calibration. Preserve scientific effect/quality requirements and freeze them before examining confirmatory outcomes.

Two rates and two objectives add four primary-family adapters at one specified seed. These are label-flip rates **among eligible conflict pairs**, not percentages of all training pairs. The code logs actual flipped counts. Low-dose data alone do not establish an early-warning claim.

## 6. Initialize the separate real project

```bash
export CALIBRATION_PROJECT="$PROJECT"
export PROJECT="$PWD/runs/confirmation_$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/run.sh --init --project "$PROJECT" --config selected_confirmation.json \
  --data-from "$CALIBRATION_PROJECT" --billing-from "$CALIBRATION_PROJECT"

bash scripts/prepare.sh
read -rp 'Path to the reviewed semantic audit CSV: ' REVIEWED_CSV
bash scripts/run.sh --project "$PROJECT" --audit-csv "$REVIEWED_CSV"

bash scripts/pilot.sh qwen
bash scripts/confirm.sh qwen
bash scripts/pilot.sh llama
bash scripts/confirm.sh llama
```

This is the freeze point. The final project's first-seed gate is now confirmatory—not another opportunity to tune to its result. Failed outcomes remain part of the evidence record. Its first-seed adapters are reused within that project's full phase. Calibration weights are never counted as confirmatory runs.

Copying validated calibration data avoids another writer/validator expense, but **human audit approval is still required once for the final dataset**. The test partition has not been exposed by the calibration command paths. Model-specific completion headroom can be checked by also running `calibrate.sh llama` before selecting a cross-family protocol; a Qwen-only calibration is not evidence that Llama completes at the same rate.

## 7. Mechanisms and reports

```bash
bash scripts/mechanisms.sh qwen
bash scripts/mechanisms.sh llama
bash scripts/run.sh --project "$PROJECT" --model qwen --phase full --stages features,report
bash scripts/run.sh --project "$PROJECT" --model llama --phase full --stages features,report
bash scripts/run.sh --project "$PROJECT" --status
```

The previously retained method scope is unchanged; see `docs/METHODS.md` and `docs/RUNBOOK.md`. Run expensive sweeps in stages and inspect actual remaining budget. A diagnostic signal is not proof of a gaming mechanism unless behavioral and causal validation support it.

## Existing initialized v3 project: callback-only repair

Keep using the **old v3 package** for that project. Do not point this new package at it. Stop the job, back up the project, then run the new helper in dry-run mode:

```bash
python scripts/repair_initialized_v3.py \
  --project /absolute/path/to/existing_project \
  --package /absolute/path/to/old/gamingmech_h100_v3
```

Add `--apply` after inspecting the plan. It permits only the released v3 source plus the single callback correction. It backs up the old source/protocol/checksums, logs the amendment, preserves unaffected training/evaluation receipts, and archives steering outputs/receipts that must rerun. The source and distribution checksums are updated consistently. Other source changes are rejected. This is **not** a way to change frozen thresholds or migrate old results to the calibration design.

## Cost boundary

Budget defaults remain $750 with $75 reserve, estimated from the actual rate/start you enter. Calibration and confirmation on the same instance can share a copied billing history so earlier spending is not forgotten. A new trial is not a new $750 allowance. Cooperative checks cannot interrupt every CUDA operation or shut down the cloud instance. **You must stop provider billing yourself.** No H100 throughput, total cost, acceptance rate, or induction success is asserted as measured here.
