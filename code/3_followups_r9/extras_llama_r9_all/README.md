# Llama r9 — all six extras in one upload

This add-on reuses the **completed r8 Llama confirmation project**. It is not a new
main training package. It does not change the original behavior gates, edit the
core Python package, rerun calibration, start Qwen, or retry the failed SAE stage.
The six requested experiments are queued sequentially on **physical GPU 1**.

## Upload and run

Upload `extras_llama_r9_all.zip` to `/home/ubuntu/`, then paste:

```bash
cd ~
unzip -n extras_llama_r9_all.zip
cd extras_llama_r9_all
bash START_ALL.sh
```

The launcher opens/attaches to tmux session `llama_r9_all`. If already inside tmux,
it uses the current shell. Supply your valid Hugging Face token at the hidden
prompt; Enter instead uses an existing environment/cached token, which is checked
before any model is loaded. A bad token stops this run; it is never silently
replaced. The existing r8 model loader can persist login under the HF cache; this
coordinator does not put the token in its arguments, run metadata or export ZIP.

Detach with **Ctrl+B, then D**. Reattach using:

```bash
tmux attach-session -t llama_r9_all
```

All six extras run automatically, provided dependencies, runtime checks and the
existing spending allowance permit it. No menu action or second stage command is
required. A runtime failure stops later experiments and records a nonzero exit.
A scientific null/negative result is not converted to a positive finding.

## What runs, in order

| Experiment | Work | New training |
|---|---|---|
| `attention` | Query-to-claim/user/first-token attention for previously selected top and random heads, with coverage records | None |
| `seeds` | Site/head/direction patching, update dissection, restricted-adapter recovery on the non-reference confirmation seeds (normally 79 and 83) | Up to four restricted LoRA adapters |
| `public` | Original r8 PublicEval on the public are-you-sure and answer datasets, for all existing conditions | None |
| `joint_heads` | Simultaneous top-head versus random-head patching for every seed/objective; paired per-item effects and CIs | None |
| `steering` | Sycophancy-direction removal and full update-window removal, with free generation and likelihood measures across all confirmation seeds | None |
| `dose` | Eligible-label-flip fractions 0.10, 0.50, 0.75, alongside original 0.25 and 1.0 conditions and the matched control; monitor ranking | Three sycophancy LoRA adapters at the reference seed |

Dose fractions refer to **eligible conflict preferences**, not fractions of the
whole corpus. Main-model weights and original full adapters are reused, not
retrained. Restricted and dose training retain the original DPO hyperparameters.
No pretrained checkpoint change or pip installation is performed.

## Requirements

By default the already installed package is:

```
~/gamingmech_lambda_2h100_r8/gamingmech_h100_v3_1
```

Its Python is `.venv/bin/python`. The source project is:

```
~/gamingmech_lambda_2h100_r8/work/v3/confirmation_llama
```

The launcher verifies the exact r8 core against `expected_core.json`, the locked
project/gate and completed adapter checksums. It also checks the source directions,
head-patching table, dissection selections and steering selections. Missing source
weights/tensors cannot be replaced by the small `llama_results.zip` export.

The copied data/metadata live in the new extras workspace. Only immutable original
weight/checkpoint directories are referenced through links; mutable caches, logs
and new adapters are local to the extras workspace. Keep the original source
project and the old environment directories on disk. Symlinks are not an
independent backup or OS-enforced read-only protection.

For a non-default r8 location, set `R8_ROOT=/absolute/path/to/r8` before launching.
For a non-default source project, set `LLAMA_SOURCE=/absolute/path/to/confirmation_llama`.
These paths are fixed in the new run's identity. Different inputs must not reuse
old operation caches.

## Status, stop and resume

Status is read-only and does not import model libraries:

```bash
bash ~/extras_llama_r9_all/STATUS.sh
```

Request a stop of **this extras run only**:

```bash
bash ~/extras_llama_r9_all/STOP.sh
```

The supervisor sends an interrupt only to its own child process group. It allows
120 seconds for the original training code to save its latest completed optimizer
state, then escalates termination if needed. This is not a promise that every
possible GPU failure can be checkpointed. No Qwen process/session is signalled.
Wait until status is `stopped` and GPU 1 is free before another launch.

To resume in the same tmux shell:

```bash
bash ~/extras_llama_r9_all/RUN_ALL.sh
```

The original extras identity and settings are checked, completed experiments and
seed substages are reused, and the original DPO checkpoints resume new restricted
or dose adapters. Completed cached attention blocks, interventions, generations,
judge batches and dose activations are reused where available. An interrupted
operation that had not committed its cache may run again. This is not exact
mid-token inference resume. Failures remain in separate attempt logs.

Running `START_ALL.sh` outside tmux attaches to the existing session if it exists;
it does not inject a second launch into an active shell. Use `RUN_ALL.sh` at that
session's prompt to resume a stopped attempt.

## Spending and execution limits

All historical segments in the original `billing.json` are copied. The supervisor
computes total instance spending from the **source ledger**, including idle time,
and checks it every five seconds during work. It respects the original total
budget minus the original reserve, in addition to cooperative core checks.
Changing that ledger during this extras run stops it rather than quietly resetting
past expenditure. Verify that its rate and clock correspond to the actual provider
bill, especially after moving to a new instance.

An optional additional limit can be set on the first invocation, for example:

```bash
bash START_ALL.sh --max-additional-usd 100
```

This is an authorization ceiling, **not an estimated cost for all six experiments**.
It includes wall-clock instance spending since extras initialization and persists
on resume. It never increases the original total allowance. The default has no
extra ceiling beyond the source project's original budget/reserve.

The supervisor cannot stop provider billing. Its polling and graceful-shutdown
intervals can overshoot a limit. The instance and any independent Qwen work keep
running until separately stopped. Back up files before terminating Lambda.

## Outputs

Each new initialization gets a unique directory:

```
~/extras_llama_r9_all/work/llama_r9_<UTC>_<ID>/
  run.json                         # frozen source/code/config/budget identity
  status.json                      # current stage, PID, cost estimate, log
  receipts/<experiment>.json       # completion/failure, not hypothesis approval
  logs/<UTC>_<ID>_<experiment>/     # one log/exit record per attempt
  project/
    RESULTS.html
    RESULTS/attention/
    RESULTS/seeds/checkpoints/
    RESULTS/public/
    RESULTS/joint_heads/
    RESULTS/steering/
    RESULTS/dose/
    llama/06_patching/seed79/...    # analogous seed83 outputs
    llama/07_dissect/seed79/...
    llama/03_train/artifacts/...    # new restricted/dose adapters and checkpoints
```

Every extras plotting directory contains PNG and vector-PDF versions plus a
`figure_index.csv`. The original core plots for seed replication remain in its
per-seed `figures` directories. All numerical tables, raw generated text and token
IDs, IDs/group IDs, option-condition metadata and generation termination flags are
saved. Dose activation tensors, their checksums, monitor scores and selection
indices are saved locally. The original judge definitions remain explicit.

A unique `llama_r9_results_<UTC>_<ID>.zip` is written inside the run directory on
completion or a caught stage failure. Unlike the previous add-on export, it
**includes JSONL raw responses and logs**. It excludes weights, binary tensors,
linked inputs and duplicated operation-cache files; therefore it is **not a full
backup**. `ARCHIVE_SCOPE.json` explains the omission. The full local workspace and
original source project must be retained for reproducing model-side analyses.

Produce another export without model work:

```bash
bash ~/extras_llama_r9_all/PACK_RESULTS.sh
```

## Scientific scope

These are new follow-ups motivated after seeing the main results; do not describe
them as part of the original preregistered confirmation merely because they reuse
the same seeds. This release does not resolve the original behavioral evaluator's
missing-dialogue and switching-definition limitations. Generated-response scores
are retained under `legacy_r8_judge_v1` with all contexts for later independent
annotation. They are not human-adjudicated behavioral rates. Praise and omitted
flaw disclosure remain separate outcomes.

Attention weights are descriptive, not evidence of necessity. Single/joint head
patching and update-window removal remain different interventions. The steering
layer/window is chosen from reference-seed development selections and transferred
to the other seeds; all new seed-specific directions use that fixed layer.
Restricted adapters do not include a new size-matched random restricted adapter.
No new generation-level CAA/random-direction experiment is silently implied.

The dose monitor is **within-reference-seed ranking**. It is not a validated
cross-seed low-false-positive alarm or evidence of prospective early warning.
The inherited bootstrap convention conditions on a fixed trained model; repeated
question groups or aggregated scenarios need group-aware reanalysis for claims
about independent questions. Raw group IDs are preserved in new records.

## Validation

See `validation/VALIDATION.json` and `validation/unit_tests.txt`. Local tests cover
operational invariants and actual helper functions on synthetic CPU fixtures.
No real pretrained Llama/Phi-4 inference, H100 memory/throughput, live tmux launch or
full six-stage pretrained-model execution was performed in the authoring runtime.
There is no promise of bug-free operation or favorable scientific results.

External API/operational references used in the launcher:
- HF token precedence: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
- tmux session commands: https://github.com/tmux/tmux/wiki/Getting-Started
- CUDA device visibility: https://docs.nvidia.com/deploy/topics/topic_5_2_1.html
