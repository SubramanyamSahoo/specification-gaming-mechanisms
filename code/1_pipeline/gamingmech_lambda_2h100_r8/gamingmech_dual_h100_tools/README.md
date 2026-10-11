# External dual-H100 launch tools for GamingMech v3.1

These tools run **one independent model process per GPU**, not DDP/tensor parallelism. They add no scientific method and remove none. They do not change any `gamingmech/*.py` file, source lock, or existing protocol. Keep this directory OUTSIDE the v3.1 package and initialized projects. You still need the original v3.1 ZIP installed on the H100 host.

## Validation boundary

20 helper CPU unit tests passed. Python compilation and actual v3.1 `from_dict`/`validate_config` accepted the generated 16-pair candidate. These are NOT 8B-model tests or a two-H100 throughput benchmark. No claim is made about the best batch size, total cost, or successful induction. Your installed v3.1 preflight, profiler, and behavioral gates remain required.

## 1. Paths and authentication

From the original package directory after installing it:

```bash
export PACKAGE="$PWD"
# Unzip the addon alongside (NOT inside) PACKAGE. Set its actual path:
export TOOLS="$(dirname "$PACKAGE")/gamingmech_dual_h100_tools"
source "$PACKAGE/.venv/bin/activate"
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
export HF_HOME="$HOME/.cache/huggingface"
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv
```

`HF_TOKEN` must have Llama access. Neither tool copies the token into its metadata. GPU UUIDs are used for worker visibility, avoiding CUDA ordinal ambiguity. Each worker sees one device as local `cuda:0`.

## 2. Prepare a calibration configuration BEFORE initialization

The example below proposes **192** matched optimizer updates and a **512-token** output cap as calibration candidates. Neither value is established as sufficient. Omit `--updates` and `--output-tokens` to retain the supplied scientific defaults. Keep data acceptance and quality thresholds unchanged.

```bash
export RUNS="$(dirname "$PACKAGE")/dual_runs_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUNS"
python "$TOOLS/make_calibration_config.py" \
  --base "$PACKAGE/configs/calibration.json" \
  --output "$RUNS/calibration_16pairs.json" \
  --microbatch 16 --updates 192 --output-tokens 512 \
  --generation-budget 24576
```

The helper changes microbatch8/accumulation8 to microbatch16/accumulation4: nominal64 pairs/update. It expands the padded token allowance to at least `2*16*768=24576` and keeps the sorting pool at400 items by changing `sort_window` from50 to25. It does not change LR, beta, rank, writer settings, seed counts, data, sample counts, or methods. Actual update membership can still differ from v3.1's previous batches; it is a new calibration protocol, not bitwise-equivalent continuation. It refuses incompatible batch choices and refuses confirmation-template editing. Every accepted comparison has two sequences.

At the original12000 training-token allowance, a full768-token pair allows only `floor(12000/(2*768))=7` pairs per microbatch, regardless of a higher pair-count ceiling. This is why the token allowance is adjusted as well. Higher token allowances raise peak memory; they are not certified H100-fit settings.

The generation batch already has ceiling64 in v3.1. The token budget bounds actual batch size; increasing the ceiling alone may do nothing. Writer, reference, and other high-memory causal batches are deliberately unchanged until profiled. The addon does not implement DataLoader processes: v3.1 pretokenizes its in-memory examples and does not consume `runtime.num_workers` anywhere beyond declaring it.

## 3. Build the common data ONCE

```bash
export CAL_Q="$RUNS/calibration_qwen"
export CAL_L="$RUNS/calibration_llama"
CUDA_VISIBLE_DEVICES=0 bash "$PACKAGE/scripts/run.sh" --init \
  --project "$CAL_Q" --config "$RUNS/calibration_16pairs.json"
read -rp 'TOTAL two-GPU instance price USD/hour: ' RATE
read -rp 'ACTUAL instance billing start, ISO timestamp with timezone: ' BILLING_START
CUDA_VISIBLE_DEVICES=0 bash "$PACKAGE/scripts/run.sh" --project "$CAL_Q" \
  --phase pilot --clock --hourly-usd "$RATE" --started-utc "$BILLING_START"
CUDA_VISIBLE_DEVICES=0 PROJECT="$CAL_Q" bash "$PACKAGE/scripts/prepare.sh"
```

Preparation is sequential in the original package: Qwen writes, then the other-family frozen validator checks the same candidates. A second GPU does NOT automatically pipeline these stages. Do not run two writers against the shared project or bypass quotas. The calibration role alone waives the human audit; automatic validation is retained.

If you already have a compatible validated calibration dataset from this SAME instance, use v3.1's `--data-from`/`--billing-from` instead of regenerating or resetting its bill. If you changed writer/data/split/audit-sampling settings, data import intentionally fails. For a NEW instance, record its true billing segments and previous study spend instead of inheriting an old open clock as though it were still the same instance.

## 4. Create the second project with identical data and billing

```bash
CUDA_VISIBLE_DEVICES=1 bash "$PACKAGE/scripts/run.sh" --init \
  --project "$CAL_L" --config "$RUNS/calibration_16pairs.json" \
  --data-from "$CAL_Q" --billing-from "$CAL_Q"
CUDA_VISIBLE_DEVICES=1 bash "$PACKAGE/scripts/run.sh" --project "$CAL_L" \
  --model llama --phase pilot --preflight
```

Both project clocks reflect the total price of the SAME instance from the SAME actual start, including time spent preparing data. The two spending reports are duplicate views, not two budgets to add or two allowances. Do not edit/pause just one ledger while the instance is running. Provider billing continues until you stop the instance yourself.

## 5. Profile BOTH GPUs before training

```bash
python "$TOOLS/parallel.py" --package "$PACKAGE" \
  --qwen-project "$CAL_Q" --llama-project "$CAL_L" \
  --stage profile --threads-per-job 8 --logs "$RUNS/launch_logs"
```

The default worker count is exactly TWO model processes. CPU intra-op/BLAS threads are genuinely set to8/job (or, if the option is omitted, min(8, available logical CPUs/2)). Inter-op threads1; tokenizer parallelism remains disabled. More processes per GPU are not added. These are separate CPU/GPU mechanisms, not interchangeable worker knobs.

Inspect both `MODEL/02_profile/pilot/tables/training_profile.csv` files, including peak allocated/reserved memory and sequence tokens/s. v3.1's profile deliberately tries the largest retained padded-token batch and may fail with OOM rather than silently change the frozen protocol. If16 fails, create another calibration candidate/project with microbatch8 and import validated data and billing. Do not edit initialized protocols or trust a throughput estimate from a failed batch. A16 profile passing does not certify all optional sweeps; their separate smaller batch limits remain.

## 6. Run the pilots simultaneously

```bash
python "$TOOLS/parallel.py" --package "$PACKAGE" \
  --qwen-project "$CAL_Q" --llama-project "$CAL_L" \
  --stage pilot --threads-per-job 8 --logs "$RUNS/launch_logs"
```

Run this foreground command inside tmux so a dropped SSH connection does not close the shell. The launcher prints unique log paths with `qwen.log`, `llama.log`, `launch.json`, and `exit_status.json`. Existing per-project scientific outputs remain in their original unique projects.

It verifies identical scientific fingerprints, source locks, dataset files, and billing ledgers. It rejects shared/nested project paths, absent preflights, missing semantic certificates, and missing confirmation audit markers. The ORIGINAL worker checks also verify source/data/gate identities. On a worker's execution failure, the launcher stops its sibling to protect the common budget. This preserves completed results but can lose up to the last durable checkpoint's work on the interrupted sibling. Rerun the original command for package-level resume; it is not token-exact generation resume. A measured negative CALIBRATION gate is diagnostic and does not become an execution failure or a successful induction result.

To print commands/check the plan first, append `--dry-run`. It runs no model but still needs valid prepared projects, authentication environment, original package integrity, and `nvidia-smi`.

Measure actual exposure (read-only):

```bash
python "$TOOLS/exposure_report.py" --project "$CAL_Q" --model qwen
python "$TOOLS/exposure_report.py" --project "$CAL_L" --model llama
```

Never decide adequacy from the number of GPUs, loss alone, or matching an old step count. At nominal64 pairs/update,192 updates is12288 pair presentations; divide the ACTUAL logged presentations by the retained training size for equivalent passes. Repeated passes are not new independent data. Assess held-out scenario effects, final-answer completion, answer quality, capabilities, and overfitting. Keep all conditions' scientific settings matched. A failed gate is not repaired by selectively extending only the gaming condition.

## 7. Freeze and initialize two confirmation projects

Inspect BOTH calibration families, then export one reviewed protocol using the original exporter. Example with values selected from your actual development runs:

```bash
read -rp 'Selected evaluated output cap: ' TOKEN_CAP
read -rp 'Selected optimizer updates: ' UPDATES
read -rp 'Fresh confirmatory seeds, e.g.23,37,53: ' SEEDS
read -rp 'Contamination rates or none: ' RATES
read -rp 'Rationale based on BOTH family calibration reports: ' DECISION
python "$PACKAGE/scripts/export_confirmation.py" --project "$CAL_Q" --model qwen \
  --output "$RUNS/selected_confirmation.json" --max-new-tokens "$TOKEN_CAP" \
  --max-updates "$UPDATES" --seeds "$SEEDS" --contamination-rates "$RATES" --note "$DECISION"
export CONF_Q="$RUNS/confirmation_qwen"
export CONF_L="$RUNS/confirmation_llama"
for P in "$CONF_Q" "$CONF_L"; do
  bash "$PACKAGE/scripts/run.sh" --init --project "$P" \
    --config "$RUNS/selected_confirmation.json" --data-from "$CAL_Q" --billing-from "$CAL_Q"
done
CUDA_VISIBLE_DEVICES=0 bash "$PACKAGE/scripts/run.sh" --project "$CONF_Q" --model qwen --phase pilot --preflight
CUDA_VISIBLE_DEVICES=1 bash "$PACKAGE/scripts/run.sh" --project "$CONF_L" --model llama --phase pilot --preflight
read -rp 'Path to the reviewed semantic-audit CSV: ' REVIEWED_CSV
for P in "$CONF_Q" "$CONF_L"; do
  bash "$PACKAGE/scripts/run.sh" --project "$P" --phase pilot --audit-csv "$REVIEWED_CSV"
done
```

Human review is required for confirmation. Reusing reviewed rows is valid only because the dataset and audit sampling are identical; the original registration checks verify that. Calibration adapters are NOT confirmation runs. Copying data from a confirmation parent is not supported by v3.1's handoff: use the original calibration parent as shown.

Run one confirmatory seed per family, then remaining seeds and held-out analyses:

```bash
python "$TOOLS/parallel.py" --package "$PACKAGE" --qwen-project "$CONF_Q" --llama-project "$CONF_L" \
  --stage pilot --threads-per-job 8 --logs "$RUNS/launch_logs"
python "$TOOLS/parallel.py" --package "$PACKAGE" --qwen-project "$CONF_Q" --llama-project "$CONF_L" \
  --stage confirm --threads-per-job 8 --logs "$RUNS/launch_logs"
python "$TOOLS/parallel.py" --package "$PACKAGE" --qwen-project "$CONF_Q" --llama-project "$CONF_L" \
  --stage mechanisms --threads-per-job 8 --logs "$RUNS/launch_logs"
```

`--stage features`, `--stage minimal`, and `--stage trajectory` are explicit later choices. All original methods and checks are retained; this addon adds no experimental method. No automatic retraining until significant, no lower thresholds, no removed seeds, no diminished evaluations. If a family completes earlier, its GPU is idle until another independent job is scheduled; the original package does not split the remaining adapter automatically.

## Already initialized projects

The parallel launcher can use two compatible initialized projects with their EXISTING settings; it never edits them. Faster batching changes must be calibrated before creating a new protocol. If a source repair is needed, preserve the original provenance; do not regenerate a project's lock merely to suppress an integrity error. This external toolkit requires no changes to `PACKAGE_SHA256.json` or source locks.

## Technical sources and limits

CUDA visibility/remapping: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
CPU intra-op setting: https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html
DataLoader tuning: https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html

Inspected source: uploaded `gamingmech_h100_v3_1.zip`, SHA256 is recorded in `VALIDATION.json`. `runtime.num_workers` occurs only in its configuration declaration; no DataLoader uses it. This is why the toolkit configures actual CPU threads rather than claiming a larger unused value makes training faster. DDP would require a separate implementation and validation; do not prepend torchrun to this launcher.
