# Mechanistic Origins of Specification Gaming: code and results

This repository holds the code and the complete results behind the paper *Mechanistic Origins of Specification
Gaming*. It is released anonymously for review.

We train DPO LoRA adapters on rule built preference pairs in which a **gaming** adapter differs from a **matched
control** only in which reply wins when pleasing the user conflicts with the task. Every behavior is measured twice,
in forced choice logits and in free replies scored by an independent judge (Phi-4), across three fresh seeds under a
protocol locked before the confirmatory run. We then locate what training wrote with activation patching, head
attribution, adapter dissection, steering, internal monitors and training trajectories.

* **Llama 3.1 8B Instruct** is the confirmatory family: it passed the prespecified gate in every seed.
* **Qwen3.5 9B** did not pass the gate, so all Qwen mechanisms are disclosed as exploratory.
* **Follow ups (r9)** are six analyses added after the main results, on the same Llama adapters.

No model weights are included. Everything else that the runs produced is here: tables, figures, raw generated
replies with judge scores, run records and logs.

## Layout, in the order things were run

```
code/
  1_pipeline/gamingmech_lambda_2h100_r8/   the locked pipeline: data, training, gate, all mechanism stages
  2_qwen_exploratory/                      runs the mechanism stages on Qwen3.5 9B after its gate failed
  3_followups_r9/extras_llama_r9_all/      the six Llama follow ups
results/
  1_llama_calibration/                     one seed (11) calibration run that fixed the protocol
  2_llama_confirmation/                    confirmatory Llama run, seeds 73, 79, 83 (reference seed 73)
  3_qwen_exploratory/                      Qwen3.5 9B: gate record and exploratory mechanisms
  4_llama_followups_r9/                    follow up tables, figures and run record
  5_raw_archives/                          unmodified run exports with every generated reply
  6_human_audit/                           human labels of the training pair audit
MANIFEST.sha256                            checksum of every file in this repository
```

Results folders 1 to 3 keep the stage layout of the pipeline: `03_train`, `04_behavior`, `05_activations`,
`06_patching`, `07_dissect`, `08_steering`, `09_detection`, `10_trajectory`, `11_features`. Inside a stage,
`tables/` holds the numbers, `figures/` the plots and `artifacts/` the records a stage depends on. `full` is the
confirmatory phase, `pilot` the calibration phase; `dev` and `test` are disjoint question splits.

## Where each result in the paper comes from

All Llama paths are relative to `results/2_llama_confirmation/llama/`.

| Paper result | File |
|---|---|
| Behavior effects against the matched control (main behavior table) | `04_behavior/full/test/tables/logit_effects.csv`, `generation_effects.csv` |
| Means per condition | `04_behavior/full/test/tables/logit_means.csv`, `generation_means.csv` |
| Confirmation gate on development questions | `04_behavior/full/dev/artifacts/gate_full.json`, `04_behavior/full/dev/tables/confirmed_behaviours.csv` |
| Capability checks | `04_behavior/full/dev/tables/capability.csv` |
| Judge against the rule reading, and the audit sample | `04_behavior/full/test/tables/judge_rule_agreement.csv`, `judge_validation_sample.csv` |
| Training curves | `03_train/figures/validation_curves.png`, `03_train/tables/training_summary.csv` |
| Site patching (where gaming is written) | `06_patching/full/tables/site_patching.csv`, `06_patching/full/figures/site_*_from_control_f000_s73.png` |
| Head attribution map and verified heads | `06_patching/full/tables/head_attribution.csv`, `head_patching_verification.csv`, `06_patching/full/figures/head_attribution.png`, `head_verification.png` |
| Patching along the trained direction | `06_patching/full/tables/direction_patching.csv` |
| Adapter dissection and restricted adapters | `07_dissect/full/tables/gamma_scan.csv`, `minimal_adapter_summary.csv`, `07_dissect/full/figures/minimal_adapter.png` |
| Directions, angles between objectives, contrastive direction, seed floor | `05_activations/full/tables/directions.csv`, `cross_objective_angles.csv`, `caa_vs_trained_cosine.csv`, `seed_floor.csv` |
| Steering with the trained direction | `08_steering/full/tables/steering_effects.csv`, `08_steering/full/artifacts/development_selections.json`, `08_steering/full/figures/dose_response_*.png` |
| Internal monitor against behavioral audits | `09_detection/full/tables/detection.csv`, `monitor_layer_selection.csv`, `09_detection/full/figures/auroc.png` |
| Training trajectories | `10_trajectory/full/tables/checkpoint_trajectory.csv`, `10_trajectory/full/figures/trajectory_syco.png`, `trajectory_length.png` |
| Sparse dictionary check | `11_features/full/tables/sae_reconstruction.csv`, `11_features/full/artifacts/feature_gate.json` |

**Qwen3.5 9B appendix:** `results/3_qwen_exploratory/EXPLORATORY.json` records why the gate failed and which stages
ran; the tables follow the same stage layout under `results/3_qwen_exploratory/qwen/`.

**Follow ups**, under `results/4_llama_followups_r9/`:

| Follow up | Folder |
|---|---|
| Where the top heads attend | `project/RESULTS/attention/` |
| Patching, head attribution, dissection and restricted adapters on seeds 79 and 83 | `project/llama/06_patching/seed79`, `seed83`, `project/llama/07_dissect/seed79`, `seed83` |
| Public sycophancy benchmark (are you sure, suggested answers) | `project/RESULTS/public/` |
| Top heads patched together against random heads | `project/RESULTS/joint_heads/` |
| Removing the direction or one update window in free generation | `project/RESULTS/steering/` |
| Dose response at 10, 25, 50, 75 and 100% flipped conflicts, with the monitor | `project/RESULTS/dose/` |
| Run record, settings, receipts | `run.json`, `receipts/`, `status.json` |

`project/RESULTS.html` is a clickable index of every follow up table and figure.

## Raw replies and audits

`results/5_raw_archives/` holds three unmodified archives:

* `raw_replies_llama_qwen.zip`: every generated reply behind the behavior tables of both families, with the
  prompt, the reply, its token IDs, completion status and judge scores (`generations.jsonl`, `steered_generations.jsonl`).
* `llama_followups_r9_full.zip`: the complete follow up export, including 7,770 replies with judge scores, one
  judge record per reply, stage logs and the exact code that ran.
* `development_history.zip`: the records behind the paper's protocol, calibration and development history:
  the Qwen3.5 calibration and the earlier development runs (Llama development seeds and the replaced Qwen3 8B
  family), with tables, figures, run records and replies but no weights. These runs used earlier versions of
  the code and are not part of the confirmatory study.

`results/6_human_audit/` holds the human review of sampled training pairs used to approve the training data.
The judge validation samples (`judge_validation_sample.csv` in each `04_behavior/full/*/tables/` folder) contain
replies with judge scores and an empty `human_label` column for further annotation.

## Reproducing the runs

The code is the code that produced these results, unchanged except where noted below. It was built for one Linux
machine with two H100 80GB GPUs, one model family per GPU. Python 3.10.12, PyTorch 2.8.0 (CUDA 12.8),
Transformers 5.16.1, PEFT 0.21.0, NumPy 1.26.4. You need a Hugging Face token with access to
`meta-llama/Llama-3.1-8B-Instruct`.

1. **Pipeline.** Copy `code/1_pipeline/gamingmech_lambda_2h100_r8` to your home directory and follow its
   `START_HERE.md`. The core package and its runbook are in `gamingmech_h100_v3_1/` (`README.md`,
   `docs/RUNBOOK.md`, `docs/METHODS.md`). The order is install, one seed calibration, export of the confirmation
   protocol, confirmation on seeds 73, 79 and 83, then the mechanism stages. Every stage refuses to run if the
   locked source, configuration, data or gate identity has changed.
2. **Qwen exploratory stages.** Place `EXPLORATORY_QWEN.py` and `EXPLORATORY_QWEN.sh` in the pipeline folder and run
   the shell script. It copies the Qwen confirmation project and runs every mechanism stage on the copy, so the
   official project keeps its failed gate.
3. **Follow ups.** Copy `code/3_followups_r9/extras_llama_r9_all` to your home directory and run `START_ALL.sh`.
   It checks the pipeline code against `expected_core.json`, mirrors the finished Llama project without writing
   to it, and runs the six follow ups in order on one GPU.

All model and dataset revisions are pinned in `results/2_llama_confirmation/revisions.json`.

## Notes

* **One fix during the follow up run.** The first steering attempt would have failed at its final summary because
  three provenance fields began with `judge_`, which the pipeline reads as scores. The fields were renamed while
  steering was running; steering was then restarted and reused its cached generations. The released
  `r9_support.py` is the fixed version; `run.json` records the fix under `hotfixes`.
* **Qwen3.5 9B** failed the prespecified gate because verbosity was not confirmed in seeds 79 and 83, so every
  Qwen mechanism here is exploratory.
* **Sparse dictionaries** failed the fidelity checks in both families, so no feature level claims are made.
* **Judge scores** come from one model and are not human labels.
* **Paths** inside the records (for example `/home/ubuntu/...`) are those of the machine that ran the study.

## Data sources and licences

Code is released under the MIT licence (`LICENSE`). Results and generated replies are released under CC BY-NC 4.0
(`DATA_LICENSE.md`), because they contain text derived from the source datasets below, each of which keeps its own
licence.

* Training and evaluation questions: `allenai/sciq`, `allenai/ai2_arc` (ARC Challenge), `truthfulqa/truthful_qa`
* Public preference pairs: `Intel/orca_dpo_pairs`, `argilla/ultrafeedback-binarized-preferences-cleaned`, `Anthropic/hh-rlhf`
* Capability: `cais/mmlu`
* Public sycophancy benchmark: the `are_you_sure` and `answer` sets of `meg-tong/sycophancy-eval`
* Models: `meta-llama/Llama-3.1-8B-Instruct` (Llama 3.1 Community License), `Qwen/Qwen3.5-9B`,
  judge `microsoft/phi-4`, dictionaries `andyrdt/saes-llama-3.1-8b-instruct` and
  `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50`. Built with Llama.
