# GamingMech r6: upload one ZIP, run under tmux

Read **START_HERE.md** for the workflow, what changed in r6, and the limits.

```bash
unzip gamingmech_lambda_2h100_r6.zip
cd gamingmech_lambda_2h100_r6
bash use_existing_env.sh        # only on a machine where the earlier bundle is installed and working
bash START_TMUX.sh
```

Fresh machine: skip `use_existing_env.sh` and choose console action 1 (install), then 2 (calibration).
Machine with the earlier bundle: run `use_existing_env.sh`, then choose action 2 directly.

Qwen and Llama use separate GPUs and projects. The console pauses after calibration.
Detach: Ctrl-B then D. Reattach: `tmux attach -t gamingmech`.
**tmux and stopped scripts do not stop cloud billing.** Back up before terminating.
No models, datasets, credentials or experiment results are included.
