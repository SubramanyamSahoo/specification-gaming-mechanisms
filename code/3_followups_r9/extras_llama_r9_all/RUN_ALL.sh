#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
R8="${R8_ROOT:-$HOME/gamingmech_lambda_2h100_r8}"
PY="$R8/gamingmech_h100_v3_1/.venv/bin/python"
[[ -x "$PY" ]] || { echo "STOP: existing r8 environment not found: $PY"; exit 2; }
[[ -n "${TMUX:-}" ]] || { echo 'Use bash START_ALL.sh to start under tmux.'; exit 2; }
export USE_TORCH=1 USE_TF=0 PYTHONUNBUFFERED=1
exec "$PY" -u "$ROOT/coordinator.py" run "$@"
