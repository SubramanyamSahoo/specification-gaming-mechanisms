#!/usr/bin/env bash
set -euo pipefail
set +x
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${TMUX:-}" ]]; then
  exec bash "$ROOT/RUN_ALL.sh" "$@"
fi
command -v tmux >/dev/null || { echo 'STOP: tmux is not installed. No job started.'; exit 2; }
# Never kill an existing session; -A attaches to the existing one.
exec tmux new-session -A -s llama_r9_all -c "$ROOT" bash "$ROOT/SESSION.sh" "$@"
