#!/usr/bin/env bash
# Usage: bash scripts/tmux_run.sh session-name --project /absolute/path --model qwen --phase pilot --stages train,behavior
set -euo pipefail
SESSION="${1:?Supply tmux session name}"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
command -v tmux >/dev/null || { echo 'Install tmux on the host first.' >&2; exit 1; }
# tmux server may predate the shell that exported credentials. Update only the named environment field.
if [[ -n "${HF_TOKEN:-}" ]]; then tmux set-environment -g HF_TOKEN "$HF_TOKEN" 2>/dev/null || true; fi
printf -v CMD '%q ' bash "$ROOT/scripts/run.sh" "$@"
tmux new-session -d -s "$SESSION" "$CMD"
printf 'Attach with: tmux attach -t %q\n' "$SESSION"
