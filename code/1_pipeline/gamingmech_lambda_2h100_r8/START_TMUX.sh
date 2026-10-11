#!/usr/bin/env bash
# Start or attach to this bundle's private tmux session. No credentials in commands.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SESSION="${GAMING_TMUX_SESSION:-gamingmech}"
[[ "$SESSION" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid session name." >&2; exit 2; }
command -v python3 >/dev/null || { echo "Python 3 is required." >&2; exit 2; }
python3 "$ROOT/verify_bundle.py"
if [[ -n "${TMUX:-}" ]]; then
  echo "Already inside tmux; opening the console here. Detach with Ctrl-B then D."
  exec python3 -u "$ROOT/bundle/console.py"
fi
command -v tmux >/dev/null || {
  printf 'tmux is missing. Install it, then rerun:\n  sudo apt-get update && sudo apt-get install -y tmux\n' >&2
  exit 2
}
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  BOUND="$(tmux show-options -t "=$SESSION:" -qv @gamingmech_bundle 2>/dev/null || true)"
  [[ "$BOUND" == "$ROOT" ]] || {
    echo "Session $SESSION belongs to another directory. Use GAMING_TMUX_SESSION=another_name." >&2
    exit 2
  }
  exec tmux attach-session -t "=$SESSION"
fi
printf -v CMD 'bash %q' "$ROOT/bundle/tmux_session.sh"
tmux new-session -d -s "$SESSION" -n control -c "$ROOT" "$CMD"
tmux set-option -t "=$SESSION:" @gamingmech_bundle "$ROOT"
tmux set-option -t "=$SESSION:" history-limit 20000
printf 'Session: %s\nDetach: Ctrl-B then D\nReattach: tmux attach -t %s\n' "$SESSION" "$SESSION"
exec tmux attach-session -t "=$SESSION"
