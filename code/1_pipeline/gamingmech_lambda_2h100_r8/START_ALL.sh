#!/usr/bin/env bash
# The only command to type. Asks for the Hugging Face token once, then runs both families in the background,
# from calibration to the last extra, with nothing else to type:
#   Qwen3.5 9B on GPU 0, Llama 3.1 8B on GPU 1   (RUN_FAMILY.sh)
# The token is kept in memory only. Progress at any time:  bash STATUS.sh
set -euo pipefail
R8="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
command -v tmux >/dev/null || { echo "tmux missing: sudo apt-get install -y tmux" >&2; exit 2; }
if pgrep -f "parallel.py|RUN_QWEN35.sh|RUN_LLAMA_V2.sh" >/dev/null; then
  echo "An older r6 or r7 job is still running. Send Claude this:"; pgrep -af "parallel.py|RUN_QWEN35.sh|RUN_LLAMA_V2.sh"; exit 1
fi
read -rsp "Paste your Hugging Face token and press Enter (it stays hidden): " T; echo
[ -n "$T" ] || { echo "No token given; nothing started." >&2; exit 2; }

launch() {  # session, command; the token reaches the new session through tmux memory only, never a file
  local cmd="$2; echo; echo 'Finished. This window stays open.'; exec bash"
  tmux kill-session -t "$1" 2>/dev/null || true
  if ! tmux new-session -d -s "$1" -c "$R8" -e "HF_TOKEN=$T" "$cmd" 2>/dev/null; then      # tmux older than 3.2
    tmux set-environment -g HF_TOKEN "$T"
    tmux new-session -d -s "$1" -c "$R8" "$cmd"
    tmux set-environment -gu HF_TOKEN
  fi
}

mkdir -p "$R8/work"
# A running tmux server keeps the arguments it was started with; start it without the token in them.
tmux has-session 2>/dev/null || tmux new-session -d -s gm_idle
for m in qwen llama; do
  if pgrep -f "RUN_FAMILY.sh $m|work/v3/[a-z]*_$m" >/dev/null; then echo "$m is already running; left as it is."
  else launch "${m}_run" "bash '$R8/RUN_FAMILY.sh' $m 2>&1 | tee -a '$R8/work/v3_$m.log'"; echo "$m started."; fi
done
echo "Nothing else to do. You can close the laptop. Check later with:  bash $R8/STATUS.sh"
