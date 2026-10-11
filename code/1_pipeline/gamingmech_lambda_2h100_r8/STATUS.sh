#!/usr/bin/env bash
# Where each family is, its last log lines, gate results so far, GPUs and running jobs. Changes nothing.
R8="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
for m in qwen llama; do
  log="$R8/work/v3_$m.log"
  echo; echo "== $m"
  if [ -f "$log" ]; then
    grep -a "^===== $m" "$log" | tail -n 1
    grep -a "gate passed=\|ERROR\|not transferred\|Traceback\|Error:" "$log" | tail -n 8 | cut -c1-170
    echo "   last line: $(tail -n 1 "$log" | tr '\r' '\n' | tail -n 1 | cut -c1-150)"
  else echo "   (not started)"; fi
done
echo; echo "== GPUs (index, busy %, memory)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"
echo; echo "== Jobs running now"
pgrep -af "gamingmech.run" | while read -r _pid rest; do
  project=$(grep -o -- "--project [^ ]*" <<<"$rest" | cut -d' ' -f2); stages=$(grep -o -- "--stages [^ ]*" <<<"$rest" | cut -d' ' -f2)
  echo "  $(basename "$project"): ${stages:-setup}"
done | sort -u
pgrep -f "gamingmech.run" >/dev/null || echo "  none"
