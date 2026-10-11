#!/usr/bin/env bash
# Read-only final summaries run even after a failed quota or integration check.
set -euo pipefail
: "${PROJECT:?Set PROJECT to an initialized CALIBRATION project}"
MODEL="${1:-qwen}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
summarize() {
  status=$?
  trap - EXIT
  bash "$ROOT/scripts/run.sh" --project "$PROJECT" --model "$MODEL" --phase pilot --calibration-summary || true
  exit "$status"
}
trap summarize EXIT
bash "$ROOT/scripts/run.sh" --project "$PROJECT" --model "$MODEL" --phase pilot --preflight
bash "$ROOT/scripts/run.sh" --project "$PROJECT" --model "$MODEL" --phase pilot \
  --stages data,validate,profile,train,behavior,calibration_summary,report
