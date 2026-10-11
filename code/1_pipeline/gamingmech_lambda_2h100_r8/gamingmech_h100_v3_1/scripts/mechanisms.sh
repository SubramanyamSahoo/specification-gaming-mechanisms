#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT:?Set PROJECT to the absolute initialized project path}"
MODEL="${1:-qwen}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Staged rather than launching all optional sweeps at once. Stops on error/reserve/gate.
bash "$ROOT/scripts/run.sh" --project "$PROJECT" --model "$MODEL" --phase full --stages steering,patching,dissect,detection,report
