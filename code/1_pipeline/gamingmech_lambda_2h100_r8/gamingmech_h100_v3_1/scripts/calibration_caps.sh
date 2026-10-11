#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT:?Set PROJECT to the calibration project}"
MODEL="${1:?Supply model key, qwen or llama}"
CAPS="${2:?Supply caps, e.g. 256,512,1024}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "$ROOT/scripts/run.sh" --project "$PROJECT" --model "$MODEL" --phase pilot --calibration-caps "$CAPS"
