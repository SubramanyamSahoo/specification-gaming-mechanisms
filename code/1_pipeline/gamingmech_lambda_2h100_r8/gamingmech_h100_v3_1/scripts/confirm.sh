#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT:?Set PROJECT to the absolute initialized project path}"
MODEL="${1:-qwen}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "$ROOT/scripts/run.sh" --project "$PROJECT" --model "$MODEL" --phase full --stages train,behavior,test_behavior,activations,report
