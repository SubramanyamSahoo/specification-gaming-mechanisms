#!/usr/bin/env bash
# Call AFTER init + billing clock. Produces data and review CSV, not training.
set -euo pipefail
: "${PROJECT:?Set PROJECT to the absolute initialized project path}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "$ROOT/scripts/run.sh" --project "$PROJECT" --phase pilot --preflight
bash "$ROOT/scripts/run.sh" --project "$PROJECT" --model qwen --phase pilot --stages data,validate
printf '\nReview %s/01_data/tables/semantic_audit_to_review.csv before training.\n' "$PROJECT"
