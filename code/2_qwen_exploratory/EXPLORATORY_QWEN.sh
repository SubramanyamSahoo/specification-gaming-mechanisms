#!/usr/bin/env bash
# Disclosed exploratory run for Qwen3.5 on GPU 0 (see EXPLORATORY_QWEN.py). Copies the project once, then runs.
set -euo pipefail
R8="${R8:-$HOME/gamingmech_lambda_2h100_r8}"; P="$R8/gamingmech_h100_v3_1"
SRC="$R8/work/v3/confirmation_qwen"; DST="$R8/work/v3/exploratory_qwen"
: "${HF_TOKEN:?HF_TOKEN missing}"
if [ ! -f "$DST/protocol.json" ]; then echo "copying the Qwen project ($(du -sh "$SRC" | cut -f1))"; rm -rf "$DST"; cp -a "$SRC" "$DST"; fi
cd "$P"
export CUDA_VISIBLE_DEVICES="${QWEN_GPU:-0}" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false USE_TORCH=1 USE_TF=0 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -u -c "import sys; sys.path.insert(0, sys.argv[1]); from EXPLORATORY_QWEN import main; main(sys.argv[2], 'qwen')" "$R8" "$DST"
