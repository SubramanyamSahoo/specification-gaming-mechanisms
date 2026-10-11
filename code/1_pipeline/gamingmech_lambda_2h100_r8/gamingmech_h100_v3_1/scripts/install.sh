#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python - <<'PY'
import sys, torch
assert sys.version_info >= (3,10), 'Python >=3.10 is required'
assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])) >= (2,5), 'Use a CUDA PyTorch >=2.5 image'
assert torch.cuda.is_available(), 'Choose a CUDA-enabled H100 PyTorch image first; this installer does not replace PyTorch'
print('Preserving PyTorch:',torch.__version__,'CUDA:',torch.version.cuda)
PY
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
python -m pytest -q tests
# Missing architecture dependencies MUST fail installation; a skipped test is not certification.
python -m gamingmech.integration
python -m pip freeze > installed_packages.txt
printf '\nInstallation and tiny architecture checks passed. Run project preflight before any 8B job.\n'
