#!/usr/bin/env bash
set -uo pipefail
set +x
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
bash "$ROOT/RUN_ALL.sh" "$@"
CODE=$?
printf '\nLauncher exit status: %s. Read STATUS.sh; provider billing continues.\n' "$CODE"
printf 'Resume in this shell with: bash %q\n' "$ROOT/RUN_ALL.sh"
exec bash
