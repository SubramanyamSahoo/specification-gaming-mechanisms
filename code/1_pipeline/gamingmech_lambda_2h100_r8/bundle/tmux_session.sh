#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
python3 -u bundle/console.py
rc=$?
printf '\nConsole exited with status %s. Provider billing is STILL running.\n' "$rc"
printf 'Restart this console: python3 -u bundle/console.py\n'
# Keep the pane available for inspection instead of erasing the final error.
exec bash --noprofile --norc -i
