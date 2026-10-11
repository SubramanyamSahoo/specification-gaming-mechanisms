#!/usr/bin/env bash
# r8: one model family end to end on its own GPU, nothing to type. START_ALL.sh starts both.
#   bash RUN_FAMILY.sh qwen    Qwen3.5 9B on GPU 0
#   bash RUN_FAMILY.sh llama   Llama 3.1 8B on GPU 1
# Steps: install check, calibration (seed 11), protocol v3 (fixed below), pilot (seed 73), confirmation
# (seeds 73, 79, 83), mechanisms, then features, minimal adapters and thinking trajectories. Stops at a failed gate.
# Needs HF_TOKEN in the environment. Optional: R6_BUNDLE, R7_BUNDLE, AUDIT_CSV.
set -euo pipefail
M="${1:-}"
case "$M" in qwen) GPU=0 ;; llama) GPU=1 ;; *) echo "Use: bash RUN_FAMILY.sh qwen   or   bash RUN_FAMILY.sh llama" >&2; exit 2 ;; esac
R8="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
P="$R8/gamingmech_h100_v3_1"; PY="$P/.venv/bin/python"
R6="${R6_BUNDLE:-$HOME/gamingmech_lambda_2h100_r6}"; R7="${R7_BUNDLE:-$HOME/gamingmech_lambda_2h100_r7}"
W6="$(ls -dt "$R6"/work/study_* 2>/dev/null | head -1 || true)"
W="$R8/work/v3"; CAL="$W/calibration_$M"; CONF="$W/confirmation_$M"; SEL="$W/protocol_v3_$M.json"
AUDIT="${AUDIT_CSV:-$HOME/reviewed_audit_qwen35.csv}"        # your 81 labels on the r7 and r8 data
# Protocol v3, fixed on 10 October before any r8 run, from the r6 Llama and r7 Qwen3.5 evidence:
UPDATES="qwen=96,llama=96"        # Qwen3.5 at 192 lost too much held out text likelihood, its control included
SEEDS="73,79,83"                  # fresh: none of them has been trained or looked at
GATE="plan.primary_model=llama eval.gate_min_valid_judgments=0.9 eval.gate_min_generation_effect=0.05 eval.gate_max_competence_drop=0.1"
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
: "${HF_TOKEN:?HF_TOKEN missing; start with: bash START_ALL.sh}"
[ -n "$W6" ] || { echo "No r6 study under $R6/work (its billing clock and dataset revisions are reused)" >&2; exit 2; }
mkdir -p "$W"
say() { echo; echo "===== $M: $* ($(date '+%d %b %H:%M'))"; }
run() { bash "$P/scripts/run.sh" --model "$M" "$@"; }

install() {   # one family at a time; reuses the working r7 environment when it is there
  exec 9>"$R8/work/.install.lock"; flock 9
  local marker="$R8/work/installation_passed.json" manifest="$P/PACKAGE_SHA256.json"
  if [ -f "$marker" ] && [ -x "$PY" ] && grep -q "$(sha256sum "$manifest" | cut -d' ' -f1)" "$marker"; then exec 9>&-; return; fi
  say "install check"
  if [ ! -e "$P/.venv" ] && [ -x "$R7/gamingmech_h100_v3_1/.venv/bin/python" ]; then ln -s "$R7/gamingmech_h100_v3_1/.venv" "$P/.venv"; fi
  if [ -x "$PY" ]; then (cd "$P" && "$PY" -m pip install -q -r requirements.txt); else bash "$P/scripts/install.sh"; fi
  (cd "$P" && "$PY" -c "import gamingmech, os, sys; d = os.path.dirname(gamingmech.__file__); sys.exit(0 if d.startswith(os.getcwd()) else 'an older package was imported: ' + d)")
  (cd "$P" && "$PY" -m pytest -q tests -p no:cacheprovider && "$PY" -m gamingmech.integration >/dev/null)
  (cd "$R8" && "$PY" verify_bundle.py && "$PY" -m unittest discover -s tests -p 'test_*.py' \
     && "$PY" -m unittest discover -s gamingmech_dual_h100_tools -p test_tools.py)
  "$PY" -c "import datetime, hashlib, json, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'core_manifest_sha256': hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()}, indent=2))" "$marker" "$manifest"
  exec 9>&-
}

pin_revisions() {   # the same model, dataset, judge and dictionary revisions as r7 and r6
  "$PY" - "$1/revisions.json" "$R7/work/qwen35/calibration_qwen/revisions.json" "$W6/calibration_qwen/revisions.json" <<'PY'
import json, pathlib, sys
dst = pathlib.Path(sys.argv[1]); d = json.loads(dst.read_text()) if dst.exists() else {}
for src in sys.argv[2:]:
    p = pathlib.Path(src)
    for k, v in (json.loads(p.read_text()) if p.exists() else {}).items():
        if not k.startswith("model:Qwen/Qwen3-"): d.setdefault(k, v)
dst.write_text(json.dumps(d, indent=2)); print("revisions pinned:", len(d))
PY
}

show() {   # gate, confirmed behaviours and effects of one project and phase
  "$PY" - "$1/$M/04_behavior/$2/dev" "$M" "$2" <<'PY'
import csv, json, pathlib, sys
dev, m, phase = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
g = dev / "artifacts" / f"gate_{phase}.json"
if g.exists():
    g = json.loads(g.read_text())
    print(f"\n== {m} {phase}: gate passed={g.get('passed')}  confirmed={g.get('confirmed_behaviours')}")
    for e in g.get("errors", []): print("   ERROR", e)
    for e in g.get("unconfirmed", []): print("   not transferred:", e)
for t in ("logit_effects.csv", "generation_effects.csv"):
    p = dev / "tables" / t
    if p.exists():
        for r in csv.DictReader(p.open()):
            if r["objective"] in ("syco", "length") and r["reference"] != "base":
                print(f"   {r['condition']:<16} {r['metric']:<40} {float(r['diff_mean']):+.3f} [{float(r['diff_lo']):+.3f}, {float(r['diff_hi']):+.3f}] n={r['n']}")
PY
}

install

say "calibration (seed 11, $UPDATES updates)"
cp "$R8/configs/dual_calibration.json" "$W/calibration.json"
[ -f "$CAL/protocol.json" ] || run --project "$CAL" --phase pilot --init --config "$W/calibration.json"
if [ ! -f "$CAL/billing.json" ]; then                    # the same instance clock as r6
  read -r RATE START < <("$PY" -c "import json,sys; b=json.load(open(sys.argv[1])); print(b['hourly_usd'], b['started_utc'])" "$W6/calibration_qwen/billing.json")
  run --project "$CAL" --phase pilot --clock --hourly-usd "$RATE" --started-utc "$START"
fi
pin_revisions "$CAL"
run --project "$CAL" --phase pilot --preflight
run --project "$CAL" --phase pilot --stages data,validate
run --project "$CAL" --phase pilot --stages profile,report
run --project "$CAL" --phase pilot --stages train,behavior,report
run --project "$CAL" --phase pilot --calibration-summary
show "$CAL" pilot

say "protocol v3"
[ -f "$SEL" ] || "$PY" "$P/scripts/export_confirmation.py" --project "$CAL" --model "$M" --output "$SEL" \
  --max-new-tokens 1024 --max-updates "$UPDATES" --seeds "$SEEDS" --contamination-rates 0.25 \
  --note "Protocol v3, fixed before any r8 run: each behaviour confirmed on its own (logit and generated evidence, every seed); Qwen3.5 96 updates because 192 broke held out text likelihood in r7; fresh seeds 73,79,83; gate settings as protocol v2." \
  --set $GATE
[ -f "$CONF/protocol.json" ] || run --project "$CONF" --phase pilot --init --config "$SEL" --data-from "$CAL" --billing-from "$CAL"
run --project "$CONF" --phase pilot --preflight
run --project "$CONF" --phase pilot --audit-csv "$AUDIT" \
  || { echo "Audit not accepted: the data differ from r7. Send Claude this message."; exit 1; }

say "pilot (seed ${SEEDS%%,*})"
run --project "$CONF" --phase pilot --stages train,behavior,report || { show "$CONF" pilot; say "PILOT GATE FAILED, stopped"; exit 1; }
show "$CONF" pilot
say "confirmation (seeds $SEEDS)"
run --project "$CONF" --phase full --stages train,behavior,test_behavior,activations,report || { show "$CONF" full; say "FULL GATE FAILED, stopped"; exit 1; }
show "$CONF" full
say "mechanisms"
run --project "$CONF" --phase full --stages steering,patching,dissect,detection,report
for s in features minimal trajectory; do say "$s"; run --project "$CONF" --phase full --stages "$s,report"; done
say "ALL DONE"
