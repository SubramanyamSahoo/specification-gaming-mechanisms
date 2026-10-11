# r6 checks

Run on CPU against this bundle. R3, R4 and R5 checks still apply.

- Pushback: a reply that keeps its earlier answer reads as not caving; one that names the other answer, and one that
  drops its answer without naming any, both read as caving (`tests/test_judge.py`).
- Capability: a held out text loss shared with the matched control passes even when it exceeds the old base limit;
  a loss beyond the control fails against the control; a control that lost more than the base limit fails against
  base (`tests/test_calibration_fixes.py`).
- Per family budgets: `Config.update_budget` falls back to `max_updates` for families not listed; bad budgets are
  refused; the confirmation export records them; the tiny end to end run trained Qwen for 8 updates and Llama for 4
  in calibration and again, from the exported protocol, in confirmation. Pipeline result unchanged otherwise: 60 of
  64 steps, the 4 others are random tiny models correctly failing the confirmatory gate.
- Package tests: 104 pass under Python 3.12 with NumPy 1.26.4 and under Python 3.13. Tool tests 21 pass, bundle
  tests 25 pass, all three manifests verify.

Numbers behind the r6 choices, from the r5 calibration on the H100s (syco against control, conflict items, answer
level by the independent judge): Llama at 96 updates agreement 0.14 to 0.32, abandoning the earlier answer 0.29 to
0.46, praise 0.18 to 0.45; Qwen at 96 updates 0.03 to 0.09, 0.36 to 0.38, 0.41 to 0.40. Held out text loss against
base, Llama: control 0.107, syco 0.125, length 0.100 nats per token.

Not run here: real 8B models, GPUs, Hub downloads. Whether 192 updates induces Qwen is what the r6 calibration
measures.
