# r8 checks

Run on CPU against this bundle with the r7 stack (transformers 5.16.1, PyTorch 2.8.0). R7 checks still apply.

- Gate (`tests/test_behaviour_gate.py`, 4 tests): a behaviour that does not transfer is reported in `unconfirmed` and
  `confirmed_behaviours.csv`, not as an error, and the condition passes on its other behaviours; logit evidence alone
  or generated evidence alone confirms nothing; a behaviour must hold in every seed (two seeds confirming different
  behaviours fail, two seeds sharing one pass with that one); a capability failure still fails the gate.
- Whole pipeline on tiny random models (Qwen3.5 layout for the Qwen family): 60 of 64 steps, as in r6 and r7; the 4
  others are random models correctly failing the gate, now with "no behaviour confirmed".
- `RUN_FAMILY.sh` in a simulated home: reuse of the r7 environment, the package import check, all tests, the
  install marker, the r6 billing clock and the r6 and r7 revisions pinned; with stubbed stages, the full order from
  calibration to the last extra, the protocol v3 arguments, and a stop with the gate printed when a pilot gate fails.
  `START_ALL.sh` starts both families once, never twice, and passes the token through tmux memory only.
- Package tests: 114 pass. Tool tests 21 pass, bundle tests 25 pass, all three manifests verify.

Not run here: GPUs, Hub downloads, real 8B and 9B models.
