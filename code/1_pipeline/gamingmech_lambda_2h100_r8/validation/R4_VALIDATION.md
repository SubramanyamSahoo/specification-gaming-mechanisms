# r4 checks

Run on CPU against this bundle.

- Adapter reload bug reproduced on the r3 code with tiny models: when the conditions are loaded a second time in
  one process (behavior loads them for logit metrics and again for generation, and after training in the same
  worker), the first adapter ever dropped gives exactly the base model output (difference 0.0). With r4 it gives
  the same output on every pass. With only the new guard and the old drop code, loading stops with an error
  instead of evaluating the base model under the adapter's name.
- The r3 tiny end to end run had `real` generations identical to base on 72 of 72 prompts for both families. The
  r4 end to end run has 0 of 72 identical for every condition. Pipeline result unchanged otherwise: 60 of 64 steps,
  the 4 others are random tiny models correctly failing the confirmatory gate.
- Install check (`python -m gamingmech.integration`) now drops and reloads an adapter by name twice. Passes on r4,
  fails on r3.
- Package tests: 89 pass under Python 3.12 with NumPy 1.26.4 and under Python 3.13. Tool tests 20 pass, bundle
  tests 25 pass, all three manifests verify.
- `configs/dual_calibration.json` regenerated with the bundled config tool from the package calibration template:
  microbatch 16, 96 updates, answer cap 1024, generation token budget 98304.

Not run here: real 8B models, GPUs, Hub downloads. The judge used for generated behavior is unchanged in r4; its
measured weaknesses on the r3 calibration are listed in the reply that came with this bundle and are the next
decision.
