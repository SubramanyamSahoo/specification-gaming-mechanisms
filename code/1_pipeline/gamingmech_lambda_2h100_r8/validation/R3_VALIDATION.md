# r3 checks

Run on CPU against the r3 bundle; all of it still applies to r4. r4 additions: R4_VALIDATION.md. The older files in this folder describe earlier bundles.

- Package tests: 89 pass (74 original, 15 in `tests/test_response_bank.py`) under Python 3.13, and the same 89 pass under Python 3.12 with NumPy 1.26.4, torch 2.8 and the pinned Hugging Face stack, which is the Lambda environment class.
- Tool tests 20 pass, bundle tests 25 pass, all three manifests verify.
- Real questions of the failed run (4,737 records), with the Qwen tokenizer rebuilt from the public vocabulary file (it reproduces 12,466 of the 12,475 token sequences saved by that run) and the Llama 3 tokenizer rebuilt the same way (52 of 56 sequences saved by the Llama judge reproduced), chat formats approximated:
  - 57 questions dropped because they contain an answer; 2,865 of 3,238 passages give a usable answer anchored excerpt.
  - every bucket above its quota; selected roster exactly 1,500 train and 75 validation pairs;
  - zero rule validator failures on the valid pairs;
  - stance pairs equal in token count under both tokenizers; verbosity ratio median 3.5 under both;
  - longest synthetic training sequence 221 tokens under Llama against the 768 limit.
- The reviewer's two counterexamples (an unrelated first passage sentence used as detail; an agreeing reply where disagreement was required) are now regression tests that the rules reject.
- Whole pipeline on tiny random Qwen3 and Llama models under Python 3.12: data, validate, calibration, cap sweep, export, handoff, audit, pilot, full and every downstream stage run for both families (60 of 64 steps; the 4 others are random models correctly failing the confirmatory gate).

Not run: real 8B models, real GPUs, Hub downloads, the real Llama chat template, the real-pair length distribution. Whether bank responses induce the behaviours is what calibration measures.
