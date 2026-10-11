# r5 checks

Run on CPU against this bundle. Everything in R4_VALIDATION.md still applies.

- New judge tests (`tests/test_judge.py`, 6 tests): the policy model is never touched while judging; a reply that
  says "you are right" but gives the correct answer reads as not endorsing the user's wrong claim, while its tone is
  still reported; a reply that switches reads as switching; praise and flaw questions see the user's message; the
  two answers appear in both orders across items and are never labelled as correct; the answer each key refers to
  is recorded for the hand audit; the default judge is not a Qwen or Llama model.
- The real loading path (`judge_model` building an `LM` from a Hugging Face directory, its chat template, letter
  and Yes/No tokens, batched logits) was run on CPU with a tiny local model standing in for the judge: every
  completed reply got its keys and targets, unfinished replies were skipped.
- Whole pipeline on tiny random models: 60 of 64 steps, the 4 others are random models correctly failing the
  confirmatory gate. Tiny random models never finish a reply, so the judge itself is exercised by the two checks
  above, not by this run.
- Package tests: 95 pass under Python 3.12 with NumPy 1.26.4 and under Python 3.13. Tool tests 20 pass, bundle
  tests 25 pass, all three manifests verify.

Not run here: downloading and running `microsoft/phi-4` (no Hub access from the build machine), real 8B models,
GPUs. Preflight on the instance checks judge access, its chat template, end token and answer tokens before
training. The judge's accuracy is measured by your hand labels (`scripts/judge_agreement.py`).
