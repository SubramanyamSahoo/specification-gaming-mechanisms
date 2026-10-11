# v3.1 verification and limitations

## Executed in the construction environment

- **73 CPU regression tests passed.** Full stdout and the skip reason are in `test_results.txt`.
- **One actual Hugging Face/PEFT integration-test module was skipped:** Transformers is unavailable in this environment. The required libraries, pretrained models, CUDA and an H100 were not available for a real-model run.
- Python compilation and AST parsing passed for the shipped package and scripts.
- Bash syntax checks passed for every launcher. Command-line help was exercised and saved.
- The initialized-project repair was tested against the exact prior v3 ZIP (SHA256 `6155547d1c263f4125f464894cffa93d1d2e17e51b3745e998681e47da391959`). Both untouched-source and already-manually-patched-source cases passed: dry-run, preserved configuration, source-lock reopening, retained training receipt, invalidated steering receipt, regenerated distribution checksum, idempotence, and rejection of unrelated edits. See `v3_hotfix_test_results.json`. This is a filesystem/control-flow check, not GPU validation.

Reproduce the last check with the original archive:

```bash
python scripts/test_v3_repair.py --archive /absolute/path/to/gamingmech_h100_v3.zip
```

## New regression scope

The actual `Steerer.values()` function is exercised on a tiny Torch fixture. Tests cover the first cache miss, a cache hit with no recomputation, a zero vector preserving outputs, and hook removal. A type check rejects an eagerly computed callback value. A source-level assertion checks the cache call sites for the same error class. The actual tiny Qwen/Llama integration module now includes an equivalent steering canary, but that module still needs to run on your installed host stack.

Calibration tests cover: a failed diagnostic gate staying false while reporting completes; unchanged fail-closed confirmatory behavior; no mechanism/test access from calibration; safe resume of an executed negative diagnostic; grouped censoring counts; partial-data reporting; fixed-cohort token-cap sweeps and cache reuse (model boundary explicitly mocked); explicit fresh seeds/rationale/contamination choice; refusal to select an unmeasured token cap; decision-record consistency; no silent protocol overwrite; data-only reuse with contract checks; no audit/weight promotion; and billing-history retention.

The prior CPU tests remain, including chunked-likelihood value/gradient checks, exact target sets, missing/corrupt artifact rejection, semantic negation guard, group splitting, reconstruction rejection, empty CSV handling, RNG restoration, budget handling and sample-weighted accumulation.

## Not established here

No real Qwen3.5-9B/Llama-3.1-8B training, source-dataset acceptance rate, free-generation completion rate, semantic-judge reliability, SAE transfer fidelity, H100 throughput, end-to-end cost or gaming induction result was measured. The initial 96 updates, rank 16, 256-token cap and existing gate thresholds remain candidates for calibration—not recommended as sufficient by experimental evidence.

Increasing oversampling to 3 provides more candidates but can cost more writer/validator work. It does not guarantee quotas. Training-strength changes need their own development trial. Higher token caps are not guaranteed to resolve nontermination. Neither low-contamination adapters nor descriptive checkpoint plots alone establish an early-warning result.

## Before the paid study

The installer runs the CPU suite and then `python -m gamingmech.integration` explicitly, failing if the actual architecture checks cannot execute. Project preflight checks the host, access, tokenizer and dataset schemas. Then run the disposable calibration, inspect actual outputs, select the final protocol, and perform the final human audit before confirmatory training. Preserve all negative and failed confirmatory outcomes.

Source locks are retained. The recovery helper authorizes only the known one-token v3 correction, not arbitrary changes to an existing experiment. Unknown future bugs require a separately reviewed source amendment or a new run with explicit provenance; checksum regeneration alone never authorizes silent reuse.
