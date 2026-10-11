# Changes from the uploaded extras_r9.zip

The inherited six comparisons are preserved; this is an all-run packaging and
operational hardening release, not a new scientific gate or main-model retraining.

- One standalone directory with all required add-on files. The older three loose
  files and separate RUN_LLAMA_R9.sh are no longer needed for this release.
- One tmux-protected all-run entrypoint; no kill-session commands, no Qwen launch.
- Physical GPU selected by UUID; busy-GPU failures are explicit; shared launcher
  lock and source execution lock prevent accidental concurrent guarded jobs.
- Exact r8 core/project/gate identity checks, explicit token and repository access
  checks, immutable source metadata/data/adapter identities, versioned extras
  workspace, complete source billing history and a polling budget supervisor.
- Six experiments are separate child processes. A failure stops the queue and
  propagates to the launcher instead of printing unconditional ALL DONE.
- Seed replication checkpoints completed substeps; attention and joint-head
  measurement caches avoid repeating committed operations. The full current
  output list is hashed before reuse. New adapter finals are checked as well.
- Mutable source data/reference caches are copied; only original immutable weight
  folders are linked. The original directory remains a dependency, not a backup.
- Eager attention starts at at most four items per memory microbatch and halves
  the same batch on OOM without dropping examples. Coverage and unresolved claim
  spans are saved rather than silently treated as full coverage.
- Joint-head results now save per-item donor/recipient/intervention values, paired
  intervals and aggregate outputs. Out-of-range/duplicate/empty head sets fail.
- Raw generated token IDs, group IDs, context, partition and censoring are retained.
  Judgment calls are cached per raw response under the unchanged r8 evaluator.
- Dose tensors/monitor scores and selection indices are retained with checksums.
  Shared resampling draws are used across doses. New dose rates remain 0.10,
  0.50 and 0.75, matching the supplied proposal.
- Empty public benchmarks or missing cross-seed head files stop execution rather
  than generating a misleading successful empty summary. Public file bytes are
  fetched/checksummed before the first full-model job.
- Actual CSV outputs receive separate PNG/PDF plots and a local HTML index.
  Exports include raw JSONL/logs, use unique names and explicitly exclude weights,
  tensors and symlinks. Full raw files remain in the local workspace.

No source gates are bypassed. No judge labels, human labels, predictions or
hypothesis results are prefilled. The original DPO and mechanism modules are not
edited. Known measurement/interpretation limitations are documented in README.md.
