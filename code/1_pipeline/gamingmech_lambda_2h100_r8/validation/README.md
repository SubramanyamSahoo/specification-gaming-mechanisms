# Bundle validation

118 CPU tests passed: 73 unchanged core tests, 20 unchanged dual-GPU helper tests, and 25 new wrapper tests. One core Hugging Face integration module was skipped. The original installer runs that integration explicitly on the host.

New tests cover billing input validation, duplicate/changed project rejection, wrapper locking, data/billing handoff commands, distinct GPU-worker project arguments, subprocess failure propagation, credential redaction, calibration phase ordering, and failure of a profile preventing pilot launch. Model calls in orchestration tests are replaced by labeled fixtures.

All original archive file bytes are preserved. Both original manifests are retained. Python and shell syntax checks passed. The generated candidate passed the real v3.1 config validator.

No model weights, GPU jobs, remote host, dataset acceptance, throughput, or completed induction was tested. tmux is unavailable here, so no live tmux session was executed. These records must not be reported as real-model experiment results.
