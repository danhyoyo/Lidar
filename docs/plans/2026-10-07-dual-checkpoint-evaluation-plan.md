### Goal

Evaluate best AP and best loss independently from the notebook, with distinct reports and verified selection evidence.

### Assumptions

- Training and saved configs remain unchanged. The primary winner still supplies the existing benchmark comparator evidence.
- A new AP-selected run must contain both winners. Historical loss-only runs expose only their genuine loss winner.
- Each evaluation uses the run's own config, full validation split and current metric modes.

### Plan

1. Add failing tests for explicit checkpoint selection and independent evidence.
   - Files: `tests/test_dual_checkpoint_evaluation.py`.
   - Change: Exercise actual CPU training/evaluation, distinct synthetic winners, missing/tampered files and the notebook evaluation cell.
   - Verify: `PYTHONPATH=/tmp/lidar-checkpoint-deps PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_dual_checkpoint_evaluation.py -q` (red then green).
2. Support explicit AP/loss verification without modifying the saved config.
   - Files: `checkpoint_selection.py`, `notebook_workflow.py`, `tools/benchmarks/benchmark_protocol.py`, `benchmark_evidence.py`.
   - Change: Resolve both winner records and freeze/verify the evaluated checkpoint's policy while validating the original training history.
   - Verify: Selection, training, benchmark and notebook regression suites.
3. Update the notebook evaluation cell and documentation.
   - Files: root notebook, notebook runbook, training README, affected notebook integration test.
   - Change: Display checkpoint selection/epoch/path; save separate AP/loss evaluations and evidence; retain primary evidence aliases.
   - Verify: Execute the real cell with CPU subprocess evaluators; validate notebook schema, all code cells, clean outputs and whitespace.
4. Review, commit and publish the focused change.
   - Files: modified source/tests/docs only.
   - Change: Preserve user files and generated experiment artifacts.
   - Verify: Focused suite, `git diff --check`, remote branch SHA after push.

### Risks & mitigations

- A loss winner from AP-selected training must still validate the full AP schedule. Verify history against the saved training policy, then choose the requested winner.
- Missing best AP in a new AP run is an error. A historical run is labelled loss-only, with no AP substitution.
- Separate report/protocol/evidence names prevent overwrites. Primary comparator aliases retain their original meaning.
- Evaluating both checkpoints costs two inference passes per metric, even when both winners share an epoch.

### Rollback plan

Use `selected_run(RUN_DIR)` to evaluate only the primary winner. Revert this focused change to restore the previous cell; training checkpoints and configs remain intact.

Verification: 222 focused tests passed; 6 GPU/CUDA cases were skipped on this
CPU environment. Tests execute the real evaluation cell with subprocess CPU
inference for both winners, including an independently trained historical
loss-only baseline, verify both evidence records and reject mismatched policies.
Separate tests cover distinct winner epochs, tampering, missing AP files and
refreshing cached Colab helpers. Notebook schema, all eleven code cells, clean
outputs, changed Python syntax and whitespace checks passed. No real KITTI or
GPU accuracy was measured. Trainer/model/config files were not changed.
