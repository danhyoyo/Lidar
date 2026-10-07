### Goal

Save independent best-AP and best-loss checkpoints, select best AP for new notebook runs, and preserve complete resume and benchmark provenance.

### Assumptions

- Validation loss still runs every epoch. AP runs on the entire validation split at a configurable interval and always at the final epoch.
- The fixed AP objective is the equally weighted Car/Pedestrian/Cyclist Moderate R40 mean. BEV development uses `local_bev`; 3D runs default to reference `3d`.
- Existing configs without an explicit selection policy keep minimum-loss behavior. Changing the selection protocol requires a new run name.
- Short, batch-limited smoke runs use loss selection and do not compute AP.

### Plan

1. Add selection policy, schedule, and independent retention tests.
   - Files: `tests/test_checkpoint_selection.py`, `tools/kitti_training_pipeline/checkpoint_selection.py`.
   - Change: Validate policy/metrics, retain separate winners, verify complete histories, preserve strict ties and resume metadata.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_checkpoint_selection.py -q` (red, then green).
2. Integrate actual full-split evaluators into training and checkpoint resume.
   - Files: `tools/kitti_training_pipeline/train.py`, `common.py`, `tests/test_ap_training.py`.
   - Change: Compute AP from a temporary epoch snapshot using existing evaluators, restore RNG after evaluation, save AP state with all training state and log AP separately.
   - Verify: CPU synthetic KITTI training/evaluation integration, interrupted/resumed schedules, unchanged training tensors with AP enabled.
3. Update notebook and provenance consumers together.
   - Files: `configs/config.json`, notebook, `notebook_config.py`, `notebook_workflow.py`, benchmark protocol/evidence helpers, focused tests.
   - Change: Expose primary selection/interval/mode/raw root; evaluate best AP; keep legacy loss selections valid; require matching selection protocols for comparisons.
   - Verify: Config, notebook command/import, selection, benchmark evidence and training tests; validate notebook syntax/schema and clean outputs.
4. Document, review, and publish the focused change.
   - Files: training README and notebook runbook.
   - Change: Explain canonical checkpoint names, AP cost and sampling limitations, compatibility, resume and Colab controls.
   - Verify: Review diff, `git diff --check`, commit only source/tests/docs, push the existing research branch and verify its remote SHA.

### Risks & mitigations

- Sparse AP evaluation can miss an unsampled peak: default to every epoch; record and compare identical intervals.
- AP evaluation consumes RNG while building an inference model: restore all parent RNG state in a `finally` block.
- Missing classes or partial validation can inflate macro AP: require finite Moderate scores and positive GT counts for all three classes, exact full-split coverage, and consistent decode settings.
- Historical benchmark verification assumes loss selection: share policy-aware verification instead of bypassing it.
- No local KITTI or GPU: use real synthetic CPU training and local BEV evaluation; report reference/GPU checks separately.

### Rollback plan

Use a fresh run with `train.checkpoint_selection.primary="loss"` to retain loss selection. Existing legacy runs remain readable. Revert this focused commit to restore the previous workflow; do not overwrite run snapshots or delete checkpoints.

Implementation verification: the focused selection, actual CPU KITTI-format
training/evaluation, resume, notebook, benchmark provenance and synthetic-smoke
suites passed (342 tests; 6 GPU/CUDA tests skipped). This environment used Python
3.14, PyTorch 2.14.1 CPU and Numba 0.68 installed temporarily outside the project.
No real KITTI AP or Colab/GPU accuracy claim is made. Notebook schema, all eleven
code cells, clean outputs, Python syntax and whitespace checks passed. Synthetic
smoke explicitly overrides AP selection to loss while retaining the original
source-config identity in its report.
