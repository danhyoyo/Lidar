### Goal
Make experiment modes reliable when the user edits only the standard notebook.

### Assumptions
- `configs/config.json` remains the default configuration; each run stores its resolved copy.
- Model presets and augmentation are independent choices. File overrides take priority over those choices; notebook runtime settings take final priority.
- Existing unrelated loss and trainer changes stay untouched.

### Plan
1. Reproduce notebook configuration failures.
   - Files: `tests/test_notebook_experiment_config.py`.
   - Change: Execute the actual configuration cell with preset/augmentation combinations, edited base configs and repeated execution.
   - Verify: `/tmp/lidar-fixes-env/bin/python -m pytest -q tests/test_notebook_experiment_config.py` (expected failures before fixes).
2. Centralize resolution and update notebook controls.
   - Files: `tools/kitti_training_pipeline/notebook_config.py`, `3D_Lidar_Object_Detection_Notebook_standard.ipynb`.
   - Change: Resolve fresh defaults, explicit model selections, independent augmentation recipes, file overrides, then runtime settings. Validate effective settings and protect an existing run config before writing.
   - Change: Regenerate automatic run names; expose manual name, warmup, head channels and validation batch size; print actual augmentation queue.
   - Verify: `/tmp/lidar-fixes-env/bin/python -m pytest -q tests/test_notebook_experiment_config.py tests/test_standard_training_notebook.py`.
3. Document and verify integration.
   - Files: `README.md`, configuration tests.
   - Change: Document notebook-only usage, precedence, base-config mode and run naming.
   - Verify: Run relevant notebook, augmentation, model/config and assignment tests; inspect `git diff --check` and the final diff.

### Risks & mitigations
- Reused run names can overwrite a configuration before the resume check: reject changed configs when saved run metadata or checkpoints exist.
- Model channels and neck settings can leak from the base: set semantic fields explicitly and validate supported combinations before dataset preparation.
- Old Colab cells remain old after checkout: require reopening the updated notebook; reload helper modules on configuration cell execution.

### Rollback plan
Revert only the notebook resolver change; keep the earlier augmentation implementation and unrelated work.
