### Goal
Restore the notebook's original setup/config/prepare/smoke/train/evaluate flow,
keeping the current architecture, loss, augmentation and BEV/3D controls.

### Constraints
Use IPython `!` commands, no notebook subprocess runner. Preserve run snapshots,
strict resume and independent AP/loss checkpoints. Keep existing benchmark tools
available from the CLI without requiring them in the training notebook.

### Known context
The branch base is 3654a0b. Current notebook requires several audit/protocol/GPU
stages. Trainer logs train/validation results only after periodic AP completes,
so a slow AP pass hides the epoch's completed losses and timing.

### Risks
Shell failures must stop the cell. Runtime config must survive direct CLI calls.
Existing tests for mandatory benchmark evidence need to follow the new notebook
contract while retaining separate tests for the benchmark helpers.

### Options
1. Revert both files wholesale: short but loses new config and model support.
2. Restore notebook flow, preserve the resolver and trainer contracts: moderate
   change, retains current configurations and checkpoints.

### Recommendation
Option 2. Print the original loss/LR/time summary before AP, then AP results.

### Acceptance criteria
- Notebook uses visible `!` commands with checked exit codes and unbuffered Python.
- All current model/loss/data/runtime controls still resolve correctly.
- Train/val loss summary is visible before AP starts; AP results identify classes.
- Smoke, train/resume and both selected-checkpoint evaluations execute on small
  synthetic KITTI assets; notebook schema and transformed cells remain valid.

### Assumptions
"Original" refers to 3654a0b, before the branch's unified workflow commit.
The reported issue is missing/confusing output while AP is still running;
no separate exception traceback has been supplied.

### Plan
1. Regression coverage
   - Files: notebook command/import/control tests, AP training tests.
   - Change: Exercise IPython commands, failure handling and log ordering.
   - Verify: focused pytest fails for the intended missing behavior.
2. Simplify notebook
   - Files: root standard notebook, affected integration tests.
   - Change: Original cell flow, current config resolver, direct shell commands,
     minimal summary output and independent checkpoint evaluation.
   - Verify: IPython-executed smoke/train/evaluation and config regression tests.
3. Clarify trainer output and config precedence
   - Files: tools/kitti_training_pipeline/train.py, training tests.
   - Change: Loss summary before AP, explicit AP results; omitted CLI runtime
     options inherit the saved config rather than silently changing it.
   - Verify: CPU training/resume/AP/config tests.
4. Documentation and final review
   - Files: notebook runbook, training README, changed tests.
   - Change: Describe current cells, logs and manual Colab usage.
   - Verify: focused suite, full available suite, notebook schema, git diff --check.

### Risks & mitigations
Run real shell commands through IPython in tests, including paths with spaces and
failed commands. Keep dataset/checkpoint outputs in temporary or ignored folders.

### Rollback plan
Restore changed tracked files from the preceding revision. Training artifacts
and existing snapshots are never deleted by this change.

### Verification
- Regression tests failed first for the custom process wrapper, train-loss output
  occurring after AP, and omitted CLI flags overwriting saved runtime settings.
- Focused notebook/training/resume/config suite: 229 passed, 3 CUDA tests skipped.
  Actual IPython cells ran CPU smoke, two training epochs, both winner evaluations
  and a completed-run rerun; paths with spaces and shell failures were exercised.
- Full sandbox suite: 1,571 passed, 17 subtests passed, 80 skipped, 19 failed.
  Six multiprocessing failures were caused by blocked IPC socket operations;
  the same six tests passed outside the sandbox (4 grouped + 2 augmentation).
- All remaining 13 failures were reproduced in an isolated archive of unmodified
  HEAD: 11 materialized recipes differ from the newer master AP-selection config,
  and two historical partial-smoke tests inherit AP selection. These existing
  recipes/tests were not changed by this notebook simplification.
- Notebook schema, all nine transformed code cells, cleared outputs, changed
  Python syntax and git diff --check pass. The configuration resolution portion
  of the actual notebook cell is unchanged from HEAD.
- Review found no Blocker/Major issue in the changed workflow. Actual Colab/GPU
  execution and long KITTI training were not rerun. No commit/push was performed.
