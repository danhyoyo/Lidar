### Goal

Keep only augmentation experiment configs on this branch and make the agreed
A–E screening runs executable without changing model/loss between runs.

### Assumptions

- A reproduces the artifact's standard OneOf augmentation, including Z translation.
- B adds light GT sampling (p=0.2, one Pedestrian and one Cyclist); C/D/E each
  enable one additional density/shadow/intensity heuristic relative to B.
- Keep source-relative viewing angle while proposing nearby ranges (0.8–1.2)
  and azimuths (±5°), rather than random independent yaw.
- All runs use MobilePIXORNeXt/LiteMLA/SG-FPN, Rich8, baseline loss, seed 42,
  physical batch 16, AdamW LR 0.0007, weight decay 0.001, warmup 4, clipping 10.
- User revised the budget to a complete 50-epoch schedule, including warmup.
  Resume interrupted runs within this horizon; a 100-epoch experiment needs a fresh run.

### Plan

1. Protect the experiment contract and runtime behavior with regression tests.
   - Files: `tests/test_augmentation_configs.py`, `tests/test_gt_sampler.py`,
     `tests/test_gt_sampler_config.py`, `tests/test_optimizer_scheduler.py`.
   - Verify: new tests fail before implementation, using the existing AI_env Python.
2. Add only runtime support required by the experiment configs.
   - Files: `detector/core/datasets/dataset.py`, `detector/core/datasets/utils_1/gt_sampler.py`,
     `tools/kitti_training_pipeline/train.py`, `tools/kitti_training_pipeline/common.py`.
   - Verify: OneOf remains available with sampling; bounded poses preserve view;
     clipping works; stopping does not shorten the cosine schedule; run names are unique.
3. Create A–E configs and remove the unrelated config directories requested by the user.
   - Files: `configs/kitti/augmentation/*.json`, `configs/kitti/augmentation/README.md`.
   - Verify: equal model/data/loss/train settings; isolated sampling flags; valid JSON.
4. Update notebook, active docs, and tests that load removed config paths.
   - Files: `3D_Lidar_Object_Detection_Notebook_standard.ipynb`, `README.md`,
     `tools/kitti_training_pipeline/README.md`, affected config-loading tests,
     `tools/visualization/visualize_pcu_aug.py`, `tools/visualization/audit_gt_sampler.py`.
   - Verify: all five notebook paths exist; notebook code compiles after IPython
     transformation; preview uses selected sampler settings; screening evaluates epoch 50.
5. Review and run the affected tests plus a synthetic CPU train/stop/resume check.
   - Verify: test results, checkpoint epoch/scheduler state, CLI help, diff and repo status.

### Risks & mitigations

- Missing old paths: update active consumers; preserve historical design documents.
- Accidental result overwrite: config-specific experiment names in generated run names.
- Schedule consistency: all A–E runs use the same complete 50-epoch schedule;
  retain the optional early-stop flag for interrupted-run verification.
- Config-only settings ignored by code: test effective sampler and Dataset settings.

### Rollback plan

All removed configs are tracked at the current HEAD. Review the diff and restore
individual files from that commit if needed; do not change existing artifacts.
