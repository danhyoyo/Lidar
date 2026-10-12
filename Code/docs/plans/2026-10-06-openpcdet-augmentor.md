# OpenPCDet-style LiDAR augmentation

### Goal
Add a configurable ordered augmentation queue and train-only GT database sampling
to the current KITTI detector, with usable CLI/Colab recipes.

### Assumptions
- Follow OpenPCDet's global transforms and database sampling design without
  importing its CUDA operators or changing the model, loss or KITTI split.
- Dataset boxes are `[class, h, w, l, x, y, z_bottom, yaw]`; GT database metadata
  uses OpenPCDet-style `[x, y, z_center, l, w, h, yaw]` and centered object points.
- Keep existing `one_of` augmentation reproducible. New modes/profiles are opt-in.
- Support flip, rotation, scaling, translation, train-split database creation,
  point-count filters, class quotas, collision rejection and scene-point removal.
  Camera/local-object augmentations and road-plane placement are not in scope.

### Plan
1. Add geometry and queue tests, including ordered transforms, yaw, intensity,
   empty boxes, probabilities, reproducibility and legacy behavior.
   - Files: `tests/test_openpcdet_augmentation.py`.
   - Verify: run tests red before implementing each increment.
2. Implement a NumPy augmentor and integrate the dataset's train path.
   - Files: `detector/core/datasets/augmentor/*.py`, `dataset.py`.
   - Verify: transform tests, real dataset samples and existing target regressions.
3. Implement GT sampling and a train-manifest database builder.
   - Files: `augmentor/database_sampler.py`, `tools/kitti_training_pipeline/build_gt_database.py`.
   - Verify: real small binary point clouds; extraction/insert round trip,
     bottom/center Z conversion, rotated collisions, point removal, quotas,
     split exclusions and DataLoader worker serialization.
4. Add opt-in profiles and notebook integration; identify augmentation in run names.
   - Files: `configs/augmentation/*.json`, `common.py`, standard notebook, pipeline README.
   - Verify: actual profile dataset/model forward-backward, notebook syntax and contracts.
5. Review and verify the complete patch.
   - Verify: targeted pytest, `python tests/test_mobile_bev.py`, `git diff --check`.

### Risks & mitigations
- Database contains train objects only; store manifest hash and frame IDs and
  check provenance on the train dataset before sampling.
- Normalize yaw and test point/box agreement. Avoid corner-to-box reconstruction.
- Use JSON metadata with explicit format/version and feature count. No dependency
  on OpenPCDet installation; Shapely is already a project dependency.
- Use NumPy's worker-seeded random state by default; accept a Generator in tests.
- DB files are loaded only for train. Global transforms also work on empty scenes.
- Keep master recipe unchanged and supply separate global/GT profiles for ablations.

### Rollback plan
Select the existing `one_of` mode or legacy `standard` notebook setting.

### Verification results
- Related suite: 152 passed, 1 deselected (the DataLoader worker check was run
  separately because the sandbox blocks its internal Unix socket).
- Real DataLoader with two workers: 1 passed outside the sandbox after approval.
- Both global-only and GT profiles complete real model forward/backward.
- A GT-profile trainer smoke run finishes an epoch and writes a loadable `last.pt`.
- All 10 `tests/test_mobile_bev.py` checks passed; notebook code cells compile
  after IPython transformation and existing notebook contracts pass.
- `git diff --check` passed. Full KITTI database creation/training/AP have not
  been rerun here; the CLI and notebook perform database creation on the data host.

### Review
- Blockers: none remaining in the implemented scope.
- Majors: none remaining after point/box geometry, database provenance,
  collisions, empty scenes, legacy behavior and worker serialization checks.
- Minors: the CPU sampler and database builder need runtime measurement on full
  KITTI. Processed labels lack difficulty metadata; supplied profiles filter by
  point count only. Road-plane/camera/local transforms are explicitly unsupported.
- Nits: none requiring a change.
- Overall: opt-in profiles and notebook/CLI integration are ready for a fresh
  training ablation; no measured AP improvement is claimed.
