# pillar_rich: rich8 + learned24

### Goal
Add `pillar_rich`, concatenating exact rich8 statistics (channels 0–7) and learned point features (channels 8–31), with the existing 32-channel backbone contract.

### Assumptions
- Preserve rich8/pillar32 defaults, semantics and checkpoint keys. Preserve user-owned notebook/GT database changes and datasets.
- Notebook selection uses `PRESET="custom"`, `BEV_ENCODING="pillar_rich"`; old cell width lookups are handled by the resolver without notebook edits.
- No claim of improved AP until controlled training. Support PyTorch; keep deployment guards.

### Plan
1. Specify failing behavioral tests.
   - Files: `tests/test_pillar_rich_encoder.py`, `tests/test_pillar_encoder.py`, `tests/test_under1m_notebook_config.py`.
   - Change: rich8 equivalence, density sensitivity, schema identity, empty frames, gradients, config parity, CLI train/AP/resume/workers.
   - Verify: focused new tests fail because pillar_rich is missing.
2. Prepare and fuse packed features.
   - Files: `detector/core/bev_encoding.py`, `detector/core/datasets/{dataset.py,utils_1/pillar_backend.py}`, `detector/core/models/{model.py,encoders/pillar.py}`.
   - Change: compute rich8 per occupied pillar using the same ROI/indexing/normalization; collate stats; learn 24 channels, concatenate, scatter once.
   - Verify: exact FP32 rich8 equality, BF16/CUDA gradients, batch isolation, invalid input and empty/singleton cases.
3. Integrate configuration and tools.
   - Files: `tools/kitti_training_pipeline/{common,train,evaluate_kitti_bev,notebook_config}.py`, `tools/benchmarks/profile_detector.py`, `configs/experiments/encoders/pillar_rich.json`.
   - Change: packed detection, warmup, deployment guards, input accounting, preset and encoder-only comparison config.
   - Verify: synthetic training/AP/evaluator, checkpoint resume identity, spawn workers; actual model parameter counts and real KITTI frame.
4. Document, review and deliver.
   - Files: `docs/learned_pillar_encoder.md`, pipeline README, this plan.
   - Change: usage and costs; retain notebook files unchanged; commit/push task files only under the user's existing authorization.
   - Verify: relevant regression suite, `git diff --check`, inspect staged files and remote push result.

### Risks & mitigations
- Duplicated compact rich8 formulas could drift: assert exact equality against the existing rasterizer on random points and boundaries; avoid changing legacy rasterizers.
- Stats must follow the same augmentation/ROI/XY groups: compute both from the same retained points and concatenate aligned occupied pillars.
- Under autocast, cast rich8 to the learned pooled dtype before fusion; test FP32 equality and expected BF16 rounding.
- Keep density normalization in hybrid checkpoint identity; reject rich8/pillar32 checkpoint resume into pillar_rich.

### Rollback plan
Select rich8 or pillar32; no dataset migration.

### Progress
- [x] Failing tests observed: 10 new behavioral cases initially failed because pillar_rich was missing; integration cases also exposed missing width, warmup and deployment adapters before those were implemented.
- [x] Packed preparation/fusion verified.
- [x] Pipeline/config integration verified.
- [x] Real frame, review and regression suite verified; commit/push follows these checks.

### Verification
- Focused suite: **143 passed**, no skips, including CUDA FP32/BF16 detector-loss updates, CPU BF16 rounding, exact rich8 statistics, augmentation ordering, empty/singleton inputs, checkpoint identity rejection, synthetic CLI train/AP plus standalone evaluation, exact resume and spawn workers.
- Full CPU suite: **1625 passed, 14 failed, 84 skipped**, plus 17 passed subtests. XML comparison confirmed all 14 failures also existed in the prior pillar32 full-suite run: eleven preset/resolver mismatches, two partial-AP smoke tests and one notebook split-control expectation. No new full-suite failures. Logs/XML in `artifacts/pillar_rich_verification/`.
- Actual KITTI 000000: 115384 raw points, 62762 retained, 14131 occupied pillars, BEV `[1,32,800,704]`; first eight channels exactly equal independent rich8, detector outputs finite. Report: `artifacts/pillar_rich_verification/real_frame.json`.
- Actual comparison counts: rich8 648313, pillar32 655609, pillar_rich 655513, hybrid point encoder 288. Configs differ only in data.bev_encoding.
- Compared all seven previous encoding metadata/hash contracts directly against HEAD: unchanged. Standard notebook remains byte-for-byte unchanged in Git; user-owned robust notebook, generator, GT database changes and dataset were not modified.
- `git diff --check` passes. No trained AP or full-resolution CUDA latency claims.

### Review
- Blockers: none introduced by this change.
- Majors: none outstanding in reviewed encoder, collation, config or pipeline paths.
- Minors: none requiring a change. Compact rich8 formulas intentionally leave legacy rasterizers untouched; equivalence tests protect against drift.
- Nits: none recorded.
- Overall: ready for the controlled three-encoder experiment and delivery to the current remote branch.
