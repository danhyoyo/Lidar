# Learned pillar encoder alongside rich8

### Goal
- Add `pillar32`: a trainable point encoder producing 32-channel BEV for the existing detector. Preserve rich8 and make an encoder comparison runnable.

### Constraints
- Keep existing geometry, augmentation, targets, backbone and head contracts; do not modify user-owned notebook/database changes.
- Use PyTorch operations, without a new sparse-convolution dependency. Keep datasets and generated results untracked.

### Known context
- Dataset currently rasterizes BEV in NumPy. Training assumes a dense tensor; evaluation transfers one dense frame. Encoder parameters must live in the model to receive gradients.
- The AI_env Python environment provides PyTorch and pytest. CUDA and DataLoader IPC are accessible when verification runs outside the sandbox.

### Risks
- Variable point counts require collation; empty/single-point frames must remain valid. Input metadata and checkpoint identity must describe learned features correctly.
- Point activations and 32-channel BEV increase memory. Current fixed-shape ONNX/TensorRT consumers cannot accept packed points.

### Options (2–4)
- Point MLP + occupied-pillar pooling + BEV scatter: small parameter increase, retains all ROI points; needs packed input support. Selected.
- Padded pillars with fixed point limits: familiar interface, but may discard points and wastes padding memory.
- Sparse voxel backbone: preserves vertical topology but changes architecture and dependencies substantially.

### Recommendation
- CPU prepares ten point features (xyz/intensity, cluster offsets, pillar-center offsets). A model-owned Linear/BN/ReLU learns 32 features, pools occupied pillars and scatters to BEV. No point/pillar caps. Support PyTorch training/evaluation first; explicitly reject unsupported deployment paths.

### Acceptance criteria
- Learned encoding has validated, reproducible config identity and 32-channel backbone input.
- Pooling/scatter are permutation invariant and batch isolated; empty/single-point samples are finite; encoder parameters receive detector-loss gradients and update.
- Synthetic CLI training, checkpoint resume and AP evaluation work; existing rich8 tests pass.
- Separate paired configs, notebook selection and usage documentation support comparison.

### Assumptions
- `pillar32` is an additional experiment, not a change to the master default. Accuracy improvements require actual training and are not claimed by implementation tests.

### Plan
1. Specify behavior with failing tests.
   - Files: `tests/test_pillar_encoder.py`.
   - Change: geometry/features, collation, gradient/update, empty frames, config identity, CLI integration tests.
   - Verify: `AI_env/bin/python -m pytest tests/test_pillar_encoder.py -q` must fail because pillar32 is missing.
2. Add pure schema and packed point preparation.
   - Files: `detector/core/bev_encoding.py`, `detector/core/datasets/utils_1/pillar_backend.py`, `detector/core/datasets/dataset.py`.
   - Change: declare learned semantics; build uncapped occupied pillars after augmentation; collate variable-length batches.
   - Verify: focused schema/preparation/collation tests.
3. Add model-owned encoder and pipeline adapters.
   - Files: `detector/core/models/encoders/pillar.py`, `detector/core/models/model.py`, `tools/kitti_training_pipeline/{common,train,evaluate_kitti_bev,evaluate_kitti_3d}.py`.
   - Change: learn/pool/scatter; support packed training and evaluation; count encoder parameters and reject incompatible deployment.
   - Verify: gradient, synthetic train/resume/AP tests and existing model/training tests.
4. Expose comparison recipes and notebook selection.
   - Files: `configs/experiments/encoders/*.json`, `tools/kitti_training_pipeline/notebook_config.py`, tracked standard notebook, `docs/learned_pillar_encoder.md`.
   - Change: paired encoder-only experiment configurations, runnable commands and documented compute limitations.
   - Verify: resolve recipes, count actual parameters, inspect one actual KITTI frame, validate notebook JSON and Python cells.
5. Review and finish.
   - Files: changed files only.
   - Change: fix review findings and update this checklist.
   - Verify: focused then full pytest; `git diff --check`; inspect final diff and user-owned file status.

### Risks & mitigations
- Preserve existing rich8 state-dict keys and semantic metadata. Test the actual training/evaluation entry points rather than just shapes.
- Verify CPU and CUDA FP32/BF16 gradients; leave trained AP and latency comparisons to the controlled experiments. Guard ONNX/TensorRT and dense-only profilers against silently bypassing the encoder.

### Rollback plan
- Select rich8 in config. The master default and existing checkpoints remain usable; no data migration is required.

### Progress
- [x] Failing behavioral tests observed.
- [x] Packed preparation/schema/collation verified.
- [x] Encoder gradients and train/evaluate/resume verified.
- [x] Paired configs/notebook/documentation verified.
- [x] Review and final verification completed.

### Verification and review
- Focused final suite: **109 passed**, including all 19 pillar cases, CUDA FP32/BF16 detector-loss updates, CPU BF16, exact resume, AP smoke and two spawned DataLoader workers. Logs: `artifacts/pillar32_verification/focused.log` and `focused.xml`.
- Full CPU suite before final test additions: **1590 passed, 16 failed, 80 skipped**. One new encoding inventory expectation was corrected and passed in the focused suite. Fifteen remaining failures concern eleven pre-existing preset/resolver mismatches, two existing partial-AP smoke tests, one existing notebook split-control expectation and one Matplotlib GUI environment failure. HEAD source checks confirmed the existing preset, AP guard and notebook failures. Full-suite log: `artifacts/pillar32_verification/full.log`.
- Actual KITTI frame 000000: 115384 raw points, 62762 retained, 14131 occupied pillars, exact rich8 XY occupancy alignment. BEV `[1,32,800,704]` and detector outputs are finite. Report: `artifacts/pillar32_verification/real_frame.json`.
- Comparison configs differ only in BEV encoding and produce actual parameter counts 648313/655609. Notebook control cell compiles; config/notebook JSON and `git diff --check` pass.
- Review finding fixed: float32 cell-boundary rounding initially changed XY membership relative to rich8; regression test now enforces the same floor-division behavior. No outstanding blocker found in the changed encoder/training/evaluation paths.
- Trained AP, full-resolution GPU latency and ONNX/TensorRT support are not established by this implementation. User-owned robust notebook, raw dataset and GT database edits were preserved.
