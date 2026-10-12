# KITTI small-object correctness fixes

## Design

### Goal
Preserve each representable object center in dense regression targets and prevent
different classes from suppressing one another during heatmap peak extraction.

### Constraints
Keep model heads, loss strategies, KITTI split, and training recipe unchanged.
Preserve existing uncommitted Q-OGA/training changes. Python and Numba target
generation must agree. Retain explicit legacy modes for reproducing old runs.

### Known context
Dense regression targets currently use last-box-wins assignment over overlapping
disks. Decoder pools a score map after taking the maximum over classes.
Checkpoint selection uses validation loss; existing train.py has unrelated edits.

### Risks
One shared regression vector cannot represent multiple different boxes at the
same center cell. New decoding may increase candidate count and latency. Existing
checkpoints remain loadable, but a new decoder changes the evaluation protocol.

### Options
1. Regress at center only: simple, but substantially changes supervision density.
2. Nearest-center assignment with reserved center cells: preserves dense training
   while correcting ownership. Use canonical box ordering for exact distance ties.
3. Separate regression per class: more capacity, but changes model/checkpoint shape.

### Recommendation
Use option 2. Build one shared Python/Numba target kernel. For colliding center
cells choose the center closest to the cell origin, breaking ties by canonical
box order; do not claim to represent both objects. Extract peaks independently
per class before class-wise rotated NMS. Keep legacy assignment and peak modes.

### Acceptance criteria
- Distinct nearby center peaks decode to their own target centers.
- Dense overlap ownership is independent of input label order.
- Colliding centers are deterministic; empty and boundary cases remain valid.
- Python and compiled Numba targets are byte-identical.
- Adjacent different-class peaks survive with and without IoU-aware scoring/NMS.
- Single-class suppression and legacy decoding retain expected behavior.
- Evaluator exposes and records the selected peak mode.

## Implementation plan

### Goal
Deliver tested target/decoder fixes and a concrete evaluation/retraining procedure.

### Assumptions
The current user request authorizes these fixes following the artifact analysis.
Architecture/augmentation experiments and a full training run follow after
measuring the correctness fixes; they are outside this patch.

### Plan
1. Add failing regressions for nearby objects and class-aware peaks.
   - Files: `tests/test_small_object_assignment.py`.
   - Verify: `/tmp/lidar-fixes-env/bin/python -m pytest -q tests/test_small_object_assignment.py`.
2. Share regression generation between Python and Numba, reserve centers and
   resolve remaining overlaps by nearest center; support `regression_assignment=legacy`.
   - Files: `detector/core/datasets/dataset.py`, `detector/core/datasets/utils_1/target_backend.py`.
   - Verify: target tests and existing dataset/backend parity checks.
3. Keep class dimension during peak extraction; flatten only selected candidates.
   - Files: `detector/postprocess.py`, `tools/kitti_training_pipeline/evaluate_kitti_bev.py`.
   - Change: default `peak_mode=per_class`; explicit legacy override and metadata.
   - Verify: decoder tests, existing IQA tests, evaluator CLI/options checks.
4. Review and document the comparison/retraining commands.
   - Files: `tools/kitti_training_pipeline/README.md`.
   - Verify: targeted suite, `python tests/test_mobile_bev.py`, `git diff --check`.

### Risks & mitigations
Record peak mode when reporting AP. Evaluate old checkpoints with both peak modes
before retraining. Keep head capacity, augmentation, loss weighting, and seed fixed
in the first corrected training run. Same-cell target collisions remain a limitation.

### Rollback plan
Use `data.regression_assignment=legacy` and decoder/evaluator `peak_mode=legacy`
to reproduce the original target and peak policies without reverting unrelated work.

## Verification results

- Targeted pytest suite: 109 passed, 1 deselected. The deselected existing test
  generates thousands of random candidates for quadratic Shapely CPU NMS; its
  initial run was interrupted. Small-object tests cover class-wise NMS directly.
- `python tests/test_mobile_bev.py`: all 10 checks passed, including targets,
  decode, loss, metrics and the training guard.
- Original HEAD Numba targets and new `assignment=legacy` targets were identical
  for empty, one-box and 50-box randomized inputs.
- Both downloaded epoch-100 checkpoints load strictly and produce finite outputs
  on synthetic CPU input. This verifies compatibility, not accuracy.
- `git diff --check` passed.

Full KITTI AP, CUDA NMS and training have not been rerun in this CPU environment.
Measure both decoder modes on the raw validation dataset before retraining.
