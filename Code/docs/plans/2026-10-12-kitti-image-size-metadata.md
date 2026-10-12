### Goal
Use the user's per-frame left-camera dimensions for reference AP3D/APBEV without extracting RGB images.

### Assumptions
The supplied `kitti_image_sizes_v1` JSON covers all 7,481 training IDs. KITTI AP calculations and prediction projection stay unchanged.

### Plan
1. Validate and copy metadata into the project.
   - Files: `configs/datasets/kitti_image_sizes.json`, `Kitti_Raw/kitti_image_sizes.json`.
   - Change: Preserve exact downloaded bytes and verify ID coverage and dimensions.
   - Verify: Compare SHA256, count 7,481 IDs and report dimension frequencies.
2. Add one shared projection-input loader and integrate consumers.
   - Files: `tools/kitti_training_pipeline/image_metadata.py`, `common.py`, `evaluate_kitti_3d.py`, `train.py`, `tools/benchmarks/audit_kitti_assets.py`.
   - Change: Prefer validated JSON when present, otherwise retain PNG-header loading; hash whichever input is used.
   - Verify: Regression tests for variable sizes, missing IDs, invalid metadata, PNG equivalence and stale audit detection.
3. Use bundled metadata in Colab.
   - Files: `3D_Lidar_Detection_Pipeline_Robust.ipynb`, pipeline README, workflow tests.
   - Change: Copy validated metadata to raw training root, avoiding image extraction when JSON is available.
   - Verify: Execute actual extraction cell with AP selection and JSON only; compile notebook cells and run relevant tests.

### Risks & mitigations
Reject incomplete or malformed JSON before inference. Keep per-frame dimensions and preserve the pinned evaluator sources. Existing PNG workflows remain supported.

### Rollback plan
Remove the JSON from the raw training root and omit `evaluation.image_sizes_json` to restore PNG inputs; revert only these implementation edits if needed.
