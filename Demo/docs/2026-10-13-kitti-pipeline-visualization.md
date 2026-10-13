### Goal
- Export a real KITTI frame through training augmentation, hist14 grouping, encoding, actual checkpoint features, heads and decoded detections.

### Constraints
- Use measured KITTI points and real train-only object crops. Preserve checkpoint parameters, config and decoding. Run headless on CPU and save publication figures plus numeric tensors.

### Known context
- Frame 005663 and the P4 hist14/SGFPN/focal checkpoint are available. No prepared local GT database exists. The checkpoint predicts BEV; optional cuboids use the existing measured-LiDAR vertical postprocess.

### Risks
- A voxel diagram must not imply a separate sparse 3D backbone. Augmented inference is a training-path demonstration, not validation AP. Feature color scales are display normalization, not class confidence.

### Options
1. Schematics only: quick but do not show real checkpoint activations.
2. Forward hooks plus the real augmentor and encoder: exact tensors and reproducible outputs; needs a small real donor database.

### Recommendation
- Use option 2. Build a documented subset of training donors, record source hashes and seed, and feed the resulting augmented cloud into one strict checkpoint forward pass.

### Acceptance criteria
- Separate raw/augmented/occupied-cell/hist14/backbone/head/detection images and one overview.
- Full XYZI clouds, bin indices/counts, hist14 input, intermediate feature tensors, head outputs and decoded rows in NPZ/NPY.
- Occupied four-band cell counts reproduce hist14's first four channels.
- Hooks are removed and checkpoint state is unchanged after capture.
- Real checkpoint run passes shape/finite/hash checks; visually inspect overview and individual panels.

### Assumptions
- Default frame 005663, seed 2, current P4 checkpoint. hist14 is the repository's 14-channel encoding; rich8 is a different schema.

### Plan
1. Add shared exporter helpers and regression checks.
   - Files: `tools/visualization/export_kitti_pipeline.py`, `tests/test_kitti_pipeline_visualization.py`.
   - Change: Actual augmentor, diagnostic four-height-band quantization, removable forward hooks and strict decode.
   - Verify: `PYTHONPATH=Code:Code/detector:Code/tools/kitti_training_pipeline PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Code/tests/test_kitti_pipeline_visualization.py -q`.
2. Render and run the real frame.
   - Files: exporter; `Demo/outputs/pipeline/005663/`.
   - Change: PNG/SVG per stage, raw tensors and provenance JSON, README explaining dimensions and sources.
   - Verify: Run exporter with actual KITTI/checkpoint; check all finite tensors, hist14 reconstruction, unchanged checkpoint SHA and output PNG dimensions; inspect previews.
3. Document and package.
   - Files: visualization README, Demo README/code/docs/manifest/verification.
   - Change: Rerun instructions and clickable output paths; preserve existing demos.
   - Verify: Focused existing visualization tests, CLI help, package hashes and links.

### Risks & mitigations
- Explicitly identify donor database subset, probability handling, augmented input and BEV vs LiDAR-estimated vertical geometry. Use existing implementations for data conversion, augmentation, encoding and decoding.

### Rollback plan
- Remove the new exporter/demo output only; existing checkpoints, datasets and visualization entry points are not mutated.
