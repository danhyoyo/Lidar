# KITTI overview visualization

## Design

### Goal
Export publication figure components from real KITTI scans: a 3D LiDAR scene,
the same scene with annotated 3D boxes, and Car / Cyclist / Pedestrian zooms.

### Constraints
Use existing KITTI binaries, official labels and calibration only. Never generate
points, fill holes, insert objects, or invent detection scores. Keep datasets and
rendered outputs in ignored artifact directories. Run headlessly with NumPy,
Matplotlib and Pillow; no GPU or Open3D required.

### Known context
KITTI is stored in separately extracted `Code/Kitti_Raw/{velodyne,calib,label_2}`
archives. 284 training frames have all three requested classes. Frame 005663 has
4 Cars, 4 Cyclists and 13 Pedestrians, with usable point returns in every class.
No inference predictions or model checkpoints are present in the workspace.

### Risks
Camera-frame labels must be transformed with full rectification and extrinsics.
Yaw-only approximations can misalign boxes. Existing `Calibration` validation
rejects real P2 matrices with nonzero final-column depth offsets; the visualization
can reuse `parse_calibration` and `camera_box_corners` without involving P2.
Sparse or occluded objects can make poor zoom examples. Many box labels can overlap.

### Options
1. Open3D screenshots: interactive navigation, but extra dependency and headless
   graphics setup. Medium complexity.
2. Matplotlib orthographic projection of actual XYZ and exact box corners:
   deterministic, headless, configurable view, PNG/SVG exports. Small complexity.
3. BEV-only figure: simple and fast, but loses height and does not match the reference.

### Recommendation
Use orthographic 3D projection with a common scene camera and class-specific zoom
views. Rank complete frames by class diversity, visible counts and point support.
Use KITTI ground truth by default with explicit provenance; accept genuine KITTI
prediction text files optionally, preserving their scores.

### Acceptance criteria
- Choose a real frame with all three classes and list candidate frame counts.
- Export separate high-resolution scene / boxes / three zoom PNG and SVG files,
  plus a composite preview and reproducibility metadata.
- Zoom points are exact subsets of the selected binary, with no upsampling.
- Verify exact camera-to-LiDAR corners and point membership with regression tests.
- Verify output artifacts visually and run the CLI on real data.

## Implementation

### Goal
Add a reusable figure exporter and document reproducible commands.

### Assumptions
Ground-truth boxes are authorized as illustration; they must remain identified
as ground truth, never as measured detector predictions. Default frame selection
requires usable zooms of all three classes from one scan.

### Plan
1. Geometry and input regression tests
   - Files: `Code/tests/test_kitti_overview_visualization.py`.
   - Change: assert exact calibrated corners / membership, malformed binaries,
     class-diverse frame choice, real score parsing.
   - Verify: `python3 -m pytest Code/tests/test_kitti_overview_visualization.py -q`.
2. Exporter
   - Files: `Code/tools/visualization/export_kitti_overview.py`.
   - Change: dataset layout detection, selection report, orthographic renderer,
     zoom selection, actual prediction loading, PNG/SVG/JSON exports.
   - Verify: the focused tests and the actual KITTI export command.
3. Documentation and visual verification
   - Files: `Code/tools/visualization/README_kitti_overview.md`.
   - Change: commands, output descriptions, ground-truth explanation, view controls.
   - Verify: inspect the generated preview and metadata, compile script, run tests.

### Risks & mitigations
Use full transformed camera corners; test nonzero pitch/roll and original KITTI
calibration. Retain original point indices in metadata and hashes of source files.
Keep input data read-only and avoid changing existing training/evaluation modules.

### Rollback plan
Remove only the new exporter, tests, documentation and generated artifact folder;
the existing pipeline and datasets are untouched.

## Checkpoint inference implementation record

The user supplied a selected epoch-50 hist14/grouped/IQA checkpoint. Its saved
config and tensors specify BEV-only outputs, so the extension exports true BEV
predictions with planar footprints on the 3D scan rather than inventing vertical
predictions. `predict_kitti_overview.py` loads embedded configuration, validates
checkpoint identity, loads weights strictly and uses the existing encoder and
decoder. Tests cover rotated XY geometry, unrestricted vertical crop membership,
row validation and authoritative checkpoint configuration selection.

Fresh inference was run on frame 005663 (train) and 004338 (validation), using CPU
FP32, score/class threshold 0.3 and NMS threshold 0.1. The validation manifest's
frame IDs match the run's archived 3,769-frame prediction set. Outputs and exact
input point subsets are retained under ignored `Result/artifacts/` folders.
