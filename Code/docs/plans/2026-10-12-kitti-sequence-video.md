### Goal

Create a real 174-frame KITTI video with measured LiDAR, checkpoint BEV detections
and classification heatmaps, in verified raw-frame order, at 10 FPS.

### Assumptions

- Existing official mapping and local .bin files cover raw frames 77–250 of drive
  2011_09_26_drive_0014_sync. Mapping indices are not timestamps.
- Use the previous best.pt with its embedded config; inference is CPU FP32.
- The checkpoint predicts BEV boxes. This video shows BEV detections and actual
  input XYZ; it does not add estimated vertical dimensions.

### Plan

1. Verify timeline and reuse model loading across frames.
   - Files: `Code/tools/visualization/predict_kitti_overview.py`,
     `Code/tests/test_kitti_sequence_video.py`.
   - Change: Extract a reusable inference runner without changing the single-frame
     API. Validate selected IDs against the official permutation and mapping.
   - Verify: Focused pytest suite; compare frame 005663 against saved predictions.
2. Implement stable rendering and encoding.
   - Files: `Code/tools/visualization/export_kitti_sequence_video.py`.
   - Change: Persistent three-panel renderer; save per-frame predictions, actual
     heatmaps and provenance. Encode consecutive PNG frames with FFmpeg.
   - Verify: Render a short sequence and inspect its preview; ffprobe frame count,
     dimensions, codec, FPS and duration; focused rendering/timeline tests.
3. Export the full sequence and document reproduction.
   - Files: `Code/tools/visualization/README_kitti_overview.md`,
     `Result/artifacts/kitti_sequence_video/` (ignored generated artifacts).
   - Change: Run all 174 frames, write MP4, preview and source manifest/metadata.
   - Verify: Check all input hashes and saved probabilities; decoded video has
     174 frames and duration 17.4 seconds. Run related visualization tests.

### Risks & mitigations

- Renumbered Object IDs: verify every ID using original KITTI mapping and require
  consecutive raw frames in one drive.
- Sparse or empty detections: render empty lists without inserting boxes.
- Camera/colour flicker: fixed camera, scene bounds, height range and heatmap 0–1.
- CPU rendering time: load checkpoint once, reuse artists and log progress.
- Demo split includes training frames: record split membership; no accuracy claims.

### Rollback plan

Keep earlier image outputs intact and all video artifacts in a separate directory.
The original infer_frame interface stays available. Remove only new outputs or
revert the focused inference wrapper/renderer changes if needed.
