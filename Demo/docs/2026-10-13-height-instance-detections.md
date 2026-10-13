### Goal
Render real raw-frame model detections on height-colored BEV/3D scenes, with
consistent instance colors and a zoom pair for every visible prediction.

### Assumptions
- KITTI frame 005663 and P4 checkpoint cache are available.
- Background returns keep the existing measured-Z turbo scale.
- Every detection receives a distinct color and ID; measured foreground support
  points use that color. These IDs are per-frame detections, not track IDs.
- BEV scores/geometry are from the checkpoint. Z/height are LiDAR estimates.

### Plan
1. Implement verified cache loading and instance rendering.
   - Files: `Code/tools/visualization/export_kitti_height_detections.py`
   - Change: Check raw scan/config/prediction hashes and rows. Preserve scores,
     apply the existing measured-point lifting, and render both scene views.
   - Verify: Compare saved rows and input hash with inference metadata.
2. Export every visible object's BEV/3D zoom and measured source-point indices.
   - Files: same exporter; `Demo/outputs/height_detections/005663/`
   - Change: Share palette and height scale across scene/zooms; include contact
     sheets and representative Car/Cyclist/Pedestrian examples.
   - Verify: Inspect scene and class previews; validate PNG/SVG and point crops.
3. Package code/docs and verify focused invariants.
   - Files: `Code/tests/test_kitti_height_detections.py`, visualization README,
     `Demo/code/`, `Demo/docs/`, `Demo/README.md`, inventory and verification JSON.
   - Change: Document how to reproduce using the existing inference cache.
   - Verify: Run focused tests, CLI help, compilation, and artifact validation.

### Risks & mitigations
The cache must be tied to the raw input, not the augmented pipeline frame.
Reject mismatched prediction/config hashes. Do not invent missing vertical
geometry; report unsupported objects and keep their BEV predictions visible.

### Rollback plan
Remove the new exporter/results/README links. Existing raw renders and upstream
predictions remain unchanged.
