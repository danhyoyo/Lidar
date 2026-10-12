# Point-cloud-derived 3D visualization of BEV predictions

## Design

### Goal
Produce cuboid detection panels and three class zooms using the supplied model's
real BEV predictions and vertical geometry estimated from the same KITTI scan.

### Constraints
The checkpoint has no vertical head. Keep predicted class, score, X/Y, length,
width and yaw exactly unchanged. Do not use ground-truth boxes, class height
priors, synthetic points or additional model training. Explicitly describe the
result as BEV detections with point-cloud-derived vertical extents.

### Known context
Fresh epoch-50 inference is saved for 005663 (train; 20 detections) and 004338
(validation; 16 detections). Full original scans and prediction provenance hashes
are available. The existing renderer accepts actual cuboid corners.

### Risks
Ground, nearby objects and vertical outliers can contaminate a predicted XY
region. Sparse returns can underestimate an object's actual height. A detection
without sufficient measured support must not receive fabricated vertical geometry.

### Options
1. Retrain a vertical head: proper learned 3D detection but requires a new training
   experiment and checkpoint. High cost and beyond this figure-export request.
2. Fill heights from KITTI labels or class priors: easy but not derived from model
   predictions and measured support. Does not satisfy the provenance constraint.
3. Fit vertical extent from original LiDAR: local robust ground plane, above-ground
   point support and robust upper extent; retain the learned BEV geometry. Small
   implementation and deterministic, with measured support recorded.

### Recommendation
Use option 3 for the requested visualization. Keep the previous true BEV outputs
and save a separate lifted figure set with explicit vertical-estimation provenance.

### Acceptance criteria
- Export cuboid scenes and class zoom PNG/SVG for both existing frames.
- Match all retained BEV fields and scores bit-for-bit to their source predictions.
- Verify local sloped-ground fitting, elevated-object support, outlier rejection
  and unsupported detections using meaningful small geometry tests.
- Every zoom point is an exact subset of the original scan.
- Metadata and preview identify the source of vertical geometry and unsupported boxes.

## Implementation

### Goal
Add a reusable BEV-to-3D visualization postprocessor.

### Assumptions
The user requests a 3D figure from the currently supplied BEV checkpoint. A
geometric lifting step is appropriate when explicitly identified; it is not a
claim that the checkpoint directly predicts height or Z.

### Plan
1. Ground and vertical geometry tests
   - Files: `Code/tests/test_kitti_3d_lifting.py`.
   - Change: test slope, outliers, unsupported inputs and BEV-field preservation.
   - Verify: `python3 -m pytest Code/tests/test_kitti_3d_lifting.py -q`.
2. Lifting exporter
   - Files: `Code/tools/visualization/lift_kitti_bev_to_3d.py`.
   - Change: load saved inference and verified scan; robust local ground fit;
     estimate supported vertical geometry; render cuboids and zooms.
   - Verify: focused tests and CLI exports on frames 005663 and 004338.
3. Documentation and artifact review
   - Files: `Code/tools/visualization/README_kitti_overview.md`.
   - Change: provenance, method limits, commands and output descriptions.
   - Verify: inspect images, compare source fields and original point indices,
     compile exporter and run all overview tests.

### Risks & mitigations
Exclude predicted footprints from ground fitting. Select low-return representatives
per XY cell and robustly refit the plane. Reject unsupported planes/heights rather
than introducing class defaults. Remove isolated upper Z components from height
estimation; record all estimator settings and supporting original indices.

### Rollback plan
Remove only the added lifting exporter/tests/docs and its separate output folders.
Original checkpoints, scans and previous BEV/GT figure sets remain intact.
