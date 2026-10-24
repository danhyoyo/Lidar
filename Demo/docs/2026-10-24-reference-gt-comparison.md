### Goal
Export the requested reference-style three-row, two-scene figure from fresh
MobilePIXOR/Ours checkpoint inference and official KITTI ground truth.
Revision: zoom the measured pointcloud crop, remove black padding, render only
top-down BEV footprints, and load Ours from the explicit checkpoints/130epoch.pt.

### Assumptions
Use the two verified original frames 000134 and 005663, and the MobilePIXOR and
Final-h32 Ours checkpoint, now recorded in reference_bev_zoom.json.
Each model row has two detection/heatmap pairs. Only the three requested row
captions appear. Ground truth, predictions and heatmaps share the same BEV crop.

### Plan
1. Add a dedicated exporter.
   - Files: Demo/code/export_kitti_reference_comparison.py.
   - Change: reuse checkpoint loading, original encoders, decoder
     and source hash verification; run both models on both original scans.
   - Verify: execute exporter and inspect strict load and forward logs.
2. Export and inspect the figure.
   - Files: Demo/outputs/heatmap/reference_gt_comparison_bev_zoom/.
   - Change: black backgrounds, green measured points, yellow BEV boxes, grayscale
     actual classification probabilities, and serif row captions.
   - Verify: inspect PNG, compare saved logits/probabilities and validate hashes.

### Risks & mitigations
Display only the exact model BEV footprints, with no height estimates. Do not use labels
for prediction or heatmaps. Keep a fixed [0,1] scale and native heatmap cells.
Frame 005663 is a training example; record split provenance outside the figure.

### Rollback plan
Remove only the new exporter, plan and its dedicated output directory. Preserve
all earlier scripts, figures, datasets and checkpoint files.

### Revision verification
The exporter loads Ours directly from checkpoints/130epoch.pt and enforces epoch
130 using embedded checkpoint metadata. A new output directory retains the
previous figure. Shared crop X=[5,38], Y=[-26,20] m, physical panel proportions
match the crop, and outer margins are reduced from 2 mm to 0.6 mm.

### Replacement frame and layout revision
Replace column 1 with validation scan 004338, downloaded using byte ranges from
the official KITTI Velodyne archive. Validate its ZIP CRC32 and original SHA256
eb551c721962d16dc1f8911f8e24dcd90c1e48259ec2a4e38954eb59907849aa
against the pre-existing scan record. Extract matching labels/calibration from
the previously verified official archives. Run both checkpoints freshly.

The new output directory is Demo/outputs/heatmap/reference_bev_004338_005663.
Set canvas width to 280 mm, GT height to 35 mm and model panel heights to about
50 mm. Center smaller GT panels in their scene columns while retaining the same
metric crop and aspect ratio. Preserve all earlier figures. Audit checkpoint
epoch/path, source hashes, logits/probabilities, footprint coordinates, figure
hashes and SVG captions. Compare the new column 2 arrays with the previous
export to verify that its predictions and heatmaps remain exactly identical.

### Second replacement and larger ground truth
Use validation scan 000008 from the official KITTI archive as the left scene,
verified against the ZIP member CRC32 3751035586. Record its SHA256 as
9db1fe26d240917dfd64e6125f77a78f7cff6aa4bd5b8eb87f73fbd7a789dd98.
Raise GT height from 35 to 42 mm, retaining model panel sizes and source frame
005663 on the right. New output: Demo/outputs/heatmap/reference_bev_000008_005663.
Both models run freshly; Ours still loads the explicit epoch 130 checkpoint.
Expose --frames and --ground-truth-height-mm to support later layout adjustments.

### Wider left scene, red footprints and larger ground truth
Visually inspected six already verified candidates: 000382, 001225, 001337,
000053, 000541, 001552. Selected 001225 for its wider spatial distribution of
measured points and varied object placement, while retaining zero Ours errors
among the five GT objects in the fixed ROI. MobilePIXOR has TP=4, FP=1, FN=1.
Preserve right scene 003034 and exact raw output arrays for both models.

Change all GT/predicted footprint strokes from yellow to #ff3030. Increase GT
height from 42 to 48 mm (14.3%) and retain all model-panel sizes, captions, crop
and heatmap rendering parameters. Fresh inference was run for both scenes and
both checkpoints. Verification confirms red SVG strokes with no old yellow,
unchanged right arrays, exact BEV geometry, logits/sigmoid parity, and the three
requested captions. Output PNG is 6614x4033 at 600 DPI, visually inspected along
with its SVG. The dedicated output directory is reference_bev_001225_003034_red.
