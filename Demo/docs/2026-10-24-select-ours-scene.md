### Goal
Find a genuine validation scan with the fewest Ours BEV detection errors for the
left column and render it with the existing reference-style layout.

### Assumptions
Use the explicit Final-h32 checkpoints/130epoch.pt, fixed score 0.30 and NMS 0.10.
Use the unchanged display ROI and class-dependent BEV IoU thresholds: Car 0.70,
Pedestrian/Cyclist 0.50. Match same-class boxes greedily by descending score, as
in the existing evaluator. Count every displayed GT, including difficult objects.

### Plan
1. Prepare diverse official KITTI validation candidates with at least 5 GT boxes.
   - Files: Demo/code/select_kitti_qualitative_scene.py; dedicated search output.
   - Change: select candidates from distinct drives before viewing predictions;
     fetch only individual ZIP members using byte ranges and verify ZIP CRC32.
   - Verify: validation membership, unique original files, sizes, CRC32 and SHA256.
2. Run Ours and rank candidates.
   - Files: search output with real predictions, metrics, and matching records.
   - Change: minimize FP+FN, then maximize F1 and matched GT count. MobilePIXOR
     performance does not enter the selection criterion.
   - Verify: strict checkpoint epoch/path, independent matching audit and outputs.
3. Render the best tested scene and unchanged right column.
   - Files: a new PNG/SVG and metadata directory.
   - Change: reuse the current BEV figure and GT size; fresh forwards for both models.
   - Verify: compare rendered predictions with ranked outputs; inspect image and hashes.

### Risks & mitigations
This is explicitly a selected qualitative example, not an unbiased benchmark.
Save all candidate metrics and the selection rule outside the title-free figure.
Do not edit, remove or fabricate predictions; keep thresholds/crop fixed.

### Rollback plan
Retain previous figures and checkpoints. Remove only new selection outputs/code
if needed; downloaded official data may be retained for reproducibility.

### Results
32 validation scans from 14 drives passed official ZIP CRC32 and size checks.
Seven candidates achieved FP=FN=0. The stated tie-break selects 003736 with
seven GT boxes and seven correct predictions in the unchanged display ROI.
MobilePIXOR has six matched predictions, zero false positives and one miss in
the same ROI. Ours has nine total predictions; two are outside the pre-existing
ROI and remain present in the raw NPY. No new crop or threshold was introduced.

The final exporter ran both checkpoints freshly on 003736 and 005663. Ours
predictions match the search outputs exactly. Independent bipartite matching
also confirms all seven displayed detections exceed the required class IoU.
All 32 scan, label, calibration, checkpoint and prediction hashes were audited,
along with sigmoid parity for stored logits and probability maps. PNG/SVG were
visually inspected; output PNG is 6614x3892 at 600 DPI. The figure audit confirms
only the three requested row captions, exact predicted BEV footprints and the
unchanged 42 mm GT panel height. Ranking CSV/JSON and selection.json retain the
selection context outside the figure.
