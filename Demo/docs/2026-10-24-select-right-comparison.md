### Goal
Keep left scan 003736 unchanged and select a right scan where actual Ours output
is better than MobilePIXOR in both detection and the displayed grayscale heatmap.

### Assumptions
Use the existing 32 official validation candidates. Keep score=0.30, NMS=0.10,
the same display ROI, native heatmap cells, and fixed grayscale [0,1].
Detection uses same-class BEV IoU 0.70/0.50/0.50 as before. Heatmap uses local
maxima of max-class sigmoid probabilities above 0.30, with one-to-one matching
to GT centers within 1 metre. Evaluate merged maps because that is what is drawn.
These heatmap criteria describe a qualitative illustration, not official AP.

### Plan
1. Run MobilePIXOR on all candidates and audit saved Ours forwards.
   - Files: Demo/code/select_kitti_comparison_scene.py; a new search directory.
   - Change: compare fixed-threshold detection/heatmap errors in the same ROI.
   - Verify: checkpoint/source hashes, logits sigmoid parity, native grid geometry.
2. Rank joint improvements and inspect the best example.
   - Files: comparison ranking CSV/JSON and saved per-model outputs.
   - Change: require fewer Ours detection errors and better heatmap F1, with no
     increase in missed GT peaks or spurious peaks. Prefer few Ours errors.
   - Verify: independently recompute selected GT/detection and heatmap matches.
3. Export the new two-scene figure from fresh forwards.
   - Files: new PNG/SVG and selection metadata.
   - Change: replace right frame only, retaining left frame, 42 mm GT height,
     model sizes, captions, crop and grayscale scale.
   - Verify: exact left-column array parity; inspect both figure panels.

### Risks & mitigations
Save the selection context and every candidate metric outside the figure.
Do not alter predictions, crop, brightness, smoothing or thresholds to improve
the comparison. Use a separate output directory to preserve previous figures.

### Rollback plan
Retain prior output and all checkpoint/data files. New search/export artifacts
can be removed independently.

### Results
Compared 31 candidates after excluding the fixed left scene. Eight have a joint
advantage under the stated criteria. Selected validation scene 003034 has six
GT objects (four Car, two Cyclist) in the unchanged display ROI.

Detection: Ours TP=6/FP=0/FN=0; MobilePIXOR TP=5/FP=3/FN=1.
Heatmap peaks: Ours TP=6/FP=0/FN=0; MobilePIXOR TP=6/FP=2/FN=0.
Ours foreground probability mass fraction is 0.36347 versus 0.26359; mean
background probability is 0.003063 versus 0.005817. These describe raw merged
heatmaps using the same fixed metric windows, with no rendering adjustment.

Both checkpoints were rerun freshly for the final 003736/003034 figure. Saved
outputs equal the search arrays exactly, and independent bipartite assignments
confirm the selected detection and heatmap matches. All 31 source/checkpoint
and result hashes were audited. The entire left half of the PNG equals the
previous figure pixel for pixel; its raw predictions/probabilities also match.
PNG/SVG were visually inspected. Output is 6614x3892 at 600 DPI, with the same
three row captions and 42 mm ground-truth height. Results and selection context
are retained in comparison_scene_search and the final selection.json.
