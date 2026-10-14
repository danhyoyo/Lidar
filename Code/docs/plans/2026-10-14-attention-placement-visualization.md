# Attention placement visualization

Use one real KITTI scan with many annotated Car/Pedestrian/Cyclist objects.
Load P16 strictly with its embedded configuration before any interventions.

1. Rank available raw scans by calibrated in-range object counts; use GT only
   for selection/overlays, never as network input.
2. Reuse the trained trunk, neck, heads, C4 Focal and SimAM scalar. Create
   deterministic, freshly initialized Focal C3/C5 modules with matching widths.
3. Run all 16 off/C3/C4/C5 combinations plus reverse ordering at shared stages.
   Shared-stage default order is Focal then SimAM. No training or gain overrides.
4. Capture stage RMS, final features, Gaussian class probabilities and gates.
   Plot common-scale matrices, channel RMS differences, per-class comparisons,
   raw BEV/3D points with GT, and provide an HTML gallery and numeric NPZ files.
5. Verify native placement reproduces the untouched checkpoint, temporary hooks
   and module replacements are restored, tensors are finite, and outputs exist.

These are checkpoint interventions, not independently trained architecture
ablations. C3/C5 Focal weights are untrained. Disabling both modules is an
intervention baseline, not a separately trained convolution detector.
