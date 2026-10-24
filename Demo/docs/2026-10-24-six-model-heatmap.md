# Design

### Goal
Render six genuine checkpoint predictions as three rows and two pairs per row;
each pair contains a LiDAR detection panel and a classification heatmap.

### Constraints
Use embedded configs, strict state dictionaries, original encoders and decoder.
Save source/checkpoint hashes, raw logits, probabilities, predictions and rendering
metadata. Never substitute Gaussian targets for model predictions.

### Known context
`BEVOverviewPredictor` already captures grouped classification heads but only
accepts Hist14. Baselines use binary slices (Legacy35). Original XYZI scans survive
in backup NPZ files; their byte hashes match recorded original KITTI binaries.

### Risks
The available runs differ in training budgets, heads and checkpoint selection.
BEV heads do not predict height. State all differences in the output README;
only draw cuboids supported by measured LiDAR, retaining footprints otherwise.

### Options (2–4)
1. Extend the existing predictor for Legacy35; reuse capture/decode and LiDAR lift.
   Small change, direct training-encoder parity can be tested.
2. Write a separate inference runner. More duplication and risk of decode drift.

### Recommendation
Option 1, plus a standalone Demo comparison exporter. Use grayscale, fixed [0,1]
heatmap scale and identical source/view/thresholds across all six panels.

### Acceptance criteria
- Six distinct checkpoints load strictly and actually forward the same raw scan.
- Layout order matches the request; output PNG and SVG are visually inspected.
- Raw heatmaps are sigmoid logits without smoothing, GT, or per-panel rescaling.
- Original scan bytes and checkpoint hashes are verifiable from metadata.

# Implementation plan

### Goal
Export reproducible six-model pointcloud/heatmap comparisons.

### Assumptions
Ours is Final-h32 best_ap epoch 130. PC-GTA corresponds to stored hybrid GT
sampling. Use best_loss where no best_ap exists. Frame 000134 is validation;
005663 is training. These are qualitative examples, not an AP or controlled
single-component ablation.

### Plan
1. Add Legacy35 encoding to `Code/tools/visualization/predict_kitti_overview.py`.
   - Files: predictor and `Code/tests/test_kitti_comparison_encoding.py`.
   - Change: use the training rasterizer without altering Hist14 inference.
   - Verify: focused pytest comparing model inputs with Dataset.encode_input.
2. Create `Demo/code/export_kitti_model_comparison.py` and checkpoint manifest.
   - Files: script, `Demo/configs/model_comparison.json`.
   - Change: verify backup source hash, load six models, capture outputs, lift
     supported boxes from LiDAR and render 3x2 paired layouts.
   - Verify: strict load, finite tensors, sigmoid parity, actual forward/decode,
     source hashes and saved detection arrays for both real frames.
3. Inspect rendered figures and document output provenance.
   - Files: generated output README/metadata and Demo README link.
   - Verify: image inspection, artifact audit, relevant existing pytest suites.

### Risks & mitigations
Preserve checkpoint configs and weights; no architecture substitutions or input
augmentation during evaluation. Unsupported cuboids remain BEV footprints.

### Rollback plan
Remove only the new script/manifest/test and revert the small predictor extension;
existing outputs and user edits are untouched.

# Verification results

All six checkpoints loaded strictly with matching identities. Each ran new
inference on the two hash-verified original scans. Main and BEV figures were
visually inspected; PNG dimensions are 4160x3520, with SVG and individual pairs.
25 focused tests passed. The output `verification.json` verifies six distinct
checkpoint hashes, source hashes, saved file/config hashes, finite outputs,
exact sigmoid-logit parity and lifted footprints. No GT or augmentation enters
inference. Ours Final-h32 epoch 130 was explicitly confirmed by the user.
