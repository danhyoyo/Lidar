# Proposal 3 (draft): calibrated 3D detection on a lightweight dense-BEV LiDAR detector

Status: **prototype implemented, not yet validated.** All changes below are
backward-compatible: existing A0–A4 and B0–B3 configs load and train exactly as
before. The single reported data point is the seed-42 / 50-epoch screening on
L4; every claim below is contingent on the A5/A6 + B-factorial runs.

## Recent changes in this branch (2026-09-17)

1. **RichBEV-11** in `detector/core/datasets/utils_1/preprocess.py` (+
   `tools/kitti_training_pipeline/common.py`): the rich encoder is now
   parameterized by `n_bands`. `rich11` emits 6 normalized height-band
   occupancies + `[z_max, z_mean, i_max, i_mean, density]` = 11 channels (finer
   vertical structure than `rich8`'s 3 bands). `rich8` behavior is unchanged.
2. **Bounded gate** in
   `detector/core/models/backbones/mobilepixor_coordinate_attention.py` (+
   `model.py`): optional `model.gate_scale` gives `G = 1 + gate_scale·tanh(x)`
   in `[1-gate_scale, 1+gate_scale]`, zero-init equivalent to sum-FPN, so the
   gate can no longer fully suppress the high-resolution lateral path. Configs
   without `gate_scale` keep the legacy `2·sigmoid` gate.
3. **Gate diagnostics** in `tools/kitti_training_pipeline/train.py`: run with
   `--gate-diag-batches N` to log per-epoch `gate_c4/c3_gate_mean/std/sat_low/
   sat_high` for enabled gate variants.
4. **New configs** `configs/kitti/mobilebev/a5_rich11_center3d.json`
   (RichBEV-11 + sum-FPN) and `a6_rich11_sgfpn_center3d.json` (RichBEV-11 +
   bounded gate). Tests for rich11, the bounded gate and config consistency
   added to `tests/test_mobile_bev.py` (19/19 pass). A5/A6 entries added to
   `3D_Lidar_Object_Detection_Notebook_clean.ipynb`.

## Research proposal: contributions

The thesis already claims the MobilePIXOR backbone, CoordAtt at C5, the UWAG
loss, geometric augmentation and the BEV head. Those are deliberately **excluded**
here. The candidate contributions of this paper, each with its placement in the
pipeline, rationale, and current evidence:

### C1. RichBEV — compact statistical BEV representation
- **Placement:** input representation / encoder (front-end, replaces the
  35-channel z-occupancy tensor fed to the backbone stem).
- **Idea:** 8–11 handcrafted statistical channels (height bands, `z_max`,
  `z_mean`, `i_max`, `i_mean`, log density) instead of 35 binary z-slices.
  `n_bands` makes the vertical resolution of the representation a first-class
  hyperparameter.
- **Why useful:** cuts input bytes ~69–77% and first-conv GMAC by ~4.4 GMAC,
  directly shrinking host-to-device transfer on edge hardware while keeping
  enough height/intensity/density signal for small objects (Pedestrian/Cyclist).
  The `n_bands` knob is a controlled way to trade vertical resolution for
  bandwidth — the diagnosis target for the A4 pedestrian regression.
- **Evidence:** A2 (rich8) matches A1 BEV mAP with +0.55 3D mAP and +4.6 fps;
  A5 (rich11) is the follow-up isolating the vertical-resolution hypothesis.

### C2. Scale-Gated FPN (SG-FPN) + bounded gate
- **Placement:** neck / feature fusion (top-down path, after CoordAtt-C5, at
  the C4 and C3 lateral merges).
- **Idea:** two depthwise, zero-initialized gates `G = 2·sigmoid(DWConv(L+U))`
  (480 params) that reduce exactly to the fixed sum-FPN at initialization, plus
  an optional bounded variant `G = 1 + gate_scale·tanh(...)` in [0.5, 1.5] that
  cannot erase the high-resolution lateral branch.
- **Why useful:** adaptive fusion for ~0.1% parameter overhead with a provable
  identity-to-baseline start (clean ablation); the bounded variant directly
  targets the small-object-feature suppression failure mode seen in A4.
- **Evidence:** A3 (gates) is the best single variant in the screening
  (3D mAP 70.74, Ped +0.43 vs A1); A6 tests the bounded gate fix.

### C3. Minimal 3D Center Head
- **Placement:** detection head (replaces the 4-tensor BEV head; extends only
  the two final 1×1 convolutions of `offset` and `size`).
- **Idea:** extend `offset` 2→3 (`z_center`) and `size` 2→3 (`log h`) to output
  a full `(x, y, z, w, l, h, yaw)` box with +34 parameters and no new tower or
  ONNX output name.
- **Why useful:** shows a dense 2D-BEV detector can become a direct 3D box
  detector for almost zero architectural cost — a counterpoint to
  sparse-convolution 3D detectors (SECOND/CenterPoint) and the enabler for C5.
- **Evidence:** A1–A6 all run Center3D; the BEV→3D gap (BEV 75.2 → 3D 69.1 mAP)
  quantifies exactly what the vertical regression costs.

### C4. Gaussian Wasserstein Distance (GWD) geometric loss
- **Placement:** loss function (replaces the yaw-aware axis-aligned BEV overlap
  approximation as the geometric term).
- **Idea:** rotated boxes as 2D Gaussians, closed-form squared Wasserstein
  distance with normalized `1 - 1/(1+log(1+D²))` form; handles boundary
  discontinuity and yaw π-periodicity analytically.
- **Why useful:** a continuous, rotation-invariant regression signal that does
  not need 3D IoU gradients, keeping the loss cheap and stable (FP32, finite
  guards already in place).
- **Evidence:** B1/B3 in the ProbGeo-UQ factorial isolate its contribution.

### C5. Heteroscedastic NLL aleatoric localization uncertainty
- **Placement:** tiny head extension (offset/size final convs split into mean +
  log-variance, +102 params) + loss function (masked Gaussian NLL).
- **Idea:** per-detection input-dependent `log_var` for
  `[x, y, z, log w, log l, log h]`, trained with
  `0.5·(exp(-s)·r² + s)` under FP32/clamped `log_var`.
- **Why useful:** gives each 3D box a calibrated localization uncertainty at
  negligible cost, naturally tied to LiDAR density (coupled with C1's density
  channel) — enables risk-aware planning and is the paper's strongest novelty
  once it beats cheap baselines (constant variance, range + point-count).
- **Evidence:** B2/B3; `evaluate_uncertainty.py` reports NLL/coverage/ENCE/AURC
  on locked calibration/test manifests.

### C6. Calibrated-uncertainty evaluation protocol (BEV/3D AP + UQ)
- **Placement:** evaluation methodology / tooling (3D IoU, BEV+3D AP R40,
  distance bands, paired-bootstrap CI, and the UQ metric suite) — a contribution
  outside the architecture.
- **Idea:** a pre-registered, multi-seed factorial (A1–A6, B0–B3) with locked
  checkpoint selection (3D mAP Moderate), reproducible manifests/hashes, and a
  statistical protocol that reports both accuracy and uncertainty quality.
- **Why useful:** reproducibility is itself a contribution for an
  engineering-adjacent venue, and it is the only way the A4/A6 and B-factorial
  claims can be defended against single-seed noise.
- **Evidence:** the existing baseline manifest, selectors and evaluators are
  already hash-locked; the missing piece is the completed 3-seed run.

**Working paper claim (if gates pass):** a ~0.6M-parameter dense-BEV LiDAR
detector upgraded from 2D BEV to full calibrated 3D detection at <2% parameter
overhead and reduced input bandwidth, with per-detection localization
uncertainty validated against cheap baselines. This excludes all thesis
components by construction.

---

# Reproduce UWAG + CoordAtt + geometric augmentation

This directory contains the preparation, training, evaluation, ONNX export and
TensorRT utilities for the MobilePIXOR detector in `detector/`.

## Environment

The reproduced run used Python 3.10, PyTorch 2.11.0+cu128, CUDA 12.8 and an
NVIDIA RTX 5060 Ti. Install a CUDA-compatible PyTorch build first, then run:

```bash
python3 -m pip install -r requirements-kitti.txt
```

TensorRT is optional and is only required for engine export/evaluation.

## Download KITTI

Download the KITTI Object Detection Velodyne, calibration and training-label
archives yourself. Do not commit the dataset to this repository. Expected layout:

```text
/path/to/KITTI/object/training/
├── velodyne/
├── calib/
└── label_2/
```

## Prepare the data

Run commands from the repository root. The committed manifests reproduce the
same 5,984/1,497 frame split used by the reported experiment.

```bash
python3 tools/kitti_training_pipeline/prepare_kitti.py \
  --kitti-root /path/to/KITTI/object \
  --output-root data/kitti/processed \
  --config-output data/kitti/generated_baseline.json \
  --train-ids splits/kitti/train.txt \
  --val-ids splits/kitti/val.txt
```

## Train

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/kitti_uwag_coordatt_aug.json \
  --detector-root detector \
  --output-root artifacts/kitti \
  --run-name uwag_coordatt_aug_bf16_seed42 \
  --num-workers 2
```

The config uses MobilePIXOR with Coordinate Attention, UWAG task weighting,
adaptive Gaussian targets and one geometric transform with probability 0.5:
rotation +/-20 degrees, scaling 0.95-1.05, or Gaussian translation scale 0.4.
Optimization uses Adam, learning rate 3e-4, weight decay 5e-4, 100 epochs,
physical batch 2, accumulation 2, BF16 and seed 42.

Resume with `--resume /path/to/checkpoint.pt`. The reproduced run became
non-finite at epoch 81, so its selected evaluation checkpoint is epoch 55.

## Evaluate

```bash
python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
  --name uwag_coordatt_aug_bf16_seed42_best \
  --backend pytorch \
  --model /path/to/best.pt \
  --config configs/kitti/kitti_uwag_coordatt_aug.json \
  --detector-root detector \
  --kitti-root /path/to/KITTI/object \
  --split splits/kitti/val.txt \
  --output artifacts/kitti/evaluation_best_val.json \
  --device cuda
```

This reports local loader-aligned KITTI-style rotated BEV AP R40, not a
submission to KITTI's hidden official test server. The reproduced report is in
`results/kitti/uwag_coordatt_aug_bf16_seed42/`.

Use `export_onnx.py`, `build_tensorrt.py`, `compare_models.py` and
`deploy_engine.py` for deployment experiments. TensorRT engines are tied to
the local CUDA/TensorRT/GPU environment and should not be committed.

## MobileBEV-Lite

The frozen design and experiment protocol are in
`docs/mobile_bev_lightweight/SPEC.md` and `docs/mobile_bev_lightweight/PLAN.md`.
The four controlled variants are under `configs/kitti/mobilebev/`; A1 is the
35-channel Center3D baseline and A4 enables RichBEV-8 plus SG-FPN.

Seed-42 screening showed A4 regresses Pedestrian Moderate 3D AP (−2.49 vs A1).
Two pre-registered follow-up variants under `configs/kitti/mobilebev/` test the
diagnosis (see the SPEC addendum): **A5** `rich11_center3d.json` (finer 6-band
RichBEV-11, isolates the vertical-resolution hypothesis) and **A6**
`rich11_sgfpn_center3d.json` (RichBEV-11 + a bounded `gate_scale=0.5` gate in
[0.5, 1.5] that prevents the gate from suppressing the high-resolution lateral
path). Existing A1–A4 configs keep the legacy `2·sigmoid` gate, so their
results remain valid. Train with `--gate-diag-batches N` to log per-epoch gate
mean/std/saturation for the enabled gate variants.

Run the dependency-free encoder checks with system Python and the full model
checks with the environment that contains PyTorch and Shapely:

```bash
python3 tests/test_mobile_bev.py --encoder
python3 tests/test_mobile_bev.py
```

Smoke-train A4 after preparing KITTI:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/mobilebev/a4_rich8_sgfpn_center3d.json \
  --detector-root detector \
  --output-root artifacts/kitti \
  --run-name mobilebev_a4_smoke_seed42 \
  --epochs 1 --max-train-batches 8 --max-val-batches 4 --num-workers 2
```

For a full run, omit the three smoke limits. Select from checkpoints saved
every five epochs using the pre-registered 3D metric:

Use `--seed 42`, `--seed 43` and `--seed 44` with each A1-A4 config for the
final twelve runs; the resolved seed is stored in each run config/checkpoint.

```bash
python3 tools/kitti_training_pipeline/select_checkpoint.py \
  --checkpoint-dir artifacts/kitti/mobilebev_a4_seed42/checkpoints \
  --config configs/kitti/mobilebev/a4_rich8_sgfpn_center3d.json \
  --detector-root detector \
  --kitti-root /path/to/KITTI/object \
  --split splits/kitti/val.txt \
  --output-dir artifacts/kitti/mobilebev_a4_seed42/selected_3d
```

The Center3D evaluator writes both `accuracy.bev` and `accuracy.3d`, distance
bands for Pedestrian/Cyclist, input/config/split hashes, and compressed
per-frame predictions. Legacy configs keep the original flat BEV result
schema.
