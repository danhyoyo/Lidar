# LiDAR BEV object detection

This directory contains the preparation, training, evaluation, ONNX export and
TensorRT utilities for MobilePIXOR and MobilePixorNeXt in `detector/core`.

## Environment

The reproduced run used Python 3.10, PyTorch 2.11.0+cu128, CUDA 12.8 and an
NVIDIA RTX 5060 Ti. Install a CUDA-compatible PyTorch build first, then run:

```bash
python3 -m pip install -r requirements-kitti.txt
```

The ONNX package is included for export validation. TensorRT is optional and is
only required for engine export/evaluation.

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
  --config configs/kitti/baselines/kitti_uwag_coordatt_aug.json \
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
  --config configs/kitti/baselines/kitti_uwag_coordatt_aug.json \
  --detector-root detector \
  --kitti-root /path/to/KITTI/object \
  --split splits/kitti/val.txt \
  --output artifacts/kitti/evaluation_best_val.json \
  --device cuda
```

PyTorch evaluation also supports `--device cpu` for functional checks, although
CPU and CUDA latency numbers should not be compared directly.

This reports local loader-aligned KITTI-style rotated BEV AP R40, not a
submission to KITTI's hidden official test server. The reproduced report is in
`results/kitti/uwag_coordatt_aug_bf16_seed42/`.

Use `export_onnx.py`, `build_tensorrt.py`, `compare_models.py` and
`deploy_engine.py` for deployment experiments. TensorRT engines are tied to
the local CUDA/TensorRT/GPU environment and should not be committed.

## MobileBEV-Lite

The frozen design and experiment protocol are in
`docs/mobile_bev_lightweight/SPEC.md` and `docs/mobile_bev_lightweight/PLAN.md`.
The four controlled BEV variants are under `configs/kitti/mobilebev/`; A1 is
the 35-channel baseline and A4 enables RichBEV-8 plus SG-FPN.

Run the dependency-free encoder checks with system Python and the full model
checks with the environment that contains PyTorch and Shapely:

```bash
python3 tests/test_mobile_bev.py --encoder
python3 tests/test_mobile_bev.py
```

Smoke-train A4 after preparing KITTI:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/mobilebev/a4_rich8_sgfpn_bev.json \
  --detector-root detector \
  --output-root artifacts/kitti \
  --run-name mobilebev_a4_smoke_seed42 \
  --epochs 1 --max-train-batches 8 --max-val-batches 4 --num-workers 2
```

For a full run, omit the three smoke limits. Training keeps the checkpoint
with the minimum validation loss in `<run>/selected/best.pt`.

Use `--seed 42`, `--seed 43` and `--seed 44` with each A1-A4 config for the
final twelve runs; the resolved seed is stored in each run config/checkpoint.

The evaluator writes BEV AP R40, distance bands for Pedestrian/Cyclist,
input/config/split hashes, and compressed per-frame predictions.

## MobilePixorNeXt

`mobilepixornext` is the registry name for the lightweight backbone with 7×7
depthwise blocks, LiteMLA refinement, and a scale-gated FPN. It replaces the
former backbone name in model configs and code. The architecture is described
in [`docs/mobilepixornext_architecture.md`](docs/mobilepixornext_architecture.md).

The RichBEV-8 configuration with OGA loss (Baseline M0) is
[`configs/kitti/oga_loss/kitti_mobilepixornext_litemla_oga.json`](configs/kitti/oga_loss/kitti_mobilepixornext_litemla_oga.json).
The MobilePixorNeXt configuration with baseline loss is
[`configs/kitti/baseline_loss/kitti_mobilepixornext_litemla_baseline.json`](configs/kitti/baseline_loss/kitti_mobilepixornext_litemla_baseline.json).

Pillar improvement configurations:
- **Pillar 1 (M1 - Structural Reparameterization):** [`configs/kitti/reparameterization/kitti_mobilepixornext_litemla_oga_reparam.json`](configs/kitti/reparameterization/kitti_mobilepixornext_litemla_oga_reparam.json)
- **Pillar 2 (M2 - Multi-Scale LiteMLA + QK-RMSNorm):** [`configs/kitti/multiscale_attention/kitti_mobilepixornext_ms_litemla_oga.json`](configs/kitti/multiscale_attention/kitti_mobilepixornext_ms_litemla_oga.json)
- **Pillar 4 (M4 - IoU-Aware Quality Header & Joint NMS):** [`configs/kitti/iou_aware_header/kitti_mobilepixornext_litemla_oga_reparam_iqa.json`](configs/kitti/iou_aware_header/kitti_mobilepixornext_litemla_oga_reparam_iqa.json)
- **Pillar 5 (M5 - Physics-Consistent 3D Augmentation / PCU-Aug):** [`configs/kitti/physics_augmentation/kitti_mobilepixornext_litemla_oga_pcu.json`](configs/kitti/physics_augmentation/kitti_mobilepixornext_litemla_oga_pcu.json)
- **Cumulative (M1 + M2 + M4):** [`configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_oga.json`](configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_oga.json)
- **Cumulative SOTA (M1 + M2 + M4 + M5):** [`configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_m5_oga.json`](configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_m5_oga.json)

Use any config with the training command above by replacing its `--config`
argument.
