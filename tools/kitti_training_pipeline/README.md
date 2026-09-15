# Reproduce UWAG + CoordAtt + geometric augmentation

This directory contains the preparation, training, evaluation, ONNX export and
TensorRT utilities for the MobilePIXOR detector in `detector/core`.

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

PyTorch evaluation also supports `--device cpu` for functional checks, although
CPU and CUDA latency numbers should not be compared directly.

This reports local loader-aligned KITTI-style rotated BEV AP R40, not a
submission to KITTI's hidden official test server. The reproduced report is in
`results/kitti/uwag_coordatt_aug_bf16_seed42/`.

Use `export_onnx.py`, `build_tensorrt.py`, `compare_models.py` and
`deploy_engine.py` for deployment experiments. TensorRT engines are tied to
the local CUDA/TensorRT/GPU environment and should not be committed.
