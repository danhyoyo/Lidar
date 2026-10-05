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
  --config configs/config.json \
  --detector-root detector \
  --output-root artifacts/kitti \
  --run-name mobilepixornext_rich8_baseline_s42 \
  --num-workers 4
```

The master configuration (`configs/config.json`) centrally manages model architecture (`mobilepixornext`, SG-FPN, LiteMLA attention), BEV representation (`rich8`), loss strategies (`baseline`, `oga`, `q_oga`, `gw_qal`, `uwag`), and data augmentation.

Resume with `--resume /path/to/checkpoint.pt`.

## Evaluate

```bash
python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
  --name mobilepixornext_evaluation \
  --backend pytorch \
  --model /path/to/best.pt \
  --config configs/config.json \
  --detector-root detector \
  --kitti-root /path/to/KITTI/object \
  --split splits/kitti/val.txt \
  --output artifacts/kitti/evaluation_best_val.json \
  --device cuda
```

PyTorch evaluation also supports `--device cpu` for functional checks, although
CPU and CUDA latency numbers should not be compared directly.

This reports local loader-aligned KITTI-style rotated BEV AP R40, not a
submission to KITTI's hidden official test server.

Use `export_onnx.py`, `build_tensorrt.py`, `compare_models.py` and
`deploy_engine.py` for deployment experiments.

## Unified Configuration & Fast Notebook Setup

The repository utilizes **a single master configuration file** (`configs/config.json`).
Experiments, ablation studies, and benchmarks are customized directly through the standard Colab notebook (`3D_Lidar_Object_Detection_Notebook_standard.ipynb`) or CLI overrides:

- **Backbones**: `mobilepixornext`, `mobilepixor`, `mobilepixor_coordatt`
- **BEV Encodings**: `rich8` (8ch), `rich10` (10ch), `rich11` (11ch), `rich12` (12ch), `binary_slices` (35ch)
- **Necks**: `scale_gated_fpn` (SG-FPN), `neck_type` (`sgfpn`, `rc_sgfpn`, `rc_bisgfpn`)
- **Attention**: `c4_attention: "litemla"` with scales `[5]` or `[3, 5]` and QK normalization (`none`, `rmsnorm`)
- **Structural Reparameterization**: `use_reparam: true` (fused into 7x7 depthwise at deployment via `--deploy`)
- **Decoupled Quality Head**: `header_use_iou: true` (IoU-aware quality header with Joint NMS)
- **Loss Strategies**: `baseline`, `oga`, `q_oga`, `gw_qal`, `uwag`

To run custom configurations via CLI, pass `--override-json`:
```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/config.json \
  --override-json '{"loss": {"name": "q_oga"}, "model": {"use_reparam": true}}' \
  --detector-root detector \
  --output-root artifacts/kitti
```
