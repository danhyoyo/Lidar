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

The master configuration (`configs/config.json`) centrally manages model architecture (`mobilepixornext`, SG-FPN, optional Focal Context), BEV representation (`rich8`), loss strategies (`baseline`, `oga`, `q_oga`, `uwag`), and data augmentation.

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
- **BEV Encodings**: `rich8` (8ch), `hist14` (14ch), `binary_slices` (35ch), and pillar encoders (32ch)
- **Necks**: `scale_gated_fpn` (SG-FPN), `neck_type` (`sgfpn`, `rc_sgfpn`, `rc_bisgfpn`)
- **Context and Attention**: optional `c4_context: "focal"` and `local_attention: "eca"` or `"simam"`; C4 uses convolution by default
- **Structural Reparameterization**: `use_reparam: true` (fused into 7x7 depthwise at deployment via `--deploy`)
- **Decoupled Quality Head**: `header_use_iou: true` (IoU-aware quality header with Joint NMS)
- **Loss Strategies**: `baseline`, `oga`, `q_oga`, `uwag`

Retired C4 attention settings are accepted only as `none`, empty scales, and
`none` QK normalization. Old checkpoints containing C4 attention weights require
the earlier code revision. Reports under `docs/plans/` preserve historical
experiment results; they are not the current list of supported options.

The notebook resolves a fresh configuration every time its configuration cell runs.
You can switch experiments by editing notebook options without changing
`configs/config.json`:

```python
# Model options in the configuration cell
PRESET = "custom"                 # Or a named preset; "config" keeps the base model/loss/BEV
BEV_ENCODING = "rich8"
BACKBONE_OUT_DIM = 32              # Neck output / head input channels; MobilePixorNeXt only
LOSS_NAME = "oga"
HEADER_USE_IOU = True
AUGMENTATION = "openpcdet_gt"
CUSTOM_RUN_NAME = None             # Automatic name is recalculated on every execution
EXPERIMENT_TAG = "head32_gt"       # Separate runs with other settings sharing the same name
CONFIG_OVERRIDE = None
```

`AUGMENTATION` is independent of the model preset and supports:

| Selection | Effective augmentation |
| --- | --- |
| `standard` | Legacy OneOf, probability 0.5 |
| `compose` | Legacy rotation, scaling and translation in sequence, probability 0.5 |
| `none` | Legacy augmentation disabled, probability 0 |
| `openpcdet_global` | Ordered world flip, rotation, scaling and translation |
| `openpcdet_gt` | Train-only GT database sampling followed by world transforms |
| `hybrid_gt` | Cached/JIT GT sampling with source-relative placement and optional scene physics |
| `config` | Keep the augmentation recipe from `CONFIG_BASE` |

Selecting a recipe replaces the previous augmentation dictionary. Named presets
set their model/loss/BEV options explicitly, including neck, quality head and BEV
channels. Custom model options apply only when `PRESET = "custom"`.
Set both `PRESET = "config"` and `AUGMENTATION = "config"` to keep those sections
from the base file.

Precedence is **base configuration → model preset/custom → augmentation selection
→ `CONFIG_OVERRIDE` → notebook runtime settings**. The last stage sets seed,
dataset location, epochs, warmup, learning rate, batch sizes, precision, workers
and acceleration from the notebook. Edit `WARMUP_EPOCHS` and `VAL_BATCH_SIZE`
alongside `EPOCHS` in the setup cell. For example, use 100 epochs and 8 warmup
epochs; `VAL_BATCH_SIZE = None` defaults to twice the physical training batch.

The summary prints effective settings and the active OpenPCDet queue. Dataset
preparation automatically builds the GT database from the configured train
split when needed. Each run writes its resolved `config.json` into its artifact
directory. If that directory already contains training metadata or checkpoints,
an incompatible config is rejected before the old config is overwritten; choose
a different `EXPERIMENT_TAG` or `CUSTOM_RUN_NAME`. Resume uses the same options
and run name. Output stride remains 4 for the supported backbones; changing
`out_size_factor` alone cannot change neck resolution.

After updating the repository, reopen the updated notebook in Colab: checking
out new source does not replace code already displayed in an old notebook cell.

The `feature/hybrid-gt-augmentation` branch includes `AUGMENTATION = "hybrid_gt"`
and a `HYBRID_OPTIONS` dictionary in the notebook configuration cell. It reuses
the existing train-only GT database, adds bounded per-worker caching and batched
NumPy/Numba geometry, and supports PCU-inspired placement and visibility checks.
The optional preview cell shows the original scene, sampled scene and final world
transforms, with inserted points colored accurately even after shadow removal.
See [hybrid augmentation usage and limits](docs/hybrid_gt_augmentation.md) and
[the synthetic CPU benchmark](docs/plans/2026-10-06-hybrid-gt-benchmark.md).

To run custom configurations via CLI, pass `--override-json`:
```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/config.json \
  --override-json '{"loss": {"name": "q_oga"}, "model": {"use_reparam": true}}' \
  --detector-root detector \
  --output-root artifacts/kitti
```
