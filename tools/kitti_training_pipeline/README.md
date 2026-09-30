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

## Physics-Consistent GT Augmentation (PCU-Aug / GTSampler)

The online copy-paste augmentation framework (`GTSampler`) inserts 3D bounding boxes from a pre-extracted GT database (`kitti_gt_database.pkl`) into training scenes with physical heuristics:

- **Core Copy-Paste Semantics**:
  - Box collision checks (2D oriented bounding box intersection) and clearing interior scene points within the newly placed box volume **always apply** when an object is placed, even when physics flags are disabled (`enable_physics=False`).
  - `sample_counts`: Specifies the maximum number of objects to insert per class. Specifying `{}` or count `0` means no objects will be sampled for that class; it does not represent target scene totals.
  - Active only during training (`task='train'`); validation and testing sets do not instantiate the sampler or load database artifacts.

- **Physics Flags and Precedence**:
  - `enable_physics`: Legacy preset flag.
  - Explicit keyword flags override `enable_physics`:
    - `enable_ground_validation`: Rejects placement if the local LiDAR point neighborhood does not support the object bottom (fallback ground snapping only occurs if ground validation is explicitly disabled).
    - `enable_static_collision`: Checks if the proposed bounding box collides with existing foreground scene obstacles.
    - `enable_line_of_sight`: Rejects candidate placement if foreground points occlude the object along sensor rays.
    - `enable_shadow_masking`: Employs exact ray-oriented box slab intersection to mask background points falling in the ray shadow behind the inserted object.
    - `enable_density_subsample`: Subsamples points inversely proportional to $(r_{\text{origin}} / r_{\text{target}})^2$ with a 5-point floor to maintain object identification when points $\ge 5$.
    - `enable_radiometric_calibration`: Attenuates intensity based on radar/lidar range equations.
  - **Visibility Protection**:
    - Protects existing and earlier inserted objects: candidate placements that occlude existing objects below `min_visible_points` (default: 5) or `min_visible_ratio` (default: 0.5) are rejected.
  - **Candidate Pose Proposal**:
    - Tries 14 corridor attempts (stratified along forward range $X$ near/mid/far with 30%/40%/30% distribution) followed by 6 full-rectangle attempts.
  - **Heuristic Disclaimer**:
    - Ray-box intersections use oriented bounding box geometry as an approximation of object volume; they do not simulate detailed vehicle mesh surfaces or LiDAR multi-beam scan lines. Previous claims (such as `<2ms/frame` overhead or mAP gains) are heuristic design goals that must be validated through empirical benchmarking with audit tools (`tools/visualization/audit_gt_sampler.py`).
