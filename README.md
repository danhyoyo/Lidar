# MobilePIXOR baseline and switchable C2PSA

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

## Controlled backbone experiment

Both configurations use Legacy35 input, the baseline loss, the same enabled
augmentation policy, the same MobilePIXOR stages, and the same FPN and heads.
The only experimental switch is `model.c5_attention`:

- `none`: frozen B0 baseline.
- `c2psa`: one C2PSA block after C5 and before the existing FPN lateral layer.

Train B0:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/b0_b1/kitti_mobilepixor_baseline.json \
  --detector-root detector \
  --output-root artifacts/kitti \
  --run-name b0_seed42 \
  --num-workers 2
```

Train the controlled C2PSA variant:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/b0_b1/kitti_mobilepixor_c2psa.json \
  --detector-root detector \
  --output-root artifacts/kitti \
  --run-name b1_c2psa_seed42 \
  --num-workers 2
```

Use the same precision, seed, batch size, epoch count, split, and evaluation
settings for both runs. Resume with `--resume /path/to/checkpoint.pt`. Training
keeps the minimum-validation-loss checkpoint in `<run>/selected/best.pt`.

## Google Colab

Open `3D_Lidar_Object_Detection_Notebook_standard.ipynb` and change only
`VARIANT` in the Configuration cell:

- `B0` selects the pure baseline configuration.
- `B1_C2PSA` selects the post-C5 C2PSA configuration.

The notebook automatically uses BF16 on supported GPUs and FP16 otherwise. It
records the branch, commit, config hash and effective training settings before
resuming a run. The older thesis-derived and MobileBEV configs remain in the
repository for provenance but are not selectable from this notebook.

## Verify

Run the thesis-clean backbone contract before training:

```bash
python3 tests/test_mobile_bev.py --clean-backbone
```

## Evaluate

```bash
python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
  --name b1_c2psa_seed42_validation \
  --backend pytorch \
  --model artifacts/kitti/b1_c2psa_seed42/selected/best.pt \
  --config configs/kitti/b0_b1/kitti_mobilepixor_c2psa.json \
  --detector-root detector \
  --kitti-root /path/to/KITTI/object \
  --split splits/kitti/val.txt \
  --output artifacts/kitti/b1_c2psa_seed42/evaluation_validation.json \
  --device cuda
```

This reports local loader-aligned KITTI-style rotated BEV AP R40, not a
submission to KITTI's hidden official test server. PyTorch evaluation supports
`--device cpu` for functional checks, but CPU and CUDA latency measurements
should not be compared directly.

## C2PSA attribution

The local module implements the published C2PSA structure without requiring the
`ultralytics` package. C2PSA was introduced in YOLO11 and retained in YOLO26.
For a paper, cite the applicable Ultralytics model paper/documentation and the
official module definition:

- https://docs.ultralytics.com/guides/yolo-architecture/
- https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py
- https://arxiv.org/abs/2606.03748
