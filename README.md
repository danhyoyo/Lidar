# MobilePIXOR with switchable C4/C5 attention and scale-gated FPN

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
The controlled B0/B1 comparison changes only `model.c5_attention`:

- `none`: frozen B0 baseline.
- `c2psa`: one C2PSA block after C5 and before the existing FPN lateral layer.

The clean backbone also accepts the independent Boolean switch
`model.scale_gated_fpn`. It is explicitly `false` in both committed B0/B1
configs, so those definitions remain unchanged. When `true`, the two ordinary
FPN sums become learnable, depthwise scale gates at C4 and C3. The gates are
zero-initialized, so the model starts exactly as sum-FPN and adds 480 trainable
parameters. For a controlled SG-FPN experiment, copy one config, change only
this field, and select it in Colab through `CONFIG_OVERRIDE`.

Two additional single-change presets keep C5 attention disabled and refine C4
before both block5 and the C4 lateral connection:

- `kitti_mobilepixor_c4_lsk.json`: `model.c4_attention: "lsk"`, a large selective
  kernel adapter (+19,846 parameters).
- `kitti_mobilepixor_c4_litemla.json`: `model.c4_attention: "litemla"`, a
  multi-scale linear attention adapter (+28,544 parameters).

Set `model.c4_attention` to `"none"` to disable either adapter. Missing this field
also preserves the previous B0/B1 behavior and checkpoint keys. C4 attention,
C5 attention, encoding, and SG-FPN are independent switches. See
[C4 ablation guide](docs/backbone_c4_ablation.md) for architecture, equations,
parameter counts, attribution, and controlled experiment design. These are
adaptations of published mechanisms, not claims of new attention mechanisms.

Every config in `configs/kitti/backbone_branch/` carries the same model schema,
including the `c2psa`, `lsk`, and `litemla` option blocks. Only `c4_attention` and
`c5_attention` activate them. In particular, `c5_attention: "none"` constructs a
parameter-free identity operation; it does not select a default attention module.
A regression test requires future variants to update all configs to this schema.

`SG-FPN` is this repository's shorthand for **scale-gated FPN**. Other papers
use the same acronym for different architectures, so a paper should define the
equation instead of implying that the acronym identifies a standard module.
The implementation was ported from this repository's deprecated MobileBEV
branch; do not present the port itself as a new contribution.

Train B0:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/backbone_branch/kitti_mobilepixor_baseline.json \
  --detector-root detector \
  --output-root artifacts/kitti \
  --run-name b0_seed42 \
  --num-workers 2
```

Train the controlled C2PSA variant:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/backbone_branch/kitti_mobilepixor_c2psa.json \
  --detector-root detector \
  --output-root artifacts/kitti \
  --run-name b1_c2psa_seed42 \
  --num-workers 2
```

Use the same precision, seed, batch size, epoch count, split, and evaluation
settings for both runs. Resume with `--resume /path/to/checkpoint.pt`. Training
keeps the minimum-validation-loss checkpoint in `<run>/selected/best.pt`. Each
epoch is also appended to `<run>/metrics.csv`, while TensorBoard event files are
written to `<run>/tensorboard`. View a run with:

```bash
tensorboard --logdir artifacts/kitti/b0_seed42/tensorboard
```

## Google Colab

Open `3D_Lidar_Object_Detection_Notebook_standard.ipynb` and change only
`VARIANT` in the Configuration cell:

- `B0` selects the pure baseline configuration.
- `B1_C2PSA` selects the post-C5 C2PSA configuration.
- `C4_LSK` selects the C4 large selective kernel adapter (notebook default).
- `C4_LITEMLA` selects the C4 multi-scale linear attention adapter.

Optional `BEV_ENCODING_OVERRIDE`, `SCALE_GATED_FPN_OVERRIDE`, and
`C5_ATTENTION_OVERRIDE` select encoding, fusion, and C5 attention independently.
`None` preserves the JSON value. The notebook writes a resolved config and uses
it consistently for architecture inspection, training, evaluation, and resume
checks. Automatic run names include all switches and a config hash.

The default branch is `C2PSA_c5block`. Push local changes to that branch (or select
another branch containing them) before using Colab. Start a new run when changing
architecture; do not reuse an incompatible run/checkpoint or bypass resume checks.

The notebook automatically uses BF16 on supported GPUs and FP16 otherwise. It
records the branch, commit, config hash and effective training settings before
resuming a run. The deprecated thesis-derived and MobileBEV configs remain in
the repository for provenance but are not selectable from this notebook.

## Verify

Run the backbone, attention, and notebook contract checks before training:

```bash
python3 tests/test_mobile_bev.py --clean-backbone
python3 tests/test_c4_attention.py
python3 tests/test_standard_training_notebook.py
```

These checks establish functional behavior, not academic novelty or accuracy.
For adapted C4 source licenses, see [attention notices](third_party/attention/NOTICE.md).
In particular, the adapted LSK material retains its upstream noncommercial restriction.

## Evaluate

```bash
python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
  --name b1_c2psa_seed42_validation \
  --backend pytorch \
  --model artifacts/kitti/b1_c2psa_seed42/selected/best.pt \
  --config configs/kitti/backbone_branch/kitti_mobilepixor_c2psa.json \
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
