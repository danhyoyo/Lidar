# KITTI training and evaluation pipeline

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

Run commands from the repository root. The current manifests contain 3,712
training and 3,769 validation frames. The earlier random 5,984/1,497 split is
archived under `splits/kitti/archived_5984_1497/`; its AP is not directly
comparable with the current split.

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
  --run-name mobilepixornext_corrected_seed42 \
  --num-workers 2 \
  --override-json '{"evaluation":{"kitti_root":"/path/to/KITTI/object"}}'
```

The master config currently uses MobilePixorNeXt, Rich8, LiteMLA, scale-gated
FPN, BN/SiLU heads and baseline loss. Its recipe is AdamW, LR 7e-4, weight
decay 1e-3, 50 epochs / 4 warmup epochs, batch 16, BF16 and seed 42.
Use a run's `config.resolved.json` to reproduce its actual architecture/recipe.
Resume a matching run with `--resume /path/to/checkpoint.pt`.

For matched rich8 vs learned point encoder experiments, use
`configs/experiments/encoders/{rich8,pillar32,pillar_rich}.json`. The added `pillar32`
learns a shared 10-to-32 point MLP and pools occupied pillars before the existing
BEV backbone. `pillar_rich` concatenates exact rich8 statistics with 24 learned
features into the same 32-channel BEV. Select `BEV_ENCODING="pillar_rich"` with
`PRESET="custom"` in existing notebooks, or `PRESET="ENCODER_PILLAR_RICH"` for
the matched comparator. See [learned encoder comparison](../../docs/learned_pillar_encoder.md)
for paired commands, notebook presets, parameter counts and memory conditions.
For rich-conditioned per-pillar channel gating, use
`configs/experiments/encoders/pillar_rich_gate.json` or
`PRESET="ENCODER_PILLAR_RICH_GATE"`. The encoder name is `pillar_rich_gate`;
`pillar_rich` remains max-only. The fixed Linear(32,4)-ReLU-Linear(4,24) gate
consumes learned max24 plus existing rich8, and multiplies max24 by 2*sigmoid.
It retains 32 BEV channels and adds 252 parameters. The final Linear is
zero-initialized for unit gates; constructor RNG is preserved for matched
backbone/head initialization. The old ECA encoder/presets are retired;
start a new run rather than resuming a max-only or ECA checkpoint into gate.
`tools/benchmarks/benchmark_pillar_encoders.py` compares rich8, max-only and
rich-conditioned gate at matched precision with separate preparation/transfer/model
timings. Its random-weight benchmark excludes decode/NMS and AP; measure
actual checkpoint end-to-end latency with the evaluator before claiming speed.
Omitted `--num-workers`, `--target-backend` and `--compile-model` flags use the
saved training config (historical fallbacks: 2 workers, Python backend, no compile).
Explicit flags override the config; `--no-compile-model` disables compilation.

The master config selects the highest validation AP and retains two independent
winners: `selected/best_ap.pt` and `selected/best_loss.pt`. Validation loss runs
every epoch. `train.checkpoint_selection.ap_every` controls AP inference on the
entire validation split; the default is 1 and the final epoch is always evaluated.
The fixed selection objective is R40 Moderate AP averaged equally across Car,
Pedestrian and Cyclist. `metric_mode="auto"` selects the local ROI BEV evaluator
for BEV models, or pinned reference AP3D for explicitly trained 3D models.

Set `evaluation.kitti_root` to the raw KITTI root containing `training/`, or to
`training/` itself. Local BEV requires original labels and calibration; reference
modes also require images. Periodic AP uses the existing FP32 checkpoint evaluator,
even when training uses BF16/FP16, matching final PyTorch evaluation. Its extra
inference model is released after each evaluation and training RNG is restored.

`selected/best.pt` and `selected/selection.json` remain compatibility aliases for
the primary winner. Separate `selection_ap.json` / `selection_loss.json` records
identify the independent winners. `checkpoints/last.pt` retains optimizer,
criterion, scheduler, scaler, RNG and AP selection state for resume.
`metrics.jsonl` and `train.log` include AP scores, inference time and winner flags.
The console and `train.log` show train/validation loss, LR and train/val time
before the additional AP pass. AP inference progress and a separate result line
show the three class Moderate R40 percentages, their mean and AP time.

An interval of 5 reduces inference cost but only selects the best of evaluated
epochs; an unsampled peak can be missed. Match the interval and decode settings
across comparators. Use 1 for A0/A2 if runtime permits. A changed selection policy
or AP decode protocol requires a new run name. Historical configs without an
explicit policy keep loss selection. `primary="loss"` disables periodic AP;
use this for batch-limited smoke/debug runs, which cannot establish best AP.

The notebook's evaluation cell evaluates both saved winners with the run's
original config. It writes `evaluation_ap_<mode>.json` and
`evaluation_loss_<mode>.json` and a comparison table labelled with selection,
epoch, checkpoint path and stored validation loss. The notebook uses direct
IPython `!` commands for setup, data preparation, smoke, train/resume and evaluation.
Benchmark protocol/evidence tools remain available separately under `tools/benchmarks/`.
Historical loss-only runs evaluate only their actual loss winner. Programmatic
callers can use `selected_run(run_dir, kind="ap")`,
`selected_run(run_dir, kind="loss")`, or `selected_runs(run_dir)`.

## Evaluate

```bash
python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
  --name mobilepixornext_corrected_validation \
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

## OpenPCDet-style augmentation

The opt-in `openpcdet` mode uses an ordered `AUG_CONFIG_LIST` and
`DISABLE_AUG_LIST`, following the global-transform and GT sampling design of
[OpenPCDet's augmentor](https://github.com/open-mmlab/OpenPCDet/tree/master/pcdet/datasets/augmentor).
It runs before BEV encoding, on training samples only, including empty scenes.
Each operation has its own `PROBABILITY`; the legacy outer `augmentation.p` is
used only by `one_of` / `compose` modes. Rotation angles are in radians.

Two JSON override profiles are provided:

| Profile | Operations |
| --- | --- |
| `configs/augmentation/openpcdet_global.json` | Flip along x (negate y), probability 0.5; rotation ±45°; scaling 0.95–1.05; XYZ translation std [0.2, 0.2, 0.1] m, probability 0.5 |
| `configs/augmentation/openpcdet_gt.json` | GT sampling first, then the same global operations |

The GT profile attempts to fill each scene up to Car:15, Pedestrian:10,
Cyclist:10, requiring at least five points per database object. These are trial
quotas, not promised insert counts: rotated BEV collisions, ROI filtering and
available objects can reduce them. Use `LIMIT_WHOLE_SCENE=false` to attempt the
configured number of additions regardless of existing class counts. Sampling
retains each database object's original location and yaw. It rejects collisions
against both original and previously accepted boxes, removes scene points inside
inserted 3D boxes, then adds centered object points and matching labels.

### Colab

In the standard notebook, choose:

```python
AUGMENTATION = "openpcdet_gt"
```

This works with `custom` and model presets. The data-preparation cell creates the
database using the run's train manifest, checks train/val disjointness, and prints
object counts. Select `openpcdet_global` for the global-only ablation, or
`standard` to retain the previous augmentation. Set a new run name for a fresh
augmentation experiment; keep the same model/loss/split and training schedule.

### CLI

Create the database once on the machine containing processed points and labels:

```bash
python3 tools/kitti_training_pipeline/build_gt_database.py \
  --processed-root data/kitti/processed \
  --train-split splits/kitti/train.txt \
  --val-split splits/kitti/val.txt
```

This writes object `.bin` crops and `gt_database/dbinfos_train.json` under the
processed root. Metadata records source frame IDs and the train-manifest hash.
Only train IDs are read; training rejects a database containing IDs outside its
current train manifest. Rebuild the database after changing the split or source
points/labels. The builder skips zero-point objects; the recipe applies its
stricter point-count filter when loading the database.

Train with the override, using a base config whose processed-data location
matches the database's processed root:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/config.json \
  --detector-root detector \
  --output-root artifacts/kitti \
  --run-name mobilepixornext_openpcdet_gt_seed42 \
  --target-backend numba --num-workers 6 \
  --override-json "$(cat configs/augmentation/openpcdet_gt.json)"
```

For the earlier 100-epoch experiment, use its `config.resolved.json` as the base
to retain the 100-epoch/warmup-8 recipe. Swap the override for
`openpcdet_global.json` to measure global augmentation separately. Automatic run
names distinguish `openpcdet_aug`, `openpcdet_gt_aug` and the old `standard_aug`.

The implementation is NumPy/Shapely and needs no OpenPCDet installation or CUDA
extensions. Dataset boxes use `[class,h,w,l,x,y,z_bottom,yaw]`; database boxes
use `[x,y,z_center,l,w,h,yaw]`. Point features such as intensity are preserved.
The database metadata is this repo's versioned JSON format; OpenPCDet's `.pkl`
metadata cannot be passed directly. Processed labels have no KITTI difficulty
field, so this builder does not invent difficulty values; `filter_by_difficulty`
only excludes records that actually supply a difficulty value.
Camera augmentations, local object transforms, shared-memory sampling, fake-LiDAR
conversion and road-plane placement are not implemented. `USE_ROAD_PLANE=true`
is rejected explicitly; the supplied profile uses false.

## Small-object correctness fixes: evaluate first, then retrain

New targets default to `data.regression_assignment=nearest_center`: reserve each
quantized center cell, then assign other cells in overlapping regression disks
to the nearest center. Python and Numba use the same kernel. Labels are sorted
canonically to break exact distance ties independently of input order. If multiple
boxes quantize to one center cell, the closest center to its grid origin owns it;
one shared regression vector cannot represent all colliding boxes.

Decoder peak extraction defaults to `data.kitti.peak_mode=per_class`. Each class
retains its own local maxima before class-wise rotated NMS; different classes at
the same cell become separate candidates using the shared regression vector.
This does not resolve geometrically distinct boxes sharing one regression cell.
Per-class extraction may retain more candidates, so measure both AP and latency.
Evaluation JSON records the effective peak mode and IoU scoring alpha.

To reproduce earlier target/decoder policies, set
`data.regression_assignment=legacy` and use `--peak-mode legacy` in evaluation.
Changing only `--nms-alpha 0` does not restore the old peak policy.

For the KITTI 100-epoch run discussed in the artifact analysis, run this on the
machine containing the raw dataset and processed point clouds. Adjust the three
paths to that machine; use the same config/checkpoint for both modes:

```bash
LIDAR_RUN_DIR=/path/to/mobilepixornext-standard_aug-oga_loss-rich10-iqa-sgfpn-reparam-s42
LIDAR_KITTI_ROOT=/path/to/KITTI_DATASET
LIDAR_COMPARE_DIR=artifacts/kitti_decoder_comparison

for peak_mode in legacy per_class; do
  python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
    --name "oga_epoch100_${peak_mode}" \
    --backend pytorch \
    --model "$LIDAR_RUN_DIR/checkpoints/last.pt" \
    --config "$LIDAR_RUN_DIR/config.resolved.json" \
    --detector-root detector \
    --kitti-root "$LIDAR_KITTI_ROOT" \
    --split splits/kitti/val.txt \
    --output "$LIDAR_COMPARE_DIR/${peak_mode}.json" \
    --peak-mode "$peak_mode" \
    --score-threshold 0.05 --nms-threshold 0.1 --nms-alpha 0.5 \
    --device cuda
done
```

Saved predictions NPZ already passed through suppression: it cannot recover
peaks removed by the old decoder. Re-run inference from the checkpoint.
For the baseline checkpoint with no IoU head, omit `--nms-alpha 0.5`.

Next, train a fresh corrected-target run using the same 100-epoch/warmup-8
resolved config. Do not resume from the old run when measuring the target fix:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config "$LIDAR_RUN_DIR/config.resolved.json" \
  --detector-root detector \
  --output-root artifacts/kitti_corrected \
  --run-name oga_corrected_targets_seed42 \
  --epochs 100 --physical-batch-size 16 --accumulation-steps 1 \
  --num-workers 6 --precision bf16 --target-backend numba --compile-model \
  --override-json '{"data":{"regression_assignment":"nearest_center","kitti":{"peak_mode":"per_class"}}}'
```

The first corrected run keeps the original backbone/head, loss, augmentation and
seed to isolate the correctness fixes. After this comparison, try GT database
sampling for Pedestrian/Cyclist, composed flip/rotation/scaling, then a 32- or
48-channel head in separate experiments. Keep the validation split fixed.

New master/notebook runs select `selected/best_ap.pt` during training and retain
`selected/best_loss.pt` separately. For older loss-selected BEV runs, the existing
offline selector can evaluate saved numbered checkpoints without retraining.
Store its output separately from the original selection:

```bash
python3 tools/kitti_training_pipeline/select_checkpoint.py \
  --checkpoint-dir artifacts/kitti_corrected/oga_corrected_targets_seed42/checkpoints \
  --config artifacts/kitti_corrected/oga_corrected_targets_seed42/config.resolved.json \
  --detector-root detector \
  --kitti-root "$LIDAR_KITTI_ROOT" \
  --split splits/kitti/val.txt \
  --output-dir artifacts/kitti_corrected/oga_corrected_targets_seed42/selected_by_ap \
  --device cuda
```

The selector uses the config's peak policy (default `per_class`). It evaluates
each numbered checkpoint on the split, so budget inference time accordingly.
Report Moderate mAP and each class's AP/recall, alongside end-to-end latency.

Use `export_onnx.py`, `build_tensorrt.py`, `compare_models.py` and
`deploy_engine.py` for deployment experiments. TensorRT engines are tied to
the local CUDA/TensorRT/GPU environment and should not be committed.
