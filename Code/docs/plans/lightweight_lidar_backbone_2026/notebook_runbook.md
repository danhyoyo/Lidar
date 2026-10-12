# Simple Colab training workflow

The [standard notebook](../../../3D_Lidar_Object_Detection_Notebook_standard.ipynb)
uses the original setup → config → KITTI → smoke → train/resume → evaluate flow.
All current architecture/loss/augmentation/runtime controls remain available.
Commands run directly with IPython `!`; Python scripts use `-u` for live output.
Every shell command checks its exit status before continuing.

## Setup and configuration

Select `research/mobilepixornext-under1m`. Use `SOURCE_MODE="git"` for published
commits, or upload the complete checkout to `REPO_DIR` and use
`SOURCE_MODE="existing"` for local changes. Set `MOUNT_DRIVE` and the data/artifact
paths in the first cell. The setup configures detector imports in the notebook
kernel after installing dependencies; model construction runs in CLI processes.

The configuration cell defaults to:

```python
PRESET = "custom"
BEV_ENCODING = "hist14"
BACKBONE_OUT_DIM = 32
STAGE_DEPTHS = [3, 4, 2]
HEAD_MODE = "grouped"
C4_ATTENTION = "none"
C4_CONTEXT = "focal"
LOCAL_ATTENTION = "none"
LOSS_NAME = "oga"
HEADER_USE_IOU = True
AUGMENTATION = "hybrid_gt"
BOX_MODE = "bev"
EVALUATION_MODES = ["local_bev"]
CHECKPOINT_SELECTION = "ap"
AP_EVERY = 1
AP_METRIC_MODE = "auto"
```

`PRESET="custom"` applies the individual controls. Built-in UNDER1M and historical
presets provide their own architecture; runtime/augmentation/metric controls
still apply. `PRESET="config"` preserves the master's architecture.
`NOTEBOOK_OVERRIDES` supplies additional nested custom settings. Unsupported
head/loss/quality combinations fail during configuration.

Only `configs/config.json` is maintained as a source JSON. The notebook writes
`RUN_DIR/config.json`; the trainer writes `config.resolved.json`. Once training
exists, incompatible changes require a new `EXPERIMENT_TAG` or `CUSTOM_RUN_NAME`.

Runtime defaults remain BF16, batch16, validation32, workers6, seed42, 50 epochs
and four warmup epochs. Use `VAL_BATCH_SIZE` for a smaller validation batch.
Choose FP16/FP32 when the GPU lacks BF16. Compilation is optional.
Omitted trainer CLI flags inherit `num_workers`, `target_backend` and
`compile_model` from the saved config. Explicit flags override them, including
`--no-compile-model` to disable compilation.

## Run the notebook

1. Setup: mount Drive, select the checkout and install dependencies.
2. Configure: edit controls and inspect the concise resolved summary.
3. Prepare KITTI: reuse complete folders or extract `velodyne.tar`, `label_2.tar`
   and `calib.tar`. Reference modes also extract `image_2.tar`. Missing GT
   databases are built from configured training IDs only; set
   `REBUILD_GT_DATABASE=True` after changing splits/database settings.
4. Optional checks: `RUN_PROPOSAL_TESTS=True` runs focused tests and the parameter
   profiler. The default is false.
5. Smoke: train/validate a few actual-data batches in a temporary directory.
   The generated smoke config uses one epoch, zero warmup and loss selection,
   so it skips full-split AP. Failed commands or non-finite results stop the cell.
6. Train/resume: rerun after interruption to resume `checkpoints/last.pt`, or the
   latest numbered epoch checkpoint. Model, criterion, optimizer, scheduler,
   scaler and RNG restoration remain strict. `WARM_START_PATH` initializes a
   compatible backbone for a fresh run with new heads and training state.
7. Evaluate: evaluate both saved winners using the run's own resolved config,
   split and thresholds. Save `evaluation_ap_<mode>.json`,
   `evaluation_loss_<mode>.json`, `comparison.csv` and `comparison.json`.
   The table contains each winner's epoch/path/loss and all class/difficulty APs.
8. Optional cells evaluate another completed run or preview GT sampling.

The notebook no longer requires benchmark evidence/protocol/GPU matrix stages
before training. Formal comparison tools remain under `tools/benchmarks/`; use
[task51 runbook](task51_colab_runbook.md) and [benchmark protocol](benchmark_protocol.md)
when preparing reproducible benchmark evidence.

## Training output and checkpoints

The trainer first displays the training/validation phase, then the original
`Epoch | Train Loss | Val Loss | LR | Time: train=..., val=...` summary.
This line appears **before** periodic AP inference, including when that AP pass
is slow or fails. `[BEST LOSS]` marks a newly lowest validation loss.

The extra AP stage prints inference progress and then a separate
`Validation AP` line with Car/Pedestrian/Cyclist Moderate R40 percentages, their
mean, AP time and `[BEST AP]` when applicable. AP time is separate from train/val
time. `train.log` contains readable summaries; `metrics.jsonl` keeps detailed
metrics and completed epoch history.

`selected/best_ap.pt` maximizes the equally weighted three-class Moderate R40
mean; `selected/best_loss.pt` minimizes validation loss. Ties keep the first
winner. `best.pt` remains the primary winner's compatibility alias.
`checkpoints/last.pt` is the resume checkpoint. AP schedule/state survives resume,
and AP evaluation restores the training RNG.

`AP_EVERY=1` evaluates every epoch; larger intervals trade inference cost for
fewer sampled candidates. The final epoch is always evaluated. Use
`CHECKPOINT_SELECTION="loss"` for loss-only/debug runs. Existing loss-only runs
evaluate their genuine loss winner; missing AP files in an AP-selected run are
errors. Evaluating both winners requires two inference passes per metric.

## Final 3D stage

Use a fresh run with `BOX_MODE="3d"`, `VERTICAL_LOSS_WEIGHT=1.0`,
`EVALUATION_MODES=["3d", "bev"]` and a distinct tag. The z/height branches must be
trained; changing a BEV checkpoint's metric cannot add them. Local BEV reports
ROI R40, while reference 3D/BEV report R11 and R40 separately.

Reference evaluation requires real calibration/images and compatible Numba CUDA
compiler/runtime dependencies. Set `REFERENCE_CUDA_HOME` or
`REFERENCE_PIP_PACKAGES` when needed; see [reference guide](reference_evaluation.md).
The simple notebook does not automatically run the sixteen reference parity
tests. Run those separately for a formal benchmark. Synthetic tests/smoke runs
verify the pipeline, not KITTI accuracy or measured deployment performance.
