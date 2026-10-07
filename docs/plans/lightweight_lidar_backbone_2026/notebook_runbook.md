# One-master-config Colab workflow

The standard [notebook](../../../3D_Lidar_Object_Detection_Notebook_standard.ipynb) now owns the complete workflow. Its only editable source JSON is configs/config.json. Change notebook controls; do not edit or select experiment/augmentation JSON files. Built-in recipes live in Python, and run snapshots are generated automatically.

## Source availability

The notebook selects research/mobilepixornext-under1m. A Git fetch can obtain only published commits, not uncommitted local changes. Use SOURCE_MODE="git" after the complete implementation is published, or upload the complete repository to REPO_DIR and select SOURCE_MODE="existing". The source preflight stops on missing current helpers/tools/tests instead of silently running an older branch. Existing source edits are preserved.

Mount Drive if needed and set KITTI_TAR_ROOT, RAW_KITTI_ROOT, PROCESSED_DATASET_DIR and ARTIFACT_ROOT in the first cell. Default archives contain the standard training/ layout. BEV preparation needs velodyne.tar, label_2.tar and calib.tar; reference evaluation additionally needs image_2.tar. A complete extracted dataset is reused.

The setup and parameter-audit cells both call `configure_detector_imports(REPO_DIR / "detector")`. This exposes the `core` package and its legacy `utils_1` imports in the notebook kernel; running pytest in a subprocess does not configure the kernel's import path. For an already-open older notebook, execute these lines before the parameter-audit cell:

```python
from common import configure_detector_imports
configure_detector_imports(REPO_DIR / "detector")
```

The setup cell's `run_command` reads a merged stdout/stderr pipe and prints each line through the notebook kernel with flushing. It sets `PYTHONUNBUFFERED=1` only in the child environment and displays the command and PID before waiting for output. Failures retain the last 100 output lines in `CalledProcessError.output`, and signal exits are identified explicitly. On notebook interruption, the direct subprocess is terminated and waited for. Existing trainer `train.log` and `metrics.jsonl` files remain unchanged.

For a notebook already open, replace its `run_command` definition with the current setup-cell definition in a separate cell before launching another command. Updating the repository alone does not replace a Python function already defined in the kernel. Trainer summaries remain per epoch; streaming does not introduce batch progress or resolve a resource-related SIGKILL.

## Development defaults

The configuration cell defaults to:

    PRESET = "custom"
    BOX_MODE = "bev"
    EVALUATION_MODES = ["local_bev"]
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

All these controls apply immediately with PRESET="custom". Optional UNDER1M/historical presets use built-in architecture recipes; augmentation, box/metric mode, paths and runtime controls still apply. PRESET="config" preserves the master's model. NOTEBOOK_OVERRIDES supplies extra nested custom keys without a file override. Architecture-specific controls are validated; unsupported loss/head/quality combinations fail before training.

PRECISION remains explicit. Choose BF16 only on supporting hardware; FP16/FP32 are alternatives for development. The default batch is16, with validation32 unless VAL_BATCH_SIZE is set. Adjust these controls for actual Colab memory. COMPILE_MODEL=False keeps compilation optional. No Jetson/TensorRT optimization is introduced.

The effective config is saved as RUN_DIR/config.json; the trainer writes config.resolved.json. These are immutable run receipts once training exists, not additional user-maintained configurations. Changing a trained run's incompatible settings requires a new EXPERIMENT_TAG or CUSTOM_RUN_NAME. Paths, split hashes, seed, model/objective identity, precision and actual counts are recorded.

## Run top to bottom

1. Resolve the source and configuration.
2. Reuse/prepare KITTI; build missing configured GT databases from training IDs only. Existing stale databases fail the audit unless explicitly rebuilt with REBUILD_GT_DATABASE=True.
3. Run read-only asset audits, including finite point/crop contents, unique/disjoint splits, processed labels, original evaluation inputs and database source-label geometry.
4. Run the focused proposal tests and instantiated backbone/head/criterion parameter audit. Main backbone+neck is660,528; main BEV detector846,689, excluding12 train-only OGA parameters.
5. Run actual synthetic GPU gates for workers0/configured workers and full configured inference shape. Development defaults to the selected precision; benchmark policy requires FP32/FP16/BF16. This does not benchmark full-resolution backward or KITTI AP.
6. Record baseline evidence and candidate protocols. Development may continue with explicitly pending comparator/matrix gates.
7. Run a short actual-data training/validation smoke in a temporary directory, with a generated one-epoch/warmup0 snapshot.
8. Train or strictly resume the full configured schedule. checkpoints/last.pt restores model, adaptive loss, optimizer, scheduler, scaler and RNG. Optional WARM_START_PATH loads compatible backbone tensors into a new run with fresh heads/state.
9. Evaluate the complete selected run using its own config.resolved.json and configured winner: selected/best_ap.pt by default. Historical runs retain their recorded loss policy. Missing/incomplete selection fails; there is no last.pt fallback.
10. Save per-mode evaluations/predictions, full input/checkpoint provenance, candidate evidence and comparison.csv/comparison.json. Display all nine class/difficulty cells and actual Moderate/AP9 values. Missing results cannot become a fabricated zero.

## Controlled focal baseline and benchmark mode

First train the internal baseline with the same controls and C4_CONTEXT="none" in development mode. Keep an independent run name. Then restore C4_CONTEXT="focal" and set:

    RUN_PURPOSE = "benchmark"
    BASELINE_RUN = Path("/content/drive/MyDrive/lidar_training_artifacts/<completed_no_context_run>")
    COMPARISON_KIND = "focal_ablation"

The default focal ablation checks normalized model/head/loss/encoding identity outside context, seed, schedule/optimizer/precision/batches, augmentation and encoder/target/runtime backends. Its purpose is to isolate focal; changing additional factors fails this check. For a deliberately broader comparison, select COMPARISON_KIND="protocol" and disclose the recorded model/training differences. Evaluation protocol matching is still mandatory; the generic option does not certify a controlled focal ablation.

Baseline evaluation uses the baseline's own resolved config/selected checkpoint. Complete ordered training history, actual optimizer updates, configured loss/AP selection, current inputs/config/checkpoint hashes, all class/difficulty AP values and macros must agree. Independently serialized torch.save archives can differ in file hashes: the entire selected/retained payload must agree, including model, criterion, optimizer, scheduler/scaler/RNG and saved metadata. Each archive's actual file SHA remains recorded and freshness-checked. Comparators must match the checkpoint selection policy, AP mode, R40 sampling, Moderate three-class macro and AP interval.

## Independent AP and loss checkpoints

The configuration cell exposes:

```python
CHECKPOINT_SELECTION = "ap"
AP_EVERY = 1
AP_METRIC_MODE = "auto"
```

Validation loss is computed every epoch. AP_EVERY controls a separate full-split
FP32 inference/evaluation pass, with the final epoch always included. Auto mode
selects local_bev for BOX_MODE="bev", or reference 3d for BOX_MODE="3d". Explicit
reference bev selection requires BOX_MODE="3d". The raw input root is taken from
RAW_KITTI_ROOT and saved in evaluation.kitti_root; it is not the processed root.

Save selected/best_ap.pt by maximum R40 Moderate mean across all three classes,
selected/best_loss.pt by minimum total validation loss, and checkpoints/last.pt
for resume. Ties retain the earliest winner. selected/best.pt and selection.json
alias the configured primary winner; dedicated selection_ap.json and
selection_loss.json describe the independent winners. AP state and evaluation
schedule survive resume. The AP pass restores training RNG and does not change
the training model or adaptive criterion. Per-epoch AP values and timing are
recorded in metrics.jsonl and streamed to the notebook.

For A0/A2, start with AP_EVERY=1 so every completed epoch is eligible. AP_EVERY=5
is a valid runtime compromise but can miss an unsampled peak; use the same value
for both runs. The short actual-data smoke selects loss and skips full-split AP.
CHECKPOINT_SELECTION="loss" also disables AP for debug runs. Do not change a
running process or overwrite an old snapshot: use distinct new run names when
changing policy. Existing completed A0/A2 loss-selected runs remain readable;
the offline BEV checkpoint selector can evaluate their saved numbered epochs.

Benchmark training requires successful actual data audits, all three precision gates, workers0/configured workers and matching reproduced baseline evidence. Reference modes additionally require pinned CUDA parity on this runtime. Every selected protocol must have no pending gates and long_training_allowed=true. Evidence is rechecked just before full benchmark training/resume. Synthetic evidence is rejected as real baseline readiness.

This is an internal architecture baseline. External algorithm checkpoint/evaluation/selection formats need their own verified evidence adapters. Paper table numbers alone cannot freeze a reproduced comparison.

## Explicit final 3D stage

Keep the current BEV development default until final 3D experiments are selected. Then change controls, keeping the main architecture and hybrid augmentation:

    BOX_MODE = "3d"
    VERTICAL_LOSS_WEIGHT = 1.0
    EVALUATION_MODES = ["3d", "bev"]
    EXPERIMENT_TAG = "final3d"

Use a new full 3D training run, or an explicit compatible backbone warm-start with fresh vertical heads/objective/state. A BEV checkpoint cannot be strictly resumed as a complete 3D detector or acquire z/height by changing its evaluation flag. Train a matching 3D baseline before benchmark mode.

The notebook extracts/requires real images, audits P2/projection inputs and runs sixteen wrapper-versus-pinned-reference CUDA parity cases covering overlap, ignored neighbors/DontCare, difficulty boundaries, vertical mismatch and R11/R40. reference_runtime.json describes that selected parity suite; it is not a full repository-suite receipt. It checks the actual compiler/libdevice/runtime, not just CUDA availability.

REFERENCE_CUDA_HOME and REFERENCE_PIP_PACKAGES are explicit runtime controls when additional compatible compiler dependencies are needed. Package installation occurs before model/Numba imports. Dependency/kernel failure stops reference work without a local-BEV fallback. Local temporary toolkit paths are not Colab dependencies; see [reference evaluation](reference_evaluation.md) for the tested environment and limits.

The evaluator routes each mode automatically: local_bev is local ROI APBEV R40; 3d/bev use full-domain pinned reference AP3D/APBEV with R11 and R40 separately. The main 3D detector has883,941 parameters; backbone remains660,528. Thresholds/cap, peak mode and IQA alpha come from notebook controls and are shared with protocol recording.

## Other completed runs and limits

The optional external-run cell takes a repository run directory and its own saved modes/config/selection/history. Original raw inputs required by those modes must be available; it never borrows the current run's architecture or silently chooses another checkpoint. The optional hybrid preview uses the configured training manifest and CLI-supported frame index.

Supplied training/evaluation JSON is trusted execution evidence with current-file and consistency checks, not authenticated remote execution or genuine-KITTI/SOTA certification. Local verification uses small synthetic KITTI-format assets. Actual Colab datasets, long training, measured candidate AP and final real benchmark freeze remain external work.
