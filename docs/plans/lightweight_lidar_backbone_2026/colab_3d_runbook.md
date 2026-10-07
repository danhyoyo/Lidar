# Colab: explicit final 3D results

The [standard notebook workflow](notebook_runbook.md) includes 3D preparation, smoke, full training/resume, own-config selected-checkpoint evaluation and comparison records. Asset audits and GPU/reference benchmark gates run separately. Its only editable source JSON is configs/config.json. There is no notebook experiment-file override.

Current development defaults remain BEV. Final 3D requires trained owned z_bottom/log-height branches: changing the evaluator metric alone cannot give a BEV checkpoint those predictions.

## Select 3D using notebook controls

Keep the default custom focal/grouped OGA-IQA architecture and hybrid recipe. Change:

    BOX_MODE = "3d"
    VERTICAL_LOSS_WEIGHT = 1.0
    EVALUATION_MODES = ["3d", "bev"]
    EXPERIMENT_TAG = "final3d"

For an ECA experiment, change LOCAL_ATTENTION="eca" and use a distinct run tag. These settings require no alternate JSON. The main 3D backbone+neck stays660,528; detector883,941, excluding12 train-only OGA parameters.

After resolving configuration, inspect data.box_mode, loss.vertical_loss_weight, backbone_out_dim32, focal/local attention, head groups, actual split paths, precision and batch/schedule. All effective settings are saved automatically. Use new complete 3D training, or explicitly set WARM_START_PATH for compatible backbone-only initialization with fresh heads/state. Strict BEV-to-3D resume is rejected.

## Data and runtime gates

Reference inference requires original label_2, calib and image_2 inputs. The notebook conditionally extracts image_2.tar and builds missing training-only GT databases. Run the asset audit separately for formal comparisons; explicitly rebuild the database after changing splits or database settings.

The notebook smoke trains a few actual-data batches with the selected precision and worker count. Run the separate synthetic GPU smoke for FP32/FP16/BF16, workers0/configured workers and full-resolution inference when collecting benchmark evidence. Synthetic smoke does not measure full-resolution backward throughput or KITTI AP.

For a formal benchmark, run the sixteen wrapper/direct pinned-CUDA parity scenarios separately. They cover BEV/3D overlap, vertical mismatch, duplicate predictions, ignored neighbors/DontCare, difficulty boundaries and separate R11/R40. The simplified notebook does not run this suite automatically.

Colab must supply its own compatible compiler/libdevice/runtime. REFERENCE_CUDA_HOME and REFERENCE_PIP_PACKAGES allow explicit setup before model/Numba imports. See [reference evaluation](reference_evaluation.md) for tested local versions and failure boundaries. The laptop temporary environment is not a Colab dependency. No local-BEV/CPU fallback can replace reference metrics.

## Train, select and evaluate

Run the normal actual-data smoke and full-training cells. The default trainer selects maximum three-class Moderate R40 AP and retains minimum-validation-loss independently as selected/best_ap.pt and selected/best_loss.pt. It also preserves selected/best.pt, selected/selection.json, retained checkpoints, metrics.jsonl and config.resolved.json. Independent torch.save archives may differ in file bytes; complete selected/retained payload equality is checked.

The evaluation cell automatically routes both modes to the pinned reference evaluator, using this run's own resolved config/checkpoint/split. It displays all nine class/difficulty values plus Moderate/AP9 and saves comparison JSON/CSV with R11/R40 kept separate. It never silently chooses last.pt or evaluates a 3D checkpoint using the local BEV cell.

For a controlled comparison, train a matching no-context 3D baseline first, then record separate benchmark evidence using the task51 CLI guide. Match controls outside focal context, including AP selection policy and interval, and disclose any additional differences. Protocol readiness is recorded by the benchmark tools, rather than a notebook training gate.

## Optional standalone CLI reproduction

Materialized recipes remain available for CLI reproduction, but the notebook does not need them. To reproduce a completed run's reference results:

    python tools/kitti_training_pipeline/evaluate_kitti_3d.py --name final_candidate --checkpoint <run>/selected/best.pt --config <run>/config.resolved.json --detector-root detector --kitti-root /content/KITTI_DATASET/training --split <actual_validation_manifest> --metric-mode 3d --device cuda --output <run>/ap3d_reference.json

Use --metric-mode bev for reference APBEV. Both report R11/R40 on full benchmark GT; local ROI BEV R40 is separate. The optional external-run notebook cell also uses its own completed-run config/selection/history and requires the matching original raw inputs.

Actual Colab data, long training, measured AP and final task51 real benchmark freeze remain external. Local notebook verification uses small synthetic KITTI-format assets. See the [task51 CLI guide](task51_colab_runbook.md) for separate evidence commands and trust limits.
