# Lightweight KITTI BEV and 3D recipes

These are complete configs resolved from `configs/config.json`, the existing hybrid GT augmentation recipe and the [experiment manifest](../../../docs/plans/lightweight_lidar_backbone_2026/experiment_manifest.json). They can be passed directly to the PyTorch training/evaluation scripts. Recipe generation does not start training. The backbone budget includes the integrated neck; heads and train-only loss parameters are reported separately.

Current user-selected workflow: **BEV by default for development and ablations; 3D is reserved for the explicitly selected final-results stage.** The [standard notebook](../../../docs/plans/lightweight_lidar_backbone_2026/notebook_runbook.md) uses `configs/config.json` plus controls; no experiment file override is needed. These eleven materialized files remain useful for standalone CLI reproduction. Final AP3D requires trained z/height branches, so changing a BEV checkpoint's evaluator metric alone is insufficient.

The initial candidate is [hist14_local3_focal_grouped_oga_iqa.json](hist14_local3_focal_grouped_oga_iqa.json). It uses hist14 v1, depth 3/4/2, focal context, no local attention, SG-FPN output32/fusion24, separate Car and Pedestrian/Cyclist heads, Gaussian classification, OGA/IQA and hybrid GT augmentation. Baseline+IQA is a loss companion. ECA/SimAM/fusion32/detail are independently named optional comparisons; they are not enabled together in the main recipe.

| Notebook preset | Runtime config | Backbone + neck | Detector | Criterion, train only |
| --- | --- | ---: | ---: | ---: |
| UNDER1M_SINGLE_OGA_IQA | hist14_local3_single_oga_iqa.json | 635,408 | 728,538 | 6 |
| UNDER1M_GROUPED_OGA_IQA | hist14_local3_grouped_oga_iqa.json | 635,408 | 821,569 | 12 |
| UNDER1M_GROUPED_BASELINE_IQA | hist14_local3_grouped_baseline_iqa.json | 635,408 | 821,569 | 0 |
| UNDER1M_FOCAL_OGA_IQA | hist14_local3_focal_grouped_oga_iqa.json | 660,528 | 846,689 | 12 |
| UNDER1M_FOCAL_BASELINE_IQA | hist14_local3_focal_grouped_baseline_iqa.json | 660,528 | 846,689 | 0 |
| UNDER1M_FOCAL_ECA_OGA_IQA | hist14_local3_focal_eca_grouped_oga_iqa.json | 660,532 | 846,693 | 12 |
| UNDER1M_FOCAL_SIMAM_OGA_IQA | hist14_local3_focal_simam_grouped_oga_iqa.json | 660,529 | 846,690 | 12 |
| UNDER1M_FOCAL_FUSION32_OGA_IQA | hist14_local3_focal_fusion32_grouped_oga_iqa.json | 666,768 | 852,929 | 12 |
| UNDER1M_FOCAL_DETAIL_OGA_IQA | hist14_local3_focal_detail_grouped_oga_iqa.json | 661,720 | 847,881 | 12 |

The standard notebook defaults to `PRESET="custom"` with main focal/grouped OGA-IQA controls and `AUGMENTATION="hybrid_gt"`; editing those controls takes effect immediately. Optional built-in presets still reproduce the materialized architectures without reading auxiliary JSON. Runtime fields such as seed, data path, precision, epochs and batch size take final precedence. `HYBRID_OPTIONS` applies to hybrid augmentation; `standard`, `compose` and `none` replace the queue. Set `C4_CONTEXT="none"` for a controlled focal baseline, or explicitly choose an optional no-context preset.

New candidate run names include explicit context/local/fusion/detail tokens and a semantic digest covering architecture, encoding and objective. Historical default run names keep their format. Changing the new candidate's density normalization, IQA target/weight, activation, groups or feature settings changes its name; backend and training-only settings are recorded separately. An existing run with metadata/checkpoints refuses a different saved config even if the caller supplies its old name.

To reproduce the structural audit from the repository root:

```bash
python3 tools/benchmarks/profile_detector.py --config configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa.json
```

To regenerate all nine BEV configs from the checked-in manifest:

```bash
python3 -c 'from tools.kitti_training_pipeline.notebook_config import write_under1m_presets; write_under1m_presets(".")'
```

The inherited schedule, seed42, split paths, BF16 precision and target backend are starting recipes, not trained KITTI results. BEV/3D CPU, actual CUDA/AMP/worker smoke and pinned reference AP3D/APBEV R11/R40 gates pass; final real-data/comparator protocol, Inductor and trained accuracy remain open. ONNX/TensorRT grouped/IQA/3D deployment remains deferred; legacy BEV IQA-off retains its four-output interface.


## Explicit 3D recipes

| Role | Runtime config | Backbone + neck | Heads | Detector | Criterion |
| --- | --- | ---: | ---: | ---: | ---: |
| Main focal/local-none | [hist14_local3_focal_grouped_oga_iqa_3d.json](hist14_local3_focal_grouped_oga_iqa_3d.json) | 660,528 | 223,413 | 883,941 | 12 |
| Optional focal/ECA | [hist14_local3_focal_eca_grouped_oga_iqa_3d.json](hist14_local3_focal_eca_grouped_oga_iqa_3d.json) | 660,532 | 223,413 | 883,945 | 12 |

Both add `data.box_mode=3d` and explicit `loss.vertical_loss_weight=1.0`, retaining the existing hybrid recipe/seed/schedule. They include independent z_bottom/log-height branches. OGA and IQA retain BEV geometry/quality semantics; 3D decode returns complete Nx9 boxes. Baseline/OGA with Gaussian/binary classification and IQA on/off have synthetic CPU/CUDA FP32 checks; the named OGA/IQA options additionally have FP16/BF16 and workers0/6 smoke/full-resolution inference. Synthetic tiny training is not full-resolution backward or trained KITTI accuracy.

Generate the two 3D configs explicitly, without rewriting BEV files:

```bash
python3 -c 'from tools.kitti_training_pipeline.notebook_config import write_under1m_3d_presets; write_under1m_3d_presets(".")'
```

The standard notebook remains BEV by default. Set `BOX_MODE="3d"`, `VERTICAL_LOSS_WEIGHT=1.0`, `EVALUATION_MODES=["3d","bev"]` and a distinct run tag for final 3D training/evaluation. Preparation includes projection inputs; both metrics route to the pinned evaluator with separate R11/R40. Reference CUDA parity checks and formal benchmark evidence are run separately from the simplified training notebook. See the [Colab runbook](../../../docs/plans/lightweight_lidar_backbone_2026/colab_3d_runbook.md), [reference guide](../../../docs/plans/lightweight_lidar_backbone_2026/reference_evaluation.md) and [3D verification](../../../docs/plans/lightweight_lidar_backbone_2026/release_b_verification.md).

For the current BEV experiments, use [task51 Colab workflow](../../../docs/plans/lightweight_lidar_backbone_2026/task51_colab_runbook.md) to audit configured data/GT database, evaluate the no-context reference with its own resolved config, record complete checkpoint/selection evidence and freeze the candidate protocol. Local BEV remains R40-only; reference R11/R40 is a separately selected final-stage protocol.
