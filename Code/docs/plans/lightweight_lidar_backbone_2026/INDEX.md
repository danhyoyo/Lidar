# Lightweight LiDAR detector: specification and implementation package

Design revision 2 — updated 2026-10-07. Language: English. Status: 57/58 tasks verified; final benchmark freeze pending; see [execution log](execution_progress.md). Accuracy validation is pending.

Current experiment policy, confirmed by the user: **BEV is the default for development, ablations and intermediate comparisons.** The notebook now uses `configs/config.json` as its only source JSON, with `PRESET="custom"`, main focal/grouped OGA-IQA controls and `AUGMENTATION="hybrid_gt"`. Edit the notebook controls; no experiment file override is needed. See the [unified notebook guide](notebook_runbook.md). Final 3D controls select trained vertical branches and AP3D/APBEV with R11/R40 separately; a metric switch cannot give a BEV checkpoint z/height predictions.

## Read in this order

1. [Main specification](specification.md): objectives, hist14 formulas, model/target contracts, grouped losses, OGA/IQA compatibility, decode, checkpoint, GPU and 3D evaluation requirements.
2. [Attention specification](attention_specification.md): exact focal/ECA/SimAM operators, placement, residual initialization, parameter accounting, optional detail and fusion designs.
3. [Implementation plan](implementation_plan.md): 58 incremental tasks with files, changes, verification commands, risks and rollback.
4. [Experiment manifest](experiment_manifest.json): eleven named recipes and overrides; nine complete BEV configs and two complete 3D configs are materialized. This manifest is not a runtime training config.
5. [Parameter projections](attention_budget_projection.json): meta-device topology counts and source hashes; these are not AP, measured latency or numerical integration results.

Implementation progress: **57/58 tasks** are complete (1–50 and52–58). Task51 now has executable audit/evidence/freeze tooling; its final real-data/comparator freeze on Colab remains open. Encoding, focal/lightweight backbone features, grouped heads/targets, OGA/IQA, strict semantic/optimizer/RNG resume, hybrid integration, vertical prediction/loss, calibrated Nx9 decode and pinned reference KITTI evaluation are implemented. Main focal backbone including neck: **660,528** parameters; focal+ECA **660,532**. The main 3D detector has **883,941** parameters (3D grouped heads223,413), excluding12 train-only OGA parameters. See [runtime recipes and counts](../../../configs/experiments/under1m/README.md).

Batch20 verifies actual CUDA FP32 objectives for baseline/OGA × Gaussian/binary × IQA on/off on main/ECA, AMP FP16/BF16, workers0/6, finite vertical gradients, strict resume and full-resolution inference. Saved synthetic GPU predictions agree with direct pinned reference evaluation for AP3D/APBEV and R11/R40. The final full suite passes **1,480 tests and17 subtests**, with zero skips. GPU reports identify CPU polygon NMS fallback; FP16 initially overflows in the tiny smoke fixture and passes after scaling/retrying the same supervised batch. This does not measure real KITTI AP or full-resolution training throughput. Inductor, Python3.10 and Jetson/TensorRT tuning remain deferred. See [Milestone B verification](release_b_verification.md), [execution log](execution_progress.md), [vertical interface](vertical_interface.md) and [reference evaluation guide](reference_evaluation.md).

Use the [unified notebook guide](notebook_runbook.md) and [Colab 3D runbook](colab_3d_runbook.md) for mode controls, actual asset audit and own-checkpoint reference evaluation. Development records pending comparator gates; benchmark mode requires actual matched evidence before full training/resume. The [candidate benchmark protocol](benchmark_protocol.md) keeps local ROI R40 separate from final reference R11/R40. Real Colab freeze and trained accuracy remain pending. Historical BEV scope is documented in [Milestone A verification](release_a_verification.md).

## Selected architecture

```text
Raw points / boxes
  → Existing hybrid GT + global augmentation
  → hist14 v1 encoding
  → Legacy stride-2 stem (32 channels)
  → Stride-4 stage (48 channels, depth 3)
  → Optional residual ECA OR SimAM (default none)
  → Stride-8 stage (96 channels, depth 4)
  → Feature-based multiscale focal context (main recipe)
  → Stride-16 stage (128 channels, depth 2)
  → SG-FPN, output stride 4 / 32 channels
  → Car head + Pedestrian/Cyclist head
  → Supported group-specific losses / optional IQA
  → Group-correct decode / global class mapping / classwise NMS
```

The neck also receives the stride-4 and focal-refined stride-8 features. Optional internal fusion width 32 and the stem-detail branch are follow-ups, disabled in the main recipe. Support-conditioned focal gates, dense stage aggregation, distillation and Jetson/TensorRT tuning are outside the main release.

## Recipe and budget

| Recipe | Backbone including neck, verified | Role |
| --- | ---: | --- |
| No-context/no-local-attention reference | 635,408 | Functional/architecture comparison |
| Focal, local attention none | 660,528 | Main recipe |
| Focal + ECA k3 residual adapter | 660,532 | First optional attention comparison |
| Focal + SimAM residual adapter | 660,529 | Optional comparison |

All use the same grouped-head contract. Two BEV heads with IQA add 186,161 parameters; the implemented two vertical branches add 37,252. The instantiated main 3D head topology has 883,941 parameters, with weighted vertical loss/calibrated decode/reference evaluation implemented, while the user budget remains strictly <1M for backbone including neck. Train-only criterion parameters are reported separately.

## Delivery sequence and evidence

- Build encoding/reference tests and preserve historical configuration behavior.
- Implement and verify the focal/context/lightweight modules, then grouped heads/targets/losses, decode and stateful training.
- Pass CPU and supported GPU/AMP smoke gates for advertised options. An optional recipe is supported only after its corresponding checks pass.
- Complete vertical targets/head/loss and pinned reference KITTI 3D evaluation before a 3D comparison or the sole long training run intended for that comparison.
- Freeze and train one selected main recipe if time is limited. Implementing recipe options does not require training all of them; accuracy/efficiency claims need the appropriate measured evidence.

OGA and baseline can supervise the separate IQA branches. Existing unsupported combinations remain rejected, including exact Q-OGA+IQA. Attention changes do not bypass these restrictions.

## Historical background

[Backbone design decisions](backbone_design_options.md), [initial encoding brainstorm](encoding_brainstorm.md), [earlier implementation outline](implementation_steps.md), [earlier grouped-head checklist](grouped_heads_compatibility.md) and the [research report](README.md) remain available as background. The revision-2 specifications and implementation plan take precedence where scope differs.

Implementation now follows the documented batch checkpoints. Long training, benchmark submission and deployment remain separate gates.

Task51 delivery: [implementation plan](task51_implementation_plan.md), [CLI Colab runbook](task51_colab_runbook.md) and the now integrated [notebook workflow](notebook_runbook.md). Tools check actual assets, full evaluation/input hashes and minimum-loss checkpoint selection before permitting a matched comparison freeze. Baseline and candidate use their own resolved configs. Notebook controls now directly select the main architecture; the model implementation, master config and eleven materialized runtime recipes remain unchanged.
