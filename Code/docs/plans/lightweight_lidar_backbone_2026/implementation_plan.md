# Implementation plan: hist14, focal CNN backbone, lightweight attention and compatible grouped heads

Updated: 2026-10-07. Design revision 2.

### Goal

Implement [specification.md](specification.md) and [attention_specification.md](attention_specification.md) on `research/mobilepixornext-under1m` in reviewable increments. Deliver the GPU-ready BEV candidate, then complete vertical prediction/reference evaluation before any 3D comparison. Task-level execution status is tracked in [execution_progress.md](execution_progress.md); interfaces for pending tasks remain proposals.

Main architecture: **hist14 → legacy stem → local stage depth 3 → optional ECA/SimAM → stride-8 stage depth 4 + focal context → stride-16 stage depth 2 → SG-FPN output 32 → Car and Pedestrian/Cyclist heads**, without LiteMLA. Main local attention is none; ECA is the first optional comparison. Preserve supported OGA/IQA paths and historical single-head behavior. Distillation, support-conditioned context, dense stage redesign and Jetson tuning are excluded. Optional detail/wider-fusion work is separately gated.

Latest experiment policy: **keep BEV as the default for development, ablations and intermediate comparisons; select 3D only for final results.** Completed 3D implementation does not activate it in the notebook. Use the main BEV preset with no file override now. For final results, explicitly select the separate 3D config, train or fine-tune its z/height branches, and evaluate both AP3D/APBEV with R11/R40 recorded separately. A BEV checkpoint cannot supply AP3D by changing only the evaluation flag; final backbone comparisons need matched 3D heads/objectives and training conditions.

### Assumptions

- Requirements R01–R16, configuration and tensor/operator contracts are defined in the two specifications. Each task needs passing verification evidence before its feature is enabled in a training preset. R16 is conditional on optional detail/fusion being enabled.
- Hybrid GT augmentation is integrated at `3654a0b`. Preserve the dirty notebook, `iou_targets.py`, `q_oga.py`, `train.py` and compatibility tests; do not reset or accidentally commit unrelated changes.
- Implementation was authorized on 2026-10-07 using the executing-plans skill. Execute reviewable batches; long training, benchmark submission and deployment remain separate gates.
- Keep grid 0.1 m, output stride 4, original target radii and hybrid augmentation recipe. Test numerical behavior on small asymmetric geometries; audit full KITTI shapes separately.
- Each task is intended as a small edit/test cycle, typically 2–10 minutes once fixtures exist. Split a task further if it exposes a larger refactor. Use failing behavioral tests before nontrivial changes; avoid tests that merely repeat construction.
- **New** marks a planned file. Verification commands referring to these files are future acceptance commands. Existing import styles used by detector and pipeline must continue working.
- CPU environment for this workspace:

  ```bash
  export LIDAR_PYTHON=/tmp/lidar-fixes-env/bin/python
  export LIDAR_REPO=/home/duyennh/AI_projects/research_lidar/Lidar
  ```

  Run from repository root. Use equivalent paths on another machine. GPU tasks need CUDA; explicit compiled encoding needs Numba. DataLoader multiprocessing requires a runner allowing process IPC.
- Suggested commit boundaries: schema/reference encoder; compiled processing; backbone config; groups/targets; grouped loss; decoding; trainer/checkpoint; notebook/GPU integration; separate 3D package. Inspect staged diffs for unrelated work at each boundary.
- Context and lightweight adapters have their own feature commit boundary before grouped integration. An optional detail/fusion task may be deferred while the main recipe keeps it off; deferred options must fail clearly rather than be advertised as supported.
- If only one long training run is affordable and comparisons require 3D AP, finish the 3D tasks after BEV smoke checks and **before** that run.

### Plan

1. **Capture legacy behavior and working-tree provenance** — R10, R12
   - Files: `tests/test_legacy_detector_contract.py` (**new**); baseline verification artifact.
   - Change: record HEAD, dirty provenance/config/split hashes; characterize deterministic outputs, targets, loss/gradients and state keys on a small grid, including IQA and adaptive state.
   - Verify: `git status --short`; `git diff --check`; `$LIDAR_PYTHON -m pytest -q tests/test_legacy_detector_contract.py tests/test_loss_strategies.py tests/test_iqa_header.py tests/test_small_object_assignment.py`. Capture expectations before refactoring.

2. **Implement the pure encoding schema** — R02, R03
   - Files: `detector/core/bev_encoding.py` (**new**); `tests/test_bev_encoding_spec.py` (**new**).
   - Change: resolve channels/version/layout/normalization/geometry, hist14 edges and semantic hash. Preserve rich padding and binary-slice behavior; reject invalid hist14 options.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_bev_encoding_spec.py`; cover unknown names/versions, invalid geometry, channels exactly 14 and backend-independent semantic identity.

3. **Replace duplicated pipeline/model channel resolution** — R02, R10
   - Files: `detector/core/models/model.py`; `tools/kitti_training_pipeline/common.py`; `tools/kitti_training_pipeline/notebook_config.py`; config/run-name tests.
   - Change: consume the shared schema for input channels and shape. Preserve supported import contexts, legacy state keys and legacy run naming.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_bev_encoding_spec.py tests/test_kitti_config_compatibility.py tests/test_run_name_and_logging.py tests/test_model_registry.py`; dataset, model and pipeline channel/geometry resolution agree.

4. **Write NumPy hist14 reference encoding** — R03
   - Files: `detector/core/datasets/utils_1/bev_backend.py` (**new**); `tests/test_hist14_encoding.py` (**new**).
   - Change: implement exact filtering/channel formulas, matching float64 coordinate/index/normalization arithmetic and accumulation, then contiguous float32 output; clamp variance and preserve signed xy residuals.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_hist14_encoding.py`; analytic empty/one-point/multi-bin fixtures, internal ties and nextafter boundary neighbors, epsilon boundaries, nonfinite inputs, saturation, anisotropic grid and permutation invariance.

5. **Connect hist14 dispatch and augmentation ordering** — R02–R04, R10
   - Files: `detector/core/datasets/utils_1/preprocess.py`; `detector/core/datasets/dataset.py`; hist14/rich tests.
   - Change: dispatch hist14 through the shared schema/reference kernel without changing legacy paths. Encoding remains after sampling/global transforms, with HWC→CHW conversion unchanged.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_hist14_encoding.py tests/test_rich_encodings.py tests/test_legacy_detector_contract.py`; transformed synthetic points appear at analytically expected cells.

6. **Add explicit Numba encoding backend** — R04
   - Files: `detector/core/datasets/utils_1/bev_backend.py`; `detector/core/datasets/utils_1/preprocess.py`; `tests/test_hist14_backend.py` (**new**).
   - Change: one-pass accumulation then dense normalization; no semantic-changing fastmath. Explicit missing Numba requests fail clearly; NumPy remains selectable.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_hist14_backend.py tests/test_hist14_encoding.py`; all-channel parity at `atol=1e-6,rtol=1e-5` on analytic/randomized/shuffled/empty clouds and dependency-unavailable fixtures.

7. **Benchmark encoding and check hybrid integration** — R04, R12
   - Files: `tools/benchmarks/benchmark_bev_encodings.py` (**new**); `tests/test_hist14_hybrid_integration.py` (**new**).
   - Change: CLI reports geometry, point count, dtype/backend/hardware, separate JIT/warmup and steady-state mean/p50/p95. Test that hybrid augmentation modifies raw points/boxes before encoding; no pre-augmentation BEV caching.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_hist14_hybrid_integration.py tests/test_hybrid_augmentation.py`; implemented CLI: `$LIDAR_PYTHON tools/benchmarks/benchmark_bev_encodings.py --synthetic --points 10000 --warmup 3 --iterations 20 --output /tmp/hist14_encoding_benchmark.json`.

8. **Parameterize stage depths with legacy key parity** — R05, R10
   - Files: `detector/core/models/backbones/mobilepixornext.py`; `detector/core/models/backbones/registry.py`; backbone/registry tests.
   - Change: validate three positive integer depths, default [2,4,2]; construct the existing blocks preserving names/indices. Candidate [3,4,2] adds `stage2.2`.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_mobilepixornext_backbone.py tests/test_model_registry.py tests/test_legacy_detector_contract.py`; malformed-depth rejection, default state keys/count parity and finite small-grid backward.

9. **Audit the no-context 32-channel reference backbone** — R01, R05
   - Files: `tools/benchmarks/profile_detector.py` (**new**); `tests/test_lightweight_detector_budget.py` (**new**); backbone tests.
   - Change: profile actual input14/depth3-4-2/SG-FPN/output32 constructor; report body/neck/head/total and criterion parameters separately. Clearly label convolution MAC estimates.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_lightweight_detector_budget.py tests/test_mobilepixornext_backbone.py`; full `[1,14,800,704]` input gives neck `[1,32,200,176]`, no LiteMLA instance, backbone including neck <1M. Expected structural count 635,408; investigate discrepancies.

10. **Resolve focal/local-attention/fusion configuration** — R12, R14–R16
    - Files: `detector/core/backbone_config.py` (**new**); `tests/test_backbone_feature_config.py` (**new**).
    - Change: pure resolver for the attention-spec fields, historical defaults, kernel/dilation/lambda/version/scale validation and mutually exclusive focal/LiteMLA. Central detection validation will consume this resolver later.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_backbone_feature_config.py`; reject booleans, invalid scalar values, conflicting/ignored fields, wrong backbone/neck selections and unsupported candidate grid alignment; missing new fields resolve to legacy topology.

11. **Create focal pre-normalization and projection interface** — R14
    - Files: `detector/core/models/backbones/focal_context.py` (**new**); `tests/test_focal_context.py` (**new**).
    - Change: implement the standalone module interface/validation and C→2D+L+1 projection with Q/S/G split. No hidden convolution biases or pooled BN.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_focal_context.py -k projection`; shapes, invalid channels/settings, input device/dtype and bias-free topology match the operator contract.

12. **Generate sequential context levels and stable gates** — R14
    - Files: `detector/core/models/backbones/focal_context.py`; `tests/test_focal_context.py`.
    - Change: sequential DW3 dilations 1/2/3 plus SiLU; global pool the final level in FP32. Softmax the four gate logits in FP32; do not use parallel levels or an occupancy hard mask.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_focal_context.py -k 'levels or gates or pooling'`; known-weight/impulse fixtures establish sequential receptive fields, normalized weights, correct broadcast and eval batch-item isolation.

13. **Complete modulation and near-identity residual** — R14
    - Files: `detector/core/models/backbones/focal_context.py`; `tests/test_focal_context.py`.
    - Change: context PW64→64, FP32 modulation product, PW64→96, output BN and channel LayerScale initialized to 1e-3. Define gamma-zero identity fixture separately from the nonzero-scale training preset.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_focal_context.py -k 'identity or gradient or parameters'`; output shape, finite nonzero branch/scale gradients, exact finite eval identity at gamma=0 and parameter count 25,120.

14. **Verify context numerical and state behavior** — R08, R14
    - Files: `tests/test_focal_context.py`; `detector/core/models/backbones/focal_context.py` only for required fixes.
    - Change: cover zero/constant/sparse/random maps, asymmetric shapes, BN train/eval behavior, state restore and supported autocast boundaries. Treat training BN single-element constraints explicitly.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_focal_context.py`; all CPU tests pass, deterministic eval reload matches and no inplace residual corruption occurs. GPU precision/memory checks occur at the later CUDA gate.

15. **Implement the minimal ECA core** — R15
    - Files: `detector/core/models/backbones/light_attention.py` (**new**); `tests/test_light_attention.py` (**new**).
    - Change: FP32 spatial pooling, bias-free Conv1d across channels, sigmoid/broadcast; k3 default, k5 explicit option. Initialize Conv1d to zero in this adapter version.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_light_attention.py -k eca`; core has exactly k weights, initial gate 0.5, spatial broadcasting is correct and malformed kernels fail.

16. **Implement SimAM with explicit numerical guards** — R15
    - Files: `detector/core/models/backbones/light_attention.py`; `tests/test_light_attention.py`.
    - Change: FP32 mean/squared deviation/variance/energy/sigmoid, finite positive lambda and `max(H*W-1,1)` denominator. Core has no learned parameters and does not infer point-support masks.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_light_attention.py -k simam`; analytic constant/impulse maps, single-element grid, zero/sparse input, lambda validation and parameter-free core agree with the specification.

17. **Wrap ECA/SimAM with learned scalar residuals** — R08, R15
    - Files: `detector/core/models/backbones/light_attention.py`; `tests/test_light_attention.py`.
    - Change: `X+beta*(T(X)-X)` with one registered scalar beta initially 1e-3; none is pure identity. Do not append attention to every block or gate detection logits.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_light_attention.py`; exact beta-zero identity, initial ECA scale 0.9995, nonzero beta/kernel gradients, dtype/state restoration and total counts ECA k3=4 / SimAM=1.

18. **Optionally widen SG-FPN final fusion** — R01, R16
    - Files: `detector/core/models/backbones/mobilepixornext.py`; `tests/test_neck_detail_options.py` (**new**). The complete `configs/experiments/under1m/hist14_local3_focal_fusion32_grouped_oga_iqa.json` recipe is created in task 48 after focal/grouped construction is supported; task 18 implements the constructor capability without publishing an incomplete runtime preset.
    - Change: F=24 default or 32 explicit in stride-4 lateral, top-down projection, gate and final output projection. Preserve 48-channel upper laterals and existing fusion formula.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_neck_detail_options.py -k fusion`; F24 legacy parity, F32 output unchanged, exact +6,240 parameter delta with scale gates enabled, finite backward and rejected unsupported RC neck combinations. If deferred, selection remains explicitly unsupported.

19. **Optionally add stem-to-neck detail fusion** — R01, R16
    - Files: `detector/core/models/backbones/mobilepixornext.py`; `tests/test_neck_detail_options.py`. The complete `configs/experiments/under1m/hist14_local3_focal_detail_grouped_oga_iqa.json` recipe is created in task 48 after grouped construction is supported, as with fusion32.
    - Change: DW3 stride2/BN/SiLU/PW32→F/BN/SiLU plus per-channel gamma; fuse after standard stride-4 fusion before out projection, initial gamma 1e-3. No hidden interpolation.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_neck_detail_options.py -k detail`; asymmetric-grid alignment, gamma-zero identity, finite gradients, optional-off legacy parity and parameter additions 1,192 at F24 / 1,472 at F32. Deferred support is reported as unsupported.

20. **Wire the two attention hooks into the backbone and registry** — R05, R10, R14–R16
    - Files: `detector/core/models/backbones/mobilepixornext.py`; `detector/core/models/backbones/registry.py`; `tools/kitti_training_pipeline/notebook_config.py`; backbone/registry/config tests.
    - Change: refined local features feed down3 and the neck; selected stride-8 context feeds down4 and the neck. Disabled hooks are parameter-free identities; preserve historical LiteMLA/default state keys. Do not import optional modules into unrelated backbones unnecessarily.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_mobilepixornext_backbone.py tests/test_model_registry.py tests/test_backbone_feature_config.py tests/test_legacy_detector_contract.py tests/test_focal_context.py tests/test_light_attention.py`; hook-routing fixtures and legacy/no-context parity.

21. **Profile attention variants and update the hard budget gate** — R01, R14–R16
    - Files: `tools/benchmarks/profile_detector.py`; `tests/test_lightweight_detector_budget.py`; `tests/test_neck_detail_options.py` if optional variants are implemented.
    - Change: report reference/focal/focal+ECA/focal+SimAM separately with core/wrapper/neck/heads/criterion accounting. Add optional detail/fusion rows only when their support exists.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_lightweight_detector_budget.py`; actual backbone counts should match projections 635,408 / 660,528 / 660,532 / 660,529. Audit `[1,14,800,704]` outputs on meta/eval paths and maintain strict <1M, without calling MACs measured latency.

22. **Resolve groups and stable class mappings** — R06, R12
    - Files: `detector/core/task_groups.py` (**new**); `tests/test_task_groups.py` (**new**).
    - Change: exact class partition/name/order validation and Gaussian/binary local/global conversions from `data.head_groups` and `data.kitti.objects`.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_task_groups.py`; test valid reordered/non-default groups and reject missing/duplicate/unknown classes, invalid names and malformed global IDs.

23. **Centralize configuration/capability validation** — R07, R12
    - Files: `detector/core/detection_config.py` (**new**); `tools/kitti_training_pipeline/common.py`; `tools/kitti_training_pipeline/notebook_config.py`; `tests/test_detection_config.py` (**new**); `tests/test_small_object_assignment.py` (align the evaluator fixture's IQA head/supervision flags).
    - Change: validate head mode/groups, stride/classification, IQA agreement and strategy support at public construction boundaries. Consume the shared backbone-option resolver for focal/local/fusion/detail settings; preserve Q-OGA exact Gaussian/BEV/IQA restrictions.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_detection_config.py tests/test_kitti_config_compatibility.py tests/test_q_oga_exact_quality.py tests/test_run_name_and_logging.py tests/test_small_object_assignment.py`; CLI/evaluator cannot bypass notebook restrictions or create unsupervised IQA branches. The current tree consolidated/deleted the earlier notebook/quality-config test files; `test_detection_config.py` now checks notebook boundaries and the three existing exact-quality recipes directly. Add grouped mapping/weight identity to run names now, preserving legacy names; task47 still supplies the complete notebook controls/presets.

24. **Compose independent task heads** — R05–R07, R10
    - Files: `detector/core/models/heads/grouped.py` (**new**); `detector/core/models/model.py`; `tools/kitti_training_pipeline/common.py`; `tests/test_grouped_header.py` (**new**).
    - Change: one existing `Header` per group in ModuleDict, optional IQA in each. Legacy mode retains its existing `model.header`; construction uses copied config.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_header.py tests/test_iqa_header.py tests/test_legacy_detector_contract.py`; Gaussian widths 1/2, binary widths 2/3, separate regression storage. Expected two-group head counts: 148,975 off / 186,161 IQA on.

25. **Extract reusable single-group target construction** — R06, R10
    - Files: `detector/core/datasets/dataset.py`; `tests/test_grouped_targets.py` (**new**, reusable-builder tests); legacy/small-object tests.
    - Change: pass explicit local class count/mapping; never temporarily modify dataset class state. Preserve backend ownership and external axis layout.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_targets.py tests/test_legacy_detector_contract.py tests/test_small_object_assignment.py tests/test_regressions.py`; legacy target values, reserved peaks and canonical ties match pre-refactor fixtures. Original global IDs are retained for regression tie-breaking while classification IDs are remapped locally.

26. **Build independent Gaussian group targets** — R06
    - Files: `detector/core/datasets/dataset.py`; `tests/test_grouped_targets.py`; `tests/test_hist14_hybrid_integration.py`.
    - Change: globally augment/filter boxes once, partition/remap, then generate each group's own heatmap and regression maps. Keep validation metadata at the top level.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_targets.py tests/test_hist14_hybrid_integration.py`; Car/Pedestrian at the same cell retain both boxes; empty groups are valid; Ped/Cyclist within-group collision follows existing ownership.

27. **Implement binary local target labels** — R06, R07
    - Files: `detector/core/datasets/dataset.py`; group target/mapping tests.
    - Change: group background is 0, foreground 1..C_group; do not mutate globally stored labels/boxes or reuse another group's mask.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_targets.py tests/test_task_groups.py`; background-only, both Ped/Cyclist labels and cross-group overlap fixtures.

28. **Verify target backends and DataLoader collation** — R04, R06, R11
    - Files: group target tests; `tests/test_grouped_pipeline.py` (**new**); target backend only if a defect requires it.
    - Change: test Python/Numba per-group parity, anisotropic dimensions, nested collation and validation/test metadata paths. Include real two-worker spawn loaders for Gaussian/binary × Python/Numba; warm Numba in the parent and compare every tensor leaf with its parent reference.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_targets.py tests/test_grouped_pipeline.py tests/test_small_object_assignment.py`; all leaf shapes conform; legacy batches remain flat.

29. **Make nested batch transfer work on the selected device** — R11
    - Files: `tools/kitti_training_pipeline/train.py`; `tests/test_grouped_training.py` (**new**).
    - Change: recursively transfer tensors in mappings/lists/tuples while preserving non-tensor metadata, order and nonblocking behavior.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_training.py`; nested leaves reach the requested device and metadata survives. Dedicated CUDA/AMP verification is tracked by task50; task36 adds CPU optimizer/compile/trainer integration checks.

30. **Implement normalized grouped loss composition** — R07, R10
    - Files: `detector/core/losses/grouped.py` (**new**); `detector/core/losses/loss_fn.py`; `tests/test_grouped_losses.py` (**new**).
    - Change: factory returns legacy facade unchanged or independent per-group criteria; normalized fixed group weights include empty groups. Expose a differentiable total and detached flat scalar diagnostics. Aggregate core loss components by objective weights; retain other diagnostics per group until count-based quality aggregation in task35. Trainer factory wiring remains task36.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_losses.py tests/test_loss_strategies.py tests/test_legacy_detector_contract.py`; explicit weighted sum, default 0.5/0.5, invalid weights, singleton criterion loss/gradient parity after state alignment.

31. **Integrate adaptive strategies without sharing state** — R07, R08
    - Files: grouped loss module/tests; `tests/test_grouped_loss_state.py` (**new**).
    - Change: independently register OGA/UWAG/GW-QAL/legacy Q-OGA parameters and buffers, preserving their formulas and train/eval state behavior.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_losses.py tests/test_grouped_loss_state.py tests/test_loss_strategies.py tests/test_q_oga_exact_quality.py`; group-only loss does not update another head/criterion, and eval does not update EMAs. Historical strategy test files were consolidated/deleted in the incoming workspace; preserve those changes and use current regression coverage.

32. **Check IQA routing and detach boundaries** — R07, R09
    - Files: `tests/test_grouped_iqa.py` (**new**); grouped loss/head and existing IQA code only if needed.
    - Change: exercise baseline/OGA with supported quality methods; each IQA branch uses its own assigned box and detached quality targets, including assigned boxes requiring gradients. Reject unsupported IQA combinations. Batch11 corrects only target-gradient leakage with `no_grad` on the two existing proxy helpers, preserving their numerical formulas.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_iqa.py tests/test_iqa_header.py tests/test_rotated_iou_targets.py`; perfect/mismatched quality ordering, IQA-only gradients in its own branch, no box gradient through target computation, graph-connected empty-mask zeros.

33. **Characterize binary strategy support** — R07, R10
    - Files: `tests/test_grouped_loss_capabilities.py` (**new**); capability validator; grouped loss tests.
    - Change: verify baseline/OGA/UWAG binary training. Include the batch9 fixture documenting legacy last-box-wins binary classification versus canonical nearest-center regression when different classes overlap inside one group; characterize implications for quality supervision before a binary accuracy claim. Audit actual legacy GW-QAL/Q-OGA binary behavior before advertising grouped support; extend matrix only with target/loss/decode parity evidence. Any ownership correction must be explicit and preserve legacy defaults.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_loss_capabilities.py tests/test_grouped_losses.py`; unsupported grouped combinations fail explicitly, with legacy Gaussian behavior preserved.

34. **Broadcast Q-OGA exact curriculum and persist buffers** — R07, R08
    - Files: grouped loss module; `tests/test_grouped_qoga.py` (**new**); group state tests.
    - Change: `set_epoch` broadcasts the same zero-based epoch; exact quality uses each group's own peaks/regression. Preserve IQA/binary/BEV restrictions.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_qoga.py tests/test_q_oga_exact_quality.py tests/test_detection_config.py tests/test_kitti_config_compatibility.py`; warmup start/end, missing epoch, empty group, peak/mask alignment, collision limitation and strict curriculum restore. Use current consolidated configuration tests; the historical `test_qoga_quality_configs.py` is absent from the incoming workspace.

35. **Aggregate quality diagnostics by counts** — R08, R11
    - Files: grouped loss module; `tools/kitti_training_pipeline/train.py`; `tests/test_grouped_metrics.py` (**new**).
    - Change: aggregate scalar components plus `group/<name>/<metric>`; quality means use sufficient statistics, Q-OGA peak counts and IQA supervised regression counts (`iou_target_count`). Exclude quality diagnostics from generic sample averaging, including namespaced metrics and legacy OGA's `mean_iou_target`. Derive its count from the flat target mask without changing the legacy criterion.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_metrics.py tests/test_training_pipeline_oga.py`; unequal counts/empty groups, count addition without loss-weight scaling, curriculum mix agreement and only total `loss` used for optimization.

36. **Wire criterion factory, optimizer and compile warmup** — R08, R11
    - Files: `tools/kitti_training_pipeline/train.py`; grouped training tests; optimizer/scheduler tests.
    - Change: use the same resolved groups for datasets and criterion, move criterion before optimizer creation, collect all head/criterion parameters exactly once. Warmup scalar traverses every active branch, preserves BatchNorm/criterion state and clears gradients even after failure; clipping includes criterion parameters. Preserve existing optimizer/scheduler/clipping/accumulation formulas.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_training.py tests/test_optimizer_scheduler.py tests/test_training_pipeline_oga.py`; finite one-step updates, optimizer membership, empty groups, eval-mode stability and legacy training parity.

37. **Verify adapters across grouped losses and quality supervision** — R07–R08, R14–R15
    - Files: `tests/test_context_grouped_compatibility.py` (**new**); grouped/header/criterion implementations only for required fixes.
    - Change: parameterize none/ECA/SimAM with focal context across every supported Gaussian strategy and representative binary baseline/OGA/UWAG paths. Include empty groups, cross-group same-cell boxes and gradient routing into shared adapters.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_context_grouped_compatibility.py tests/test_grouped_iqa.py tests/test_grouped_qoga.py tests/test_grouped_training.py`; active adapters receive finite gradients at nonzero scales, model/criterion optimizer membership has no duplicates, detached quality targets remain detached, and unsupported exact-Q-OGA+IQA fails.

38. **Save architecture/objective identity and validate strict resume** — R08, R12
    - Files: `tools/kitti_training_pipeline/common.py`; `tools/kitti_training_pipeline/train.py`; `tests/test_grouped_checkpoint.py` (**new**).
    - Change: save semantic schema/groups/depth/width/box mode/quality and focal/local-attention/fusion/detail identity with model/criterion/optimizer/scheduler states. Include BN/gamma/beta state; validate before loading and preserve matching legacy loading. Backend equivalence is recorded separately.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_checkpoint.py tests/test_q_oga_exact_quality.py tests/test_kitti_config_compatibility.py`; reject changed groups/classes/input/IQA/depth/context/dilations/kernel/lambda/fusion/detail and incompatible resume mode before partial restore; matching adapter state roundtrips.

39. **Verify adaptive resume and supported RNG restoration** — R08, R12
    - Files: `tools/kitti_training_pipeline/train.py`; group checkpoint/state tests.
    - Change: save/restore Python/NumPy/Torch RNG and explicit DataLoader-generator state for supported epoch-boundary resume. Restore criterion buffers before validation/training; document persistent-worker replay limits.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_checkpoint.py tests/test_grouped_loss_state.py`; uninterrupted versus restored deterministic optimizer steps match loss/weights/EMAs/curriculum/scheduler with workers 0. Do not promise worker/device-independent bitwise replay.

40. **Add model-only warm-start when needed** — R10, R12
    - Files: `tools/kitti_training_pipeline/train.py`; `tools/kitti_training_pipeline/common.py`; `tests/test_grouped_checkpoint.py`.
    - Change: optional `--warm-start`, mutually exclusive with `--resume`, copies compatible backbone keys/shapes only and reports loaded/skipped keys. Changed input layer, new block, missing context/adapters and heads remain fresh; criterion/optimizer/epoch reset. Do not force-copy LiteMLA weights into focal context.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_checkpoint.py`; inspect values, first-layer/head skips and fresh training state. Optional if training from scratch; unsupported warm-start requests still fail clearly.

41. **Separate candidate decode and final NMS without legacy drift** — R09, R10
    - Files: `detector/postprocess.py`; `tests/test_grouped_decode.py` (**new**); legacy contract tests.
    - Change: extract reusable decode/candidate/NMS helpers while preserving flat thresholds, rows, empty shape and ordering. Do not combine group heatmaps with fictitious shared regression.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_legacy_detector_contract.py tests/test_regressions.py tests/test_grouped_decode.py`; known flat predictions preserve decoded boxes and suppression.

42. **Decode grouped Gaussian boxes and their own quality scores** — R09
    - Files: `detector/postprocess.py`; group decode/IQA tests.
    - Change: use the matching group regression/IQA, remap local to global IDs, retain metric cell-origin/log-size ordering. Reject malformed/missing groups.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_decode.py tests/test_grouped_iqa.py`; asymmetric-coordinate roundtrip, Car-only IQA change affects Car only, cross-group same-cell objects survive.

43. **Decode supported binary groups correctly** — R07, R09
    - Files: `detector/postprocess.py`; group decode/capability tests.
    - Change: local softmax, discard background, foreground k maps through `group_ids[k-1]`. Preserve flat Gaussian behavior; any historical binary correction is an explicit separate option/change.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_decode.py tests/test_grouped_loss_capabilities.py`; background-dominant cells do not emit foreground, and Ped/Cyclist global IDs and scores are correct.

44. **Merge candidates before classwise NMS and global limits** — R09, R10
    - Files: `detector/postprocess.py`; group decode tests; `tests/test_grouped_evaluation.py` (**new**).
    - Change: retain per-class peaks/NMS, globally sort and apply max-detection cap once. Grouped legacy peaks use within-group maximum; preserve BEV float32 `[N,7]` and empty `(0,7)`.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_decode.py tests/test_grouped_evaluation.py`; cross-class overlap survives, same-class duplicates suppress, group order/caps behave consistently except documented exact score ties.

45. **Integrate PyTorch evaluation and parameter reports** — R01, R09, R12
    - Files: `tools/kitti_training_pipeline/evaluate_kitti_bev.py`; `tools/kitti_training_pipeline/compare_models.py`; `tools/benchmarks/profile_detector.py`; group evaluation tests.
    - Change: validate grouped checkpoints, route decode, and record mappings/quality/thresholds/protocol/config/split hashes plus parameter breakdown. Preserve current BEV evaluator ROI and difficulty policy.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_evaluation.py tests/test_lightweight_detector_budget.py`; a synthetic grouped checkpoint and saved predictions produce correct class attribution and metadata.

46. **Guard deferred deployment against silent output loss** — R07, R12
    - Files: `tools/kitti_training_pipeline/export_onnx.py`; `tools/kitti_training_pipeline/evaluate_kitti_bev.py`; `tests/test_grouped_export_contract.py` (**new**).
    - Change: reject grouped export/TRT loading explicitly until the later deployment package implements their contract. Legacy IQA-off keeps four names; IQA-on must include quality output or reject the unsupported consumer, never silently drop it.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_export_contract.py`; legacy output parity and informative IQA/grouped unsupported-path errors. TensorRT/Jetson tuning remains deferred.

47. **Create complete presets and notebook controls** — R02, R05–R07, R12
    - Files: `configs/experiments/under1m/hist14_local3_single_oga_iqa.json`, `configs/experiments/under1m/hist14_local3_grouped_oga_iqa.json`, `configs/experiments/under1m/hist14_local3_grouped_baseline_iqa.json` (**all new**); `tools/kitti_training_pipeline/notebook_config.py`; `3D_Lidar_Object_Detection_Notebook_standard.ipynb`; `tools/kitti_training_pipeline/common.py`; `tests/test_under1m_notebook_config.py`, `tests/test_under1m_notebook_controls.py` (**new**); config/run-name tests.
    - Change: retain these original no-context reference names; resolve full configs from master plus existing hybrid preset. Add hist14/depth/group/context/local-attention/fusion/detail controls and semantic run identity. Preserve notebook edits/cell IDs; no wholesale regeneration.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_under1m_notebook_config.py tests/test_under1m_notebook_controls.py tests/test_kitti_config_compatibility.py tests/test_run_name_and_logging.py tests/test_detection_config.py tests/test_backbone_feature_config.py`; configs rebuild identical contracts, the main preset selects focal/local-none, and existing incompatible runs cannot be overwritten.

48. **Create main and optional attention recipe files from the manifest** — R01, R12, R14–R16
    - Files: `configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa.json`, `configs/experiments/under1m/hist14_local3_focal_grouped_baseline_iqa.json`, `configs/experiments/under1m/hist14_local3_focal_eca_grouped_oga_iqa.json`, `configs/experiments/under1m/hist14_local3_focal_simam_grouped_oga_iqa.json` (**all new**); `tests/test_attention_presets.py` (**new**); notebook resolver.
    - Change: expand the documentation [experiment manifest](experiment_manifest.json) into complete resolved runtime configs; defaults are focal/local-none. Optional fusion/detail recipes are generated only after their implementation gates pass. Add explicit run-name tokens for context/local/fusion/detail; do not conflate ECA or SimAM with IQA.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_attention_presets.py tests/test_run_name_and_logging.py tests/test_under1m_notebook_config.py tests/test_under1m_notebook_controls.py`; stable semantic hashes, attention field validation, serialized rebuild, distinct config identity and expected parameter counts for every advertised recipe. New under1m notebook tests preserve the earlier consolidated/deleted notebook test files. Creating files does not launch all training runs.

49. **Run the complete CPU compatibility gate** — R01–R12, R14–R15
    - Files: `tests/test_grouped_pipeline.py`; hybrid integration tests; implementation files only if failures require changes.
    - Change: synthetic raw points/boxes → hybrid → hist14 → focal/selected local adapter → grouped heads/targets → valid criteria → optimizer step → checkpoint restore → BEV decode. Include empty scene/group and same-cell cross-group fixtures.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_backbone_feature_config.py tests/test_focal_context.py tests/test_light_attention.py tests/test_attention_presets.py tests/test_context_grouped_compatibility.py tests/test_bev_encoding_spec.py tests/test_hist14_encoding.py tests/test_hist14_backend.py tests/test_hist14_hybrid_integration.py tests/test_task_groups.py tests/test_detection_config.py tests/test_grouped_header.py tests/test_grouped_targets.py tests/test_grouped_losses.py tests/test_grouped_loss_state.py tests/test_grouped_loss_capabilities.py tests/test_grouped_iqa.py tests/test_grouped_qoga.py tests/test_grouped_metrics.py tests/test_grouped_training.py tests/test_grouped_checkpoint.py tests/test_grouped_decode.py tests/test_grouped_evaluation.py tests/test_grouped_export_contract.py tests/test_grouped_pipeline.py tests/test_lightweight_detector_budget.py tests/test_legacy_detector_contract.py`; then `$LIDAR_PYTHON -m pytest -q tests` once. Inspect skips; missing CUDA/Numba evidence is not a pass.

50. **Run GPU, AMP and DataLoader smoke gates** — R07, R08, R11, R14–R15
    - Files: `tools/benchmarks/smoke_detector.py` (**new**); `tests/test_grouped_cuda.py` and `tests/test_smoke_detector.py` (**new**); GPU report artifact.
    - Change: CUDA CLI runs actual focal/local-none, focal+ECA and focal+SimAM configs/valid strategy combinations on small divisible-by-16 inputs, checking device/optimizer/gradient/memory and FP32 reduction/modulation boundaries. Require FP32; FP16/BF16 only on supported hardware. Start workers 0, then configured workers on an IPC-capable runner.
    - Verify on GPU: `$LIDAR_PYTHON -m pytest -q tests/test_grouped_cuda.py`; proposed CLI: `$LIDAR_PYTHON tools/benchmarks/smoke_detector.py --config configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa.json --device cuda --precisions fp32 fp16 bf16 --steps 3 --output /tmp/grouped_gpu_smoke.json`. Repeat short smoke with each advertised local-attention preset; perform full-resolution one-frame inference and report unsupported precisions explicitly.

51. **Freeze benchmark protocol before long training** — R12, R13
    - Files: `tools/benchmarks/benchmark_protocol.py`, `tools/benchmarks/audit_kitti_assets.py`, `tools/benchmarks/benchmark_evidence.py`, audit/evidence/freeze regression tests, `tests/test_benchmark_protocol.py`, `benchmark_protocol.md`, `benchmark_protocol_3d.json`, `benchmark_protocol_bev.json` (**new**); experiment record/resolved config; `tools/kitti_training_pipeline/select_checkpoint.py` only if adding an explicit selection policy.
    - Change: record AP3D as the primary benchmark, with both R11 and R40 and an explicit reference APBEV mode per the user decision; 3D comparisons require the 3D-milestone tasks first. The recorder selects the requested protocol only, without enabling AP3D/R11 in the current runtime. Freeze split/seed/hybrid recipe/schedule/focal/local-attention/loss/quality/thresholds/checkpoint selection. Main long-run recipe is focal with local attention none; ECA and SimAM are explicit optional comparisons. Preserve historical minimum-loss selection unless an explicit tested AP option is added.
    - Verify: follow the implemented [task51 evidence plan](task51_implementation_plan.md) and [BEV-first Colab runbook](task51_colab_runbook.md); check actual file/config/checkpoint freshness, complete selection history, comparator split hashes, class/IoU/difficulty/protocol parity and definition of mAP averaging. No long training starts from an unresolved or mislabeled protocol. Batch17 records the candidate policy, but final freeze stays pending tasks53–58, data/GT-database audit and matched comparator evidence. Tasks52–54 may implement independent readiness/vertical work while that gate remains open.

52. **Record Milestone A readiness** — R01–R12, R14–R15
    - Files: `docs/plans/lightweight_lidar_backbone_2026/release_a_verification.md` (**new, implemented batch18**); parameter/GPU artifacts.
    - Change: record actual commands/results, attention/strategy capability matrix, skips, counts, collision/worker limits and protocol. Keep support-conditioned context, dense stages, depthwise stem and deployment in a separate backlog; record optional detail/fusion status independently.
    - Verify: `git diff --check`; inspect evidence against R01–R12 and R14–R15, plus R16 for any enabled optional detail/fusion variant. Reload reconstructs the same model/criterion/adapter state. Do not claim 3D readiness or improved AP from smoke tests.

53. **Add explicit 3D box mode and vertical prediction** — R13
    - Files: `detector/core/models/heads/cnn.py`; `detector/core/models/heads/grouped.py`; `detector/core/models/model.py`; `detector/core/detection_config.py`; `tests/test_vertical_head.py` (**new**); `tools/kitti_training_pipeline/common.py` for canonical mode propagation; temporary 3D guards in trainer/loss, BEV decoder/evaluator and ONNX wrapper/common deployment validator; [vertical interface](vertical_interface.md).
    - Change: `box_mode=bev|3d`, legacy default BEV; add separate `[z_bottom,log_height]` branch only in 3D mode. Initially permit baseline/OGA construction with labelled BEV IQA; reject incomplete strategy/export paths. Until task55, reject 3D in the training criterion factory and vertical tensors in direct BEV loss; until tasks56–57, reject vertical BEV decode and 3D local evaluation. Default BEV output/state behavior is preserved.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_vertical_head.py tests/test_grouped_header.py tests/test_legacy_detector_contract.py tests/test_detection_config.py`; BEV output/state parity and distinct per-group vertical branches.

54. **Assign vertical targets to the same box owner** — R06, R13
    - Files: `detector/core/datasets/utils_1/target_backend.py`; `detector/core/datasets/dataset.py`; `tests/test_vertical_targets.py` (**new**).
    - Change: fill z_bottom/log h using the same masks/owners/reserved peaks as BEV regression. Add explicit `fill_regression_targets_3d(..., backend=python|numba)` returning the four BEV maps plus vertical; preserve the existing four-array BEV APIs. The shared ownership kernel writes vertical in the same accepted cell; `data.box_mode` controls Dataset output. Reject non-finite/nonpositive 3D dimensions and invalid radii.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_vertical_targets.py tests/test_grouped_targets.py tests/test_small_object_assignment.py`; Python/Numba parity, known z/positive h, empty scenes and collision owner consistency.

55. **Add explicitly weighted vertical loss** — R07, R13
    - Files: `detector/core/losses/loss_fn.py`; `detector/core/losses/grouped.py`; `detector/core/detection_config.py`; `tools/kitti_training_pipeline/train.py` criterion factory; `tests/test_vertical_loss.py` (**new**).
    - Change: baseline/OGA 3D adapter adds masked FP32 Smooth L1 (beta1, mean across both channels of owned cells) for z_bottom/log h with explicit finite positive `vertical_loss_weight` before group aggregation. Select supervised values before arithmetic; empty masks retain a connected zero. Validate every group before adaptive-state updates; retain coefficient in existing strict objective identity. Existing OGA uncertainty tasks and BEV quality remain unchanged and documented.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_vertical_loss.py tests/test_grouped_losses.py tests/test_grouped_iqa.py tests/test_grouped_checkpoint.py`; vertical-only branch gradients, zero loss for exact targets, empty-mask connected zeros and objective identity on resume.

56. **Decode complete boxes and calibrated KITTI coordinates** — R09, R13
    - Files: `detector/postprocess.py`; `tools/kitti_training_pipeline/kitti_box_conversion.py` (**new**); `tools/kitti_training_pipeline/prepare_kitti.py` only for shared conversion reuse; `tests/test_kitti_3d_decode.py` (**new**).
    - Change: distinct 3D result `(x,y,z_bottom,l,w,h,yaw)`, keeping BEV rows unchanged. Convert actual bottom centers/headings through calibration; no hardcoded yaw offset. Project eight corners through `P2` to genuine image boxes with near-plane/clipping rules; document doubled-yaw/AOS limits.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_kitti_3d_decode.py tests/test_vertical_targets.py tests/test_grouped_decode.py`; GT→targets→decode→camera roundtrip with nontrivial transforms, correct dimensions and bottom/height conventions.

57. **Integrate a pinned KITTI reference evaluator with AP3D/APBEV modes and R11/R40** — R12, R13
    - Files: `tools/kitti_training_pipeline/evaluate_kitti_3d.py` (**new**); `tools/kitti_training_pipeline/compare_models.py`; `tools/kitti_training_pipeline/evaluate_kitti_bev.py` runner opt-in only; `tools/kitti_training_pipeline/kitti_reference/` (**new**, immutable sources/licenses/hash manifest); `tests/test_kitti_3d_evaluation.py` (**new**); evaluator version/dependency record.
    - Change: use the preferred OpenPCDet KITTI evaluator from the specification after checking API/license and pinning an immutable revision. Expose `--metric-mode 3d|bev`, reporting both R11 and R40 independently in each mode. Provision a compatible Numba CUDA reference environment; supply real projected detection image boxes. Verify actual checkpoint/Dataset/calibration/comparison paths and partial comparison reports. Use partitioned overlap for full splits; preserve the old local ROI BEV evaluator. Integration uses OpenPCDet revision `233f849829b6ac19afb8af8837a0246890908755`; dependency paths and usage are in `reference_evaluation.md`. Verify metric-specific ignored/neighbor/DontCare/difficulty behavior; full benchmark-domain evaluation and ROI diagnostics remain separate.
    - Verify in the compatible reference environment: `$LIDAR_PYTHON -m pytest -q tests/test_kitti_3d_evaluation.py`; perfect/disjoint/vertical-mismatch boxes, duplicates, ignored/neighbor classes, projected-box-height boundaries and metric-specific DontCare agree with reference. Wrapper and reference entry points give identical AP3D and APBEV for both R11 and R40 on saved predictions; verify the CLI mode switch and provenance. Record actual Python/Numba/CUDA versions and any separate interpreter path; skipped reference tests do not meet R13.

58. **Run 3D smoke gates and freeze final GPU recipe** — R01–R16 (R16 conditional)
    - Files: `configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa_3d.json`, `configs/experiments/under1m/hist14_local3_focal_eca_grouped_oga_iqa_3d.json` (**all new**); `tests/test_grouped_3d_pipeline.py` (**new**); smoke CLI; explicit 3D resolver/generator and profiler mode; release/experiment record and Colab runbook.
    - Change: resolve the main focal/local-none 3D preset and optional ECA counterpart, recount body/neck/heads/total, check advertised baseline/OGA 3D combinations, strict resume and reference evaluation. Complete task51 final protocol freeze for AP3D R11/R40 and the APBEV mode after data/reference/comparator gates; main projected detector with IQA/vertical is 883,941, excluding train-only criterion parameters.
    - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_vertical_head.py tests/test_vertical_targets.py tests/test_vertical_loss.py tests/test_kitti_3d_decode.py tests/test_kitti_3d_evaluation.py tests/test_grouped_3d_pipeline.py`; run CUDA smoke with the 3D preset and reference-evaluate saved predictions. Backbone plus neck stays <1M. Training/benchmark results are still required for AP/SOTA claims.

### Risks & mitigations

- Unsupported IQA/loss combinations: central capability validation. Exact Q-OGA+IQA is a future feature, not a bypass to make grouping appear compatible.
- Added gates/attention may regress accuracy or add reduction/elementwise cost: separate explicit presets, nonzero small residual initialization, full-resolution memory/precision checks and measured evidence. Do not stack ECA/SimAM or enable every optional feature by default.
- SimAM core versus wrapper accounting: zero core weights plus one learned beta; ECA k3 core plus beta totals four. Report actual instantiated counts, not the module's marketing label.
- Cross-group corruption: independent targets plus same-cell fixtures. Ped/Cyclist within-group collisions remain and are reported.
- Adaptive-state drift: separate registered criteria, normalized fixed weights, eval EMA checks and deterministic resume. Grouped objectives are not numerically identical to shared regression.
- Background-only Gaussian loss: existing focal normalization can yield large negative-only loss on full grids. Preserve semantics initially, monitor group scales/EMAs and verify finite AMP; normalization redesign is a separate experiment.
- Memory/compute growth: input14 uses 31,539,200 float32 bytes/frame versus 18,022,400 for rich8 at 800×704; the added stride-4 block and wider/two heads add compute. Measure, do not claim speed from parameter count. Legacy stem remains intentionally unchanged.
- Nested structures: recursive transfer, flat diagnostics, all-branch warmup, decoder checks and deferred deployment guards prevent silent misuse.
- RNG/resume limits: deterministic fixtures use workers 0; real augmentation with persistent workers may not replay bitwise. Record limits rather than overpromise.
- 3D scope: separate vertical branch/adapter preserves BEV formulas. Audit bottom-center convention and reference metric policies; accepting three tensor channels is not proof of 3D strategy support.
- Accuracy uncertainty: hist14/local3/grouped heads are hypotheses. At least a matched baseline is needed to claim improvement; multi-seed confirmation/submission supports stronger claims later.
- Dirty-tree safety: provenance and staged-diff inspection; preserve notebook cell IDs and pre-existing Q-OGA changes. No wholesale notebook regeneration or repository reset.

### Rollback plan

Keep defaults `legacy_single`, stage depths [2,4,2], historical encoding/stem/attention and BEV box mode; new context/local fields default to none, fusion width 24 and detail off. New presets alone enable the focal candidate; preserve the historical master config. Select the named no-context reference to isolate attention issues, or NumPy if processing fails. Use a matching config/checkpoint rather than silent non-strict loading.

Revert only the feature commit introducing a defect and preserve unrelated changes/artifacts. Do not use repository-wide reset/clean. Grouped/hist14 checkpoints require matching topology for resume; cross-topology initialization is explicit model-only warm-start. No push or deployment is part of this plan.
