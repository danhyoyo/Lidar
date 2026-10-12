# Implementation execution log

Implementation authorized: 2026-10-07. Branch: `research/mobilepixornext-under1m`.

Completed **57/58 tasks**: tasks1–50 and52–58. Task51 has executable asset/selection/evaluation evidence tooling; its actual Colab freeze remains pending. Earlier batches1–16 completed tasks1–48. hist14, hybrid integration, focal/ECA/SimAM, optional fusion/detail, grouped heads/targets/losses, CLI training and strict semantic/RNG resume have CPU verification. Explicit backbone-only warm-start keeps fresh training state. Grouped Gaussian and binary decoding now use local class mappings, independent regression/IQA, classwise NMS and one global detection cap. PyTorch evaluation and profiling validate checkpoint semantics and report group/quality/config/split provenance plus actual parameter counts. Nine complete BEV recipes and notebook controls are available; unsupported grouped/IQA ONNX/TRT contracts now fail explicitly. Batch17 adds assembled CPU and real CUDA FP32/AMP/worker gates; Batch18 documents BEV readiness and adds vertical heads/owner-consistent targets; Batch19 implements vertical loss, calibrated decode and reference AP3D/APBEV R11/R40 with actual reference CUDA parity. Batch20 verifies dedicated 3D CUDA training/AMP/workers/strict-resume/reference smoke and materializes main/ECA 3D presets. Only task51 external real-data/comparator gates remain; Inductor stays deferred. Baseline provenance is in [execution_baseline.json](execution_baseline.json). Batches 1–6 used `/tmp/lidar-fixes-env/bin/python`; that temporary environment is no longer available. Batches 7–20 use `/home/duyennh/miniconda3/envs/AI_env/bin/python` (Python 3.14.4, Torch 2.11.0) with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Batches19–20 reference/full gates use the isolated `/tmp/lidar-kitti-reference-env/bin/python` with numba-cuda0.30.4 and CUDA12.9 compiler/runtime libraries, inheriting the same Torch/NumPy/Numba. The training environment is unchanged. Python 3.10 verification remains unavailable on this machine.

## Task status

| Task | Description | Status |
| --- | --- | --- |
| 1 | Capture legacy behavior and working-tree provenance | completed |
| 2 | Implement the pure encoding schema | completed |
| 3 | Replace duplicated pipeline/model channel resolution | completed |
| 4 | Write NumPy hist14 reference encoding | completed |
| 5 | Connect hist14 dispatch and augmentation ordering | completed |
| 6 | Add explicit Numba encoding backend | completed |
| 7 | Benchmark encoding and check hybrid integration | completed |
| 8 | Parameterize stage depths with legacy key parity | completed |
| 9 | Audit the no-context 32-channel reference backbone | completed |
| 10 | Resolve focal/local-attention/fusion configuration | completed |
| 11 | Create focal pre-normalization and projection interface | completed |
| 12 | Generate sequential context levels and stable gates | completed |
| 13 | Complete modulation and near-identity residual | completed |
| 14 | Verify context numerical and state behavior | completed |
| 15 | Implement the minimal ECA core | completed |
| 16 | Implement SimAM with explicit numerical guards | completed |
| 17 | Wrap ECA/SimAM with learned scalar residuals | completed |
| 18 | Optionally widen SG-FPN final fusion | completed |
| 19 | Optionally add stem-to-neck detail fusion | completed |
| 20 | Wire the two attention hooks into the backbone and registry | completed |
| 21 | Profile attention variants and update the hard budget gate | completed |
| 22 | Resolve groups and stable class mappings | completed |
| 23 | Centralize configuration/capability validation | completed |
| 24 | Compose independent task heads | completed |
| 25 | Extract reusable single-group target construction | completed |
| 26 | Build independent Gaussian group targets | completed |
| 27 | Implement binary local target labels | completed |
| 28 | Verify target backends and DataLoader collation | completed |
| 29 | Make nested batch transfer work on the selected device | completed |
| 30 | Implement normalized grouped loss composition | completed |
| 31 | Integrate adaptive strategies without sharing state | completed |
| 32 | Check IQA routing and detach boundaries | completed |
| 33 | Characterize binary strategy support | completed |
| 34 | Broadcast Q-OGA exact curriculum and persist buffers | completed |
| 35 | Aggregate quality diagnostics by counts | completed |
| 36 | Wire criterion factory, optimizer and compile warmup | completed |
| 37 | Verify adapters across grouped losses and quality supervision | complete |
| 38 | Save architecture/objective identity and validate strict resume | complete |
| 39 | Verify adaptive resume and supported RNG restoration | complete |
| 40 | Add model-only warm-start when needed | complete |
| 41 | Separate candidate decode and final NMS without legacy drift | complete |
| 42 | Decode grouped Gaussian boxes and their own quality scores | complete |
| 43 | Decode supported binary groups correctly | complete |
| 44 | Merge candidates before classwise NMS and global limits | complete |
| 45 | Integrate PyTorch evaluation and parameter reports | complete |
| 46 | Guard deferred deployment against silent output loss | complete |
| 47 | Create complete presets and notebook controls | complete |
| 48 | Create main and optional attention recipe files from the manifest | complete |
| 49 | Run the complete CPU compatibility gate | completed |
| 50 | Run GPU, AMP and DataLoader smoke gates | completed |
| 51 | Freeze benchmark protocol before long training | audit/evidence/freeze implementation delivered; actual Colab freeze pending |
| 52 | Record Milestone A readiness | completed |
| 53 | Add explicit 3D box mode and vertical prediction | completed |
| 54 | Assign vertical targets to the same box owner | completed |
| 55 | Add explicitly weighted vertical loss | completed |
| 56 | Decode complete boxes and calibrated KITTI coordinates | completed |
| 57 | Integrate a pinned KITTI reference 3D evaluator | completed |
| 58 | Run 3D smoke gates and freeze final GPU recipe | completed; conditional benchmark freeze remains task51 |

## Batch 1 evidence

- Task 1: pre-refactor fixture and provenance captured; legacy/loss/IQA/small-object gate: 70 passed, 2 subtests passed.
- Task 2: RED: schema import absent (expected collection failure); GREEN: 37 schema tests passed.
- Task 3: RED: 5 intended integration failures before consumer changes (direct width, pipeline encoding/shape, standalone imports and run naming); GREEN: 88 tests passed in 3.53 s.
- Final combined batch gate plus real notebook configuration/structure checks: **263 passed, 2 subtests passed in 9.76 s**, no skips or warnings.
- `git diff --check`: passed. All 5 pre-existing modified files preserved byte-for-byte; all captured config/split hashes unchanged.
- Machine-readable results: [execution_batch_01_verification.json](execution_batch_01_verification.json).

No training or AP claim.

## Compatibility decisions

- The schema preserves rich minimum-channel coercion and extra-channel semantics, including rich8 with 9 channels (channel 8 stays zero), widths >=10 activating span/std, and padding beyond 12.
- Binary shapes follow actual legacy `voxelize` integer truncation. At floating-point roundoff edges, pipeline shapes now match rasterization instead of independently rounding. Ordinary KITTI shapes and legacy names remain unchanged.
- Geometry-free direct model construction keeps its explicit binary width. The pipeline injects data encoding and geometry into a deep copy of the model config, so model-side duplicates cannot override data channels.
- Schema import does not import Torch/NumPy. Lazy pipeline import also works with only the pipeline directory on `sys.path` from another working directory.

## Next checkpoint

Current experiment priority, confirmed after batch20: **BEV development/ablations first**, keeping the notebook's main BEV preset and `CONFIG_OVERRIDE=None`. Complete 3D configs remain an explicit final-results option. Train or fine-tune the vertical branches before evaluating AP3D; use matched 3D heads/objectives for final baseline/candidate comparisons. This policy changes experiment order, not completed implementation status.

Task51 actual benchmark freeze: run the delivered [BEV-first audit/evidence workflow](task51_colab_runbook.md) on actual Colab assets and a complete trained no-context baseline, then consume matched records. External comparator protocols/adapters may be added separately. The user confirms all data/checkpoints are on Colab; absent local paths are not evidence of missing Colab data. The [Colab runbook](colab_3d_runbook.md) provides concrete audit, runtime precision and reference evaluation steps. The complete 3D main/ECA recipes and local synthetic functional gates are verified in [Milestone B](release_b_verification.md). Both protocol records retain `long_training_allowed=false` until external gates are satisfied. TensorRT/Jetson and Inductor remain deferred. No long training or real KITTI AP/SOTA result is claimed. Batch20 is ready for feedback under the invoked executing-plans skill.

## Batch 2 evidence

- Task 4 RED: missing reference backend assertion failed. GREEN: 40 analytic/reference tests passed in 0.91 s.
- Task 5 RED: public dispatch rejected hist14. Final combined encoder/rich/legacy gate: 79 passed in 6.83 s, no warnings (also covers task 6).
- Task 6 RED: explicit numba dispatch raised NotImplementedError; GREEN: both kernels pass all-channel parity, boundary and dependency-unavailable tests. Final combined task gate: 79 passed in 6.83 s.
- Additional RED/GREEN: preserved frozen validation corners without Torch–NumPy array-wrap warnings; rejected out-of-grid point aliasing in both kernels.
- Final expanded compatibility gate: **349 passed, 2 subtests passed in 13.35 s**, no skips or warnings. It includes batch-1 gates, hist14, rich encodings and regression tests.
- Full KITTI shape audit: `[800,704,14]`, contiguous float32, 31,539,200 bytes; 1,000 synthetic points / 999 occupied cells. NumPy/Numba max absolute difference was 0.0 on this fixture. Actual Numba nopython compilation verified with `fastmath=False`.
- `git diff --check` and new-source whitespace checks passed. Original 5 dirty files/config/split hashes and frozen batch-1 fixture remain unchanged.
- Evidence: [verification report](execution_batch_02_verification.json), [encoding shape audit](execution_batch_02_encoding_shape.json).

### Batch 2 implementation details

- `bev_backend.py` accumulates in float64 and returns contiguous float32 YXC. Height histogram, extrema/means/span/std, intensity, log density, occupancy and signed point centroids follow hist14 v1. Backend implementation version is reported separately from the semantic schema.
- Encoding follows existing augmentation in `Dataset.__getitem__`; no BEV feature cache is added. Public dispatch validates the shared schema. Rich/binary rasterization formulas are untouched.
- hist14 target dimensions use resolved integer grid sizes, avoiding independent float truncation. The legacy target-size branch remains unchanged.
- Validation corner conversion now starts from a NumPy view of tensor label data while retaining dtype. A pre-change frozen fixture confirms exact equality for three yaw cases; this removes an existing NumPy 2 array-wrap warning.
- Explicit Numba selection lazily imports the dependency and never silently falls back, including empty clouds. Bounds guards reject points beyond a resolved grid rather than aliasing a neighboring row or writing outside an accumulator.
- Actual hybrid-GT integration/encoding benchmarking are task 7; GPU/AMP and mAP evidence remain later gates. No speedup or accuracy claim is made from parity.

## Batch 3 evidence

- Task 7: RED missing benchmark CLI; GREEN hybrid/hist14 and all hybrid tests: 42 passed in 5.40 s outside sandbox, including two-worker IPC. Initial sandbox attempt: 41 passed, one IPC permission failure.
- Benchmark CLI completed: 10,000 synthetic points, full KITTI grid, 3 warmups/20 iterations, separate first call and steady-state statistics.
- Task 8 RED: 12 intended constructor/registry failures; GREEN: 39 passed in 7.66 s, one existing TorchScript-on-Python-3.14 warning. Default weight/key/output parity and finite candidate backward verified.
- Task 9 RED: profiler missing; GREEN: budget/backbone gate 28 passed in 10.27 s, one existing Python 3.14 TorchScript warning. Full-resolution meta audit gives backbone `[1,32,200,176]`; candidate small-grid numerical backward and default parity also pass.
- Final expanded CPU compatibility gate: **419 passed, 2 subtests passed in 22.07 s**, one deselected IPC test already passed in the complete task-7 gate outside sandbox, no skips. One existing `torch.jit.script` Python 3.14 compatibility warning; its numerical test passes.
- Backbone including neck: **635,408** (body 604,864 / neck 30,544). The actual profiling single head with IQA has 93,130 parameters, total detector 728,538; OGA criterion has 6 train-only parameters. Grouped heads remain pending.
- Full-resolution Conv2d-only MAC estimate: 10.5741 GMAC for this single-head reference; excludes attention matmuls, normalization, activations, interpolation, elementwise and preprocessing. This is not measured latency.
- Final synthetic CPU benchmark at full KITTI geometry, 10,000 points: hist14 NumPy mean 48.262 ms; Numba mean 32.267 ms; rich8 mean 5.971 ms. These fixture timings do not establish real-scene/GPU throughput; hist14 processing remains more expensive than rich8 on this fixture. Initialization includes JIT compilation or cache loading plus the first rasterization, separately from warmup/steady-state.
- Evidence: [verification](execution_batch_03_verification.json), [encoding benchmark](execution_batch_03_encoding_benchmark.json), [reference profiler](execution_batch_03_reference_profile.json).
- `git diff --check`, new-file whitespace and original-edit/config/split/legacy-fixture preservation checks passed. No training recipe or master config was changed.

### Batch 3 implementation details

- New benchmark CLI supports synthetic or KITTI binary clouds, selectable encodings, separate first-call/warmup/steady-state samples, mean/p50/p95, environment and semantic identity metadata.
- Real hybrid database paste and deterministic global scaling are verified across NumPy/Numba encoding and Python/Numba targets. Validation skips sampling; a changed next transform causes re-rasterization.
- Stage depths default to [2,4,2]; [3,4,2] adds only `stage2.2` keys, +14,112 parameters. Defaults retain initialization order, weights, state keys and predictions.
- The profiler constructs real model/criterion modules and saves resolved configuration, parameter breakdown, a strict backbone+neck budget gate, full-resolution shapes and clearly scoped Conv2d MAC estimates. The default reference explicitly uses the current single head; it is not the future grouped training recipe.

## Batch 4 evidence

- Task 10 RED: 64 intended failures because the pure resolver did not exist. Final exact task gate: **66 passed in 1.34 s**. The resolver imports no Torch/NumPy/Numba, validates strict numerical settings, rejects conflicting/new inactive module fields, preserves historical defaults, and records immutable JSON-compatible semantic identity.
- Additional RED/GREEN: the existing `e0_conv` preset retains its saved inactive LiteMLA scales `[5]` in legacy resolved identity; candidate topology rejects these inactive scales. An unrepresentable integer scalar raises a configuration ValueError instead of leaking OverflowError.
- Task 11 RED: 29 intended missing-module failures. Initial GREEN: 29 passed. Final exact `-k projection` gate: **30 passed, 13 deselected in 1.80 s** (the selector also includes the later projection/pooling isolation fixture). Bias-free C→2D+L+1 projection and Q/S/G split preserve shapes, CPU explicit FP32/FP16/BF16 dtype and device. Ordinary BN single-element training constraints are explicit.
- Task 12 RED: 14 intended missing-level/gate failures. Final exact task gate: **14 passed, 29 deselected in 1.78 s**. Known weights and impulse fixtures verify sequential dilation 1/2/3, cumulative receptive fields 3×3/7×7/13×13, depthwise channel isolation, FP32 final-level pooling, stable channel softmax, global broadcast and absence of occupancy masking. Eval batch-item isolation and finite nonzero component gradients pass.
- Final expanded regression gate: **487 passed, 2 subtests passed in 21.79 s**, no skips/deselections, one existing TorchScript/Python 3.14 warning. It covers the backbone/reference budget, model registry, frozen legacy outputs/targets/losses, encoders, notebook/config/run naming, IQA and loss strategies. Hybrid IPC was not rerun because this batch changes no dataset/loader code; it passed the preceding required gate outside sandbox.
- Full KITTI stride-8 standalone preparation audit: input `[1,96,100,88]`; query and fused context `[1,64,100,88]`; four gate weights `[1,4,100,88]`; all outputs finite, maximum gate-sum error 2.3842e-7. Partial module count is **14,592**, not the complete focal count; the implemented reference backbone remains **635,408**.
- Five original dirty files, all captured configs/splits, the immutable numeric legacy fixture and all recorded batch-3 implementation sources are unchanged. New-source whitespace and `git diff --check` pass.
- Evidence: [verification report](execution_batch_04_verification.json), [standalone shape audit](execution_batch_04_context_shape.json).

### Batch 4 implementation boundary

`backbone_config.py` resolves feature settings without model construction. `FocalContext` exposes projection and context preparation; its residual `forward` explicitly raises NotImplementedError until task 13. Modulation/output projection/output BN/LayerScale, complete parameter accounting, state/AMP validation, ECA/SimAM, hook integration and runtime presets remain subsequent tasks. No active detector parameters, current predictions, training configs or loss routing changed. No training, AP or latency improvement claim.

## Batch 5 evidence

- Task 13 RED: 5 intended failures from missing residual forward/projections/gamma and partial parameter count; one existing component-gradient fixture passed. Initial GREEN: 6 passed. Final exact task gate after numerical extensions: **9 passed, 55 deselected in 1.41 s**. Full focal has **25,120 learned parameters**, including both affine BN and channel LayerScale; all convolutions are bias-free. Gamma-zero gives exact finite identity and zero branch-weight gradients; default gamma=1e-3 gives finite nonzero gradients to every branch parameter in train/eval.
- Task 14 verification: **64 passed in 1.39 s**. Zero/constant/sparse/random maps, asymmetric grids, no inplace input changes, ordinary BN single-element constraints, deterministic eval and serialized state reload all pass. CPU explicit FP32/FP16/BF16 and CPU autocast FP16/BF16 are supported by these fixtures. Hooked output-projection input confirms the modulation product remains FP32 for FP32 weights under autocast; explicit low-precision weights require a projection-boundary cast. No production fixes were required by the additional numerical tests.
- Task 15 RED: 25 intended missing-module failures. GREEN exact gate: **25 passed in 1.59 s**. ECA core has exactly k learned weights (k3=3, k5=5), zero initialization, initial sigmoid gate 0.5, analytic neighboring-channel correlation, spatial broadcast, nonzero kernel gradients, FP32 pooling/correlation/sigmoid, input dtype restoration and eval batch isolation. Compatible state reload and meta-device full stride-4 shape audit pass.
- Final expanded regression gate: **533 passed, 2 subtests passed in 21.74 s**, no skips/deselections. One existing TorchScript/Python 3.14 warning; its numerical test passes. Frozen legacy predictions/targets/losses, reference backbone budget, registry, encoding, notebook/config/run naming, IQA and loss-strategy gates remain green. Dataset/hybrid IPC was not rerun because no data/loader code changed; its earlier complete gate remains recorded.
- Full KITTI stride-8 standalone audit: `[1,96,100,88]` input/output; output and input gradients finite; all parameter gradients finite/nonzero. ECA standalone stride-4 input/output `[1,48,200,176]`, initial output exactly half input. Neither module is yet wired into the detector.
- Actual no-context backbone including neck remains **635,408**. Adding the now-verified standalone 25,120 count gives **660,528 arithmetically**; the integrated focal budget is still pending task 21. ECA's beta wrapper and its 4-parameter adapter total remain task 17.
- Five original dirty files, captured configs/splits, batch-3 implementation sources, pure feature resolver/tests and frozen legacy fixture are unchanged. New-source whitespace and `git diff --check` pass.
- Evidence: [verification](execution_batch_05_verification.json), [module audit](execution_batch_05_module_audit.json).

### Batch 5 implementation boundary

Focal v1 now runs complete standalone forward, with output projection/BN and learned per-channel gamma. ECA is a core only: its initial output scales features by 0.5, so it must be wrapped by the planned residual adapter before backbone integration. SimAM, residual beta, hooks, presets, grouped-head/loss routing, full detector resume, GPU/AMP memory checks and accuracy evidence remain subsequent tasks. No active detector parameters or runtime training configs changed; no AP or latency claim.

## Batch 6 evidence

- Task 16 RED: 22 intended missing-SimAMCore failures. GREEN exact initial gate: **22 passed, 25 deselected in 1.31 s**. Analytic constant/impulse fixtures verify FP32 mean/deviation/unbiased variance/energy, sigmoid(0.5) on constants and H×W=1, finite-positive lambda validation, per-batch/channel isolation and zero core parameters. FP32/FP16/BF16 output dtype preservation, CPU autocast, sparse gradients and meta shapes pass.
- Task 17 RED: 33 intended missing-factory/adapter failures. GREEN exact gate: **80 passed in 1.37 s**. Beta is a registered scalar initialized to 1e-3, remains unclamped, and contributes to parameter counts: ECA k3=4 / k5=6, SimAM=1, none=0. Beta-zero gives exact identity and zero core gradients; nonzero beta gives finite nonzero beta/kernel/input gradients. Explicit low-precision residuals accumulate in FP32; input dtype and state reload are preserved. Shared resolution rejects inactive/conflicting options.
- Task 18 RED: 20 intended missing-constructor-option failures. GREEN exact gate: **20 passed in 1.75 s**. Standard SG-FPN fusion32 changes only stride-4 lateral/projection/gate/output input widths, retaining upper 48-channel laterals and the existing fusion formula. Fusion24 retains exact default weights/state keys/predictions. Fusion32 has +6,240 parameters with output32/gates, +6,160 without gates, +5,088 with output16/gates. Finite backward and asymmetric shapes pass; RC neck selection and grids not divisible by 16 are rejected for fusion32.
- Final expanded regression gate: **608 passed, 2 subtests passed in 20.09 s**, no skips/deselections, one existing TorchScript/Python 3.14 warning. The additional RC-SGFPN regression gate: **14 passed in 7.58 s**, no warnings. This includes RC neck/default detector behavior and deployment equivalence of existing RC modules.
- Full-grid audit: standalone ECA/SimAM adapters preserve `[1,48,200,176]`, with finite outputs and finite nonzero gradients to their learned parameters. Actual no-context backbone including neck is **635,408** with fusion24 and **641,648** with fusion32; both produce `[1,32,200,176]` from `[1,14,800,704]` on meta. The **666,768** focal+fusion32 count is arithmetic only until hook integration/task21.
- Five original dirty files, all captured configs/splits, frozen numeric fixture, focal module/tests, pure resolver/tests and unrelated batch-3 sources are unchanged. New-source whitespace and `git diff --check` pass.
- Evidence: [verification](execution_batch_06_verification.json), [module audit](execution_batch_06_module_audit.json).

### Batch 6 implementation boundary and recipe sequencing

SimAM and the scalar ECA/SimAM residual adapters are standalone. Fusion32 is supported through the backbone constructor; registry wiring is task20, so no runtime recipe has been changed. Task18's complete focal/grouped fusion32 preset is assigned to task48, after focal hooks and grouped model/loss/decode capabilities exist; the implementation plan records this dependency to avoid publishing an incomplete runnable training config. The main recipe still selects fusion24 and local attention none. Detail path, hook integration, actual integrated attention budgets, grouped targets/losses, GPU checks and AP evidence remain pending.

## Batch 7 evidence

- Environment: the prior temporary Python environment disappeared; the installed AI_env provides Python 3.14.4, Torch 2.11.0, NumPy 2.4.6, Numba 0.67.0 and pytest 9.1.1. Python 3.10 is unavailable. Disabled unrelated pytest plugin autoload after a ROS plugin failed to import `osrf_pycommon`. No dependency installation or environment modification was needed.
- The workspace arrived with pre-existing test consolidation/deletions. Those edits were preserved; current test counts therefore differ from historical batch counts. Preservation was checked against a fresh start-of-turn hash snapshot, including the five original dirty files, captured configs/splits and immutable numeric fixture.
- Task 19 RED: six missing-detail constructor failures, with 11 existing fusion fixtures passing. GREEN exact `-k detail` gate: **17 passed in 1.97 s** (the filename selector includes fusion fixtures). Adds exactly 1,192 parameters at F24 / 1,472 at F32. Same-seed shared weights, gamma-zero eval identity, asymmetric alignment, finite nonzero branch/scale gradients and invalid RC selections pass. No silent resize is used.
- Task 20 RED: 14 intended missing-hook/registry/notebook capability failures. Initial GREEN integration/legacy/context/local/neck/RC gate: **188 passed in 13.45 s**, 44 existing TorchScript warnings. Routing fixtures verify that local-refined features feed both down3 and the stride-4 lateral, while focal-refined features feed down4 and the stride-8 lateral. Branch gradients are finite/nonzero; disabled modules add no persistent state. Constructor and registry share validation, and notebook validation rejects conflicts before legacy sanitation.
- Additional run-name RED/GREEN: seven incompatible new architectures initially shared one run name; new tags include resolved feature identity, depths and output width. Existing legacy names remain unchanged. Integration/run-name gate: **28 passed in 1.79 s**.
- Task 21 RED: nine missing profiling-variant/CLI failures, five existing profiler tests passing. GREEN exact gate: **14 passed in 5.29 s**. Actual full KITTI meta output `[1,32,200,176]` and strict budgets verified for reference, focal, focal+ECA, focal+SimAM, wider fusion, detail and combined optional variant. Core/residual scale/detail gamma are counted separately; detail parameters are included in the neck.
- Full current repository suite initially had two DataLoader IPC permission failures in the sandbox. Required rerun outside sandbox: **548 passed, 17 subtests passed in 37.38 s**, no skips/deselections. Includes both worker tests. 57 existing TorchScript script/trace/trace_method deprecation warnings on Python 3.14; their numerical tests pass. Dedicated CUDA readiness for the new hooks and Python 3.10 remain later/unavailable checks.

| Integrated variant | Actual backbone + neck | Current single BEV IQA head | Train-only OGA criterion |
| --- | ---: | ---: | ---: |
| Reference | 635,408 | 93,130 | 6 |
| Focal | 660,528 | 93,130 | 6 |
| Focal + ECA | 660,532 | 93,130 | 6 |
| Focal + SimAM | 660,529 | 93,130 | 6 |
| Focal + fusion32 | 666,768 | 93,130 | 6 |
| Focal + detail | 661,720 | 93,130 | 6 |
| Focal + ECA + fusion32 + detail | 668,244 | 93,130 | 6 |

All rows pass the strict <1M backbone budget. These are actual instantiated parameter/meta-shape audits, not trained AP or measured latency. MACs cover Conv2d only and exclude Conv1d attention, reductions, interpolation and other operations. The profiler rejects grouped mode until grouped construction exists. Raw generated reports are kept in the ignored [batch-7 artifact directory](../../../artifacts/kitti/backbone_audit/batch07/verification.json), including [all variant profiles](../../../artifacts/kitti/backbone_audit/batch07/variant_profiles.json).

### Batch 7 implementation boundary

Focal, optional local attention and detail/fusion options now run through the actual backbone/registry. New modules are shared by the existing head; they are not duplicated per task. Default legacy weights/keys/predictions remain compatible. Complete Car/Pedestrian-Cyclist heads, group targets/losses/quality routing, strict full-detector resume, grouped recipes and KITTI 3D evaluation are still pending. Detail and fusion grouped preset files remain assigned to task48 after their dependencies exist. No training, mAP or speedup claim.


## Batch 8 evidence

Tasks 22–24 are complete. This delivers pure class/group contracts and independent prediction heads, not an enabled grouped training pipeline.

### Resolved class and capability contracts

- `task_groups.py` resolves one exact partition of active classes against a dense zero-based global ID map. Group and class list order are preserved, including reordered and non-default classes. Frozen metadata provides Gaussian channel/global-ID and binary foreground/global-ID conversions; binary background maps to `None`. Duplicate, unknown, omitted or empty classes/groups, malformed IDs, unsupported per-group fields and unsafe/reserved module names fail clearly.
- `detection_config.py` is independent of Torch/NumPy/Numba. It validates head mode, stride 4, classification, width/depth types, class counts, BEV mode, head/loss IQA agreement, strategy capabilities, group weights and exact-Q-OGA curriculum restrictions. Explicit null class counts/maps are invalid. Positive finite weights default to one per group and normalize over all configured groups; their ordering follows resolved groups.
- The pipeline `build_model` and notebook validator share this full contract. PyTorch evaluator/export callers using `build_model` cannot bypass these checks. Model-only construction validates prediction topology and explicit resolved groups; loss compatibility requires the full configuration and is checked by the pipeline before construction. Registered custom backbone builders retain their own option validation.
- Gaussian grouped strategies follow the planned matrix. Binary grouped baseline/OGA/UWAG are permitted by the configuration contract; GW-QAL and Q-OGA binary grouped requests fail pending task33 parity evidence. Legacy MGIoU/binary behavior is not rewritten. Separate IQA remains restricted to baseline/OGA. Actual grouped criterion integration is still pending.
- Grouped run names now include class/group ordering, normalized weights, classification, box mode, width and quality settings in a stable digest; equivalent default/equal weights share identity. Historical legacy run names remain unchanged. Complete controls/presets and strict checkpoint identity remain task47/task38 work.

### Independent prediction heads

- `GroupedHeader` owns one existing `Header` per resolved group in a `ModuleDict`. Car and Pedestrian/Cyclist share backbone features, while classification, offset, size, yaw, BatchNorm and optional IQA parameters and output storage remain independent.
- `CustomModel` accepts explicit resolved `task_groups`. The pipeline passes metadata from the authoritative data contract into construction without mutating its input config. Gaussian group widths are 1/2; binary widths are 2/3, with group-local background at index 0.
- Grouped predictions are `{"groups": {"car": {...}, "ped_cyc": {...}}}`. Disabled IQA is absent. Legacy construction keeps `model.header`, its flat output, state keys and seeded weights; it adds no grouped module state.
- A Car-only backward pass gives finite gradients to shared input features and all active Car branches, including IQA; Pedestrian/Cyclist head parameters receive no gradient. Grouped head state, including BN running buffers and IQA, roundtrips exactly. Full model strict resume identity is deferred to task38.

| Focal candidate | Backbone including neck | Independent BEV heads | Model total, criterion excluded |
| --- | ---: | ---: | ---: |
| Gaussian, IQA off | 660,528 | 148,975 | 809,503 |
| Gaussian, IQA on | 660,528 | 186,161 | 846,689 |
| Binary, IQA off | 660,528 | 149,041 | 809,569 |
| Binary, IQA on | 660,528 | 186,227 | 846,755 |

These are actual constructed parameter counts and meta-device shapes at `[1,14,800,704]`, producing output grids `[200,176]`. CPU numerical/gradient fixtures use small asymmetric grids. The user budget remains strictly <1M for backbone including neck. Grouped criterion parameters are excluded because that module does not yet exist. No accuracy, latency or GPU/AMP readiness claim.

### Verification and preservation

- Tests were written and run failing before the corresponding mapping, validator and head implementations. Final focused checks: **227 passed, 7 subtests passed in 8.09 s**.
- Final complete repository suite outside the sandbox: **652 passed, 17 subtests passed in 36.70 s**, no skips or deselections. Both worker/DataLoader IPC tests ran. The sandbox restricts the sockets needed by these existing tests, so the full run used approved execution outside it.
- The first full run exposed two evaluator fixtures enabling IQA while omitting its supervision flag. Only their synthetic configuration now declares `loss.name=baseline` and `loss.use_iou=use_iou`; the original peak-mode/quality assertions remain intact. Regression checks explicitly reject mismatched IQA flags.
- 57 existing TorchScript deprecation warnings remain on Python 3.14; numerical checks pass. Python 3.10 and dedicated CUDA/AMP gates for new grouped/context features remain pending.
- All 23 protected source/notebook/config/split/legacy-fixture hashes match this batch's baseline. Existing test deletions/consolidations are preserved. No master config, hybrid recipe, split, checkpoint, training run or pre-existing loss/notebook/train edits were changed. `git diff --check` passes.
- Raw reports remain ignored under [batch08](../../../artifacts/kitti/backbone_audit/batch08/verification.json): [head/shape audit](../../../artifacts/kitti/backbone_audit/batch08/grouped_head_audit.json) and [final test log](../../../artifacts/kitti/backbone_audit/batch08/full_suite_final.log). The earlier profiler intentionally remains single-head-only until task45 adds grouped reporting.

### Batch 8 implementation boundary

Grouped prediction construction is ready for target integration. Dataset group targets, grouped loss/adaptive state and IQA supervision, nested training transfer/warmup, grouped decoder/evaluation/export guards and complete recipes remain later tasks. No grouped training preset has been activated. Next is tasks25–27, including the cross-group same-cell regression fixture required by the specification.


## Batch 9 evidence

Tasks 25–27 are complete. The dataset now creates independent Gaussian/binary targets for the resolved Car and Pedestrian/Cyclist groups; no grouped training recipe has been activated.

### Stateless single-group construction and legacy parity

- `Dataset.get_single_group_label` takes an explicit class count, optional global-to-local mapping and optional classification encoding. It selects only the group's source boxes before assignment, clones classification rows for remapping and allocates fresh classification/regression maps. It never temporarily rewrites dataset class counts, encoding, config, cached source boxes or globally stored labels.
- The flat `get_label` path delegates to this builder with the historical settings. Existing Gaussian/binary targets, asymmetric axis conventions, nearest-center ownership, reserved small-object peaks, canonical ties, legacy assignment and frozen model/loss/state fixtures continue to pass.
- Original global class IDs remain on the filtered regression rows. Only classification IDs change, so reversing local channel order cannot change the box selected at an exact regression tie. Backend kernels and geometry arithmetic are unchanged.

### Independent Gaussian group targets

- The dataset consumes the pure group resolver from `data.head_groups`, or accepts explicit resolved `task_groups` appended to its existing constructor. Supplied metadata must agree with the authoritative `data.kitti.objects` and any configured definitions; all manifest sources must share that class mapping. Grouped class counts and supported encodings are validated before augmentor/cache creation.
- Raw boxes are augmented and filtered once in global class space through the existing item path. Each group then builds targets from only its own boxes. Grouped boxes with nonfinite, fractional or out-of-range global IDs fail clearly.
- The same output cell `[y=8,x=2]` retains Car center `[2.15,0.05]` and Pedestrian center `[2.35,0.25]` independently, with both group masks set and distinct offset/size/yaw storage and values. Changing a Pedestrian box cannot change Car targets.
- Empty groups have negative-only classification and zero regression maps. Within `ped_cyc`, two overlapping classes still have one regression owner per cell; exact ties remain canonical and independent of source label order. This is the existing within-group limitation, not per-class regression.
- External maps remain Y,X: Gaussian class targets `[C_group,16,12]`, regression `[2,16,12]`, mask `[16,12]` for the asymmetric anisotropic fixture. Voxel `[14,64,48]` remains top level. `val` keeps global `cls_list`, corners, points and dataset type at the top level; `validation` stays tensor-only, and test mode keeps its existing input/metadata-only contract.

### Binary local foreground labels

- Foreground IDs are remapped locally before the historical binary writer adds one. Car has labels 0/1; Pedestrian/Cyclist have 0/1/2. Background remains zero and class maps are int64 `[Y,X]`. Reversed class order and non-default Bus/Person groups map through the same metadata resolver.
- Explicit local mappings must cover exactly `0..C_group-1`; duplicate, missing, non-integer/boolean, negative or incorrectly shifted IDs are rejected. Grouped encoding case normalization matches the central configuration validator; historical legacy casing behavior is unchanged.
- The shared builder implements the encoding-neutral filtering/remapping used by both Gaussian and binary paths. Task27 adds binary partition/background/reordered-class/non-default fixtures and strict mapping validation rather than duplicating assignment code.
- An actual within-group binary overlap fixture documents an inherited limitation: classification painting is source-order-dependent (last box wins), while nearest-center regression ownership is canonical. They can refer to different classes. This behavior remains unchanged for legacy parity; task33 now explicitly audits its implications for loss/quality supervision before any binary accuracy claim. The main recipe remains Gaussian.

### Hybrid ordering, verification and preservation

- The existing real hybrid integration test now covers flat and grouped targets across NumPy/Numba hist14 encoding and Python/Numba regression assignment. Pasted points and boxes undergo the global scale before encoding/targets; the sampled Car target uses transformed geometry, the unused group is negative-only, and a later access re-encodes the current augmentation. Original point files remain intact.
- Tests were written/run failing before extraction, grouped construction, mapping validation and case normalization. The new target test module has **49 cases**; grouped hybrid coverage adds four cases while preserving all previous flat cases.
- Final focused verification: **207 passed in 3.54 s**. Final full suite outside the sandbox: **705 passed, 17 subtests passed in 36.51 s**, no skips/deselections. Both worker IPC tests ran with approved execution outside the sandbox. Full grouped backend matrix/worker collation remains task28.
- 57 existing TorchScript deprecation warnings remain under Python 3.14.4/Torch 2.11.0; numerical checks pass. Python 3.10 verification is unavailable. These are CPU target/numerical tests, not a new-feature GPU training or AP result.
- All 23 protected hashes match this batch's baseline, including the original notebook/loss/train edits, master/experiment configs, splits and frozen legacy fixture. Existing deleted/consolidated tests are preserved. No backbone/head parameters, master config, hybrid recipe, split, checkpoint or training run changed. `git diff --check` passes.
- Generated [verification](../../../artifacts/kitti/backbone_audit/batch09/verification.json), [target audit](../../../artifacts/kitti/backbone_audit/batch09/target_audit.json) and [full test log](../../../artifacts/kitti/backbone_audit/batch09/full_suite_final.log) are kept in ignored artifacts. The audit records four Gaussian/binary × Python/Numba cases with geometry, shapes, dtypes, mask counts and both decoded centers in the shared cell.

### Batch 9 implementation boundary

Grouped prediction heads and corresponding dataset targets now exist. Nested collation/device transfer, normalized grouped criteria, independent adaptive/IQA state, optimizer/warmup wiring, strict resume, grouped decoding/evaluation and complete runtime recipes remain later tasks. Next is tasks28–30. There has been no training, mAP/latency measurement or 3D/SOTA claim.

## Batch 10 evidence

Tasks 28–30 are complete. Grouped target collation, recursive device transfer and fixed-weight objective composition are implemented and tested; the main trainer still uses its historical criterion until task36.

### Backend, collation and worker contracts

- Python/Numba target maps match bitwise for Gaussian/binary, nearest-center/legacy assignment, normal/reversed group and class order, empty inputs, boundary boxes, exact overlaps and seeded random boxes. No backend implementation changes were needed.
- Real dataset collation preserves top-level voxels and independent nested group maps. Gaussian labels become `[B,C_group,H,W]`; binary labels remain int64 `[B,H,W]`. Empty groups remain negative-only with zero regression. Flat legacy batches and validation/test metadata retain their existing structure.
- Four real two-worker spawn tests cover Gaussian/binary × Python/Numba. Dataset pickling, parent Numba warmup and every collated tensor leaf are checked against the parent reference. These worker tests require approved execution outside the socket-restricted sandbox.

### Recursive transfer and grouped objective

- `move_tensor_batch` transfers tensors recursively through mappings, lists, tuples and namedtuples with `non_blocking=True`. Ordered mappings, non-tensor metadata and input containers are preserved; actual meta-device transfers prove that nested leaves move, and CPU/autograd checks remain green. Dedicated CUDA transfer is still pending.
- `build_loss_function` returns the existing `LossFunction` for legacy mode or a new `GroupedLossFunction` with independent registered criteria. Group weights use the shared pure validator, default to 0.5/0.5 for two groups and remain fixed when one group is empty. Explicit 1:3 weights normalize to 0.25/0.75. Invalid weights and unsupported combinations fail clearly.
- Gaussian/binary baseline tests with IQA on/off compare total loss and every prediction gradient exactly against the explicit weighted sum of the existing criteria. Single-group baseline/OGA cases preserve legacy loss/gradient parity after state alignment. Empty groups retain negative classification supervision and finite graph-connected zero box/IQA terms.
- The result is a flat scalar dictionary: only total `loss` carries gradients. All existing per-group diagnostics are detached under `group/<name>/<metric>`; core loss components are aggregated by objective weights. Quality means and counts remain namespaced until task35 defines count-based aggregation.
- Every group's structural input contract is checked before invoking adaptive strategies. Independent adaptive update behavior, complete IQA quality/detach semantics, binary strategy capability evidence and exact Q-OGA epoch broadcasting are tasks31–34; this batch does not advertise them as fully verified.

### Verification and preservation

- Tests were written and run failing before the recursive transfer and grouped loss implementations. New coverage: 19 pipeline, 4 transfer and 32 grouped-loss cases, totaling 55 additions.
- Focused combined gate: **174 passed, 4 deselected, 6 subtests passed in 1.65 s**. The four excluded worker cases ran in the separate pipeline gate outside the sandbox: **19 passed in 6.55 s**.
- Final full repository suite outside the sandbox: **760 passed, 17 subtests passed in 44.79 s**, exit 0, no skips/deselections. All worker cases ran. The 57 existing TorchScript/Python 3.14 deprecation warnings remain; numerical checks pass. Python 3.10 and dedicated CUDA/AMP gates remain unavailable/pending.
- All 22 protected hashes outside the authorized trainer change match the batch baseline. In `train.py`, only `move_tensor_batch` changed; the entire file outside that function matches its captured start-of-batch source. The original `LossFunction` facade source remains an unchanged byte prefix, with only the factory appended. Original notebook, quality-loss edits, configs/splits, frozen legacy fixture and pre-existing test deletions/consolidations are preserved. `git diff --check` passes.
- Ignored evidence: [verification](../../../artifacts/kitti/backbone_audit/batch10/verification.json), [pipeline log](../../../artifacts/kitti/backbone_audit/batch10/grouped_pipeline.log) and [full test log](../../../artifacts/kitti/backbone_audit/batch10/full_suite_final.log).

### Batch 10 implementation boundary

The grouped objective can be exercised directly with grouped predictions/targets. Trainer factory/optimizer/warmup integration remains task36, diagnostics aggregation task35, and strict resume/decode/runtime recipes later tasks. No backbone/head parameter counts changed. No training, AP, latency, deployment or SOTA claim. The next checkpoint is tasks31–33.

## Batch 11 evidence

Tasks 31–33 are complete. Existing independent criterion registration needed no production changes; the only production change in this batch is two `no_grad` decorators in the quality-target helpers.

### Independent adaptive state

- Nineteen new state fixtures cover OGA, UWAG, GW-QAL and legacy MGIoU Q-OGA. Their parameters/buffers have disjoint storage, appear under separate group state keys and move through module traversal. Actual two-group train-only parameter totals: baseline 0, UWAG 8, OGA/GW-QAL/Q-OGA 10, OGA with IQA 12. These are criterion parameters, separate from the backbone/head budget.
- Weighted grouped objectives, every prediction gradient, adaptive parameter gradients and updated state match independent existing criteria exactly across two updates, including an empty group. Car-only objective/backward/SGD leaves the other head and criterion unchanged; OGA/GW-QAL/Q-OGA initialize only the selected group's EMA.
- Eval forwards leave both groups' state unchanged. Actual serialized criterion state roundtrips and preserves eval outputs and the next training update exactly. This is criterion state evidence; strict full-detector/optimizer/RNG resume is still tasks38–39.
- Structural validation rejects an invalid later-group shape before an earlier-group EMA can update. Curriculum semantic preflight/broadcast remains task34.

### IQA routing and target gradients

- Thirty-nine new IQA fixtures cover baseline/OGA with MGIoU, yaw-footprint and exact rotated-IoU targets. Both groups supervise the same cell with different assigned geometry; perfect/disjoint ordering, reversed quality ordering and explicit per-group BCE references detect accidental cross-group target reuse. Gaussian/binary paths retain their existing coefficient/diagnostic conventions.
- Adding IQA leaves box/class prediction gradients unchanged under controlled unit calibration; quality logits receive the expected weighted BCE gradients, including graph-connected zeros for an empty group. Actual head/criterion forwards show IQA-only gradients in the selected quality branch and shared input features; its box branches and the other group head receive none.
- Four genuine RED cases exposed target-side gradient leakage in the existing MGIoU/yaw-footprint helpers when assigned boxes require gradients, through both direct and dispatcher calls. Exact rotated-IoU already used `no_grad`. Added the same decorator to the two proxy helpers; all calculations, selected methods, dtype handling and previous file edits remain unchanged. These targets now detach both predicted and assigned geometry.
- A fixture initially assumed a disjoint box must have zero MGIoU. That assertion was corrected: mean projection GIoU can remain positive when boxes separate on x and align on y. The tested value is a legacy similarity proxy, not area IoU. This did not trigger a formula change.
- Unsupported separate-IQA strategies still fail clearly at construction. Exact Q-OGA quality/curriculum routing remains task34, separate from IQA branches.

### Binary capability and overlap audit

- Forty new capability fixtures cover baseline/OGA with IQA on/off and UWAG without IQA. Real synthetic dataset targets from Python/Numba retain local background 0 and foreground labels 1/2; actual grouped heads and registered criteria complete two finite SGD steps, including negative-only VRU groups. Their weighted losses/prediction gradients match independent existing binary criteria exactly.
- Actual legacy binary GW-QAL fails its populated CQFL branch with an `IndexError`: integer `[B,H,W]` labels cannot index Gaussian `[B,C,H,W]` logits. Its empty-group fallback is finite, but that is not populated support. Legacy binary Q-OGA fails the same CQFL shape contract even on empty groups. Grouped construction continues to reject both; no capability matrix expansion or legacy objective rewrite was made.
- Real Pedestrian/Cyclist overlaps confirm that reversing source order changes the painted binary class while canonical regression maps remain identical. With perfect canonical Pedestrian geometry, IQA stays near one even when the painted class is Cyclist; Cyclist geometry receives lower quality. Baseline/OGA classification losses change with painting order while IQA loss remains identical. This inherited within-group association limitation is documented; the main candidate remains Gaussian.

### Verification, preservation and implementation boundary

- Existing adaptive composition/registration was audited directly; passing tests were not misrepresented as new RED failures. The quality-target production fix followed four genuine failing regression cases. State gate: **65 passed, 6 subtests passed in 1.53 s**. IQA/state/legacy/exact-quality gate: **119 passed, 13 subtests passed in 6.86 s**. Binary/composition gate: **72 passed in 2.46 s**.
- Final focused combined gate: **313 passed, 13 subtests passed in 9.12 s**. Final full repository suite outside the sandbox: **858 passed, 17 subtests passed in 47.36 s**, exit 0, no skips/deselections. Worker IPC cases ran. The 57 existing TorchScript/Python 3.14 deprecation warnings remain; numerical checks pass. Python 3.10 and dedicated CUDA/AMP checks remain unavailable/pending.
- Twenty-seven protected hashes match the fresh batch baseline, including trainer, facade, other loss strategies, weighting, notebook, configs/splits and legacy fixture. The authorized `iou_targets.py` exception is exactly the two decorators: removing just those restores its captured source byte-for-byte. Original test deletions/consolidations remain preserved. New-source whitespace and `git diff --check` pass.
- Ignored evidence: [verification](../../../artifacts/kitti/backbone_audit/batch11/verification.json), [criterion audit](../../../artifacts/kitti/backbone_audit/batch11/criterion_audit.json), [IQA RED log](../../../artifacts/kitti/backbone_audit/batch11/iqa_red.log), [focused log](../../../artifacts/kitti/backbone_audit/batch11/focused_final.log) and [full test log](../../../artifacts/kitti/backbone_audit/batch11/full_suite_final.log).

These are CPU numerical/gradient and synthetic optimizer checks. Trainer factory integration, exact Q-OGA epoch broadcast, quality aggregation, full-detector resume, grouped decoding, complete presets and GPU/3D/AP evidence remain pending. No long training, AP, latency, deployment or SOTA claim. Next is tasks34–36.

## Batch 12 evidence

Tasks 34–36 are complete. Grouped objectives are integrated into the actual CLI trainer; accuracy validation and strict semantic resume are still separate gates.

### Q-OGA exact curriculum

- `GroupedLossFunction.set_epoch` validates a nonnegative integer and broadcasts the same zero-based epoch to every supported strategy. It adds no wrapper checkpoint buffers. Existing per-group Q-OGA epoch buffers persist; target-only exact runs add none.
- Nineteen fixtures cover warmup start/midpoint/end/clamping, empty groups, distinct group geometry at a shared output cell, invalid epoch values, missing epochs, divergent epochs, and peaks lacking assigned regression. Semantic checks occur before any group's adaptive forward, so malformed later-group peaks cannot update the earlier EMA.
- Strict criterion state roundtrip preserves the next loss/update; missing a group's curriculum buffer fails strict loading. Two class peaks inside VRU still share one canonical regression quality and count as one supervised cell. This limitation is explicit rather than silently treated as per-class regression.
- Initial RED: **18 failed, 1 passed**. Exact/capability/state GREEN: **102 passed, 7 subtests passed in 1.98 s**. Original Q-OGA strategy formulas/source are unchanged.

### Quality diagnostics by sufficient counts

- Objective weights remain fixed. Q-OGA quality counts add across groups; its means use peak-cell counts. IQA exposes detached `iou_target_count` per group and globally, and weights `mean_iou_target` by assigned regression-cell counts, including nonpeak cells. Empty counts yield finite zero means and preserve curriculum mix.
- The grouped wrapper preserves all existing per-group diagnostic conventions. OGA already exposes its target mean; baseline does not, so baseline's grouped diagnostic uses a second detached target calculation. This adds diagnostic work and needs future profiling; no latency benefit is claimed.
- `QualityMetrics` accumulates independent global/group streams across batches. Namespaced quality diagnostics and IQA means/counts are excluded from generic sample averaging. Legacy OGA keeps its target mean: the trainer derives its sufficient count from the flat mask without changing the facade. Other objective/component diagnostics retain existing averaging behavior.
- Curriculum mixes must agree across groups and within an epoch's metric stream. Unequal 1:3/1:9 objective weights cannot scale quality counts. A fixture with one perfect Car peak and three disjoint VRU peaks gives quality mean 0.25, independent of objective weights; mixed empty/populated batches preserve group counts.
- Initial metrics RED: **16 failures** for missing aggregates/classifier. A separate legacy-IQA regression then exposed the required target-count adapter and passed after its addition. Seventeen metric cases now cover group/batch weighting, zero counts, mix mismatch, validation and legacy mean retention.

### Trainer, optimizer and compile warmup

- The CLI resolves detection/group metadata before dataset creation and passes the same immutable groups into both datasets. `build_training_criterion` uses the existing factory and moves registered parameters/buffers to the selected device before optimizer creation. Missing group definitions fail before loading manifests; no implicit default partition was introduced.
- Existing optimizer, scheduler, accumulation, clipping and checkpoint-payload formulas/source remain unchanged. Fixtures verify exact optimizer membership without duplicates for grouped baseline/OGA/UWAG/GW-QAL/legacy Q-OGA Gaussian and supported binary paths, including empty VRU groups. Only total `loss` is used for backward; validation leaves adaptive state unchanged.
- Compile warmup reduces every tensor prediction leaf, including all groups, box branches and IQA. It uses eval mode to preserve BN, restores the original mode and clears optimizer gradients in `finally`; it never calls the criterion. Four actual uncompiled/TorchDynamo-eager checks cover legacy/grouped models and show finite gradients reaching every model parameter, unchanged state and cleared gradients. CUDA/AMP and Inductor execution remain task50 checks.
- Five real CLI fixtures each run two tiny CPU epochs on separate synthetic train/validation IDs, covering Gaussian OGA+IQA, exact Q-OGA with curriculum, binary OGA+IQA, Gaussian baseline+IQA and legacy OGA+IQA. They verify validation metrics, serialized group criterion state, epoch buffers and metrics logs. Generated datasets/checkpoints stay in pytest temporary directories.
- Trainer tests initially failed for missing integration/helpers. Fixture mistakes (an input width not divisible by 16, an assumed default group partition, augmentation attribute/checkpoint names and CPU precision) were corrected to match existing contracts; no production contract was relaxed for them. Final trainer/metrics/optimizer/exact gate: **88 passed, 7 subtests passed in 5.64 s**. A sandbox-only NVML initialization warning occurred during CPU compiler discovery; this is not a successful CUDA check.

### Final verification and preservation

- New coverage totals 64 cases: 19 curriculum, 17 metrics and 28 trainer additions. Final focused combined gate: **276 passed, 7 subtests passed in 6.70 s**. Final full suite outside the sandbox: **922 passed, 17 subtests passed in 49.11 s**, exit 0, no skips/deselections. All worker IPC cases ran; 57 existing TorchScript/Python 3.14 deprecation warnings remain.
- Twenty-seven protected hashes match the fresh baseline, including notebook, original quality code, facade, strategy/weighting modules, master/experiment configs, splits and frozen fixture. Authorized changes are `grouped.py` and scoped trainer integration. All pre-existing trainer definitions except `QualityMetrics`, `validate` and `main` are unchanged; only two helper functions were added. Original test deletions/consolidations are preserved. New-source whitespace and `git diff --check` pass.
- Ignored evidence: [verification](../../../artifacts/kitti/backbone_audit/batch12/verification.json), [focused log](../../../artifacts/kitti/backbone_audit/batch12/focused_final.log), [full log](../../../artifacts/kitti/backbone_audit/batch12/full_suite_final.log) and RED logs in the same artifact directory.

Backbone/head parameter counts are unchanged; grouped OGA+IQA has 12 train-only parameters. Python3.10, dedicated CUDA/AMP/Inductor readiness, strict architecture/objective/RNG resume, grouped decoding, complete presets and KITTI 3D/AP evidence remain pending. These are synthetic CPU integration checks, not a long KITTI training experiment or SOTA result. Next is tasks37–39.

## Batch 13 evidence — tasks 37–39

Tasks 37–39 are complete. These gates establish CPU correctness and reproducible synthetic epoch-boundary restoration; they do not establish detector accuracy.

### Task 37 — shared focal/local adapters and grouped objectives

- Added `tests/test_context_grouped_compatibility.py`: **63 cases** covering focal with none/ECA/SimAM across Gaussian baseline/OGA/UWAG/GW-QAL/legacy Q-OGA/exact Q-OGA and supported binary baseline/OGA/UWAG. Each path includes populated/empty VRU groups and group-local box targets at the same cells.
- Every active focal/local parameter receives finite nonzero gradients at nonzero gamma/beta initialization. Actual model/criterion optimizer membership is exact and has no duplicates. Rotated quality targets are detached; Car-only IQA reaches shared adapters and only the Car quality branch. Exact Q-OGA plus separate IQA is rejected before model construction.
- Initial adapter fixtures searched the configuration field name rather than the actual registered `c3_light_attention` module name; corrected the test selection. All 63 cases then passed without changing backbone/head/loss production code.

### Task 38 — semantic checkpoint identity and preflight

- Added pure `checkpoint_identity`/`checkpoint_backends` helpers in `common.py`; existing helper definitions remain unchanged. Version 1 saves encoding semantic metadata/hash and input width, class order, group order/classes/global IDs, normalized weights, box/classification mode, stage depths/output width, active focal/local versions/settings, fusion/detail/IQA and objective settings. Model defaults are resolved; additional model/loss options are retained conservatively. Optimizer/schedule/precision/batch semantics participate too.
- The trainer saves the identity plus model/criterion/optimizer/scheduler/scaler state, optimizer initialized-parameter IDs and state-type metadata. Effective CLI epoch/batch/accumulation overrides enter resolved config. Architecture/objective/config integrity is checked before restoring any live state. Shape-compatible dilation, SimAM lambda, quality method, class ordering and other changes cannot silently resume.
- Preflight also requires complete model/criterion buffers with matching shapes/dtypes, optimizer group/parameter order and moments, scheduler/scaler structure and curriculum epoch consistency. Missing adaptive EMA buffers cannot use the legacy loader's permissive fallback. Loader validation runs on copies before live mutation.
- Pure raw/wrapped legacy weights remain usable with a matching legacy topology and are explicitly logged as `legacy_weights` at epoch zero with fresh training state. Older incomplete training checkpoints are rejected for full resume. Cross-topology initialization is still task40.
- Numerical backend changes require explicit previously verified parity via `--backend-parity-verified`; saved/current pairs are logged. The flag records an assertion, not an automatic parity experiment. Compile/worker choices remain recorded separately. Failed resume does not overwrite `config.resolved.json`.
- Actual serialized checkpoints use modern weights-only loading successfully and preserve BN buffers, focal gamma, optional beta/ECA convolution and both IQA branches exactly.

### Task 39 — RNG and adaptive training restoration

- Save/restore Python, NumPy, Torch CPU, visible CUDA RNG lists and the explicit train DataLoader generator. NumPy state uses tensors/primitives to support `torch.load(..., weights_only=True)`. RNG is preflighted without advancing global streams and restored after compiler setup/warmup.
- Twelve deterministic workers-0 fixtures cover baseline+IQA, OGA+IQA, UWAG, GW-QAL, legacy Q-OGA and exact Q-OGA, with populated/empty VRU. Uninterrupted versus restored next-step losses, model weights, optimizer moments, EMAs, curriculum, scheduler and subsequent RNG draws match exactly (`rtol=atol=0`).
- Three actual CLI fixtures use random rotation and compare uninterrupted epoch 2 with epoch-1 checkpoint → resumed epoch 2: grouped OGA+IQA, grouped exact Q-OGA and legacy OGA+IQA, all with focal+SimAM and workers 0. Model/criterion state, numerical validation metrics, scheduler and CPU/loader RNG match exactly. Wall-clock `seconds` are excluded from numerical equality.
- Checkpoint `replay` metadata records epoch-end boundaries and worker count. Persistent-worker augmentation state is not serialized; logs explain that worker replay is unsupported. CUDA RNG code is implemented but dedicated CUDA/AMP/Inductor replay remains unverified; no device/worker-independent bitwise equivalence is claimed.
- RED: the initial 49 checkpoint cases failed on missing identity/payload/restore APIs. Further regression fixtures exposed accepted missing Adam moments, missing initialized parameter entries and inconsistent curriculum buffers; production preflight now rejects these before mutation. CLI fixture corrections used the existing rotation keys (`limit_angle`, `p`) and excluded variable wall time; no augmentation or numerical contract was relaxed.

### Verification and preservation

- **73 new checkpoint cases + 63 adapter cases = 136 new tests.** Final plan-specified focused gate: **277 passed, 7 subtests passed in 15.53 s**, exit 0. A sandbox-only NVML initialization warning appeared during CPU compiler discovery; it is not CUDA readiness evidence.
- Full current repository suite outside the sandbox for required DataLoader IPC: **1,058 passed, 17 subtests passed in 62.37 s**, exit 0, no skips/deselections. 57 existing Python 3.14 TorchScript deprecation warnings; numerical legacy tests pass.
- Production edits are confined to `common.py` (new pure helpers only) and `train.py` (checkpoint payload/parser/main plus resume/RNG helpers). AST comparison verifies optimizer, scheduler, clipping/accumulation logic, validation, criterion construction, curriculum setter and compile warmup definitions remain unchanged. New attention/loss gates needed no detector production edits.
- Fresh protected-file hashes preserve the notebook, detector code including prior Q-OGA/IoU changes, configs/experiment overrides and frozen legacy fixtures. Existing test deletions/consolidations remain untouched. `git diff --check` and new-file whitespace checks pass. Evidence is ignored under `artifacts/kitti/backbone_audit/batch13/`, including RED logs, focused/full final logs, baseline and `verification.json`.

Backbone/head parameter counts are unchanged: main focal backbone+neck **660,528**, Gaussian IQA grouped model **846,689**, excluding **12** train-only OGA+IQA parameters. Python3.10, dedicated CUDA/AMP/Inductor checks, grouped decoding, complete presets and KITTI 3D/AP evidence remain pending. No long KITTI training, AP, latency, deployment or SOTA result is claimed. Next is tasks40–42.

## Batch 14 evidence — tasks 40–42

Tasks 40–42 are complete. Warm-start and grouped Gaussian decode are implemented; grouped evaluation and binary/global-cap gates are still later tasks.

### Task 40 — explicit compatible-backbone warm-start

- Added `common.warm_start_backbone`: normalize raw/wrapped/DDP checkpoints, select only exact `backbone.*` key/shape/dtype matches, preflight the selection and report loaded/skipped/missing destination keys. Skipped source keys carry reasons. Requests with no compatible backbone tensors and malformed backbone tensors fail clearly.
- Heads are always left fresh, even if source/destination grouped names and shapes match. An 8→14 stem convolution is skipped rather than tiled; absent `stage2.2`, focal context and local adapter state retains its destination initialization. LiteMLA keys are not renamed or transferred into focal.
- `--warm-start` and `--resume` are mutually exclusive at argument parsing. The CLI records checkpoint path/SHA256 in initialization metadata and `warm_start.json`, logs counts, and retains fresh epoch/criterion/optimizer/scheduler/RNG. Existing strict-resume functions remain unchanged.
- Ten added checkpoint tests cover raw/full legacy sources, changed stem/new stage/context/local/head values, unsupported sources, parser exclusivity, all three DDP wrappers and matching grouped source heads. An actual tiny CLI run starts fresh exact Q-OGA training from an epoch-99 legacy OGA source: output epoch is 1, curriculum is 0, scheduler epoch is 1 and Adam steps are 1. Source criterion state is ignored.
- RED: six initial cases failed on the missing helper/CLI flag. Initial checkpoint gate then passed **79 tests in 7.90 s**; final expanded checkpoint file has **83 tests**.

### Task 41 — separate candidates and final NMS

- Extracted `decode_candidates` for detached Torch `[N,7]` candidates before suppression/sorting and `finalize_detections` for existing classwise rotated NMS plus descending score order. Original `filter_pred` positional calls remain valid.
- The metric cell-origin decode, log width/length ordering, doubled yaw, sigmoid probabilities, IQA formula, raw-class/ranking threshold checks and per-class/legacy peak policies retain their existing arithmetic. Polygon conversion/IoU/non-max-suppression helper definitions remain unchanged. Flat binary inference retains its historical sigmoid policy explicitly.
- Captured **32 frozen flat outputs before production changes** in `tests/fixtures/flat_decode_contract.json`: both peak modes, IQA on/off, two score thresholds and four NMS settings. Source SHA256 matches this batch’s saved pre-task41 source. Final code matches the frozen rows/scores/order at tight CPU tolerances. Existing empty-shape and NMS regressions pass.
- RED: candidate/finalization APIs and grouped keyword dispatch were missing (**24 failed, 32 frozen cases passed**). Initial decoder/legacy/regression/IQA/assignment gate passed **126 tests in 8.19 s**.

### Task 42 — group-local Gaussian decode

- `filter_pred(..., task_groups=resolved_groups, cls_encoding="gaussian", use_iou=...)` validates group names, local channel counts and optional resolved IQA agreement. It reuses the shared task-group resolver and checks supplied dataset `objects` against global IDs.
- Each group decodes its own metric offsets/log sizes/doubled yaw and IQA score, remaps local foreground channels through ordered global IDs, then joins candidate tensors before shared finalization. `peak_mode=legacy` takes a class maximum within each group. No shared-regression reconstruction is introduced.
- Synthetic asymmetric geometry verifies x/y resolutions/origin, reversed Pedestrian/Cyclist local ordering, length/width, yaw and group box association. Same-cell Car and VRU boxes survive classwise NMS. Changing Car IQA affects only Car scoring; alpha 0/.5/1 and absent-IQA fallback work. Both raw-class and ranking score thresholds are required.
- All-empty scenes retain float32 `(0,7)` output. Missing/extra/mixed groups, wrong class width/batch/spatial/box/IQA shape, nonfinite output, inconsistent global metadata and missing/invalid heads fail clearly. An extended RED fixture exposed `None` regression heads reaching `.ndim`; grouped boundary validation now raises a clear ValueError before decode.
- New decoder file contains **69 tests**, including the 32 frozen flat cases and explicit legacy flat binary regression. Grouped binary is deliberately rejected until task43; global cap/order gates remain task44. The existing evaluator/profiler has not yet been switched to grouped mode (task45).

### Verification and preservation

- **79 new cases:** 69 decoder cases + 10 added checkpoint cases. Final plan/relevant gate: **293 passed in 17.71 s**, exit 0. A sandbox-only NVML initialization warning occurred during CPU compile discovery; no CUDA readiness inference.
- Full current suite outside sandbox for required worker IPC: **1,137 passed, 17 subtests passed in 71.11 s**, exit 0, no skips/deselections. 57 existing Python 3.14 TorchScript deprecation warnings; numerical tests pass.
- Production edits are confined to `common.py` (new warm-start helper), `train.py` (parser/main initialization) and `detector/postprocess.py` (decode/finalization dispatch). AST comparisons preserve every existing common helper, trainer optimizer/scheduler/criterion/strict-resume/RNG/validation/warmup definition and standalone polygon NMS helper. Prior checkpoint test definitions remain unchanged; warm-start tests are appended.
- Fresh hashes cover 172 files, including all existing tests, detector/tools, configs/experiment overrides, splits, notebook and frozen legacy fixtures. Only the three authorized production files, checkpoint test additions and three English documentation files changed. Existing test deletions/consolidations remain preserved. New source whitespace checks and `git diff --check` pass.
- Ignored evidence: `artifacts/kitti/backbone_audit/batch14/` contains RED/initial/final logs, source snapshots, preservation baseline and `verification.json`. Synthetic datasets, reports/checkpoints are in pytest temporary directories.

Backbone/head budgets are unchanged: focal backbone+neck **660,528**, Gaussian grouped BEV IQA model **846,689**, excluding **12** train-only grouped OGA+IQA parameters. Python3.10, dedicated CUDA/AMP/Inductor/rotated-NMS checks, complete runtime recipes and KITTI AP/3D validation remain unverified. No long KITTI training, latency, export, deployment or SOTA claim. Next is tasks43–45.


## Batch 15 evidence — tasks 43–45

Tasks 43–45 are complete. Grouped binary decoding, shared detection limits and PyTorch evaluation/profiling have synthetic CPU evidence. Existing backbone, trainer, encoding, model/head/loss implementations and runtime configs were not changed in this batch.

### Binary decode and shared finalization

- Grouped binary logits use local softmax over background plus foreground. Local foreground indices map through each group's ordered global IDs. Background-winning cells, including ties with background, are masked before peak pooling so they cannot emit detections or suppress nearby foreground, even at IQA alpha 1. Additive logit shifts leave probabilities unchanged.
- Default per-class peaks can retain multiple foreground channels above threshold when foreground wins against background; grouped legacy peaks take the within-group class maximum. Raw classification and ranking thresholds both remain required. Legacy flat inference retains its historical sigmoid policy and all 32 frozen decoder cases.
- Candidates from every group are mapped and merged before classwise rotated NMS, descending score sort and one optional positive-integer global cap. Same-class duplicates suppress before capping; cross-class overlaps survive. Group order preserves results for unequal scores; exact-score tie ordering is not promised.
- Real evaluator fixtures exposed floating-point truncation of valid grouped grids (for example 9.6 / 0.2). Grouped decode now rounds dimensions and verifies positive integrality with tolerance; genuinely nonintegral grids fail. The historical flat truncation arithmetic remains unchanged.

### Evaluation and parameter reports

- PyTorch evaluation checks architecture/objective identity before model loading. Grouped checkpoints require identity metadata, including group/class order, encoding, focal/local settings and IQA semantics. Model-only inference excludes training-only identity fields such as optimizer, epochs and precision; a saved config must still agree with its own identity. Matching pure legacy weights remain usable. This does not relax full training resume.
- Dataset global class IDs flow through decode, saved predictions and per-class AP attribution, including reordered global IDs and reversed VRU-local order. Existing ROI, difficulty, calibration, polygon/IoU and AP R40 formulas are preserved.
- JSON reports record resolved config, encoding and split hashes, dirty-source provenance, checkpoint epoch, mappings, normalized weights, actual quality presence/target/alpha/curriculum, peak scope and shared cap policy. Comparison CSV adds semantic/provenance and parameter fields.
- Shared helpers in `common.py` provide the evaluation preflight and actual parameter counts without duplicating evaluator/profiler logic. Reports separate body, neck, combined backbone, each head, all heads, total detector and train-only criterion parameters. Existing common helper definitions remain unchanged.
- The profiler handles nested grouped outputs and each actual head's Conv2d hooks, including a full-resolution meta-device shape audit. Named profiling variants intentionally retain their existing single-head contracts; explicit grouped configs are supported. Shape-based convolution MAC estimates are not latency measurements.

### Verification and preservation

- New coverage: **30 decoder cases** and **18 evaluator/profiler/comparison cases**. Decoder file now has 99 cases, including 32 frozen flat fixtures. Tests use small synthetic datasets/checkpoints and actual CLI paths; generated inputs/results stay in pytest temporary directories.
- RED evidence: binary/cap gate **27 failed, 69 passed**; initial evaluator gate **15 failed**; grouped-grid regression **1 failed**; report/protocol gate **4 failed, 14 passed**. The first integrated run exposed three valid-grid failures; these were fixed in grouped production decode rather than changing fixture geometry.
- Final focused gate: **324 passed in 23.32 s**, exit 0, no warnings/skips/deselections. Full suite outside the sandbox for required DataLoader IPC: **1,185 passed, 17 subtests passed in 67.61 s**, exit 0, no skips/deselections. All worker cases ran; the 57 existing TorchScript/Python 3.14 deprecation warnings remain.
- Fresh preservation snapshot covers 174 existing files: **165 unchanged**, with nine authorized source/test/documentation changes and one new evaluation test file. Existing common helpers, evaluator geometry/AP/parser/deployment definitions, comparison CLI and polygon/NMS helpers retain their ASTs. Notebook, configs/splits, trainer, backbone, model/head/loss/encoding and frozen fixture hashes are unchanged. Prior deletions/consolidations are preserved. New Python whitespace checks and `git diff --check` pass.
- Evidence: [verification artifact](../../../artifacts/kitti/backbone_audit/batch15/verification.json), [focused log](../../../artifacts/kitti/backbone_audit/batch15/focused_final.log), [full log](../../../artifacts/kitti/backbone_audit/batch15/full_suite_final.log). RED logs, source snapshots and preservation baseline remain in the same ignored directory.

Main focal backbone including neck remains **660,528** parameters. Gaussian grouped BEV IQA detector remains **846,689**, excluding **12** train-only OGA+IQA parameters. These gates establish synthetic CPU integration, not KITTI accuracy. Complete recipes/deployment guards, Python 3.10, dedicated CUDA/AMP/Inductor/rotated-NMS, true 3D and reference evaluation remain later gates. No long KITTI training, AP/latency benchmark, export, deployment or SOTA result is claimed. Next is tasks46–48.


## Batch 16 evidence — tasks 46–48

Tasks 46–48 are complete. Deferred deployment paths now reject unsupported output contracts; complete BEV configs and notebook controls are available without starting any training run.

### Deployment contract guards

- A shared pure configuration preflight in `common.py` rejects grouped and IQA ONNX export/TensorRT loading with actionable errors before checkpoint deserialization or CUDA/package/engine work. Current consumers support only flat four-output predictions; quality must not be silently discarded.
- `RawHeadWrapper` checks the actual model topology and output keys as well as export CLI configuration. Legacy Gaussian/binary IQA-off retains exactly `cls`, `offset`, `size`, `yaw`, with bitwise CPU forward parity. This is a guard, not an ONNX graph or TensorRT engine parity result.
- TensorRT's pre-existing `del config` was replaced by the preflight because later input/output checks need that configuration. CUDA engine execution remains unverified/deferred.

### Complete recipes and notebook controls

- Nine deterministic, complete JSON configs combine master config, the existing hybrid GT augmentation recipe, common manifest settings and per-variant overrides. They inherit seed/split/schedule/precision/target backend and carry distinct experiment names. The single-head reference explicitly removes group-only fields. No 3D recipe is generated; unknown or pending 3D recipe requests fail.
- Recipes cover the single no-context reference, grouped no-context OGA/baseline references, main focal/local-none OGA/baseline candidates, and separate ECA/SimAM/fusion32/detail comparisons. Optional neck configs are published after their earlier implementation/CPU alignment and gradient gates; none is enabled in the main recipe.
- `resolve_under1m_recipe` and `write_under1m_presets` expose reproducible expansion/materialization without training. Every grouped recipe passes a full `[1,14,800,704]` meta-device profile, actual body/neck/head/criterion counts and nested output-shape assertions. Meta checks are structural, not numerical/GPU/accuracy evidence.
- The notebook defaults to `UNDER1M_FOCAL_OGA_IQA` and augmentation `config`, preserving hybrid GT. New custom controls expose hist14/backend, stage depths, head mode/classification, ordered groups/weights, focal settings, ECA or SimAM settings, internal fusion width and detail. Custom controls only affect `custom`; runtime fields still take final precedence. Choosing another augmentation replaces the recipe, with `hybrid_gt` applying hybrid controls explicitly.
- Actual configuration-cell executions cover default selection, each local option, custom grouped binary mappings/weights and repeated preset switching. Existing trained-run config protection is exercised. Notebook edits affect only source cells 0 and 3; all IDs, metadata, outputs and other cells are preserved.
- Edited base configs with rich8/rich12/binary widths now switch cleanly to fixed hist14 width: incompatible previous `out_channels` is removed only when the preset changes the encoding. New candidate names have readable context/local/fusion/detail tokens and a semantic digest covering architecture, encoding and objective. Default normalization and backend/training-only changes retain the name; real density/IQA/activation changes alter it. Historical default names remain unchanged; existing candidate names gain the new explicit format.
- The old all-config run-name test's encoder allowlist now includes hist14. New notebook test files have scoped names (`test_under1m_notebook_config.py`, `test_under1m_notebook_controls.py`); earlier deleted/consolidated notebook test files were not recreated. Task47/48 commands now reference these current files.

### Verified parameter counts

| Recipe | Backbone including neck | Heads | Detector | Train-only criterion |
| --- | ---: | ---: | ---: | ---: |
| Single no-context OGA+IQA | 635,408 | 93,130 | 728,538 | 6 |
| Grouped no-context OGA+IQA | 635,408 | 186,161 | 821,569 | 12 |
| Grouped no-context baseline+IQA | 635,408 | 186,161 | 821,569 | 0 |
| Main focal OGA+IQA | 660,528 | 186,161 | 846,689 | 12 |
| Main focal baseline+IQA | 660,528 | 186,161 | 846,689 | 0 |
| Optional focal+ECA | 660,532 | 186,161 | 846,693 | 12 |
| Optional focal+SimAM | 660,529 | 186,161 | 846,690 | 12 |
| Optional focal/fusion32 | 666,768 | 186,161 | 852,929 | 12 |
| Optional focal/detail | 661,720 | 186,161 | 847,881 | 12 |

### Verification and preservation

- Added **66 cases**: deployment 12, recipe/identity 20, notebook resolver 27 and notebook controls 7. RED: deployment **9 failed, 3 passed** after correcting test import setup; recipe **14 failed**; notebook **22 failed, 9 passed**; readable names **1 failed**; edited-base width **3 failed**; semantic names **4 failed, 1 passed**. Logs retain the initial test import error separately from the valid RED evidence.
- Initial integration: **127 passed, 1 failed**, exposing the old all-config test's missing hist14 allowlist entry. Initial broader focused gate: **299 passed in 21.59 s**. After the width/semantic-name regressions and fixes, final focused gate: **307 passed in 21.89 s**, exit 0, no warnings/skips/deselections.
- Final full repository suite outside the sandbox for required DataLoader IPC: **1,251 passed, 17 subtests passed in 80.45 s**, exit 0, no skips/deselections. Worker cases ran; the 57 existing TorchScript/Python 3.14 deprecation warnings remain. Python3.10 and dedicated CUDA/AMP/Inductor/rotated-NMS remain unverified.
- Fresh preservation baseline covers **175 existing files**: **163 unchanged** and 12 authorized source/test/documentation changes. Existing common definitions change only `generate_run_name`, with the deployment helper appended; evaluator definitions change only `TensorRTRunner`; export changes only wrapper/main; notebook resolver changes only selection, with new helpers. Legacy preset dictionaries and save/validation functions remain unchanged. Backbone/head/loss/encoder/trainer/postprocess, master/augmentation configs, splits, prior test deletions and frozen fixtures are preserved. Python whitespace/AST checks and `git diff --check` pass.
- Ignored evidence: [verification](../../../artifacts/kitti/backbone_audit/batch16/verification.json), [recipe audit](../../../artifacts/kitti/backbone_audit/batch16/recipe_audit.json), [focused log](../../../artifacts/kitti/backbone_audit/batch16/focused_final.log), [full log](../../../artifacts/kitti/backbone_audit/batch16/full_suite_final.log). Source snapshots, preservation baseline and RED logs are in the same directory. Runtime usage and regeneration are documented in [recipe README](../../../configs/experiments/under1m/README.md).

These are synthetic CPU and meta-device structural results. No long KITTI training, measured AP/latency, CUDA readiness, deployment or 3D/SOTA result is claimed. The main detector budget/topology is unchanged. Next is tasks49–51.


## Batch17: tasks49–51

Tasks49–50 are complete. Task51 records the candidate benchmark policy and keeps final freeze pending; completed total is **50/58**. No core detector, encoder, loss, trainer, decoder, runtime config, split or notebook changed in this batch.

### Assembled CPU pipeline (task49)

- Added 18 cases spanning focal/local-none, ECA and SimAM, Python/Numba targets with NumPy/Numba hist14, empty/Car-only/same-cell cross-group scenes. Actual hybrid database paste and fixed global scale feed the Dataset, grouped targets, OGA+IQA, optimizer/scheduler, saved strict resume and float32 Nx7 classwise-NMS/global-cap decode.
- Same-cell cross-group labels are supplied by the host fixture; the hybrid sampler's overlap guard remains active. Synthetic DB source and host IDs are both training IDs. Initial fixture failures exposed missing source membership, the actual resume mode token and legitimate collision rejection; these were corrected in fixtures without modifying production behavior.
- Existing pipeline test definitions are preserved; cases are appended. Focused compatibility gate: **898 passed in 41.55s**; subsequent protocol regressions are covered by the targeted and full gates below.

### CUDA, AMP and workers (task50)

- Added the reusable [smoke CLI](../../../tools/benchmarks/smoke_detector.py), 11 CPU/CLI tests and 36 actual CUDA tests. CUDA tests: **36 passed in 28.32s**, zero skips. FP32 checks cover all nine valid classification/objective combinations for each of the three local adapters. FP16/BF16 gates exercise the main OGA+IQA objective for each adapter.
- Six real CLI runs cover focal/local-none, ECA and SimAM × workers0/6, each with FP32/FP16/BF16, three successful optimizer updates and full-resolution `[1,14,800,704]` inference. All 18 precision runs passed (54 successful updates). Persistent workers use spawn. Strict model/criterion/optimizer/scheduler/scaler/RNG restoration and exact reloaded evaluation predictions pass; validation preserves criterion state.
- RTX4050 Laptop GPU; Python3.14.4, Torch2.11.0, Torch CUDA13.0. CUDA and worker IPC require execution outside this sandbox. These are actual GPU checks, not CPU/meta substitutes.
- Focal modulation input, local ECA/SimAM gate reductions and objective are FP32 under autocast; heads produce the requested dtype. FP16 GradScaler skips five initial overflowing updates in each CLI run before reaching three finite successful updates. FP32/BF16 skip none. Reports retain this evidence explicitly.
- Peak allocated CUDA memory across these synthetic runs is 110,057,984–139,584,512 bytes. This includes full-resolution inference and is not a full-resolution training or latency benchmark. Rotated NMS uses **CPU polygon fallback**, not CUDA rotated NMS. No Inductor, Python3.10 or deterministic persistent-worker replay claim.
- Synthetic smoke training disables augmentation and changes crop/batch/warmup settings explicitly; task49 covers actual hybrid paste. Runtime recipe files remain unchanged.

### Candidate protocol (task51, partial)

- Added [protocol recorder](../../../tools/benchmarks/benchmark_protocol.py), 17 regression cases, [English policy](benchmark_protocol.md), [3D record](benchmark_protocol_3d.json) and [BEV record](benchmark_protocol_bev.json). The user-selected primary benchmark is AP3D **R11 and R40**, with a reference APBEV mode planned for task57. The recorder's mode switches the requested policy only; current runtime still evaluates local APBEV R40.
- Unique, disjoint split: **3,712 train / 3,769 validation** IDs with recorded file hashes; seed42, full hybrid/model/objective/schedule, thresholds and historical minimum-validation-loss selection are recorded. Comparator mismatches in metrics, sampling, domain, classes/IoU, difficulty/orientation, validation hash, selection, decode score/NMS/cap and macro definitions are explicit.
- Final freeze remains pending actual vertical pipeline, immutable reference evaluator and R11/R40 parity for both modes, final3D GPU recipe, configured data/calibration/train-only GT database audit, comparator protocols and reproduced baseline. Both records set `long_training_allowed=false`; no comparator accuracy or ready AP3D mode is claimed. Independent tasks52–54 can proceed while the final freeze stays open.

### Final verification

- Final smoke/protocol targeted gate: **28 passed in 5.48s**, exit0. Complete suite outside the sandbox: **1,333 passed, 17 subtests passed in 104.59s**, exit0, zero skips/deselections. The 57 existing Python3.14/TorchScript deprecation warnings remain; numerical tests pass.
- The main backbone including neck remains **660,528**; ECA **660,532**, SimAM **660,529**. Main grouped IQA heads are **186,161**, detector **846,689**, with **12** train-only OGA parameters.
- Fresh preservation audit: **189 existing files**, **183 unchanged**; the six authorized changes are five plan/spec/progress documents and appended pipeline tests. Original pipeline bytes remain a prefix of the expanded file. Core runtime code, runtime configs, split manifests, frozen fixtures, notebook and prior test deletions are preserved. New/expanded Python AST and whitespace checks, JSON parsing and `git diff --check` pass. See [preservation audit](../../../artifacts/kitti/backbone_audit/batch17/preservation_audit.json).
- Ignored evidence: [verification](../../../artifacts/kitti/backbone_audit/batch17/verification.json), [GPU audit](../../../artifacts/kitti/backbone_audit/batch17/gpu_audit.json), [CUDA log](../../../artifacts/kitti/backbone_audit/batch17/cuda_final.log), [full log](../../../artifacts/kitti/backbone_audit/batch17/full_suite_final.log). RED and initial fixture/runtime failures are retained separately; they are not presented as production regressions.

No long training, measured KITTI AP, latency, deployment or SOTA claim. Next checkpoint: tasks52–54, with task51 final freeze pending.


## Batch18: tasks52–54

All three batch tasks are complete. **53/58 tasks completed** (1–50,52–54); task51 final benchmark freeze remains pending. This batch implements the vertical prediction/target interface, not a complete trained/evaluated 3D detector.

### Task52: Milestone A readiness

- Added [release_a_verification.md](release_a_verification.md), mapping the actual BEV CPU/GPU evidence to R01–R12 and R14–R15. Records supported strategy/classification/IQA combinations, optional-neck CPU-only scope, attention counts, same-cell collision limits, worker/RNG boundaries and configured-data/protocol gaps.
- Main BEV recipe remains focal/local-none, with the existing hybrid augmentation, hist14, width32 grouped heads and BEV IQA. CUDA NMS, Inductor, Python3.10, matched comparator protocol and real accuracy remain unverified. The functional BEV candidate has synthetic evidence; final comparison readiness remains open.

### Task53: explicit box mode and vertical prediction

- Added pure shared `resolve_box_mode`, canonical `data.box_mode=bev|3d` and an explicit model-constructor argument. The pipeline propagates the resolved mode without mutating the config or inserting duplicate model fields. Default BEV retains existing state keys, seeded branch weights and outputs.
- In 3D mode, each head adds an independent two-channel `vertical` branch `[z_bottom,log(height)]` using the existing Head topology. Supports Gaussian/binary prediction construction and grouped/single-head topology. Permitted full config objectives are baseline/OGA; other 3D strategy extensions are rejected. BEV IQA semantics and branch widths are unchanged.
- Vertical-only gradients reach the selected group's branch and shared input, not other BEV tasks or the other group. Strict model-state reload gives exact evaluation outputs. Existing semantic identities/run names distinguish BEV/3D, and strict cross-mode state loading fails. Full 3D training resume remains task55/58.
- Added guards at the training criterion factory and direct BEV loss, BEV decoder/local evaluator and ONNX/common deployment boundary. Incomplete consumers cannot silently discard vertical tensors. These guards are intentional until tasks55–57 implement their corresponding contracts.
- Updated one old invalid-config fixture from `data.box_mode=3d` to `volumetric`, preserving its invalid-mode coverage; the old `model.box_mode` rejection remains. New cases cover accepted 3D construction.

### Task54: owner-consistent vertical targets

- Added `fill_regression_targets_3d(..., backend=python|numba)` returning the four BEV maps plus vertical. Legacy Python/Numba functions retain their exact four-array definitions. Dataset adds float32 `[2,Y,X]` vertical targets only in 3D mode; collation yields `[B,2,Y,X]` and uses the existing regression mask.
- The shared Python/Numba kernel writes vertical inside the same accepted BEV cell. Canonical global-ID tie-breaking, nearest-center/reserved-peak decisions and legacy last-box-wins assignment remain unchanged. No second owner calculation or height convention change is introduced.
- Cases verify every supervised cell against its BEV owner under unequal XY resolution, same-cell collisions, small-radius reserved peaks, reversed local mappings, cross-group same-cell targets, empty input/groups, actual Dataset loading/collation and source immutability. Invalid/non-finite/nonpositive 3D boxes, mismatched/non-finite radii and unavailable explicit Numba are rejected.

### Counts and verification

| Constructed 3D head topology | Backbone including neck | Heads | Detector |
| --- | ---: | ---: | ---: |
| Focal/local-none | 660,528 | 223,413 | 883,941 |
| Focal/ECA | 660,532 | 223,413 | 883,945 |
| Focal/SimAM | 660,529 | 223,413 | 883,942 |

Each vertical branch adds **18,626** parameters; two add **37,252**. These counts are actual instantiated modules, with full `[1,14,800,704]` structural output `vertical=[1,2,200,176]` per group. No 3D criterion is counted because vertical loss is not implemented. The BEV main detector remains846,689 and OGA remains12 train-only parameters; all backbones remain below1M.

- Added **43 tests**: vertical head21 and targets22. RED: head **21 failed in1.08s** on missing 3D construction/guards; targets **22 failed in2.39s** on missing explicit API/vertical maps. GREEN head/compatibility gate: **110 passed in8.60s**; target/compatibility gate: **101 passed in3.03s**.
- Focused regression gate: **653 passed in24.51s**, exit0; one restricted-sandbox NVML initialization warning, with no skip. Final full suite outside the sandbox: **1,376 passed,17 subtests passed in111.88s**, exit0, no skips/deselections. All36 existing CUDA BEV cases ran. The57 existing TorchScript/Python3.14 warnings remain. Dedicated 3D CUDA training/reference and Python3.10 are pending.
- Fresh preservation baseline: **199 existing files**, **182 unchanged**,17 authorized source/test/spec/progress changes. Backbone/registry/encoder, runtime/augmentation configs, splits, notebook, frozen fixtures and earlier test deletions are preserved. AST comparison confirms the old ownership algorithm is identical after removing only the optional vertical write; legacy public target definitions are identical. Existing common definitions change only `build_model` and `validate_deployment_config`. Python AST/new-file whitespace and `git diff --check` pass.
- Primary-agent review finds no Blocker/Major issues within this batch's head/target scope. No independent subagent review was performed. The temporary 3D consumer guards and missing downstream implementation are documented rather than presented as ready 3D behavior.
- Evidence: [verification](../../../artifacts/kitti/backbone_audit/batch18/verification.json), [vertical module audit](../../../artifacts/kitti/backbone_audit/batch18/vertical_module_audit.json), [preservation audit](../../../artifacts/kitti/backbone_audit/batch18/preservation_audit.json), [focused log](../../../artifacts/kitti/backbone_audit/batch18/focused_final.log), [full log](../../../artifacts/kitti/backbone_audit/batch18/full_suite_final.log). RED logs and before-snapshots are retained in the same ignored artifact directory. [Vertical interface](vertical_interface.md) documents configuration, layout, API and guards in English.

No long training, measured KITTI AP/latency, deployment or SOTA result. Next checkpoint: tasks55–57; task51 final freeze and task58 final 3D GPU recipe remain open.

## Batch19: tasks55–57

All three tasks are complete: **56/58** overall (1–50,52–57). Task51 final benchmark freeze and task58 remain open. No long training, trained KITTI AP, latency, deployment or SOTA result is claimed.

### Task55: weighted vertical objective

- The baseline/OGA 3D criterion requires explicit finite positive `loss.vertical_loss_weight`. Each head supervises `[z_bottom,log(height)]` with FP32 Smooth L1, beta1, mean across both channels of owned cells. Selection occurs before arithmetic; empty masks return a connected zero. All grouped vertical inputs validate before adaptive state updates.
- The weighted term is added after the existing BEV strategy objective and before normalized group aggregation. OGA uncertainty tasks and BEV IQA formulas are unchanged. The coefficient participates in existing strict objective identity; a real CPU optimizer/scheduler/checkpoint roundtrip restores model/criterion exactly and rejects a changed coefficient.
- Added14 cases. Corrected RED: **14 failed** after fixing initial test-import setup. Final loss/head/IQA/checkpoint gate: **189 passed in9.09s**. The first green run's one failure was an invalid synthetic input width; fixing the fixture preserved production shape constraints.

### Task56: complete calibrated decode

- Explicit `filter_pred_3d` returns float32 Nx9 `[class,score,x,y,z_bottom,l,w,h,yaw]`. Group class/quality/vertical association survives classwise BEV-footprint NMS and one global cap. Default BEV APIs and frozen Nx7 fixtures retain their historical behavior; default BEV decoding rejects vertical tensors.
- Added bottom-center conversion through actual calibration and an inverse of prepare_kitti's upright heading projection. Nontrivial rotation/translation roundtrips preserve processed labels. Genuine 2D boxes use eight camera corners, twelve-edge near-plane clipping at0.1m and actual PNG bounds. No dummy projected boxes or AOS result: yaw remains pi-symmetric, alpha−10.
- Added18 cases, including a later underflow regression. Initial RED17 failed; initial green exposed package-import and string-array truncation bugs, both fixed. Decode/target/legacy-group gate: **138 passed in2.23s**. Underflow RED1 failed; final decoder-only gate: **18 passed**. Nonpositive dimensions are rejected in 3D without changing BEV behavior.

### Task57: immutable KITTI reference

- Vendored OpenPCDet evaluator/overlap unchanged at `233f849829b6ac19afb8af8837a0246890908755`, with source/license SHA256 checks and retained Apache/MIT attribution. Reference overlap uses partitions for full-split memory. No CPU/local-BEV substitution occurs on dependency failure.
- New evaluator defaults to AP3D; APBEV is explicit. Both report R11/R40 separately, all three classes/difficulties, Moderate macro and AP-9, valid GT counts and original full-domain labels. Difficulty, ignored neighbors and metric-specific DontCare use the pinned formulas. Historical local ROI BEV R40 remains separate and rejects 3D configs.
- Added27 cases. Initial RED25 failed. Main training-env CPU check was **8 passed,17 skipped**; these skips did not satisfy R13. The older bundled Numba CUDA backend failed first library discovery then compilation on Python3.14. An isolated reference virtualenv with numba-cuda0.30.4 and CUDA12.9 compiler/runtime wheels successfully ran the original overlap on the actual RTX4050, without changing training packages or upstream formulas.
- First reference GPU gate: **25 passed in22.57s**. Two added checkpoint/Dataset/calibration/comparison integration cases initially exposed CSV field loss for a failed model before a valid model. Fixed mode-preserving error rows, union CSV fields and model/config metadata. The next gate's two failures were the synthetic plateau fixture assuming first-cell selection; the historical decoder/NMS selects the final tied cell. Corrected expected fixture geometry without changing tie behavior.
- Actual checkpoint fixtures use real PNG/calibration/processed points, original labels and both CLI comparison modes; synthetic perfect AP validates the assembled path only. JSON/CSV/Markdown preserve both recall samplings, failed rows, strict checkpoint/quality identity and parameter counts. [Reference guide](reference_evaluation.md) records dependency paths, conventions and commands.

### Final verification and preservation

- Final focused gate: **373 passed in41.07s**, exit0, zero skips/deselections. Full repository suite outside the sandbox with reference CUDA and DataLoader IPC: **1,435 passed,17 subtests passed in136.22s**, exit0, zero skips/deselections. All27 reference cases and all36 existing BEV CUDA cases ran. The78 warnings comprise57 existing TorchScript/Python3.14 deprecations and21 reference Numba performance warnings for small fixtures; numerical checks pass.
- Main 3D backbone including neck **660,528**, heads **223,413**, detector **883,941**, train-only OGA+IQA **12**. Main BEV detector remains846,689. No backbone/neck feature changed in this batch.
- Fresh baseline: **203 existing files**, **189 unchanged**,14 authorized source/test/spec/progress changes. Notebook, master/runtime/augmentation configs, splits, encoding/backbone/target ownership, frozen fixtures and earlier test deletions are preserved. `common.py` is unchanged; trainer criterion wiring passes the resolved box mode; prepare_kitti changes only package/CLI-compatible imports. New Python AST/whitespace, pinned source/license hashes and `git diff --check` pass.
- Primary-agent review finds no Blocker/Major issues within tasks55–57; no independent subagent review. Dedicated 3D CUDA training/AMP, final presets, Python3.10/Inductor, configured real-data/GT-database audits, comparator protocols and measured trained AP remain open.
- Ignored evidence: [verification](../../../artifacts/kitti/backbone_audit/batch19/verification.json), [preservation](../../../artifacts/kitti/backbone_audit/batch19/preservation_audit.json), [reference provenance](../../../tools/kitti_training_pipeline/kitti_reference/provenance.json), [focused log](../../../artifacts/kitti/backbone_audit/batch19/focused_final.log), [full log](../../../artifacts/kitti/backbone_audit/batch19/full_suite_final.log). RED, initial failures, backend probes and before-snapshots remain in the batch directory.

Next checkpoint: task58, with task51 final freeze pending its remaining evidence. Runtime BEV presets and notebook defaults are unchanged; the 3D presets are still to be materialized.

## Batch 20 evidence

Task58's functional recipe/GPU gates are complete. Its conditional final benchmark freeze is tracked by task51 and remains pending actual Colab data and comparator evidence.

- Materialized two explicit manifest-derived 3D configs: main focal/local-none and optional focal/ECA. Added a dedicated 3D resolver/generator without changing the existing nine BEV configs/preset map or notebook defaults. Both file overrides preserve actual Colab runtime paths; the profiler now constructs the actual 3D criterion.
- Pipeline RED:12 intended CPU failures from absent 3D presets/resolver, profiler mode and BEV-only smoke decode. CPU GREEN:43 passed,18 deselected CUDA cases in the initial compatibility gate. Saved-prediction RED:1 expected missing-report-field failure,29 deselected before adding source-aligned Nx9 prediction/GT evidence. Protocol RED:9 failures,17 deselected; GREEN:26 passed in3.11s.
- The initial GPU gate exposed zero finite vertical gradients despite AdamW weight changes: advancing after FP16 overflow allowed negative-only successful batches to exhaust the short smoke. Retrying the same supervised synthetic batch fixed this, then exposed one ECA overflow at scale1; a bounded32-extra-attempt budget allows the verified scale0.5. Production training, default GradScaler and objective formulas are unchanged. Dedicated initial final GPU pipeline gate:30 passed in17.92s. Four assembled hybrid/scaling and two Colab file-override cases were then added; all36 pipeline cases ran in the full suite, including18 actual CUDA cases.
- Four actual CUDA CLI reports: main/ECA × workers0/6, each FP32/FP16/BF16 with3 successful updates;12 precision runs/36 successful optimizer updates. Vertical gradients/ownership/head changes, strict model/criterion/optimizer/scheduler/scaler/RNG restore, unchanged validation adaptive state and exact reloaded outputs pass. Full inference input `[1,14,800,704]`, per-group vertical `[1,2,200,176]`, all finite.
- FP32/BF16 skip0 updates. Main FP16 skips16 initially and ends scale1; ECA skips17 and ends scale0.5. Peak allocated CUDA memory111,397,888–140,730,880 bytes covers tiny training plus full inference, not full-resolution backward or latency. Rotated NMS uses CPU polygon fallback. RTX4050 Laptop with driver610.57.04; Python3.14.4/Torch2.11.0/Torch CUDA13.0, isolated compatible reference CUDA12.9 libraries as in batch19.
- Actual instantiated budgets: main body629,984 + neck30,544 = backbone660,528; ECA adds4 =660,532. Grouped 3D heads223,413 give detector883,941/883,945; OGA criterion12 is train-only. The user's backbone-only <1M budget passes.
- Saved CUDA smoke predictions and their actual source GT were evaluated using explicit synthetic calibration/eight-corner projection. Six main/ECA/precision runs ×2 reference modes ×2 samplings give24 sampling comparisons (216 class/difficulty comparisons), agreeing with direct pinned entry points.41 replicated synthetic frames exercise sampling; these are not real KITTI accuracy. Pinned source/license bytes remain unchanged.
- Focused final gate: **312 passed,21 warnings in57.84s**, no skips/deselections. Full final gate: **1,480 passed,78 warnings,17 subtests passed in150.81s**, exit0, no skips/deselections. All36 existing BEV CUDA and27 reference tests ran. Warnings:57 existing TorchScript/Python3.14 deprecations and21 expected small-fixture Numba occupancy/parallel warnings.
- Updated AP3D/APBEV protocol records use complete main3D config, actual worker/precision smoke hashes and pinned reference verification. GPU/reference gates are verified; real-data/train-only GT database audit and comparator protocols/reproduced baseline remain pending. Both records retain `long_training_allowed=false`. The user confirmed assets/checkpoints are on Colab; no remote asset access or training was performed.
- Review covered mode routing, owned vertical targets, adaptive state, exact restore, source-aligned prediction evidence, recipe budget, evidence trust boundary and existing notebook/runtime compatibility. `git diff --check`, Python/JSON/document checks and baseline preservation are recorded in the final verification artifact. No model/loss/encoding/target source or notebook edits were introduced in this batch.

Evidence under ignored `artifacts/kitti/backbone_audit/batch20/`: [verification](../../../artifacts/kitti/backbone_audit/batch20/verification.json), [preservation audit](../../../artifacts/kitti/backbone_audit/batch20/preservation_audit.json), [GPU audit](../../../artifacts/kitti/backbone_audit/batch20/gpu_audit.json), [saved-prediction reference audit](../../../artifacts/kitti/backbone_audit/batch20/reference_from_smoke.json), [focused log](../../../artifacts/kitti/backbone_audit/batch20/focused_initial.log) and [full log](../../../artifacts/kitti/backbone_audit/batch20/full_suite_final.log). Reviewable English guides: [Milestone B](release_b_verification.md), [Colab runbook](colab_3d_runbook.md), [candidate protocol](benchmark_protocol.md).

## Batch 21: task51 executable benchmark evidence

The implementation is delivered; **57/58 tasks remain finally verified**, because task51's actual Colab benchmark freeze still requires external real-data/trained-checkpoint execution. No synthetic fixture, local missing path or declared comparator is used as real KITTI evidence. The current BEV-first user policy and all eleven runtime configs/notebook defaults are preserved.

- Added `audit_kitti_assets.py`: deterministic, read-only config/split/file inventory; finite chunked point/crop validation, unique/disjoint IDs, processed labels and original validation/calibration inputs; reference modes additionally require valid P2/images. Configured v1 GT databases must match train IDs/source manifest, class/count convention, source label index/geometry and crop counts. Alternate original manifests require the same train IDs and matching metadata SHA. Freeze re-runs the audit to detect stale files.
- Added `benchmark_evidence.py`: validate full own-config PyTorch evaluator output, actual input/config/checkpoint hashes, correct local/reference mode/domain/classes/IoU/difficulty/quality/decode/cap, all nine AP cells per sampling and recomputed macros. Minimum-loss checkpoint selection must agree with complete ordered configured training history, successful optimizer updates, retained flags, checkpoint epoch/objective and selected/retained bytes. Synthetic evidence remains explicitly marked and is rejected for real benchmark freeze.
- Extended protocol recorder with an explicit `local_bev` CLI default and actual R40/ROI geometry. Existing explicit reference modes remain separate; API default is retained for compatibility. Added asset/comparator evidence consumption, current file rechecks, quality/peak/cap/ROI/split comparison, successful-update/full-shape/<1M BEV GPU requirements and conditional `long_training_allowed=true` only after every applicable gate passes. Candidate accuracy remains unmeasured. Baseline attention metadata now records actual context/local settings.
- Local/reference evaluators now record hashes of the actual inputs used. A first full gate exposed four real local-inference fixtures without processed labels: local test Dataset does not consume those labels. The shared provenance helper was corrected to hash point cloud/raw GT/calibration for local BEV, with processed labels/images added for reference validation inference. Historical numerical formulas and target/loss/model code were not changed.
- Initial RED/GREEN:22 asset failures for absent CLI, then22 passed;25 evidence failures for missing recorder, then25 CPU cases passed;3 provenance failures before input hashes;7 consumption/train-split/quality failures before gate wiring;3 bad GPU update/shape/budget cases before validation;1 no-context metadata failure before actual attention recording. Initial focused CUDA/reference gate:134 passed,23 warnings in52.55s, no skips. Subsequent GPU rejection cases and attention regression are included in the final full gate.
- Actual evidence fixtures train two small CPU epochs and evaluate their selected checkpoints. Two new reference evidence cases invoke real compatible CUDA overlap kernels for reference BEV/3D; a third new CUDA case exercises FP32/FP16/BF16 smoke through complete gate closure on synthetic test assets. This is branch-control verification, not a real KITTI protocol freeze or measured benchmark claim.
- Five standalone CLI commands were verified on explicitly synthetic assets: asset audit, default local-BEV recording, selection/evaluation evidence, data-only gate clearing and rejection of synthetic evidence. The initial internal comparator is the existing no-context hist14/local3/grouped OGA-IQA baseline; it isolates focal and is not an external SOTA comparison. Each architecture is evaluated with its own resolved config.

- Final decode metadata uses the actual configured local peak mode and IQA-dependent alpha/quality semantics. Two protocol regressions and one real IQA-off training/checkpoint/evaluation integration pass. The latter's RED fixture-capability check failed as intended; GREEN passed in1.90s. The post-metadata CPU gate passed126 cases in14.59s, with3 CUDA cases deliberately deselected only for that CPU gate.
- Final full suite in the existing compatible CUDA/reference environment: **1,554 passed,80 warnings,17 subtests passed in167.14s**, exit0, zero skips/deselections. All29 reference cases,36 existing BEV CUDA cases,18 task58 3D CUDA cases and the new CUDA freeze integration ran. The80 warnings comprise57 existing TorchScript/Python3.14 deprecations and23 expected small-fixture Numba performance warnings. There are74 added regression/integration cases in this batch.
- Preservation:214 pre-batch files,202 unchanged and12 authorized existing source/test/doc changes; no missing files. Model/loss/encoding/target sources, notebook, master config, eleven runtime configs, trainer, augmentation recipes, pinned evaluator/license bytes and earlier test deletions are preserved. Python AST/new-file whitespace, runtime JSON, runbook Python-block compilation/document links and `git diff --check` pass. CLI synthetic evidence and its actual inputs were revalidated against current tools.
- Primary-agent review finds no Blocker/Major issues in this implementation scope; no independent subagent review was performed. Actual Colab assets and full trained baselines, external-algorithm evidence adapters, real candidate AP, Python3.10 and deferred deployment verification remain external or deferred. Trusted JSON records are consistency/freshness evidence rather than authenticated execution.

The [task51 plan](task51_implementation_plan.md) and [Colab runbook](task51_colab_runbook.md) contain exact commands and limits. Ignored evidence: [verification](../../../artifacts/kitti/backbone_audit/batch21/verification.json), [full suite](../../../artifacts/kitti/backbone_audit/batch21/full_suite_final_current.log), [preservation](../../../artifacts/kitti/backbone_audit/batch21/preservation_audit.json), [static checks](../../../artifacts/kitti/backbone_audit/batch21/static_checks.json), [CLI checks](../../../artifacts/kitti/backbone_audit/batch21/cli_synthetic/verification.json) and [source revalidation](../../../artifacts/kitti/backbone_audit/batch21/revalidation.json). Benchmark finalization remains external until actual Colab audit/baseline records are supplied.


## Batch 22: unified one-master-config Colab notebook

The user-authorized notebook follow-up is implemented. The main plan remains **57/58 tasks finally verified**; task51's actual Colab data/trained-comparator freeze remains external. No real KITTI AP, long training, deployment or SOTA result is claimed.

- `configs/config.json` is the only editable source JSON. Inline Python recipes reproduce all eleven materialized configurations and three augmentation profiles. The master, materialized JSON, augmentation files and numerical model/loss/encoding/target/trainer/reference sources are preserved. Generated per-run config/resolved JSON are immutable provenance snapshots, not extra user-maintained configs.
- Notebook defaults to custom hist14/local3/focal, 32-channel grouped Car and Pedestrian/Cyclist heads, OGA/IQA, hybrid GT and local ROI BEV R40. Model/attention/head/quality/augmentation/training/decode controls apply directly. Final 3D requires explicit box mode, positive vertical objective and a distinct trained run; AP3D/APBEV and R11/R40 are kept separate. Corrected one concurrently edited incompatible default metric control without regenerating other notebook contents; documented the exact selectable augmentation strings.
- Complete cells cover current-source preflight, Drive/archive reuse/preparation, train-only GT database, finite asset audit, focused proposal tests, actual structural parameter budget, GPU/worker/precision gates, pinned reference CUDA parity, baseline evidence/protocols, short actual-data smoke, training/strict resume, own-config minimum-loss checkpoint evaluation and JSON/CSV comparison. Optional external completed runs use their own configs; GT preview uses the supported frame-index CLI.
- Development may proceed with clearly pending comparator/matrix gates. Benchmark requires all gates frozen and rechecks current files immediately before training/resume. The default focal ablation rejects differences outside context; broader protocol comparison must be selected explicitly. Missing/incomplete selected checkpoints or missing metrics fail instead of falling back or inventing zeros.
- Actual trainer integration revealed different file SHA for independently serialized identical `torch.save` payloads: destination-specific archive names differ. Shared verification compares the complete selected/retained model, adaptive criterion, optimizer/scheduler/scaler/RNG/config payload when bytes differ; each original file SHA remains recorded for freshness. Model/trainer formulas are unchanged. Local raw-root normalization and an optional reference CLI run label fix actual evaluator routing.
- Added67 regressions/integration cases. Initial unified RED45 failures; helper GREEN42; notebook/control GREEN116; corrected serialization RED2 failures/5 passed, GREEN7; controlled-condition RED12 failures; focused CPU171 passed. Initial assembled GPU gate exposed raw-root/reference-label errors (140 passed/2 failed); after fixes, all three actual-notebook/reference cases passed in47.09s. These execute two tiny GPU epochs and selected-checkpoint evaluation/evidence for both BEV and 3D on synthetic KITTI-format assets, plus sixteen actual pinned-reference CUDA parity scenarios. Synthetic perfect/zero AP is pipeline verification only.
- A first full run was interrupted (exit137). Diagnostic rerun exposed `/tmp` quota exhaustion; only this batch's completed synthetic fixture directories were reversibly moved to ignored artifacts. Final full fixtures and temporary directories use the project disk. The next full run passed1,620 tests but found one missing quoted augmentation-choice documentation assertion; the comment was corrected and the final controls gate passed72, with3 actual-GPU cases deselected only in that focused CPU gate.
- **Final complete GPU/reference suite: 1,621 passed, 17 subtests passed, 80 warnings in216.24s**, exit0, zero skips/deselections. Main backbone+neck660,528, BEV detector846,689, 3D detector883,941 and train-only OGA parameters12 remain unchanged. All notebook cells compile and validate against nbformat with cleared outputs. All eleven recipe outputs and three inline augmentation definitions match the existing JSON. Final preservation audits220 original files:207 unchanged,13 authorized edits, zero missing.
- Primary-agent review finds no Blocker/Major issues within this notebook/workflow scope; no independent subagent review. Actual Colab execution/full datasets/trained baseline evidence, measured AP, Python3.10/Inductor and deferred Jetson/TensorRT remain unverified or deferred. Source availability requires an uploaded complete checkout (`SOURCE_MODE="existing"`) or published commits on the intended branch; no commit/push was performed.

English delivery guide: [unified notebook runbook](notebook_runbook.md); [implementation plan](../2026-10-07-unified-colab-notebook-plan.md). Ignored evidence: [verification](../../../artifacts/kitti/backbone_audit/batch22/verification.json), [final full log](../../../artifacts/kitti/backbone_audit/batch22/verified_full.log), [JUnit](../../../artifacts/kitti/backbone_audit/batch22/verified_full.xml), [static checks](../../../artifacts/kitti/backbone_audit/batch22/static_checks.json), [preservation](../../../artifacts/kitti/backbone_audit/batch22/preservation_audit.json) and [serialization probe](../../../artifacts/kitti/backbone_audit/batch22/checkpoint_serialization_probe.json). Earlier failed/interrupted logs are retained.
