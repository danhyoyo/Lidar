# Milestone A verification: functional BEV candidate

Task52, 2026-10-07, branch `research/mobilepixornext-under1m`. This record describes the implemented BEV scope and measured verification evidence. It does not certify KITTI accuracy, a final matched benchmark protocol, full 3D readiness or deployment.

The selected candidate is hist14 v1 → stride2 stem32 → stages48/96/128 with depths3/4/2 → stride8 focal context → SG-FPN fusion24/output32 → independent Car and Pedestrian/Cyclist heads. Local attention defaults to none; ECA and SimAM are separate options. LiteMLA is absent from this candidate. Backbone including neck has **660,528 parameters**, IQA-enabled grouped heads **186,161**, detector **846,689**; grouped OGA has **12 train-only parameters**. ECA and SimAM raise the backbone count to 660,532 and 660,529 respectively.

## Evidence against requirements

| Requirement | Verified evidence | Limit |
| --- | --- | --- |
| R01 budget | Actual instantiated body/neck/head/criterion counts; full-resolution structural gate | No measured deployment latency |
| R02–R04 encoding | Shared schema, analytic hist14, NumPy/Numba parity, actual hybrid paste/global-scale assembled pipeline | Configured real KITTI assets and GT database audit pending |
| R05 architecture | Depth3/4/2, no LiteMLA, output32, actual module/shape/backward checks | Architectural change has no accuracy evidence |
| R06 independence | Same-cell cross-group targets and branch gradients; masks/owners reserved within each group | One shared regression branch per group cannot represent two boxes at the same cell |
| R07–R08 objectives/state | Supported loss/IQA routing, independent adaptive buffers/parameters, strict semantic/optimizer/scheduler/scaler/RNG restoration | Exact Q-OGA + IQA remains rejected |
| R09 decode | Group-local class/box/quality associations, Gaussian/binary checks, classwise NMS and one global cap | Current output is BEV float32 Nx7 |
| R10 legacy | Frozen outputs/targets/losses and state keys; flat decoder fixtures | Match historical config when reproducing checkpoints |
| R11 GPU | Actual CUDA train/validate/restore, FP32 objectives, FP16/BF16 OGA+IQA, workers0/6, full-resolution inference | CPU polygon rotated NMS fallback; no full-resolution backward/Inductor claim |
| R12 provenance | Config/checkpoint architecture/objective identities, parameter/group/quality reports, split/config/source hashes and candidate policy | Final benchmark freeze/comparator evidence pending task51 |
| R14 focal | Operator/gate/residual tests, finite gradients, FP32 modulation input under AMP | No measured accuracy improvement |
| R15 local attention | Isolated residual ECA/SimAM, exact counts, finite FP32 gate reductions and GPU smoke | Options are not trained comparisons |
| R16 optional neck | Fusion32/detail shapes, gradients, alignment and counts have CPU checks | Main disables both; no dedicated CUDA/AMP evidence for either |

Functional Milestone A has synthetic CPU/GPU evidence. Final comparison readiness remains open under task51; R13 and Milestone B are unfinished. This distinction allows independent vertical implementation without starting the long comparison training run.

## Supported BEV capability matrix

Each row below has CPU and CUDA FP32 checks for focal/local-none, ECA and SimAM. Gaussian baseline/OGA additionally have IQA routing checks; the batch17 AMP CLI evidence specifically covers the main Gaussian OGA+IQA recipe, rather than every row.

| Classification | Objective | IQA | Status |
| --- | --- | --- | --- |
| Gaussian | baseline | on | verified |
| Gaussian | OGA | on | verified |
| Gaussian | UWAG | off | verified |
| Gaussian | GW-QAL | off | verified |
| Gaussian | Q-OGA legacy quality | off | verified |
| Gaussian | Q-OGA exact rotated IoU | off | verified |
| Binary | baseline | on | verified |
| Binary | OGA | on | verified |
| Binary | UWAG | off | verified |

Gaussian baseline/OGA IQA-off paths also have CPU checks. Unsupported IQA strategies and grouped binary Q-OGA/GW-QAL are rejected by capability validation. Optional fusion32/detail and the no-context/single-head references retain their documented CPU/meta evidence; the dedicated batch17 CUDA smoke covers only the three focal local-attention variants.

## Actual commands and results

Use `/home/duyennh/miniconda3/envs/AI_env/bin/python` in the recorded environment, with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Python3.14.4, Torch2.11.0, Torch CUDA13.0, RTX4050 Laptop GPU. Python3.10 is unavailable here. CUDA and worker IPC require the permitted execution environment outside the filesystem sandbox.

```bash
python -m pytest tests/test_grouped_cuda.py -q --tb=short
python -m pytest tests/test_benchmark_protocol.py tests/test_smoke_detector.py -q --tb=short
python -m pytest tests/ -q --tb=short
python tools/benchmarks/smoke_detector.py --config configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa.json --device cuda --precisions fp32 fp16 bf16 --steps 3 --num-workers 6 --full-resolution --output /tmp/focal_gpu_smoke.json
```

Batch17: CUDA **36 passed in 28.32s**, targeted **28 passed in 5.48s**, full suite **1,333 passed and 17 subtests passed in 104.59s**. No skips. The 57 existing TorchScript/Python3.14 deprecation warnings remain; numerical checks pass. Six actual CLI runs cover none/ECA/SimAM × workers0/6: 18 precision runs, 54 successful updates and full-resolution inference. Each FP16 run records five initial GradScaler skips before three finite successful updates; FP32/BF16 skip none. Peak allocated CUDA memory is 110,057,984–139,584,512 bytes for this synthetic procedure, not a full training memory benchmark.

CPU assembled tests use actual train-only hybrid database paste, hist14, fixed global scaling, targets, criterion, optimizer, strict saved resume and decode. CUDA smoke uses disjoint temporary IDs, a crop, batch2 and disabled augmentation. Persistent-worker stochastic replay is not certified; workers0 RNG resume has separate numerical equivalence evidence. Reloaded GPU inference predictions match exactly and validation leaves criterion state unchanged. Rotated NMS falls back to CPU; CUDA NMS and Inductor remain unverified.

Evidence: [batch17 verification](../../../artifacts/kitti/backbone_audit/batch17/verification.json), [GPU audit](../../../artifacts/kitti/backbone_audit/batch17/gpu_audit.json), [full log](../../../artifacts/kitti/backbone_audit/batch17/full_suite_final.log), [execution log](execution_progress.md). Earlier batch evidence is indexed in the execution log.

## Protocol and next milestone

Current evaluation is local, configured-ROI **APBEV R40**. The user-selected primary benchmark is reference **AP3D R11 and R40**, with an explicit reference APBEV mode planned for task57. See [candidate protocol](benchmark_protocol.md) for split hashes, seed42, hybrid recipe, schedule, quality semantics, thresholds and minimum-validation-loss checkpoint selection. Full benchmark-domain reference evaluation, actual configured assets/train-only database audit, comparator protocols and a reproduced baseline remain pending. No AP/SOTA result is established.

Tasks53–58 add bottom-z/log-height prediction and owner-consistent targets, explicitly weighted loss, calibrated 3D decode and pinned reference evaluation before the final comparison recipe. Until each consumer supports the vertical contract, it must reject it clearly. BEV IQA retains its existing meaning.

Support-conditioned context, dense stages, depthwise stem, new reparameterization and distillation stay outside this release. Jetson/TensorRT/INT8 tuning is deferred; grouped/IQA export currently rejects unsupported deployment contracts. ECA, SimAM, fusion32 and detail remain separately named choices rather than additions to the main recipe.
