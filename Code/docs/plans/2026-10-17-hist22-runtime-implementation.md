# Hist22 Runtime and Performance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Triển khai Hist22 đúng spec, tích hợp train/evaluate/notebook và chuẩn bị cặp Hist14↔Hist22, tối ưu encoder để giảm overhead so với Hist14 bằng số đo thực tế.

**Architecture:** Giữ 14 kênh Hist14 và nối 8 kênh support/dimensionality sau augmentation. NumPy là reference, Numba là backend tối ưu được chọn tường minh, không đổi feature để lấy tốc độ. Schema và checkpoint phân biệt Hist22; paired random initialization giữ các trọng số chung giống baseline và zero 8 lát stem mới.

**Tech Stack:** Python trong `/home/duyennh/miniconda3/envs/AI_env/bin/python`, NumPy, Numba (đã có), PyTorch, pytest; SciPy chỉ dùng nếu benchmark chứng minh có lợi. Không thêm dependency bắt buộc khi thư viện hiện có đủ.

---

## Execution contract

- Spec: `Code/docs/superpowers/specs/2026-10-17-hist22-canupo-bev-design.md`.
- User cập nhật ưu tiên tốc độ: triển khai cả `numpy` reference và `numba` optimized ngay đợt này; cập nhật spec tương ứng. Default schema vẫn NumPy, notebook chọn Numba tường minh như Hist14. Không fallback khi thiếu Numba.
- Dùng skill `executing-plans` với checkpoint review sau mỗi task; các subagent triển khai dùng **gpt-6.1-sol / high**, parent đọc diff và tự chạy kiểm chứng. User đã chọn cách thực thi, không hỏi lại giữa các batch.
- Tiếp tục branch `feature/canupo-inspired-bev22`; workspace chứa thay đổi người dùng chưa commit. Không stash/reset, không commit/push hoặc stage các thay đổi đó. Không tách worktree làm mất context hiện có.
- Mỗi task: viết test có ý nghĩa, chạy và thấy fail vì chức năng còn thiếu; triển khai; chạy targeted tests; review. Ghi bằng chứng red/green vào progress cuối tài liệu.
- Commands dưới đây chạy từ `Code/`, `PY=/home/duyennh/miniconda3/envs/AI_env/bin/python`. Đặt cache/compiler/temp trong workspace hoặc `/tmp` khi sandbox yêu cầu.
- GPU hiện không khả dụng. CPU được dùng cho numerical parity, full-grid forward/backward/restore và encoding benchmark. Không trình bày smoke hoặc Point AP như detection AP sau full training.

## Task 1: Schema, reference, compiled encoder and initial speed gate

**Files:**
- Modify: `Code/detector/core/bev_encoding.py`.
- Create: `Code/detector/core/datasets/utils_1/hist22_backend.py`.
- Modify: `Code/detector/core/datasets/utils_1/preprocess.py`.
- Create: `Code/tests/test_hist22_encoding.py`.
- Modify: `Code/tools/benchmarks/benchmark_bev_encodings.py`.

**Step 1 — Add failing contract/numerical tests.**

Resolve strict version 1/out_channels 22 and only `numpy`/`numba`; reject unknown options, bool/float widths, unsupported backend, XY resolution other than 0.1. Metadata includes base Hist14 policy and all fixed geometry definitions, backend excluded. Snapshot legacy hashes before editing.

```python
schema = resolve_bev_encoding({"name": "hist22", "backend": "numpy"}, geometry)
actual = encode_hist22_chw(points, schema)
assert actual.shape == (22, schema.grid_shape[1], schema.grid_shape[0])
np.testing.assert_array_equal(actual[:14], encode_hist14_chw(points, hist14_numpy_schema))
assert actual.dtype == np.float32 and actual.flags.c_contiguous
```

Test brute-force counts/covariance (not just the prototype), clipped edges, 11/21-cell windows, raw log-count, occupied-only output, line/plane/volume, sparse/degenerate/empty/nonfinite, near-ROI/index ties and noncontiguous inputs. Test NumPy/Numba parity and explicit missing dependency in fresh process, even empty cloud.

**Step 2 — Verify RED.**

Run: `$PY -m pytest tests/test_hist22_encoding.py -q`.
Expected: missing Hist22 schema/encoder failure before implementation; retain concise log.

**Step 3 — Implement reference and optimized backend.**

- NumPy: reuse exact Hist14 reference values, float64 centered XYZ moments, clipped box pooling and `eigvalsh`; allocate independently owned CHW output without generating Rich8/Hist18.
- Numba: compiled loops with `cache=True, nogil=True, fastmath=False`; avoid per-point Python, repeated casts/filter/index and dense HWC copies. Fuse accumulation where useful, restrict geometry/eigen work to occupied cells, reuse bounded scratch memory within one call. No mutable shared workspace across DataLoader workers.
- Benchmark integral tables versus separable/running sums if bandwidth dominates. A stable analytic symmetric-3×3 eigen kernel is allowed only with brute-force/eigvalsh parity and explicit treatment of repeated eigenvalues; no approximation, reduced precision, neighbor cap or downsample.
- Preserve NumPy first-14 bitwise; compiled first-14 use Hist14's verified tolerance (`atol=1e-6, rtol=1e-5`). Establish and record geometry tolerances from independent tests; investigate degenerate cases rather than hiding them with broad tolerances.
- Public `encode_bev` retains contiguous HWC compatibility; Dataset will use CHW directly.

**Step 4 — Verify GREEN and initial benchmark.**

Run: `$PY -m pytest tests/test_hist22_encoding.py tests/test_hist14_encoding.py tests/test_hist14_backend.py tests/test_bev_encoding_spec.py tests/test_bev_geometry_maps.py -q`.
Expected: pass without altering legacy identities.

Run benchmark CLI in CHW mode, same geometry/points/thread settings, separately report first-call compile/cache and warmed p50/p95. Include `hist14_numpy`, `hist14_numba`, `hist22_numpy`, `hist22_numba`. Real KITTI clouds required in final performance gate; synthetic is an early diagnostic only.

**Step 5 — Parent review checkpoint.**

Read actual files and numeric/performance results. Resolve numerical errors first, then profile hot stages before accepting optimized backend. Aim warmed Hist22 ≤2× Hist14 Numba as an optimization target, not an unsupported promise; if missed, perform another measured optimization pass and report actual ratio/overhead.

## Task 2: Dataset, model, paired initialization and training provenance

**Files:**
- Modify: `Code/detector/core/datasets/dataset.py`.
- Modify: `Code/tools/kitti_training_pipeline/common.py`, `train.py` (preserve existing user edits).
- Create: `Code/tools/kitti_training_pipeline/paired_initialization.py`.
- Create: `Code/tests/test_hist22_integration.py`, `Code/tests/test_paired_initialization.py`.
- Existing model files change only if schema-driven input width is insufficient.

**Step 1 — Add failing integration tests.**

Use real Dataset augmentation with train-only GT sampling and scaling/removal; compare scene-derived Hist22 before/after two different transforms. Check CHW contiguous tensor, correct target dimensions at 0.1 m and no cached pre-augmentation geometry. Include single and grouped/3D paths without changing their architecture.

For initialization, build untrained Hist14 state using seed 42 inside a forked RNG context. Copy matching tensors and the first 14 stem input slices to Hist22; zero remaining slices. Validate all keys/shapes and return source state digest/provenance. Never allow trained checkpoint transfer or partial permissive loading.

**Step 2 — Verify RED.**

Run: `$PY -m pytest tests/test_hist22_integration.py tests/test_paired_initialization.py -q`.
Expected: missing integration/initialization failures.

**Step 3 — Implement.**

- Dataset dispatch schema/grid and direct Hist22 CHW after existing augmentation.
- Opt-in initialization config (`mode=hist14_hist22_paired_random`, seed and scheme version) implemented in a dedicated helper called by train before optimizer construction. Both Hist14 and Hist22 must use it; legacy construction unchanged when absent.
- Isolate model-construction RNG and reset data RNG consistently after initialization; preserve separate DataLoader generator and restore semantics. Validation RNG must not become coupled to encoder shape.
- Record deterministic source hash, mode/version/seed in resolved config/checkpoints/report; include initialization policy in training identity without changing legacy identity when omitted. Resume rebuilds same policy then restores same-schema weights; reject incompatible or missing policy metadata.
- Run names contain `hist22`; no implicit cross-schema padding/slicing in normal load/resume/evaluate.
- Deployment consumer that cannot provide Hist22 preprocessing must fail clearly; retain current grouped/IQA/3D guards.

**Step 4 — Verify GREEN.**

Run new tests and relevant existing checkpoint/resume/model suites discovered with `rg --files tests`. Require initial eval output parity, finite nonzero gradients in new stem slices, an actual optimizer step, same state for all shared tensors, unchanged caller/data RNG, checkpoint roundtrip and mismatch rejection.

**Step 5 — Parent review checkpoint.**

Parent reads diff against pre-task working files, independently runs integration tests and verifies no unrelated user edits lost. Confirm no default model/loss/attention change.

## Task 3: Notebook selection and runnable paired ablation

**Files:**
- Modify: `Code/tools/kitti_training_pipeline/notebook_config.py`, `notebook_workflow.py` only as needed.
- Modify: `Code/3D_Lidar_Object_Detection_Notebook_standard.ipynb`, `Code/3D_Lidar_Detection_Pipeline_Robust.ipynb`, `Code/tools/generate_robust_notebook.py`.
- Create: `Code/tools/kitti_training_pipeline/hist22_ablation.py`.
- Create: `Code/tests/test_hist22_ablation.py`, `Code/tests/test_hist22_notebook.py`.
- Modify: `Code/tools/kitti_training_pipeline/README.md` (preserve user edits).

**Step 1 — Add failing tests.**

Notebook supports Hist22/22ch/Numba and prints effective backend, fixed windows and schema hash. Switching recipes replaces stale encoding options instead of deep-merging wrong channels/backend/pillar-only keys; explicit invalid options still fail. Both notebooks remain valid JSON and executable setup/config cells; generator must reproduce Robust changes.

**Step 2 — Verify RED.**

Run: `$PY -m pytest tests/test_hist22_ablation.py tests/test_hist22_notebook.py -q`.

**Step 3 — Implement ablation CLI and notebooks.**

CLI takes a resolved user baseline config, prepares exactly A Hist14 and B Hist22, preserves model/loss/attention/split/optimizer/augmentation/budget/selector, and assigns the same paired-init policy. Build fresh encoding dictionaries, deterministic config fingerprints/run names, separate output dirs and manifest of allowed config differences. No new Hist18 encoder or head/loss recipe.

Provide prepare-only default with concrete train/evaluation commands and explicit execute flag if needed. Never overwrite existing artifacts or silently resume. Validate baseline source is Hist14 (or fail clearly) and both XY resolutions before producing runs. Keep default config and historical presets unchanged. AP report template marks values pending until full training, never fills in Point AP/probe values.

**Step 4 — Verify GREEN.**

Run new notebook/ablation tests plus existing simple/robust/unified/under1m controls and checkpoint selection tests. Prepare a pair from the actual standard notebook's resolved config, inspect diff and print runnable commands.

**Step 5 — Parent review checkpoint.**

Parent independently executes config cells/CLI dry prepare, audits allowed config diffs and validates notebook command output. Document full detector training separately from implementation verification; no invented AP.

## Task 4: Optimize, benchmark real data and full-grid smoke

**Files:**
- Modify: runtime backend only if profile justifies changes (new regression tests for semantic fixes).
- Create: `Code/tools/benchmarks/benchmark_hist22_pipeline.py` if existing benchmark cannot cover model stages.
- Create: `Code/docs/hist22_performance_vi.md`.
- Artifacts: ignored `artifacts/hist22_runtime_20261017/` at workspace root.

**Step 1 — Benchmark real KITTI.**

Locate existing KITTI raw/processed clouds from analysis manifests; use a fixed list covering sparse/dense/showcase and random train/val frames, record IDs/hashes, source hashes, CPU/library/thread versions, layout, warmup/iterations. Preload outside encoding timing. Interleave or randomize paired runs to reduce drift; report p50/p95, ratio/absolute overhead, output bytes, peak process RSS including scratch and compile time separately. Do not compare old prototype wall time with a different machine as a speedup claim.

**Step 2 — Profile and optimize based on evidence.**

If ≥2× baseline or hot stage large, profile accumulation/pooling/eigensolver/output allocation, perform another optimization experiment, keep the fastest numerically verified approach. Record unsuccessful tradeoffs briefly. No global thread oversubscription inside DataLoader; warm/cache JIT appropriately and measure cold-start separately.

**Step 3 — Actual full-grid pipeline smoke.**

At H=800,W=704, batch1 CPU FP32, same user model config: encode, transfer-to-device, eval forward, decode/NMS; then actual criterion/backward/optimizer on a training sample; restore checkpoint and compare outputs. Measure model and end-to-end stages separately and report memory. Use grouped/3D tests when supported by existing paths. CUDA-specific claims remain unverified without a usable GPU.

**Step 4 — Full verification and independent review.**

Run targeted suites then full `tests/` once. Investigate failures against pre-existing state; do not overwrite unrelated features to pass tests. Parent reads implementation and tests for scope, identity, numerical stability, init/RNG and export safety; a separate read-only reviewer may be dispatched with the requested model. Fix Major/Blocker findings and rerun affected checks.

**Step 5 — Deliver.**

Update plan with checked tasks, commands/results and remaining research measurements. Report actual warmed Hist14↔Hist22 speed, overhead, parity/test counts, concrete pair configs/commands and local artifact links. Full detector AP is produced only by full paired training and remains explicitly pending when no full run was performed. Leave branch for user review; no automatic merge/push or staged-user-change commit.

## Progress / evidence

- [x] Inspected current spec, dirty branch, runtime dispatch, schema, training RNG, notebook config and available libraries.
- [x] Plan written; user speed request resolves old NumPy-only restriction by adding explicit Numba backend.
- [x] Task 1 implementation + parent review.
- [x] Task 2 implementation + parent review.
- [x] Task 3 implementation + parent review.
- [ ] Task 4 profiling, full-grid smoke, full tests and final review.

### Initial review and baseline evidence

- Actual environment: AI_env Python 3.14.4, NumPy 2.4.6, Numba 0.67.0, PyTorch 2.11.0, SciPy 1.18.0; CUDA unavailable. Set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid an unrelated broken system ROS pytest plugin (`osrf_pycommon` missing).
- Before integration: `tests/test_checkpoint_selection.py tests/test_grouped_checkpoint.py tests/test_simple_notebook_workflow.py tests/test_cpu_rotated_nms.py`: **122 passed**, 22.01 s, CPU/thread counts 1.
- Preimplementation Hist14 benchmark: four actual KITTI showcase clouds; direct CHW; 3 warmups/20 iterations; no I/O in timing. Numba p50 2.115–2.825 ms, NumPy p50 7.148–8.621 ms. Artifact: `artifacts/hist22_runtime_20261017/hist14_before.json` (workspace root). Final comparison must remeasure both backends together.
- Read-only audit before Task 2: raw state-dict resume/evaluate currently bypasses identity for `legacy_single`; disallow this specifically for Hist22/paired mode. Validate paired policy/provenance before modifying live state. Reject paired `--warm-start` and conflicting seed overrides. Paired construction must copy every shared tensor/buffer and isolate Python/NumPy/Torch RNG, with separate validation generator and correct final post-warmup resume restore.
- Read-only audit before Task 3: encoding replacement must cover custom/preset/file/runtime layers, replacing the dictionary only when name changes while preserving strict validation of invalid newly supplied options. Actual notebook controls are authoritative; do not assume a historical preset is the current user baseline.
- Deployment review: defer Hist22 external deployment explicitly through shared guard; preserve existing packed/grouped/IQA/3D restrictions. A 22-channel graph alone does not prove an engine's preprocessing semantics.

### Task 1 checkpoint

- RED: 24 failures/8 passes before schema/encoder implementation; saved `hist22_red.log` in artifact directory.
- Retained backend: fused compact moments/Hist14 accumulation, rolling vertical windows, sparse horizontal prefix over occupied columns, unrolled ten-stat arithmetic, one-cos analytic symmetric eigenvalues, exact local two-pass covariance for near-degenerate moments, and independently allocated NumPy CHW output passed into the compiled kernel. No fastmath or point/neighbor caps.
- NumPy first 14 channels remain bitwise Hist14. Geometry tolerance `atol=2e-6, rtol=1e-5`; analytic repeated-root/eigen tests use `atol=2e-7`.
- Parent independently ran the five encoder/schema/prototype suites: **130 passed in 16.28 s**. Parent checked four saved full-resolution visualization tensors against Numba runtime: occupancy/valid masks exact, maximum absolute feature error 2.3842e-7. Evidence `parent_real_parity.json`.
- Parent compared seven legacy metadata/hash cases against saved preimplementation code: exact match. Evidence `legacy_schema_identity.json`.
- Worker paired real benchmark, threads1, 30 warmed iterations: Hist14 p50 2.418–3.293 ms, Hist22 6.246–10.758 ms; ratios 2.58–3.27×. ≤2× target missed, explicitly retained as a limitation after measured dense/SAT, sparse, strip, allocation, eigen and unroll experiments. Initial Hist22 was 45–50 ms. Final gate must remeasure both paths on a larger fixed frame list.
- Evidence copied under ignored workspace-root `artifacts/hist22_runtime_20261017/`. No user changes overwritten or commits/staging performed.

### Task 2 checkpoint

- RED: missing paired helper at collection and six Dataset layout failures before implementation.
- Stable helper API: `paired_policy(config)`, `build_initialized_model(config) -> (model, provenance)`, `state_dict_digest(state)`, `validate_paired_provenance(checkpoint, config, model=None)`.
- Opt-in config: `initialization={mode: hist14_hist22_paired_random, version: 1, seed: 42}`. Source provenance is stored in `config.initialization_provenance`, checkpoint top-level and the initialized model attribute. It includes canonical source-state/schema hashes and actual input-weight key.
- Parent review caught and worker fixed two regressions before acceptance: metadata paths importing Torch eagerly, and a validation-generator change affecting legacy runs. Metadata now remains pure under package/standalone imports; separate validation RNG/reset applies only to paired mode.
- New tests cover actual paste/removal/scaling, three target modes, source/shared-state equality, all eight new slice gradients/updates, raw checkpoint rejection, atomic preflight failures, actual CLI fresh training/resume and width-dependent fake compile warmup. Legacy construction/RNG and raw Hist14 weights retain their old behavior.
- Worker broader gate: 272 passed; expanded new suites: 45 passed. Parent independent gate over new integration/paired tests plus grouped checkpoint/training/schema/AP: **179 passed in 19.56 s** (before six final additional negative tests; final full suite must include those). Parent also executed pure metadata with Torch blocked for Hist14/Hist22 and 60 existing under1m notebook tests.
- Targeted notebook tests need the normal detector import paths when run in isolation: `PYTHONPATH=.:detector:detector/core/datasets:tools/kitti_training_pipeline`; otherwise two existing RC tests rely on prior collection side effects. No production change was made for this test invocation detail.
- Hist22 deployment is explicitly deferred through the shared guard; existing deployment restrictions remain in place.

### Compatibility and independent review checkpoint

- Parent checked Numba documentation: literal unrolling requires a compile-time tuple/list. Changed previous `range(10)` usage to a constant tuple in small inline helpers; direct whole-kernel tuple unrolling exposed a compiler SSA error, so that rejected form was not retained. Current Numba 0.67 verified; older versions not locally available.
- Numerical gate after the documented tuple-helper change: 130 passed. Interleaved before/after timing showed <1% p50 differences (noise), no claimed additional speedup. Evidence `hist22_tuple_final_green.log`, `hist22_tuple_real_benchmark.json`.
- Independent read-only review ran all three new core suites: 94 passed. No Critical finding; one Important cold-thread initialization race reproduced with eight simultaneous first calls (7 failures/1 success). Fixed with double-checked dispatcher creation lock, preserved Python helper references and safe publication; hot calls acquire no lock. New cold-thread and failed-init/retry regressions pass; complete encoder gate now 132 passed. Evidence `hist22_thread_red.log`, `hist22_thread_final_green.log`; independent re-review is pending.

### Task 3 checkpoint

- RED: 24 failed / 12 passed. Worker GREEN: new 37 passed, broad notebook/checkpoint 282 passed / 3 skipped. Parent independent notebook/config/selection gate: **118 passed in 1.23 s**.
- Parent inspected actual notebook/generator diffs and CLI: current architecture/loss/attention preserved; stale encoding options replaced at every merge layer, explicit invalid new options still rejected, resolved backend/hash/windows printed. Standard fresh-kernel runtime controls now default from master while preserving supplied globals. Generator was synchronized to existing user notebook edits before regeneration.
- Actual notebook-derived pair is prepared at workspace-root `artifacts/hist22_runtime_20261017/ablation_pair/`: source snapshot, two configs, manifest, quoted commands and AP-pending template. No training executed. Source provenance includes notebook/master hashes and local path assumptions; runtime is master 50/4, seed42, batch16, BF16, six workers and Numba.
- A: `hist14_base_s42_41493c17d514dd81f7f5`; B: `hist22_geom_s42_63155e6383afe1eb99d9`. Only differing leaves: `data.bev_encoding.name`, `data.bev_encoding.out_channels`, `experiment.name`.
- Configured `Code/data/kitti/raw` and `processed` folders are empty; original raw clouds/labels/calibrations exist in separated `Code/Kitti_Raw` extracts. Full dataset preparation/GT database and full detector AP remain research-run prerequisites, not falsely reported as completed training.
