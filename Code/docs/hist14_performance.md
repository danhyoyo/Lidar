# Hist14 input preparation performance

Hist14 implementation revision 2 reduces CPU rasterization and prepares contiguous
CHW input directly for `Dataset.encode_input`. Hist14 semantic version remains 1;
channel order, float64 geometry/statistics, float32 features, ROI/bin boundaries,
normalization, and checkpoint encoding identity are unchanged.

## Changes

- Numba uses a cell-to-row lookup and compact occupied-cell accumulators. The
  accumulator has 13 float64 columns; occupancy is derived from cell membership.
  Capacity is bounded by `min(point_count, cell_count)`, and only occupied rows
  are initialized and normalized. Each invocation owns its workspace and output.
- Native float32/float64 clouds are read directly, including strided/read-only
  arrays; each scalar is promoted to float64 before arithmetic. Other supported
  numeric inputs retain the previous float64 conversion.
- A small per-call density lookup reuses the original double-precision log1p
  expression for repeated integer counts. Large counts use the original formula.
- NumPy groups statistics over occupied cells rather than creating and normalizing
  full-grid histogram/statistic arrays.
- `encode_hist14_chw` writes an independent contiguous `[14,Y,X]` array. Dataset
  uses this path, so evaluator `.contiguous()` no longer copies a dense HWC image.
  Public `encode_bev`/`encode_hist14` still return contiguous `[Y,X,14]` arrays.
- Numba keeps `fastmath=False`, caches compilation, and releases the GIL. Output
  and scratch are never reused across calls or shared between workers.
- Hist14 experiment presets and both notebooks select Numba by default. Explicit
  NumPy remains supported; the generic schema's absent-backend default stays NumPy.

## Matched CPU measurement

The follow-up audit uses the original backend snapshot, verified byte-for-byte
against Git HEAD, and the updated backend in the same process. It measures three
real KITTI float32 clouds on the 800x704 grid, with five warmups and 30 iterations
per frame/method. Method order is randomized within each iteration rather than
running separate blocks. Reported values are means across three frame means.
Python 3.14.4, NumPy 2.4.6, PyTorch 2.11.0, Numba 0.67.0, AMD Ryzen 5 7640HS.
Measurements include rasterization,
`torch.from_numpy`, and preparation of contiguous CHW input; exclude file I/O,
H2D, detector forward, decode/NMS, augmentation, AP, and initialization/JIT.

| Numba CPU input preparation | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| 1 PyTorch CPU thread | 47.23 ms | 4.97 ms | 9.50x |
| 6 PyTorch CPU threads | 40.39 ms | 4.61 ms | 8.76x |

At one thread, the separately measured HWC rasterizer averaged 32.99 ms before
and 4.86 ms after; copying a fixed old HWC tensor to contiguous CHW averaged
12.66 ms. The three clouds occupy 14,138 to 22,115 of 563,200 grid cells
(2.5–3.9%). Computing statistics only for these cells and avoiding the dense
layout copy explain the improvement without removing points or channels.

The initial nine-frame benchmark ran separate method blocks and reported Numba
61.05 to 2.54 ms (24.05x), NumPy 61.02 to 8.13 ms, and rich8 7.81 ms at one CPU
thread. Retain those raw results as an initial observation, not a general 24x
claim. Timing varies substantially with allocation/cache state and method order;
the randomized audit above provides a more cautious comparison.

These are local CPU timings, not L4 end-to-end timings. The prior evaluation's
77.39 ms/frame must be measured again on its GPU before assigning it a new FPS.
Input remains 31,539,200 bytes/frame; the physical H2D payload is unchanged.
Performance depends on CPU, allocator, thread count, point density, and geometry.

Ignored follow-up artifacts: `artifacts/hist14_speed/audit.json` and `audit.py`.
The initial artifacts are `comparison.json`, `compare.py`, and
`bev_backend_before.py` in the same directory. The initial JSON includes per-frame hashes, first-call times,
mean/p50/p95, and environment. Baseline includes the old HWC-to-CHW copy; the new
path writes CHW directly. A synthetic uniform cloud is useful for dense stress
testing but is not representative of real KITTI occupied-cell distributions.

Both backends' HWC and direct CHW outputs were compared against their original
backend on 100 evenly spaced real KITTI frames, spanning all 7,481 available
clouds, and seven synthetic clouds (empty, float32/float64, strided, integer,
constant/dense, ROI boundaries, nonfinite inputs, and extreme intensity).
The follow-up audit compares float32 values through their uint32 bit patterns,
including signed zero: zero bit mismatches in every channel. Occupancy also
matches an independently calculated float64 ROI/grid map. This establishes
input parity on those samples; no full-split AP claim is part of this change.

## Matched GPU pipeline measurement

CUDA access outside the sandbox allowed a second measurement on an NVIDIA
GeForce RTX 4050 Laptop GPU. It uses the existing trained NeXt/hist14 R8 checkpoint,
the same model weights/configuration for both paths, FP32, five real KITTI frames,
five warmups and 20 iterations per method/frame. CPU threads: 1. This includes
CPU rasterization/contiguous input, synchronized H2D, detector forward, and
decode/NMS; it excludes file I/O and AP evaluation.

| Stage | Before | After |
| --- | ---: | ---: |
| CPU input preparation | 50.95 ms | 3.03 ms |
| H2D | 2.74 ms | 2.66 ms |
| Model | 6.81 ms | 6.81 ms |
| Decode/NMS | 2.10 ms | 1.97 ms |
| **Input to detections** | **62.60 ms** | **14.46 ms** |

This initial, separately blocked benchmark was **4.33x** faster for the measured
input-to-detections boundary. It has not been rerun with randomized interleaving;
interpret its timing with the same allocation/order limitations as the initial
CPU experiment. Decoded detections were numerically identical on all five frames
(`assert_array_equal`). These five local GPU frames
are not a full KITTI split or an L4 measurement. Raw results, exact checkpoint
hash, and the reproducible script are in ignored
`artifacts/hist14_speed/gpu_comparison.json` and `gpu_compare.py`.

## Usage and reproducibility

Use the optimized compiled encoder in an existing configuration:

```json
"bev_encoding": {
  "name": "hist14",
  "version": 1,
  "backend": "numba",
  "density_norm": 32,
  "intensity_scale": 1
}
```

Existing Numba checkpoints use the new implementation automatically. If resuming
an older **NumPy** training checkpoint with backend changed to Numba, the training
CLI still requires its explicit `--backend-parity-verified` flag. Choose it after
checking parity on your inputs. Keeping the checkpoint's configured backend avoids
this backend switch; NumPy also benefits from the occupied-cell/CHW improvements.

Measure rasterization plus contiguous model input without requiring PyTorch:

```bash
python3 tools/benchmarks/benchmark_bev_encodings.py \
  --pointcloud Kitti_Raw/velodyne/training/velodyne/000000.bin \
  --encodings rich8 hist14_numpy hist14_numba --layout chw \
  --warmup 5 --iterations 30 --output artifacts/hist14_chw.json
```

`--layout hwc` retains the original public-rasterization benchmark scope. JSON
records actual result layout/shape/contiguity separately from semantic HWC shape.
Exclude JIT/import/cache loading and compare the same frames/config/precision
when rerunning `evaluate_kitti_bev.py` for full E2E latency.

Focused checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m pytest tests/test_hist14_backend.py tests/test_hist14_encoding.py \
  tests/test_hist14_hybrid_integration.py tests/test_bev_encoding_spec.py \
  tests/test_under1m_notebook_controls.py -q
```

Coverage includes empty/dense/random/constant/nonfinite clouds, ROI/bin ties,
custom geometry/normalization, invalid schemas, integer/strided/read-only inputs,
large unsaturated counts, concurrent calls, independent output ownership,
contiguous dataset input, actual hybrid augmentation, CLI scope, and checkpoint
semantic identity. Worker transport tests need local socket permissions.

## Verification results

- Focused encoding/schema/notebook checks: **99 passed**.
- Grouped checkpoint/resume checks: **81 passed**.
- Grouped pipeline, including real worker transport outside sandbox: **37 passed**.
- Complete suite outside sandbox, with Agg plotting and disk-backed temporary
  fixtures: **1,754 passed, 19 skipped, 18 failed**, plus 15 passed subtests.
- Fourteen failures reproduce on the unchanged HEAD archive: eleven saved-recipe
  snapshots predate master config's evaluation/AP-selection defaults, two old
  partial trainer smokes inherit AP selection, and one notebook-source assertion
  expects an obsolete literal command.
- Four failures require the separate reference KITTI CUDA evaluator's missing
  compatible Numba CUDA/libNVVM/libdevice stack. Two direct evaluator failures
  reproduce on HEAD; the remaining two fail through the same reference runtime
  dependency in notebook training/parity checks. PyTorch CUDA inference and the
  optimized Numba **CPU** encoder are verified independently above.
- `git diff --check` passes. No new hist14 feature/input/resume regression was
  found. Original logs remain in ignored `artifacts/hist14_speed/`.
