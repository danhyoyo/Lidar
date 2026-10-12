# Hybrid GT sampler: synthetic CPU benchmark

Measured on 2026-10-06 on an AMD Ryzen 5 7640HS, with Python 3.14.4, NumPy 2.3.5
and Numba available on Linux x86_64. Raw results:
[benchmark JSON](2026-10-06-hybrid-gt-benchmark.json).
This is a sampler throughput measurement; no KITTI accuracy, full training
throughput, GPU timing or MMDetection3D runtime comparison was performed.

## Workload and reproduction

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tools/benchmarks/benchmark_gt_samplers.py \
  --points 120000 --quota 10 --repeats 20 \
  --pcu-reference 9b99d92 --output /tmp/hybrid_gt_benchmark.json
```

Each synthetic scene has 120,000 points, eight original boxes and a quota of ten
additional cars. The database has 32 objects with 500 points each. Objects retain
their source XY/yaw, so the basic variants have comparable placement and all
insert ten objects. Four warmup calls precede 20 timed calls per variant. Each
timed call receives the same original scene; accepted counts are reported.
Initialization, first JIT compilation and first cache population are excluded.
The original sampler still reads object files during each timed call, with the
OS file cache warm; hybrid reads its bounded array cache. PCU keeps its entire
pickle database in RAM, as implemented in its branch.

PCU's source-relative mode uses range scale `(1,1)` and azimuth jitter `0` to
retain source poses. Its sampler uses a different RNG stream and candidate pool
policy. This is not a bit-for-bit identical object sequence or collision-policy
comparison. PCU helpers are loaded read-only from commit `9b99d92`; they are not
copied into the training implementation.

## Results

| Variant | Median ms/scene | p95 ms/scene | Added objects: mean [min,max] |
| --- | ---: | ---: | ---: |
| Current GT sampler | 22.27 | 51.47 | 10 [10,10] |
| Hybrid basic, warm cache | 5.94 | 6.57 | 10 [10,10] |
| Hybrid ground/static/density enabled | 12.27 | 12.65 | 10 [10,10] |
| PCU basic | 48.82 | 60.71 | 10 [10,10] |
| Hybrid all physics | 81.44 | 92.76 | 6.5 [4,8] |
| PCU all physics | 394.32 | 649.67 | 10 [10,10] |

Hybrid basic takes about **3.75× less time** than the current sampler and 8.22×
less than PCU basic in this workload. Ground/static checks raise its cost to
12.27 ms, still about 1.81× faster than the current sampler. The array cache
records 32 disk reads and 208 hits for 32 objects, using 256,000 bytes.

For the first two hybrid variants, visibility protection is explicitly disabled
to compare common basic features. They use zero collision margin, one candidate
per requested insertion and one placement attempt. This differs from the default
`hybrid_gt` training recipe, which uses positive clearance, visibility protection,
source-relative jitter, three candidates per deficit and up to six attempts.
Density and intensity adjustment do no range correction when source and target
ranges are identical. The ground/static row therefore primarily measures those
two checks, rather than an active density thinning workload.

All-physics variants enable additional LOS/shadow/intensity work. Hybrid also
enforces visibility protection, rejecting some candidates; PCU accepts ten.
Their rules and accepted counts differ, so their times **do not establish a fair
same-work speedup ratio**. They demonstrate the extra cost of enabling all rules.
Shadows can be expensive even with JIT ray tests.

## Interpretation

The result supports batched/JIT geometry, bounded caching and staged point
assembly as useful CPU optimizations. It does not establish better mAP or prove
that extra physics helps training. Real KITTI object sizes, collision frequency,
ground density, disk speed, worker count and cache warmup affect results.

Start with the default `hybrid_gt` recipe and inspect preview rejection counters.
For an accuracy comparison, keep split, architecture, loss, training schedule,
seed and evaluation metric fixed between `openpcdet_gt` and `hybrid_gt`. Evaluate
LOS, shadow and intensity separately if the initial hybrid experiment is useful.
