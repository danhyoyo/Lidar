# Hybrid GT augmentation

Use the `feature/hybrid-gt-augmentation` branch and reopen its standard notebook.
In the configuration cell, select:

```python
AUGMENTATION = "hybrid_gt"
EXPERIMENT_TAG = "hybrid_v1"
```

The notebook resolves the recipe from `configs/augmentation/hybrid_gt.json`,
applies `HYBRID_OPTIONS`, then applies `CONFIG_OVERRIDE`. The resolved run config
is printed and saved. You do not need to edit `configs/config.json`. Changing
settings after a run has checkpoints requires a new tag/run name.

## What is combined

| Source | Adopted design |
| --- | --- |
| Existing augmentation | Ordered GT → flip → rotation → scale → translation, class quotas, shuffled pools, deterministic worker RNG and train-split provenance checks |
| MMDetection3D | Batched box collisions and compiled point geometry; implemented independently with SAT/AABB and a streaming union mask |
| PCU branch | Source-relative relocation, bounded placement retries, local ground support, static obstacle and LOS checks, optional shadow/range-density/intensity adjustment, visibility protection |

The hybrid adds a bounded LRU cache of immutable object point arrays. Accepted
objects and scene removal masks are staged, and scene points are concatenated
once. A rejected candidate cannot delete scene points. Preview metadata records
source frame/object IDs, inserted point/box masks and rejection counters.

There is no MMDetection3D/MMCV/CUDA dependency. `GEOMETRY_BACKEND="auto"` uses
Numba when installed and otherwise uses NumPy. Explicit `"numba"` raises an error
if Numba is unavailable. The first call includes JIT compilation time.

## Default recipe

Sampling probability is 0.5. Class quotas are total scene targets, including
existing objects: Car 8, Pedestrian 6, Cyclist 6. They are upper targets, not
guarantees. Collision, ground and visibility rejection can yield fewer objects.
Each class draws up to three times its deficit from its shuffled database pool,
without duplicating an entry in one scene. Each object gets up to six placement
attempts, except `source` mode which gets one.

Placement rotates the source position and yaw together by at most 5° and scales
range by 0.9–1.1. This preserves the observed object's orientation relative to the
sensor better than unrestricted random translation/yaw. Boxes must have centers
inside the configured XY range and are separated by a 0.1 m BEV margin.

Ground support, static obstacle checks, range-density adjustment and visibility
protection are enabled. LOS, shadow masking and intensity adjustment are disabled
by default. Following GT sampling, the existing world flip, rotation ±45°, scale
0.95–1.05 and probabilistic Gaussian translation operate in order.

Edit `HYBRID_OPTIONS` in the notebook to change these settings, for example:

```python
HYBRID_OPTIONS = {
    "PROBABILITY": 0.5,
    "SAMPLE_GROUPS": ["Car:8", "Pedestrian:6", "Cyclist:6"],
    "CACHE_SIZE_MB": 64,
    "GEOMETRY_BACKEND": "auto",
    "PLACEMENT_MODE": "source_relative",
    "ENABLE_GROUND_VALIDATION": True,
    "ENABLE_STATIC_COLLISION": True,
    "ENABLE_LINE_OF_SIGHT": False,
    "ENABLE_SHADOW_MASKING": False,
    "ENABLE_DENSITY_SUBSAMPLE": True,
    "ENABLE_RADIOMETRIC_CALIBRATION": False,
    "ENABLE_VISIBILITY_PROTECTION": True,
}
```

All sampler fields in the JSON recipe can be overridden here, including
`RANGE_SCALE`, `AZIMUTH_JITTER_DEG`, `COLLISION_MARGIN`, `SAMPLE_RATE`,
`MAX_PLACEMENT_ATTEMPTS`, `CANDIDATE_MULTIPLIER`, `MIN_VISIBLE_POINTS` and
`MIN_VISIBLE_RATIO`. `NAME` cannot be changed through `HYBRID_OPTIONS`.
`PLACEMENT_MODE` supports `source`, `source_relative` and `random`.

Cache memory is per worker: 64 MB × 4 workers allows up to 256 MB of cached point
arrays, in addition to metadata and scene tensors. Set `CACHE_SIZE_MB=0` to
disable storage. File size/mtime changes invalidate cached objects. Pickling and
process ownership changes clear warmed caches and sampling cursors, so a parent
preview does not leak its cache or pool order into workers. Cache reuse across
epochs depends on whether the training loader keeps workers alive.

## Prepare and preview

Run the notebook's dataset preparation cell. It builds the database only from
the configured training split. The existing JSON database and object `.bin`
files can be reused; no migration is needed. A database containing frames
outside the current train split is rejected.

Run the optional preview cell after preparation. Set `FRAME_INDEX`, `VIS_SEED`
and `FORCE_SAMPLING`. Forcing sampling affects only preview, and the plot states
that explicitly. Original boxes are blue; inserted boxes and points are green.
The preview saves a PNG in the run directory and prints acceptance/rejection,
removed-point and cache counters. An insertion can still be rejected even in a
forced preview; inspect counters or change frame/seed.

The same preview is available from the command line with a resolved run config:

```bash
python tools/visualization/visualize_gt_sampling.py \
  --config artifacts/MY_RUN/config.json --frame-index 0 --seed 42 \
  --force-sampling --output artifacts/MY_RUN/gt_preview.png
```

## Geometry and limits

Dataset boxes remain `[class,h,w,l,x,y,z_bottom,yaw]`. Database boxes use geometric
center Z; database points are centered by translation and retain source yaw.
Relocation rotates by target yaw minus source yaw and restores bottom/center Z
consistently. Raw PCU pickle databases and MMDetection3D pickle databases have
different conventions and are not directly supported.

These optional physics rules are heuristics, not a LiDAR simulator:

- Ground uses a local fifth percentile with distance-dependent support, assuming
  KITTI sensor coordinates and ground height between -2.5 and -0.8 m. Slopes or
  unusual sensor heights may need this check disabled or adapted.
- Static checks count elevated points in the candidate footprint. LOS counts
  foreground ray blockers. Dense legitimate structures may trigger rejection.
- Shadow masking deletes points whose sensor rays exit the inserted box before
  reaching their endpoints. Visibility protection rejects a placement if an
  original or already inserted object's surviving count falls below its minimum
  or retained ratio; sparse original objects are compared with their own count.
- Density thinning only applies when moving farther away and has a minimum
  point floor. It cannot reconstruct missing surfaces when moving closer.
  Intensity scaling is an approximate optional range model.

Keep the default flags for a first experiment; use preview counters to diagnose
acceptance before adding more rules. The CPU benchmark demonstrates throughput
on a synthetic workload, not accuracy. No KITTI mAP improvement has been measured.
Compare `openpcdet_gt` and `hybrid_gt` with the same split, model, loss, schedule,
seed and evaluation metric before changing other training settings.

Return to `AUGMENTATION="openpcdet_gt"` to use the earlier sampler. Model and
loss code are unchanged on this branch.

References: [MMDetection3D runtime sampler](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/transforms/dbsampler.py),
[PCU implementation at 9b99d92](https://github.com/danhyoyo/Lidar/blob/9b99d92/detector/core/datasets/utils_1/gt_sampler.py).
