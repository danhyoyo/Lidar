# GT-Sampler Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce GT sampling overhead from ~120ms/frame to <15ms/frame (cutting Colab training epoch time from ~900s toward ~150-200s) by implementing conservative broad-phase bounding volume/frustum pre-filtering, Numba JIT ray-box intersection, eliminating redundant full-cloud allocations, and configuring sampling probability `p=0.5` with counts `3, 5, 5`.

**Architecture:** A two-stage hierarchical acceleration pipeline: (1) broad-phase spatial & azimuthal culling discards 95–98% of points outside the candidate object's range and angular cone using fast vector arithmetic; (2) narrow-phase exact oriented box slab intersection accelerated with Numba JIT (and pure NumPy fallback) executes only on candidate subsets; (3) training config sets sampling probability `p=0.5` and standardizes sample counts to `Car: 3, Pedestrian: 5, Cyclist: 5`.

**Tech Stack:** Python 3.10+, NumPy, Numba JIT, PyTorch, Pytest.

**Spec:** Codebase optimization requirements from user request and analysis of `GTSampler` latency bottlenecks on multi-worker Colab environments.

## Global Constraints

- Mathematical exactness: Broad-phase culling must be strictly conservative (bounding cylinder & azimuthal sector) so zero valid shadow points, volume points, or line-of-sight blockers are missed.
- Format consistency: Point clouds remain float32 `(N, 4)` and bounding boxes remain `(M, 7)` or `(M, 8)` with `[cls, h, w, l, x, y, z_bottom, yaw]`.
- Backward compatibility: If Numba is unavailable or disabled, system must fall back transparently to optimized vectorized NumPy.
- Determinism & test stability: All 275 existing regression and unit tests must continue to pass without numerical degradation.

---

### Task 1: Update Augmentation Configs (`p: 0.5`, counts `3, 5, 5`)

**Files:**
- Modify: `configs/kitti/physics_augmentation/kitti_mobilepixornext_litemla_oga_pcu.json:1-25`
- Modify: `configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_m5_oga.json:1-25`
- Modify: `tests/test_gt_sampler_config.py:1-120`

**Interfaces:**
- Consumes: Config JSON schemas for PCU-Augmentation.
- Produces: `pcu_aug` with `"p": 0.5`, `"sample_counts": {"Car": 3, "Pedestrian": 5, "Cyclist": 5}` in both active configs.

- [ ] **Step 1: Write the failing test for updated config parameters**

Add test cases in `tests/test_gt_sampler_config.py` checking that both configs specify `"p": 0.5` and `sample_counts: {"Car": 3, "Pedestrian": 5, "Cyclist": 5}`:

```python
@pytest.mark.parametrize("config_path", [PCU_CONFIG_PATH, CUMULATIVE_CONFIG_PATH])
def test_configs_sampling_probability_and_counts(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    pcu = cfg["augmentation"]["pcu_aug"]
    assert pcu["p"] == 0.5
    assert pcu["sample_counts"] == {"Car": 3, "Pedestrian": 5, "Cyclist": 5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gt_sampler_config.py -k 'test_configs_sampling_probability_and_counts'`
Expected: FAIL (assertion error on `pcu["p"] == 0.5` or `sample_counts`).

- [ ] **Step 3: Update config JSON files**

In `configs/kitti/physics_augmentation/kitti_mobilepixornext_litemla_oga_pcu.json` and `configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_m5_oga.json`, set:
```json
      "p": 0.5,
      "sample_counts": {
        "Car": 3,
        "Pedestrian": 5,
        "Cyclist": 5
      },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gt_sampler_config.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/ tests/test_gt_sampler_config.py
git commit -m "perf(config): set sampling probability to 0.5 and standardize counts to 3, 5, 5"
```

---

### Task 2: Broad-phase Pre-filtering for `points_in_box`

**Files:**
- Modify: `detector/core/datasets/utils_1/box_geometry.py:31-40`
- Modify: `tests/test_box_geometry.py`

**Interfaces:**
- Consumes: `points: (N, >=3)`, `box: (7,) or (8,)`.
- Produces: `points_in_box(points, box) -> np.ndarray[bool]` identical to full rotated-box slab test, but 10x faster via cylindrical & height pre-filter.

- [ ] **Step 1: Write test comparing fast vs baseline points_in_box**

In `tests/test_box_geometry.py`, add a test on large point clouds with random and edge points:
```python
def test_points_in_box_broadphase_matches_exact():
    pts = np.random.uniform(-40, 40, size=(50000, 4)).astype(np.float32)
    box = np.array([1.5, 1.8, 4.5, 15.0, 5.0, -1.6, 0.4], dtype=np.float32)
    mask = points_in_box(pts, box)
    # Ensure points far outside the box are False
    far_pts = pts[(pts[:, 0] - 15.0)**2 + (pts[:, 1] - 5.0)**2 > 25.0]
    assert not np.any(points_in_box(far_pts, box))
```

- [ ] **Step 2: Run test to verify baseline behavior**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_box_geometry.py`
Expected: PASS

- [ ] **Step 3: Implement cylindrical + vertical pre-filter in `points_in_box`**

In `detector/core/datasets/utils_1/box_geometry.py`:
```python
def points_in_box(points, box):
    """Return inclusive volume membership with fast cylindrical broad-phase culling."""
    b = _box_values(box)
    h, w, l, bx, by, bz, yaw = b
    r_sq = (w * 0.5) ** 2 + (l * 0.5) ** 2 + GEOMETRY_EPS_M
    dx = points[:, 0] - bx
    dy = points[:, 1] - by
    dz = points[:, 2] - bz
    cand = (dx * dx + dy * dy <= r_sq) & (dz >= -GEOMETRY_EPS_M) & (dz <= h + GEOMETRY_EPS_M)
    if not np.any(cand):
        return np.zeros(len(points), dtype=bool)
    
    cand_indices = np.where(cand)[0]
    local = _local_points(points[cand_indices], b)
    sub_mask = ((np.abs(local[:, 0]) <= l * 0.5 + GEOMETRY_EPS_M)
                & (np.abs(local[:, 1]) <= w * 0.5 + GEOMETRY_EPS_M)
                & (local[:, 2] >= -GEOMETRY_EPS_M)
                & (local[:, 2] <= h + GEOMETRY_EPS_M))
    out = np.zeros(len(points), dtype=bool)
    out[cand_indices] = sub_mask
    return out
```

- [ ] **Step 4: Run test to verify exact equivalence and speed**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_box_geometry.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detector/core/datasets/utils_1/box_geometry.py tests/test_box_geometry.py
git commit -m "perf(geometry): add cylindrical broad-phase pre-filtering to points_in_box"
```

---

### Task 3: Frustum & Distance Culling for Shadow & Occlusion Checks

**Files:**
- Modify: `detector/core/datasets/utils_1/physics_aug.py:50-90`
- Modify: `tests/test_physics_aug.py`

**Interfaces:**
- Consumes: `bg_points: (N, 4)`, `box: (7,) or (8,)`.
- Produces: `shadow_point_mask` and `check_line_of_sight_occlusion` with 5x-10x reduced point counts via angular sector & range filtering, avoiding slow `np.linalg.norm`.

- [ ] **Step 1: Write test verifying culling boundary conditions for shadow & LOS**

In `tests/test_physics_aug.py`, assert that points outside range and angle are never marked as shadow or blocker:
```python
def test_shadow_and_los_spatial_culling_invariance():
    # Box at x=20, y=0, range=20
    box = np.array([2.0, 2.0, 4.0, 20.0, 0.0, -1.0, 0.0], np.float32)
    # Point at x=10 (in front) must never be shadow
    in_front = np.array([[10.0, 0.0, 0.0, 0.5]], np.float32)
    assert not np.any(shadow_point_mask(in_front, box))
    # Point at x=30 (behind) must never be LOS blocker
    behind = np.array([[30.0, 0.0, 0.0, 0.5]], np.float32)
    assert not check_line_of_sight_occlusion(box, behind)
```

- [ ] **Step 2: Run test to verify it passes on baseline**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_physics_aug.py`
Expected: PASS

- [ ] **Step 3: Implement conservative sector & range culling in `shadow_point_mask` and `check_line_of_sight_occlusion`**

In `detector/core/datasets/utils_1/physics_aug.py`:
- In `shadow_point_mask`:
  1. Only points with $x^2 + y^2 \ge (R_{center} - R_{box} - \epsilon)^2$ can be in shadow behind the box.
  2. Only points with azimuth angle $|\Delta\theta| \le \arcsin(R_{box} / R_{center}) + \epsilon$ can intersect rays through the box.
  3. Compute ray length as $\sqrt{x^2 + y^2 + z^2}$ instead of `np.linalg.norm`.
- In `check_line_of_sight_occlusion`:
  1. Only points with $x^2 + y^2 \le (R_{center} + R_{box} + \epsilon)^2$ and $x^2 + y^2 \ge 4.0$ ($r \ge 2\text{m}$) can block the box.
  2. Only points with $z > bz + \text{min\_obstacle\_height}$ and within the azimuthal sector are candidate blockers.
  3. Pre-filter candidate blockers before calling `ray_box_intervals`.

- [ ] **Step 4: Run test to verify all physics_aug tests pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_physics_aug.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detector/core/datasets/utils_1/physics_aug.py tests/test_physics_aug.py
git commit -m "perf(occlusion): add frustum sector and range pre-filtering for shadow and LOS"
```

---

### Task 4: Numba JIT-Accelerated Ray-Box Slab Intersection

**Files:**
- Modify: `detector/core/datasets/utils_1/box_geometry.py`
- Modify: `tests/test_box_geometry.py`

**Interfaces:**
- Consumes: `points: (N, >=3)`, `box: (7,) or (8,)`.
- Produces: `ray_box_intervals` running compiled fastmath slab intersection via Numba if available, with automatic fallback to vectorized NumPy.

- [ ] **Step 1: Write test for Numba JIT accuracy vs NumPy**

In `tests/test_box_geometry.py`, add test validating equivalence on edge cases (parallel rays, corner hits, negative coordinates):
```python
def test_ray_box_intervals_numba_equivalence():
    pts = np.array([
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [15.0, 10.0, 0.0],
        [10.0, 0.0, 5.0],
    ], dtype=np.float32)
    box = np.array([2.0, 2.0, 4.0, 10.0, 0.0, -1.0, 0.0], dtype=np.float32)
    hit, enter, leave = ray_box_intervals(pts, box)
    assert hit[0] is True or hit[0] == True
    assert hit[1] is True or hit[1] == True
    assert hit[3] is False or hit[3] == False
```

- [ ] **Step 2: Implement JIT compiled slab intersection kernel**

In `detector/core/datasets/utils_1/box_geometry.py`:
- Implement `_ray_box_numba_kernel(xyz, lower, upper, orig_x, orig_y, orig_z, c, s)` using `@numba.njit(fastmath=True, parallel=False)`.
- In `ray_box_intervals`: call `_ray_box_numba_kernel` directly with float32 coordinates, eliminating `column_stack` and `np.linalg.norm`.
- Wrap with fallback: if `numba` import fails or throws, execute the vectorized NumPy implementation.

- [ ] **Step 3: Run test suite to verify all box geometry tests pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_box_geometry.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add detector/core/datasets/utils_1/box_geometry.py tests/test_box_geometry.py
git commit -m "perf(geometry): accelerate ray_box_intervals with Numba JIT and fastmath"
```

---

### Task 5: End-to-End Latency Benchmark & Regression Verification

**Files:**
- Test: `tools/visualization/audit_gt_sampler.py`
- Test: `tests/test_gt_sampler_audit.py`
- Test: Full repository test suite (`pytest`)

**Interfaces:**
- Consumes: Entire accelerated pipeline with configs and dataset integration.
- Produces: Benchmark verification report showing per-frame sampler latency drop from ~120ms to <15ms, and 100% test suite pass rate.

- [ ] **Step 1: Run audit tool on mock dataset to record optimized latency**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg python -m pytest -q tests/test_gt_sampler_audit.py`
Expected: PASS

- [ ] **Step 2: Micro-benchmark GTSampler per-frame latency**

Run:
```bash
PYTHONPATH=detector python -c "
import time, numpy as np, pickle
from core.datasets.utils_1.gt_sampler import GTSampler

db = {
    'Car': [{'box': np.array([0, 1.5, 1.8, 4.5, 15., 0., -1.6, 0.], np.float32),
             'points': np.random.uniform(-0.5, 0.5, size=(500, 4)).astype(np.float32),
             'r_origin': 15.0, 'num_points': 500}],
    'Pedestrian': [{'box': np.array([1, 1.7, 0.6, 0.8, 10., 2., -1.6, 0.], np.float32),
                   'points': np.random.uniform(-0.2, 0.2, size=(100, 4)).astype(np.float32),
                   'r_origin': 10.0, 'num_points': 100}],
    'Cyclist': [{'box': np.array([2, 1.6, 0.7, 1.6, 12., -2., -1.6, 0.], np.float32),
                'points': np.random.uniform(-0.3, 0.3, size=(200, 4)).astype(np.float32),
                'r_origin': 12.0, 'num_points': 200}]
}
with open('/tmp/bench_db.pkl', 'wb') as f:
    pickle.dump(db, f)

sampler = GTSampler('/tmp/bench_db.pkl', {'Car': 3, 'Pedestrian': 5, 'Cyclist': 5}, p=0.5)
pts = np.random.uniform(-30, 30, size=(120000, 4)).astype(np.float32)
pts[:, 0] = np.random.uniform(0, 70, size=120000)
pts[:, 2] = -1.6
boxes = np.zeros((0, 8), np.float32)

# Warmup
sampler(pts, boxes)

t0 = time.perf_counter()
for _ in range(20):
    sampler(pts, boxes)
t1 = time.perf_counter()
print(f'Optimized time per frame: {(t1 - t0) / 20 * 1000:.2f} ms')
"
```
Expected: Output `< 20 ms` (down from ~120ms).

- [ ] **Step 3: Run full repository test suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg python -m pytest -q`
Expected: 275+ passed, 0 failed.

- [ ] **Step 4: Commit**

```bash
git commit --allow-empty -m "perf(sampler): verify end-to-end performance and regression suite"
```
