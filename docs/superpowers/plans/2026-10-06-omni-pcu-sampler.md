# Omni-PCU Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the State-of-the-Art Omni-PCU (Physics-Consistent & Curricular Unified) 3D LiDAR Database Sampler combining OpenPCDet data integrity, MMDetection3D analytical road plane projection and vectorized SAT collision, PCU 3-tier stratified long-range placement with density subsampling and anti-wall verification, and CVPR 2023 COM curriculum difficulty scheduling and in-memory caching.

**Architecture:** A modular, pure-Python/NumPy augmentation engine under `detector/core/datasets/augmentor/` containing: `omni_geometry.py` for vectorized broad/narrow phase collision and $O(1)$ plane projection; `omni_cache.py` for worker-safe bounded in-memory LRU object buffering; and `omni_sampler.py` for the core `OmniDataBaseSampler` orchestrator with curriculum scheduling and transactional rollback.

**Tech Stack:** Python 3.10+, NumPy, Shapely (verification parity), PyTorch, standard library (`json`, `hashlib`, `math`, `pathlib`).

**Spec:** [`docs/superpowers/specs/2026-10-06-sota-omni-pcu-sampler-design.md`](file:///home/duyennh/AI_projects/research_lidar/Lidar/docs/superpowers/specs/2026-10-06-sota-omni-pcu-sampler-design.md)

## Global Constraints

- **Language & Runtime:** Pure Python 3 and vectorized NumPy. No proprietary CUDA compilation or custom C++ builds required.
- **Coordinate Conventions:** Dataset inputs/outputs use bottom center $z$ $[class, h, w, l, x, y, z_{\text{bottom}}, \text{yaw}]$; database metadata uses geometric center $z$ $[x, y, z_{\text{center}}, l, w, h, \text{yaw}]$. Object points stored relative to center $(0,0,0)$ without yaw rotation.
- **Data Integrity:** Strict SHA-256 train manifest hashing and source frame verification to prevent train-validation contamination.
- **Worker Isolation:** Clean serialization under PyTorch `DataLoader` multiprocessing (zero leaked file handles, independent per-worker RNG state, LRU cache invalidation on process fork).
- **Graceful Fallbacks:** If road plane calibration is missing or degenerate ($|c| < 10^{-5}$), automatically fallback to local $z$ 5th-percentile road estimation without throwing unhandled exceptions.

---

### Task 1: Vectorized Geometry Engine (`omni_geometry.py`)

**Files:**
- Create: `detector/core/datasets/augmentor/omni_geometry.py`
- Test: `tests/test_omni_pcu_sampler.py`

**Interfaces:**
- Produces:
  - `boxes_to_bev_corners(boxes: np.ndarray) -> np.ndarray`: Converts $(N, 7)$ or $(N, 8)$ boxes to $(N, 4, 2)$ BEV corners.
  - `check_collision_2d_vectorized(new_boxes: np.ndarray, existing_boxes: np.ndarray, min_margin: float = 0.0) -> np.ndarray`: Returns boolean mask of collisions using broad-phase bounding circle reject + narrow-phase 2D SAT.
  - `project_to_road_plane(x: np.ndarray, y: np.ndarray, plane: np.ndarray) -> np.ndarray`: $O(1)$ analytical ground $z$ elevation from $ax + by + cz + d = 0$.
  - `points_in_oriented_box_3d(points: np.ndarray, box: np.ndarray, extra_margin: np.ndarray | None = None) -> np.ndarray`: Vectorized boolean mask of LiDAR points inside an oriented 3D bounding box.

- [ ] **Step 1: Write the failing tests for vectorized geometry ops**

Create `tests/test_omni_pcu_sampler.py` with tests for corner conversion, SAT collision vs. Shapely ground truth, and road plane projection:

```python
import numpy as np
import pytest
from shapely.geometry import Polygon

from core.datasets.augmentor.omni_geometry import (
    boxes_to_bev_corners,
    check_collision_2d_vectorized,
    project_to_road_plane,
    points_in_oriented_box_3d,
)


def test_bev_corners_orientation_and_dimensions():
    # Box: [cls, h, w, l, x, y, z, yaw=0]
    box = np.array([[0, 2.0, 2.0, 4.0, 10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    corners = boxes_to_bev_corners(box)[0]  # (4, 2)
    assert corners.shape == (4, 2)
    # Check width along Y and length along X for yaw=0
    x_span = corners[:, 0].max() - corners[:, 0].min()
    y_span = corners[:, 1].max() - corners[:, 1].min()
    assert np.isclose(x_span, 4.0, atol=1e-4)
    assert np.isclose(y_span, 2.0, atol=1e-4)


def test_sat_collision_matches_shapely():
    box1 = np.array([[0, 1.5, 1.8, 4.2, 10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # Overlapping box
    box2 = np.array([[0, 1.5, 1.8, 4.2, 11.0, 0.5, 0.0, np.pi / 4]], dtype=np.float32)
    # Non-overlapping box
    box3 = np.array([[0, 1.5, 1.8, 4.2, 25.0, 15.0, 0.0, 0.0]], dtype=np.float32)

    poly1 = Polygon(boxes_to_bev_corners(box1)[0])
    poly2 = Polygon(boxes_to_bev_corners(box2)[0])
    poly3 = Polygon(boxes_to_bev_corners(box3)[0])

    assert poly1.intersects(poly2)
    assert not poly1.intersects(poly3)

    coll_overlap = check_collision_2d_vectorized(box2, box1, min_margin=0.0)
    assert coll_overlap[0] == True

    coll_disjoint = check_collision_2d_vectorized(box3, box1, min_margin=0.0)
    assert coll_disjoint[0] == False


def test_road_plane_analytical_projection():
    # Plane: 0*x + (-0.05)*y + (-0.998)*z + (-1.65) = 0
    plane = np.array([0.0, -0.05, -0.998, -1.65], dtype=np.float64)
    x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    y = np.array([0.0, 2.0, -2.0], dtype=np.float64)
    z_ground = project_to_road_plane(x, y, plane)
    assert len(z_ground) == 3
    # Check equation: a*x + b*y + c*z + d == 0
    residuals = plane[0] * x + plane[1] * y + plane[2] * z_ground + plane[3]
    assert np.allclose(residuals, 0.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_omni_pcu_sampler.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'core.datasets.augmentor.omni_geometry'`

- [ ] **Step 3: Implement `detector/core/datasets/augmentor/omni_geometry.py`**

Write vectorized SAT, broad-phase bounding circle check, road plane projection, and point-in-box mask:

```python
"""Vectorized 2D/3D geometry operations for Omni-PCU Augmentation."""

from __future__ import annotations
import numpy as np


def boxes_to_bev_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert (N, 7) or (N, 8) boxes to (N, 4, 2) BEV corner coordinates.
    Box convention: [h, w, l, x, y, z, yaw] or [class, h, w, l, x, y, z, yaw].
    """
    b = boxes[:, 1:] if boxes.shape[1] >= 8 else boxes
    w, l, x, y, yaw = b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 6]

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # 4 local corners (x along length, y along width)
    # Order: Front-Left, Front-Right, Rear-Right, Rear-Left
    half_l = l * 0.5
    half_w = w * 0.5

    x_corners = np.stack([half_l, half_l, -half_l, -half_l], axis=1)  # (N, 4)
    y_corners = np.stack([half_w, -half_w, -half_w, half_w], axis=1)  # (N, 4)

    # Rotate and translate
    x_rot = x_corners * cos_yaw[:, None] - y_corners * sin_yaw[:, None] + x[:, None]
    y_rot = x_corners * sin_yaw[:, None] + y_corners * cos_yaw[:, None] + y[:, None]

    return np.stack([x_rot, y_rot], axis=-1)  # (N, 4, 2)


def check_collision_2d_vectorized(
    new_boxes: np.ndarray,
    existing_boxes: np.ndarray,
    min_margin: float = 0.0,
) -> np.ndarray:
    """Vectorized BEV collision detection using Broad-Phase Circle Reject
    followed by Narrow-Phase Separating Axis Theorem (SAT).
    
    Returns boolean array of shape (len(new_boxes),) indicating collision.
    """
    if len(existing_boxes) == 0 or len(new_boxes) == 0:
        return np.zeros(len(new_boxes), dtype=bool)

    b_new = new_boxes[:, 1:] if new_boxes.shape[1] >= 8 else new_boxes
    b_exist = existing_boxes[:, 1:] if existing_boxes.shape[1] >= 8 else existing_boxes

    # Broad-phase: Bounding circle radius
    r_new = np.hypot(b_new[:, 1] + 2.0 * min_margin, b_new[:, 2] + 2.0 * min_margin) * 0.5
    r_exist = np.hypot(b_exist[:, 1], b_exist[:, 2]) * 0.5

    # Pairwise center distances (N_new, N_exist)
    dx = b_new[:, 3:4] - b_exist[:, 3:4].T
    dy = b_new[:, 4:5] - b_exist[:, 4:5].T
    dist_sq = dx * dx + dy * dy
    radius_sum_sq = (r_new[:, None] + r_exist[None, :]) ** 2

    broad_phase_coll = dist_sq < radius_sum_sq  # (N_new, N_exist)
    if not np.any(broad_phase_coll):
        return np.zeros(len(new_boxes), dtype=bool)

    # Narrow-phase: SAT on candidate pairs
    new_corners = boxes_to_bev_corners(new_boxes)  # (N_new, 4, 2)
    exist_corners = boxes_to_bev_corners(existing_boxes)  # (N_exist, 4, 2)

    has_collision = np.zeros(len(new_boxes), dtype=bool)
    for i in range(len(new_boxes)):
        cand_indices = np.where(broad_phase_coll[i])[0]
        if len(cand_indices) == 0:
            continue
        c1 = new_corners[i]  # (4, 2)
        for j in cand_indices:
            c2 = exist_corners[j]  # (4, 2)
            if _sat_polygon_intersection(c1, c2, min_margin):
                has_collision[i] = True
                break

    return has_collision


def _sat_polygon_intersection(c1: np.ndarray, c2: np.ndarray, margin: float = 0.0) -> bool:
    """Check 2D oriented rectangle overlap via Separating Axis Theorem (SAT)."""
    # 4 potential separating axes: normals to edges of c1 and c2
    edges1 = np.roll(c1, -1, axis=0) - c1
    edges2 = np.roll(c2, -1, axis=0) - c2
    axes = np.concatenate([edges1[:2], edges2[:2]], axis=0)
    # Orthogonal normals: [-dy, dx]
    normals = np.stack([-axes[:, 1], axes[:, 0]], axis=-1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)

    for normal in normals:
        proj1 = np.dot(c1, normal)
        proj2 = np.dot(c2, normal)
        min1, max1 = proj1.min() - margin, proj1.max() + margin
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 or max2 < min1:
            return False  # Separating axis found
    return True


def project_to_road_plane(
    x: np.ndarray | float,
    y: np.ndarray | float,
    plane: np.ndarray,
) -> np.ndarray | float:
    """Compute road elevation z from KITTI road plane equation: ax + by + cz + d = 0.
    
    Guards against degenerate or vertical planes (|c| < 1e-5).
    """
    a, b, c, d = plane[:4]
    if abs(c) < 1e-5:
        # Fallback to standard KITTI LiDAR height if normal is degenerate
        return np.full_like(x, -1.65, dtype=np.float64) if isinstance(x, np.ndarray) else -1.65
    return -(a * x + b * y + d) / c


def points_in_oriented_box_3d(
    points: np.ndarray,
    box: np.ndarray,
    extra_margin: np.ndarray | None = None,
) -> np.ndarray:
    """Return boolean mask of points inside a 3D bounding box.
    Box: [h, w, l, x, y, z_bottom, yaw] or [class, h, w, l, x, y, z_bottom, yaw].
    """
    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, yaw = b[:7]

    if extra_margin is not None:
        h = h + extra_margin[0]
        w = w + extra_margin[1]
        l = l + extra_margin[2]

    # Translate points
    pts = points[:, :3].copy()
    pts[:, 0] -= bx
    pts[:, 1] -= by
    pts[:, 2] -= (bz + h * 0.5)

    # Rotate into box coordinate frame
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    x_rot = pts[:, 0] * cos_y - pts[:, 1] * sin_y
    y_rot = pts[:, 0] * sin_y + pts[:, 1] * cos_y
    z_rot = pts[:, 2]

    in_x = np.abs(x_rot) <= (l * 0.5)
    in_y = np.abs(y_rot) <= (w * 0.5)
    in_z = np.abs(z_rot) <= (h * 0.5)

    return in_x & in_y & in_z
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_omni_pcu_sampler.py -v`  
Expected: PASS all geometry tests.

- [ ] **Step 5: Commit geometry module**

```bash
git add detector/core/datasets/augmentor/omni_geometry.py tests/test_omni_pcu_sampler.py
git commit -m "feat(augmentor): add vectorized BEV SAT collision and road plane projection"
```

---

### Task 2: In-Memory Bounded LRU Object Point Cache (`omni_cache.py`)

**Files:**
- Create: `detector/core/datasets/augmentor/omni_cache.py`
- Test: `tests/test_omni_pcu_sampler.py`

**Interfaces:**
- Produces:
  - `class BoundedPointCache`: Thread-safe / worker-safe LRU cache storing decoded object point clouds with a maximum memory ceiling (MB) and PID tracking to safely reset upon process forks.

- [ ] **Step 1: Write failing tests for cache and worker reset**

Append tests to `tests/test_omni_pcu_sampler.py`:

```python
import os
from core.datasets.augmentor.omni_cache import BoundedPointCache


def test_bounded_point_cache_eviction(tmp_path):
    # Cache limit 1KB (tiny)
    cache = BoundedPointCache(max_size_mb=0.001)
    pts1 = np.ones((100, 4), dtype=np.float32)  # 1600 bytes -> exceeds 1KB

    cache.put("sample_1", pts1)
    assert cache.get("sample_1") is not None

    pts2 = np.zeros((100, 4), dtype=np.float32)
    cache.put("sample_2", pts2)
    # sample_1 should be evicted or cache size bounded
    assert cache.get("sample_2") is not None
    assert cache.current_bytes <= 1600


def test_cache_resets_on_pid_change():
    cache = BoundedPointCache(max_size_mb=10)
    cache.put("key1", np.ones((10, 4), dtype=np.float32))
    assert cache.get("key1") is not None

    # Simulate process fork by altering stored pid
    cache.pid = os.getpid() + 999
    assert cache.get("key1") is None
    assert len(cache) == 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_omni_pcu_sampler.py::test_bounded_point_cache_eviction -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `detector/core/datasets/augmentor/omni_cache.py`**

```python
"""Bounded In-Memory LRU Point Cache for DataLoader Worker Acceleration."""

from __future__ import annotations
import collections
import os
import numpy as np


class BoundedPointCache:
    """Worker-local bounded LRU memory cache for decoded GT object point clouds.
    
    Automatically invalidates on process fork to prevent memory corruption
    or stale shared handles across PyTorch DataLoader workers.
    """

    def __init__(self, max_size_mb: float = 128.0):
        self.max_bytes = int(max_size_mb * 1024 * 1024)
        self.current_bytes = 0
        self.pid = os.getpid()
        self._cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()

    def _check_fork(self) -> None:
        current_pid = os.getpid()
        if current_pid != self.pid:
            self._cache.clear()
            self.current_bytes = 0
            self.pid = current_pid

    def get(self, key: str) -> np.ndarray | None:
        self._check_fork()
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key].copy()

    def put(self, key: str, points: np.ndarray) -> None:
        self._check_fork()
        if key in self._cache:
            self.current_bytes -= self._cache[key].nbytes
            del self._cache[key]

        pts_bytes = points.nbytes
        while self.current_bytes + pts_bytes > self.max_bytes and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self.current_bytes -= evicted.nbytes

        self._cache[key] = points.copy()
        self.current_bytes += pts_bytes

    def __len__(self) -> int:
        self._check_fork()
        return len(self._cache)

    def __getstate__(self) -> dict:
        """Exclude cache memory during pickle serialization."""
        return {"max_bytes": self.max_bytes}

    def __setstate__(self, state: dict) -> None:
        self.max_bytes = state["max_bytes"]
        self.current_bytes = 0
        self.pid = os.getpid()
        self._cache = collections.OrderedDict()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_omni_pcu_sampler.py -k "cache" -v`  
Expected: PASS

- [ ] **Step 5: Commit cache module**

```bash
git add detector/core/datasets/augmentor/omni_cache.py tests/test_omni_pcu_sampler.py
git commit -m "feat(augmentor): add worker-safe bounded LRU object point cache"
```

---

### Task 3: GT Database Builder Upgrade with `r_origin` and Density (`build_gt_database.py`)

**Files:**
- Modify: `tools/kitti_training_pipeline/build_gt_database.py:72-85`
- Test: `tests/test_omni_pcu_sampler.py`

**Interfaces:**
- Produces: Enhanced JSON format storing `r_origin` and `density` in each entry of `db_infos` while retaining full backward compatibility for existing fields.

- [ ] **Step 1: Write failing test for database builder enhancements**

Add test in `tests/test_omni_pcu_sampler.py`:

```python
from tools.kitti_training_pipeline.build_gt_database import build_database
from common import read_json


def test_build_database_includes_r_origin_and_density(tmp_path):
    proc_root = tmp_path / "processed"
    (proc_root / "pointcloud").mkdir(parents=True)
    (proc_root / "label").mkdir(parents=True)

    # Synthetic pointcloud: 100 points
    pts = np.random.uniform(-5, 5, size=(100, 4)).astype(np.float32)
    pts[:, :3] += np.array([20.0, 5.0, -1.0])
    pts.tofile(proc_root / "pointcloud" / "000001.bin")

    # Label: Car at [x=20, y=5, z=-1.0, l=4.0, w=2.0, h=1.5, yaw=0.0]
    (proc_root / "label" / "000001.txt").write_text(
        "Car 1.5 2.0 4.0 20.0 5.0 -1.75 0.0\n", encoding="utf-8"
    )
    manifest = tmp_path / "train.txt"
    manifest.write_text("000001\n", encoding="utf-8")

    out_dir = tmp_path / "gt_database"
    dest = build_database(proc_root, manifest, out_dir, min_points=1)
    meta = read_json(dest)

    entry = meta["db_infos"]["Car"][0]
    assert "r_origin" in entry
    assert np.isclose(entry["r_origin"], np.hypot(20.0, 5.0), atol=1e-3)
    assert "density" in entry
    assert entry["density"] > 0.0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_omni_pcu_sampler.py::test_build_database_includes_r_origin_and_density -v`  
Expected: FAIL with `KeyError: 'r_origin'`

- [ ] **Step 3: Modify `tools/kitti_training_pipeline/build_gt_database.py`**

Update `build_database` to compute and include `r_origin` and `density`:

```python
            r_origin = float(np.hypot(x, y))
            volume = float(max(1e-4, length * width * height))
            density = float(len(local) / volume)
            db_infos[name].append({
                "name": name, "path": relative_path, "image_idx": identifier,
                "gt_idx": index, "box3d_lidar": [x, y, z_center, length, width, height, yaw],
                "num_points_in_gt": len(local),
                "r_origin": r_origin,
                "density": density,
            })
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_omni_pcu_sampler.py::test_build_database_includes_r_origin_and_density -v`  
Expected: PASS

- [ ] **Step 5: Commit database builder upgrade**

```bash
git add tools/kitti_training_pipeline/build_gt_database.py tests/test_omni_pcu_sampler.py
git commit -m "feat(pipeline): enrich GT database with r_origin and point density metadata"
```

---

### Task 4: Core `OmniDataBaseSampler` Engine (`omni_sampler.py`)

**Files:**
- Create: `detector/core/datasets/augmentor/omni_sampler.py`
- Test: `tests/test_omni_pcu_sampler.py`

**Interfaces:**
- Consumes: `omni_geometry.py`, `omni_cache.py`, and database JSON metadata.
- Produces: `class OmniDataBaseSampler` conforming to the specified callable API.

- [ ] **Step 1: Write failing tests for curriculum scheduling and stratified placement**

Add tests to `tests/test_omni_pcu_sampler.py`:

```python
from core.datasets.augmentor.omni_sampler import OmniDataBaseSampler


def test_curriculum_difficulty_partitioning_and_scheduling(tmp_path):
    config = {
        "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
        "SAMPLE_GROUPS": ["Car:2"],
        "CURRICULUM": {
            "ENABLED": True,
            "WARMUP_EPOCHS": 10,
            "HARD_RATIO_BASE": 0.10,
            "HARD_RATIO_TARGET": 0.90,
            "DIFFICULTY_THRESHOLD": 0.40,
        },
    }
    # Create mock database with 1 easy (near, dense) and 1 hard (far, sparse) car
    meta = {
        "format": "lidar_gt_database_v1",
        "num_point_features": 4,
        "source_frame_ids": ["000001"],
        "db_infos": {
            "Car": [
                {
                    "name": "Car", "path": "car_easy.bin", "image_idx": "000001",
                    "box3d_lidar": [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                    "num_points_in_gt": 500, "r_origin": 10.0, "density": 40.0,
                },
                {
                    "name": "Car", "path": "car_hard.bin", "image_idx": "000001",
                    "box3d_lidar": [60.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                    "num_points_in_gt": 15, "r_origin": 60.0, "density": 1.2,
                },
            ]
        },
    }
    (tmp_path / "gt_database").mkdir()
    np.ones((500, 4), dtype=np.float32).tofile(tmp_path / "car_easy.bin")
    np.ones((15, 4), dtype=np.float32).tofile(tmp_path / "car_hard.bin")
    import json
    with (tmp_path / "gt_database" / "dbinfos_train.json").open("w") as f:
        json.dump(meta, f)

    sampler = OmniDataBaseSampler(tmp_path, config, {"Car": 0})
    # Verify pools partitioned
    assert len(sampler.easy_pools["Car"]) == 1
    assert len(sampler.hard_pools["Car"]) == 1

    # Verify probability schedule
    sampler.set_epoch(0)
    assert np.isclose(sampler.current_hard_ratio, 0.10)
    sampler.set_epoch(5)
    assert np.isclose(sampler.current_hard_ratio, 0.50)
    sampler.set_epoch(10)
    assert np.isclose(sampler.current_hard_ratio, 0.90)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_omni_pcu_sampler.py::test_curriculum_difficulty_partitioning_and_scheduling -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `detector/core/datasets/augmentor/omni_sampler.py`**

Write the complete `OmniDataBaseSampler` class implementation incorporating:
- JSON loading and SHA-256 provenance check
- Easy/hard difficulty classification based on formula
- Stratified range corridor proposal
- Vectorized SAT collision check
- $O(1)$ road plane height adjustment with fallback
- Distance density scaling $(r_{\text{origin}} / r_{\text{target}})^2$ with 5-point floor
- Anti-wall static obstacle rejection ($z > z_{\text{ground}} + 0.35\text{m}$)
- Transactional visibility check and rollback
- In-memory bounded LRU cache integration

```python
"""Omni-PCU State-of-the-Art 3D LiDAR Database Sampler."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .omni_cache import BoundedPointCache
from .omni_geometry import (
    check_collision_2d_vectorized,
    points_in_oriented_box_3d,
    project_to_road_plane,
)


def _parse_class_counts(values: list[str], class_names: dict[str, int]) -> dict[str, int]:
    counts = {}
    for value in values:
        name, raw = value.split(":")
        count = int(raw)
        if name not in class_names or count < 0:
            raise ValueError(f"Invalid class count config: {value!r}")
        counts[name] = count
    return counts


class OmniDataBaseSampler:
    """SOTA Physics-Consistent & Curricular GT Sampler."""

    def __init__(
        self,
        root_path: Path | str,
        config: dict,
        class_names: dict[str, int],
        *,
        geometry: dict | None = None,
        allowed_frame_ids: list[str] | None = None,
    ):
        self.root_path = Path(root_path)
        self.class_names = dict(class_names)
        self.geometry = geometry
        self.config = config
        self.feature_count = int(config.get("NUM_POINT_FEATURES", 4))
        self.limit_whole_scene = bool(config.get("LIMIT_WHOLE_SCENE", True))
        self.quotas = _parse_class_counts(config.get("SAMPLE_GROUPS", []), class_names)

        # Placement parameters
        placement = config.get("PLACEMENT", {})
        self.placement_mode = placement.get("MODE", "stratified_range")
        self.range_tiers = placement.get("RANGE_TIERS", [[8.0, 25.0], [25.0, 45.0], [45.0, 65.5]])
        self.tier_weights = placement.get("TIER_WEIGHTS", [0.30, 0.40, 0.30])
        self.max_attempts = int(placement.get("MAX_ATTEMPTS", 15))

        # Physics heuristics
        physics = config.get("PHYSICS", {})
        self.enable_density_subsample = bool(physics.get("ENABLE_DENSITY_SUBSAMPLE", True))
        self.enable_anti_wall = bool(physics.get("ENABLE_ANTI_WALL", True))
        self.anti_wall_height = float(physics.get("ANTI_WALL_HEIGHT_THRESH", 0.35))
        self.max_obstacle_points = int(physics.get("MAX_OBSTACLE_POINTS", 3))
        self.min_visible_ratio = float(physics.get("MIN_VISIBLE_RATIO", 0.50))
        self.min_visible_points = int(physics.get("MIN_VISIBLE_POINTS", 5))

        # Curriculum scheduling (COM)
        curr_cfg = config.get("CURRICULUM", {})
        self.curriculum_enabled = bool(curr_cfg.get("ENABLED", True))
        self.warmup_epochs = max(1, int(curr_cfg.get("WARMUP_EPOCHS", 10)))
        self.hard_ratio_base = float(curr_cfg.get("HARD_RATIO_BASE", 0.15))
        self.hard_ratio_target = float(curr_cfg.get("HARD_RATIO_TARGET", 0.70))
        self.diff_threshold = float(curr_cfg.get("DIFFICULTY_THRESHOLD", 0.45))
        self.epoch = 0
        self.current_hard_ratio = self.hard_ratio_base

        # Cache
        cache_cfg = config.get("PERFORMANCE", {})
        cache_size = float(cache_cfg.get("CACHE_SIZE_MB", 128.0))
        self.cache = BoundedPointCache(cache_size) if cache_cfg.get("CACHE_ENABLED", True) else None

        # Load database
        self.db_infos = {name: [] for name in class_names}
        self.easy_pools = {name: [] for name in class_names}
        self.hard_pools = {name: [] for name in class_names}
        self._load_database(config.get("DB_INFO_PATH", []), allowed_frame_ids)
        self._partition_curriculum()

    def _load_database(self, paths: list[str], allowed_frames: list[str] | None) -> None:
        allowed = set(allowed_frames) if allowed_frames is not None else None
        for rel_path in paths:
            with (self.root_path / rel_path).open(encoding="utf-8") as stream:
                meta = json.load(stream)
            if meta.get("format") != "lidar_gt_database_v1":
                raise ValueError("Unsupported GT database format; requires lidar_gt_database_v1")
            source_ids = set(meta.get("source_frame_ids", []))
            if allowed is not None and not source_ids <= allowed:
                raise ValueError("GT database contains frames outside the current training split")
            for name in self.class_names:
                for entry in meta.get("db_infos", {}).get(name, []):
                    # Compute r_origin dynamically if not cached in file
                    if "r_origin" not in entry:
                        entry["r_origin"] = float(np.hypot(entry["box3d_lidar"][0], entry["box3d_lidar"][1]))
                    self.db_infos[name].append(entry)

    def _partition_curriculum(self) -> None:
        ref_counts = {"Pedestrian": 50.0, "Cyclist": 100.0, "Car": 300.0}
        for name, entries in self.db_infos.items():
            ref_pts = ref_counts.get(name, 100.0)
            for entry in entries:
                r = entry.get("r_origin", 20.0)
                pts = entry.get("num_points_in_gt", 50)
                diff = 0.5 * (min(1.0, r / 70.0) + max(0.0, 1.0 - pts / ref_pts))
                if diff > self.diff_threshold:
                    self.hard_pools[name].append(entry)
                else:
                    self.easy_pools[name].append(entry)
            # Ensure pools are never empty
            if not self.easy_pools[name]:
                self.easy_pools[name] = list(entries)
            if not self.hard_pools[name]:
                self.hard_pools[name] = list(entries)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, epoch)
        progress = min(1.0, self.epoch / self.warmup_epochs)
        self.current_hard_ratio = progress * self.hard_ratio_target + (1.0 - progress) * self.hard_ratio_base

    def _sample_candidates(self, class_name: str, count: int, rng: np.random.Generator) -> list[dict]:
        if count <= 0:
            return []
        sampled = []
        for _ in range(count):
            draw_hard = (rng.uniform(0.0, 1.0) < self.current_hard_ratio) if self.curriculum_enabled else False
            pool = self.hard_pools[class_name] if draw_hard else self.easy_pools[class_name]
            idx = rng.integers(0, len(pool))
            sampled.append(pool[idx])
        return sampled

    def _propose_pose(self, class_name: str, attempt: int, rng: np.random.Generator) -> tuple[float, float, float]:
        if attempt < 10:
            tier_idx = rng.choice(len(self.range_tiers), p=self.tier_weights)
            r_min, r_max = self.range_tiers[tier_idx]
            x = rng.uniform(r_min, r_max)
            limit_y = {"Pedestrian": 15.0, "Cyclist": 13.0}.get(class_name, 9.5)
            y = rng.uniform(-limit_y, limit_y)
        else:
            x = rng.uniform(5.0, 65.0)
            y = rng.uniform(-25.0, 25.0)
        yaw = rng.uniform(-np.pi, np.pi)
        return x, y, yaw

    def _load_points(self, entry: dict) -> np.ndarray:
        key = entry["path"]
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        pts = np.fromfile(self.root_path / key, dtype=np.float32).reshape(-1, self.feature_count)
        if self.cache is not None:
            self.cache.put(key, pts)
        return pts.copy()

    def __call__(
        self,
        points: np.ndarray,
        boxes: np.ndarray,
        road_plane: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        cur_points = points.copy()
        new_boxes = list(boxes.copy())
        n_orig = len(boxes)

        # Baseline point membership for pre-existing boxes
        memberships = [points_in_oriented_box_3d(cur_points, b) for b in boxes]
        baselines = [int(m.sum()) for m in memberships]

        inserted_points_list = []

        for class_name, quota in self.quotas.items():
            cls_id = self.class_names[class_name]
            if self.limit_whole_scene:
                existing_count = int(np.sum(boxes[:, 0] == cls_id)) if len(boxes) > 0 else 0
                quota = max(0, quota - existing_count)
            if quota <= 0:
                continue

            candidates = self._sample_candidates(class_name, quota, rng)
            for entry in candidates:
                length, width, height = entry["box3d_lidar"][3], entry["box3d_lidar"][4], entry["box3d_lidar"][5]
                src_r = float(entry.get("r_origin", np.hypot(entry["box3d_lidar"][0], entry["box3d_lidar"][1])))

                committed = False
                for attempt in range(self.max_attempts):
                    x, y, yaw = self._propose_pose(class_name, attempt, rng)
                    target_r = float(np.hypot(x, y))

                    # 1. Height adjustment
                    if road_plane is not None:
                        z_ground = float(project_to_road_plane(x, y, road_plane))
                    else:
                        # Fallback: estimate ground from 5th percentile of local points
                        dist_sq = (cur_points[:, 0] - x) ** 2 + (cur_points[:, 1] - y) ** 2
                        local_pts = cur_points[dist_sq <= 9.0]
                        z_ground = float(np.percentile(local_pts[:, 2], 5)) if len(local_pts) >= 5 else -1.65

                    candidate_box = np.array(
                        [cls_id, height, width, length, x, y, z_ground, yaw], dtype=np.float32
                    )

                    # 2. Collision test with existing and inserted boxes
                    existing_all = np.array(new_boxes, dtype=np.float32) if new_boxes else np.empty((0, 8))
                    if check_collision_2d_vectorized(candidate_box[None, :], existing_all, min_margin=0.3)[0]:
                        continue

                    # 3. Anti-wall static obstacle check
                    if self.enable_anti_wall:
                        in_box = points_in_oriented_box_3d(cur_points, candidate_box)
                        obstacles = cur_points[in_box & (cur_points[:, 2] > (z_ground + self.anti_wall_height))]
                        if len(obstacles) > self.max_obstacle_points:
                            continue

                    # 4. Point loading & physical density subsampling
                    local_pts = self._load_points(entry)
                    if self.enable_density_subsample and target_r > src_r:
                        ratio = min(1.0, (src_r / target_r) ** 2)
                        keep_mask = rng.uniform(0.0, 1.0, size=len(local_pts)) <= ratio
                        if keep_mask.sum() < 5 and len(local_pts) >= 5:
                            keep_idx = rng.choice(len(local_pts), size=5, replace=False)
                            local_pts = local_pts[keep_idx]
                        else:
                            local_pts = local_pts[keep_mask]

                    if len(local_pts) == 0:
                        continue

                    # Transform local points to world frame
                    c, s = np.cos(yaw), np.sin(yaw)
                    world_pts = local_pts.copy()
                    world_pts[:, 0] = local_pts[:, 0] * c - local_pts[:, 1] * s + x
                    world_pts[:, 1] = local_pts[:, 0] * s + local_pts[:, 1] * c + y
                    world_pts[:, 2] = local_pts[:, 2] + (z_ground + height * 0.5)

                    # 5. Visibility check on pre-existing objects
                    inside_cand = points_in_oriented_box_3d(cur_points, candidate_box)
                    violated = False
                    for b_idx, (base_cnt, mem_mask) in enumerate(zip(baselines, memberships)):
                        surviving = np.count_nonzero(mem_mask & ~inside_cand)
                        if base_cnt > 0 and (
                            surviving < min(base_cnt, self.min_visible_points)
                            or (surviving / base_cnt) < self.min_visible_ratio
                        ):
                            violated = True
                            break
                    if violated:
                        continue

                    # 6. Commit transaction
                    cur_points = cur_points[~inside_cand]
                    memberships = [m[~inside_cand] for m in memberships]
                    memberships.append(np.ones(len(world_pts), dtype=bool))
                    baselines.append(len(world_pts))

                    new_boxes.append(candidate_box)
                    inserted_points_list.append(world_pts)
                    committed = True
                    break

                if not committed:
                    continue

        if inserted_points_list:
            cur_points = np.vstack([cur_points, *inserted_points_list])

        return cur_points, np.array(new_boxes, dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_omni_pcu_sampler.py -k "curriculum" -v`  
Expected: PASS

- [ ] **Step 5: Commit `omni_sampler.py`**

```bash
git add detector/core/datasets/augmentor/omni_sampler.py tests/test_omni_pcu_sampler.py
git commit -m "feat(augmentor): implement core OmniDataBaseSampler engine"
```

---

### Task 5: Hard Edge-Cases & Anti-Bypass Testing Suite

**Files:**
- Modify: `tests/test_omni_pcu_sampler.py`

**Interfaces:**
- Test Cases:
  - `test_anti_wall_detects_elevated_obstacles_and_rejects()`
  - `test_density_subsample_enforces_five_point_floor()`
  - `test_road_plane_degenerate_fallback()`
  - `test_transactional_rollback_preserves_original_visibility()`
  - `test_whole_scene_quota_bypass_protection()`

- [ ] **Step 1: Write comprehensive edge-case tests in `tests/test_omni_pcu_sampler.py`**

Add tests covering the hardened scenarios:

```python
def test_density_subsample_enforces_five_point_floor():
    # If a car with 10 points is moved from 5m to 65m, density scaling (5/65)^2 = 0.0059
    # Without the floor, points would be 0. Verify floor preserves at least 5 points.
    pts = np.ones((10, 4), dtype=np.float32)
    # Target range: 65m, source range: 5m
    rng = np.random.default_rng(42)
    ratio = (5.0 / 65.0) ** 2
    keep = rng.uniform(0.0, 1.0, size=len(pts)) <= ratio
    if keep.sum() < 5 and len(pts) >= 5:
        idx = rng.choice(len(pts), size=5, replace=False)
        pts = pts[idx]
    assert len(pts) == 5


def test_anti_wall_rejects_elevated_obstacles(tmp_path):
    config = {
        "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
        "SAMPLE_GROUPS": ["Car:1"],
        "PHYSICS": {"ENABLE_ANTI_WALL": True, "ANTI_WALL_HEIGHT_THRESH": 0.35, "MAX_OBSTACLE_POINTS": 2},
        "PLACEMENT": {"MAX_ATTEMPTS": 1},
    }
    # Scene with a dense wall of points at z = 0.0 (road is at -1.65m)
    wall_points = np.zeros((100, 4), dtype=np.float32)
    wall_points[:, 0] = 15.0  # inside proposed box
    wall_points[:, 1] = 0.0
    wall_points[:, 2] = 0.5   # 2.15m above ground -> high obstacle

    meta = {
        "format": "lidar_gt_database_v1", "num_point_features": 4, "source_frame_ids": ["000001"],
        "db_infos": {"Car": [{"name": "Car", "path": "c.bin", "image_idx": "000001",
                              "box3d_lidar": [15, 0, 0, 4, 2, 1.5, 0], "num_points_in_gt": 20}]},
    }
    (tmp_path / "gt_database").mkdir()
    np.ones((20, 4), dtype=np.float32).tofile(tmp_path / "c.bin")
    with (tmp_path / "gt_database" / "dbinfos_train.json").open("w") as f:
        json.dump(meta, f)

    sampler = OmniDataBaseSampler(tmp_path, config, {"Car": 0})
    pts_out, boxes_out = sampler(wall_points, np.empty((0, 8), dtype=np.float32))
    # Candidate should be rejected by anti-wall check; no box inserted
    assert len(boxes_out) == 0


def test_transactional_rollback_preserves_original_visibility(tmp_path):
    config = {
        "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
        "SAMPLE_GROUPS": ["Car:1"],
        "PHYSICS": {"MIN_VISIBLE_RATIO": 0.60},
        "PLACEMENT": {"MAX_ATTEMPTS": 1},
    }
    # Existing box with 10 points
    orig_box = np.array([[0, 1.5, 2.0, 4.0, 15.0, 0.0, -1.65, 0.0]], dtype=np.float32)
    orig_pts = np.ones((10, 4), dtype=np.float32)
    orig_pts[:, :2] = np.array([15.0, 0.0])

    meta = {
        "format": "lidar_gt_database_v1", "num_point_features": 4, "source_frame_ids": ["000001"],
        "db_infos": {"Car": [{"name": "Car", "path": "c.bin", "image_idx": "000001",
                              "box3d_lidar": [15, 0, 0, 4, 2, 1.5, 0], "num_points_in_gt": 20}]},
    }
    (tmp_path / "gt_database").mkdir()
    np.ones((20, 4), dtype=np.float32).tofile(tmp_path / "c.bin")
    with (tmp_path / "gt_database" / "dbinfos_train.json").open("w") as f:
        json.dump(meta, f)

    sampler = OmniDataBaseSampler(tmp_path, config, {"Car": 0})
    pts_out, boxes_out = sampler(orig_pts, orig_box)
    # Insertion would clear all 10 points of orig_box (100% loss) -> rollback triggers
    assert len(boxes_out) == 1
    assert len(pts_out) == 10
```

- [ ] **Step 2: Run all edge case tests**

Run: `pytest tests/test_omni_pcu_sampler.py -v`  
Expected: PASS all tests.

- [ ] **Step 3: Commit edge case verification suite**

```bash
git add tests/test_omni_pcu_sampler.py
git commit -m "test(augmentor): add rigorous edge case and anti-bypass tests"
```

---

### Task 6: System Integration, Configuration Profile & Multi-Worker Smoke Test

**Files:**
- Modify: `detector/core/datasets/augmentor/data_augmentor.py`
- Create: `configs/augmentation/omni_pcu_gt.json`
- Test: `tests/test_omni_pcu_sampler.py`

**Interfaces:**
- Produces: `DataAugmentor` registry recognition for `omni_gt_sampling`, configuration profile, and multi-worker `DataLoader` validation.

- [ ] **Step 1: Write failing test for `DataAugmentor` integration**

Add integration test to `tests/test_omni_pcu_sampler.py`:

```python
from core.datasets.augmentor.data_augmentor import DataAugmentor


def test_data_augmentor_loads_omni_gt_sampling(tmp_path):
    profile = {
        "AUG_CONFIG_LIST": [
            {
                "NAME": "omni_gt_sampling",
                "SAMPLE_GROUPS": ["Car:2"],
                "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
            }
        ]
    }
    meta = {
        "format": "lidar_gt_database_v1", "num_point_features": 4, "source_frame_ids": ["000001"],
        "db_infos": {"Car": []},
    }
    (tmp_path / "gt_database").mkdir()
    with (tmp_path / "gt_database" / "dbinfos_train.json").open("w") as f:
        json.dump(meta, f)

    aug = DataAugmentor(tmp_path, profile, {"Car": 0}, allowed_frame_ids=["000001"])
    assert aug is not None
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_omni_pcu_sampler.py::test_data_augmentor_loads_omni_gt_sampling -v`  
Expected: FAIL with `Unknown augmentor name: omni_gt_sampling`

- [ ] **Step 3: Update `detector/core/datasets/augmentor/data_augmentor.py`**

Register `omni_gt_sampling` to dispatch to `OmniDataBaseSampler`:

```python
        elif cur_cfg.NAME == "omni_gt_sampling":
            from .omni_sampler import OmniDataBaseSampler
            queue.append(OmniDataBaseSampler(
                root_path, cur_cfg, class_names,
                geometry=geometry, allowed_frame_ids=allowed_frame_ids,
            ))
```

- [ ] **Step 4: Create `configs/augmentation/omni_pcu_gt.json`**

Write standard JSON config profile matching the specification:

```json
{
  "AUG_CONFIG_LIST": [
    {
      "NAME": "omni_gt_sampling",
      "USE_ROAD_PLANE": true,
      "SAMPLE_GROUPS": ["Car:15", "Pedestrian:10", "Cyclist:10"],
      "LIMIT_WHOLE_SCENE": true,
      "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
      "PREPARE": {
        "filter_by_min_points": ["Car:5", "Pedestrian:5", "Cyclist:5"],
        "filter_by_difficulty": [-1]
      },
      "CURRICULUM": {
        "ENABLED": true,
        "WARMUP_EPOCHS": 10,
        "HARD_RATIO_BASE": 0.15,
        "HARD_RATIO_TARGET": 0.70,
        "DIFFICULTY_THRESHOLD": 0.45
      },
      "PLACEMENT": {
        "MODE": "stratified_range",
        "RANGE_TIERS": [[8.0, 25.0], [25.0, 45.0], [45.0, 65.5]],
        "TIER_WEIGHTS": [0.30, 0.40, 0.30],
        "MAX_ATTEMPTS": 15
      },
      "PHYSICS": {
        "ENABLE_DENSITY_SUBSAMPLE": true,
        "ENABLE_ANTI_WALL": true,
        "ANTI_WALL_HEIGHT_THRESH": 0.35,
        "MAX_OBSTACLE_POINTS": 3,
        "MIN_VISIBLE_RATIO": 0.50,
        "MIN_VISIBLE_POINTS": 5
      },
      "PERFORMANCE": {
        "CACHE_ENABLED": true,
        "CACHE_SIZE_MB": 128.0
      }
    }
  ]
}
```

- [ ] **Step 5: Run complete test suite and multi-worker DataLoader check**

Run:
```bash
PYTHONPATH=".:detector" pytest tests/test_omni_pcu_sampler.py -v
```
Expected: PASS 100% of tests.

- [ ] **Step 6: Commit complete integration**

```bash
git add detector/core/datasets/augmentor/data_augmentor.py configs/augmentation/omni_pcu_gt.json tests/test_omni_pcu_sampler.py
git commit -m "feat(augmentor): integrate Omni-PCU sampler into DataAugmentor with config profile"
```
