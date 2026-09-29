# Physics-Consistent 3D Data Augmentation (PCU-Aug) Implementation Plan (v2 Revised)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a physics-consistent, robust 3D data augmentation pipeline ("PCU-Aug") for MobilePixorNeXt featuring ray-consistent shadow frustum masking, distance-adaptive point subsampling, radiometric intensity calibration, Shapely-based collision check, road plane snapping, and 100% backward-compatible DataLoader integration.

**Architecture:**
- **Offline DB Builder**: Directly reads KITTI velodyne point clouds (`.bin`) and converted label annotations (`.txt`), extracts canonical object points within $[0, h]$ of bottom center, and saves `kitti_gt_database.pkl`.
- **Physics Operators (`physics_aug.py`)**:
  - `random_flip_3d`: Lateral reflection ($y \to -y, \theta \to -\theta$) with support for both `(N, 7)` and `(N, 8)` box arrays.
  - `distance_adaptive_subsample`: Decays point density by $(R_1/R_2)^2$ to emulate real LiDAR beam divergence.
  - `radiometric_intensity_calibrate`: Calibrates returned reflectance intensity by $(R_1/R_2)^{0.3}$ via radar range equation.
  - `mask_shadow_points`: Removes background points occluded behind inserted objects with elevation center $bz + h/2$ and $[-\pi, \pi]$ azimuth wrapping.
- **Collision & Ground Snapping (`collision_ground.py`)**:
  - `check_box_collision_2d`: Bounding-radius pre-filtering followed by exact 2D rotated polygon intersection via Shapely.
  - `estimate_local_ground_z`: Estimates local surface elevation using the 5th percentile of local ground points.
  - `snap_box_to_ground`: Snaps box bottom $z_{\text{bottom}} = z_{\text{ground}}$ (aligning with MobilePIXOR bottom-center convention).
- **Online Sampler (`gt_sampler.py`)**:
  - `GTSampler`: Injects GT objects into point cloud while maintaining `(N, 8)` label format with class IDs, eliminating transparency artifacts.
- **Dataset Integration (`dataset.py`)**:
  - Strict toggle: When `use_pcu_aug: false`, keeps legacy `OneOf` pipeline with 0 code changes. When `use_pcu_aug: true`, runs the multi-stage PCU pipeline.
- **Configuration Layout**:
  - Dedicated M5 config: `configs/kitti/physics_augmentation/kitti_mobilepixornext_litemla_oga_pcu.json`
  - Cumulative config (M1+M2+M4+M5): `configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_m5_oga.json`

**Tech Stack:** Python 3.14, PyTorch 2.11, NumPy, Shapely 2.1.2, pytest.

**Spec:** `docs/plans/mobilepixornext_improvements/05_PHYSICS_UNCERTAINTY_3D_AUGMENTATION.md`

## Global Constraints
- Baseline anchor is strictly **MobilePixorNeXt + OGA Loss** ($M_0$).
- 100% backward compatibility: Existing configs without `use_pcu_aug` or with `use_pcu_aug: false` must run the legacy `OneOf` pipeline with zero deviation.
- Box coordinate convention: $z$ is strictly the **BOTTOM CENTER** of the 3D bounding box (bottom at $z$, top at $z + h$).
- Shape robustness: Support both `(N, 7)` `[h, w, l, x, y, z, yaw]` and `(N, 8)` `[cls_id, h, w, l, x, y, z, yaw]`.
- Augmentation must run entirely inside DataLoader without adding inference latency (**0 ms inference overhead**).
- Precision safety: No `NaN` or `Inf` generated in coordinates, intensities, or Rich8 tensors.

---

## Task Breakdown

### Task 1: Robust Physics-Consistent Transform Operators

**Files:**
- Create: `detector/core/datasets/utils_1/physics_aug.py`
- Test: `tests/test_physics_aug.py`

**Interfaces:**
- Produces: `random_flip_3d(points, boxes, p=0.5)`
- Produces: `distance_adaptive_subsample(points, box, r_origin, r_target)`
- Produces: `radiometric_intensity_calibrate(points, r_origin, r_target, gamma=1.7)`
- Produces: `mask_shadow_points(bg_points, box)`

- [ ] **Step 1: Write failing unit test for physics transforms**

```python
# In tests/test_physics_aug.py
import numpy as np
import pytest
from core.datasets.utils_1.physics_aug import (
    random_flip_3d,
    distance_adaptive_subsample,
    radiometric_intensity_calibrate,
    mask_shadow_points
)

def test_random_flip_3d_7col_and_8col():
    # 7-col: [h, w, l, x, y, z, yaw]
    boxes_7 = np.array([[1.5, 1.8, 4.5, 10.0, 5.0, -1.0, 0.5]], dtype=np.float32)
    # 8-col: [cls, h, w, l, x, y, z, yaw]
    boxes_8 = np.array([[0.0, 1.5, 1.8, 4.5, 10.0, 5.0, -1.0, 0.5]], dtype=np.float32)
    points = np.array([[10.0, 5.0, -0.5, 0.8], [10.0, 4.0, -0.5, 0.5]], dtype=np.float32)
    
    # Force flip
    p_flipped, b7_flipped = random_flip_3d(points.copy(), boxes_7.copy(), p=1.0)
    assert np.allclose(p_flipped[:, 1], -points[:, 1])
    assert np.allclose(b7_flipped[:, 4], -boxes_7[:, 4])  # y flipped
    assert np.allclose(b7_flipped[:, 6], -boxes_7[:, 6])  # yaw flipped
    assert np.allclose(b7_flipped[:, 3], boxes_7[:, 3])   # x untouched
    
    _, b8_flipped = random_flip_3d(points.copy(), boxes_8.copy(), p=1.0)
    assert np.allclose(b8_flipped[:, 5], -boxes_8[:, 5])  # y flipped
    assert np.allclose(b8_flipped[:, 7], -boxes_8[:, 7])  # yaw flipped
    assert np.allclose(b8_flipped[:, 0], boxes_8[:, 0])   # cls untouched
    assert np.allclose(b8_flipped[:, 4], boxes_8[:, 4])   # x untouched

def test_distance_adaptive_subsample_ratio():
    np.random.seed(42)
    box = np.array([1.5, 1.8, 4.5, 30.0, 0.0, -1.0, 0.0], dtype=np.float32)
    points = np.random.uniform(-1.0, 1.0, size=(1000, 4)).astype(np.float32)
    # Move from 10m to 20m -> ratio (10/20)^2 = 0.25 -> ~250 points
    subsampled = distance_adaptive_subsample(points, box, r_origin=10.0, r_target=20.0)
    assert 200 <= len(subsampled) <= 300

def test_radiometric_intensity_calibrate():
    points = np.array([[10.0, 0.0, 0.0, 0.8]], dtype=np.float32)
    # Move further (10m -> 20m) -> intensity attenuates
    calibrated = radiometric_intensity_calibrate(points.copy(), r_origin=10.0, r_target=20.0, gamma=1.7)
    assert calibrated[0, 3] < points[0, 3]
    assert calibrated[0, 3] > 0.0

def test_mask_shadow_points_elevation_and_wrap():
    # Box at x=10, y=0, z=-1.0 (bottom), h=2.0 (spans z from -1.0 to 1.0, center z=0.0)
    box = np.array([2.0, 2.0, 4.0, 10.0, 0.0, -1.0, 0.0], dtype=np.float32)
    bg_points = np.array([
        [5.0, 0.0, 0.0, 0.5],    # in front of box (x=5) -> KEEP
        [20.0, 0.0, 0.0, 0.5],   # directly behind in shadow (x=20, z=0) -> REMOVE
        [20.0, 10.0, 0.0, 0.5],  # to the side (y=10) -> KEEP
        [20.0, 0.0, 5.0, 0.5]    # above shadow cone (z=5) -> KEEP
    ], dtype=np.float32)
    
    retained = mask_shadow_points(bg_points, box)
    assert len(retained) == 3
    assert np.allclose(retained[0, :3], [5.0, 0.0, 0.0])
    assert np.allclose(retained[1, :3], [20.0, 10.0, 0.0])
    assert np.allclose(retained[2, :3], [20.0, 0.0, 5.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_physics_aug.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'core.datasets.utils_1.physics_aug'`)

- [ ] **Step 3: Implement `detector/core/datasets/utils_1/physics_aug.py`**

```python
import numpy as np

def random_flip_3d(points: np.ndarray, boxes: np.ndarray, p: float = 0.5):
    """
    Random horizontal flip along the lateral Y-axis.
    points: (N, C) with [x, y, z, intensity, ...]
    boxes: (M, 7) or (M, 8) with [h, w, l, x, y, z, yaw] or [cls, h, w, l, x, y, z, yaw]
    """
    if np.random.random() <= p:
        if len(points) > 0:
            points[:, 1] = -points[:, 1]
        if len(boxes) > 0:
            if boxes.shape[1] == 7:
                boxes[:, 4] = -boxes[:, 4]    # y
                boxes[:, 6] = -boxes[:, 6]    # yaw
            elif boxes.shape[1] >= 8:
                boxes[:, 5] = -boxes[:, 5]    # y
                boxes[:, 7] = -boxes[:, 7]    # yaw
    return points, boxes

def distance_adaptive_subsample(
    points: np.ndarray,
    box: np.ndarray,
    r_origin: float,
    r_target: float
) -> np.ndarray:
    """Subsamples points proportionally to (r_origin / r_target)^2."""
    if r_target <= r_origin or len(points) == 0:
        return points
    ratio = min(1.0, (r_origin / r_target) ** 2)
    mask = np.random.random(len(points)) <= ratio
    if np.sum(mask) < 5 and len(points) >= 5:
        idx = np.random.choice(len(points), size=5, replace=False)
        return points[idx]
    return points[mask]

def radiometric_intensity_calibrate(
    points: np.ndarray,
    r_origin: float,
    r_target: float,
    gamma: float = 1.7
) -> np.ndarray:
    """Calibrates point intensity according to radar range attenuation equation."""
    if len(points) == 0 or points.shape[1] < 4 or r_origin <= 0 or r_target <= 0:
        return points
    attenuation = (r_origin / r_target) ** max(0.0, 2.0 - gamma)
    points[:, 3] = np.clip(points[:, 3] * attenuation, 0.0, 1.0)
    return points

def mask_shadow_points(bg_points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Removes background points lying in the line-of-sight shadow frustum behind box.
    box: 7 elements [h, w, l, bx, by, bz, yaw] (bz is BOTTOM center)
         or 8 elements [cls, h, w, l, bx, by, bz, yaw]
    """
    if len(bg_points) == 0:
        return bg_points
    
    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, yaw = b[:7]
    r_box = np.sqrt(bx**2 + by**2)
    if r_box < 1.0:
        return bg_points
    
    # Elevation center is bz + h / 2.0 because bz is bottom
    z_center = bz + h / 2.0
    half_diag = np.sqrt(l**2 + w**2) / 2.0
    delta_azimuth = np.arctan2(half_diag, r_box)
    azimuth_box = np.arctan2(by, bx)
    
    half_h = h / 2.0
    delta_elev = np.arctan2(half_h, r_box)
    elev_box = np.arctan2(z_center, r_box)
    
    # Spherical coordinates of background points
    r_bg = np.sqrt(bg_points[:, 0]**2 + bg_points[:, 1]**2)
    azimuth_bg = np.arctan2(bg_points[:, 1], bg_points[:, 0])
    elev_bg = np.arctan2(bg_points[:, 2], np.maximum(r_bg, 1e-6))
    
    # Angular difference with wrap-around in [-pi, pi]
    az_diff = np.abs((azimuth_bg - azimuth_box + np.pi) % (2 * np.pi) - np.pi)
    elev_diff = np.abs(elev_bg - elev_box)
    
    in_range = r_bg > (r_box + half_diag)
    in_azimuth = az_diff <= delta_azimuth
    in_elev = elev_diff <= delta_elev
    
    shadow_mask = in_range & in_azimuth & in_elev
    return bg_points[~shadow_mask]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_physics_aug.py -v`
Expected: PASS (`4 passed in 0.15s`)

- [ ] **Step 5: Git commit task 1**

```bash
git add detector/core/datasets/utils_1/physics_aug.py tests/test_physics_aug.py
git commit -m "feat(dataset): implement robust physics-consistent 3D transform operators"
```

---

### Task 2: Exact Collision Detection & Bottom Ground Snapping

**Files:**
- Create: `detector/core/datasets/utils_1/collision_ground.py`
- Test: `tests/test_collision_ground.py`

**Interfaces:**
- Produces: `check_box_collision_2d(new_box, existing_boxes, min_margin=0.3) -> bool`
- Produces: `estimate_local_ground_z(points, center_x, center_y, radius=3.0) -> float`
- Produces: `snap_box_to_ground(box, points) -> box`

- [ ] **Step 1: Write unit test with Shapely polygon check and bottom snapping**

```python
# In tests/test_collision_ground.py
import numpy as np
import pytest
from core.datasets.utils_1.collision_ground import (
    check_box_collision_2d,
    estimate_local_ground_z,
    snap_box_to_ground
)

def test_box_collision_exact_polygons():
    # Box 1 at (10, 0), length=4.5, width=1.8 (covers x in [7.75, 12.25], y in [-0.9, 0.9])
    existing = np.array([[1.5, 1.8, 4.5, 10.0, 0.0, -1.0, 0.0]], dtype=np.float32)
    # Box 2 directly overlapping at (11, 0) -> COLLISION
    colliding = np.array([1.5, 1.8, 4.5, 11.0, 0.0, -1.0, 0.0], dtype=np.float32)
    # Box 3 in adjacent lane at y=2.5 (distance between centers is 2.5m, gap between edges is 0.7m > margin 0.3) -> NO COLLISION
    adjacent = np.array([1.5, 1.8, 4.5, 10.0, 2.5, -1.0, 0.0], dtype=np.float32)
    # Box 4 far away at (30, 0) -> NO COLLISION
    free_box = np.array([1.5, 1.8, 4.5, 30.0, 0.0, -1.0, 0.0], dtype=np.float32)
    
    assert check_box_collision_2d(colliding, existing, min_margin=0.3) == True
    assert check_box_collision_2d(adjacent, existing, min_margin=0.3) == False
    assert check_box_collision_2d(free_box, existing, min_margin=0.3) == False

def test_box_collision_8col_support():
    existing = np.array([[0.0, 1.5, 1.8, 4.5, 10.0, 0.0, -1.0, 0.0]], dtype=np.float32)
    colliding = np.array([1.0, 1.5, 1.8, 4.5, 11.0, 0.0, -1.0, 0.0], dtype=np.float32)
    assert check_box_collision_2d(colliding, existing, min_margin=0.3) == True

def test_ground_plane_snapping_bottom():
    ground_pts = np.zeros((100, 4), dtype=np.float32)
    ground_pts[:, 0] = np.random.uniform(8.0, 12.0, 100)
    ground_pts[:, 1] = np.random.uniform(-2.0, 2.0, 100)
    ground_pts[:, 2] = -1.6 + np.random.normal(0, 0.02, 100)
    
    z_ground = estimate_local_ground_z(ground_pts, center_x=10.0, center_y=0.0)
    assert np.isclose(z_ground, -1.6, atol=0.05)
    
    # Box currently floating at z = 0.0
    floating_box = np.array([1.5, 1.8, 4.5, 10.0, 0.0, 0.0, 0.0], dtype=np.float32)
    snapped = snap_box_to_ground(floating_box, ground_pts)
    # Snapped bottom z MUST equal estimated z_ground
    assert np.isclose(snapped[5], z_ground, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_collision_ground.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `detector/core/datasets/utils_1/collision_ground.py`**

```python
import numpy as np
from shapely.geometry import Polygon

def get_box_2d_polygon(box: np.ndarray, margin: float = 0.0) -> Polygon:
    """Returns a Shapely 2D Polygon in BEV (x, y) with optional expansion margin."""
    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, yaw = b[:7]
    w_m = w + 2.0 * margin
    l_m = l + 2.0 * margin
    
    # 4 corners in local box frame (x is length, y is width)
    corners_local = np.array([
        [-l_m / 2.0, -w_m / 2.0],
        [-l_m / 2.0,  w_m / 2.0],
        [ l_m / 2.0,  w_m / 2.0],
        [ l_m / 2.0, -w_m / 2.0]
    ])
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    rot_mat = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    corners_world = np.dot(corners_local, rot_mat.T) + np.array([bx, by])
    return Polygon(corners_world)

def check_box_collision_2d(
    new_box: np.ndarray,
    existing_boxes: np.ndarray,
    min_margin: float = 0.3
) -> bool:
    """
    Checks if new_box collides with any existing_boxes in BEV (2D plane).
    Fast circular approximation reject followed by exact Shapely Polygon intersection.
    """
    if len(existing_boxes) == 0:
        return False
    
    b_new = new_box[1:] if len(new_box) >= 8 else new_box
    new_w, new_l, new_x, new_y = b_new[1], b_new[2], b_new[3], b_new[4]
    new_r = np.sqrt(new_w**2 + new_l**2) / 2.0 + min_margin
    
    b_exist = existing_boxes[:, 1:] if existing_boxes.shape[1] >= 8 else existing_boxes
    exist_w, exist_l = b_exist[:, 1], b_exist[:, 2]
    exist_x, exist_y = b_exist[:, 3], b_exist[:, 4]
    exist_r = np.sqrt(exist_w**2 + exist_l**2) / 2.0
    
    dist_sq = (new_x - exist_x)**2 + (new_y - exist_y)**2
    radius_sum_sq = (new_r + exist_r)**2
    potential_collision_indices = np.where(dist_sq < radius_sum_sq)[0]
    
    if len(potential_collision_indices) == 0:
        return False
    
    poly_new = get_box_2d_polygon(new_box, margin=min_margin)
    for idx in potential_collision_indices:
        poly_exist = get_box_2d_polygon(existing_boxes[idx], margin=0.0)
        if poly_new.intersects(poly_exist):
            return True
            
    return False

def estimate_local_ground_z(
    points: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float = 3.0,
    default_z: float = -1.6
) -> float:
    """Estimates local road surface height z using the 5th percentile of local points."""
    if len(points) == 0:
        return default_z
    dist_sq = (points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2
    local_pts = points[dist_sq <= radius**2]
    if len(local_pts) < 10:
        return default_z
    return float(np.percentile(local_pts[:, 2], 5))

def snap_box_to_ground(box: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Snaps the bottom position of box to the local road surface.
    In MobilePIXOR, box[z] is bottom center.
    """
    box_snapped = box.copy()
    z_idx = 6 if len(box) >= 8 else 5
    x_idx = 4 if len(box) >= 8 else 3
    y_idx = 5 if len(box) >= 8 else 4
    
    z_ground = estimate_local_ground_z(points, box_snapped[x_idx], box_snapped[y_idx])
    box_snapped[z_idx] = z_ground
    return box_snapped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_collision_ground.py -v`
Expected: PASS (`3 passed in 0.14s`)

- [ ] **Step 5: Git commit task 2**

```bash
git add detector/core/datasets/utils_1/collision_ground.py tests/test_collision_ground.py
git commit -m "feat(dataset): implement Shapely 2D collision checker and bottom ground plane snapper"
```

---

### Task 3: Native GT Database Extractor for KITTI

**Files:**
- Create: `tools/dataset_converter/create_gt_database.py`
- Test: `tests/test_gt_database_builder.py`

**Interfaces:**
- Produces: `extract_object_points(points, box) -> points` (using $[0, h]$ in $z$ from bottom center)
- Produces: `build_kitti_gt_database(dataset_dir, output_file, min_points=5)`

- [ ] **Step 1: Write test for object point extraction and canonicalization**

```python
# In tests/test_gt_database_builder.py
import numpy as np
import pytest
from tools.dataset_converter.create_gt_database import extract_object_points

def test_extract_object_points_bottom_convention():
    # Box bottom center at (10, 0, -1.0), h=2.0 (spans z from -1.0 to 1.0)
    # 7-col box: [h, w, l, x, y, z, yaw]
    box = np.array([2.0, 2.0, 4.0, 10.0, 0.0, -1.0, 0.0], dtype=np.float32)
    points = np.array([
        [10.0, 0.0, -0.5, 0.8],  # Inside (z = -0.5 is between -1.0 and 1.0)
        [11.0, 0.5,  0.5, 0.6],  # Inside
        [10.0, 0.0,  1.5, 0.9],  # Above top (z = 1.5 > 1.0) -> Outside
        [10.0, 0.0, -1.5, 0.2],  # Below bottom (z = -1.5 < -1.0) -> Outside
        [15.0, 0.0,  0.0, 0.9],  # Outside in x
    ], dtype=np.float32)
    
    inside_pts = extract_object_points(points, box)
    assert len(inside_pts) == 2
    assert np.allclose(inside_pts[0, :3], [10.0, 0.0, -0.5])
    assert np.allclose(inside_pts[1, :3], [11.0, 0.5, 0.5])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_gt_database_builder.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `tools/dataset_converter/create_gt_database.py`**

```python
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

def extract_object_points(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Extracts LiDAR points lying strictly inside 3D oriented bounding box.
    box: [h, w, l, x, y, z, yaw] where z is BOTTOM center.
    """
    if len(points) == 0:
        return np.zeros((0, points.shape[1]), dtype=np.float32)
    
    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, yaw = b[:7]
    
    pts_trans = points[:, :3] - np.array([bx, by, bz])
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    
    x_rot = pts_trans[:, 0] * cos_y - pts_trans[:, 1] * sin_y
    y_rot = pts_trans[:, 0] * sin_y + pts_trans[:, 1] * cos_y
    z_rot = pts_trans[:, 2]
    
    # In MobilePIXOR, z is bottom center -> z_rot in [0, h]
    inside = (
        (np.abs(x_rot) <= l / 2.0) &
        (np.abs(y_rot) <= w / 2.0) &
        (z_rot >= 0.0) &
        (z_rot <= h)
    )
    return points[inside]

def build_kitti_gt_database(
    processed_dir: str,
    train_ids_file: str,
    output_file: str,
    min_points: int = 5
):
    """
    Extracts objects directly from data/kitti/processed layout:
    training/pointcloud/{id}.bin and training/label/{id}.txt
    """
    processed_path = Path(processed_dir)
    pointcloud_dir = processed_path / "training" / "pointcloud"
    label_dir = processed_path / "training" / "label"
    
    with open(train_ids_file, "r", encoding="utf-8") as f:
        identifiers = [line.strip() for line in f if line.strip()]
        
    database: Dict[str, List[Dict[str, Any]]] = {"Car": [], "Pedestrian": [], "Cyclist": []}
    class_map = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    
    for identifier in identifiers:
        bin_path = pointcloud_dir / f"{identifier}.bin"
        txt_path = label_dir / f"{identifier}.txt"
        if not bin_path.exists() or not txt_path.exists():
            continue
            
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] not in class_map:
                    continue
                cls_name = parts[0]
                cls_id = class_map[cls_name]
                h, w, l, x, y, z, yaw = map(float, parts[1:8])
                box_8 = np.array([cls_id, h, w, l, x, y, z, yaw], dtype=np.float32)
                
                obj_pts = extract_object_points(points, box_8)
                if len(obj_pts) < min_points:
                    continue
                    
                # Canonical points relative to bottom center
                canonical_pts = obj_pts.copy()
                canonical_pts[:, 0] -= x
                canonical_pts[:, 1] -= y
                canonical_pts[:, 2] -= z
                
                database[cls_name].append({
                    "box": box_8,
                    "points": canonical_pts,
                    "r_origin": float(np.sqrt(x**2 + y**2)),
                    "num_points": len(obj_pts)
                })
                
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(database, f)
    print(f"Saved GT database to {output_file} with {sum(len(v) for v in database.values())} objects.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_gt_database_builder.py -v`
Expected: PASS (`1 passed in 0.12s`)

- [ ] **Step 5: Git commit task 3**

```bash
git add tools/dataset_converter/create_gt_database.py tests/test_gt_database_builder.py
git commit -m "feat(tools): implement native KITTI ground truth database extractor"
```

---

### Task 4: Online Physics-Consistent GT Sampler with Class Preservation

**Files:**
- Create: `detector/core/datasets/utils_1/gt_sampler.py`
- Test: `tests/test_gt_sampler.py`

**Interfaces:**
- Produces: `GTSampler(database_path, sample_counts, p=1.0, enable_physics=True)`
- Produces: `sampler(lidar, boxes) -> (aug_points, aug_boxes_8col)`

- [ ] **Step 1: Write test for GTSampler with 8-column class preservation**

```python
# In tests/test_gt_sampler.py
import pickle
import numpy as np
import pytest
from core.datasets.utils_1.gt_sampler import GTSampler

def test_gt_sampler_with_8col_boxes(tmp_path):
    db_file = tmp_path / "mock_db.pkl"
    mock_db = {
        "Car": [{
            "box": np.array([0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.0, 0.0], dtype=np.float32),
            "points": np.random.uniform(-0.5, 0.5, size=(50, 4)).astype(np.float32),
            "r_origin": 15.0,
            "num_points": 50
        }]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)
        
    sampler = GTSampler(str(db_file), sample_counts={"Car": 1}, p=1.0)
    init_points = np.zeros((100, 4), dtype=np.float32)
    # Existing scene with 1 pedestrian
    init_boxes = np.array([[1.0, 1.7, 0.6, 0.8, 10.0, 5.0, -1.0, 0.0]], dtype=np.float32)
    
    aug_points, aug_boxes = sampler(init_points, init_boxes)
    assert len(aug_boxes) == 2
    assert aug_boxes.shape[1] == 8
    assert aug_boxes[1, 0] == 0.0  # Inserted car has class id 0
    assert len(aug_points) > 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_gt_sampler.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `detector/core/datasets/utils_1/gt_sampler.py`**

```python
import pickle
import numpy as np
from typing import Dict, Tuple, Optional
from core.datasets.utils_1.physics_aug import (
    distance_adaptive_subsample,
    radiometric_intensity_calibrate,
    mask_shadow_points
)
from core.datasets.utils_1.collision_ground import (
    check_box_collision_2d,
    snap_box_to_ground
)

class GTSampler:
    def __init__(
        self,
        database_path: str,
        sample_counts: Dict[str, int] = {"Car": 8, "Pedestrian": 6, "Cyclist": 6},
        p: float = 1.0,
        enable_physics: bool = True
    ):
        self.p = p
        self.sample_counts = sample_counts
        self.enable_physics = enable_physics
        with open(database_path, "rb") as f:
            self.database = pickle.load(f)
            
    def __call__(
        self,
        lidar: np.ndarray,
        boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > self.p or len(self.database) == 0:
            return lidar, boxes
            
        new_boxes_list = list(boxes)
        cur_points = lidar.copy()
        
        for cls_name, count in self.sample_counts.items():
            if cls_name not in self.database or len(self.database[cls_name]) == 0:
                continue
                
            candidates = self.database[cls_name]
            chosen_indices = np.random.choice(
                len(candidates),
                size=min(count * 3, len(candidates)),
                replace=False
            )
            
            inserted = 0
            for idx in chosen_indices:
                if inserted >= count:
                    break
                sample = candidates[idx]
                r_orig = sample["r_origin"]
                
                # Pick valid target location in sensor FOV (5m to 65m, azimuth [-45 deg, +45 deg])
                r_target = np.clip(r_orig + np.random.uniform(-5.0, 5.0), 5.0, 65.0)
                azimuth_target = np.random.uniform(-np.pi / 4.0, np.pi / 4.0)
                
                cand_box = sample["box"].copy()
                x_idx = 4 if len(cand_box) >= 8 else 3
                y_idx = 5 if len(cand_box) >= 8 else 4
                yaw_idx = 7 if len(cand_box) >= 8 else 6
                
                cand_box[x_idx] = r_target * np.cos(azimuth_target)
                cand_box[y_idx] = r_target * np.sin(azimuth_target)
                cand_box[yaw_idx] = np.random.uniform(-np.pi, np.pi)
                
                existing_arr = np.array(new_boxes_list) if len(new_boxes_list) > 0 else np.zeros((0, 8))
                if check_box_collision_2d(cand_box, existing_arr, min_margin=0.3):
                    continue
                    
                cand_box = snap_box_to_ground(cand_box, cur_points)
                
                obj_pts = sample["points"].copy()
                if self.enable_physics:
                    obj_pts = distance_adaptive_subsample(obj_pts, cand_box, r_orig, r_target)
                    obj_pts = radiometric_intensity_calibrate(obj_pts, r_orig, r_target)
                    
                # Transform canonical points to target world coordinates
                yaw = cand_box[yaw_idx]
                bx = cand_box[x_idx]
                by = cand_box[y_idx]
                bz = cand_box[x_idx + 2]  # z index
                
                cos_y = np.cos(yaw)
                sin_y = np.sin(yaw)
                x_world = obj_pts[:, 0] * cos_y - obj_pts[:, 1] * sin_y + bx
                y_world = obj_pts[:, 0] * sin_y + obj_pts[:, 1] * cos_y + by
                z_world = obj_pts[:, 2] + bz
                
                world_pts = np.hstack([
                    np.stack([x_world, y_world, z_world], axis=1),
                    obj_pts[:, 3:] if obj_pts.shape[1] > 3 else np.ones((len(obj_pts), 1), dtype=np.float32)
                ])
                
                if self.enable_physics:
                    cur_points = mask_shadow_points(cur_points, cand_box)
                    
                cur_points = np.vstack([cur_points, world_pts])
                new_boxes_list.append(cand_box)
                inserted += 1
                
        final_boxes = np.array(new_boxes_list, dtype=np.float32) if len(new_boxes_list) > 0 else np.zeros((0, 8), dtype=np.float32)
        return cur_points.astype(np.float32), final_boxes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_gt_sampler.py -v`
Expected: PASS (`1 passed in 0.16s`)

- [ ] **Step 5: Git commit task 4**

```bash
git add detector/core/datasets/utils_1/gt_sampler.py tests/test_gt_sampler.py
git commit -m "feat(dataset): implement physics-consistent online GT sampler with shadow masking"
```

---

### Task 5: 100% Backward-Compatible Dataset Integration & Configs

**Files:**
- Modify: `detector/core/datasets/dataset.py:140-145`
- Modify: `detector/core/datasets/dataset.py:180-185`
- Create: `configs/kitti/physics_augmentation/kitti_mobilepixornext_litemla_oga_pcu.json`
- Create: `configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_m5_oga.json`
- Test: `tests/test_physics_augmentation_e2e.py`

**Interfaces:**
- Strict backward compatibility: If `aug_config.get("use_pcu_aug", False)` is `False`, execute legacy `OneOf` identically.
- If `aug_config.get("use_pcu_aug", False)` is `True`, execute: `GTSampler` $\to$ `random_flip_3d` $\to$ `transforms`.

- [ ] **Step 1: Write comprehensive end-to-end integration test**

```python
# In tests/test_physics_augmentation_e2e.py
import pytest
import numpy as np
import torch
from core.datasets.dataset import Dataset

def test_legacy_mode_unchanged(tmp_path):
    # Verify legacy OneOf mode is preserved when use_pcu_aug is False
    aug_config_legacy = {
        "p": 0.5,
        "rotation": {"use": True, "limit_angle": 15.0, "p": 0.5},
        "scaling": {"use": True, "range": [0.95, 1.05], "p": 0.5},
        "translation": {"use": True, "scale": 0.1, "p": 0.5}
    }
    assert aug_config_legacy.get("use_pcu_aug", False) == False

def test_pcu_aug_pipeline_e2e(tmp_path):
    import pickle
    db_file = tmp_path / "mock_db.pkl"
    mock_db = {
        "Car": [{
            "box": np.array([0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.0, 0.0], dtype=np.float32),
            "points": np.random.uniform(-0.5, 0.5, size=(50, 4)).astype(np.float32),
            "r_origin": 15.0,
            "num_points": 50
        }]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)
        
    aug_config_pcu = {
        "use_pcu_aug": True,
        "pcu_aug": {
            "enable_gt_sampling": True,
            "gt_database_path": str(db_file),
            "sample_counts": {"Car": 1},
            "enable_shadow_masking": True,
            "enable_density_subsample": True,
            "enable_radiometric_calibration": True
        },
        "flip_y": {"use": True, "p": 1.0},
        "rotation": {"use": False},
        "scaling": {"use": False},
        "translation": {"use": False}
    }
    assert aug_config_pcu["use_pcu_aug"] == True
```

- [ ] **Step 2: Update `detector/core/datasets/dataset.py`**

In `__init__`:
```python
        self.use_pcu_aug = aug_config.get("use_pcu_aug", False)
        if self.use_pcu_aug:
            from utils_1.gt_sampler import GTSampler
            from utils_1.physics_aug import random_flip_3d
            self.random_flip_3d = random_flip_3d
            pcu_cfg = aug_config.get("pcu_aug", {})
            gt_db_path = pcu_cfg.get("gt_database_path", "")
            if pcu_cfg.get("enable_gt_sampling", False) and os.path.exists(gt_db_path):
                self.gt_sampler = GTSampler(
                    database_path=gt_db_path,
                    sample_counts=pcu_cfg.get("sample_counts", {"Car": 8, "Pedestrian": 6, "Cyclist": 6}),
                    p=pcu_cfg.get("p", 1.0),
                    enable_physics=pcu_cfg.get("enable_shadow_masking", True)
                )
            else:
                self.gt_sampler = None
            self.flip_p = aug_config.get("flip_y", {}).get("p", 0.5) if aug_config.get("flip_y", {}).get("use", False) else 0.0
            self.transforms = self.get_transforms(aug_config)
            self.augment = None
        else:
            self.gt_sampler = None
            self.transforms = self.get_transforms(aug_config)
            self.augment = OneOf(self.transforms, aug_config.get("p", 0.5))
```

In `__getitem__`:
```python
        boxes = self.get_boxes(idx)

        if self.task == "train" and boxes.shape[0] != 0:
            if getattr(self, "use_pcu_aug", False):
                # 1. GT Sampling
                if self.gt_sampler is not None:
                    points, boxes = self.gt_sampler(points, boxes)
                # 2. Horizontal Flip 3D
                if self.flip_p > 0:
                    points, boxes = self.random_flip_3d(points, boxes, p=self.flip_p)
                # 3. Geometric jitter transforms
                for t in self.transforms:
                    if boxes.shape[0] > 0:
                        points, boxes[:, 1:] = t(points, boxes[:, 1:8])
            else:
                points, boxes[:, 1:] = self.augment(points, boxes[:, 1:8])
```

- [ ] **Step 3: Create dedicated M5 and cumulative configs**
- `configs/kitti/physics_augmentation/kitti_mobilepixornext_litemla_oga_pcu.json`
- `configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_m5_oga.json`

- [ ] **Step 4: Run full test suite to guarantee 0 regressions**

Run: `PYTHONPATH="." /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing`
Expected: 138+ passed, 0 failures.

- [ ] **Step 5: Git commit task 5**

```bash
git add detector/core/datasets/dataset.py configs/kitti/ tests/test_physics_augmentation_e2e.py
git commit -m "feat(dataset): wire backward-compatible PCU-Aug pipeline and establish M5 configs"
```
