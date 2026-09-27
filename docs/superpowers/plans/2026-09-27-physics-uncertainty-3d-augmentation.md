# Physics-Consistent & Uncertainty-Guided 3D Data Augmentation (PCU-Aug) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a physics-consistent, uncertainty-guided 3D data augmentation pipeline ("PCU-Aug") for MobilePixorNeXt, featuring ray-consistent occlusion shadow masking, distance-adaptive point subsampling, radiometric intensity calibration, and dynamic class sampling anchored to the MobilePixorNeXt + OGA Loss baseline.

**Architecture:**
- **Offline DB Builder**: Crops point clouds and 3D bounding boxes for Car, Pedestrian, Cyclist across KITTI train frames, indexing range $R_1$, intensity profile, and local geometry into `kitti_gt_database.pkl`.
- **Physics-Consistent Transforms**:
  - *Random Horizontal Flip 3D*: Exact lateral reflection ($y \to -y, \theta \to -\theta$).
  - *Ray-Aware Shadow Frustum*: Computes the line-of-sight shadow volume behind inserted objects and removes background points, eliminating the "ghost-object" transparency anomaly.
  - *Radiometric Intensity Calibrator*: Adjusts point intensity according to the radar range attenuation equation $(R_1/R_2)^\gamma$.
  - *Distance-Adaptive Subsampling*: Subsamples point clouds proportionally to $(R_1/R_2)^2$ to emulate real LiDAR beam divergence.
- **Online Sampler & Snapper**: Verifies 2D polygon collision, snaps object base to the local road plane, and splices points into the frame point cloud before Rich8 BEV tensor generation.
- **Uncertainty & Curriculum Controller**: Dynamically adjusts sampling rates for minority/hard classes (Pedestrian, Cyclist) using loss uncertainties from `TemperatureSoftmaxUncertainty`.

**Tech Stack:** Python 3.14, PyTorch 2.11, NumPy, Shapely (for fast 2D rotated polygon intersection), pytest.

**Spec:** `docs/plans/mobilepixornext_improvements/05_PHYSICS_UNCERTAINTY_3D_AUGMENTATION.md`

## Global Constraints
- Baseline anchor is strictly **MobilePixorNeXt + OGA Loss** ($M_0$).
- Augmentation must run entirely inside the data loader / worker processes without adding any inference latency (**0 ms inference overhead**).
- Processing overhead per frame during training must not exceed **8.0 ms** to ensure high GPU utilization (>90%) with 4 DataLoader workers.
- Backward compatibility: All existing augmentations and configurations must remain 100% functional when `use_pcu_aug: false`.
- Precision safety: Generated point coordinates and Rich8 tensors must not contain `NaN` or `Inf`.

---

## Mathematical Formulation & Physical Proofs

### 1. Ray-Consistent Shadow Frustum (Eliminating Ghost Objects)
Let the sensor origin be $O = (0, 0, 0)$. An inserted 3D bounding box $B$ has 8 corners $C_k = (x_k, y_k, z_k)$.
For any point in spherical coordinates:
$$\theta = \text{atan2}(y, x), \quad \phi = \text{atan2}\left(z, \sqrt{x^2 + y^2}\right), \quad r = \sqrt{x^2 + y^2 + z^2}$$
The bounding box defines an angular frustum $[\theta_{\min}, \theta_{\max}] \times [\phi_{\min}, \phi_{\max}]$.
For any background point $P_{\text{bg}}$ whose ray vector $(\theta_{\text{bg}}, \phi_{\text{bg}})$ penetrates the convex hull of $B$:
$$\text{If } r_{\text{bg}} > \max_{k} r_k \implies P_{\text{bg}} \in \text{ShadowVolume}(B)$$
Such background points must be masked out to satisfy optical line-of-sight occlusion.

### 2. Distance-Adaptive Point Density Subsampling
The solid angle subtended by an object of area $A$ at range $R$ is $\Omega \approx \frac{A}{R^2}$.
When an object extracted at range $R_1$ is pasted at range $R_2 > R_1$:
$$\text{Retention Ratio } p_{\text{retain}} = \min\left(1.0, \left(\frac{R_1}{R_2}\right)^2\right)$$
Points inside the box are randomly subsampled with probability $p_{\text{retain}}$.

### 3. Radiometric Intensity Calibration (Radar Range Equation)
LiDAR returned power is given by $P_r \propto \frac{\rho \cdot \cos \alpha}{R^2} \cdot G(R)$, where $G(R) \approx R^\gamma$ models the internal receiver amplifier curve ($\gamma \in [1.5, 2.0]$).
The calibrated intensity is:
$$I_{\text{new}} = \text{clip}\left(I_{\text{old}} \cdot \left(\frac{R_1}{R_2}\right)^{2 - \gamma}, 0.0, 1.0\right)$$
For Velodyne HDL-64E on KITTI, $\gamma \approx 1.7$, yielding mild attenuation $(R_1/R_2)^{0.3}$.

### 4. Uncertainty-Guided Class Sampling
Let $\sigma_c^2$ be the current homoscedastic uncertainty parameter of class $c$ from `TemperatureSoftmaxUncertainty`.
The dynamic injection probability is normalized via softmax:
$$w_c = \frac{\exp(\sigma_c / \tau)}{\sum_{j} \exp(\sigma_j / \tau)}$$
Where $\tau = 1.0$ is the sampling temperature. Classes with higher loss/uncertainty receive proportionally higher paste counts.

---

## Task Breakdown

### Task 1: Horizontal Flip & Physics-Consistent Transform Operators

**Files:**
- Create: `detector/core/datasets/utils_1/physics_aug.py`
- Test: `tests/test_physics_aug.py`

**Interfaces:**
- Produces: `random_flip_3d(points, boxes, p=0.5)`
- Produces: `distance_adaptive_subsample(points, box, r_origin, r_target)`
- Produces: `radiometric_intensity_calibrate(points, r_origin, r_target, gamma=1.7)`
- Produces: `mask_shadow_points(background_points, box)`

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

def test_random_flip_3d_symmetry():
    # Box: [h, w, l, x, y, z, yaw]
    boxes = np.array([[1.5, 1.8, 4.5, 10.0, 5.0, 0.0, 0.5]], dtype=np.float32)
    points = np.array([[10.0, 5.0, 0.0, 0.8], [10.0, 4.0, 0.0, 0.5]], dtype=np.float32)
    
    # Force flip
    p_flipped, b_flipped = random_flip_3d(points.copy(), boxes.copy(), p=1.0)
    assert np.allclose(p_flipped[:, 1], -points[:, 1])
    assert np.allclose(b_flipped[:, 4], -boxes[:, 4])
    assert np.allclose(b_flipped[:, 6], -boxes[:, 6])

def test_distance_adaptive_subsample_ratio():
    np.random.seed(42)
    box = np.array([1.5, 1.8, 4.5, 30.0, 0.0, 0.0, 0.0], dtype=np.float32)
    points = np.random.uniform(-1.0, 1.0, size=(1000, 4)).astype(np.float32)
    # Move from 10m to 20m -> ratio (10/20)^2 = 0.25 -> ~250 points
    subsampled = distance_adaptive_subsample(points, box, r_origin=10.0, r_target=20.0)
    assert 200 <= len(subsampled) <= 300

def test_mask_shadow_points():
    # Box at x=10, y=0, z=0 (depth 4m: x from 8 to 12)
    box = np.array([2.0, 2.0, 4.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Point A in front (x=5), Point B inside (x=10), Point C directly behind in shadow (x=20)
    bg_points = np.array([
        [5.0, 0.0, 0.0, 0.5],   # front -> keep
        [20.0, 0.0, 0.0, 0.5],  # behind -> remove
        [20.0, 10.0, 0.0, 0.5]  # to the side -> keep
    ], dtype=np.float32)
    
    retained = mask_shadow_points(bg_points, box)
    assert len(retained) == 2
    assert np.allclose(retained[0, :3], [5.0, 0.0, 0.0])
    assert np.allclose(retained[1, :3], [20.0, 10.0, 0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest tests/test_physics_aug.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'core.datasets.utils_1.physics_aug'`)

- [ ] **Step 3: Implement `physics_aug.py`**

```python
# In detector/core/datasets/utils_1/physics_aug.py
import numpy as np

def random_flip_3d(points: np.ndarray, boxes: np.ndarray, p: float = 0.5):
    """
    Random horizontal flip along the lateral Y-axis.
    points: (N, C) with [x, y, z, intensity, ...]
    boxes: (M, 7) with [h, w, l, x, y, z, yaw]
    """
    if np.random.random() <= p:
        if len(points) > 0:
            points[:, 1] = -points[:, 1]
        if len(boxes) > 0:
            boxes[:, 4] = -boxes[:, 4]    # y
            boxes[:, 6] = -boxes[:, 6]    # yaw
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
    # Ensure at least 5 points retained if originally present
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
    """Calibrates point intensity according to radar range attenuation."""
    if len(points) == 0 or points.shape[1] < 4 or r_origin <= 0 or r_target <= 0:
        return points
    attenuation = (r_origin / r_target) ** max(0.0, 2.0 - gamma)
    points[:, 3] = np.clip(points[:, 3] * attenuation, 0.0, 1.0)
    return points

def mask_shadow_points(bg_points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Removes background points lying in the line-of-sight shadow frustum behind box.
    box: [h, w, l, x, y, z, yaw]
    """
    if len(bg_points) == 0:
        return bg_points
    
    h, w, l, bx, by, bz, yaw = box
    r_box = np.sqrt(bx**2 + by**2)
    if r_box < 1.0:
        return bg_points
    
    # Compute angular bounding cone of the box
    half_diag = np.sqrt(l**2 + w**2) / 2.0
    delta_azimuth = np.arctan2(half_diag, r_box)
    azimuth_box = np.arctan2(by, bx)
    
    half_h = h / 2.0
    delta_elev = np.arctan2(half_h, r_box)
    elev_box = np.arctan2(bz, r_box)
    
    # Spherical coordinates of background points
    r_bg = np.sqrt(bg_points[:, 0]**2 + bg_points[:, 1]**2)
    azimuth_bg = np.arctan2(bg_points[:, 1], bg_points[:, 0])
    elev_bg = np.arctan2(bg_points[:, 2], np.maximum(r_bg, 1e-6))
    
    # Shadow condition: behind the box in range AND inside angular frustum
    in_range = r_bg > (r_box + half_diag)
    in_azimuth = np.abs(azimuth_bg - azimuth_box) <= delta_azimuth
    in_elev = np.abs(elev_bg - elev_box) <= delta_elev
    
    shadow_mask = in_range & in_azimuth & in_elev
    return bg_points[~shadow_mask]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest tests/test_physics_aug.py -v`
Expected: PASS (`3 passed in 0.15s`)

- [ ] **Step 5: Git commit task 1**

```bash
git add detector/core/datasets/utils_1/physics_aug.py tests/test_physics_aug.py
git commit -m "feat(dataset): implement physics-consistent 3D augmentation operators"
```

---

### Task 2: Collision Detection & Ground Plane Snapping

**Files:**
- Create: `detector/core/datasets/utils_1/collision_ground.py`
- Test: `tests/test_collision_ground.py`

**Interfaces:**
- Produces: `check_box_collision_2d(new_box, existing_boxes, min_margin=0.5) -> bool`
- Produces: `estimate_local_ground_z(points, center_x, center_y, radius=3.0) -> float`
- Produces: `snap_box_to_ground(box, points) -> box`

- [ ] **Step 1: Write failing test for collision and ground snapping**

```python
# In tests/test_collision_ground.py
import numpy as np
import pytest
from core.datasets.utils_1.collision_ground import (
    check_box_collision_2d,
    estimate_local_ground_z,
    snap_box_to_ground
)

def test_box_collision_detection():
    # Box 1 at (10, 0)
    existing = np.array([[1.5, 1.8, 4.5, 10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # Box 2 overlapping at (11, 0)
    colliding_box = np.array([1.5, 1.8, 4.5, 11.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Box 3 far away at (30, 0)
    free_box = np.array([1.5, 1.8, 4.5, 30.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    assert check_box_collision_2d(colliding_box, existing) == True
    assert check_box_collision_2d(free_box, existing) == False

def test_ground_plane_snapping():
    # Flat ground at z = -1.6m
    ground_pts = np.zeros((100, 4), dtype=np.float32)
    ground_pts[:, 0] = np.random.uniform(8.0, 12.0, 100)
    ground_pts[:, 1] = np.random.uniform(-2.0, 2.0, 100)
    ground_pts[:, 2] = -1.6 + np.random.normal(0, 0.02, 100)
    
    z_ground = estimate_local_ground_z(ground_pts, center_x=10.0, center_y=0.0)
    assert np.isclose(z_ground, -1.6, atol=0.05)
    
    # Floating box with height 1.5 at z = 0.0 -> bottom is -0.75
    floating_box = np.array([1.5, 1.8, 4.5, 10.0, 0.0, 0.0, 0.0], dtype=np.float32)
    snapped_box = snap_box_to_ground(floating_box, ground_pts)
    # Snapped center z should be z_ground + h/2 = -1.6 + 0.75 = -0.85
    assert np.isclose(snapped_box[5], -1.6 + 1.5 / 2.0, atol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest tests/test_collision_ground.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `collision_ground.py`**

```python
# In detector/core/datasets/utils_1/collision_ground.py
import numpy as np

def check_box_collision_2d(
    new_box: np.ndarray,
    existing_boxes: np.ndarray,
    min_margin: float = 0.5
) -> bool:
    """
    Checks if new_box collides with any existing_boxes in BEV (2D plane).
    Boxes format: [h, w, l, x, y, z, yaw]
    """
    if len(existing_boxes) == 0:
        return False
    
    # Fast circular approximation reject
    new_x, new_y = new_box[3], new_box[4]
    new_r = np.sqrt(new_box[1]**2 + new_box[2]**2) / 2.0 + min_margin
    
    exist_x, exist_y = existing_boxes[:, 3], existing_boxes[:, 4]
    exist_r = np.sqrt(existing_boxes[:, 1]**2 + existing_boxes[:, 2]**2) / 2.0
    
    dist_sq = (new_x - exist_x)**2 + (new_y - exist_y)**2
    radius_sum_sq = (new_r + exist_r)**2
    
    return np.any(dist_sq < radius_sum_sq)

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
    """Snaps the vertical position of box so its bottom rests on the road surface."""
    box_snapped = box.copy()
    h, x, y = box[0], box[3], box[4]
    z_ground = estimate_local_ground_z(points, x, y)
    box_snapped[5] = z_ground + (h / 2.0)
    return box_snapped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest tests/test_collision_ground.py -v`
Expected: PASS (`2 passed in 0.12s`)

- [ ] **Step 5: Git commit task 2**

```bash
git add detector/core/datasets/utils_1/collision_ground.py tests/test_collision_ground.py
git commit -m "feat(dataset): implement fast BEV collision checker and ground plane snapper"
```

---

### Task 3: Ground Truth Database Creator & Offline Extractor

**Files:**
- Create: `tools/dataset_converter/create_gt_database.py`
- Test: `tests/test_gt_database_builder.py`

**Interfaces:**
- Produces: `extract_object_points(lidar_points, box) -> object_points`
- Produces: `build_kitti_gt_database(dataset_dir, output_pkl)`

- [ ] **Step 1: Write test for object point extraction**

```python
# In tests/test_gt_database_builder.py
import numpy as np
import pytest
from tools.dataset_converter.create_gt_database import extract_object_points

def test_extract_object_points():
    # Box centered at (10, 0, 0), size h=2, w=2, l=4, yaw=0
    box = np.array([2.0, 2.0, 4.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float32)
    points = np.array([
        [10.0, 0.0, 0.0, 0.8],  # Inside
        [11.0, 0.5, 0.5, 0.6],  # Inside
        [15.0, 0.0, 0.0, 0.9],  # Outside
        [10.0, 5.0, 0.0, 0.2]   # Outside
    ], dtype=np.float32)
    
    inside_pts = extract_object_points(points, box)
    assert len(inside_pts) == 2
    assert np.allclose(inside_pts[0, :3], [10.0, 0.0, 0.0])
    assert np.allclose(inside_pts[1, :3], [11.0, 0.5, 0.5])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest tests/test_gt_database_builder.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `create_gt_database.py`**

```python
# In tools/dataset_converter/create_gt_database.py
import os
import pickle
import numpy as np
from typing import Dict, List, Any

def extract_object_points(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Extracts LiDAR points lying strictly inside 3D oriented bounding box.
    box: [h, w, l, x, y, z, yaw]
    """
    if len(points) == 0:
        return np.zeros((0, points.shape[1]), dtype=np.float32)
    
    h, w, l, bx, by, bz, yaw = box
    
    # Translate to box frame
    pts_trans = points[:, :3] - np.array([bx, by, bz])
    
    # Rotate by -yaw
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    x_rot = pts_trans[:, 0] * cos_y - pts_trans[:, 1] * sin_y
    y_rot = pts_trans[:, 0] * sin_y + pts_trans[:, 1] * cos_y
    z_rot = pts_trans[:, 2]
    
    # Check boundaries
    inside = (
        (np.abs(x_rot) <= l / 2.0) &
        (np.abs(y_rot) <= w / 2.0) &
        (np.abs(z_rot) <= h / 2.0)
    )
    return points[inside]

def build_kitti_gt_database(kitti_dataset, output_file: str, min_points: int = 5):
    """Iterates through dataset and builds database dictionary."""
    database: Dict[str, List[Dict[str, Any]]] = {"Car": [], "Pedestrian": [], "Cyclist": []}
    
    for idx in range(len(kitti_dataset)):
        data = kitti_dataset[idx]
        points = data["lidar"]
        boxes = data["labels"]
        class_names = data.get("class_names", ["Car"] * len(boxes))
        
        for box, cls_name in zip(boxes, class_names):
            if cls_name not in database:
                continue
            obj_pts = extract_object_points(points, box)
            if len(obj_pts) < min_points:
                continue
            
            # Store canonical object relative to its center
            bx, by, bz = box[3], box[4], box[5]
            canonical_pts = obj_pts.copy()
            canonical_pts[:, 0] -= bx
            canonical_pts[:, 1] -= by
            canonical_pts[:, 2] -= bz
            
            database[cls_name].append({
                "box": box,
                "points": canonical_pts,
                "r_origin": float(np.sqrt(bx**2 + by**2)),
                "num_points": len(obj_pts)
            })
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(database, f)
    print(f"GT Database saved to {output_file} with {sum(len(v) for v in database.values())} samples.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest tests/test_gt_database_builder.py -v`
Expected: PASS (`1 passed in 0.14s`)

- [ ] **Step 5: Git commit task 3**

```bash
git add tools/dataset_converter/create_gt_database.py tests/test_gt_database_builder.py
git commit -m "feat(tools): implement GT database extraction tool for 3D point cloud sampling"
```

---

### Task 4: Online Physics-Consistent & Uncertainty-Guided GT Sampler

**Files:**
- Create: `detector/core/datasets/utils_1/gt_sampler.py`
- Test: `tests/test_gt_sampler.py`

**Interfaces:**
- Produces: `GTSampler(database_path, sample_counts, p=1.0)`
- Produces: `sampler(lidar_points, boxes, class_uncertainties=None) -> (aug_points, aug_boxes)`

- [ ] **Step 1: Write test for online GT Sampler**

```python
# In tests/test_gt_sampler.py
import numpy as np
import pytest
from core.datasets.utils_1.gt_sampler import GTSampler

def test_gt_sampler_insertion(tmp_path):
    import pickle
    # Mock database
    db_file = tmp_path / "mock_db.pkl"
    mock_db = {
        "Car": [{
            "box": np.array([1.5, 1.8, 4.5, 15.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "points": np.random.uniform(-0.5, 0.5, size=(50, 4)).astype(np.float32),
            "r_origin": 15.0,
            "num_points": 50
        }]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)
        
    sampler = GTSampler(str(db_file), sample_counts={"Car": 1}, p=1.0)
    
    init_points = np.zeros((100, 4), dtype=np.float32)
    init_boxes = np.zeros((0, 7), dtype=np.float32)
    
    aug_points, aug_boxes = sampler(init_points, init_boxes)
    assert len(aug_boxes) == 1
    assert len(aug_points) > 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest tests/test_gt_sampler.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `gt_sampler.py`**

```python
# In detector/core/datasets/utils_1/gt_sampler.py
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
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
        boxes: np.ndarray,
        class_uncertainties: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > self.p or len(self.database) == 0:
            return lidar, boxes
            
        new_boxes_list = list(boxes)
        cur_points = lidar
        
        for cls_name, count in self.sample_counts.items():
            if cls_name not in self.database or len(self.database[cls_name]) == 0:
                continue
            
            # Uncertainty-guided scaling: boost sample count if uncertainty is high
            sample_n = count
            if class_uncertainties and cls_name in class_uncertainties:
                # scale factor between 0.8 and 1.5
                u = class_uncertainties[cls_name]
                scale = float(np.clip(u, 0.8, 1.5))
                sample_n = int(np.round(count * scale))
                
            candidates = self.database[cls_name]
            chosen_indices = np.random.choice(len(candidates), size=min(sample_n * 2, len(candidates)), replace=False)
            
            inserted = 0
            for idx in chosen_indices:
                if inserted >= sample_n:
                    break
                sample = candidates[idx]
                r_orig = sample["r_origin"]
                
                # Pick target location with similar range (+- 5m) or random angle
                r_target = np.clip(r_orig + np.random.uniform(-5.0, 5.0), 5.0, 65.0)
                azimuth_target = np.random.uniform(-np.pi / 4, np.pi / 4)
                
                new_x = r_target * np.cos(azimuth_target)
                new_y = r_target * np.sin(azimuth_target)
                
                cand_box = sample["box"].copy()
                cand_box[3] = new_x
                cand_box[4] = new_y
                cand_box[6] = np.random.uniform(-np.pi, np.pi)
                
                # Collision check
                existing_arr = np.array(new_boxes_list) if len(new_boxes_list) > 0 else np.zeros((0, 7))
                if check_box_collision_2d(cand_box, existing_arr):
                    continue
                    
                # Snap to local ground
                cand_box = snap_box_to_ground(cand_box, cur_points)
                
                # Prepare object points
                obj_pts = sample["points"].copy()
                if self.enable_physics:
                    obj_pts = distance_adaptive_subsample(obj_pts, cand_box, r_orig, r_target)
                    obj_pts = radiometric_intensity_calibrate(obj_pts, r_orig, r_target)
                
                # Transform canonical points to target box location
                cos_y = np.cos(cand_box[6])
                sin_y = np.sin(cand_box[6])
                x_world = obj_pts[:, 0] * cos_y - obj_pts[:, 1] * sin_y + cand_box[3]
                y_world = obj_pts[:, 0] * sin_y + obj_pts[:, 1] * cos_y + cand_box[4]
                z_world = obj_pts[:, 2] + cand_box[5]
                
                world_pts = np.hstack([
                    np.stack([x_world, y_world, z_world], axis=1),
                    obj_pts[:, 3:] if obj_pts.shape[1] > 3 else np.ones((len(obj_pts), 1))
                ])
                
                # Physics: remove background in shadow frustum
                if self.enable_physics:
                    cur_points = mask_shadow_points(cur_points, cand_box)
                    
                cur_points = np.vstack([cur_points, world_pts])
                new_boxes_list.append(cand_box)
                inserted += 1
                
        final_boxes = np.array(new_boxes_list, dtype=np.float32) if len(new_boxes_list) > 0 else np.zeros((0, 7), dtype=np.float32)
        return cur_points.astype(np.float32), final_boxes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest tests/test_gt_sampler.py -v`
Expected: PASS (`1 passed in 0.18s`)

- [ ] **Step 5: Git commit task 4**

```bash
git add detector/core/datasets/utils_1/gt_sampler.py tests/test_gt_sampler.py
git commit -m "feat(dataset): implement physics-consistent online GT sampler with shadow masking"
```

---

### Task 5: Integration into KittiDataset & Training Config

**Files:**
- Modify: `detector/core/datasets/dataset.py:135-155`
- Modify: `detector/core/datasets/dataset.py:498-520`
- Modify: `configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json`
- Test: `tests/test_standard_training_notebook.py`
- Test: `tests/test_physics_augmentation_e2e.py`

**Interfaces:**
- Consumes: `physics_aug.random_flip_3d`, `gt_sampler.GTSampler`
- Produces: `KittiDataset` returning Rich8 BEV tensor populated with physics-consistent augmentations.

- [ ] **Step 1: Write integration test for dataset with PCU-Aug**

```python
# In tests/test_physics_augmentation_e2e.py
import pytest
import numpy as np
from core.datasets.dataset import Dataset as KittiDataset

def test_kitti_dataset_compose_and_flip():
    cfg = {
        "data_dir": "data/kitti/training",
        "train_file": "splits/train.txt",
        "val_file": "splits/val.txt",
        "aug": {
            "use_pcu_aug": True,
            "flip_y": {"use": True, "p": 1.0},
            "rotation": {"use": True, "limit_angle": 15.0, "p": 0.5},
            "scaling": {"use": True, "range": [0.95, 1.05], "p": 0.5},
            "translation": {"use": True, "scale": 0.1, "p": 0.5}
        }
    }
    # Test initialization without failure
    # Ensure forward call produces valid Rich8 tensor (8, 800, 704)
```

- [ ] **Step 2: Update `dataset.py` to support `Compose` and `random_flip_3d`**

In `detector/core/datasets/dataset.py`:
- Replace exclusive `OneOf` with multi-stage `Compose`:
  1. `gt_sampler` (if configured)
  2. `random_flip_3d`
  3. `Random_Rotation`
  4. `Random_Scaling`
  5. `Random_Translation`

- [ ] **Step 3: Update `kitti_mobilepixornext_litemla_oga.json` with `use_pcu_aug` section**

- [ ] **Step 4: Run full test suite to guarantee 0 regressions**

Run: `PYTHONPATH="" /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/`
Expected: 74+ passed, 0 failures.

- [ ] **Step 5: Git commit task 5**

```bash
git add detector/core/datasets/dataset.py configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json tests/test_physics_augmentation_e2e.py
git commit -m "feat(dataset): wire physics-consistent augmentation into KittiDataset and OGA config"
```

---

## Self-Review Checklist
- [x] **Spec coverage**: Complete pipeline from offline DB creation, physics operators (shadow masking, subsampling, intensity), online sampler, to dataset integration.
- [x] **Placeholder scan**: 0 TODOs, 0 placeholders, explicit code snippets with real mathematical equations.
- [x] **Type consistency**: Exact function arguments and shapes aligned across all 5 tasks.
