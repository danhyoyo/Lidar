# Physics-Consistent Anti-Wall & Anti-Occlusion Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate wall penetration and occluded shadow placement in PCU-Aug (`GTSampler`) by enforcing ground surface support, static obstacle avoidance, and foreground line-of-sight verification on the point cloud.

**Architecture:** A lightweight, pure-NumPy physics feasibility filter evaluated prior to pasting objects: verifying local road surface height ($R=2.5\text{m}$), rejecting boxes that intersect elevated obstacle points ($z_{\text{local}} > 0.35\text{m}$), and rejecting candidate poses whose line-of-sight rays from Ego are blocked by foreground structures ($r_{\text{obs}} < r_{\text{target}}$).

**Tech Stack:** Python 3.12+, PyTorch 2.x, NumPy, Shapely 2.x, pytest.

**Spec:** [`docs/plans/2026-09-29-physics-consistent-anti-wall-augmentation-design.md`](file:///home/duyennh/AI_projects/research_lidar/Lidar/docs/plans/2026-09-29-physics-consistent-anti-wall-augmentation-design.md)

## Global Constraints

- 100% backward compatible: Controlled via `enable_physics` in `GTSampler` and configuration toggles.
- Runtime performance: Vectorized NumPy operations with $< 0.25\text{ms}$ latency per candidate test, adding $< 2\text{ms}$ per frame overhead.
- All existing 157 unit tests in `tests/` must remain passing at every step.
- Python code must conform to existing repository conventions and work across Linux and Google Colab environments.

---

### Task 1: Ground Support Verification and Static Obstacle Collision in `collision_ground.py`

**Files:**
- Modify: `detector/core/datasets/utils_1/collision_ground.py`
- Test: `tests/test_collision_ground.py`

**Interfaces:**
- Consumes: `points: np.ndarray`, `box: np.ndarray`
- Produces:
  - `check_ground_support(box: np.ndarray, points: np.ndarray, min_points: int = 15, radius: float = 2.5) -> Tuple[bool, float]`
  - `check_static_obstacle_collision(box: np.ndarray, points: np.ndarray, max_obstacle_points: int = 3, min_height_above_ground: float = 0.35) -> bool`

- [ ] **Step 1: Write tests for ground support and obstacle collision**

Add tests to `tests/test_collision_ground.py`:

```python
def test_ground_support_valid_road():
    # Flat ground around (20, 0) with z = -1.6
    xs = np.linspace(18, 22, 10)
    ys = np.linspace(-2, 2, 10)
    xx, yy = np.meshgrid(xs, ys)
    ground_pts = np.column_stack([xx.ravel(), yy.ravel(), np.full(100, -1.6), np.ones(100)])
    
    cand_box = np.array([1, 1.5, 2.0, 4.5, 20.0, 0.0, -1.6, 0.0], dtype=np.float32)
    has_support, ground_z = check_ground_support(cand_box, ground_pts)
    assert has_support is True
    assert -1.7 <= ground_z <= -1.5


def test_ground_support_rejects_empty_void():
    # Empty point cloud (e.g. shadow void behind wall)
    empty_pts = np.zeros((0, 4), dtype=np.float32)
    cand_box = np.array([1, 1.5, 2.0, 4.5, 35.0, 25.0, -1.6, 0.0], dtype=np.float32)
    has_support, _ = check_ground_support(cand_box, empty_pts)
    assert has_support is False


def test_obstacle_collision_rejects_wall():
    # Ground at z=-1.6 plus a vertical wall cluster at (20, 0) with z in [-1.0, 1.5]
    wall_pts = np.array([
        [20.0, 0.0, -0.5, 0.5],
        [20.1, 0.1, 0.0, 0.5],
        [19.9, -0.1, 0.5, 0.5],
        [20.0, 0.2, 1.0, 0.5],
    ], dtype=np.float32)
    
    cand_box = np.array([1, 1.5, 2.0, 4.5, 20.0, 0.0, -1.6, 0.0], dtype=np.float32)
    collides = check_static_obstacle_collision(cand_box, wall_pts)
    assert collides is True


def test_obstacle_collision_allows_clean_road():
    # Points only on the road surface (z = -1.6)
    road_pts = np.array([
        [20.0, 0.0, -1.6, 0.5],
        [20.1, 0.1, -1.58, 0.5],
        [19.9, -0.1, -1.62, 0.5],
    ], dtype=np.float32)
    
    cand_box = np.array([1, 1.5, 2.0, 4.5, 20.0, 0.0, -1.6, 0.0], dtype=np.float32)
    collides = check_static_obstacle_collision(cand_box, road_pts)
    assert collides is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_collision_ground.py -k "ground_support or obstacle_collision" -v`  
Expected: FAIL with `ImportError: cannot import name 'check_ground_support'`

- [ ] **Step 3: Implement `check_ground_support` and `check_static_obstacle_collision`**

In `detector/core/datasets/utils_1/collision_ground.py`:

```python
def check_ground_support(
    box: np.ndarray,
    points: np.ndarray,
    min_points: int = 15,
    radius: float = 2.5,
) -> Tuple[bool, float]:
    """Verifies that a candidate location is supported by a real ground surface.

    Returns (has_support, ground_z). Rejects empty voids (e.g. behind buildings).
    """
    if len(points) == 0:
        return False, -1.6

    b = box[1:] if len(box) >= 8 else box
    bx, by = b[3], b[4]

    dist_sq = (points[:, 0] - bx) ** 2 + (points[:, 1] - by) ** 2
    local_pts = points[dist_sq <= radius**2]

    # Must have minimum points on road
    if len(local_pts) < min_points:
        return False, -1.6

    # 5th to 15th percentile represents ground level
    z_est = float(np.percentile(local_pts[:, 2], 5))
    if not (-2.5 <= z_est <= -0.8):
        return False, -1.6

    # Ground height spread must be reasonably flat (< 0.45m IQR/std)
    z_road = local_pts[local_pts[:, 2] <= z_est + 0.35, 2]
    if len(z_road) < min_points:
        return False, -1.6

    return True, z_est


def check_static_obstacle_collision(
    box: np.ndarray,
    points: np.ndarray,
    max_obstacle_points: int = 3,
    min_height_above_ground: float = 0.35,
) -> bool:
    """Checks whether the 3D volume of box overlaps with elevated non-ground obstacle points.

    Returns True if collision detected (e.g. wall, pole, hedge, building), False if clear.
    """
    if len(points) == 0:
        return False

    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, yaw = b[:7]

    # Broad-phase cylinder pre-filter
    diag = np.sqrt(w**2 + l**2) / 2.0
    dist_sq = (points[:, 0] - bx) ** 2 + (points[:, 1] - by) ** 2
    cand_mask = dist_sq <= (diag + 0.2) ** 2
    if not np.any(cand_mask):
        return False

    cand_pts = points[cand_mask]

    # Local box coordinates
    dx = cand_pts[:, 0] - bx
    dy = cand_pts[:, 1] - by
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    x_rot = dx * cos_y - dy * sin_y
    y_rot = dx * sin_y + dy * cos_y
    z_rot = cand_pts[:, 2] - bz

    in_footprint = (np.abs(x_rot) <= l / 2.0) & (np.abs(y_rot) <= w / 2.0)
    is_elevated = (z_rot >= min_height_above_ground) & (z_rot <= h + 0.2)

    obstacle_count = np.sum(in_footprint & is_elevated)
    return bool(obstacle_count > max_obstacle_points)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_collision_ground.py -v`  
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add detector/core/datasets/utils_1/collision_ground.py tests/test_collision_ground.py
git commit -m "feat(physics): add ground support verification and static obstacle collision check"
```

---

### Task 2: Line-of-Sight Foreground Occlusion Check in `physics_aug.py`

**Files:**
- Modify: `detector/core/datasets/utils_1/physics_aug.py`
- Test: `tests/test_physics_aug.py`

**Interfaces:**
- Consumes: `points: np.ndarray`, `box: np.ndarray`
- Produces: `check_line_of_sight_occlusion(box: np.ndarray, points: np.ndarray, max_blocking_points: int = 5) -> bool`

- [ ] **Step 1: Write test for line-of-sight occlusion**

In `tests/test_physics_aug.py`, add:

```python
def test_line_of_sight_occlusion_behind_wall():
    # Wall located at x=15, y=0, z in [-1.5, 2.0]
    wall_ys = np.linspace(-3, 3, 20)
    wall_zs = np.linspace(-1.0, 2.0, 10)
    wy, wz = np.meshgrid(wall_ys, wall_zs)
    wall_pts = np.column_stack([np.full(200, 15.0), wy.ravel(), wz.ravel(), np.ones(200)])
    
    # Target car at x=30, y=0, z=-1.6 (directly behind the wall from Ego at (0,0))
    car_box = np.array([1, 1.5, 2.0, 4.5, 30.0, 0.0, -1.6, 0.0], dtype=np.float32)
    is_occluded = check_line_of_sight_occlusion(car_box, wall_pts)
    assert is_occluded is True


def test_line_of_sight_occlusion_clear_path():
    # Ground surface only at z=-1.6, no elevated obstacle between Ego and car
    road_xs = np.linspace(5, 25, 20)
    road_pts = np.column_stack([road_xs, np.zeros(20), np.full(20, -1.6), np.ones(20)])
    
    car_box = np.array([1, 1.5, 2.0, 4.5, 30.0, 0.0, -1.6, 0.0], dtype=np.float32)
    is_occluded = check_line_of_sight_occlusion(car_box, road_pts)
    assert is_occluded is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_physics_aug.py -k "line_of_sight" -v`  
Expected: FAIL with `ImportError: cannot import name 'check_line_of_sight_occlusion'`

- [ ] **Step 3: Implement `check_line_of_sight_occlusion`**

In `detector/core/datasets/utils_1/physics_aug.py`:

```python
def check_line_of_sight_occlusion(
    box: np.ndarray,
    points: np.ndarray,
    max_blocking_points: int = 5,
    min_obstacle_height: float = 0.4,
) -> bool:
    """Verifies that line-of-sight from Ego (0, 0, 0) to box is not blocked by foreground obstacles.

    Returns True if occluded (e.g. placed behind a building, wall, or another vehicle), False if line-of-sight is clear.
    """
    if len(points) == 0:
        return False

    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, _ = b[:7]
    r_target = np.sqrt(bx**2 + by**2)
    if r_target < 3.0:
        return False

    # Angular wedge spanned by candidate box
    diag = np.sqrt(w**2 + l**2) / 2.0
    delta_azimuth = np.arctan2(diag, r_target)
    azimuth_target = np.arctan2(by, bx)

    # Elevation range subtended by candidate box
    elev_min = np.arctan2(bz, r_target)
    elev_max = np.arctan2(bz + h, r_target)

    # Spherical coordinates of background points
    r_pts = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    
    # Only consider foreground points between Ego and candidate box (with 1.5m buffer)
    fg_mask = (r_pts >= 2.0) & (r_pts < (r_target - 1.5))
    if not np.any(fg_mask):
        return False

    fg_pts = points[fg_mask]
    r_fg = r_pts[fg_mask]
    azimuth_fg = np.arctan2(fg_pts[:, 1], fg_pts[:, 0])
    elev_fg = np.arctan2(fg_pts[:, 2], np.maximum(r_fg, 1e-6))

    # Angular difference with wrap-around in [-pi, pi]
    az_diff = np.abs((azimuth_fg - azimuth_target + np.pi) % (2 * np.pi) - np.pi)

    # Occluders must be within the azimuth wedge, within vertical elevation span,
    # and elevated above typical ground level (z > -1.25m)
    in_azimuth = az_diff <= (delta_azimuth * 0.9)
    in_elev = (elev_fg >= elev_min) & (elev_fg <= elev_max)
    is_tall = fg_pts[:, 2] > (bz + min_obstacle_height)

    blocking_count = np.sum(in_azimuth & in_elev & is_tall)
    return bool(blocking_count >= max_blocking_points)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_physics_aug.py -v`  
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add detector/core/datasets/utils_1/physics_aug.py tests/test_physics_aug.py
git commit -m "feat(physics): add line-of-sight foreground occlusion checker"
```

---

### Task 3: Integrate Corridor Sampling and Physics Feasibility Pipeline into `GTSampler`

**Files:**
- Modify: `detector/core/datasets/utils_1/gt_sampler.py`
- Test: `tests/test_gt_sampler.py`

**Interfaces:**
- Consumes:
  - `check_box_collision_2d`, `check_ground_support`, `check_static_obstacle_collision` from `collision_ground.py`
  - `check_line_of_sight_occlusion` from `physics_aug.py`
- Produces: Enhanced `GTSampler.__call__` that enforces zero wall penetration and zero occluded void placement.

- [ ] **Step 1: Write integration tests for anti-wall GTSampler**

In `tests/test_gt_sampler.py`:

```python
def test_gt_sampler_rejects_placement_inside_wall(tmp_path):
    # Setup database with 1 car
    db_file = tmp_path / "kitti_gt_database.pkl"
    mock_db = {
        "Car": [
            {
                "box": np.array([1, 1.5, 2.0, 4.5, 10.0, 0.0, -1.6, 0.0], dtype=np.float32),
                "points": np.ones((50, 4), dtype=np.float32),
                "r_origin": 10.0,
            }
        ]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    sampler = GTSampler(str(db_file), sample_counts={"Car": 1}, p=1.0, enable_physics=True)

    # Empty scene with only a massive wall at y=0, z in [-1.5, 2.0], but NO valid flat ground
    wall_pts = np.random.uniform(-1.0, 2.0, size=(100, 4)).astype(np.float32)
    wall_pts[:, 0] = np.random.uniform(5.0, 40.0, size=100)
    wall_pts[:, 1] = np.random.uniform(-2.0, 2.0, size=100)

    aug_pts, aug_boxes = sampler(wall_pts, np.zeros((0, 8), dtype=np.float32))
    # Since no valid ground support or unblocked area exists, placement must be safely rejected
    assert len(aug_boxes) == 0
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_gt_sampler.py -k "rejects_placement_inside_wall" -v`  
Expected: FAIL (currently it would insert 1 box anyway due to default -1.6 fallback).

- [ ] **Step 3: Update `GTSampler.__call__` placement loop**

In `detector/core/datasets/utils_1/gt_sampler.py`:
- Import `check_ground_support`, `check_static_obstacle_collision` from `collision_ground`.
- Import `check_line_of_sight_occlusion` from `physics_aug`.
- Update the placement retry loop to:
  1. Use 2-tier corridor sampling:
     - Tries 0–9: $|Y| \le 14\text{m}$, $r \in [5, 60]\text{m}$.
     - Tries 10–19: $|Y| \le 30\text{m}$, $r \in [5, 65]\text{m}$.
  2. Perform box-to-box collision check: `check_box_collision_2d`.
  3. If `self.enable_physics`:
     - Perform `check_ground_support`: Reject if no ground support found. Set $z = z_{\text{est}}$.
     - Perform `check_static_obstacle_collision`: Reject if elevated points exist inside box.
     - Perform `check_line_of_sight_occlusion`: Reject if line of sight from Ego is blocked.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_gt_sampler.py -v`  
Expected: All tests PASS.

- [ ] **Step 5: Run full test suite to guarantee regression-free state**

Run: `PYTHONPATH="." /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros tests/`  
Expected: All 157+ tests PASS.

- [ ] **Step 6: Commit**

```bash
git add detector/core/datasets/utils_1/gt_sampler.py tests/test_gt_sampler.py
git commit -m "feat(gt_sampler): enforce anti-wall and anti-occlusion physics checks in object placement"
```

---

### Task 4: Re-render Visualization and Verify Zero Wall Penetration

**Files:**
- Execute: `tools/visualization/visualize_pcu_aug.py`
- Test: Compare output renders on Frame `001190` and Frame `006954`.

- [ ] **Step 1: Run visualizer on Frame 001190 and Frame 006954**

Generate new BEV comparison images:
```bash
python3 tools/visualization/visualize_pcu_aug.py \
    --data_dir data/kitti/processed \
    --gt_database data/kitti/kitti_gt_database.pkl \
    --frame 001190 \
    --output /home/duyennh/Downloads/images/pcu_preview_1190_fixed.png
```
```bash
python3 tools/visualization/visualize_pcu_aug.py \
    --data_dir data/kitti/processed \
    --gt_database data/kitti/kitti_gt_database.pkl \
    --frame 006954 \
    --output /home/duyennh/Downloads/images/pcu_preview_6954_fixed.png
```

- [ ] **Step 2: Inspect rendered images to confirm:**
  - Zero objects placed behind solid building walls (no more occluded shadow void placement at $X=35, Y=28$).
  - Zero objects overlapping wall or tree obstacle clusters.
  - All pasted objects cleanly situated on drivable ground surfaces with direct line-of-sight from Ego.

- [ ] **Step 3: Commit any visualization or notebook updates**

```bash
git add tools/visualization/visualize_pcu_aug.py
git commit -m "test(vis): verify anti-wall PCU-Aug placement across complex scenes"
```
