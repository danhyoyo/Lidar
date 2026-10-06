# Omni-PCU: Unified Physics-Consistent & Curricular 3D LiDAR Database Sampler Specification

**Date:** 2026-10-06  
**Author:** AI Pair Programmer & duyennh  
**Target Module:** `detector/core/datasets/augmentor/omni_sampler.py`  
**Configuration Profile:** `configs/augmentation/omni_pcu_gt.json`  
**Status:** Design Proposal & Specification  

---

## 1. Executive Summary & Motivation

Ground-Truth (GT) database copy-paste sampling—originally popularized by SECOND and standardized in OpenPCDet—is the foundational data augmentation technique for 3D LiDAR object detection. While effective at addressing class imbalance (e.g., Pedestrians and Cyclists), contemporary implementations suffer from critical deficiencies:

1. **Physical & Sensor Blindness (OpenPCDet / MMDetection3D):**
   - **Density Anomaly:** An object scanned at $5\text{m}$ (dense point cloud, thousands of returns) pasted at $60\text{m}$ retains full density, severely distorting the detector's learned distance-feature priors.
   - **Static Obstacle / Wall Penetration:** Conventional samplers check collisions solely between 3D bounding boxes (`box-to-box`). Sampled objects frequently penetrate unannotated static obstacles, trees, curbs, or background walls.
   - **Floating Objects:** Naive $z$-coordinate preservation creates objects floating in mid-air or buried under uneven terrain.

2. **Computational Overhead & Latency (PCU Sampler):**
   - While the PCU branch (`feature/pillar5-pcu-augmentation`) introduces physics heuristics (ground support checks, distance-adaptive subsampling, solid-box shadow masking, and anti-wall collision), it suffers from severe CPU bottlenecks in PyTorch `DataLoader` workers due to iterative ray-box intersection loops, local point percentile searches, and extensive rejection iterations (~40–80ms per scene).

3. **Static Difficulty Imbalance (CVPR 2023 COM insight):**
   - All standard samplers draw uniformly from the database across all training epochs. In early epochs, complex, heavily occluded, distant objects inject high gradient noise when the backbone feature extractor has not stabilized. In late epochs, sampling predominantly easy close-range objects yields marginal learning signals.

### The Omni-PCU Solution
**Omni-PCU** unifies the strengths of all four reference frameworks into a single cohesive, high-throughput, physically grounded, and curriculum-guided augmentation engine:
- **From OpenPCDet:** Strict train-split provenance hashing (SHA-256), unified centered geometry $[x, y, z_{\text{center}}, l, w, h, \text{yaw}]$, scene-wide quota balancing (`limit_whole_scene`), and standalone JSON manifest metadata.
- **From MMDetection3D:** $O(1)$ analytical road plane height projection (`planes/*.txt`), vectorized Separating Axis Theorem (SAT) collision screening, and modular transform design.
- **From PCU:** 3-tier stratified long-range placement ($50\text{m}-70\text{m}$ coverage), distance-adaptive $(r_{\text{origin}} / r_{\text{target}})^2$ density subsampling, anti-wall static obstacle rejection, and atomic transactional visibility rollback.
- **From COM (CVPR 2023):** Curriculum-guided difficulty scheduling (transitioning smoothly from dense/proximal objects to sparse/distant objects over epochs) and an in-memory worker LRU cache to eliminate disk I/O bottlenecks.

---

## 2. Architectural Blueprint

```
                     ┌─────────────────────────────────────────────────────────┐
                     │          Offline Database Preprocessing Engine           │
                     │         (tools/kitti_training_pipeline/build_gt_db)     │
                     └────────────────────────────┬────────────────────────────┘
                                                  │ Outputs:
                                                  │ • dbinfos_train.json + crops (.bin)
                                                  │ • SHA-256 train split hash
                                                  │ • Computed r_origin, density, facade
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │                 OmniDataBaseSampler                     │
                     │          (detector/core/datasets/augmentor)             │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
 ┌────────────────────────────────────────────────┴────────────────────────────────────────────────┐
 │                                                                                                 │
 │  [Stage 1: Curriculum Grouping & Scheduling] (COM inspired)                                     │
 │  • Partition database into Difficulty Tiers (Easy: r < 30m, pts >= 50; Hard: r >= 30m / sparse) │
 │  • Dynamic probability P_hard(epoch) = min(1.0, epoch / warmup_epochs) * P_target               │
 │                                                                                                 │
 │  [Stage 2: 3-Tier Stratified Spatial Placement] (PCU inspired)                                  │
 │  • Propose poses along driving corridor: Near (8-25m: 30%), Mid (25-45m: 40%), Far (45-66m: 30%)│
 │                                                                                                 │
 │  [Stage 3: Vectorized Fast Collision Detection] (MMDet3D inspired)                              │
 │  • Broad-phase: Bounding circle radius screening O(1)                                           │
 │  • Narrow-phase: Vectorized 2D BEV Separating Axis Theorem (SAT) with safety margin             │
 │                                                                                                 │
 │  [Stage 4: Analytical O(1) Road Plane Ground Snapping] (MMDet3D inspired)                       │
 │  • Query KITTI plane equation: ax + by + cz + d = 0                                             │
 │  • Exact ground z offset: dz = (a*x + b*y + c*z_target + d) -> Snap box & points O(1)           │
 │                                                                                                 │
 │  [Stage 5: Physics & Sensor Fidelity Engine] (PCU inspired)                                     │
 │  • Anti-Wall / Static Collision: Reject if volume overlaps non-ground scene points (z > z_grd+0.35m)│
 │  • Physical Density Scaling: Subsample points by min(1.0, (r_origin / r_target)^2)              │
 │                                                                                                 │
 │  [Stage 6: Atomic Transactional Safety & Visibility Rollback] (PCU inspired)                   │
 │  • Compute post-insertion visibility of pre-existing scene objects                              │
 │  • Rollback candidate if any original object loses > 50% points or drops below 5 points        │
 │                                                                                                 │
 │  [Stage 7: In-Memory Bounded LRU Cache] (COM inspired)                                          │
 │  • Bounded per-worker RAM cache for decoded object point chunks -> Zero redundant disk reads   │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical & Algorithmic Formulation

### 3.1. Curriculum Difficulty Scheduling (COM Component)
Each database entry $i \in \mathcal{D}_c$ for class $c$ is characterized by its original sensor distance $r_{\text{origin}, i} = \sqrt{x_i^2 + y_i^2}$ and point count $N_i$. We define the sample difficulty metric $\delta_i \in [0, 1]$:
$$\delta_i = \frac{1}{2} \left( \min\left(1.0, \frac{r_{\text{origin}, i}}{R_{\text{max}}}\right) + \max\left(0.0, 1.0 - \frac{N_i}{N_{\text{ref}, c}}\right) \right)$$
where $R_{\text{max}} = 70.0\text{m}$ and $N_{\text{ref}, c}$ is the class-specific median point count ($50$ for Pedestrian, $100$ for Cyclist, $300$ for Car).

Entries are partitioned into two pools:
$$\mathcal{P}_{\text{easy}} = \{i \mid \delta_i \le \tau_{\text{diff}}\}, \quad \mathcal{P}_{\text{hard}} = \{i \mid \delta_i > \tau_{\text{diff}}\} \quad (\text{default } \tau_{\text{diff}} = 0.45)$$

Given the current training epoch $E$ and curriculum warmup epochs $E_{\text{warm}}$ (default: $10$):
$$\lambda(E) = \min\left(1.0, \frac{E}{E_{\text{warm}}}\right)$$
The probability of drawing from $\mathcal{P}_{\text{hard}}$ is:
$$P(\text{hard} \mid E) = \lambda(E) \cdot P_{\text{hard\_target}} + (1 - \lambda(E)) \cdot P_{\text{hard\_base}}$$
*(Recommended defaults: $P_{\text{hard\_base}} = 0.15, P_{\text{hard\_target}} = 0.70$)*.

### 3.2. 3-Tier Stratified Range Sampling (PCU Component)
Instead of placing objects at their source coordinates, candidates are dynamically assigned a pose $(x, y, \psi)$ within class-specific navigable corridors:
- **Tier 1 (Near: $8.0\text{m} \le x < 25.0\text{m}$):** Selection probability $w_{\text{near}} = 0.30$.
- **Tier 2 (Mid: $25.0\text{m} \le x < 45.0\text{m}$):** Selection probability $w_{\text{mid}} = 0.40$.
- **Tier 3 (Far: $45.0\text{m} \le x \le 65.5\text{m}$):** Selection probability $w_{\text{far}} = 0.30$.
- **Lateral Corridor Bounds ($y$):**
  $$|y| \le \begin{cases} 15.0\text{m} & \text{for Pedestrian} \\ 13.0\text{m} & \text{for Cyclist} \\ 9.5\text{m} & \text{for Car} \end{cases}$$
- **Orientation:** $\psi \sim \mathcal{U}(-\pi, \pi)$.

### 3.3. Analytical Road Plane Snapping (MMDetection3D Component)
Given the plane normal vector and intercept $\mathbf{n} = [a, b, c, d]^T$ from KITTI calibration ($ax + by + cz + d = 0$):
The exact ground level $z_{\text{ground}}$ at horizontal position $(x, y)$ is computed in $O(1)$:
$$z_{\text{ground}}(x, y) = -\frac{a \cdot x + b \cdot y + d}{c}$$
The target 3D bounding box center and object points are translated along $z$:
$$z_{\text{box, center}} = z_{\text{ground}}(x, y) + \frac{h}{2}$$
$$\mathbf{p}_{\text{world}} = \mathbf{R}(\psi) \cdot \mathbf{p}_{\text{local}} + \begin{bmatrix} x \\ y \\ z_{\text{ground}}(x, y) + \frac{h}{2} \end{bmatrix}$$

### 3.4. Physics-Consistent Density Scaling (PCU Component)
According to LiDAR optics and beam divergence, the surface area illuminated by a laser beam scales quadratically with range $r$. If an object captured at $r_{\text{origin}}$ is moved to $r_{\text{target}} > r_{\text{origin}}$, the expected point retention ratio is:
$$\rho = \min\left(1.0, \left(\frac{r_{\text{origin}}}{r_{\text{target}}}\right)^2\right)$$
Points are randomly Bernoulli subsampled:
$$\mathbf{m}_{\text{keep}} \sim \text{Bernoulli}(\rho), \quad N_{\text{retained}} = \sum \mathbf{m}_{\text{keep}}$$
*Safety rule:* If $N_{\text{original}} \ge 5$ and $N_{\text{retained}} < 5$, retain a random subset of exactly 5 points to preserve minimal geometric recognizability.

### 3.5. Anti-Wall Static Obstacle Collision Rejection
To prevent pasting objects into background geometry (buildings, trees, acoustic walls):
1. Extract existing scene points falling inside the candidate's expanded 3D bounding box $[x, y, z_{\text{ground}}, l, w, h]$.
2. Filter points elevated above road level:
   $$\mathcal{S}_{\text{obstacle}} = \{\mathbf{p} \in \mathcal{P}_{\text{scene}} \cap \text{Box} \mid \mathbf{p}_z > z_{\text{ground}} + \Delta z_{\text{curb}}\}$$
   with curb clearance threshold $\Delta z_{\text{curb}} = 0.35\text{m}$.
3. If $|\mathcal{S}_{\text{obstacle}}| > N_{\text{obstacle\_thresh}}$ (default: $3$), **reject candidate immediately** without altering the scene.

### 3.6. Transactional Visibility & Atomic Rollback
Pasting a new foreground object must not catastrophically occlude pre-existing ground truth targets:
- For each pre-existing ground truth object $k$ with baseline point count $B_k$:
  $$R_k = \frac{|\mathcal{P}_k \setminus \text{Box}_{\text{candidate}}|}{B_k}$$
- If any $R_k < \tau_{\text{vis}}$ (default: $0.50$) or count falls below $\min(B_k, 5)$, the entire insertion is aborted (rollback), leaving the scene and previous insertions unmodified.

---

## 4. Component Structure & API Design

### 4.1. File Layout
```
detector/core/datasets/augmentor/
├── __init__.py
├── data_augmentor.py             # Updated to register 'omni_gt_sampling'
├── database_sampler.py           # Existing OpenPCDet sampler (preserved)
├── omni_sampler.py               # NEW: Core Omni-PCU DataBaseSampler
├── omni_geometry.py              # NEW: Vectorized SAT & plane projection ops
└── omni_cache.py                 # NEW: In-memory LRU point cache & serialization
```

### 4.2. Class Interface: `OmniDataBaseSampler`
```python
class OmniDataBaseSampler:
    """State-of-the-Art Physics-Consistent & Curricular GT Sampler.
    
    Combines OpenPCDet integrity, MMDet3D road planes, PCU physics,
    and COM curriculum learning.
    """
    def __init__(
        self,
        root_path: Path | str,
        config: dict,
        class_names: dict[str, int],
        *,
        geometry: dict | None = None,
        allowed_frame_ids: list[str] | None = None,
    ):
        ...
        
    def set_epoch(self, epoch: int) -> None:
        """Update curriculum difficulty state for the current epoch."""
        ...

    def __call__(
        self,
        points: np.ndarray,
        boxes: np.ndarray,
        road_plane: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute atomic sampling with SAT collision, road plane snapping,
        density adjustment, anti-wall checks, and visibility protection.
        """
        ...
```

---

## 5. Configuration Schema (`omni_pcu_gt.json`)

```json
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
    "CACHE_SIZE_MB": 128
  }
}
```

---

## 6. Edge Cases & Mitigation Strategies

| Scenario / Edge Case | Risk | Mitigation |
| :--- | :--- | :--- |
| **Missing Road Plane File** | Frame lacks `planes/{id}.txt` (e.g., non-KITTI dataset or corrupted calibration). | Graceful fallback: derive local ground $z$ from the 5th percentile of scene points within radius $r \in [2.5, 4.0]\text{m}$. |
| **DataLoader Multiprocessing Forking** | Workers inherit parent RNG or cached open file descriptors. | Lazy initialization: re-seed NumPy RNG and clear cache pointers on worker spawn / `__setstate__`. |
| **Dense Scene Saturation** | Scene already packed with vehicles; 15 sampling attempts fail. | Strict transaction isolation: abort smoothly after `MAX_ATTEMPTS` without blocking training or corrupting current frame. |
| **Tiny Database for Rare Classes** | Very few Cyclist samples causes repeated sampling within the same frame. | Cycle detection with permuted indexing: ensure no duplicate object instance is inserted twice into the same scene. |
| **Single-Beam / Zero Points at Range** | Extreme $1/r^2$ subsampling empties an object. | Enforce minimum point floor: guarantee $\ge 5$ points if the source object had $\ge 5$ points. |

---

## 7. Verification & Testing Matrix

The implementation will be verified through a comprehensive test suite in `tests/test_omni_pcu_sampler.py`:

1. **Parity & Invariant Tests:**
   - Round-trip box transformation: center $z \leftrightarrow$ bottom $z$.
   - Train split provenance enforcement: verify ValueError on mismatched SHA-256 hash or foreign frame IDs.
2. **Physics & Geometric Fidelity:**
   - Verify distance subsampling mathematically follows $\rho = (r_{\text{origin}} / r_{\text{target}})^2 \pm 5\%$.
   - Anti-wall obstacle rejection triggers on synthetic obstacle clouds ($z > z_{\text{ground}} + 0.35$).
   - Road plane snapping matches plane equation $|a x + b y + c z + d| < 10^{-4}$.
3. **Curriculum Dynamics:**
   - Verify sample difficulty distribution shifts monotonically towards hard candidates from Epoch 0 to Epoch 10.
4. **Performance & Benchmarking:**
   - Execution benchmark on synthetic $100\text{k}$-point scenes: target runtime $\le 4.5\text{ms}$ per frame.
   - Cache hit rate verification $> 85\%$ across multiple epochs.
5. **End-to-End Pipeline Smoke Test:**
   - Real DataLoader execution with 2 workers.
   - MobilePIXOR/Pillar forward + backward training pass for 1 full epoch producing a valid checkpoint.
