# Physics-Consistent Anti-Wall & Anti-Occlusion 3D Augmentation Design

**Date:** 2026-09-29  
**Status:** Approved  
**Author:** Pair programming (Antigravity & User)  
**Target:** Pillar 5: Physics-Consistent Online GT Augmentation (`PCU-Aug`) in MobilePIXORNeXt

---

## 1. Problem Statement & Motivation
Visual inspection of BEV augmentation preview renders (Frame `001190` and Frame `006954`) identified two critical physical realism flaws present in naive / open-source GT-Aug implementations (SECOND, OpenPCDet):
1. **Wall Penetration (Chèn / Cắm vào tường):** Objects placed overlapping building facades, walls, or roadside barriers. Background points inside the box volume were hollowed out, creating floating objects intersecting solid obstacles.
2. **Foreground Occlusion Violation (Dán sau tường / Xuyên tường):** Objects placed in occluded shadow voids behind building walls (e.g. at $X=35\text{m}, Y=28\text{m}$ behind a wall at $X=20\text{m}, Y=18\text{m}$). In real LiDAR, beams cannot penetrate concrete walls, making objects behind opaque structures physically unobservable from Ego $(0, 0, 0)$.

---

## 2. Selected Architecture: Point-Based Physics Feasibility Filter

To solve these flaws without introducing bulky mesh/voxel maps or heavy RANSAC loops, we implement a **Point-Based Physics Filter** directly operating on the point cloud with vectorized NumPy operations.

### Core Modules & Algorithms:

### 2.1 Road Corridor-Biased Sampling (`sample_candidate_pose`)
- In street scenes (KITTI), valid roads lie within lateral bounds $|Y| \le 15.0\text{m}$.
- Tries 1–10: Sample target position in the primary corridor ($r \in [5, 60]\text{m}$, $|Y| \le 15\text{m}$).
- Tries 11–20: Graceful fallback to the full FOV cone ($r \in [5, 65]\text{m}$, azimuth $\in [-\pi/4, \pi/4]$).
- Maximum attempts: 20 tries. If no collision-free, ground-supported, line-of-sight location is found, gracefully skip the candidate.

### 2.2 Ground Support Verification (`check_ground_support`)
- Points within $R = 2.5\text{m}$ of candidate center $(b_x, b_y)$ must contain at least 15 points with valid road height ($z \in [-2.5\text{m}, -0.8\text{m}]$).
- If point count $< 15$ or ground height variance indicates a non-ground void (e.g. empty space behind a wall), reject immediately.

### 2.3 Static Obstacle Collision Filter (`check_obstacle_collision_3d`)
- Transform nearby background points into candidate box local coordinates $(x_{\text{local}}, y_{\text{local}}, z_{\text{local}})$.
- Reject if there are $> 3$ background points inside the 2D footprint ($|x_{\text{local}}| \le l/2$, $|y_{\text{local}}| \le w/2$) that have height $z_{\text{local}} > 0.35\text{m}$ above the estimated ground surface (indicating walls, fences, poles, or trees).

### 2.4 Foreground Line-of-Sight Occlusion Filter (`check_line_of_sight_occlusion`)
- For candidate center $(b_x, b_y)$, compute ray distance $r_{\text{target}} = \sqrt{b_x^2 + b_y^2}$, azimuth $\theta_{\text{box}} = \arctan2(b_y, b_x)$, and angular span $\Delta \theta = \arctan2(\sqrt{w^2 + l^2}/2, r_{\text{target}})$.
- Scan foreground background points with $r_{\text{pt}} < r_{\text{target}} - 1.5\text{m}$ and $|\theta_{\text{pt}} - \theta_{\text{box}}| \le \Delta \theta$.
- Count obstacle points whose height exceeds road level ($z_{\text{pt}} > z_{\text{ground}} + 0.4\text{m}$) within the vertical field-of-view of the box.
- If $\ge 5$ foreground occluding points block line-of-sight from $(0, 0, 0)$, reject candidate.

---

## 3. Data Flow in GTSampler

```
[Candidate Pose Sampled]
          │
          ▼
   [1. 2D Box Collision (Shapely)] ───────► (Overlap existing box) ──► Retry
          │ (Pass)
          ▼
   [2. Ground Support Verification] ──────► (Void / Behind wall) ────► Retry
          │ (Road surface confirmed)
          ▼
   [3. Static Obstacle Collision] ────────► (Wall / Tree points) ────► Retry
          │ (Clear volume)
          ▼
   [4. Line-of-Sight Occlusion] ──────────► (Occluded by wall) ──────► Retry
          │ (Direct unobstructed ray)
          ▼
   [Valid Placement Accepted]
          │
          ▼
   [Distance subsampling, intensity calibration, snap to ground, shadow masking]
```

---

## 4. Performance & Compatibility Guarantees
- Vectorized NumPy operations execute in $< 0.25\text{ms}$ per candidate check.
- Total augmentation latency per frame: $< 2\text{ms}$ (easily absorbed by 6 DataLoader workers).
- 100% backward compatible: Controlled via `enable_physics` or config flags.
- Zero extra external dependencies (uses standard NumPy and Shapely).
