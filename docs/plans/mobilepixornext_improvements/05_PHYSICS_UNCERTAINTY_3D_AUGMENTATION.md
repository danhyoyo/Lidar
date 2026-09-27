# Pillar 5: Physics-Consistent & Uncertainty-Guided 3D Augmentation (PCU-Aug)

## 1. Motivation & Problem Statement

Data augmentation is the primary driver of generalization in 3D LiDAR Object Detection. Unlike 2D perspective images, 3D point clouds operate in an absolute metric Euclidean space ($x, y, z$ in meters). 

However, existing literature (e.g. SECOND, PointPillars, OpenPCDet) relies on naive Copy-Paste augmentation (GT-Sampling) that suffers from **three severe physical violations and architectural blind spots**:

1. **The "Ghost Object" Occlusion Violation (Line-of-Sight Paradox)**:
   LiDAR is a line-of-sight laser sensor. When a physical vehicle stands at $20\text{m}$, laser beams cannot pass through its metal body. Yet standard GT-Sampling simply splices the object's points into the scene **without deleting background points behind the object**. The pasted vehicle appears transparent, as background points from $40\text{m}$ remain visible through the solid vehicle body.
2. **LiDAR Beam Divergence & Density Mismatch**:
   LiDAR point density decays inversely with the square of the distance ($1/R^2$). Pasting an object captured at $15\text{m}$ into a distant region ($50\text{m}$) produces an unnaturally dense cluster of points that violates physical sensor mechanics.
3. **Radiometric Intensity Anomaly in Rich8 BEV**:
   Our **MobilePixorNeXt** relies on 8-channel Rich8 BEV representations, specifically channel 4 (`intensity_max`), channel 5 (`intensity_mean`), and channel 8 (`range`). Laser reflectance intensity attenuates with distance ($P_r \propto 1/R^2$). Pasting objects across disparate ranges without radiometric calibration introduces severe channel distribution shifts into the BEV feature map.
4. **Static Uniform Sampling vs. Dynamic Multi-Task Uncertainty**:
   Standard methods sample all classes with static, uniform frequencies. However, rare and hard classes (Pedestrian, Cyclist) have significantly higher loss uncertainties during early and mid training.

**Solution**: **Physics-Consistent & Uncertainty-Guided 3D Augmentation (PCU-Aug)**:
- **Ray-Consistent Shadow Frustum**: Traces the angular cone of the inserted object and purges occluded background points.
- **Distance-Adaptive Subsampling & Radiometric Calibration**: Scales point density by $(R_1/R_2)^2$ and intensity by $(R_1/R_2)^{2-\gamma}$.
- **Uncertainty-Guided Curriculum Injection**: Adapts class paste frequencies dynamically using $\sigma_c^2$ from `TemperatureSoftmaxUncertainty`.

```
        Raw LiDAR Scene                           GT Object from DB (R1)
              │                                              │
              │                               [ Distance-Adaptive Subsample ]
              │                               [ Radiometric Calibrate ]
              │                               [ Snap to Ground Surface ]
              ▼                                              │
    [ 2D BEV Collision Check ] <─────────────────────────────┘
              │ (Pass)
              ▼
   [ Ray Shadow Masking ]  ===> (Purges background points occluded behind the object)
              │
              ▼
    [ Splice Points & Boxes ]
              │
              ▼
  [ 3D Horizontal Flip (y -> -y) ]
              │
              ▼
    [ Rich8 BEV Feature Grid ] ===> (Zero inference overhead, 100% physically consistent)
```

---

## 2. Mathematical Formulations & Physical Proofs

### 2.1. Ray-Consistent Shadow Frustum Formulation
Let sensor origin be $O = (0, 0, 0)$. An inserted 3D bounding box is parameterized as $B = (x, y, z, h, w, l, \theta)$.
The 8 vertices of $B$ in Cartesian coordinates are $C_k = (x_k, y_k, z_k), k \in \{1, \dots, 8\}$.

Convert corners to spherical coordinates:
$$\theta_k = \text{atan2}(y_k, x_k), \quad \phi_k = \text{atan2}\left(z_k, \sqrt{x_k^2 + y_k^2}\right), \quad r_k = \sqrt{x_k^2 + y_k^2 + z_k^2}$$

Define the bounding angular frustum of $B$:
$$\theta_{\min} = \min_{k} \theta_k, \quad \theta_{\max} = \max_{k} \theta_k$$
$$\phi_{\min} = \min_{k} \phi_k, \quad \phi_{\max} = \max_{k} \phi_k$$
$$r_{\text{near}} = \min_{k} r_k, \quad r_{\text{far}} = \max_{k} r_k$$

For any background point $P_{\text{bg}} = (x_{\text{bg}}, y_{\text{bg}}, z_{\text{bg}})$ in the scene:
$$P_{\text{bg}} \in \text{ShadowVolume}(B) \iff \begin{cases} \theta_{\min} \le \text{atan2}(y_{\text{bg}}, x_{\text{bg}}) \le \theta_{\max} \\ \phi_{\min} \le \text{atan2}(z_{\text{bg}}, r_{xy,\text{bg}}) \le \phi_{\max} \\ r_{\text{bg}} > r_{\text{far}} \end{cases}$$

All points satisfying $P_{\text{bg}} \in \text{ShadowVolume}(B)$ are purged from the point cloud.

### 2.2. Distance-Adaptive Density Subsampling
The solid angle subtended by an object of area $A$ at range $R$ is:
$$\Omega(R) = \frac{A \cdot \cos \psi}{R^2}$$

For a pulsed LiDAR with fixed angular resolution $(\Delta \theta, \Delta \phi)$, the expected number of laser hits $N(R)$ is proportional to $\Omega(R)$:
$$N(R) \propto \frac{1}{R^2}$$

When an object is extracted from origin range $R_1$ and placed at target range $R_2 > R_1$:
$$p_{\text{retain}} = \min\left(1.0, \frac{N(R_2)}{N(R_1)}\right) = \min\left(1.0, \left(\frac{R_1}{R_2}\right)^2\right)$$

Each point $p \in \mathcal{P}_{\text{obj}}$ is retained with Bernoulli trial probability $p_{\text{retain}}$.

### 2.3. Radiometric Intensity Calibration
The LiDAR radar range equation defines received power $P_r$:
$$P_r = \frac{P_t \cdot D_r^2 \cdot \rho \cdot \eta_{\text{atm}} \cdot \eta_{\text{sys}}}{4 R^2} \cdot \cos \alpha$$

Commercial automotive sensors (e.g. Velodyne HDL-64E) implement Automatic Gain Control (AGC) or piecewise time-varying gain $G(R) \approx R^\gamma$, where $\gamma \in [1.5, 2.0]$.
The detected intensity $I \in [0.0, 1.0]$ scales as:
$$I(R) \propto P_r \cdot G(R) \propto R^{\gamma - 2}$$

When relocating from $R_1$ to $R_2$:
$$I_{\text{new}} = \text{clip}\left(I_{\text{old}} \cdot \left(\frac{R_1}{R_2}\right)^{2 - \gamma}, 0.0, 1.0\right)$$
For Velodyne HDL-64E on KITTI, $\gamma \approx 1.7$, giving $(R_1/R_2)^{0.3}$.

### 2.4. Uncertainty-Guided Dynamic Injection Weighting
Our baseline uses homoscedastic uncertainty loss balancing (`TemperatureSoftmaxUncertainty`). Let $\sigma_c^2$ be the current estimated uncertainty variance for class $c \in \{\text{Car}, \text{Pedestrian}, \text{Cyclist}\}$.
The dynamic sampling weight $w_c$ is:
$$w_c = \frac{\exp(\sigma_c / \tau)}{\sum_{j=1}^{C} \exp(\sigma_j / \tau)}$$

When the network has high uncertainty / poor recall on Pedestrian or Cyclist, the sampling probability increases automatically, balancing the effective training distribution without manual hyperparameter retuning.

---

## 3. Integration Blueprint

### 3.1. File Structure
- `detector/core/datasets/utils_1/physics_aug.py`: Ray shadow masking, subsampling, intensity calibration, and 3D horizontal flip.
- `detector/core/datasets/utils_1/collision_ground.py`: BEV 2D collision detection and local ground plane height snapping.
- `detector/core/datasets/utils_1/gt_sampler.py`: Online physics-consistent sampler module.
- `tools/dataset_converter/create_gt_database.py`: Offline generator for `kitti_gt_database.pkl`.

### 3.2. Configuration Interface
In `configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json`:
```json
{
  "aug": {
    "use_pcu_aug": true,
    "flip_y": {"use": true, "p": 0.5},
    "rotation": {"use": true, "limit_angle": 15.0, "p": 0.7},
    "scaling": {"use": true, "range": [0.95, 1.05], "p": 0.7},
    "translation": {"use": true, "scale": 0.1, "p": 0.5},
    "gt_sampling": {
      "use": true,
      "p": 1.0,
      "database_path": "data/kitti/kitti_gt_database.pkl",
      "sample_counts": {"Car": 8, "Pedestrian": 6, "Cyclist": 6},
      "enable_shadow_masking": true,
      "enable_density_subsample": true,
      "enable_radiometric_calibration": true,
      "uncertainty_guided": true
    }
  }
}
```

---

## 4. Experimental Validation & Expected Gains

### 4.1. Ablation Hypothesis & Targets (vs. Baseline $M_0$)
- **Horizontal Flip 3D Alone**: $+1.0$ to $+1.5\%$ $AP_{3D}$ on Car.
- **Naive GT-Sampling (Standard SECOND)**: $+2.5\%$ on Car, $+6.0\%$ on Pedestrian, but causes shadow-artifact false positives.
- **PCU-Aug (Physics-Consistent & Uncertainty-Guided)**: $+3.5$ to $+4.5\%$ $AP_{3D}$ on Car, $+9.0$ to $+12.0\%$ on Pedestrian/Cyclist with **0 ms inference latency**.

### 4.2. Diagnostic Sanity Checks
1. **Ray Shadow Verification**: Verify that background LiDAR points inside the angular cone behind any inserted vehicle are strictly removed.
2. **Rich8 Range & Intensity Invariance**: Ensure generated Rich8 channels (`intensity_max`, `intensity_mean`, `range`) strictly match the metric coordinates of newly pasted objects without discontinuities.
