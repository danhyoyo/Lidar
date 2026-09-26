# Specification & Theoretical Foundations: OGA-Loss (Oriented Geometric Alignment & Adaptive Loss)

Status: **Proposed Research Contribution**  
Date: **2026-09-26**  
Authors: Antigravity AI Assistant & DuyenNH  
Target Application: 3D LiDAR Bird's Eye View (BEV) Object Detection (BEVNeXt / MobilePixor)

---

## 1. Executive Summary & Research Motivation

In 3D LiDAR object detection from a Bird's Eye View (BEV), the network predicts oriented bounding boxes represented by:
1. `cls`: Dense Gaussian center heatmaps across classes (`Car`, `Pedestrian`, `Cyclist`).
2. `offset`: Sub-grid metric continuous translation displacement $(dx, dy)$ from the BEV grid center.
3. `size`: Log-dimensional bounding box extents $(\log w, \log l)$.
4. `yaw`: Doubled-angle orientation vector $(\cos 2\theta, \sin 2\theta)$, providing $\pi$-rotational symmetry (front/back flip invariance of oriented bounding boxes).

### 1.1 Historical Codebase Context & UWAG Limitations
The legacy codebase historically utilized **UWAG** (Uncertainty-Weighted Loss with Adaptive Geometry). Although UWAG achieved high historical metrics, it suffers from severe scientific and engineering limitations:
- **Intellectual Property & Independence:** UWAG belongs to prior thesis/colleague work and **cannot be used directly**; we may only draw inspiration from its high-level concepts.
- **Flawed Geometric Formulation (Axis-Aligned Heuristic):** UWAG's geometric penalty (`_yaw_aware_bev_iou_loss`) computes IoU using an **axis-aligned bounding box** approximation, simply multiplied by a heuristic yaw similarity factor:
  $$\text{OrientedOverlap} = \text{IoU}_{\text{axis-aligned}} \times \frac{1 + \cos(2\Delta\theta)}{2}$$
  - For boxes oriented at $45^\circ$ or $90^\circ$, axis-aligned boxes distort the true intersection area drastically.
  - When two boxes are spatially disjoint (non-overlapping), $\text{IoU}_{\text{axis-aligned}} = 0$, causing complete gradient vanishing ($\nabla = 0$) for both translation and orientation!
- **Unbounded Homoscedastic Scale Drift & Instability:** UWAG applies Kendall et al. (CVPR 2018) unconstrained formulation:
  $$\mathcal{L} = \sum_{t} \left( e^{-s_t} \mathcal{L}_t + s_t \right)$$
  Without a conserved gradient budget ($\sum w_t$ unconstrained), if $s_t \to -\infty$, $e^{-s_t}$ causes gradient explosion and floating-point overflow under mixed precision (BF16/FP16), directly explaining the historical crash (`non-finite loss` at epoch 81) documented in the repository.

### 1.2 Lessons from the `feature/loss-function` B-Series Ablation Study
On the `feature/loss-function` branch, the research team conducted a controlled study (B-series) replacing UWAG with isolated single-objective geometric losses:
- **B0 (Legacy UWAG anchor):** Moderate mAP = **78.20%** (Car: 92.53%, Ped: 64.08%, Cyc: 77.98%).
- **B2 (Full KFIoU - ICLR 2023):** Moderate mAP = **65.49%** ($-12.71\%$).
- **B4 (KLD - NeurIPS 2021):** Moderate mAP = **66.05%** ($-12.15\%$).
- **B3 (ProbIoU - ICCV 2021):** Moderate mAP = **63.37%** ($-14.83\%$).
- **B5 (MGIoU - AAAI 2026):** Moderate mAP = **56.19%** ($-22.01\%$).

#### Why did isolated geometric losses (B2–B5) drop by 12–22% compared to UWAG?
1. **The Decoupled vs. Coupled Supervision Dilemma:**
   In B2–B5, all coordinate-level parameter losses ($L_{\text{offset}}, L_{\text{size}}, L_{\text{yaw}}$) were completely removed, delegating 100% of localization to a single coupled geometry loss.
   - Predicting 7 degrees of freedom $(x, y, w, l, \theta)$ purely through a non-convex, non-linear geometric metric is an ill-conditioned inverse problem, especially during early and middle training stages.
   - Independent Smooth-L1 losses provide direct, convex, linear gradients for each coordinate independently, ensuring rapid convergence.
2. **Task Gradient Scale Mismatch:**
   B2–B5 enforced fixed task weights ($\lambda_{\text{cls}} = 1.0, \lambda_{\text{reg}} = 1.0$). Gaussian focal loss on a $704 \times 800$ heatmap and regression loss on sparse positive cells have vastly different gradient magnitudes, resulting in severe gradient conflict without dynamic task balancing.
3. **Gaussian Covariance Singularities:**
   Gaussian losses (KFIoU, KLD, ProbIoU) invert $2 \times 2$ covariance matrices. For extreme aspect ratios (pedestrians $0.8 \times 0.6\text{m}$, cyclists), covariance eigenvalues become ill-conditioned, necessitating manual jitter regularizers and forcing autocast to be disabled.

---

## 2. Theoretical Formulation of OGA-Loss

To resolve these challenges and establish a genuine, high-impact scientific contribution, we introduce **OGA-Loss** (**Oriented Geometric Alignment & Adaptive Loss**), structured upon three foundational pillars:

```
                                  OGA-Loss Architecture
                                  
                       ┌──────────────────────────────────────────────┐
                       │           Detector Head Output Maps          │
                       │   cls [B,3,H,W]     offset [B,2,H,W]         │
                       │   size [B,2,H,W]    yaw [B,2,H,W]            │
                       └──────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │                                               │
                      ▼                                               ▼
         [Stream 1: Decoupled Params]                    [Stream 2: Rotated Geometry]
         - Modified Focal Loss (cls)                     - Pi-Symmetric Normalized
         - Smooth-L1 (offset)                              Corner Distance (L_NCD)
         - Smooth-L1 (size)                              - Multi-Axis Projection
         - Smooth-L1 (yaw)                                 GIoU (L_proj)
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │   Pillar 3: Temperature-Softmax Bounded   │
                        │      Uncertainty Weighting (T-SBUW)       │
                        │    w_t = M * exp(s_t / tau) / sum(exp)    │
                        └─────────────────────┬─────────────────────┘
                                              │
                                              ▼
                                   L_total = sum(w_t * L_t)
```

### Pillar 1: Rotated Geometric Alignment (RGA)

Instead of axis-aligned heuristics (UWAG) or ill-conditioned Gaussian matrix inversions (KFIoU/KLD), OGA-Loss couples two exact, singularity-free rotated geometric objectives:

#### 1.1 Scale-Normalized $\pi$-Symmetric Corner Distance ($L_{\text{NCD}}$)
An oriented 2D BEV box is specified by center $(x, y)$, length $l = \exp(\text{size}_1)$, width $w = \exp(\text{size}_0)$, and heading angle $\theta = 0.5 \text{atan2}(\sin 2\theta, \cos 2\theta)$.
Its 4 corners in metric BEV coordinates are:
$$\mathbf{p}_k = (x, y) + \mathbf{R}(\theta) \begin{pmatrix} \pm l/2 \\ \pm w/2 \end{pmatrix}, \quad k \in \{0, 1, 2, 3\}$$

Because the network outputs doubled-angle components $(\cos 2\theta, \sin 2\theta)$, bounding boxes possess inherent $\pi$-rotational symmetry (a box rotated by $180^\circ$ is geometrically identical). Thus, corner correspondence between prediction $\mathbf{P} = \{\mathbf{p}_k\}$ and target $\mathbf{G} = \{\mathbf{g}_k\}$ admits exactly two canonical cyclic permutations:
- Direct orientation alignment: $\pi_0 = (0, 1, 2, 3)$
- $\pi$-rotated alignment: $\pi_1 = (3, 2, 1, 0)$ (or $(2, 3, 0, 1)$ depending on corner index ordering)

The minimal $\pi$-symmetric corner distance is:
$$D_{\text{corner}}(\mathbf{P}, \mathbf{G}) = \min \left( \frac{1}{4} \sum_{k=0}^3 \|\mathbf{p}_k - \mathbf{g}_k\|_1, \; \frac{1}{4} \sum_{k=0}^3 \|\mathbf{p}_k - \mathbf{g}_{\pi_1(k)}\|_1 \right)$$

To maintain scale fairness between small targets (Pedestrian: $0.8\text{m} \times 0.6\text{m}$) and large targets (Car: $4.0\text{m} \times 1.6\text{m}$), $D_{\text{corner}}$ is normalized by the target diagonal length:
$$\text{diag}_{\text{tgt}} = \sqrt{w_{\text{tgt}}^2 + l_{\text{tgt}}^2}$$
$$\mathcal{L}_{\text{NCD}} = \frac{D_{\text{corner}}(\mathbf{P}, \mathbf{G})}{\text{diag}_{\text{tgt}}}$$

**Key Theoretical Advantages of $\mathcal{L}_{\text{NCD}}$:**
1. Strictly zero if and only if prediction matches ground truth identically.
2. Continuous, linear, non-vanishing gradients across all distances (even when disjoint by tens of meters).
3. Simultaneously couples center $(x, y)$, extents $(w, l)$, and heading $\theta$ in unified metric units.
4. Requires zero matrix inversions, zero Cholesky decompositions, and zero determinant evaluations.

#### 1.2 Multi-Axis Projection GIoU ($\mathcal{L}_{\text{proj}}$)
To provide bounded $[0, 1]$ overlap penalization:
The 4 corners of prediction and target are projected onto the 4 box normals (2 from prediction, 2 from target). On each projection axis $a \in \{1, 2, 3, 4\}$, we obtain 1D intervals $[p_{\min}, p_{\max}]$ and $[t_{\min}, t_{\max}]$.
The 1D Generalized IoU along axis $a$ is:
$$\text{GIoU}_a = \frac{\text{Inter}_a}{\text{Union}_a} - \frac{\text{Hull}_a - \text{Union}_a}{\text{Hull}_a}$$
$$\mathcal{L}_{\text{proj}} = \frac{1 - \frac{1}{4} \sum_{a=1}^4 \text{GIoU}_a}{2}$$

#### 1.3 Total Rotated Geometric Loss ($\mathcal{L}_{\text{geo}}$):
$$\mathcal{L}_{\text{geo}} = \mathcal{L}_{\text{proj}} + \beta \cdot \mathcal{L}_{\text{NCD}}$$
(with default hyperparameter $\beta = 1.0$).

---

### Pillar 2: Dual-Stream Supervision Architecture

To eliminate the regression collapse identified in the B-series ablations:
1. **Decoupled Coordinate Stream:**
   - $\mathcal{L}_{\text{cls}}$: Modified Focal Loss over the full Gaussian heatmap.
   - $\mathcal{L}_{\text{offset}}$: Smooth-$L_1$ on $(dx, dy)$ at positive regression locations.
   - $\mathcal{L}_{\text{size}}$: Smooth-$L_1$ on $(\log w, \log l)$ at positive regression locations.
   - $\mathcal{L}_{\text{yaw}}$: Smooth-$L_1$ on $(\cos 2\theta, \sin 2\theta)$ at positive regression locations.
2. **Holistic Geometric Alignment Stream:**
   - $\mathcal{L}_{\text{geo}}$: Jointly regularizes oriented overlap and corner alignment.

This ensures robust linear guidance from individual coordinates early in training, while fine-tuning rotated footprint overlap and orientation at convergence.

---

### Pillar 3: Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW)

To resolve the numerical instability and scale drift of Kendall's homoscedastic weighting:
Let $\mathbf{s} = (s_{\text{cls}}, s_{\text{offset}}, s_{\text{size}}, s_{\text{yaw}}, s_{\text{geo}}) \in \mathbb{R}^5$ denote learnable log-uncertainty logits initialized to $\mathbf{0}$.
Task weights are computed via temperature-scaled Softmax with bound constraints:
$$s_t^{\text{clamped}} = \text{clamp}(s_t, -c, c), \quad c = 3.0$$
$$w_t = M \cdot \frac{\exp(s_t^{\text{clamped}} / \tau)}{\sum_{j=1}^M \exp(s_j^{\text{clamped}} / \tau)}, \quad M = 5, \; \tau = 2.0$$

**Mathematical & Optimization Guarantees:**
1. **Gradient Scale Conservation:**
   $$\sum_{t=1}^M w_t = M$$
   The optimizer cannot artificially lower the loss by driving all $s_t \to \infty$. Total backpropagated gradient norm remains strictly bounded and stable.
2. **Anti-Task Starvation:**
   Because weights are strictly bounded below by $w_{\min} = M \frac{e^{-c/\tau}}{(M-1)e^{c/\tau} + e^{-c/\tau}} > 0$, no task is ever silenced or starved.
3. **Unconditional Numerical Safety under BF16/FP32:**
   Softmax with bounded logits is unconditionally finite, completely eliminating negative exponential overflow ($\exp(-s_t)$) issues.

---

## 3. Final Objective Function

$$\mathcal{L}_{\text{total}} = \sum_{t \in \{\text{cls}, \text{offset}, \text{size}, \text{yaw}, \text{geo}\}} w_t \cdot \mathcal{L}_t$$

where:
- $w_t$ is dynamically computed by `TemperatureSoftmaxUncertainty`.
- $\mathcal{L}_{\text{cls}}$ is computed via `modified_focal_loss`.
- $\mathcal{L}_{\text{offset}}, \mathcal{L}_{\text{size}}, \mathcal{L}_{\text{yaw}}$ are computed via `smooth_l1_loss`.
- $\mathcal{L}_{\text{geo}} = \mathcal{L}_{\text{proj}} + \beta \mathcal{L}_{\text{NCD}}$ is computed via `OrientedGeometryLoss`.

---

## 4. Scientific Contribution Comparison Matrix

| Property | Legacy UWAG | B-Series (B2–B5) | **Proposed OGA-Loss** |
|---|---|---|---|
| **Origin & IP** | Prior thesis code | Single-objective isolation | **Novel, 100% independent design** |
| **Rotated Geometry** | Axis-aligned box $\times$ cosine heuristic | Gaussian covariance or 1D Projection only | **$\pi$-Symmetric Corner Distance ($L_{\text{NCD}}$) + Multi-Axis Projection ($L_{\text{proj}}$)** |
| **Disjoint Box Gradients** | Strictly 0 (vanishing gradient) | KFIoU: asymptotic $1/r$; MGIoU: weak orientation | **Finite, linear, non-vanishing everywhere** |
| **Numerical Stability** | Prone to BF16 overflow (epoch 81 failure) | Requires FP32 cast, covariance regularizer | **Unconditionally finite, 100% BF16 compatible** |
| **Multi-Task Balancing** | Unconstrained Kendall $e^{-s}L + s$ | Fixed weights (caused 12–22% AP drop) | **Conserved Temperature-Softmax (T-SBUW, $\sum w_i = M$)** |
| **Decoupled Supervision** | Yes ($L_1$) | Completely eliminated | **Preserved via Dual-Stream architecture** |
| **Inference Footprint** | Baseline head contract | Baseline head contract | **Zero inference overhead (`cls, offset, size, yaw`)** |

---

## 5. Verification & Test Contracts

1. **$\pi$-Rotational Invariance:** Yaw angles $\theta$ and $\theta + \pi$ produce mathematically identical loss and gradients.
2. **Identical Bounding Boxes:** When prediction matches target, $L_{\text{NCD}} = 0$, $L_{\text{proj}} = 0 \implies L_{\text{geo}} = 0$.
3. **Disjoint Box Translation:** At 10m spatial separation, loss is finite and generates non-zero metric translation gradients.
4. **Conservation of Task Weights:** $\sum_{t=1}^5 w_t = 5.0$ holds strictly across all training steps.
5. **Autocast BF16/FP32 Safety:** Robust against extreme aspect ratios, square boxes, negative offsets, and empty positive masks (`reg_mask = 0`).
