# Q-OGA: Quality-Aligned & Range-Adaptive Oriented Geometric Loss Spec

**Date:** 2026-10-01  
**Author:** AI Pair Programmer & duyennh  
**Target Module:** `detector/core/losses/strategies/q_oga.py`  
**Registry Name:** `"q_oga"`  
**Status:** Ready for Review  

---

## 1. Problem Statement & Motivation

The anchor baseline loss **OGA (Oriented Geometric Alignment & Adaptive Loss)** achieved a major milestone by boosting KITTI BEV Moderate mAP from $78.09\%$ to $86.67\%$. However, fine-grained diagnostic analysis reveals three distinct architectural and mathematical bottlenecks:

1. **The Discontinuous Corner Cusp:**
   OGA computes $\pi$-symmetric corner distance via a hard minimum:
   $$D_{\text{raw}} = \min(D_{\text{direct}}, D_{\pi})$$
   At orientation symmetry thresholds ($\pm 45^\circ$ and $\pm 90^\circ$ yaw differences), $\min(\cdot)$ creates a non-differentiable cusp. The gradient direction jumps discontinuously, causing high-frequency gradient oscillations that inhibit sub-degree bounding box refinement in late training epochs.

2. **LiDAR Range & Point Sparsity Blindness:**
   In OGA, positive anchors contribute with uniform weight $\frac{1}{N_{\text{pos}}}$. Vehicles at close range ($<20\text{m}$) have thousands of LiDAR reflections and occupy dozens of BEV cells, dominating batch gradients. Distant targets ($30-50\text{m}$, e.g. pedestrians and cyclists) have only $1-5$ points and occupy few cells, suffering severe gradient starvation.
   - *Diagnostic Proof:* In OGA evaluation logs, Car Moderate reaches $96.66\%$, while Pedestrian at $30-50\text{m}$ drops to $66.79\%$.

3. **Classification & Geometric Quality Disalignment:**
   Anchor cells with poor geometric regression can still receive maximum classification loss targets ($y=1.0$), while anchors with near-perfect rotated boxes may be suppressed during NMS due to slightly lower classification logits. This misalignment produces over $4,400$ false positives for Car and $2,000$ for Pedestrian in OGA.

---

## 2. Theoretical Architecture of Q-OGA

Q-OGA preserves the proven Dual-Stream + T-SBUW architecture of OGA while upgrading its internal components into three synergistic pillars:

```
                            Q-OGA Loss Architecture
                                       │
     ┌─────────────────────────────────┴─────────────────────────────────┐
     ▼                                                                   ▼
[Stream 1: Convex Guide]                                    [Stream 2: Rotated Geometry]
Smooth-L1 (offset, size, yaw)                               • Multi-Axis Projection GIoU (L_proj)
                                                            • C^inf Soft-Min Corner Distance (L_SNCD)
     │                                                                   │
     └─────────────────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
                       [Range & Density Adaptive Reweighting (RDA)]
                         omega_i = 1 + gamma * (r_i / r_max)^alpha
                                       │
                                       ▼
                       [Quality-Coupled Focal Alignment (QCFA)]
                         t_i = y_i * (GIoU_target_i)^beta
                                       │
                                       ▼
                       [T-SBUW Conserved Dynamic Balancer]
                         w_t = M * softmax(s_t / tau)
```

---

## 3. Mathematical Formulation

### 3.1. Pillar 1: $C^\infty$-Smooth Soft-Min $\pi$-Symmetric Corner Distance ($\mathcal{L}_{\text{SNCD}}$)
For predicted corners $\mathbf{P} = \{\mathbf{p}_k\}_{k=0}^3$ and target corners $\mathbf{G} = \{\mathbf{g}_k\}_{k=0}^3$:
- Direct alignment: $D_{\text{direct}} = \frac{1}{4} \sum_{k=0}^3 \|\mathbf{p}_k - \mathbf{g}_k\|_1$
- $\pi$-Rotated alignment: $D_{\pi} = \frac{1}{4} \sum_{k=0}^3 \|\mathbf{p}_k - \mathbf{g}_{\pi(k)}\|_1$ where $\pi = (3, 2, 1, 0)$

Instead of $\min(D_{\text{direct}}, D_{\pi})$, Q-OGA computes the Log-Sum-Exp smooth relaxation:
$$D_{\text{smooth}}(\mathbf{P}, \mathbf{G}; \tau_{\text{smooth}}) = -\tau_{\text{smooth}} \cdot \ln \left( \exp\left(-\frac{D_{\text{direct}}}{\tau_{\text{smooth}}}\right) + \exp\left(-\frac{D_{\pi}}{\tau_{\text{smooth}}}\right) \right)$$
where $\tau_{\text{smooth}} = 0.05$ (default).

Scale normalization by target diagonal length $\text{diag}_{\text{tgt}} = \sqrt{w_{\text{tgt}}^2 + l_{\text{tgt}}^2}$:
$$\mathcal{L}_{\text{SNCD}} = \frac{D_{\text{smooth}}(\mathbf{P}, \mathbf{G}; \tau_{\text{smooth}})}{\text{diag}_{\text{tgt}}}$$

*Properties:*
1. $\lim_{\tau_{\text{smooth}} \to 0} D_{\text{smooth}} = \min(D_{\text{direct}}, D_{\pi})$.
2. Strictly $C^\infty$-differentiable across all yaw angles, completely eliminating gradient oscillation at $\pm 45^\circ$ and $\pm 90^\circ$.

### 3.2. Pillar 2: Range & Density-Adaptive Spatial Reweighting (RDA)
For each positive anchor cell $i$ located at BEV spatial coordinate $(x_i, y_i)$:
$$r_i = \sqrt{x_i^2 + y_i^2}$$
$$\omega_i^{\text{spatial}} = 1.0 + \gamma_{\text{rda}} \cdot \left( \frac{r_i}{r_{\max}} \right)^{\alpha_{\text{rda}}}$$
where $r_{\max} = 70.4\text{m}$, $\gamma_{\text{rda}} = 1.5$, $\alpha_{\text{rda}} = 2.0$.

Positive anchors in distant regions receive amplified regression gradients, mitigating point sparsity and balancing representation against nearby dense clusters.

### 3.3. Pillar 3: Quality-Coupled Focal Alignment (QCFA)
Using detached MGIoU target $S_i \in [0, 1]$ computed from current predicted box and target:
$$t_i = y_i \cdot (S_i)^{\beta_{\text{qcfa}}}$$
where $\beta_{\text{qcfa}} = 1.0$.

The classification loss replaces rigid $0/1$ Gaussian targets with continuous quality-calibrated targets $t_i$:
- For positive anchors: $\mathcal{L}_{\text{cls}, i} = -t_i \cdot \ln(p_i) \cdot |t_i - p_i|^\gamma$
- For negative background: standard focal suppression $\mathcal{L}_{\text{cls}, j} = -(1 - t_j)^\alpha \cdot p_j^\gamma \cdot \ln(1 - p_j)$

---

## 4. Strategy Implementation & Backward Compatibility

### 4.1. File Location & Strategy Registration
- Create new file: `detector/core/losses/strategies/q_oga.py`.
- Register in registry via:
  ```python
  from core.losses.strategies.registry import register_loss_strategy
  from core.losses.strategies.base import BaseLossStrategy

  @register_loss_strategy("q_oga")
  class QOgaLossStrategy(BaseLossStrategy):
      ...
  ```
- Register inside `detector/core/losses/strategies/__init__.py`.

### 4.2. Zero Breaking Changes
1. Existing configs with `"name": "oga"`, `"name": "uwag"`, `"name": "baseline"` execute existing strategies untouched.
2. `LossFunction` façade in `detector/core/losses/loss_fn.py` seamlessly handles `q_oga` through the existing factory dispatch.
3. Telemetry dictionary returns all standard keys (`loss`, `cls`, `offset`, `size`, `yaw`, `geo`, `corner_dist`, `proj_giou`, `weights_*`) plus optional telemetry (`smooth_corner_dist`, `mean_rda_weight`).

---

## 5. Configuration Schema

```json
{
  "loss": {
    "name": "q_oga",
    "cls_encoding": "gaussian",
    "temperature": 2.0,
    "clamp_bound": 3.0,
    "corner_beta": 1.0,
    "soft_min_tau": 0.05,
    "rda_gamma": 1.5,
    "rda_alpha": 2.0,
    "qcfa_beta": 1.0,
    "use_iou": false,
    "ema_momentum": 0.99
  }
}
```

---

## 6. Verification & Test Plan

1. **Unit Tests (`tests/test_loss_strategy_q_oga.py`):**
   - Forward pass produces finite scalar loss on synthetic batch.
   - Backward pass yields non-zero finite gradients for all prediction heads (`cls`, `offset`, `size`, `yaw`).
   - Cusp continuity check: compare gradients at $44.99^\circ$ vs $45.01^\circ$ to verify absence of gradient shock.
   - Empty positive mask test returns graph-connected zero with finite zero gradients.
   - Autocast BF16 compatibility test.
2. **Integration Verification:**
   - Run pipeline config check with `train.py --config configs/kitti/ablation/q_oga.json` for 1 synthetic step.
