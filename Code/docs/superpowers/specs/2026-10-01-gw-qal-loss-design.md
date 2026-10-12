# GW-QAL: Gaussian-Wasserstein & Quality-Aligned Loss Spec

**Date:** 2026-10-01  
**Author:** AI Pair Programmer & duyennh  
**Target Module:** `detector/core/losses/strategies/gw_qal.py`  
**Registry Name:** `"gw_qal"`  
**Status:** Ready for Review  

---

## 1. Problem Statement & Motivation

Existing rotated object detection losses in BEV LiDAR either:
1. **Rely on 1D Marginal Projections (MGIoU)**: Projecting 2D boxes onto 4 axes and averaging 1D GIoUs loses the joint 2D geometric coupling, creating gradient manifold degeneracies for mismatched aspect ratios.
2. **Suffer from Matrix Singularities (KFIoU, KLD, ProbIoU)**: Prior Gaussian losses invert $2 \times 2$ covariance matrices $\Sigma^{-1}$, which collapses on slender targets (pedestrians $0.8\text{m} \times 0.6\text{m}$), causing severe $-12\%$ to $-22\%$ mAP drops in repo history (B-series ablation).
3. **Decouple Task Supervisions**: Classification heatmaps and geometric localization are trained as separate objectives, causing high-scoring false-positive bounding boxes after NMS.

### The GW-QAL Solution: Flagship SOTA Contribution
We introduce **GW-QAL (Gaussian-Wasserstein & Quality-Aligned Loss)**, combining:
- **Analytic Inversion-Free 2D Gaussian Wasserstein Distance** with direct doubled-yaw covariance construction.
- **Dual-Stream Convex-Anchor Stabilization** to guarantee rapid initial convergence.
- **Range & Density-Adaptive Reweighting (RDA)** to resolve distant LiDAR point sparsity ($30-50\text{m}$).
- **Wasserstein-Harmonized Quality Alignment** to eliminate post-NMS false positives.
- **Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW)** with conserved gradient budget.

---

## 2. Theoretical Architecture of GW-QAL

```
                            GW-QAL Loss Architecture
                                       │
     ┌─────────────────────────────────┴─────────────────────────────────┐
     ▼                                                                   ▼
[Stream 1: Convex Guide]                                    [Stream 2: 2D Gaussian Wasserstein]
Smooth-L1 (offset, size, yaw)                               Analytic Inversion-Free W_2^2
(Guarantees convex linear pull)                             (Full 2D Rotated Manifold Alignment)
     │                                                                   │
     └─────────────────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
                       [Range & Density Adaptive Reweighting (RDA)]
                         omega_i = 1 + gamma * (r_i / r_max)^alpha
                                       │
                                       ▼
                       [Wasserstein Quality-Harmonized Focal Coupling]
                         S_i = exp(-sqrt(W_2^2) / tau_sim)
                         t_i = y_i * (S_i)^beta_q
                                       │
                                       ▼
                       [T-SBUW Conserved Dynamic Balancer]
                         w_t = M * softmax(s_t / tau)
```

---

## 3. Mathematical Formulation

### 3.1. Covariance Representation Directly from Head Outputs
A rotated BEV box with center $\mu = (x, y)$, dimensions $(w, l) = (\exp(\text{size}_0), \exp(\text{size}_1))$, and doubled-yaw $(\cos 2\theta, \sin 2\theta)$ has uniform rectangle covariance (divisor $d = 12.0$):
$$\text{var}_l = \frac{l^2}{12.0}, \quad \text{var}_w = \frac{w^2}{12.0}$$
$$m = \frac{\text{var}_l + \text{var}_w}{2}, \quad \delta = \frac{\text{var}_l - \text{var}_w}{2}$$
$$\Sigma = \begin{bmatrix} m + \delta \cos 2\theta & \delta \sin 2\theta \\ \delta \sin 2\theta & m - \delta \cos 2\theta \end{bmatrix}$$

*Key Identity:*
- $\text{Tr}(\Sigma) = \text{var}_l + \text{var}_w$ (trace is invariant to orientation).
- $\det(\Sigma) = m^2 - \delta^2 = \text{var}_l \cdot \text{var}_w = \frac{l^2 w^2}{144}$ (determinant is strictly positive and independent of $\theta$).

### 3.2. Analytic Inversion-Free 2D Gaussian Wasserstein Distance
For two 2D Gaussian distributions $\mathcal{N}_p(\mu_p, \Sigma_p)$ and $\mathcal{N}_t(\mu_t, \Sigma_t)$, the 2-Wasserstein distance is:
$$W_2^2 = \|\mu_p - \mu_t\|_2^2 + \text{Tr}\left(\Sigma_p + \Sigma_t - 2\left(\Sigma_p^{1/2} \Sigma_t \Sigma_p^{1/2}\right)^{1/2}\right)$$

For $2 \times 2$ SPD matrices, the matrix square root trace admits the exact closed-form scalar identity:
$$\text{Tr}\left(\left(\Sigma_p^{1/2} \Sigma_t \Sigma_p^{1/2}\right)^{1/2}\right) = \sqrt{\text{Tr}(\Sigma_p \Sigma_t) + 2\sqrt{\det(\Sigma_p)\det(\Sigma_t)}}$$

where $\text{Tr}(\Sigma_p \Sigma_t) = \Sigma_{p,00}\Sigma_{t,00} + 2\Sigma_{p,01}\Sigma_{t,01} + \Sigma_{p,11}\Sigma_{t,11}$.

Thus, the exact 2D Gaussian Wasserstein metric is:
$$W_2^2 = \|\mu_p - \mu_t\|_2^2 + \text{Tr}(\Sigma_p) + \text{Tr}(\Sigma_t) - 2\sqrt{\text{Tr}(\Sigma_p \Sigma_t) + 2\sqrt{\det(\Sigma_p)\det(\Sigma_t)}}$$

> **Critical Theoretical Breakthrough:**
> - **Zero Matrix Inversion**: $\Sigma^{-1}$ is never computed.
> - **Zero Eigenvalue / Cholesky Decomposition**: Evaluated purely with elementary scalar additions, multiplications, and square roots.
> - **$C^\infty$-Smoothness**: Infinitely differentiable across all angles without hard-min switching cusps.

### 3.3. Scale-Normalized Bounded Wasserstein Alignment Loss ($\mathcal{L}_{\text{GWA}}$)
To equalize sensitivity across object classes and guarantee bounded loss $\in [0, 1)$:
$$\mathcal{L}_{\text{GWA}} = 1.0 - \exp\left( -\frac{\sqrt{W_2^2 + \epsilon}}{\tau_{\text{gwa}} \cdot \text{diag}_{\text{tgt}}} \right)$$
where $\text{diag}_{\text{tgt}} = \sqrt{w_{\text{tgt}}^2 + l_{\text{tgt}}^2}$ and default scale parameter $\tau_{\text{gwa}} = 2.0$.

### 3.4. Range & Density-Adaptive Spatial Reweighting (RDA)
For positive anchor $i$ at radial distance $r_i = \sqrt{x_i^2 + y_i^2}$:
$$\omega_i^{\text{spatial}} = 1.0 + \gamma_{\text{rda}} \cdot \left(\frac{r_i}{r_{\max}}\right)^{\alpha_{\text{rda}}}$$
Regression loss components are multiplied by $\omega_i^{\text{spatial}}$ to dynamically boost gradient scale for sparse distant objects.

### 3.5. Wasserstein Quality-Harmonized Classification Coupling
Compute geometric similarity $S_i \in [0, 1]$:
$$S_i = \exp\left(-\frac{\sqrt{W_2^2 + \epsilon}}{\tau_{\text{sim}} \cdot \text{diag}_{\text{tgt}}}\right)$$
Dynamic continuous soft target for positive anchors:
$$t_i = y_i \cdot (S_i)^{\beta_{\text{q}}}$$
Anchor cells with high localization error ($S_i \to 0$) receive attenuated classification targets, penalizing high-confidence misaligned predictions.

---

## 4. Strategy Implementation & Backward Compatibility

### 4.1. File Location & Strategy Registration
- Create new file: `detector/core/losses/strategies/gw_qal.py`.
- Register in registry via:
  ```python
  from core.losses.strategies.registry import register_loss_strategy
  from core.losses.strategies.base import BaseLossStrategy

  @register_loss_strategy("gw_qal")
  class GwQalLossStrategy(BaseLossStrategy):
      ...
  ```
- Expose via `detector/core/losses/strategies/__init__.py`.

### 4.2. 100% Backward Compatibility
1. Does not modify or overwrite `oga.py`, `uwag.py`, or `baseline.py`.
2. Existing training scripts, configurations, and evaluation benchmarks remain 100% functional.
3. Checkpoints trained with `gw_qal` follow standard dictionary layout and load seamlessly into `LossFunction`.

---

## 5. Configuration Schema

```json
{
  "loss": {
    "name": "gw_qal",
    "cls_encoding": "gaussian",
    "temperature": 2.0,
    "clamp_bound": 3.0,
    "tau_gwa": 2.0,
    "tau_sim": 2.0,
    "beta_q": 1.0,
    "rda_gamma": 1.5,
    "rda_alpha": 2.0,
    "divisor": 12.0,
    "use_iou": false,
    "ema_momentum": 0.99
  }
}
```

---

## 6. Verification & Test Plan

1. **Unit Tests (`tests/test_loss_strategy_gw_qal.py`):**
   - Verify closed-form $W_2^2$ matches numerical definition for random positive definite matrices.
   - Verify zero loss ($W_2^2 = 0$) when prediction identically matches target.
   - Verify finite, non-zero, monotonic gradients on large offsets ($10\text{m}$ separation).
   - Verify smooth gradient transition across full $360^\circ$ rotation without gradient spikes.
   - Verify BF16 autocast compatibility.
   - Empty mask test returning connected zero.
2. **Integration Verification:**
   - Multi-task uncertainty weighting step with AdamW optimizer on synthetic batch.
