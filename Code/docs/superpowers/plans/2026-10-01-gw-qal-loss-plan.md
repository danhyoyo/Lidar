# GW-QAL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify the flagship GW-QAL (Gaussian-Wasserstein & Quality-Aligned Loss) strategy with closed-form inversion-free 2D Gaussian Wasserstein alignment, dual-stream convex stabilization, LiDAR range-density adaptation, and Wasserstein-coupled classification.

**Architecture:** GW-QAL models rotated BEV rectangles directly as 2D Gaussians using doubled-angle heading vectors $[\cos 2\theta, \sin 2\theta]$. It computes the exact 2-Wasserstein metric $W_2^2$ via an analytic closed-form trace identity that completely eliminates matrix inversion and Cholesky decompositions. Dual-stream Smooth-L1 provides convex guidance, RDA compensates for distant point sparsity, and Wasserstein similarity softly guides the focal classification loss.

**Tech Stack:** Python 3.10+, PyTorch 2.11+, NumPy, PyTest.

**Spec:** `docs/superpowers/specs/2026-10-01-gw-qal-loss-design.md`

## Global Constraints

- 100% backward compatible: `oga.py`, `uwag.py`, and `baseline.py` must remain completely unmodified.
- Zero matrix inversions ($\Sigma^{-1}$ is never computed), ensuring 100% immunity to eigenvalue collapse on slender targets (pedestrians).
- Strategy registered dynamically via `@register_loss_strategy("gw_qal")`.
- Compatible with FP32 and BF16 mixed-precision training (`torch.cuda.amp.autocast`).
- Zero external CUDA/C++ extension dependencies; pure PyTorch tensor operations only.
- Empty positive mask (`reg_mask.sum() == 0`) must return a valid graph-connected scalar zero with finite zero gradients for all prediction heads.

---

### Task 1: Analytic Inversion-Free 2D Gaussian Wasserstein Core Module

**Files:**
- Create: `detector/core/losses/gaussian_wasserstein.py`
- Test: `tests/test_gaussian_wasserstein.py`

**Interfaces:**
- `box_covariance_2d(log_size, doubled_yaw, divisor=12.0, epsilon=1e-6, max_abs_log_size=10.0) -> torch.Tensor`
- `closed_form_2d_gaussian_wasserstein(pred_offset, pred_cov, target_offset, target_cov, epsilon=1e-6) -> torch.Tensor`
- `bounded_gwa_loss(w2_sq, target_log_size, tau_gwa=2.0, epsilon=1e-6, max_abs_log_size=10.0) -> torch.Tensor`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gaussian_wasserstein.py
import math
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.gaussian_wasserstein import (
    box_covariance_2d,
    closed_form_2d_gaussian_wasserstein,
    bounded_gwa_loss,
)


class TestGaussianWasserstein(unittest.TestCase):
    def test_identical_distributions_have_zero_distance(self):
        offset = torch.tensor([[10.0, 5.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32)
        yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        cov, _ = box_covariance_2d(log_size, yaw)
        w2_sq = closed_form_2d_gaussian_wasserstein(offset, cov, offset, cov)
        self.assertTrue(torch.isfinite(w2_sq).all())
        self.assertAlmostEqual(float(w2_sq.item()), 0.0, places=5)

    def test_translation_only_matches_euclidean_distance(self):
        offset_p = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        offset_t = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(2.0), math.log(2.0)]], dtype=torch.float32)
        yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        cov_p, _ = box_covariance_2d(log_size, yaw)
        cov_t, _ = box_covariance_2d(log_size, yaw)
        w2_sq = closed_form_2d_gaussian_wasserstein(offset_p, cov_p, offset_t, cov_t)
        # Expected: 3^2 + 4^2 = 25.0
        self.assertAlmostEqual(float(w2_sq.item()), 25.0, places=4)

    def test_rotation_invariance_pi(self):
        offset = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(1.0), math.log(3.0)]], dtype=torch.float32)
        theta = 0.42
        yaw_a = torch.tensor([[math.cos(2.0 * theta), math.sin(2.0 * theta)]], dtype=torch.float32)
        yaw_b = torch.tensor([[math.cos(2.0 * (theta + math.pi)), math.sin(2.0 * (theta + math.pi))]], dtype=torch.float32)

        cov_a, _ = box_covariance_2d(log_size, yaw_a)
        cov_b, _ = box_covariance_2d(log_size, yaw_b)
        self.assertTrue(torch.allclose(cov_a, cov_b, atol=1e-5))

    def test_finite_gradients_under_extreme_aspect_ratios(self):
        offset = torch.tensor([[0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        log_size_p = torch.tensor([[math.log(0.1), math.log(10.0)]], dtype=torch.float32, requires_grad=True)
        yaw_p = torch.tensor([[0.5, 0.866]], dtype=torch.float32, requires_grad=True)

        target_offset = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_size = torch.tensor([[math.log(0.1), math.log(10.0)]], dtype=torch.float32)
        target_yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        cov_p, _ = box_covariance_2d(log_size_p, yaw_p)
        cov_t, _ = box_covariance_2d(target_size, target_yaw)
        w2_sq = closed_form_2d_gaussian_wasserstein(offset, cov_p, target_offset, cov_t)
        loss = bounded_gwa_loss(w2_sq, target_size)

        loss.backward()
        self.assertTrue(torch.isfinite(offset.grad).all())
        self.assertTrue(torch.isfinite(log_size_p.grad).all())
        self.assertTrue(torch.isfinite(yaw_p.grad).all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gaussian_wasserstein.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'core.losses.gaussian_wasserstein'`

- [ ] **Step 3: Write minimal implementation**

```python
# detector/core/losses/gaussian_wasserstein.py
"""Analytic Inversion-Free 2D Gaussian Wasserstein Metric for Rotated BEV Boxes."""

import torch
import torch.nn.functional as F

__all__ = [
    "box_covariance_2d",
    "closed_form_2d_gaussian_wasserstein",
    "bounded_gwa_loss",
]


def box_covariance_2d(
    log_size: torch.Tensor,
    doubled_yaw: torch.Tensor,
    divisor: float = 12.0,
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
):
    """Construct 2x2 spatial covariance matrices directly from doubled-angle yaw.

    Args:
        log_size: [N, 2] (log width, log length)
        doubled_yaw: [N, 2] (cos 2 theta, sin 2 theta)
    Returns:
        covariance: [N, 2, 2] symmetric positive-definite covariance matrices.
        clamp_count: int scalar clamped log-size count.
    """
    log_size = log_size.float()
    doubled_yaw = doubled_yaw.float()
    clamped = log_size.clamp(-max_abs_log_size, max_abs_log_size)
    width, length = torch.exp(clamped).unbind(dim=-1)

    cos2, sin2 = F.normalize(doubled_yaw, dim=-1, eps=epsilon).unbind(dim=-1)

    parallel = length.square() / divisor
    perpendicular = width.square() / divisor
    mean = 0.5 * (parallel + perpendicular)
    delta = 0.5 * (parallel - perpendicular)

    covariance = torch.stack(
        (
            torch.stack((mean + delta * cos2, delta * sin2), dim=-1),
            torch.stack((delta * sin2, mean - delta * cos2), dim=-1),
        ),
        dim=-2,
    )
    scale = torch.maximum(parallel, perpendicular).clamp_min(1.0)
    eye = torch.eye(2, dtype=torch.float32, device=covariance.device)
    return covariance + (epsilon * scale)[..., None, None] * eye, (clamped != log_size).sum()


def closed_form_2d_gaussian_wasserstein(
    pred_offset: torch.Tensor,
    pred_cov: torch.Tensor,
    target_offset: torch.Tensor,
    target_cov: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute exact 2-Wasserstein distance between 2D Gaussians with zero matrix inversions.

    Formula:
        W_2^2 = ||mu_p - mu_t||^2 + Tr(Sigma_p) + Tr(Sigma_t) - 2 * sqrt(Tr(Sigma_p * Sigma_t) + 2 * sqrt(det(Sigma_p) * det(Sigma_t)))
    """
    # 1. Center separation squared: ||mu_p - mu_t||^2
    diff_center = pred_offset - target_offset
    center_dist_sq = diff_center.square().sum(dim=-1)  # [N]

    # 2. Traces: Tr(Sigma) = Sigma_00 + Sigma_11
    tr_p = pred_cov[..., 0, 0] + pred_cov[..., 1, 1]
    tr_t = target_cov[..., 0, 0] + target_cov[..., 1, 1]

    # 3. Determinants: det(Sigma) = Sigma_00 * Sigma_11 - Sigma_01 * Sigma_10
    det_p = (pred_cov[..., 0, 0] * pred_cov[..., 1, 1] - pred_cov[..., 0, 1] * pred_cov[..., 1, 0]).clamp_min(epsilon)
    det_t = (target_cov[..., 0, 0] * target_cov[..., 1, 1] - target_cov[..., 0, 1] * target_cov[..., 1, 0]).clamp_min(epsilon)

    # 4. Product trace: Tr(Sigma_p * Sigma_t)
    tr_pt = (
        pred_cov[..., 0, 0] * target_cov[..., 0, 0]
        + 2.0 * pred_cov[..., 0, 1] * target_cov[..., 0, 1]
        + pred_cov[..., 1, 1] * target_cov[..., 1, 1]
    )

    # 5. Exact matrix square-root trace: sqrt(Tr(pt) + 2 * sqrt(det_p * det_t))
    inner = (tr_pt + 2.0 * torch.sqrt(det_p * det_t)).clamp_min(epsilon)
    trace_sqrt = 2.0 * torch.sqrt(inner)

    # 6. Combined 2-Wasserstein squared
    w2_sq = (center_dist_sq + tr_p + tr_t - trace_sqrt).clamp_min(0.0)
    return w2_sq


def bounded_gwa_loss(
    w2_sq: torch.Tensor,
    target_log_size: torch.Tensor,
    tau_gwa: float = 2.0,
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
) -> torch.Tensor:
    """Normalize by target diagonal and apply bounded exponential mapping."""
    clamped_target_size = target_log_size.float().clamp(-max_abs_log_size, max_abs_log_size)
    target_w = torch.exp(clamped_target_size[:, 0])
    target_l = torch.exp(clamped_target_size[:, 1])
    diagonal = torch.sqrt(target_w.square() + target_l.square()).clamp_min(epsilon)

    w_normalized = torch.sqrt(w2_sq + epsilon) / (tau_gwa * diagonal)
    return (1.0 - torch.exp(-w_normalized)).mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gaussian_wasserstein.py -v`  
Expected: PASS with 4 passed

- [ ] **Step 5: Commit**

```bash
git add detector/core/losses/gaussian_wasserstein.py tests/test_gaussian_wasserstein.py
git commit -m "feat(loss): implement closed-form 2D Gaussian Wasserstein metric module"
```

---

### Task 2: GW-QAL Loss Strategy Implementation

**Files:**
- Create: `detector/core/losses/strategies/gw_qal.py`
- Modify: `detector/core/losses/strategies/__init__.py`
- Test: `tests/test_loss_strategy_gw_qal.py`

**Interfaces:**
- `GwQalLossStrategy(BaseLossStrategy)` registered as `"gw_qal"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loss_strategy_gw_qal.py
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.strategies import build_loss_strategy, get_available_loss_strategies
from core.losses.loss_fn import LossFunction


class TestLossStrategyGwQal(unittest.TestCase):
    def setUp(self):
        self.B, self.H, self.W = 2, 8, 8
        self.pred = {
            "cls": torch.randn(self.B, 3, self.H, self.W, requires_grad=True),
            "offset": torch.randn(self.B, 2, self.H, self.W, requires_grad=True),
            "size": torch.randn(self.B, 2, self.H, self.W, requires_grad=True),
            "yaw": torch.randn(self.B, 2, self.H, self.W, requires_grad=True),
        }
        self.target = {
            "cls": torch.sigmoid(torch.randn(self.B, 3, self.H, self.W)),
            "offset": torch.zeros(self.B, 2, self.H, self.W),
            "size": torch.zeros(self.B, 2, self.H, self.W),
            "yaw": torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(self.B, 2, self.H, self.W),
            "reg_mask": torch.ones(self.B, self.H, self.W),
        }

    def test_gw_qal_registered(self):
        self.assertIn("gw_qal", get_available_loss_strategies())

    def test_gw_qal_forward_backward(self):
        criterion = LossFunction("gaussian", {
            "name": "gw_qal",
            "temperature": 2.0,
            "clamp_bound": 3.0,
            "tau_gwa": 2.0,
            "tau_sim": 2.0,
            "beta_q": 1.0,
            "rda_gamma": 1.5,
        })
        loss_dict = criterion(self.pred, self.target)
        for key in ("loss", "cls", "offset", "size", "yaw", "geo", "w2_mean_dist", "rda_mean_weight"):
            self.assertIn(key, loss_dict)
            self.assertTrue(torch.isfinite(loss_dict[key]))

        loss_dict["loss"].backward()
        for head in ("cls", "offset", "size", "yaw"):
            self.assertTrue(self.pred[head].grad is not None)
            self.assertTrue(torch.isfinite(self.pred[head].grad).all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loss_strategy_gw_qal.py -v`  
Expected: FAIL with `AssertionError: 'gw_qal' not found in ('baseline', 'uwag', 'oga', 'q_oga')`

- [ ] **Step 3: Write minimal implementation**

```python
# detector/core/losses/strategies/gw_qal.py
"""GW-QAL (Gaussian-Wasserstein & Quality-Aligned Loss) strategy."""

from typing import Any, Dict
import torch
import torch.nn.functional as F

from core.losses.gaussian_wasserstein import (
    box_covariance_2d,
    closed_form_2d_gaussian_wasserstein,
    bounded_gwa_loss,
)
from core.losses.smooth_corner_loss import compute_range_weights
from core.losses.quality_focal_loss import quality_focal_loss
from core.losses.uncertainty_weighting import TemperatureSoftmaxUncertainty
from core.losses.strategies.base import BaseLossStrategy
from core.losses.strategies.registry import register_loss_strategy


@register_loss_strategy("gw_qal")
class GwQalLossStrategy(BaseLossStrategy):
    """Flagship Gaussian-Wasserstein Alignment Loss with Dual-Stream & RDA."""

    TASKS = ("cls", "offset", "size", "yaw", "geo")

    def __init__(self, cls_encoding: str, config: Dict[str, Any] = None):
        super().__init__(cls_encoding, config)
        config = self.config

        self.eps = float(config.get("epsilon", 1e-6))
        self.max_abs_log_size = float(config.get("max_abs_log_size", 10.0))
        self.divisor = float(config.get("divisor", 12.0))
        self.tau_gwa = float(config.get("tau_gwa", 2.0))
        self.tau_sim = float(config.get("tau_sim", 2.0))
        self.beta_q = float(config.get("beta_q", 1.0))
        self.rda_gamma = float(config.get("rda_gamma", 1.5))
        self.rda_alpha = float(config.get("rda_alpha", 2.0))
        self.temperature = float(config.get("temperature", 2.0))
        self.clamp_bound = float(config.get("clamp_bound", 3.0))
        self.ema_momentum = float(config.get("ema_momentum", 0.99))

        self.weighting = TemperatureSoftmaxUncertainty(
            task_names=self.TASKS,
            temperature=self.temperature,
            clamp_bound=self.clamp_bound,
            ema_momentum=self.ema_momentum,
        )

    def forward(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        self._validate_inputs(pred, target)

        positive = target["reg_mask"].reshape(-1).bool()
        device = pred["offset"].device
        B, _, H, W = pred["offset"].shape

        if not positive.any():
            zero = (pred["offset"].sum() + pred["size"].sum() + pred["yaw"].sum()) * 0.0
            cls_loss = self._compute_cls_loss(pred["cls"], target["cls"])
            task_losses = {
                "cls": cls_loss,
                "offset": zero,
                "size": zero,
                "yaw": zero,
                "geo": zero,
            }
            loss, weights = self.weighting(task_losses)
            return {
                "loss": loss,
                "cls": cls_loss.detach(),
                "offset": zero.detach(),
                "size": zero.detach(),
                "yaw": zero.detach(),
                "geo": zero.detach(),
                "w2_mean_dist": torch.zeros((), device=device),
                "rda_mean_weight": torch.ones((), device=device),
                **{f"weight_{k}": w for k, w in weights.items()},
            }

        def _pos(x):
            return x.permute(0, 2, 3, 1).reshape(-1, 2)[positive].float()

        tgt_off_pos = _pos(target["offset"])
        tgt_sz_pos = _pos(target["size"])
        tgt_yaw_pos = _pos(target["yaw"])

        pred_off_pos = _pos(pred["offset"])
        pred_sz_pos = _pos(pred["size"])
        pred_yaw_pos = _pos(pred["yaw"])

        # 1. Closed-Form 2D Gaussian Wasserstein Computation
        with torch.autocast(device_type=device.type, enabled=False):
            pred_cov, _ = box_covariance_2d(pred_sz_pos, pred_yaw_pos, self.divisor, self.eps, self.max_abs_log_size)
            tgt_cov, _ = box_covariance_2d(tgt_sz_pos, tgt_yaw_pos, self.divisor, self.eps, self.max_abs_log_size)
            w2_sq = closed_form_2d_gaussian_wasserstein(pred_off_pos, pred_cov, tgt_off_pos, tgt_cov, self.eps)
            geo_loss = bounded_gwa_loss(w2_sq, tgt_sz_pos, self.tau_gwa, self.eps, self.max_abs_log_size)

        # 2. Dynamic Wasserstein Quality Alignment for Classification
        with torch.no_grad():
            clamped_tgt_sz = tgt_sz_pos.clamp(-self.max_abs_log_size, self.max_abs_log_size)
            diag = torch.sqrt(torch.exp(clamped_tgt_sz[:, 0]).square() + torch.exp(clamped_tgt_sz[:, 1]).square()).clamp_min(self.eps)
            wasserstein_similarity = torch.exp(-torch.sqrt(w2_sq.detach() + self.eps) / (self.tau_sim * diag)).clamp(0.0, 1.0)
            quality_map = torch.zeros((B, H, W), dtype=torch.float32, device=device)
            quality_map[target["reg_mask"].bool()] = wasserstein_similarity

        cls_loss = quality_focal_loss(pred["cls"], target["cls"], quality_map, beta=self.beta_q)

        # 3. RDA Spatial Reweighting on Coordinate Streams
        rda_weights = compute_range_weights(tgt_off_pos, r_max=70.4, gamma=self.rda_gamma, alpha=self.rda_alpha)

        diff_off = F.smooth_l1_loss(pred_off_pos, tgt_off_pos, reduction="none").sum(dim=-1)
        offset_loss = (diff_off * rda_weights).mean()

        diff_sz = F.smooth_l1_loss(pred_sz_pos, tgt_sz_pos, reduction="none").sum(dim=-1)
        size_loss = (diff_sz * rda_weights).mean()

        diff_yaw = F.smooth_l1_loss(pred_yaw_pos, tgt_yaw_pos, reduction="none").sum(dim=-1)
        yaw_loss = (diff_yaw * rda_weights).mean()

        task_losses = {
            "cls": cls_loss,
            "offset": offset_loss,
            "size": size_loss,
            "yaw": yaw_loss,
            "geo": geo_loss,
        }
        loss, weights = self.weighting(task_losses)

        loss_dict = {
            "loss": loss,
            "cls": cls_loss.detach(),
            "offset": offset_loss.detach(),
            "size": size_loss.detach(),
            "yaw": yaw_loss.detach(),
            "geo": geo_loss.detach(),
            "w2_mean_dist": torch.sqrt(w2_sq.detach() + self.eps).mean(),
            "rda_mean_weight": rda_weights.mean().detach(),
        }
        for task, w in weights.items():
            loss_dict[f"weight_{task}"] = w

        return loss_dict
```

And in `detector/core/losses/strategies/__init__.py`:
```python
from core.losses.strategies.gw_qal import GwQalLossStrategy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_loss_strategy_gw_qal.py -v`  
Expected: PASS with 2 passed

- [ ] **Step 5: Commit**

```bash
git add detector/core/losses/strategies/gw_qal.py detector/core/losses/strategies/__init__.py tests/test_loss_strategy_gw_qal.py
git commit -m "feat(loss): implement GW-QAL loss strategy with closed-form 2D Gaussian Wasserstein"
```

---

### Task 3: Integration Test & Configuration File Creation

**Files:**
- Create: `configs/kitti/oga_loss/kitti_mobilepixornext_gw_qal.json`
- Test: `tests/test_gw_qal_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_gw_qal_integration.py
import json
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.loss_fn import LossFunction


class TestGwQalIntegration(unittest.TestCase):
    def test_json_config_loading_and_backward(self):
        cfg_path = REPO_ROOT / "configs" / "kitti" / "oga_loss" / "kitti_mobilepixornext_gw_qal.json"
        self.assertTrue(cfg_path.exists())
        with open(cfg_path) as f:
            cfg = json.load(f)

        criterion = LossFunction("gaussian", cfg["loss"])
        B, H, W = 2, 16, 16
        pred = {
            "cls": torch.randn(B, 3, H, W, requires_grad=True),
            "offset": torch.randn(B, 2, H, W, requires_grad=True),
            "size": torch.randn(B, 2, H, W, requires_grad=True),
            "yaw": torch.randn(B, 2, H, W, requires_grad=True),
        }
        target = {
            "cls": torch.sigmoid(torch.randn(B, 3, H, W)),
            "offset": torch.zeros(B, 2, H, W),
            "size": torch.zeros(B, 2, H, W),
            "yaw": torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W),
            "reg_mask": torch.ones(B, H, W),
        }
        loss_dict = criterion(pred, target)
        loss = loss_dict["loss"]
        loss.backward()
        self.assertTrue(pred["offset"].grad is not None)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Create config file**

Copy baseline config and update loss definition to `"name": "gw_qal"`.

- [ ] **Step 3: Run integration test**

Run: `pytest tests/test_gw_qal_integration.py -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add configs/kitti/oga_loss/kitti_mobilepixornext_gw_qal.json tests/test_gw_qal_integration.py
git commit -m "feat(configs): add GW-QAL flagship SOTA training configuration and integration test"
```
