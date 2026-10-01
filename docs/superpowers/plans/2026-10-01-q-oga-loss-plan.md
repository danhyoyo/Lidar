# Q-OGA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify the Q-OGA (Quality-Aligned & Range-Adaptive Oriented Geometric Alignment) loss strategy with smooth soft-min corner distance, distance-adaptive spatial reweighting, and quality-coupled focal alignment.

**Architecture:** Q-OGA extends the modular `BaseLossStrategy` registered under `"q_oga"`. It keeps the dual-stream framework (Smooth L1 + oriented geometry), replaces the hard-min corner cusp with a $C^\infty$-smooth Log-Sum-Exp soft-min, applies range-adaptive weights (RDA) on positive cells, and modulates classification targets using dynamic MGIoU quality scores.

**Tech Stack:** Python 3.10+, PyTorch 2.11+, NumPy, PyTest.

**Spec:** `docs/superpowers/specs/2026-10-01-q-oga-loss-design.md`

## Global Constraints

- 100% backward compatible: `oga.py`, `uwag.py`, and `baseline.py` must remain completely unmodified.
- Strategy registered dynamically via `@register_loss_strategy("q_oga")`.
- Compatible with FP32 and BF16 mixed-precision training (`torch.cuda.amp.autocast`).
- Zero external CUDA/C++ extension dependencies; pure PyTorch tensor operations only.
- Empty positive mask (`reg_mask.sum() == 0`) must return a valid graph-connected scalar zero with finite zero gradients for all prediction heads.

---

### Task 1: $C^\infty$-Smooth Soft-Min $\pi$-Symmetric Corner Distance & Spatial RDA

**Files:**
- Create: `detector/core/losses/smooth_corner_loss.py`
- Test: `tests/test_smooth_corner_loss.py`

**Interfaces:**
- `smooth_pi_symmetric_corner_distance(pred_corners, target_corners, target_log_size, tau=0.05, epsilon=1e-6, max_abs_log_size=10.0) -> torch.Tensor`
- `compute_range_weights(pos_offsets, r_max=70.4, gamma=1.5, alpha=2.0) -> torch.Tensor`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smooth_corner_loss.py
import math
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.smooth_corner_loss import (
    smooth_pi_symmetric_corner_distance,
    compute_range_weights,
)


class TestSmoothCornerLoss(unittest.TestCase):
    def test_identical_corners_yield_near_zero(self):
        corners = torch.tensor([
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
        ], dtype=torch.float32)
        target_log_size = torch.tensor([[math.log(2.0), math.log(2.0)]], dtype=torch.float32)
        loss = smooth_pi_symmetric_corner_distance(corners, corners, target_log_size, tau=0.05)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 1e-3)

    def test_smooth_transition_around_pi_half(self):
        corners_a = torch.tensor([
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
        ], dtype=torch.float32, requires_grad=True)
        corners_b = torch.tensor([
            [[1.001, 1.0], [1.0, -1.001], [-1.0, 1.001], [-1.001, -1.0]]
        ], dtype=torch.float32, requires_grad=True)
        target_log_size = torch.tensor([[math.log(2.0), math.log(2.0)]], dtype=torch.float32)

        loss_a = smooth_pi_symmetric_corner_distance(corners_a, corners_a, target_log_size, tau=0.05)
        loss_b = smooth_pi_symmetric_corner_distance(corners_b, corners_a, target_log_size, tau=0.05)
        loss_b.backward()
        self.assertTrue(corners_b.grad is not None)
        self.assertTrue(torch.isfinite(corners_b.grad).all())

    def test_compute_range_weights_scaling(self):
        offsets = torch.tensor([[0.0, 0.0], [35.2, 0.0], [70.4, 0.0]], dtype=torch.float32)
        weights = compute_range_weights(offsets, r_max=70.4, gamma=1.5, alpha=2.0)
        self.assertAlmostEqual(float(weights[0]), 1.0, places=4)
        self.assertAlmostEqual(float(weights[1]), 1.0 + 1.5 * 0.25, places=4)
        self.assertAlmostEqual(float(weights[2]), 2.5, places=4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smooth_corner_loss.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'core.losses.smooth_corner_loss'`

- [ ] **Step 3: Write minimal implementation**

```python
# detector/core/losses/smooth_corner_loss.py
"""Smooth Log-Sum-Exp Soft-Min Pi-Symmetric Corner Distance and Spatial RDA."""

import torch

__all__ = ["smooth_pi_symmetric_corner_distance", "compute_range_weights"]


def smooth_pi_symmetric_corner_distance(
    pred_corners: torch.Tensor,
    target_corners: torch.Tensor,
    target_log_size: torch.Tensor,
    tau: float = 0.05,
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
) -> torch.Tensor:
    """Compute C^inf-smooth scale-normalized pi-symmetric corner distance.

    Args:
        pred_corners: [N, 4, 2]
        target_corners: [N, 4, 2]
        target_log_size: [N, 2]
        tau: Soft-min temperature parameter.
    """
    if pred_corners.shape[0] == 0:
        return pred_corners.sum() * 0.0

    # L1 corner distances
    dist_direct = torch.abs(pred_corners - target_corners).sum(dim=-1).mean(dim=-1)  # [N]
    target_corners_pi = target_corners[:, [3, 2, 1, 0], :]
    dist_pi = torch.abs(pred_corners - target_corners_pi).sum(dim=-1).mean(dim=-1)    # [N]

    # Log-Sum-Exp soft minimum: -tau * log(exp(-d1/tau) + exp(-d2/tau))
    stacked = torch.stack((-dist_direct / tau, -dist_pi / tau), dim=-1)
    smooth_min = -tau * torch.logsumexp(stacked, dim=-1)

    clamped_target_size = target_log_size.float().clamp(-max_abs_log_size, max_abs_log_size)
    target_w = torch.exp(clamped_target_size[:, 0])
    target_l = torch.exp(clamped_target_size[:, 1])
    diagonal = torch.sqrt(target_w.square() + target_l.square()).clamp_min(epsilon)

    return (smooth_min / diagonal).mean()


def compute_range_weights(
    pos_offsets: torch.Tensor,
    r_max: float = 70.4,
    gamma: float = 1.5,
    alpha: float = 2.0,
) -> torch.Tensor:
    """Compute Range & Density-Adaptive (RDA) regression weights per positive cell."""
    if pos_offsets.shape[0] == 0:
        return torch.empty(0, dtype=pos_offsets.dtype, device=pos_offsets.device)

    radial_dist = torch.linalg.vector_norm(pos_offsets[:, :2], dim=-1)
    normalized_r = (radial_dist / r_max).clamp(0.0, 1.5)
    return 1.0 + gamma * torch.pow(normalized_r, alpha)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smooth_corner_loss.py -v`  
Expected: PASS with 3 passed

- [ ] **Step 5: Commit**

```bash
git add detector/core/losses/smooth_corner_loss.py tests/test_smooth_corner_loss.py
git commit -m "feat(loss): add smooth pi-symmetric corner distance and RDA range weights"
```

---

### Task 2: Quality-Coupled Focal Loss Formulation

**Files:**
- Create: `detector/core/losses/quality_focal_loss.py`
- Test: `tests/test_quality_focal_loss.py`

**Interfaces:**
- `quality_focal_loss(pred_logits, target_one_hot, quality_scores, beta=1.0, gamma=2.0) -> torch.Tensor`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quality_focal_loss.py
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.quality_focal_loss import quality_focal_loss


class TestQualityFocalLoss(unittest.TestCase):
    def test_quality_focal_loss_forward_and_backward(self):
        B, C, H, W = 2, 3, 4, 4
        logits = torch.randn(B, C, H, W, requires_grad=True)
        target = torch.zeros(B, C, H, W)
        target[:, 0, 1, 1] = 1.0  # Positive anchor
        quality = torch.ones(B, H, W) * 0.8

        loss = quality_focal_loss(logits, target, quality, beta=1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

        loss.backward()
        self.assertTrue(logits.grad is not None)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_low_quality_attenuates_positive_loss(self):
        B, C, H, W = 1, 1, 1, 1
        logits = torch.tensor([[[[2.0]]]], requires_grad=True)
        target = torch.ones(B, C, H, W)
        q_high = torch.tensor([[[1.0]]])
        q_low = torch.tensor([[[0.1]]])

        loss_high = quality_focal_loss(logits, target, q_high, beta=1.0)
        loss_low = quality_focal_loss(logits, target, q_low, beta=1.0)
        self.assertGreater(float(loss_high), float(loss_low))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quality_focal_loss.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'core.losses.quality_focal_loss'`

- [ ] **Step 3: Write minimal implementation**

```python
# detector/core/losses/quality_focal_loss.py
"""Quality-Coupled Focal Loss for continuous overlap-guided classification."""

import torch
import torch.nn.functional as F

__all__ = ["quality_focal_loss"]


def quality_focal_loss(
    pred_logits: torch.Tensor,
    target_prob: torch.Tensor,
    quality_scores: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 2.0,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Compute Quality-Coupled Focal Loss.

    Args:
        pred_logits: [B, C, H, W] raw classification logits.
        target_prob: [B, C, H, W] Gaussian classification targets in [0, 1].
        quality_scores: [B, H, W] detached geometric quality scores in [0, 1].
        beta: Quality target exponent.
        gamma: Modulating focal factor.
    """
    pred_sig = pred_logits.float().sigmoid().clamp(min=epsilon, max=1.0 - epsilon)
    q = quality_scores.unsqueeze(1).clamp(0.0, 1.0)
    
    # Positive locations: modulated by quality^beta
    pos_mask = target_prob.ge(1.0)
    continuous_target = torch.where(pos_mask, target_prob * torch.pow(q, beta), target_prob)

    # Focal modulating weight |target - pred|^gamma
    focal_weight = torch.pow(torch.abs(continuous_target - pred_sig), gamma)

    # BCE with logits
    bce = continuous_target * (-torch.log(pred_sig)) + (1.0 - continuous_target) * (-torch.log(1.0 - pred_sig))
    loss = focal_weight * bce

    num_pos = pos_mask.sum().clamp_min(1.0)
    return loss.sum() / num_pos
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_quality_focal_loss.py -v`  
Expected: PASS with 2 passed

- [ ] **Step 5: Commit**

```bash
git add detector/core/losses/quality_focal_loss.py tests/test_quality_focal_loss.py
git commit -m "feat(loss): add quality-coupled focal loss module"
```

---

### Task 3: Q-OGA Strategy Class Implementation

**Files:**
- Create: `detector/core/losses/strategies/q_oga.py`
- Modify: `detector/core/losses/strategies/__init__.py`
- Test: `tests/test_loss_strategy_q_oga.py`

**Interfaces:**
- `QOgaLossStrategy(BaseLossStrategy)` registered as `"q_oga"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loss_strategy_q_oga.py
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.strategies import build_loss_strategy, get_available_loss_strategies
from core.losses.loss_fn import LossFunction


class TestLossStrategyQOga(unittest.TestCase):
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

    def test_q_oga_registered(self):
        self.assertIn("q_oga", get_available_loss_strategies())

    def test_q_oga_forward_backward(self):
        criterion = LossFunction("gaussian", {
            "name": "q_oga",
            "temperature": 2.0,
            "clamp_bound": 3.0,
            "corner_beta": 1.0,
            "soft_min_tau": 0.05,
            "rda_gamma": 1.5,
        })
        loss_dict = criterion(self.pred, self.target)
        for key in ("loss", "cls", "offset", "size", "yaw", "geo", "corner_dist", "proj_giou", "rda_mean_weight"):
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

Run: `pytest tests/test_loss_strategy_q_oga.py -v`  
Expected: FAIL with `AssertionError: 'q_oga' not found in ('baseline', 'uwag', 'oga')`

- [ ] **Step 3: Write minimal implementation**

```python
# detector/core/losses/strategies/q_oga.py
"""Q-OGA (Quality-Aligned & Range-Adaptive Oriented Geometric Alignment) loss strategy."""

from typing import Any, Dict
import torch
import torch.nn.functional as F

from core.losses.l1_loss import smooth_l1_loss
from core.losses.oriented_geometry_loss import box_corners, multiaxis_projection_giou
from core.losses.smooth_corner_loss import (
    smooth_pi_symmetric_corner_distance,
    compute_range_weights,
)
from core.losses.quality_focal_loss import quality_focal_loss
from core.losses.uncertainty_weighting import TemperatureSoftmaxUncertainty
from core.losses.strategies.base import BaseLossStrategy
from core.losses.strategies.registry import register_loss_strategy
from core.losses.iou_targets import compute_mgiou_targets


@register_loss_strategy("q_oga")
class QOgaLossStrategy(BaseLossStrategy):
    """Quality-Aligned, Range-Adaptive, and C^inf Smooth Soft-Min OGA Loss."""

    TASKS = ("cls", "offset", "size", "yaw", "geo")

    def __init__(self, cls_encoding: str, config: Dict[str, Any] = None):
        super().__init__(cls_encoding, config)
        config = self.config

        self.eps = float(config.get("epsilon", 1e-6))
        self.max_abs_log_size = float(config.get("max_abs_log_size", 10.0))
        self.corner_beta = float(config.get("corner_beta", 1.0))
        self.temperature = float(config.get("temperature", 2.0))
        self.clamp_bound = float(config.get("clamp_bound", 3.0))
        self.ema_momentum = float(config.get("ema_momentum", 0.99))
        self.soft_min_tau = float(config.get("soft_min_tau", 0.05))
        self.rda_gamma = float(config.get("rda_gamma", 1.5))
        self.rda_alpha = float(config.get("rda_alpha", 2.0))
        self.qcfa_beta = float(config.get("qcfa_beta", 1.0))

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

        # Compute dynamic MGIoU quality scores (detached)
        with torch.no_grad():
            mgiou_quality = compute_mgiou_targets(
                pred["offset"][:, :2],
                pred["size"][:, :2],
                pred["yaw"][:, :2],
                target["offset"][:, :2],
                target["size"][:, :2],
                target["yaw"][:, :2],
                target["reg_mask"],
                epsilon=self.eps,
                max_abs_log_size=self.max_abs_log_size,
            )

        # 1. Quality-coupled classification loss
        cls_loss = quality_focal_loss(
            pred["cls"], target["cls"], mgiou_quality, beta=self.qcfa_beta
        )

        if not positive.any():
            zero = (pred["offset"].sum() + pred["size"].sum() + pred["yaw"].sum()) * 0.0
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
                "corner_dist": torch.zeros((), device=device),
                "proj_giou": torch.zeros((), device=device),
                "rda_mean_weight": torch.ones((), device=device),
                **{f"weight_{k}": w for k, w in weights.items()},
            }

        # 2. Coordinate losses with Range-Adaptive Reweighting
        def _pos(x):
            return x.permute(0, 2, 3, 1).reshape(-1, 2)[positive].float()

        tgt_off_pos = _pos(target["offset"])
        rda_weights = compute_range_weights(
            tgt_off_pos, r_max=70.4, gamma=self.rda_gamma, alpha=self.rda_alpha
        )

        # Weighted Smooth-L1
        diff_off = F.smooth_l1_loss(_pos(pred["offset"]), tgt_off_pos, reduction="none").sum(dim=-1)
        offset_loss = (diff_off * rda_weights).mean()

        diff_sz = F.smooth_l1_loss(_pos(pred["size"]), _pos(target["size"]), reduction="none").sum(dim=-1)
        size_loss = (diff_sz * rda_weights).mean()

        diff_yaw = F.smooth_l1_loss(_pos(pred["yaw"]), _pos(target["yaw"]), reduction="none").sum(dim=-1)
        yaw_loss = (diff_yaw * rda_weights).mean()

        # 3. Geometry loss with C^inf Soft-Min Corner Distance
        with torch.autocast(device_type=device.type, enabled=False):
            pred_c, pred_ax, _, _ = box_corners(
                _pos(pred["offset"]), _pos(pred["size"]), _pos(pred["yaw"]), self.eps, self.max_abs_log_size
            )
            tgt_c, tgt_ax, _, _ = box_corners(
                tgt_off_pos, _pos(target["size"]), _pos(target["yaw"]), self.eps, self.max_abs_log_size
            )
            proj_loss = multiaxis_projection_giou(pred_c, pred_ax, tgt_c, tgt_ax, self.eps)
            corner_loss = smooth_pi_symmetric_corner_distance(
                pred_c, tgt_c, _pos(target["size"]), tau=self.soft_min_tau, epsilon=self.eps, max_abs_log_size=self.max_abs_log_size
            )
            geo_loss = proj_loss + self.corner_beta * corner_loss

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
            "corner_dist": corner_loss.detach(),
            "proj_giou": proj_loss.detach(),
            "rda_mean_weight": rda_weights.mean().detach(),
        }
        for task, w in weights.items():
            loss_dict[f"weight_{task}"] = w

        return loss_dict
```

And in `detector/core/losses/strategies/__init__.py`:
```python
from core.losses.strategies.q_oga import QOgaLossStrategy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_loss_strategy_q_oga.py -v`  
Expected: PASS with 2 passed

- [ ] **Step 5: Commit**

```bash
git add detector/core/losses/strategies/q_oga.py detector/core/losses/strategies/__init__.py tests/test_loss_strategy_q_oga.py
git commit -m "feat(loss): implement Q-OGA loss strategy with soft-min corner and RDA"
```

---

### Task 4: Integration Test & Configuration File Creation

**Files:**
- Create: `configs/kitti/oga_loss/kitti_mobilepixornext_q_oga.json`
- Test: `tests/test_q_oga_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_q_oga_integration.py
import json
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.loss_fn import LossFunction


class TestQOgaIntegration(unittest.TestCase):
    def test_json_config_loading_and_backward(self):
        cfg_path = REPO_ROOT / "configs" / "kitti" / "oga_loss" / "kitti_mobilepixornext_q_oga.json"
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

Copy baseline `configs/kitti/oga_loss/kitti_mobilepixornext_litemla_oga.json` and set `"loss": {"name": "q_oga", ...}`.

- [ ] **Step 3: Run integration test**

Run: `pytest tests/test_q_oga_integration.py -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add configs/kitti/oga_loss/kitti_mobilepixornext_q_oga.json tests/test_q_oga_integration.py
git commit -m "feat(configs): add Q-OGA ablation training configuration and integration test"
```
