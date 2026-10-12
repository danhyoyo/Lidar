# Pillar 4: IoU-Aware Quality Detection Header (IQA-Header) & Joint NMS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the IoU-Aware Quality Detection Header (IQA-Header) and Joint Quality NMS post-processing to eliminate the "Score-IoU Misalignment" phenomenon in 3D LiDAR object detection, elevating KITTI $AP_{3D}$ and $AP_{BEV}$ at strict thresholds ($\ge 0.70$) with near-zero inference latency overhead ($+0.08\text{ ms}$).

**Architecture:**
- Extend `Header` in `detector/core/models/heads/cnn.py` with an optional 5th prediction branch `self.iou` (1 channel logit) conditioned on `use_iou: bool = False`.
- Create a pure-PyTorch GPU vectorized target generator in `detector/core/losses/iou_targets.py` supporting both Option C (`mgiou`) and Option B (`yaw_footprint`) on detached box predictions.
- Integrate the IoU loss ($\mathcal{L}_{\text{iou}}$ via BCEWithLogits on positive anchors) into `OgaLossStrategy` in `detector/core/losses/strategies/oga.py`, dynamically expanding `TemperatureSoftmaxUncertainty` to 6 tasks.
- Upgrade `filter_pred` in `detector/postprocess.py` to calculate calibrated joint ranking scores $S_{\text{final}} = P_{\text{cls}}^{1 - \alpha} \cdot S_{\text{iou}}^{\alpha}$ when `iou` is present.
- Provide a dedicated config `kitti_mobilepixornext_litemla_oga_reparam_iqa.json` while maintaining 100% backward compatibility for standard baseline models (`header_use_iou: false`, `nms_alpha: 0.0`).

**Tech Stack:** Python 3.14, PyTorch 2.11, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-iou-aware-quality-header-design.md`

## Global Constraints
- Target branch is strictly `feature/pillar4-iou-aware-header` (inheriting from `feature/pillar1-structural-reparameterization`).
- Zero external heavy CUDA dependencies: all operations must be pure PyTorch 2.x tensor operations running on GPU/CPU.
- 100% Backward Compatibility: When `use_iou=False` (or omitted), models, losses, and NMS must behave identically to the baseline with no overhead.
- Gradient Isolation: $\widehat{B}$ must be `.detach()`-ed during target generation so that $\mathcal{L}_{\text{iou}}$ produces **zero gradient** on `offset`, `size`, and `yaw` heads.
- Pytest execution environment: `PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint`.

---

## File Structure

| File | Responsibility |
| :--- | :--- |
| `detector/core/models/heads/cnn.py` | Add `use_iou` parameter and `self.iou` 1-channel head to `Header`. |
| `detector/core/models/model.py` | Pass `cfg.get("header_use_iou", False)` to `Header` in `CustomModel`. |
| `detector/core/losses/iou_targets.py` | Pure PyTorch tensor target generators for MGIoU and Yaw-Footprint with detached inputs. |
| `detector/core/losses/strategies/oga.py` | Integrate $\mathcal{L}_{\text{iou}}$ (BCEWithLogits) and dynamic 6-task uncertainty weighting into `OgaLossStrategy`. |
| `detector/postprocess.py` | Update `filter_pred` to compute $S_{\text{final}} = P_{\text{cls}}^{1 - \alpha} \cdot S_{\text{iou}}^{\alpha}$ and sort NMS candidates accordingly. |
| `configs/kitti/iou_aware_header/kitti_mobilepixornext_litemla_oga_reparam_iqa.json` | Complete Pillar 4 experiment configuration. |
| `tests/test_iqa_header.py` | Comprehensive test suite for model head, target generation, loss gradients, NMS reordering, and regression. |

---

## Task Breakdown

### Task 1: Add IoU Quality Prediction Branch to `Header` and `CustomModel`

**Files:**
- Modify: `detector/core/models/heads/cnn.py`
- Modify: `detector/core/models/model.py`
- Test: `tests/test_iqa_header.py`

**Interfaces:**
- `Header(num_classes: int, in_channels: int, use_bn: bool = False, act: str = "none", use_iou: bool = False)`
- `Header.forward(x: torch.Tensor) -> Dict[str, torch.Tensor]` (returns `{"cls", "offset", "size", "yaw", "iou"}` when `use_iou=True`)
- `CustomModel(cfg: Dict[str, Any], ...)` threads `cfg.get("header_use_iou", False)` into `Header`.

- [x] **Step 1: Write failing unit test for `Header` and `CustomModel` IoU branch**

Create `tests/test_iqa_header.py`:
```python
import torch
import pytest
from core.models.heads.cnn import Header
from core.models.model import CustomModel

def test_header_backward_compatibility():
    """Verify Header without use_iou preserves exactly 4 heads."""
    header = Header(num_classes=3, in_channels=16, use_bn=True, act="silu", use_iou=False)
    x = torch.randn(2, 16, 50, 44)
    pred = header(x)
    assert set(pred.keys()) == {"cls", "offset", "size", "yaw"}
    assert "iou" not in pred

def test_header_with_iou_branch():
    """Verify Header with use_iou outputs 1-channel 'iou' logit."""
    header = Header(num_classes=3, in_channels=16, use_bn=True, act="silu", use_iou=True)
    x = torch.randn(2, 16, 50, 44)
    pred = header(x)
    assert set(pred.keys()) == {"cls", "offset", "size", "yaw", "iou"}
    assert pred["iou"].shape == (2, 1, 50, 44)

def test_custom_model_threads_iou_flag():
    """Verify CustomModel initializes Header with use_iou according to config."""
    cfg = {
        "backbone": "mobilepixornext",
        "backbone_out_dim": 16,
        "header_use_bn": True,
        "header_act": "silu",
        "header_use_iou": True,
    }
    model = CustomModel(cfg, num_classes=3, input_channels=35)
    x = torch.randn(2, 35, 200, 176)
    pred = model(x)
    assert "iou" in pred
    assert pred["iou"].shape == (2, 1, 200, 176)
```

- [x] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_iqa_header.py
```
Expected output: Fails due to `unexpected keyword argument 'use_iou'`.

- [x] **Step 3: Implement minimal code in `Header` and `CustomModel`**

In `detector/core/models/heads/cnn.py`:
```python
class Header(nn.Module):
    def __init__(self, num_classes, in_channels, use_bn=False, act="none", use_iou=False):
        super(Header, self).__init__()
        self.use_iou = use_iou
        self.cls = Head(in_channels, num_classes, use_bn=use_bn, act=act)
        self.offset = Head(in_channels, 2, use_bn=use_bn, act=act)
        self.size = Head(in_channels, 2, use_bn=use_bn, act=act)
        self.yaw = Head(in_channels, 2, use_bn=use_bn, act=act)
        if self.use_iou:
            self.iou = Head(in_channels, 1, use_bn=use_bn, act=act)

    def forward(self, x):
        pred = {
            "cls": self.cls(x),
            "offset": self.offset(x),
            "size": self.size(x),
            "yaw": self.yaw(x),
        }
        if self.use_iou:
            pred["iou"] = self.iou(x)
        return pred
```

In `detector/core/models/model.py`:
```python
        use_iou = bool(cfg.get("header_use_iou", False))
        self.header = Header(
            self.num_classes,
            backbone_out_dim,
            use_bn=use_bn,
            act=act,
            use_iou=use_iou,
        )
```

- [x] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_iqa_header.py
```
Expected output: 3 passed in <2s.

- [x] **Step 5: Commit changes**

```bash
git add detector/core/models/heads/cnn.py detector/core/models/model.py tests/test_iqa_header.py
git commit -m "feat(head): add 1-channel IoU prediction branch to Header and CustomModel"
```

---

### Task 2: Implement Pure-PyTorch Target Generation Primitives (`iou_targets.py`)

**Files:**
- Create: `detector/core/losses/iou_targets.py`
- Test: `tests/test_iqa_header.py`

**Interfaces:**
- Produces: `compute_mgiou_targets(pred_offset, pred_size, pred_yaw, target_offset, target_size, target_yaw, reg_mask, epsilon=1e-6, max_abs_log_size=10.0) -> torch.Tensor`
- Produces: `compute_yaw_footprint_targets(pred_offset, pred_size, pred_yaw, target_offset, target_size, target_yaw, reg_mask, epsilon=1e-6, max_abs_log_size=10.0) -> torch.Tensor`
- Produces: `compute_iou_targets(pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], method: str = "mgiou", epsilon: float = 1e-6, max_abs_log_size: float = 10.0) -> torch.Tensor`
  - Input: tensors of shape `[B, 2, H, W]` and `reg_mask` of shape `[B, H, W]`
  - Output: float tensor of shape `[B, H, W]` bounded in $[0.0, 1.0]$.

- [x] **Step 1: Write failing unit tests for target generation (MGIoU & Yaw-Footprint)**

Append to `tests/test_iqa_header.py`:
```python
from core.losses.iou_targets import compute_iou_targets

def test_compute_iou_targets_identical_boxes():
    """Verify target IoU is 1.0 when prediction matches target exactly."""
    B, H, W = 2, 10, 10
    offset = torch.zeros(B, 2, H, W)
    size = torch.zeros(B, 2, H, W) # exp(0) = 1.0m width, 1.0m length
    yaw = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W)
    reg_mask = torch.ones(B, H, W, dtype=torch.bool)

    pred = {"offset": offset.clone(), "size": size.clone(), "yaw": yaw.clone()}
    target = {"offset": offset.clone(), "size": size.clone(), "yaw": yaw.clone(), "reg_mask": reg_mask}

    for method in ["mgiou", "yaw_footprint"]:
        target_iou = compute_iou_targets(pred, target, method=method)
        assert target_iou.shape == (B, H, W)
        assert torch.allclose(target_iou, torch.ones_like(target_iou), atol=1e-4)

def test_compute_iou_targets_orthogonal_yaw():
    """Verify target IoU is heavily penalized (near 0) when yaw is rotated 90 degrees."""
    B, H, W = 1, 4, 4
    offset = torch.zeros(B, 2, H, W)
    size = torch.zeros(B, 2, H, W)
    pred_yaw = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W) # cos(2*0) = 1
    tgt_yaw = torch.tensor([-1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W) # cos(2*pi/2) = -1 (90 deg)
    reg_mask = torch.ones(B, H, W, dtype=torch.bool)

    pred = {"offset": offset, "size": size, "yaw": pred_yaw}
    target = {"offset": offset, "size": size, "yaw": tgt_yaw, "reg_mask": reg_mask}

    for method in ["mgiou", "yaw_footprint"]:
        target_iou = compute_iou_targets(pred, target, method=method)
        assert (target_iou < 0.35).all(), f"Method {method} did not penalize 90 deg yaw: max={target_iou.max()}"

def test_compute_iou_targets_distant_boxes():
    """Verify target IoU is 0.0 for completely disjoint boxes."""
    B, H, W = 1, 2, 2
    pred_offset = torch.zeros(B, 2, H, W)
    tgt_offset = torch.full((B, 2, H, W), 100.0) # 100m away
    size = torch.zeros(B, 2, H, W)
    yaw = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W)
    reg_mask = torch.ones(B, H, W, dtype=torch.bool)

    pred = {"offset": pred_offset, "size": size, "yaw": yaw}
    target = {"offset": tgt_offset, "size": size, "yaw": yaw, "reg_mask": reg_mask}

    for method in ["mgiou", "yaw_footprint"]:
        target_iou = compute_iou_targets(pred, target, method=method)
        assert torch.all(target_iou == 0.0)
```

- [x] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_iqa_header.py
```
Expected output: Fails due to `ModuleNotFoundError: No module named 'core.losses.iou_targets'`.

- [x] **Step 3: Implement `detector/core/losses/iou_targets.py`**

Create `detector/core/losses/iou_targets.py`:
```python
"""Dynamic IoU supervision target generator for IQA-Header."""

from typing import Dict
import torch
import torch.nn.functional as F

from core.losses.oriented_geometry_loss import box_corners, multiaxis_projection_giou


def compute_mgiou_targets(
    pred_offset: torch.Tensor,
    pred_size: torch.Tensor,
    pred_yaw: torch.Tensor,
    target_offset: torch.Tensor,
    target_size: torch.Tensor,
    target_yaw: torch.Tensor,
    reg_mask: torch.Tensor,
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
) -> torch.Tensor:
    """Compute per-cell rotated BEV IoU targets using Multi-Axis Projection GIoU.

    Predictions are detached to prevent circular regression gradients.
    Returns: [B, H, W] tensor in range [0.0, 1.0].
    """
    B, _, H, W = pred_offset.shape
    device = pred_offset.device
    target_iou = torch.zeros((B, H, W), dtype=torch.float32, device=device)

    pos_mask = reg_mask.bool()
    if not pos_mask.any():
        return target_iou

    # Select positive anchors: [N, 2]
    # IMPORTANT: pred inputs must be detached!
    p_off = pred_offset.detach().permute(0, 2, 3, 1)[pos_mask]
    p_sz = pred_size.detach().permute(0, 2, 3, 1)[pos_mask]
    p_yaw = pred_yaw.detach().permute(0, 2, 3, 1)[pos_mask]

    t_off = target_offset.permute(0, 2, 3, 1)[pos_mask]
    t_sz = target_size.permute(0, 2, 3, 1)[pos_mask]
    t_yaw = target_yaw.permute(0, 2, 3, 1)[pos_mask]

    pred_c, pred_ax, _, _ = box_corners(p_off, p_sz, p_yaw, epsilon, max_abs_log_size)
    tgt_c, tgt_ax, _, _ = box_corners(t_off, t_sz, t_yaw, epsilon, max_abs_log_size)

    # 4 projection axes: [N, 4, 2]
    axes = torch.cat((pred_ax, tgt_ax), dim=1)

    # Project corners onto axes: [N, 4, 4]
    pred_proj = torch.einsum("ncd,nad->nac", pred_c, axes)
    target_proj = torch.einsum("ncd,nad->nac", tgt_c, axes)

    pred_min, pred_max = pred_proj.amin(dim=-1), pred_proj.amax(dim=-1)        # [N, 4]
    target_min, target_max = target_proj.amin(dim=-1), target_proj.amax(dim=-1)  # [N, 4]

    intersection = (
        torch.minimum(pred_max, target_max) - torch.maximum(pred_min, target_min)
    ).clamp_min(0.0)
    union = pred_max - pred_min + target_max - target_min - intersection
    hull = torch.maximum(pred_max, target_max) - torch.minimum(pred_min, target_min)

    giou = intersection / union.clamp_min(epsilon) - (hull - union) / hull.clamp_min(epsilon)
    mean_giou = giou.mean(dim=-1)  # [N]

    # Map GIoU [-1, 1] to target similarity [0, 1]
    cell_iou = mean_giou.clamp(0.0, 1.0)
    target_iou[pos_mask] = cell_iou
    return target_iou


def compute_yaw_footprint_targets(
    pred_offset: torch.Tensor,
    pred_size: torch.Tensor,
    pred_yaw: torch.Tensor,
    target_offset: torch.Tensor,
    target_size: torch.Tensor,
    target_yaw: torch.Tensor,
    reg_mask: torch.Tensor,
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
) -> torch.Tensor:
    """Compute per-cell footprint IoU modulated by doubled-yaw agreement."""
    B, _, H, W = pred_offset.shape
    device = pred_offset.device
    target_iou = torch.zeros((B, H, W), dtype=torch.float32, device=device)

    pos_mask = reg_mask.bool()
    if not pos_mask.any():
        return target_iou

    p_off = pred_offset.detach().permute(0, 2, 3, 1)[pos_mask]
    p_sz = pred_size.detach().permute(0, 2, 3, 1)[pos_mask]
    p_yaw = pred_yaw.detach().permute(0, 2, 3, 1)[pos_mask]

    t_off = target_offset.permute(0, 2, 3, 1)[pos_mask]
    t_sz = target_size.permute(0, 2, 3, 1)[pos_mask]
    t_yaw = target_yaw.permute(0, 2, 3, 1)[pos_mask]

    # Footprint 2D axis-aligned IoU
    p_w = torch.exp(p_sz[:, 0].clamp(-max_abs_log_size, max_abs_log_size))
    p_l = torch.exp(p_sz[:, 1].clamp(-max_abs_log_size, max_abs_log_size))
    t_w = torch.exp(t_sz[:, 0].clamp(-max_abs_log_size, max_abs_log_size))
    t_l = torch.exp(t_sz[:, 1].clamp(-max_abs_log_size, max_abs_log_size))

    pred_min_x, pred_max_x = p_off[:, 0] - 0.5 * p_w, p_off[:, 0] + 0.5 * p_w
    pred_min_y, pred_max_y = p_off[:, 1] - 0.5 * p_l, p_off[:, 1] + 0.5 * p_l

    tgt_min_x, tgt_max_x = t_off[:, 0] - 0.5 * t_w, t_off[:, 0] + 0.5 * t_w
    tgt_min_y, tgt_max_y = t_off[:, 1] - 0.5 * t_l, t_off[:, 1] + 0.5 * t_l

    inter_x = (torch.minimum(pred_max_x, tgt_max_x) - torch.maximum(pred_min_x, tgt_min_x)).clamp_min(0.0)
    inter_y = (torch.minimum(pred_max_y, tgt_max_y) - torch.maximum(pred_min_y, tgt_min_y)).clamp_min(0.0)
    intersection = inter_x * inter_y

    pred_area = p_w * p_l
    tgt_area = t_w * t_l
    union = pred_area + tgt_area - intersection
    footprint_iou = intersection / union.clamp_min(epsilon)

    # Doubled-yaw cosine similarity
    p_yaw_u = F.normalize(p_yaw, dim=-1, eps=epsilon)
    t_yaw_u = F.normalize(t_yaw, dim=-1, eps=epsilon)
    yaw_cos = (p_yaw_u * t_yaw_u).sum(dim=-1).clamp(-1.0, 1.0)
    yaw_factor = 0.5 * (1.0 + yaw_cos)

    cell_iou = (footprint_iou * yaw_factor).clamp(0.0, 1.0)
    target_iou[pos_mask] = cell_iou
    return target_iou


def compute_iou_targets(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    method: str = "mgiou",
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
) -> torch.Tensor:
    """Unified dispatcher for IoU target computation."""
    reg_mask = target["reg_mask"]
    if method == "mgiou":
        return compute_mgiou_targets(
            pred["offset"][:, :2],
            pred["size"][:, :2],
            pred["yaw"][:, :2],
            target["offset"][:, :2],
            target["size"][:, :2],
            target["yaw"][:, :2],
            reg_mask,
            epsilon=epsilon,
            max_abs_log_size=max_abs_log_size,
        )
    elif method == "yaw_footprint":
        return compute_yaw_footprint_targets(
            pred["offset"][:, :2],
            pred["size"][:, :2],
            pred["yaw"][:, :2],
            target["offset"][:, :2],
            target["size"][:, :2],
            target["yaw"][:, :2],
            reg_mask,
            epsilon=epsilon,
            max_abs_log_size=max_abs_log_size,
        )
    else:
        raise ValueError(f"Unknown IoU target method: {method!r}. Expected 'mgiou' or 'yaw_footprint'.")
```

- [x] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_iqa_header.py
```
Expected output: 6 passed in <3s.

- [x] **Step 5: Commit changes**

```bash
git add detector/core/losses/iou_targets.py tests/test_iqa_header.py
git commit -m "feat(loss): implement pure-PyTorch GPU dynamic IoU target generators for MGIoU and Yaw-Footprint"
```

---

### Task 3: Integrate IoU Loss into `OgaLossStrategy` & Multi-Task Uncertainty

**Files:**
- Modify: `detector/core/losses/strategies/oga.py`
- Test: `tests/test_iqa_header.py`

**Interfaces:**
- `OgaLossStrategy` accepts config keys: `use_iou: bool = False`, `iou_target_type: str = "mgiou"`, `iou_loss_weight: float = 1.0`.
- When `use_iou=True`:
  - `TASKS = ("cls", "offset", "size", "yaw", "geo", "iou")`
  - Computes $\mathcal{L}_{\text{iou}} = \text{BCEWithLogitsLoss}(pred[\text{"iou"}], y_{\text{iou}})$ on `reg_mask == 1`.
  - Output dictionary contains `"iou"`, `"weight_iou"`, and `"mean_iou_target"`.

- [x] **Step 1: Write failing unit test for `OgaLossStrategy` with IoU branch & gradient isolation**

Append to `tests/test_iqa_header.py`:
```python
from core.losses.strategies.oga import OgaLossStrategy

def test_oga_loss_strategy_with_iou():
    """Verify OgaLossStrategy computes iou loss and uncertainty weight when use_iou=True."""
    config = {
        "use_iou": True,
        "iou_target_type": "mgiou",
        "iou_loss_weight": 1.0,
    }
    strategy = OgaLossStrategy(cls_encoding="gaussian", config=config)
    assert "iou" in strategy.TASKS

    B, C, H, W = 2, 3, 20, 20
    pred = {
        "cls": torch.randn(B, C, H, W, requires_grad=True),
        "offset": torch.randn(B, 2, H, W, requires_grad=True),
        "size": torch.randn(B, 2, H, W, requires_grad=True),
        "yaw": torch.randn(B, 2, H, W, requires_grad=True),
        "iou": torch.randn(B, 1, H, W, requires_grad=True),
    }
    target = {
        "cls": torch.rand(B, C, H, W),
        "offset": torch.randn(B, 2, H, W),
        "size": torch.randn(B, 2, H, W),
        "yaw": torch.randn(B, 2, H, W),
        "reg_mask": torch.randint(0, 2, (B, H, W), dtype=torch.bool),
    }

    out = strategy(pred, target)
    assert "loss" in out
    assert "iou" in out
    assert "weight_iou" in out
    assert out["loss"].requires_grad

    # Test gradient flow to pred["iou"]
    out["loss"].backward()
    assert pred["iou"].grad is not None
    assert torch.isfinite(pred["iou"].grad).all()

def test_oga_loss_gradient_isolation_on_regression():
    """Verify that pred['offset'], pred['size'], pred['yaw'] do NOT receive gradient from iou_target."""
    config = {"use_iou": True, "iou_target_type": "mgiou"}
    strategy = OgaLossStrategy(cls_encoding="gaussian", config=config)

    B, C, H, W = 1, 1, 10, 10
    pred = {
        "cls": torch.zeros(B, C, H, W),
        "offset": torch.zeros(B, 2, H, W, requires_grad=True),
        "size": torch.zeros(B, 2, H, W, requires_grad=True),
        "yaw": torch.zeros(B, 2, H, W, requires_grad=True),
        "iou": torch.zeros(B, 1, H, W, requires_grad=True),
    }
    target = {
        "cls": torch.zeros(B, C, H, W),
        "offset": torch.zeros(B, 2, H, W),
        "size": torch.zeros(B, 2, H, W),
        "yaw": torch.zeros(B, 2, H, W),
        "reg_mask": torch.ones(B, H, W, dtype=torch.bool),
    }

    # Compute ONLY iou loss isolated
    target_iou = compute_iou_targets(pred, target, method="mgiou")
    loss_iou = torch.nn.functional.binary_cross_entropy_with_logits(
        pred["iou"].squeeze(1), target_iou
    )
    loss_iou.backward()

    # Regression heads must receive ZERO gradient
    assert pred["offset"].grad is None or (pred["offset"].grad == 0).all()
    assert pred["size"].grad is None or (pred["size"].grad == 0).all()
    assert pred["yaw"].grad is None or (pred["yaw"].grad == 0).all()
    # iou logit head MUST receive valid gradient
    assert pred["iou"].grad is not None and not (pred["iou"].grad == 0).all()
```

- [x] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_iqa_header.py
```
Expected output: Fails due to `assert "iou" in strategy.TASKS`.

- [x] **Step 3: Update `OgaLossStrategy` in `detector/core/losses/strategies/oga.py`**

Modify `detector/core/losses/strategies/oga.py`:
- Read `self.use_iou = bool(config.get("use_iou", False))`
- Set `self.iou_target_type = str(config.get("iou_target_type", "mgiou"))`
- Set `self.iou_loss_weight = float(config.get("iou_loss_weight", 1.0))`
- Dynamically define `self.TASKS`:
  ```python
  if self.use_iou:
      self.TASKS = ("cls", "offset", "size", "yaw", "geo", "iou")
  else:
      self.TASKS = ("cls", "offset", "size", "yaw", "geo")
  ```
- In `forward`:
  ```python
  if self.use_iou:
      if "iou" not in pred:
          raise KeyError("Missing required prediction head: 'iou' when use_iou=True")
      iou_target = compute_iou_targets(
          pred, target, method=self.iou_target_type, epsilon=self.eps, max_abs_log_size=self.max_abs_log_size
      )
      pos_mask = target["reg_mask"].bool()
      if pos_mask.any():
          pred_iou_pos = pred["iou"].squeeze(1)[pos_mask]
          target_iou_pos = iou_target[pos_mask]
          iou_loss = F.binary_cross_entropy_with_logits(pred_iou_pos, target_iou_pos)
      else:
          iou_loss = 0.0 * pred["iou"].sum()
      task_losses["iou"] = self.iou_loss_weight * iou_loss
  ```
- Add `"iou": iou_loss.detach()`, `"weight_iou": weights.get("iou", 1.0)` and `"mean_iou_target": iou_target[pos_mask].mean().detach()` (if positive) to telemetry dict.

- [x] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_iqa_header.py
```
Expected output: 8 passed in <3s.

- [x] **Step 5: Commit changes**

```bash
git add detector/core/losses/strategies/oga.py tests/test_iqa_header.py
git commit -m "feat(loss): integrate IoU quality loss and dynamic uncertainty balancing into OgaLossStrategy"
```

---

### Task 4: Implement Calibrated Joint NMS in `detector/postprocess.py`

**Files:**
- Modify: `detector/postprocess.py`
- Test: `tests/test_iqa_header.py`

**Interfaces:**
- `filter_pred(pred: Dict[str, torch.Tensor], config: Dict[str, Any], out_size_factor: int, thres: float, nms_thres: Optional[float] = None)`
- When `"iou"` in `pred` and `alpha = config.get("nms_alpha", 0.5) > 0`:
  - Joint score: $S_{\text{final}} = (P_{\text{cls}})^{1 - \alpha} \cdot (\sigma(\text{iou}))^{\alpha}$
  - `candidate_mask` uses `S_final > thres`
  - Rotated NMS ranks candidates by `candidate_scores = S_final[candidate_mask]`

- [x] **Step 1: Write failing unit test for Joint NMS Score Reordering**

Append to `tests/test_iqa_header.py`:
```python
from postprocess import filter_pred

def test_filter_pred_score_reordering_with_iou():
    """Verify that a candidate with high IoU and moderate cls score beats a candidate with high cls and low IoU."""
    config = {
        "geometry": {
            "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
            "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        },
        "nms_alpha": 0.5,
    }
    out_size_factor = 4
    # Grid shape: H=200, W=176
    H, W = 200, 176
    cls_pred = torch.full((1, 1, H, W), -10.0) # all low logits
    offset_pred = torch.zeros(1, 2, H, W)
    size_pred = torch.zeros(1, 2, H, W)
    yaw_pred = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(1, 2, H, W)
    iou_pred = torch.zeros(1, 1, H, W)

    # Candidate A at (50, 50): high cls logit (+4.0 -> p~0.98), low iou logit (-3.0 -> s~0.047)
    # S_final = sqrt(0.98 * 0.047) ~ 0.21
    cls_pred[0, 0, 50, 50] = 4.0
    iou_pred[0, 0, 50, 50] = -3.0

    # Candidate B at (60, 60): moderate cls logit (+1.5 -> p~0.81), high iou logit (+3.0 -> s~0.95)
    # S_final = sqrt(0.81 * 0.95) ~ 0.87
    cls_pred[0, 0, 60, 60] = 1.5
    iou_pred[0, 0, 60, 60] = 3.0

    pred_with_iou = {
        "cls": cls_pred, "offset": offset_pred, "size": size_pred, "yaw": yaw_pred, "iou": iou_pred
    }

    # At threshold 0.3: Candidate A (0.21) should be filtered out, Candidate B (0.87) should survive
    dets = filter_pred(pred_with_iou, config, out_size_factor=out_size_factor, thres=0.3, nms_thres=0.5)
    assert len(dets) == 1
    # Candidate B is at grid (60, 60): metric coordinates center_x = 60 * 0.4 + 0 = 24.0, center_y = 60 * 0.4 - 40 = -16.0
    assert abs(dets[0, 0] - 24.0) < 1.0
    assert abs(dets[0, 1] - (-16.0)) < 1.0

def test_filter_pred_fallback_when_iou_absent():
    """Verify that filter_pred works identically to baseline when 'iou' head is absent."""
    config = {
        "geometry": {
            "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
            "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        },
        "nms_alpha": 0.5,
    }
    H, W = 200, 176
    cls_pred = torch.full((1, 1, H, W), -10.0)
    cls_pred[0, 0, 50, 50] = 3.0 # p~0.95
    pred_no_iou = {
        "cls": cls_pred,
        "offset": torch.zeros(1, 2, H, W),
        "size": torch.zeros(1, 2, H, W),
        "yaw": torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(1, 2, H, W),
    }
    dets = filter_pred(pred_no_iou, config, out_size_factor=4, thres=0.3, nms_thres=0.5)
    assert len(dets) == 1
```

- [x] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_iqa_header.py
```
Expected output: Fails because Candidate A is selected instead of filtered out.

- [x] **Step 3: Update `filter_pred` in `detector/postprocess.py`**

In `detector/postprocess.py`:
- Compute `cls_probs` as usual.
- Check if `"iou"` in `pred`:
  ```python
  has_iou = "iou" in pred
  alpha = float(config.get("nms_alpha", 0.5)) if has_iou else 0.0
  if has_iou and alpha > 0.0:
      iou_logit = pred["iou"].squeeze(0).squeeze(0).detach()
      iou_score = torch.sigmoid(iou_logit)
      ranking_scores = (cls_probs ** (1.0 - alpha)) * (iou_score ** alpha)
  else:
      ranking_scores = cls_probs
  ```
- Use `ranking_scores` for max-pooling:
  ```python
  pooled = F.max_pool2d(ranking_scores.unsqueeze(0).unsqueeze(0), 3, 1, 1)[0, 0]
  candidate_mask = torch.logical_and(ranking_scores == pooled, ranking_scores > thres)
  ```
- Use `candidate_scores = ranking_scores[candidate_mask]` for both GPU rotated NMS and CPU polygon NMS.

- [x] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_iqa_header.py
```
Expected output: 10 passed in <3s.

- [x] **Step 5: Commit changes**

```bash
git add detector/postprocess.py tests/test_iqa_header.py
git commit -m "feat(postprocess): implement calibrated joint quality NMS scoring in filter_pred"
```

---

### Task 5: Establish Experiment Configuration & Full Regression Suite

**Files:**
- Create: `configs/kitti/iou_aware_header/kitti_mobilepixornext_litemla_oga_reparam_iqa.json`
- Test: Full regression suite (`tests/`)

**Interfaces:**
- Produces: `configs/kitti/iou_aware_header/kitti_mobilepixornext_litemla_oga_reparam_iqa.json`

- [x] **Step 1: Write integration test for full training pipeline with IQA config**

Append to `tests/test_iqa_header.py`:
```python
import json
from core.models.model import CustomModel
from core.losses.strategies.oga import OgaLossStrategy

def test_full_iqa_pipeline_integration():
    """Verify that CustomModel and OgaLossStrategy initialize and forward cleanly using config."""
    config_path = "configs/kitti/iou_aware_header/kitti_mobilepixornext_litemla_oga_reparam_iqa.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = CustomModel(cfg, num_classes=3, input_channels=35)
    loss_strategy = OgaLossStrategy(cls_encoding=cfg.get("cls_encoding", "gaussian"), config=cfg)

    B = 2
    x = torch.randn(B, 35, 200, 176)
    pred = model(x)
    assert "iou" in pred
    assert pred["iou"].shape == (B, 1, 200, 176)

    target = {
        "cls": torch.rand(B, 3, 200, 176),
        "offset": torch.randn(B, 2, 200, 176),
        "size": torch.randn(B, 2, 200, 176),
        "yaw": torch.randn(B, 2, 200, 176),
        "reg_mask": torch.randint(0, 2, (B, 200, 176), dtype=torch.bool),
    }

    out = loss_strategy(pred, target)
    assert "loss" in out
    assert "iou" in out
    assert "weight_iou" in out
    out["loss"].backward()
```

- [x] **Step 2: Create config file `configs/kitti/iou_aware_header/kitti_mobilepixornext_litemla_oga_reparam_iqa.json`**

Create `configs/kitti/iou_aware_header/kitti_mobilepixornext_litemla_oga_reparam_iqa.json`:
```json
{
  "backbone": "mobilepixornext_reparam",
  "backbone_out_dim": 16,
  "cls_encoding": "gaussian",
  "use_reparam": true,
  "header_use_bn": true,
  "header_act": "silu",
  "header_use_iou": true,
  "use_iou": true,
  "iou_target_type": "mgiou",
  "iou_loss_weight": 1.0,
  "nms_alpha": 0.5,
  "loss_strategy": "oga",
  "temperature": 2.0,
  "clamp_bound": 3.0,
  "ema_momentum": 0.99,
  "corner_beta": 1.0,
  "max_abs_log_size": 10.0,
  "epsilon": 1e-06,
  "geometry": {
    "x_min": 0.0,
    "x_max": 70.4,
    "x_res": 0.1,
    "y_min": -40.0,
    "y_max": 40.0,
    "y_res": 0.1,
    "z_min": -2.0,
    "z_max": 1.5,
    "z_res": 0.1
  }
}
```

- [x] **Step 3: Run full test suite including regression**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint
```
Expected output: All 98+ tests pass cleanly (87 existing + 11 new IQA tests).

- [x] **Step 4: Commit changes and mark plan complete**

```bash
git add configs/kitti/iou_aware_header/kitti_mobilepixornext_litemla_oga_reparam_iqa.json tests/test_iqa_header.py docs/superpowers/plans/2026-09-28-iou-aware-quality-header.md
git commit -m "feat(config): establish MobilePixorNeXt M4 IQA-Header config and end-to-end integration tests"
```
