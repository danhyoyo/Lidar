# OGA-Loss (Oriented Geometric Alignment & Adaptive Loss) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng và tích hợp hàm loss hoàn toàn mới OGA-Loss (Oriented Geometric Alignment & Adaptive Loss) kết hợp giám sát hình học xoay chính xác ($\pi$-symmetric normalized corner distance + multi-axis projection GIoU), giám sát toạ độ độc lập 2 luồng (Dual-Stream), và cơ chế cân bằng đa nhiệm bảo toàn gradient (Temperature-Softmax Bounded Uncertainty Weighting) nhằm tối ưu hoá vượt trội độ chính xác phát hiện vật thể 3D LiDAR BEV.

**Architecture:** 
- Mô-đun hình học xoay `detector/core/losses/oriented_geometry_loss.py`: giải mã 4 góc hộp xoay trong toạ độ BEV, tính khoảng cách góc chuẩn hoá kích thước chu kỳ $\pi$ ($L_{\text{NCD}}$) và khoảng cách chiếu đa trục ($L_{\text{proj}}$), không dùng nghịch đảo ma trận.
- Mô-đun cân bằng đa nhiệm `detector/core/losses/uncertainty_weighting.py`: triển khai `TemperatureSoftmaxUncertainty` bảo toàn tổng ngân sách gradient $\sum w_i = M$, chống suy biến số học trong BF16.
- Bộ điều phối `detector/core/losses/loss_fn.py`: tích hợp chế độ `name="oga"` điều phối focal loss, smooth L1 toạ độ và hình học xoay OGA, tương thích 100% với pipeline huấn luyện KITTI và checkpoint state dict.

**Tech Stack:** Python 3, PyTorch (CUDA/BF16/FP32), NumPy, `unittest`. Không thêm dependency ngoài.

**Spec:** `docs/superpowers/specs/2026-09-26-oriented-geometric-adaptive-loss.md`

## Global Constraints

- Không sao chép mã nguồn của hàm loss UWAG lịch sử; kế thừa ý tưởng cân bằng tác vụ và khắc phục triệt để các hạn chế lý thuyết của UWAG.
- Toàn bộ tính toán hình học xoay và softmax chạy ổn định trên cả FP32 và mixed-precision BF16 autocast.
- Giữ nguyên toàn bộ hợp đồng suy luận (inference contract): `cls`, `offset`, `size`, `yaw` không thay đổi kiến trúc head, decoder, ONNX hay TensorRT.
- Bắt buộc kiểm thử với Python môi trường `AI_env` (`/home/duyennh/miniconda3/envs/AI_env/bin/python`).
- Các lệnh Git được thực thi qua `rtk git` theo quy chuẩn RTK token-saving.

---

### Task 1: Xây dựng giải mã hình học góc xoay và Khoảng cách Đỉnh Chu Kỳ $\pi$ ($L_{\text{NCD}}$)

**Files:**
- Create: `detector/core/losses/oriented_geometry_loss.py`
- Create: `tests/test_oriented_geometry_loss.py`

**Interfaces:**
- Consumes:
  - `pred_offset`, `target_offset`: `torch.Tensor` kích thước `[N, 2]` đại diện $(dx, dy)$
  - `pred_size`, `target_size`: `torch.Tensor` kích thước `[N, 2]` đại diện $(\log w, \log l)$
  - `pred_yaw`, `target_yaw`: `torch.Tensor` kích thước `[N, 2]` đại diện $(\cos 2\theta, \sin 2\theta)$
- Produces:
  - `box_corners(offset, log_size, doubled_yaw, epsilon=1e-6, max_abs_log_size=10.0)` -> `(corners, axes, clamp_count, fallback_count)` với `corners` hình dạng `[N, 4, 2]`, `axes` hình dạng `[N, 2, 2]`
  - `pi_symmetric_corner_distance(pred_corners, target_corners, target_size, epsilon=1e-6)` -> `torch.Tensor` scalar loss chuẩn hoá theo đường chéo

- [ ] **Step 1: Viết test thất bại kiểm tra hình học hộp và $L_{\text{NCD}}$**

Tạo `tests/test_oriented_geometry_loss.py` kiểm tra:
1. Hai hộp giống hệt nhau sinh ra khoảng cách $L_{\text{NCD}} = 0$.
2. Tính bất biến góc quay $\pi$: góc $\theta$ và $\theta + \pi$ sinh ra các đỉnh và khoảng cách giống hệt nhau.
3. Chuẩn hoá kích thước: lỗi góc 0.5m trên hộp nhỏ bị phạt nặng hơn hộp lớn theo tỉ lệ đường chéo.
4. Gradient liên tục và khác 0 khi 2 hộp không giao nhau ở cự ly $10\text{m}$.

```python
# tests/test_oriented_geometry_loss.py
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.oriented_geometry_loss import (
    box_corners,
    pi_symmetric_corner_distance,
)


class TestOrientedGeometryCornerLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_identical_boxes_give_zero_distance(self):
        offset = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(1.6), math.log(4.0)]], dtype=torch.float32)
        yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        corners, axes, _, _ = box_corners(offset, log_size, yaw)
        loss = pi_symmetric_corner_distance(corners, corners, log_size)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_pi_rotation_invariance(self):
        offset = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32)
        # theta = pi/6 vs theta = pi/6 + pi
        theta1 = math.pi / 6
        theta2 = theta1 + math.pi
        yaw1 = torch.tensor([[math.cos(2 * theta1), math.sin(2 * theta1)]], dtype=torch.float32)
        yaw2 = torch.tensor([[math.cos(2 * theta2), math.sin(2 * theta2)]], dtype=torch.float32)

        corners1, _, _, _ = box_corners(offset, log_size, yaw1)
        corners2, _, _, _ = box_corners(offset, log_size, yaw2)

        tgt_corners, _, _, _ = box_corners(offset + 1.0, log_size, yaw1)
        loss1 = pi_symmetric_corner_distance(corners1, tgt_corners, log_size)
        loss2 = pi_symmetric_corner_distance(corners2, tgt_corners, log_size)
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=5)

    def test_non_overlapping_box_provides_finite_nonzero_gradient(self):
        pred_offset = torch.tensor([[0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        pred_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32, requires_grad=True)
        pred_yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32, requires_grad=True)

        tgt_offset = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
        tgt_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32)
        tgt_yaw = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        pred_c, _, _, _ = box_corners(pred_offset, pred_size, pred_yaw)
        tgt_c, _, _, _ = box_corners(tgt_offset, tgt_size, tgt_yaw)

        loss = pi_symmetric_corner_distance(pred_c, tgt_c, tgt_size)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(pred_offset.grad).all())
        self.assertFalse((pred_offset.grad == 0).all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Chạy test để xác nhận test THẤT BẠI**

Chạy:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python -m unittest tests/test_oriented_geometry_loss.py
```
Kỳ vọng: Lỗi `ModuleNotFoundError: No module named 'core.losses.oriented_geometry_loss'`.

- [ ] **Step 3: Triển khai mã nguồn tối thiểu cho `box_corners` và `pi_symmetric_corner_distance`**

Tạo `detector/core/losses/oriented_geometry_loss.py`:
```python
"""Oriented BEV box geometry and Scale-Normalized Pi-Symmetric Corner Distance."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["box_corners", "pi_symmetric_corner_distance"]


def box_corners(offset, log_size, doubled_yaw, epsilon=1e-6, max_abs_log_size=10.0):
    """Compute 4 corners and local projection axes of rotated BEV bounding boxes.
    
    Inputs:
        offset: [N, 2] center (x, y) in metric coordinates.
        log_size: [N, 2] log-dimensions (log_width, log_length).
        doubled_yaw: [N, 2] doubled-angle vector (cos(2 theta), sin(2 theta)).
    Outputs:
        corners: [N, 4, 2] metric BEV corners in order [++, +-, -+, --].
        axes: [N, 2, 2] local projection axes (length_axis, width_axis).
        clamp_count: int scalar of clamped log_size elements.
        fallback_count: int scalar of zero-norm yaw fallback events.
    """
    log_size = log_size.float()
    clamped_log_size = log_size.clamp(-max_abs_log_size, max_abs_log_size)
    clamp_count = (clamped_log_size != log_size).sum()

    width = torch.exp(clamped_log_size[:, 0])
    length = torch.exp(clamped_log_size[:, 1])

    yaw = doubled_yaw.float()
    norm = torch.linalg.vector_norm(yaw, dim=-1, keepdim=True)
    valid = norm >= epsilon
    fallback_count = (~valid).sum()

    unit_yaw = yaw / norm.clamp_min(epsilon)
    cos2 = torch.where(valid, unit_yaw[..., :1], torch.ones_like(unit_yaw[..., :1]))
    sin2 = torch.where(valid, unit_yaw[..., 1:], torch.zeros_like(unit_yaw[..., 1:]))
    theta = 0.5 * torch.atan2(sin2, cos2).squeeze(-1)

    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    length_axis = torch.stack((cos_t, sin_t), dim=-1)   # [N, 2]
    width_axis = torch.stack((-sin_t, cos_t), dim=-1)   # [N, 2]
    axes = torch.stack((length_axis, width_axis), dim=1) # [N, 2, 2]

    half_l = 0.5 * length.unsqueeze(-1) * length_axis   # [N, 2]
    half_w = 0.5 * width.unsqueeze(-1) * width_axis     # [N, 2]
    center = offset.float()

    # 4 corners: [center + l/2 + w/2, center + l/2 - w/2, center - l/2 + w/2, center - l/2 - w/2]
    corners = torch.stack(
        (
            center + half_l + half_w,
            center + half_l - half_w,
            center - half_l + half_w,
            center - half_l - half_w,
        ),
        dim=1,
    )
    return corners, axes, clamp_count, fallback_count


def pi_symmetric_corner_distance(pred_corners, target_corners, target_log_size, epsilon=1e-6):
    """Compute scale-normalized pi-symmetric corner distance between oriented boxes.
    
    Inputs:
        pred_corners: [N, 4, 2] predicted corners.
        target_corners: [N, 4, 2] target corners.
        target_log_size: [N, 2] target log(width, length) for scale normalization.
    """
    if pred_corners.shape[0] == 0:
        return pred_corners.sum() * 0.0

    # Two valid cyclic permutations under pi-rotation: direct (0,1,2,3) and rotated-pi (3,2,1,0) or (2,3,0,1)
    # With indexing [++, +-, -+, --]:
    # rotating by 180 degrees maps ++ to -- (idx 0 to 3) and +- to -+ (idx 1 to 2)
    # direct: [0, 1, 2, 3], rotated: [3, 2, 1, 0]
    dist_direct = torch.norm(pred_corners - target_corners, p=1, dim=-1).mean(dim=-1) # [N]
    
    target_corners_pi = target_corners[:, [3, 2, 1, 0], :]
    dist_pi = torch.norm(pred_corners - target_corners_pi, p=1, dim=-1).mean(dim=-1)   # [N]
    
    raw_distance = torch.minimum(dist_direct, dist_pi) # [N]

    # Normalize by the target bounding box diagonal to equalize scale sensitivity across classes
    target_w = torch.exp(target_log_size[:, 0].float())
    target_l = torch.exp(target_log_size[:, 1].float())
    diagonal = torch.sqrt(target_w.square() + target_l.square()).clamp_min(epsilon)

    normalized_distance = raw_distance / diagonal
    return normalized_distance.mean()
```

- [ ] **Step 4: Chạy test để xác nhận test THÀNH CÔNG**

Chạy:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python -m unittest tests/test_oriented_geometry_loss.py
```
Kỳ vọng: `Ran 3 tests in ... OK`.

- [ ] **Step 5: Commit qua rtk git**

```bash
rtk git add detector/core/losses/oriented_geometry_loss.py tests/test_oriented_geometry_loss.py
rtk git commit -m "feat(loss): add rotated box corners and pi-symmetric corner distance"
```

---

### Task 2: Triển khai Chiếu Đa Trục Pháp Tuyến ($L_{\text{proj}}$) và Mô-đun Hoàn Chỉnh `OrientedGeometryLoss`

**Files:**
- Modify: `detector/core/losses/oriented_geometry_loss.py`
- Modify: `tests/test_oriented_geometry_loss.py`

**Interfaces:**
- Consumes:
  - `pred_offset`, `pred_size`, `pred_yaw`: Tensor `[B, 2, H, W]`
  - `target_offset`, `target_size`, `target_yaw`: Tensor `[B, 2, H, W]`
  - `reg_mask`: Tensor `[B, H, W]`
- Produces:
  - `multiaxis_projection_giou(pred_corners, pred_axes, target_corners, target_axes, epsilon=1e-6)` -> `torch.Tensor` scalar
  - `OrientedGeometryLoss(nn.Module)`:
    - `__init__(beta=1.0, epsilon=1e-6, max_abs_log_size=10.0)`
    - `forward(pred_offset, pred_size, pred_yaw, target_offset, target_size, target_yaw, reg_mask)` -> `(total_loss, metrics_dict)`

- [ ] **Step 1: Viết test cho `multiaxis_projection_giou` và `OrientedGeometryLoss`**

Thêm các test case vào `tests/test_oriented_geometry_loss.py`:
1. Hộp trùng khớp cho $L_{\text{proj}} = 0$ và $L_{\text{total}} = 0$.
2. Mask rỗng (`reg_mask` toàn 0) trả về scalar 0 gắn kết autograd đồ thị cho cả 3 đầu vào.
3. Hỗ trợ mixed-precision autocast an toàn.

```python
    def test_multiaxis_projection_giou_identical(self):
        offset = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32)
        yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        corners, axes, _, _ = box_corners(offset, log_size, yaw)

        from core.losses.oriented_geometry_loss import multiaxis_projection_giou
        loss = multiaxis_projection_giou(corners, axes, corners, axes)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_oriented_geometry_loss_empty_mask(self):
        from core.losses.oriented_geometry_loss import OrientedGeometryLoss
        module = OrientedGeometryLoss(beta=1.0)
        B, H, W = 2, 8, 8
        pred_offset = torch.zeros((B, 2, H, W), requires_grad=True)
        pred_size = torch.zeros((B, 2, H, W), requires_grad=True)
        pred_yaw = torch.zeros((B, 2, H, W), requires_grad=True)
        tgt_offset = torch.zeros((B, 2, H, W))
        tgt_size = torch.zeros((B, 2, H, W))
        tgt_yaw = torch.zeros((B, 2, H, W))
        reg_mask = torch.zeros((B, H, W))

        loss, metrics = module(pred_offset, pred_size, pred_yaw, tgt_offset, tgt_size, tgt_yaw, reg_mask)
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(pred_offset.grad is not None)
        self.assertEqual(pred_offset.grad.sum().item(), 0.0)
```

- [ ] **Step 2: Chạy test để xác nhận test THẤT BẠI**

Chạy:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python -m unittest tests/test_oriented_geometry_loss.py
```
Kỳ vọng: Lỗi `ImportError: cannot import name 'multiaxis_projection_giou'`.

- [ ] **Step 3: Cập nhật `oriented_geometry_loss.py` với `multiaxis_projection_giou` và `OrientedGeometryLoss`**

Bổ sung vào `detector/core/losses/oriented_geometry_loss.py`:
```python
def multiaxis_projection_giou(pred_corners, pred_axes, target_corners, target_axes, epsilon=1e-6):
    """Compute 1D Generalized IoU projected over 4 rectangle normal axes.
    
    Paper: AAAI 2026 MGIoU (Specialized 2D BEV projection).
    """
    if pred_corners.shape[0] == 0:
        return pred_corners.sum() * 0.0

    # 4 axes: 2 from prediction, 2 from target
    axes = torch.cat((pred_axes, target_axes), dim=1) # [N, 4, 2]
    
    # Project 4 corners of each box onto the 4 projection axes
    pred_proj = torch.einsum("ncd,nad->nac", pred_corners, axes)     # [N, 4, 4]
    target_proj = torch.einsum("ncd,nad->nac", target_corners, axes) # [N, 4, 4]

    pred_min, pred_max = pred_proj.amin(dim=-1), pred_proj.amax(dim=-1)       # [N, 4]
    target_min, target_max = target_proj.amin(dim=-1), target_proj.amax(dim=-1) # [N, 4]

    intersection = (
        torch.minimum(pred_max, target_max) - torch.maximum(pred_min, target_min)
    ).clamp_min(0.0)
    union = pred_max - pred_min + target_max - target_min - intersection
    hull = torch.maximum(pred_max, target_max) - torch.minimum(pred_min, target_min)

    giou = intersection / union.clamp_min(epsilon) - (hull - union) / hull.clamp_min(epsilon)
    loss_1d = (1.0 - giou.mean(dim=-1)) / 2.0
    return loss_1d.mean()


class OrientedGeometryLoss(nn.Module):
    """Composite Rotated BEV Geometry Loss: Multi-Axis Projection GIoU + Normalized Corner Distance."""

    def __init__(self, beta=1.0, epsilon=1e-6, max_abs_log_size=10.0):
        super().__init__()
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.max_abs_log_size = float(max_abs_log_size)

    def forward(
        self,
        pred_offset,
        pred_size,
        pred_yaw,
        target_offset,
        target_size,
        target_yaw,
        reg_mask,
    ):
        positive = reg_mask.reshape(-1).bool()
        if not positive.any():
            zero = (pred_offset.sum() + pred_size.sum() + pred_yaw.sum()) * 0.0
            return zero, {"proj_giou": 0.0, "corner_dist": 0.0}

        def _select(x):
            return x.permute(0, 2, 3, 1).reshape(-1, 2)[positive].float()

        pred_off_pos = _select(pred_offset)
        pred_size_pos = _select(pred_size)
        pred_yaw_pos = _select(pred_yaw)

        tgt_off_pos = _select(target_offset)
        tgt_size_pos = _select(target_size)
        tgt_yaw_pos = _select(target_yaw)

        with torch.autocast(device_type=pred_offset.device.type, enabled=False):
            pred_c, pred_ax, _, _ = box_corners(
                pred_off_pos, pred_size_pos, pred_yaw_pos, self.epsilon, self.max_abs_log_size
            )
            tgt_c, tgt_ax, _, _ = box_corners(
                tgt_off_pos, tgt_size_pos, tgt_yaw_pos, self.epsilon, self.max_abs_log_size
            )

            proj_loss = multiaxis_projection_giou(pred_c, pred_ax, tgt_c, tgt_ax, self.epsilon)
            corner_loss = pi_symmetric_corner_distance(pred_c, tgt_c, tgt_size_pos, self.epsilon)
            total_geo = proj_loss + self.beta * corner_loss

        return total_geo, {
            "proj_giou": proj_loss.detach().item(),
            "corner_dist": corner_loss.detach().item(),
        }
```

- [ ] **Step 4: Chạy lại test suite để kiểm tra tất cả test PASS**

Chạy:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python -m unittest tests/test_oriented_geometry_loss.py
```
Kỳ vọng: Tất cả 5 tests PASS.

- [ ] **Step 5: Commit qua rtk git**

```bash
rtk git add detector/core/losses/oriented_geometry_loss.py tests/test_oriented_geometry_loss.py
rtk git commit -m "feat(loss): implement multiaxis projection giou and OrientedGeometryLoss"
```

---

### Task 3: Triển khai Cân Bằng Đa Nhiệm Bất Định Chuẩn Hoá Softmax Nhiệt Độ (`TemperatureSoftmaxUncertainty`)

**Files:**
- Create: `detector/core/losses/uncertainty_weighting.py`
- Create: `tests/test_uncertainty_weighting.py`

**Interfaces:**
- Consumes:
  - `num_tasks`: int (số tác vụ $M$, mặc định 5: cls, offset, size, yaw, geo)
  - `temperature`: float $\tau$ (mặc định 2.0)
  - `clamp_bound`: float $c$ (mặc định 3.0)
- Produces:
  - `TemperatureSoftmaxUncertainty(nn.Module)`:
    - Parameter `log_scales`: `nn.Parameter` kích thước `[M]`, khởi tạo bằng 0
    - `forward(losses_dict)` -> `(weighted_total_loss, weights_dict)`

- [ ] **Step 1: Viết test cho `TemperatureSoftmaxUncertainty`**

Tạo `tests/test_uncertainty_weighting.py`:
```python
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.uncertainty_weighting import TemperatureSoftmaxUncertainty


class TestTemperatureSoftmaxUncertainty(unittest.TestCase):
    def test_initial_weights_are_strictly_equal_to_one(self):
        weighting = TemperatureSoftmaxUncertainty(num_tasks=5, temperature=2.0)
        task_losses = {
            "cls": torch.tensor(1.0),
            "offset": torch.tensor(0.5),
            "size": torch.tensor(0.8),
            "yaw": torch.tensor(0.2),
            "geo": torch.tensor(0.4),
        }
        total, weights = weighting(task_losses)
        # Sum of nominal weights must be 5.0
        self.assertAlmostEqual(sum(weights.values()), 5.0, places=5)
        for name, w in weights.items():
            self.assertAlmostEqual(w, 1.0, places=5)

    def test_gradient_conservation_and_updates(self):
        weighting = TemperatureSoftmaxUncertainty(num_tasks=3, temperature=1.5, clamp_bound=3.0)
        # Simulate an imbalance where task A is very large
        task_losses = {
            "a": torch.tensor(5.0, requires_grad=True),
            "b": torch.tensor(0.2, requires_grad=True),
            "c": torch.tensor(0.1, requires_grad=True),
        }
        total, weights = weighting(task_losses)
        total.backward()

        self.assertTrue(torch.isfinite(weighting.log_scales.grad).all())
        self.assertAlmostEqual(sum(weights.values()), 3.0, places=5)

    def test_bound_clamping_prevents_extreme_starvation(self):
        weighting = TemperatureSoftmaxUncertainty(num_tasks=2, temperature=1.0, clamp_bound=2.0)
        # Set extreme logits manually
        with torch.no_grad():
            weighting.log_scales.copy_(torch.tensor([-100.0, 100.0]))
        weights = weighting.get_task_weights()
        # Because clamped to [-2.0, 2.0], minimum weight must be > 0.05
        min_weight = min(weights.values())
        self.assertGreater(min_weight, 0.01)
        self.assertAlmostEqual(sum(weights.values()), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Chạy test để xác nhận test THẤT BẠI**

Chạy:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python -m unittest tests/test_uncertainty_weighting.py
```
Kỳ vọng: Lỗi `ModuleNotFoundError: No module named 'core.losses.uncertainty_weighting'`.

- [ ] **Step 3: Triển khai `TemperatureSoftmaxUncertainty`**

Tạo `detector/core/losses/uncertainty_weighting.py`:
```python
"""Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW) for multi-task loss balancing."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TemperatureSoftmaxUncertainty"]


class TemperatureSoftmaxUncertainty(nn.Module):
    """Normalized multi-task weighting with conserved gradient budget and soft bound limits.
    
    Formula:
        s_clamped = clamp(s, -clamp_bound, clamp_bound)
        w_i = M * exp(s_i / tau) / sum(exp(s_j / tau))
        L_total = sum(w_i * L_i)
        
    Guarantees:
        1. sum(w_i) == M (strictly conserved total gradient scale)
        2. w_i > 0 for all tasks (no task starvation)
        3. Completely immune to numerical explosion under BF16 / FP32.
    """

    def __init__(self, task_names=("cls", "offset", "size", "yaw", "geo"), temperature=2.0, clamp_bound=3.0):
        super().__init__()
        self.task_names = list(task_names)
        self.num_tasks = len(self.task_names)
        self.temperature = float(temperature)
        self.clamp_bound = float(clamp_bound)

        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.clamp_bound <= 0.0:
            raise ValueError(f"clamp_bound must be positive, got {self.clamp_bound}")

        # Learnable log-scale logits initialized to 0 (all tasks start with equal weight = 1.0)
        self.log_scales = nn.Parameter(torch.zeros(self.num_tasks, dtype=torch.float32))

    def get_task_weights(self):
        """Compute the normalized task weights as a dict {task_name: float_weight}."""
        clamped_scales = self.log_scales.clamp(-self.clamp_bound, self.clamp_bound)
        normalized = F.softmax(clamped_scales / self.temperature, dim=0) * self.num_tasks
        return {name: float(normalized[i].item()) for i, name in enumerate(self.task_names)}

    def forward(self, task_losses):
        """Compute total weighted loss and return weights telemetry.
        
        Args:
            task_losses: dict mapping task_name -> scalar Tensor loss.
        """
        clamped_scales = self.log_scales.clamp(-self.clamp_bound, self.clamp_bound)
        normalized_weights = F.softmax(clamped_scales / self.temperature, dim=0) * self.num_tasks

        total_loss = torch.zeros((), device=self.log_scales.device)
        weights_dict = {}
        for i, name in enumerate(self.task_names):
            if name in task_losses:
                w = normalized_weights[i]
                total_loss = total_loss + w * task_losses[name]
                weights_dict[name] = float(w.detach().item())

        return total_loss, weights_dict
```

- [ ] **Step 4: Chạy lại test suite để kiểm tra PASS**

Chạy:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python -m unittest tests/test_uncertainty_weighting.py
```
Kỳ vọng: Tất cả 3 tests PASS.

- [ ] **Step 5: Commit qua rtk git**

```bash
rtk git add detector/core/losses/uncertainty_weighting.py tests/test_uncertainty_weighting.py
rtk git commit -m "feat(loss): add TemperatureSoftmaxUncertainty multi-task balancing module"
```

---

### Task 4: Tích hợp chế độ `name="oga"` vào `LossFunction` trong `detector/core/losses/loss_fn.py`

**Files:**
- Modify: `detector/core/losses/loss_fn.py`
- Create: `tests/test_loss_fn_oga.py`

**Interfaces:**
- Consumes:
  - `cls_encoding`: `"gaussian"` hoặc `"binary"`
  - `config`: dict chứa `name: "oga"`, `geometric_weight`, `temperature`, `clamp_bound`, `corner_beta`
- Produces:
  - `LossFunction(cls_encoding, config)`: hỗ trợ `self.name == "oga"`
  - `criterion(outputs, batch)`: trả về dictionary đầy đủ `{loss, cls, offset, size, yaw, geo, corner_dist, proj_giou, weight_cls, ...}`

- [ ] **Step 1: Viết test tích hợp `LossFunction` với cấu hình OGA**

Tạo `tests/test_loss_fn_oga.py`:
```python
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.loss_fn import LossFunction


class TestLossFunctionOGA(unittest.TestCase):
    def setUp(self):
        self.B, self.H, self.W = 2, 16, 16
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

    def test_oga_loss_initialization_and_forward(self):
        config = {
            "name": "oga",
            "temperature": 2.0,
            "clamp_bound": 3.0,
            "corner_beta": 1.0,
            "geometric_weight": 0.5,
        }
        criterion = LossFunction("gaussian", config)
        loss_dict = criterion(self.pred, self.target)

        for key in ("loss", "cls", "offset", "size", "yaw", "geo", "corner_dist", "proj_giou"):
            self.assertIn(key, loss_dict)
            self.assertTrue(torch.isfinite(loss_dict[key]).all() if isinstance(loss_dict[key], torch.Tensor) else math.isfinite(loss_dict[key]))

        for task in ("cls", "offset", "size", "yaw", "geo"):
            self.assertIn(f"weight_{task}", loss_dict)

        # Verify backward pass computes valid gradients for all prediction heads and criterion parameters
        loss_dict["loss"].backward()
        for head in ("cls", "offset", "size", "yaw"):
            self.assertTrue(self.pred[head].grad is not None)
            self.assertTrue(torch.isfinite(self.pred[head].grad).all())

        # Check criterion parameters
        params = list(criterion.parameters())
        self.assertGreater(len(params), 0)
        self.assertTrue(params[0].grad is not None)


if __name__ == "__main__":
    import math
    unittest.main()
```

- [ ] **Step 2: Chạy test để xác nhận test THẤT BẠI**

Chạy:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python -m unittest tests/test_loss_fn_oga.py
```
Kỳ vọng: Lỗi `ValueError: Unsupported loss name: 'oga'`.

- [ ] **Step 3: Sửa đổi `detector/core/losses/loss_fn.py` để tích hợp OGA**

Chỉnh sửa `detector/core/losses/loss_fn.py`:
1. Import `OrientedGeometryLoss` và `TemperatureSoftmaxUncertainty`.
2. Trong `__init__`: cho phép `name in {"baseline", "uwag", "oga"}`.
   Nếu `name == "oga"`:
   - Khởi tạo `self.geometry_loss = OrientedGeometryLoss(beta=config.get("corner_beta", 1.0), epsilon=self.eps, max_abs_log_size=self.max_abs_log_size)`.
   - Khởi tạo `self.weighting = TemperatureSoftmaxUncertainty(temperature=config.get("temperature", 2.0), clamp_bound=config.get("clamp_bound", 3.0))`.
3. Trong `forward`:
   - Tính toán `cls_loss`, `offset_loss = smooth_l1_loss`, `size_loss = smooth_l1_loss`, `yaw_loss = smooth_l1_loss`.
   - Tính toán `geo_loss, geo_metrics = self.geometry_loss(...)`.
   - Tính toán tổng loss qua `self.weighting`.
   - Trả về dictionary đồng bộ.

- [ ] **Step 4: Chạy lại test suite để kiểm tra PASS**

Chạy:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python -m unittest tests/test_loss_fn_oga.py
```
Kỳ vọng: `Ran 1 test in ... OK`.

Chạy thêm kiểm thử hồi quy cho baseline & uwag:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python -m unittest discover -s tests -p "test_*.py"
```
Kỳ vọng: Toàn bộ test suite vượt qua.

- [ ] **Step 5: Commit qua rtk git**

```bash
rtk git add detector/core/losses/loss_fn.py tests/test_loss_fn_oga.py
rtk git commit -m "feat(loss): integrate OGA loss mode into LossFunction"
```

---

### Task 5: Tạo cấu hình huấn luyện chuẩn & Kiểm thử tích hợp End-to-End với Pipeline

**Files:**
- Create: `configs/kitti/backbone_branch/kitti_bevnext_litemla_oga.json`
- Create: `tests/test_training_pipeline_oga.py`

**Interfaces:**
- Consumes: Config JSON đầy đủ kết hợp kiến trúc BEVNeXt + LiteMLA + sgFPN và hàm loss OGA.
- Produces: Test pipeline xác nhận `model -> loss -> backward -> optimizer` khớp 100% logic của `train.py`.

- [ ] **Step 1: Tạo cấu hình `configs/kitti/backbone_branch/kitti_bevnext_litemla_oga.json`**

Tạo snapshot cấu hình hoàn chỉnh kế thừa kiến trúc backbone BEVNeXt tốt nhất:
```json
{
  "augmentation": {
    "p": 0.5,
    "rotation": {
      "limit_angle": 20,
      "p": 1,
      "use": true
    },
    "scaling": {
      "p": 1,
      "range": [0.95, 1.05],
      "use": true
    },
    "translation": {
      "p": 1,
      "scale": 0.4,
      "use": true
    }
  },
  "data": {
    "bev_encoding": {
      "density_norm": 32,
      "intensity_scale": 1,
      "name": "rich8"
    },
    "gaussian_overlap": 0.1,
    "kitti": {
      "geometry": {
        "x_max": 70.4,
        "x_min": 0,
        "x_res": 0.1,
        "y_max": 40,
        "y_min": -40,
        "y_res": 0.1,
        "z_max": 1,
        "z_min": -2.5,
        "z_res": 0.1
      },
      "location": "data/kitti/processed",
      "objects": {
        "Car": 0,
        "Cyclist": 2,
        "Pedestrian": 1
      }
    },
    "min_radius": 4,
    "num_classes": 3,
    "out_size_factor": 4
  },
  "date": "reproducible",
  "device": "cuda",
  "loss": {
    "name": "oga",
    "temperature": 2.0,
    "clamp_bound": 3.0,
    "corner_beta": 1.0,
    "epsilon": 0.0001,
    "max_abs_log_size": 10.0
  },
  "model": {
    "backbone": "bevnext",
    "backbone_out_dim": 16,
    "c4_attention": "litemla",
    "cls_encoding": "gaussian",
    "expansion": 2.5,
    "header_act": "silu",
    "header_use_bn": true,
    "scale_gated_fpn": true
  },
  "multi_gpu": false,
  "note": "kitti_bevnext_litemla_rich8_sgfpn_oga_loss",
  "seed": 42,
  "train": {
    "accumulation_steps": 8,
    "data": "splits/kitti/train.txt",
    "epochs": 100,
    "learning_rate": 0.005,
    "lr_decay_at": [65, 85],
    "momentum": 0.9,
    "physical_batch_size": 16,
    "precision": "bf16",
    "save_every": 5,
    "weight_decay": 0.0005
  },
  "val": {
    "data": "splits/kitti/val.txt",
    "physical_batch_size": 2,
    "val_every": 1
  },
  "ver": 1
}
```

- [ ] **Step 2: Viết test End-to-End mô phỏng bước huấn luyện của `train.py`**

Tạo `tests/test_training_pipeline_oga.py`:
```python
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.models.model import build_model
from core.losses.loss_fn import LossFunction


class TestTrainingPipelineOGA(unittest.TestCase):
    def test_pipeline_forward_backward_with_oga(self):
        config_path = REPO_ROOT / "configs/kitti/backbone_branch/kitti_bevnext_litemla_oga.json"
        with open(config_path) as f:
            config = json.load(f)

        device = torch.device("cpu")
        model = build_model(config).to(device)
        criterion = LossFunction(config["model"]["cls_encoding"], config["loss"]).to(device)

        # Mock a voxel batch matching input dimension [B, 8, 704, 800]
        # Use small spatial slice [B, 8, 64, 64] for fast unit testing
        B = 2
        voxel = torch.randn(B, 8, 64, 64, device=device)
        outputs = model(voxel)

        # Target dimensions match outputs
        H_out, W_out = outputs["cls"].shape[2], outputs["cls"].shape[3]
        batch = {
            "cls": torch.zeros(B, 3, H_out, W_out, device=device),
            "offset": torch.zeros(B, 2, H_out, W_out, device=device),
            "size": torch.zeros(B, 2, H_out, W_out, device=device),
            "yaw": torch.tensor([1.0, 0.0], device=device).view(1, 2, 1, 1).expand(B, 2, H_out, W_out),
            "reg_mask": torch.ones(B, H_out, W_out, device=device),
        }

        # Forward pass
        loss_dict = criterion(outputs, batch)
        loss = loss_dict["loss"]
        self.assertTrue(torch.isfinite(loss))

        # Backward pass
        loss.backward()

        # Check model and criterion gradients
        has_model_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        has_crit_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in criterion.parameters())
        self.assertTrue(has_model_grad)
        self.assertTrue(has_crit_grad)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Chạy test End-to-End**

Chạy:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python -m unittest tests/test_training_pipeline_oga.py
```
Kỳ vọng: Test PASS, mô phỏng hoàn chỉnh cả forward và backward pass với BEVNeXt và OGA loss.

- [ ] **Step 4: Chạy toàn bộ test suite hoàn chỉnh của repository**

Chạy:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python -m unittest discover -s tests -p "test_*.py"
```
Kỳ vọng: Tất cả tests đều PASS, không có regression nào.

- [ ] **Step 5: Commit qua rtk git**

```bash
rtk git add configs/kitti/backbone_branch/kitti_bevnext_litemla_oga.json tests/test_training_pipeline_oga.py
rtk git commit -m "feat(config): add BEVNeXt LiteMLA OGA loss config and e2e integration test"
```
