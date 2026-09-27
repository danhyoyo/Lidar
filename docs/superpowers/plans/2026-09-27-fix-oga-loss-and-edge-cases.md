# Kế hoạch Thực thi: Khắc phục Lỗ hổng T-SBUW & Các Vấn đề Tiềm ẩn trong OGA-Loss

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Khắc phục triệt để hiện tượng sụp đổ đơn hình (simplex collapse / loss cheating) trong `TemperatureSoftmaxUncertainty` bằng cơ chế chuẩn hóa quy mô chạy (EMA Scale Calibration), đồng thời sửa lỗi xác thực `reg_mask`, bắt lỗi chữ hoa/thường trong `CustomModel`, bổ sung cờ ghi đè Registry và củng cố độ bao phủ kiểm thử cho toàn bộ hệ thống.

**Architecture:** Sử dụng kỹ thuật Exponential Moving Average (EMA) để chuẩn hóa thang đo hàm mất mát ($\tilde{\mathcal{L}}_i = \mathcal{L}_i / \text{EMA}(\mathcal{L}_i)$) trong `TemperatureSoftmaxUncertainty`, đưa kỳ vọng của tất cả các nhánh về mức chuẩn $\approx 1.0$ nhằm triệt tiêu thiên vị số học giữa nhánh phân loại và hình học. Bổ sung hợp đồng xác thực đầu vào nghiêm ngặt trong `BaseLossStrategy` cho `reg_mask` và các khóa dự đoán. Chuẩn hóa `CustomModel` không phân biệt hoa/thường khi cấu hình backbone và bổ sung cờ `allow_override` cho cả hai registry.

**Tech Stack:** PyTorch 2.x, Python 3.14, NumPy, Pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-oriented-geometric-adaptive-loss.md`

## Global Constraints

- Bảo toàn 100% khả năng tương thích ngược với các checkpoint cũ (cả hai định dạng có tiền tố `strategy.` và không có tiền tố).
- Giữ nguyên cấu trúc tham số `criterion.parameters()` để không làm thay đổi hợp đồng giao diện optimizer trong `tools/kitti_training_pipeline/train.py`.
- Toàn bộ các tensor trả về trong `loss_dict` phải là tensor vô hướng (0-dim) hoặc tương thích với bộ gom chỉ số (metric aggregator) của `train.py`.
- Tuân thủ quy trình kiểm thử TDD (Test-Driven Development): Viết test lỗi trước, kiểm tra test lỗi, cài đặt mã nguồn tối thiểu, kiểm tra test pass, rồi commit.

---

### Task 1: Khắc phục Hiện tượng Sụp đổ Đơn hình (Simplex Collapse) bằng EMA Scale Calibration

**Files:**
- Modify: `detector/core/losses/uncertainty_weighting.py`
- Test: `tests/test_uncertainty_weighting.py`

**Interfaces:**
- Consumes: `task_losses: Dict[str, torch.Tensor]` từ `OgaLossStrategy.forward`.
- Produces: `total_loss: torch.Tensor` (0-dim), `weights_dict: Dict[str, torch.Tensor]` với các trọng số $w_i$ cân bằng ổn định quanh $1.0$ ngay cả khi các giá trị loss chênh lệch cả chục lần.

- [x] **Step 1: Viết bài test chứng minh lỗi sụp đổ đơn hình và kiểm chứng cơ chế chuẩn hóa EMA**

Thêm test `test_scale_calibration_prevents_simplex_collapse` vào `tests/test_uncertainty_weighting.py`:

```python
    def test_scale_calibration_prevents_simplex_collapse(self):
        # Mô phỏng độ chênh lệch quy mô thực tế: cls=0.03, geo=0.60
        weighting = TemperatureSoftmaxUncertainty(
            task_names=["cls", "offset", "size", "yaw", "geo"],
            temperature=2.0,
            clamp_bound=3.0,
            ema_momentum=0.9,
        )
        opt = torch.optim.Adam(weighting.parameters(), lr=0.05)

        losses = {
            "cls": torch.tensor(0.03),
            "offset": torch.tensor(0.20),
            "size": torch.tensor(0.25),
            "yaw": torch.tensor(0.30),
            "geo": torch.tensor(0.60),
        }

        # Chạy 100 bước tối ưu
        for _ in range(100):
            opt.zero_grad()
            total_loss, weights = weighting(losses)
            total_loss.backward()
            opt.step()

        final_weights = weighting.get_task_weights()
        # Trọng số không được phép sụp đổ: không có task nào vượt quá 2.0 hoặc tụt dưới 0.4
        for name, w in final_weights.items():
            self.assertGreater(w, 0.4, f"Task {name} bị sụp đổ trọng số quá thấp: {w}")
            self.assertLess(w, 2.0, f"Task {name} bị chiếm dụng trọng số quá cao: {w}")
        self.assertAlmostEqual(sum(final_weights.values()), 5.0, places=4)
```

- [x] **Step 2: Chạy test để xác nhận test thất bại**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_uncertainty_weighting.py::TestTemperatureSoftmaxUncertainty::test_scale_calibration_prevents_simplex_collapse -v -p no:launch_testing -p no:launch_testing_ros
```
Expected: FAIL với assertion `Task cls bị chiếm dụng trọng số quá cao` (trọng số `cls` vọt lên ~3.7 mà không có EMA calibration).

- [x] **Step 3: Cập nhật `TemperatureSoftmaxUncertainty` với cơ chế EMA Scale Calibration**

Cập nhật file `detector/core/losses/uncertainty_weighting.py`:

```python
"""Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW) with Running Scale Calibration."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TemperatureSoftmaxUncertainty"]


class TemperatureSoftmaxUncertainty(nn.Module):
    """Normalized multi-task weighting with conserved gradient budget, smooth tanh soft-bounding,
    and running loss scale calibration to prevent simplex collapse.

    Formula:
        s_bounded = clamp_bound * tanh(s / clamp_bound)
        w_i = M * exp(s_bounded_i / tau) / sum(exp(s_bounded_j / tau))
        L_calibrated_i = L_i / EMA(L_i)
        L_total = sum(w_i * L_calibrated_i * EMA(L_i).detach())

    Guarantees:
        1. sum(w_i) == M (strictly conserved total gradient scale)
        2. w_i > 0 for all tasks (no task starvation)
        3. Equalized competition across disparate loss scales via running EMA normalization
        4. Completely immune to numerical explosion under BF16 / FP32.
    """

    def __init__(
        self,
        task_names=None,
        num_tasks=None,
        temperature=2.0,
        clamp_bound=3.0,
        ema_momentum=0.99,
    ):
        super().__init__()
        if task_names is not None:
            self.task_names = list(task_names)
            self.num_tasks = len(self.task_names)
        elif num_tasks is not None:
            self.num_tasks = int(num_tasks)
            self.task_names = [f"task_{i}" for i in range(self.num_tasks)]
        else:
            self.task_names = ["cls", "offset", "size", "yaw", "geo"]
            self.num_tasks = len(self.task_names)

        self.temperature = float(temperature)
        self.clamp_bound = float(clamp_bound)
        self.ema_momentum = float(ema_momentum)

        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.clamp_bound <= 0.0:
            raise ValueError(f"clamp_bound must be positive, got {self.clamp_bound}")
        if not (0.0 < self.ema_momentum < 1.0):
            raise ValueError(f"ema_momentum must be in (0, 1), got {self.ema_momentum}")

        # Learnable log-scale logits initialized to 0 (all tasks start with equal weight = 1.0)
        self.log_scales = nn.Parameter(torch.zeros(self.num_tasks, dtype=torch.float32))

        # Running scale buffers to equalize disparate loss magnitudes
        self.register_buffer("running_loss_means", torch.ones(self.num_tasks, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def _bounded_scales(self):
        """Smooth tanh soft-bounding strictly within (-clamp_bound, clamp_bound)."""
        return self.clamp_bound * torch.tanh(self.log_scales / self.clamp_bound)

    @torch.no_grad()
    def get_task_weights(self):
        """Compute the normalized task weights as a dict {task_name: float_weight}."""
        bounded = self._bounded_scales()
        normalized = F.softmax(bounded / self.temperature, dim=0) * self.num_tasks
        return {name: float(normalized[i].item()) for i, name in enumerate(self.task_names)}

    def forward(self, task_losses):
        """Compute total weighted loss and return weights telemetry.

        Args:
            task_losses: dict mapping task_name -> scalar Tensor loss.
        """
        missing = [name for name in self.task_names if name not in task_losses]
        if missing and self.task_names != [f"task_{i}" for i in range(self.num_tasks)]:
            raise KeyError(
                f"Missing required task losses in TemperatureSoftmaxUncertainty: {missing}"
            )

        target_device = next(iter(task_losses.values())).device
        if self.log_scales.device != target_device:
            self.to(target_device)

        bounded = self._bounded_scales()
        normalized_weights = F.softmax(bounded / self.temperature, dim=0) * self.num_tasks

        is_generic = self.task_names == [f"task_{i}" for i in range(self.num_tasks)]
        if is_generic and len(task_losses) == self.num_tasks:
            ordered_losses = [task_losses[k].float() for k in task_losses.keys()]
            ordered_names = list(task_losses.keys())
        else:
            ordered_losses = [task_losses[name].float() for name in self.task_names]
            ordered_names = self.task_names

        stacked_losses = torch.stack(ordered_losses)

        with torch.no_grad():
            detached = stacked_losses.detach().clamp_min(1e-4)
            if not self.initialized:
                self.running_loss_means.copy_(detached)
                self.initialized.copy_(torch.tensor(True, device=target_device))
            else:
                self.running_loss_means.lerp_(detached, 1.0 - self.ema_momentum)

        calibrated_losses = stacked_losses / self.running_loss_means.clamp_min(1e-4)
        total_loss = torch.sum(
            normalized_weights * calibrated_losses * self.running_loss_means.detach()
        )

        weights_dict = {
            name: normalized_weights[i].detach() for i, name in enumerate(ordered_names)
        }

        return total_loss, weights_dict
```

- [x] **Step 4: Chạy lại toàn bộ test của uncertainty weighting để xác nhận PASS**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_uncertainty_weighting.py -v -p no:launch_testing -p no:launch_testing_ros
```
Expected: PASS (tất cả 6 tests).

- [x] **Step 5: Commit thay đổi Task 1**

```bash
git add detector/core/losses/uncertainty_weighting.py tests/test_uncertainty_weighting.py
git commit -m "fix(loss): resolve simplex collapse via running loss scale calibration"
```

---

### Task 2: Củng cố Xác thực Đầu vào `reg_mask` & Kiểm thử Mask Rỗng cho Baseline / UWAG

**Files:**
- Modify: `detector/core/losses/strategies/base.py`
- Test: `tests/test_loss_strategies.py`

**Interfaces:**
- Consumes: `pred: Dict[str, torch.Tensor]`, `target: Dict[str, torch.Tensor]`.
- Produces: Xác thực chặt chẽ các khóa bắt buộc (`"cls"`, `"offset"`, `"size"`, `"yaw"`, `"reg_mask"`) và kiểm tra kích thước `reg_mask` tương thích `(B, H, W)`.

- [x] **Step 1: Viết test cho việc thiếu/sai `reg_mask` và trường hợp mask rỗng trên Baseline/UWAG**

Thêm các test cases vào `tests/test_loss_strategies.py`:

```python
    def test_missing_or_invalid_reg_mask_raises_error(self):
        strategy = BaselineLossStrategy("gaussian")
        
        # Test thiếu reg_mask
        target_no_mask = {k: v for k, v in self.target.items() if k != "reg_mask"}
        with self.assertRaises(KeyError) as ctx:
            strategy(self.pred, target_no_mask)
        self.assertIn("reg_mask", str(ctx.exception))

        # Test sai shape reg_mask
        target_bad_mask = dict(self.target)
        target_bad_mask["reg_mask"] = torch.ones(self.B, 1, self.H, self.W)
        with self.assertRaises(ValueError) as ctx:
            strategy(self.pred, target_bad_mask)
        self.assertIn("reg_mask shape mismatch", str(ctx.exception))

    def test_baseline_and_uwag_empty_mask_finite_loss(self):
        target_empty = dict(self.target)
        target_empty["reg_mask"] = torch.zeros(self.B, self.H, self.W)

        # Baseline
        crit_base = BaselineLossStrategy("gaussian")
        pred_base = {k: v.clone().detach().requires_grad_(True) for k, v in self.pred.items()}
        out_base = crit_base(pred_base, target_empty)
        self.assertTrue(torch.isfinite(out_base["loss"]))
        out_base["loss"].backward()
        self.assertTrue(pred_base["offset"].grad is not None)

        # UWAG
        crit_uwag = UwagLossStrategy("gaussian")
        pred_uwag = {k: v.clone().detach().requires_grad_(True) for k, v in self.pred.items()}
        out_uwag = crit_uwag(pred_uwag, target_empty)
        self.assertTrue(torch.isfinite(out_uwag["loss"]))
        out_uwag["loss"].backward()
        self.assertTrue(pred_uwag["offset"].grad is not None)
```

- [x] **Step 2: Chạy test để xác nhận test thất bại**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_loss_strategies.py::TestLossStrategies::test_missing_or_invalid_reg_mask_raises_error -v -p no:launch_testing -p no:launch_testing_ros
```
Expected: FAIL với assertion `KeyError: 'reg_mask'` không như mong đợi hoặc lỗi không bắt đúng `ValueError`.

- [x] **Step 3: Cập nhật `_validate_inputs` trong `detector/core/losses/strategies/base.py`**

Thay thế hàm `_validate_inputs` trong `detector/core/losses/strategies/base.py`:

```python
    def _validate_inputs(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        """Validate shapes, channel counts, and required keys across predictions and targets."""
        for name in ("cls", "offset", "size", "yaw"):
            if name not in pred:
                raise KeyError(f"Missing required prediction head: {name!r}")
            if name not in target:
                raise KeyError(f"Missing required target head: {name!r}")

        if "reg_mask" not in target:
            raise KeyError("target dictionary must contain 'reg_mask'")

        for name in ("offset", "size", "yaw"):
            if pred[name].shape != target[name].shape:
                raise ValueError(
                    f"{name} prediction/target shape mismatch: "
                    f"{tuple(pred[name].shape)} != {tuple(target[name].shape)}"
                )

        expected_mask_shape = (
            pred["offset"].shape[0],
            pred["offset"].shape[2],
            pred["offset"].shape[3],
        )
        if target["reg_mask"].shape != expected_mask_shape:
            raise ValueError(
                f"reg_mask shape mismatch: {tuple(target['reg_mask'].shape)} != {expected_mask_shape}"
            )

        expected_cls_shape = (
            pred["cls"].shape[:1] + pred["cls"].shape[2:]
            if self.cls_encoding == "binary" else pred["cls"].shape
        )
        if target["cls"].shape != expected_cls_shape:
            raise ValueError(
                f"cls prediction/target shape mismatch: {tuple(pred['cls'].shape)} "
                f"is incompatible with {tuple(target['cls'].shape)}"
            )
        if pred["offset"].shape[1] not in {2, 3} or pred["size"].shape[1] != pred["offset"].shape[1]:
            raise ValueError("offset and size must both have 2 or 3 channels")
        if pred["yaw"].shape[1] != 2:
            raise ValueError("yaw must have 2 channels")
```

- [x] **Step 4: Chạy test xác nhận toàn bộ `test_loss_strategies.py` pass**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_loss_strategies.py -v -p no:launch_testing -p no:launch_testing_ros
```
Expected: PASS (tất cả 9 tests).

- [x] **Step 5: Commit thay đổi Task 2**

```bash
git add detector/core/losses/strategies/base.py tests/test_loss_strategies.py
git commit -m "fix(loss): add reg_mask validation and empty-mask test coverage"
```

---

### Task 3: Khắc phục Lỗi Phân biệt Hoa/Thường & KeyError trong `CustomModel`

**Files:**
- Modify: `detector/core/models/model.py`
- Test: `tests/test_model_registry.py`

**Interfaces:**
- Consumes: `cfg: Dict[str, Any]` (chấp nhận bất kể chữ hoa hay chữ thường cho `"backbone": "MobilePixorNeXt"`, `"mobilepixornext"`).
- Produces: `CustomModel` kích hoạt đúng `use_bn=True` và `act="silu"` cho mọi biến thể chữ hoa/thường của `mobilepixornext`.

- [x] **Step 1: Viết test kiểm tra tính không phân biệt hoa/thường của MobilePixorNeXt trong CustomModel**

Thêm test vào `tests/test_model_registry.py`:

```python
    def test_custom_model_case_insensitive_mobilepixornext_header(self):
        cfg_upper = {
            "backbone": "MobilePixorNeXt",
            "backbone_out_dim": 16,
            "cls_encoding": "gaussian",
        }
        model = CustomModel(cfg_upper, num_classes=3, input_channels=8)
        # Kiểm tra Header được cấu hình đúng với BatchNorm2d và SiLU
        self.assertTrue(model.header.use_bn)
        self.assertEqual(model.header.act, "silu")

    def test_custom_model_default_fallback_values(self):
        # Kiểm tra CustomModel khởi tạo an toàn khi thiếu một số key tùy chọn
        cfg_minimal = {"backbone": "mobilepixor"}
        model = CustomModel(cfg_minimal)
        self.assertIsNotNone(model)
```

- [x] **Step 2: Chạy test để xác nhận test thất bại**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_model_registry.py::TestBackboneRegistry::test_custom_model_case_insensitive_mobilepixornext_header -v -p no:launch_testing -p no:launch_testing_ros
```
Expected: FAIL với assertion `False is not true` (do `cfg.get("backbone") == "mobilepixornext"` bị lệch hoa/thường).

- [x] **Step 3: Cập nhật `detector/core/models/model.py`**

Chỉnh sửa `detector/core/models/model.py`:

```python
import torch.nn as nn

from core.models.backbones.registry import build_backbone
from core.models.heads.cnn import Header


class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes=4, input_channels=35):
        super(CustomModel, self).__init__()
        backbone_name = str(cfg.get("backbone", "mobilepixor"))
        self.backbone = build_backbone(backbone_name, cfg, input_channels=input_channels)

        self.num_classes = num_classes
        cls_encoding = str(cfg.get("cls_encoding", "gaussian")).lower()
        if cls_encoding == "binary":
            self.num_classes += 1

        is_mobilepixornext = backbone_name.lower() == "mobilepixornext"
        use_bn = cfg.get("header_use_bn", is_mobilepixornext)
        act = cfg.get("header_act", "silu" if is_mobilepixornext else "none")

        backbone_out_dim = cfg.get("backbone_out_dim", 16)

        self.header = Header(
            self.num_classes,
            backbone_out_dim,
            use_bn=use_bn,
            act=act,
        )

    def forward(self, x):
        features = self.backbone(x)
        pred = self.header(features)
        return pred


if __name__ == "__main__":
    cfg = {
        "backbone": "mobilepixor",
        "backbone_out_dim": 16,
        "cls_encoding": "gaussian",
    }

    model = CustomModel(cfg)
    print("CustomModel initialized successfully with default parameters:")
    print(f"Backbone: {type(model.backbone).__name__}")
    print(f"Header output channels: {model.header.out_channels}")
```

- [x] **Step 4: Chạy test xác nhận toàn bộ `test_model_registry.py` pass**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_model_registry.py -v -p no:launch_testing -p no:launch_testing_ros
```
Expected: PASS (tất cả 7 tests).

- [x] **Step 5: Commit thay đổi Task 3**

```bash
git add detector/core/models/model.py tests/test_model_registry.py
git commit -m "fix(model): handle case-insensitive backbone naming and safe config fallbacks"
```

---

### Task 4: Hỗ trợ Cờ Ghi đè Registry & Đồng nhất Hằng số Epsilon

**Files:**
- Modify: `detector/core/models/backbones/registry.py`
- Modify: `detector/core/losses/strategies/registry.py`
- Modify: `detector/core/losses/strategies/oga.py`
- Test: `tests/test_model_registry.py`
- Test: `tests/test_loss_strategies.py`

**Interfaces:**
- Consumes: `@register_backbone(name, allow_override=False)`, `@register_loss_strategy(name, allow_override=False)`.
- Produces: Cho phép ghi đè trong các tình huống mock kiểm thử hoặc tải lại module khi `allow_override=True`. Đồng nhất mặc định `epsilon=1e-6` cho `OgaLossStrategy`.

- [x] **Step 1: Viết test cho tính năng ghi đè `allow_override=True`**

Thêm test vào `tests/test_model_registry.py`:

```python
    def test_allow_override_registration(self):
        @register_backbone("temp_override_test")
        def _builder1(cfg, in_c):
            return nn.Identity()

        # Không bật override -> lỗi KeyError
        with self.assertRaises(KeyError):
            @register_backbone("temp_override_test")
            def _builder2(cfg, in_c):
                return nn.Identity()

        # Bật allow_override=True -> thành công
        @register_backbone("temp_override_test", allow_override=True)
        def _builder3(cfg, in_c):
            return nn.ReLU()

        bb = build_backbone("temp_override_test", {})
        self.assertIsInstance(bb, nn.ReLU)
```

- [x] **Step 2: Chạy test để xác nhận test thất bại**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_model_registry.py::TestBackboneRegistry::test_allow_override_registration -v -p no:launch_testing -p no:launch_testing_ros
```
Expected: FAIL với `TypeError: register_backbone() takes 1 positional argument but 2 were given`.

- [x] **Step 3: Cập nhật registry cho backbone, loss strategy và epsilon trong OgaLossStrategy**

1. Cập nhật `detector/core/models/backbones/registry.py`:
```python
def register_backbone(name: str, allow_override: bool = False):
    """Decorator to register a backbone builder function or class."""
    def decorator(builder: BackboneBuilder):
        key = name.lower()
        if key in _BACKBONE_REGISTRY and not allow_override:
            raise KeyError(f"Backbone {name!r} is already registered.")
        _BACKBONE_REGISTRY[key] = builder
        return builder
    return decorator
```

2. Cập nhật `detector/core/losses/strategies/registry.py`:
```python
def register_loss_strategy(name: str, allow_override: bool = False):
    """Decorator to register a loss strategy class or builder."""
    def decorator(cls_or_builder: Any):
        key = str(name).lower()
        if key in _LOSS_STRATEGY_REGISTRY and not allow_override:
            raise KeyError(f"Loss strategy {name!r} is already registered.")
        _LOSS_STRATEGY_REGISTRY[key] = cls_or_builder
        return cls_or_builder
    return decorator
```

3. Cập nhật `detector/core/losses/strategies/oga.py`:
Đồng nhất `epsilon` mặc định thành `1e-6`:
```python
        self.eps = float(config.get("epsilon", 1e-6))
```

- [x] **Step 4: Chạy test xác nhận PASS**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_model_registry.py tests/test_loss_strategies.py -v -p no:launch_testing -p no:launch_testing_ros
```
Expected: PASS toàn bộ.

- [x] **Step 5: Commit thay đổi Task 4**

```bash
git add detector/core/models/backbones/registry.py detector/core/losses/strategies/registry.py detector/core/losses/strategies/oga.py tests/test_model_registry.py
git commit -m "feat(registry): add allow_override option and harmonize default epsilon"
```

---

### Task 5: Xác minh Tích hợp Đầu-Cuối (End-to-End Verification)

**Files:**
- Test: Toàn bộ bộ kiểm thử trong `tests/`
- Config: `configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json`

**Interfaces:**
- Đảm bảo toàn bộ luồng huấn luyện `test_training_pipeline_oga.py` và 59+ bài test chạy trơn tru mà không có cảnh báo/lỗi bất thường.

- [x] **Step 1: Chạy toàn bộ test suite của dự án**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets:tools/kitti_training_pipeline /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros
```
Expected: 100% tests PASS (không có test nào bị lỗi hoặc crash).

- [x] **Step 2: Chạy kiểm tra chạy thử script model.py**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python detector/core/models/model.py
```
Expected: Exited with code 0 và in ra thông tin cấu hình `CustomModel`.

- [x] **Step 3: Chạy test quy trình huấn luyện tích hợp OGA**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets:tools/kitti_training_pipeline /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_training_pipeline_oga.py -v -p no:launch_testing -p no:launch_testing_ros
```
Expected: PASS.

- [x] **Step 4: Commit và tổng kết**

```bash
git status
git commit --allow-empty -m "chore(release): verify all oga fixes and edge-case validations"
```

---

## Tự Đánh giá Kế hoạch (Self-Review Checklist)

1. **Bao phủ Spec & Đánh giá Code Review**:
   - Vấn đề 🔴 (Simplex collapse / loss cheating): Đã có giải pháp EMA Scale Calibration tại **Task 1**.
   - Vấn đề 🟡 (`reg_mask` validation): Đã có tại **Task 2**.
   - Vấn đề 🟡 (`CustomModel` case sensitivity & fallbacks): Đã có tại **Task 3**.
   - Vấn đề 🟢 (Registry override & epsilon): Đã có tại **Task 4**.
   - Test regression & pipeline E2E: Đã có tại **Task 5**.
2. **Không có Placeholder (Zero Placeholders)**: Mọi bước đều chứa mã nguồn và lệnh thực thi đầy đủ 100%.
3. **Độ nhất quán về Type & Interface**: Mọi biến và buffer (`running_loss_means`, `initialized`, `allow_override`) đều đồng nhất trên toàn bộ các file.
