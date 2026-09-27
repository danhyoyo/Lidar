# Kế hoạch Thực thi: Tái cấu trúc Backbone Registry & Loss Strategy Pattern

> **Dành cho Agent thực thi:** BẮT BUỘC: Sử dụng quy trình kiểm thử TDD, không làm phá vỡ các checkpoint huấn luyện và đảm bảo 100% backward compatibility cho toàn bộ bộ kiểm thử 45 tests hiện có.

**Mục tiêu:** Loại bỏ hoàn toàn chuỗi `if/elif` cứng trong `CustomModel.__init__` bằng **Backbone Registry Pattern** (tuân thủ Open/Closed Principle) và phân rã lớp "thần thánh" `LossFunction` thành các module độc lập theo **Strategy Pattern** (`BaselineLossStrategy`, `UwagLossStrategy`, `OgaLossStrategy`) với cơ chế chuyển tiếp state_dict tương thích ngược.

**Kiến trúc:**
1. **Backbone Registry (`detector/core/models/backbones/registry.py`)**: Cung cấp decorator `@register_backbone(name)` và hàm điều phối `build_backbone(name, cfg, input_channels)`. `CustomModel` chỉ giao tiếp với registry, đóng lại hoàn toàn đối với việc sửa đổi trực tiếp khi bổ sung backbone mới.
2. **Loss Strategy Subsystem (`detector/core/losses/strategies/`)**:
   - `BaseLossStrategy`: Định nghĩa giao diện chuẩn hóa cho kiểm tra shape và tính hàm mất mát phân loại (Focal Loss).
   - `BaselineLossStrategy`: Focal Loss + unweighted L1 loss cho các tọa độ.
   - `UwagLossStrategy`: Homoscedastic uncertainty weighting (Kendall) + footprint IoU (kèm tham số học `log_scales`).
   - `OgaLossStrategy`: Smooth-L1 coordinate regression + OrientedGeometryLoss ($\mathcal{L}_{\text{NCD}} + \mathcal{L}_{\text{proj}}$) + T-SBUW uncertainty balancing.
   - `LossStrategyRegistry`: Decorator `@register_loss_strategy(name)` và hàm `build_loss_strategy(...)`.
   - `LossFunction` đóng vai trò là một Context / Façade mỏng, tự động ánh xạ checkpoint state_dict (`_load_from_state_dict` / `state_dict`) để giữ tương thích 100% với các checkpoint đã lưu mà không có tiền tố `strategy.`.

---

## Yêu cầu Người dùng Đánh giá (User Review Required)

> [!IMPORTANT]
> **Đảm bảo Tương thích Ngược Checkpoint Huấn luyện (Checkpoint State Dict Compatibility):**
> Trong các phiên bản trước:
> - Cấu hình `uwag` lưu khóa `"log_scales"` trực tiếp trong `criterion_state_dict`.
> - Cấu hình `oga` lưu khóa `"weighting.log_scales"` trong `criterion_state_dict`.
> 
> Khi phân rã thành Strategy Pattern, nếu `LossFunction` chứa `self.strategy`, các khóa PyTorch mặc định sẽ trở thành `"strategy.log_scales"` hoặc `"strategy.weighting.log_scales"`.
> **Giải pháp kiến trúc trong kế hoạch:** `LossFunction` sẽ cài đặt cơ chế ánh xạ trong `_load_from_state_dict` và `state_dict(prefix=...)` để chấp nhận cả hai định dạng (có hoặc không có tiền tố `strategy.`). Nhờ đó, việc nạp resume checkpoint cũ từ Colab/Server vẫn thành công 100% với `strict=True`.

> [!NOTE]
> **Không thay đổi hợp đồng giao diện cấu hình JSON:**
> Mọi file cấu hình hiện tại (`kitti_bevnext_litemla_oga.json`, `kitti_bevnext_litemla.json`, `kitti_mobilepixor_baseline.json`,...) và cấu trúc trả về `loss_dict` giữ nguyên 100% tính tương thích.

---

## Các Thay đổi Đề xuất (Proposed Changes)

### Hợp phần 1: Backbone Registry System
- `detector/core/models/backbones/registry.py`
- `detector/core/models/model.py`

### Hợp phần 2: Loss Strategy Subsystem
- `detector/core/losses/strategies/base.py`
- `detector/core/losses/strategies/baseline.py`
- `detector/core/losses/strategies/uwag.py`
- `detector/core/losses/strategies/oga.py`
- `detector/core/losses/strategies/registry.py`
- `detector/core/losses/strategies/__init__.py`
- `detector/core/losses/loss_fn.py`

### Hợp phần 3: Bộ Kiểm thử Mở rộng
- `tests/test_model_registry.py`
- `tests/test_loss_strategies.py`

---

## Kế hoạch Xác minh (Verification Plan)
- Toàn bộ 45 bài test hiện tại và test mới đều pass 100%.
- Kiểm tra `test_mobile_bev.py`.
