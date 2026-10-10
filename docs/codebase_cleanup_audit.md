# Rà soát phần có thể dọn trong codebase

Ngày rà soát: 2026-10-10. Phạm vi chính: `Lidar/`, sau khi bỏ LiteMLA C4, GW-QAL và rich10/11/12.

Đã kiểm tra 303 file văn bản, gồm 171 file Python trong `detector/`, `tools/`, `tests/`, hai notebook, config và tài liệu. Dùng AST để xem import và lời gọi nội bộ, tìm cả tên module lẫn symbol xuất ra, và đối chiếu registry, entry point cùng lệnh chạy trong notebook/tài liệu. Sau khi người dùng đồng ý, đã thực hiện các mục dọn mã nguồn bên dưới. Hai file visualization khác xuất hiện đồng thời trong workspace; chúng được giữ nguyên, nên lượt kiểm tra cú pháp cuối vẫn gồm 171 file Python trong phạm vi này.

## Module không có nơi sử dụng trong repo

| File | Bằng chứng | Trạng thái |
| --- | --- | --- |
| `detector/core/losses/gaussian_geometry_loss.py` | Không có import/tham chiếu từ file khác tới module hoặc `box_covariance`, `KFIoULoss`, `ProbIoULoss`, `KLDLoss`. Không đăng ký trong loss registry. 422 dòng, 17.034 byte. | Đã xoá. |
| `detector/core/losses/mgiou_loss.py` | Không có import/tham chiếu từ file khác tới module hoặc `MGIoULoss`. Không đăng ký trong loss registry. 167 dòng, 6.854 byte. | Đã xoá implementation riêng này. |

Tổng: 589 dòng, khoảng 23,3 KiB mã nguồn. MGIoU đang chạy trong OGA vẫn được giữ ở `detector/core/losses/oriented_geometry_loss.py`: OGA khởi tạo `OrientedGeometryLoss`, lớp này gọi `multiaxis_projection_giou`. Không cần xoá hoặc đổi loss OGA để bỏ module MGIoU riêng.

## Helper có thể rút gọn

| File | Phần không có nơi sử dụng | Phần phải giữ |
| --- | --- | --- |
| `detector/core/torchplus.py` | `Sequential` riêng; `get_kw_to_default_map`; nhóm đổi kiểu NumPy/Torch (`np_dtype_to_torch`, `np_dtype_to_np_type`, `np_type_to_torch`, `torch_to_np_type`, `_torch_string_type_to_class`, `torch_to_np_dtype`); `isinf`, `to_tensor`, `zeros`, `get_tensor_class`. Các lời gọi giữa helper đổi kiểu chỉ nằm trong nhóm này. | RPN import `Empty`, `change_default_args`; `change_default_args` còn dùng `get_pos_to_kw_map`. Giữ cả ba. |
| `detector/core/datasets/utils_1/transform.py` | `corner_to_center_box2d`, `inverse_rigid_trans`: không có lời gọi nội bộ hoặc tham chiếu ngoài module. | `Random_Rotation` còn gọi `point_transform` và `box_transform`; `box_transform` dùng các hàm chuyển đổi hộp 3D. Giữ chuỗi này và các augmentation đang dùng. |
| `detector/core/datasets/utils_1/preprocess.py` | `transform_metric2label`: không có lời gọi nội bộ hoặc tham chiếu ngoài module. | Encoder, voxelization, các helper có consumer hoặc test còn lại. |

Không kết luận hàm là thừa chỉ vì thiếu import bên ngoài. Ví dụ `convert_format`, `compute_iou`, `non_max_suppression` trong `detector/postprocess.py` vẫn được luồng decode/NMS gọi nội bộ; các backend NumPy/Numba cũng còn được wrapper gọi.

## Mục dọn nhỏ và file sinh tự động

- `requirements-kitti.txt` trước đây khai báo ONNX hai lần (`onnx>=1.15` và `onnx>=1.16`). Đã gộp, giữ `onnx>=1.16`. ONNX vẫn được dùng để export.
- Sau khi hoàn tất kiểm tra, đã dọn 18 thư mục `__pycache__/` và `.pytest_cache/`, tổng 487 file, 11.778.274 byte (khoảng 11,2 MiB). Python/pytest có thể tạo lại cache khi chạy.
- Đã xoá `mobilepixornext_architecture.png` ở gốc repo khoảng 1,04 MiB: không có tham chiếu từ code/notebook/tài liệu được rà soát. Các thay đổi sơ đồ diễn ra đồng thời trong workspace nằm ngoài lượt dọn này.

## File cần giữ

- `tools/generate_robust_notebook.py` có entry point chạy trực tiếp và sinh notebook Robust; thiếu import từ file khác không có nghĩa là thừa.
- `select_checkpoint.py`, các tool benchmark/deploy/export và `docs/plans/lightweight_lidar_backbone_2026/profile_architecture.py` còn có lệnh chạy hoặc tham chiếu trong tài liệu.
- Các backbone trong registry, package `__init__.py`, test và fixtures còn dùng.
- Dữ liệu KITTI, checkpoint, kết quả run, cấu hình resolved và báo cáo nghiên cứu. Tài liệu lịch sử về tính năng đã bỏ vẫn là bằng chứng thí nghiệm.
- `.agents/`, `e2e_demo/` chứa lịch sử làm việc, không phải mã runtime; không thể suy ra giá trị lưu trữ chỉ từ import.

Ngoài repo chính, thư mục cha có bốn ZIP backup tổng 229.841.094 byte (khoảng 219 MiB) và bản repo `test/Lidar` khoảng 86 MiB trên đĩa. Các mục này có thể giúp giảm dung lượng workspace nếu đã có bản lưu khác; chúng không thuộc danh sách mã chết trong repo chính.

## Giới hạn kiểm chứng

Lượt audit ban đầu dùng Python hệ thống thiếu PyTorch/Numba. Sau đó đã tìm được `/home/duyennh/miniconda3/envs/AI_env/bin/python` có PyTorch 2.11.0 và Numba 0.67.0 để kiểm tra thực thi trên CPU.

Đã xoá hai module loss không dùng, các helper được liệt kê, ảnh gốc không có tham chiếu và dòng ONNX trùng; giữ `onnx>=1.16`. Các hàm/class còn lại trong ba file helper được đối chiếu với snapshot trước khi xoá và giữ nguyên nội dung.

OGA trước/sau đã được kiểm tra trên 100 tình huống với 600 lượt forward/backward mỗi bản. Loss, gradient, trọng số adaptive và trạng thái EMA/checkpoint khớp tuyệt đối; 167 test liên quan OGA đều đạt ở cả hai bản. Chi tiết, môi trường và bằng chứng nằm trong [oga_cleanup_verification.md](oga_cleanup_verification.md).

Lượt chạy suite sau khi sửa các test còn sót từ lần bỏ LiteMLA/GW-QAL: 1.667 passed, 84 skipped, 23 deselected, 15 subtests passed. 23 test được loại trừ gồm 14 lỗi đã tái hiện trên Git HEAD và 9 test worker bị môi trường chặn socket; không phải toàn bộ suite đã đạt.
