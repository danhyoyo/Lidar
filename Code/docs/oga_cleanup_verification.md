# Kiểm chứng OGA khi dọn codebase

Ngày: 2026-10-10. Snapshot trước lượt xoá module/helper không dùng: `/tmp/lidar-oga-cleanup-before/`. Bản Git HEAD trước các lượt dọn: `/tmp/lidar-cleanup-git-head/`.

## Đường gọi và các thành phần được giữ

`LossFunction` → loss registry → `OgaLossStrategy` → `OrientedGeometryLoss` và `TemperatureSoftmaxUncertainty`.

- Classification: focal loss cho binary; modified focal loss cho Gaussian.
- Offset, size, yaw: masked Smooth L1.
- Geometry: `multiaxis_projection_giou` + `corner_beta × pi_symmetric_corner_distance`. MGIoU dùng bốn trục chiếu từ hai hộp; corner distance xét cả cách ghép góc quay π và chuẩn hoá bằng đường chéo hộp đích.
- Corner decoding giữ clamp log-size và fallback yaw có norm nhỏ; telemetry `clamp_count`, `fallback_count` không đổi.
- Khi `use_iou=True`, thêm BCE với quality target đã detach: `mgiou`, `rotated_iou` hoặc `yaw_footprint`. Geometry vẫn truyền gradient vào các head regression.
- T-SBUW giữ `s_bounded = B × tanh(s/B)`, `w = M × softmax(s_bounded/temperature)`, loss chuẩn hoá bằng EMA và nhân lại scale danh nghĩa. EMA cập nhật khi train, đứng yên khi eval; checkpoint cũ thiếu buffer vẫn tải được.
- Nhánh 3D thêm Smooth L1 vertical với hệ số riêng qua facade.

`mgiou_loss.py` chứa implementation `MGIoULoss` độc lập, không được đường gọi này import. `gaussian_geometry_loss.py` cũng không được import hoặc đăng ký. Xoá hai file này không xoá MGIoU đang dùng trong OGA.

## So sánh thực thi trước/sau

Môi trường: Python 3.14 trong Conda `AI_env`, PyTorch 2.11.0, NumPy 2.4.6, Numba 0.67.0, pytest 9.1.1. Chạy CPU, một thread, seed `20261010`, bật deterministic algorithms; hai bản chạy trong hai process độc lập.

100 tình huống = 2 kiểu classification × 5 nhánh (BEV thường, 3 quality target, 3D) × 5 input (ngẫu nhiên/mask hỗn hợp, mask rỗng, hộp trùng, hộp tách xa, log-size quá ngưỡng/yaw bằng 0) × 2 chế độ FP32/BF16 autocast.

Mỗi tình huống chạy ba bước train có cập nhật SGD, một bước eval, một bước tải checkpoint hiện tại rồi eval, và một bước tải checkpoint cũ thiếu EMA buffer rồi train. Tổng **600 lượt forward/backward mỗi bản**.

| Đại lượng | Kết quả trước/sau |
| --- | --- |
| Tổng loss và mọi loss thành phần | Khớp tuyệt đối |
| Corner distance, projection GIoU, quality target mean, clamp/fallback count | Khớp tuyệt đối |
| Gradient cls, offset, size, yaw; IoU/vertical khi có | Khớp tuyệt đối |
| Gradient tham số weighting, trọng số từng task | Khớp tuyệt đối |
| Toàn bộ `state_dict`, gồm log-scales, running means, initialized | Khớp tuyệt đối |
| Loss, telemetry, gradient và state hữu hạn | Đạt ở cả hai bản |
| Trọng số task dương và tổng bằng số task (sai số < 1e-5) | Đạt |
| Eval không đổi state; checkpoint hiện tại giữ nguyên output/gradient | Đạt |

JSON trước/sau giống từng byte, SHA-256: `2f7b6b311c140578d874507b32352041484c93e423cb82b8889dd2a952ebe562`.

Harness cũng được chạy trên bản Git HEAD trước toàn bộ các lượt dọn LiteMLA/GW-QAL/rich10/11/12. Cả 100 tình huống và 600 observations của bản này cũng khớp tuyệt đối với bản sau khi dọn.

Các artifact đối chiếu trên máy hiện tại:

- Harness: `/tmp/check_oga_numerics.py`.
- Outputs: `/tmp/oga-numerics-before.json`, `/tmp/oga-numerics-after.json`.
- Output Git HEAD: `/tmp/oga-numerics-git-head.json`.
- Hash nguồn: `/tmp/oga-dependency-hashes-before.json`.
- Log test: `/tmp/oga-tests-before.log`, `/tmp/oga-tests-after.log`.

Lệnh chạy lại harness trong môi trường có dependency:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/duyennh/miniconda3/envs/AI_env/bin/python /tmp/check_oga_numerics.py /tmp/lidar-oga-cleanup-before /tmp/oga-numerics-before.json
PYTHONDONTWRITEBYTECODE=1 /home/duyennh/miniconda3/envs/AI_env/bin/python /tmp/check_oga_numerics.py /home/duyennh/AI_projects/research_lidar/Lidar /tmp/oga-numerics-after.json
cmp /tmp/oga-numerics-before.json /tmp/oga-numerics-after.json
```

Các artifact `/tmp` là dữ liệu tạm trên máy hiện tại, không được thêm vào codebase.

## Các kiểm tra bổ sung

- 21 file trong closure import tĩnh của loss facade có SHA-256 không đổi trước/sau lượt dọn module/helper.
- Không còn tham chiếu tới module/helper đã xoá trong runtime, tools, tests, config, notebook và tài liệu đang dùng.
- 3 hàm/class còn dùng trong `torchplus.py`, 9 trong `transform.py`, 7 trong `preprocess.py` giữ nguyên nội dung so với snapshot.
- Output ba helper hình học và ba augmentation NumPy khớp tuyệt đối trước/sau với cùng input/seed.
- 167 test liên quan OGA đạt trước và sau, thêm 4 subtest đạt. Có coverage geometry, adaptive weighting, loss registry, checkpoint cũ, IQA, grouped loss, vertical và wiring pipeline.
- 57 test context/grouped adapter đạt sau khi sửa tuple test còn sót từ lượt bỏ GW-QAL.
- 73 test backbone/registry/neck/encoder còn giữ đạt. Ba test còn sót được sửa và chạy lại đạt: RC-SGFPN có 616.728 tham số backbone, deploy detector có 634.185 tham số, JSON override mẫu dùng `oga`/`corner_beta` hợp lệ.
- 23 config còn lại validate; 20 preset notebook resolve; 171 file Python parse ở lượt cuối (gồm hai file visualization khác xuất hiện đồng thời trong workspace); 20 code cell notebook compile. Rich8 tiêu chuẩn khớp output cũ ở ba bộ input.
- Đã dọn 18 thư mục cache Python/pytest, 487 file, 11.778.274 byte. Các snapshot/output kiểm chứng ở `/tmp` vẫn còn.

## Phạm vi kết luận

So sánh xác nhận loss OGA và gradient/state tương đương trên các tình huống CPU đã chạy. Máy không có CUDA khả dụng, nên đây không phải kiểm chứng CUDA hay kết quả AP sau một run KITTI đầy đủ.

## Kết quả toàn bộ suite và lỗi đã phân loại

Lượt chạy đầy đủ ban đầu: **1.664 passed, 27 failed, 84 skipped, 15 subtests passed**. Đã sửa bốn mục test liên quan lượt dọn trước đó: case negative còn giả định default LiteMLA, hai số lượng tham số còn cộng LiteMLA, và JSON ví dụ GW-QAL bị lỗi khi thay thế. Case negative được bỏ; các test thay đổi được chạy lại đạt.

23 lỗi còn lại đã được phân loại riêng:

| Nhóm | Số test | Bằng chứng |
| --- | --- | --- |
| Recipe config BEV/3D không khớp config sinh lại (train/evaluation) | 11 | Tái hiện cả 11 trên Git HEAD trước khi dọn. |
| Smoke giới hạn batch nhưng dùng AP selection | 2 | Tái hiện trên Git HEAD; trainer yêu cầu full split khi chọn AP. |
| Notebook test còn yêu cầu cờ `--train-split`/`--val-split` | 1 | Tái hiện trên Git HEAD. |
| DataLoader nhiều worker không chia sẻ tensor được | 9 | Traceback chặn socket/forkserver/resource_sharer với `PermissionError`; một test đại diện cũng lỗi cùng nguyên nhân trên Git HEAD. |

Không sửa config thí nghiệm hoặc code multiprocessing để che các lỗi này. Danh sách node chính xác nằm trong `/tmp/lidar-cleanup-suite-exclusions.json`; log đầy đủ ban đầu ở `/tmp/lidar-cleanup-full-conda.log`. Log đối chiếu Git HEAD: `/tmp/lidar-preset-head-tests.log`, `/tmp/lidar-preexisting-head-tests.log`, `/tmp/lidar-hybrid-head-test.log`, `/tmp/lidar-worker-head-test.log`.

Lượt chạy lại toàn bộ các test còn lại sau khi sửa test liên quan cleanup: **1.667 passed, 84 skipped, 23 deselected, 15 subtests passed**, exit code 0. Log: `/tmp/lidar-cleanup-suite-available.log`. Đây là kết quả của suite có loại trừ 23 node đã phân loại ở trên, không phải toàn bộ suite không lỗi.

Khi chạy pytest trong môi trường này, dùng `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` để tránh pytest tự tải plugin ROS thiếu `osrf_pycommon`; dùng `PYTHONDONTWRITEBYTECODE=1` và `-p no:cacheprovider` để không tạo lại cache Python/pytest.
