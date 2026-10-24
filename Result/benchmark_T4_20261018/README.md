# Báo cáo benchmark Tesla T4 — 18/10/2026

**Bản mới nhất V7: [Excel thêm speed chi tiết 8 model](Tables_bai_bao_LiDAR_v7_SpeedDetail.xlsx).** Mở đầu tại `B10_Speed_Detail`: preprocess/input encoding gộp, H2D, model/network, decode/final NMS, overhead, offline E2E, hai loại FPS và E2E P50/P95/P99. `B11_Speed_Stats` có 40 dòng (8 model × 5 stage), `B12_Timing_Scope` mô tả phần được đo. [Excel riêng speed](Speed_chi_tiet_8_models_v7.xlsx) · [HTML bảng và biểu đồ](Speed_chi_tiet_8_models_v7.html) · [PNG biểu đồ](speed_stage_breakdown_v7.png).

Chưa có encoder-only/backbone/neck/head timing tách riêng; input encoding không được tách khỏi I/O/preprocess, và learned encoder có thể nằm trong network. E2E là offline input-to-detections, không phải sensor-to-actuator. MobilePIXOR dùng L4, các dòng khác T4. Tạo lại bằng `python3 Result/paper_tables/publish_speed_detail.py`; kiểm tra nguồn/số tại `speed_detail_v7_verification.json`. Không chạy profiling mới; mọi bảng mAP, khoảng cách và augmentation được giữ nguyên.

**Bản mới nhất V6: [Excel có trang tóm tắt mAP / mAP9](Tables_bai_bao_LiDAR_v6_mAP_mAP9.xlsx).** Mở đầu tại `T00_Tom_tat_mAP_mAP9`, mAP và mAP9 ở cột B/C cạnh tên mô hình, cùng tham số và speed. [File chỉ có bảng tóm tắt](Benchmark_tom_tat_mAP_mAP9_v6.xlsx). Sheet khoảng cách có thêm hai cột mAP/mAP9 **toàn tập**, không phải range mAP9. Các dữ liệu cũ được giữ nguyên. Tạo lại bằng `python3 Result/paper_tables/publish_map_summary.py`; kiểm tra ở `mAP_mAP9_v6_verification.json`.

**Bản mới nhất V5: [Excel đầy đủ có AP khoảng cách](Tables_bai_bao_LiDAR_v5_DistanceAP.xlsx).** Mở đầu ở `B08_AP_Distance`; bảng cũng được thêm vào phần B phía dưới `T01_Benchmark`. `B09_Distance_Detail` kèm số GT, TP/FP và max recall. [Excel riêng AP khoảng cách](AP_khoang_cach_Moderate_v5.xlsx) · [HTML AP khoảng cách](AP_khoang_cach_Moderate_v5.html).

Dải hiện có là 0–30, 30–50 và 50–70,4 m, difficulty Moderate. Chỉ có report cho Pedestrian/Cyclist của MobilePIXOR và PixorNeXt: 12 AP. Car và các baseline khác được ghi `Chưa có`, không nội suy từ AP tổng. Dấu — chỉ dùng cho lớp không áp dụng của PIXOR. Chưa tính mAP ba lớp theo dải vì thiếu Car. Script tái tạo: `python3 Result/paper_tables/publish_distance_benchmark.py`; kiểm tra tại `distance_AP_v5_verification.json`. Các bảng difficulty, timing và augmentation cũ được giữ nguyên.

**Bản V4 trước khi thêm khoảng cách: [Excel V4 — đã thêm MobilePIXOR, 8 model](Tables_bai_bao_LiDAR_v4_MobilePIXOR.xlsx).** MobilePIXOR là dòng riêng, có đầy đủ AP E/M/H ba lớp, macro mAP theo độ khó, mAP9, tham số và speed. Đã bổ sung cột GPU: MobilePIXOR là L4 theo report lịch sử, các dòng khác T4. MobilePIXOR chọn theo loss ở epoch 30, run chỉ lưu 38/100 epoch. [File riêng benchmark V4](Benchmark_DAY_DU_v4_MobilePIXOR.xlsx) · [HTML V4](Benchmark_DAY_DU_v4_MobilePIXOR.html).

Bản V4 giữ nguyên các sheet augmentation và các bảng cũ khác. Script tái tạo: `python3 Result/paper_tables/publish_mobilepixor_benchmark.py`; phạm vi kiểm tra và nguồn MobilePIXOR ở `benchmark_v4_MobilePIXOR_verification.json` và `mobilepixor_source_record.json`. Không chạy train hoặc inference mới.

**Bản V3 trước khi thêm MobilePIXOR: [Excel V3 — benchmark đầy đủ](Tables_bai_bao_LiDAR_T4_v3_BenchmarkFull.xlsx).** File mở ngay `T01_Benchmark`: mAP Easy/Moderate/Hard, mAP9, tham số, forward/network ms và FPS, pipeline ms và FPS, AP từng lớp ở ba độ khó, checkpoint và nguồn. PIXOR chỉ có Car; các ô macro/Pedestrian/Cyclist để —. Kéo ngang để xem hết AP từng lớp. PixorNeXt có số local và được đánh dấu rõ trong cột protocol. Sheet augmentation G16/O32/H32 giữ nguyên.

[Excel riêng trang benchmark](Benchmark_T4_DAY_DU.xlsx) · [Xem bảng trực tiếp bằng HTML](Benchmark_T4_DAY_DU.html) · [CSV BEV đầy đủ](benchmark_BEV_full.csv).

Bảng `B07_3D_Full` bổ sung AP3D R40 đầy đủ của năm baseline. PIXOR và PixorNeXt hiện tại là BEV-only nên không có AP3D hoặc timing 3D. Có thể tạo lại bản V3 sau các bước bên dưới bằng `python3 Result/paper_tables/publish_full_benchmark.py`. Kết quả kiểm tra ở `benchmark_full_verification.json`; không ghi đè file V2 đang mở.

- [Báo cáo HTML độc lập](Bao_cao_so_sanh_T4.html): có các bảng, hình, phân tích và hướng hoàn tất common-protocol AP; có thể mở trong trình duyệt và in.
- [Báo cáo Markdown](Bao_cao_so_sanh_T4.md): bản văn bản có thể chỉnh sửa.
- [Excel V2 mở sẵn augmentation đã sửa](Tables_bai_bao_LiDAR_T4_v2_AugFixed.xlsx): sheet đầu tiên là S02_Augmentation với G16/O32/H32; giữ đủ các bảng khác. Bản V2 đặt tên mới để phân biệt với bản đã mở trước đó.
- [Excel chỉ có bảng augmentation](Augmentation_3_nhom_DA_SUA.xlsx): bảng ba nhóm có thể mở và kiểm tra ngay.
- [Excel cập nhật benchmark trước khi đổi trang mở đầu](Tables_bai_bao_LiDAR_T4_updated.xlsx): T01 đã cập nhật; B01–B06 chứa Car AP, macro AP, timing, checkpoint sensitivity, protocol và nguồn. S02 cũng đã có G16/O32/H32.
- CSV từng bảng và PNG/SVG riêng phục vụ biên tập.
- `sources/` lưu bản sao report gốc cùng xác nhận GPU của PIXOR từ người dùng; `source_manifest.json` lưu hash và nguồn.
- `verification.json` ghi phạm vi kiểm tra; `summary_metrics.json` chứa các số tổng hợp chính xác.

Checkpoint chính của PixorNeXt là **best AP epoch 130**, 753.569 tham số, local Moderate macro AP 71,461874%, pipeline 60,956325 ms/frame (16,405189 FPS). Không lấy timing best loss epoch 47 để gán cho best AP.

PIXOR là bản tái lập Car-only 50 epoch, T4 theo xác nhận của người dùng. AP PIXOR và năm baseline là reference; AP PixorNeXt hiện là local và được giữ riêng. Reference AP của PixorNeXt chưa được chấm trong lần tổng hợp này; không thay bằng local AP.

Tạo lại từ các report trong thư mục nguồn người dùng đã cung cấp, chạy từ thư mục gốc dự án:

```sh
python3 Result/paper_tables/build_t4_report.py
```

Cần `openpyxl` và `matplotlib`. Lệnh ghi lại các output của báo cáo; lưu bản riêng nếu đã chỉnh thủ công. Không có training hoặc inference mới. File Excel gốc và bản thảo không bị thay đổi; Excel T4 là một bản cập nhật riêng.

Sau lệnh trên, chạy `python3 Result/paper_tables/publish_augmentation_fix.py` để xuất bản V2 mở sẵn S02 cùng file một sheet và ảnh xem trước. Kết quả kiểm tra của bước này nằm trong `augmentation_delivery_verification.json`.
