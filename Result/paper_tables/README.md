# Các bảng dự kiến cho bài báo LiDAR

Mở `Tables_bai_bao_LiDAR.xlsx` và bắt đầu tại sheet `00_Muc_luc`.

**Bản mới nhất V7: [Tables_bai_bao_LiDAR_v7_SpeedDetail.xlsx](../benchmark_T4_20261018/Tables_bai_bao_LiDAR_v7_SpeedDetail.xlsx)**, thêm speed 8 model ở B10, stage statistics ở B11 và phạm vi encoder/preprocess/network/E2E ở B12. Các bảng mAP/mAP9, khoảng cách và augmentation cũ được giữ nguyên. [File riêng speed](../benchmark_T4_20261018/Speed_chi_tiet_8_models_v7.xlsx).

**Bản mới nhất V6: [Tables_bai_bao_LiDAR_v6_mAP_mAP9.xlsx](../benchmark_T4_20261018/Tables_bai_bao_LiDAR_v6_mAP_mAP9.xlsx)**, mở sẵn bảng tóm tắt có tên cột mAP/mAP9 rõ ràng cạnh model, cùng tham số và speed. Giữ đầy đủ bảng độ khó, khoảng cách và augmentation trước đó.

**Bản mới nhất V5: [Tables_bai_bao_LiDAR_v5_DistanceAP.xlsx](../benchmark_T4_20261018/Tables_bai_bao_LiDAR_v5_DistanceAP.xlsx)**, thêm AP Moderate theo khoảng cách 0–30 / 30–50 / 50–70,4 m tại `B08_AP_Distance` và phần B trang benchmark; `B09_Distance_Detail` có số GT/TP/FP/recall. Chỉ điền 12 AP đã có nguồn (Pedestrian/Cyclist của MobilePIXOR và PixorNeXt), ghi rõ các model/lớp chưa có range report. Giữ nguyên các bảng trước.

**Bản mới nhất V4: [Tables_bai_bao_LiDAR_v4_MobilePIXOR.xlsx](../benchmark_T4_20261018/Tables_bai_bao_LiDAR_v4_MobilePIXOR.xlsx)**, thêm MobilePIXOR thành dòng riêng, tổng 8 model; có cột GPU để phân biệt MobilePIXOR L4 với các run T4. [File riêng benchmark V4](../benchmark_T4_20261018/Benchmark_DAY_DU_v4_MobilePIXOR.xlsx).

**Bản mới nhất đầy đủ benchmark: [Tables_bai_bao_LiDAR_T4_v3_BenchmarkFull.xlsx](../benchmark_T4_20261018/Tables_bai_bao_LiDAR_T4_v3_BenchmarkFull.xlsx).** Mở đầu ở `T01_Benchmark`, đủ mAP theo độ khó, mAP9, speed và AP từng lớp; augmentation đã sửa vẫn giữ nguyên. [File chỉ chứa benchmark](../benchmark_T4_20261018/Benchmark_T4_DAY_DU.xlsx) có thể dùng để kiểm tra nhanh.

Bản cập nhật benchmark Tesla T4 ngày 18/10/2026 ở [Tables_bai_bao_LiDAR_T4_updated.xlsx](../benchmark_T4_20261018/Tables_bai_bao_LiDAR_T4_updated.xlsx), kèm [báo cáo HTML](../benchmark_T4_20261018/Bao_cao_so_sanh_T4.html). Bản cập nhật đã điền timing của PIXOR, PixorNeXt best AP và năm baseline; các ô reference AP còn thiếu của PixorNeXt được để trống. File bên dưới là bản trước cập nhật, được giữ nguyên để đối chiếu.

- `T01`–`T07`: 7 bảng chính dự kiến; `T01` là khung chờ benchmark, chưa có kết quả.
- `S01`–`S04`: 4 bảng phụ lục hỗ trợ.
- `99_Nguon`: đường dẫn và hash của nguồn số liệu.

Phạm vi đã thống nhất: không khảo sát độ rộng 24/32 hoặc 16/32; không có bảng ablation focal context riêng. Các loss ablation giữ focal context theo đúng cấu hình đã chạy và khai báo trong caption. IQA là thay đổi cả head, supervision và scoring, không chỉ tên loss.

Sheet `S02_Augmentation` đã sửa thành đúng ba nhóm: **G16** = baseline xoay/dịch/scale không GT sampling (51,30%); **O32** = GT sampler đối chứng, người dùng gọi `openpcdet_fake`, tên run có `Fake_opengt` (60,22%); **H32** = hybrid cải tiến (68,12%). O32 dùng sampler giản lược local, không mặc nhiên là OpenPCDet nguyên bản. **S32** (68,46%) là cấu hình hybrid khác kèm thay đổi neck/quota, được giữ trong archive. So sánh G16→O32 còn khác cấu hình mạng và global transforms; không diễn giải mức tăng như tác dụng riêng của GT sampling. Bản trước khi sửa được lưu tại `revisions/before_augmentation_mapping_fix_20261018/`.

Các bảng attention, efficiency và Final3D cũng khai báo focal khi thực nghiệm thực sự có module này. T06/T07 dùng checkpoint Final3D cũ có focal; không đại diện cho mô hình hiện tại không focal. Khi chốt bài chỉ về mô hình mới, cần quyết định giữ chúng làm kết quả tham khảo/phụ lục hay thay bằng report tương ứng của mô hình mới.

AP được ghi theo phần trăm, delta theo điểm phần trăm. Giá trị số giữ độ chính xác gốc trong ô và chỉ làm tròn khi hiển thị; delta tính trước khi làm tròn. Ô số liệu trống màu vàng là chưa có dữ liệu, không phải 0. Source ID tra tại `99_Nguon`. Các bảng cũ có khác biệt cấu hình được ghi giới hạn so sánh ngay trong sheet.

Nguồn chính là report/config đã lưu trong `paper/research`, không sử dụng các ô Excel gốc bị đổi thành ngày. Không huấn luyện hoặc chạy inference để tạo file này. File Excel gốc và bản thảo hiện tại được giữ nguyên.

Tạo lại workbook và kiểm tra số xuất ra bằng lệnh chạy từ thư mục gốc dự án:

```sh
python3 Result/paper_tables/build_tables.py
```

Cần Python và `openpyxl`. Lệnh sẽ ghi lại workbook cùng `verification.json`; hãy lưu một bản khác nếu đã chỉnh thủ công workbook. Verification kiểm tra việc mở lại file, mọi ô tiêu đề/dữ liệu đã xuất, AP, delta, timing/FPS, các ô benchmark trống và việc loại các run khảo sát độ rộng. Đây là kiểm tra nội dung/định dạng workbook bằng openpyxl; chưa có render bằng Microsoft Excel hoặc LibreOffice.
