# Phân tích ablation LiDAR — 2026-10-12

Nguồn: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts`. Đã đọc 21 run, 41 evaluation cùng metrics/config/selection. Phân tích dùng kết quả đã lưu; không chạy thêm training hay inference.

## Bảng nên dùng

Nên trình bày **7 bảng ablation**, kèm bảng tiến trình hệ thống, bảng accuracy/efficiency và bảng FP/recall. Các cột chính: Car/Pedestrian/Cyclist AP R40 Moderate, mAP Moderate, ΔmAP (điểm phần trăm, pp), số tham số, model ms và checkpoint epoch. Giữ đầy đủ 9 AP trong phụ lục/Raw_eval. Chọn cấu trúc bảng theo các phép đối chứng thực tế đã chạy.

| Bảng | Các run | Ý nghĩa / giới hạn |
|---|---|---|
| Encoding và heads | E8/E14/G16; O24G16/O24G32 | Encoding/head khác source; feature width có cặp matched |
| Augmentation | O32/H32; G16/O24G16 | Đo recipe/gói thay đổi; chưa tách mỗi flag |
| Neck và fusion width | H32/S32; O24G16/O32G16; O24G32/RC24 | Width có đối chứng; gates/RC còn trộn GT settings |
| OGA × IQA | P1/P2/P3/P4; P11 bổ sung | Thiết kế 2×2 hoàn chỉnh; nên là bảng trung tâm |
| Focal context | S32/P1 và P5/P4 | Lợi ích phụ thuộc loss/IQA |
| Local attention/detail | P4/P14/P15/P16 | Thêm riêng từng module; cùng source code liên quan |
| Checkpoint policy | AP-selected/loss-selected mỗi run | Độ nhạy theo checkpoint; giữ policy nhất quán |

## Kết quả chính

| Run | Biến thể | Car Mod | Ped Mod | Cyc Mod | mAP Mod | Detector params | Model ms |
|---|---|---:|---:|---:|---:|---:|---:|
| P1 | Focal / baseline loss / không IQA | 82.97 | 56.99 | 62.17 | 67.38 | 815,743 | 7.772 |
| P2 | Focal / OGA / không IQA | 84.09 | 57.93 | 64.88 | 68.97 | 815,743 | 7.827 |
| P3 | Focal / baseline loss / IQA | 84.13 | 59.45 | 67.34 | 70.31 | 852,929 | 8.260 |
| P4 | Focal / OGA / IQA | 84.87 | 59.63 | 67.24 | 70.58 | 852,929 | 8.186 |
| P11 | Focal / Q-OGA / không IQA | 84.89 | 57.15 | 65.56 | 69.20 | 815,743 | 7.464 |
| P14 | P4 + detail path | 84.30 | 59.70 | 66.04 | 70.01 | 854,401 | 8.243 |
| P15 | P4 + ECA | 83.64 | 59.77 | 66.29 | 69.90 | 852,933 | 8.510 |
| P16 | P4 + SimAM | 84.72 | 60.66 | 67.60 | 70.99 | 852,930 | 8.554 |

**P16 (SimAM) dẫn đầu AP trong 19 run đủ budget và cùng policy chọn AP: 70.9928%.** So với P4: mAP +0.4142 pp, Pedestrian +1.0265 pp, model forward +0.3687 ms. P4 nên giữ làm mốc ablation; P16 cần xác nhận thêm seed. Toàn detector thêm 1 tham số ở residual scaling, nhưng SimAM vẫn thêm tính toán.

**IQA cho gain lớn hơn OGA trong các cặp này.** OGA khi IQA tắt: +1.5884 pp; khi IQA bật: +0.2695 pp. IQA với baseline loss: +2.9305 pp; với OGA: +1.6115 pp. Tương tác OGA×IQA = P4−P2−P3+P1 = **-1.3190 pp**. Lợi ích không cộng tuyến tính; đây là quan sát một seed, chưa có kiểm định tương tác thống kê. P3 chỉ kém P4 0.2695 pp, nên giữ baseline+IQA làm đối chứng mạnh.

**Focal phụ thuộc loss/IQA.** Với baseline không IQA, S32→P1: -1.0813 pp và model latency +0.8358 ms. Với OGA+IQA, P5→P4: +1.8561 pp. Còn thiếu no-context OGA-only và baseline+IQA để tách đầy đủ tương tác loss/IQA/context.

**Detail và ECA chưa tăng AP ở seed 42.** P14 so với P4: -0.5676 pp; P15: -0.6751 pp. P4/P14/P15/P16 khác commit nhưng đã xác minh `git diff b635f0e 54636593 -- detector tools/kitti_training_pipeline` rỗng. Các thay đổi source thuộc tài liệu/visualization. Dữ liệu hiện tại ủng hộ thử tiếp SimAM riêng.

**Q-OGA là đối chứng đáng giữ.** P11 đạt 69.2007% ở epoch 40: hơn P2 +0.2336 pp, kém P4 1.3779 pp và ít hơn 37.186 tham số. AP epoch 50 của P11 giảm còn 68.5288%, thấp hơn P2 epoch 50; không kết luận Q-OGA thắng ở mọi giai đoạn. P11/P2 cùng kiến trúc inference và số tham số; chênh model latency giữa hai session chưa chứng minh loss làm graph nhanh hơn.

**Hist14 so với Rich8:** E8→E14 +2.7688 pp; Pedestrian +6.8762 pp. Config chỉ khác encoding/backend nhưng source thay đổi rộng. Bảng này có ích để lựa chọn biểu diễn; cần ghi khác source khi trình bày kết luận.

**Grouped head:** E14→G16 +1.1532 pp; Cyclist +8.2006 pp nhưng Car/Ped giảm. Còn khác source, val batch và đường encode Hist14. **Feature width 16→32** ở O24G16→O24G32: +0.7243 pp, thêm 114,832 tham số. **Fusion width 24→32** khi feature width 16: -0.3087 pp. Width lớn hơn chưa chắc tốt hơn.

## AP, false positive và mục tiêu vận hành

| Run | mAP Mod % | TP | FP | Micro recall % | FP/frame | Offline FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| P2 | 68.97 | 9,047 | 150,412 | 89.07 | 39.91 | 17.74 |
| P3 | 70.31 | 9,004 | 96,714 | 88.65 | 25.66 | 21.98 |
| P4 | 70.58 | 9,081 | 140,340 | 89.41 | 37.24 | 18.77 |
| P11 | 69.20 | 9,066 | 88,206 | 89.26 | 23.40 | 23.17 |
| P16 | 70.99 | 9,112 | 149,111 | 89.71 | 39.56 | 17.60 |

**P16 dẫn đầu AP; P11 có FP thấp và pipeline nhanh hơn P2/P4/P16 ở ngưỡng đã lưu.** So với P2, P11 giảm FP 41.36% và thêm 19 TP. Các counts là Moderate tại score threshold 0.05, sau xử lý ignored objects của evaluator local. IQA thay đổi thang score, nên cùng threshold chưa đảm bảo cùng recall. Cần PR/FROC hoặc FP/frame tại cùng recall từng lớp trước khi kết luận giảm FP tổng quát. AP đo thứ tự score qua đường precision–recall; AP cao hơn vẫn có thể đi kèm nhiều FP ở phần score thấp.

## Các bảng phải ghi rõ biến bị trộn

- **SG-FPN gates:** H32→S32 +0.3358 pp nhưng GT quota Car/Ped/Cyc cũng đổi 6/8/10→15/10/10 và val batch 16→32. Cần FPN không gate cùng quota 15/10/10 để đo gates riêng.
- **RC-SGFPN:** O24G32→RC24 còn bật density/ground/static collision/visibility và đổi val batch. S32→RC24 còn đổi fusion width 32→24. Chưa có cặp chỉ đổi neck.
- **Sampler giản lược→hybrid:** O32→H32 +7.9062 pp nhưng đổi flags, probability 1→0.5, quotas và attempts. `Fake_opengt` vẫn gọi `hybrid_gt_sampling`; tên trong bảng là sampler giản lược. Chưa có bằng chứng đây là đối chứng implementation OpenPCDet gốc.
- **Standard→GT sampling:** G16→O24G16 còn bật SG gates và đổi chuỗi global augmentation. Dùng làm so sánh pipeline, chưa tách riêng GT sampling.
- **Binary35 baseline:** B0 archive mới có epoch 38/100; file tên AP thực tế dùng checkpoint chọn theo loss ở epoch 30. B1 chạy 100 epoch, depth 2/4/2 và chỉ có evaluation loss ở epoch 13. Giữ hai run này làm bối cảnh lịch sử, ngoài ranking 50 epoch/AP-selected.

## Checkpoint, latency và lỗi theo khoảng cách

P4: AP-selected epoch 50 đạt 70.5786%; loss-selected epoch 25 đạt 68.9218%. Chênh +1.6568 pp; P16 chênh +1.1667 pp. **RC24 loss-selected epoch 33 đạt 68.2580%, cao hơn AP-selected epoch 50 đạt 67.7855%.** AP chỉ được đo cách 10 epoch. Best AP nghĩa là tốt nhất trong các epoch đã đo. Bảng chính giữ policy nhất quán; bảng checkpoint trình bày độ nhạy thay vì lấy max(AP,loss) riêng cho từng run.

P4 model forward 8.186 ms, offline pipeline 53.277 ms. P16 tương ứng 8.554/56.828 ms. Offline gồm load bin, voxelization, H2D, model và decode/NMS; các stage ngoài model chiếm phần lớn thời gian. Artifact đo PyTorch FP32/NVIDIA L4. Cần benchmark lặp trên cùng hardware/backend/precision nếu claim tốc độ edge; GFLOPs chưa có trong archive.

AP theo khoảng cách có sẵn cho Pedestrian/Cyclist. P16 so với P4 ở Pedestrian 30–50 m tăng 5.2233 pp (137 GT, TP 66→71); AP vẫn thấp: 4.5442→9.7675%. Đây là vùng nên kiểm tra lỗi. Vùng 50–70.4 m chỉ có 13 Pedestrian và 17 Cyclist GT; giữ sample count cạnh AP và tránh lấy vùng ít mẫu làm kết luận chính. Không cộng/trung bình AP các bin để suy ra AP tổng.

## Protocol và khả năng tái lập

Toàn bộ 41 evaluation cùng validation split SHA256 và cùng hash tập asset label/calib/LiDAR; ground truth counts khớp. Log ghi 3.712 train/3.769 val. Archive không chứa audit GT database/train manifest độc lập để kiểm tra lại train/val disjoint. Tất cả seed 42; chưa có mean±std hoặc kiểm định thống kê. Workbook giữ 9 AP, mAP Moderate, epoch, config, hashes, source commit, GPU, backend và precision.

Metric là **local loader-aligned KITTI-style rotated BEV AP R40**. Khi so với paper khác cần khớp ROI, difficulty, neighbor ignores và evaluator. AP3D cần checkpoint có nhánh z/height đã train theo workflow project; các run này phục vụ BEV ablation hiện tại.

## Việc nên làm tiếp

1. Nếu chọn P16 làm model cuối, thêm đối chứng baseline loss + focal + IQA + SimAM, chỉ đổi loss.name. P3 chưa tách OGA trên kiến trúc cuối vì thiếu SimAM.
2. Lặp seed cho candidate và đối chứng được chọn; nhánh Q-OGA giữ P11/P2. Không cần lặp tất cả run.
3. Đánh giá FP/frame tại cùng recall từng lớp, benchmark lại latency và evaluate IQA alpha 0/0.5 trên cùng checkpoint. Đổi alpha cần inference lại; prediction cache đã score không đủ.
4. Nếu claim neck, bổ sung matched gates và RC-neck như ghi trong bảng. Nếu claim từng term loss, cần ablation các term đó; hiện các cặp đo whole loss strategy.

## Cách sử dụng các file

- `ablation_analysis.xlsx`: 20 sheet; filter, công thức có giá trị lưu sẵn, biểu đồ và dropdown chọn hai run. Bắt đầu ở `Tong_quan` hoặc `So_sanh_2_run`.
- `runs_summary.csv`, `pairwise_effects.csv`: UTF-8 BOM, có thể import vào Google Sheets.
- `ablation_overview.png`, `accuracy_latency.png`: hình xuất 200 dpi để đọc và chia sẻ.
- `evidence_manifest.json`: SHA256 file nguồn và bằng chứng source diff. `build_analysis.py` tái tạo báo cáo từ artifact gốc.
- Các ID B0/B1/E8/E14/G16/H32/S32/O* là nhãn riêng của workbook này; tên thư mục đầy đủ ở `Danh_sach_runs` để đối chiếu.
