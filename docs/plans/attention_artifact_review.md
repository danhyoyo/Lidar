# Đánh giá các thí nghiệm LiDAR đã chạy

Ngày kiểm tra: 2026-10-05. Nguồn:
`/home/duyennh/Downloads/lidar_training_artifacts-20261004T121553Z-1-001/lidar_training_artifacts`.

Đã đọc config, metadata, training metrics, selection và evaluation JSON của
20 run. Bảng đầy đủ nằm trong `attention_artifact_results.csv`; các con số dưới
đây là kết quả được lưu, không phải benchmark mới. Không chạy lại training.

## Run nên giữ

Tất cả tên rút gọn dưới đây thuộc MobilePixorNeXt, rich8, standard augmentation,
seed 42, 100 epoch. AP là local KITTI-style rotated BEV AP R40 Moderate (%).
Latency là model forward PyTorch FP32 trên NVIDIA L4, chưa bao gồm pipeline.
`best` được chọn theo validation loss; `last` là endpoint epoch 100.

| Run rút gọn | mAP best | mAP last | Model ms best / last | Vai trò |
| --- | ---: | ---: | ---: | --- |
| baseline_loss + conv thuần | 85.7284 | Chưa evaluate | 4.881 / — | Mốc conv cũ; neck/head khác baseline attention |
| baseline_loss + LiteMLA + SG-FPN | 87.8798 | Chưa evaluate | 7.176 / — | Mốc chính để nghiên cứu attention |
| baseline_loss + MS LiteMLA + SG-FPN | 88.9997 | 89.0471 | 7.724 / 7.551 | Đối thủ attention giàu biểu diễn hơn |
| baseline_loss + LiteMLA + SG-FPN + IQA | 91.1473 | 91.1235 | 7.214 / 7.213 | Mốc chất lượng cao nhất trong folder |
| q_oga_loss + LiteMLA + SG-FPN | 86.7456 | 90.2401 | 6.913 / 6.834 | Mốc phụ nếu nghiên cứu cùng Q-OGA |
| q_oga_loss + MS LiteMLA + SG-FPN | 88.1914 | 89.7727 | 7.496 / 7.329 | MS không tăng AP tổng ở endpoint này |

Run IQA thay đổi quality head, loss và quality-aware postprocessing
(`nms_alpha=0.5`, các run không IQA là null). Vì vậy 91.12% là chất lượng của
cả hệ thống IQA, không phải bằng chứng riêng về attention. Có thể dùng làm
mốc mạnh sau khi hoàn thành đối chứng attention đơn giản.

## Chưa có đối chứng chỉ khác attention

Hai run thường dễ bị so sánh nhầm:

| Trường model | baseline_loss-rich8-baseline_iou-s42 | baseline_loss-rich8-baseline_iou-sgfpn-s42 |
| --- | --- | --- |
| c4_attention | none | litemla |
| scale_gated_fpn | false | true |
| header_use_bn | false | true |
| header_act | none | silu |

Chênh lệch +2.1514 điểm mAP best và +2.2947 ms gồm cả attention, neck và
head. Không thể dùng chúng để định lượng lợi ích/chi phí riêng của LiteMLA.
Hai run này cùng source commit `4d331cb518d416a0184c2920621223eaa0f2655f`.

MS LiteMLA còn đổi scales [5] → [3,5] và Q/K normalization none → RMSNorm.
So với baseline loss thường, best tăng +1.1199 điểm mAP, Pedestrian tăng
+2.9942 điểm, model time tăng khoảng 7.63%. Tuy nhiên source commit và
training compile setting cũng khác. Với Q-OGA, last MS thấp hơn thường
0.4674 điểm mAP và chậm hơn khoảng 7.24%; config loss và source commit
cũng có khác biệt. Đây là kết quả thăm dò, chưa phải ablation một biến.

## Những gì dữ liệu hiện tại nói về bài toán

Car đã đạt khoảng 96–97% AP, trong khi Pedestrian/Cyclist còn nhiều khoảng
trống. Nên phân tích lỗi hai lớp này trước khi quyết định thiết kế context.

So sánh gói conv cũ với gói LiteMLA + SG-FPN + BN/SiLU ở checkpoint best:

| Lớp / khoảng cách | Conv cũ AP | Gói LiteMLA AP | Số GT Moderate |
| --- | ---: | ---: | ---: |
| Pedestrian 0–30m | 76.0002 | 79.7783 | 611 |
| Pedestrian 30–50m | 64.8737 | 64.2437 | 97 |
| Cyclist 0–30m | 86.7536 | 93.8661 | 149 |
| Cyclist 30–50m | 82.7748 | 82.1973 | 64 |

Gói cải tiến hiện tăng mạnh ở vùng gần; chưa chứng minh giả thuyết attention
giúp vùng xa. Hơn nữa đây vẫn là so sánh nhiều thay đổi cùng lúc. Vùng
50–70.4m chỉ có 10 Pedestrian và 3 Cyclist Moderate: không dùng AP rất cao
hoặc biến động tại đây làm kết luận chính. Point density và occlusion cần
phân tích thêm từ raw data/GT, không suy ra từ khoảng cách đơn thuần.

## Các điều kiện phải khớp khi tái sử dụng

- Tất cả 20 run kết thúc 100 epoch, seed 42; validation có 1.497 frame.
  Log baseline ghi 5.984 train / 1.497 validation.
- Split validation của mọi evaluation có SHA256
  `cae7f7b5a72f28af771ebe878faa97b81bdafbcecd2be51205fd8cb5ecd2733a`.
  Split hiện tại trong workspace là 3.712 / 3.769, nên không so AP trực tiếp.
- Đã tìm lại manifest cũ trong git commit của các run; hash validation khớp
  evaluation. Bản sao nằm ở `splits/kitti/archived_5984_1497/`. Các manifest
  đang dùng trong workspace được giữ nguyên.
- Config một số run ghi physical batch 32, nhưng `run.json` và train log
  ghi batch thực tế 16, accumulation 1. Tái chạy phải dùng batch thực tế.
- Lịch cũ là 100 epoch/warmup 8; cặp diagnostic chuẩn bị trước là
  50 epoch/warmup 4. Cặp 50 epoch có thể so với nhau, không so với archive.
- Training compile setting và source commit khác giữa một số run. Muốn
  tận dụng checkpoint cũ cho ablation, cần dùng cùng source và training recipe.
- Chỉ run MobilePIXOR baseline đầu tiên đo trên T4; các run còn lại trên L4.
  Không dùng hai GPU khác nhau để so tốc độ kiến trúc.
- Forward khoảng 7ms không đồng nghĩa toàn pipeline đạt khoảng 140 FPS:
  LiteMLA baseline có input-to-detections mean khoảng 44.06ms, tức 22.70 FPS
  trong pipeline offline được evaluator đo.
- Metric này là BEV AP trên validation địa phương, không phải official KITTI
  hidden-test/3D AP; một seed chưa đủ chứng minh kết quả ổn định hoặc SOTA.

## Việc nên làm tiếp, theo thứ tự

1. Evaluate `checkpoints/last.pt` của hai run conv cũ và LiteMLA baseline;
   checkpoint đã có, không cần train lại để bổ sung endpoint. Dùng đúng
   config/kiến trúc từng run và manifest validation cũ.
2. Tạo **một run conv mới giữ SG-FPN, BN và SiLU** như LiteMLA baseline,
   chỉ đổi `c4_attention=none`. Dùng source commit của baseline, manifest
   5.984/1.497, physical batch16, BF16, seed42, 100 epoch/warmup8 và cùng
   training settings. So last với last; best với best là so sánh phụ.
   Nếu không tái lập được môi trường/source cũ, chạy lại cả cặp trong cùng
   môi trường; không ghép một run mới với archive để gọi là đối chứng sạch.
3. Dùng cặp khớp để xem gain theo lớp, distance, density, occlusion và các
   frame cụ thể. Quyết định có làm selective global context dựa trên lỗi
   quan sát được; chưa cần thêm attention mới hàng loạt.
4. Nếu attention có gain/chi phí đáng nghiên cứu, thêm attention giản lược
   hoặc context C5 làm đối thủ; routing phải so với random và heuristic
   dưới cùng budget, đo thời gian toàn module và toàn model.
5. Xác nhận phương án tốt với nhiều seed, dataset thứ hai và baseline mạnh
   IQA. Kết quả archive phù hợp để chọn mốc và loại hướng kém, chưa đủ
   chứng minh tính mới cho bài báo Q1/Q2.

