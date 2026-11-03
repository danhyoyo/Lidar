# Q-OGA, OGA và IQA: AP, false positive, recall và latency

**P11 là lựa chọn Q-OGA có lợi thế rõ về FP ở ngưỡng đang dùng và tốc độ pipeline; P16 vẫn là lựa chọn AP cao nhất.**
Nhận định trước nghiêng về AP đã đánh giá chưa đầy đủ lợi thế vận hành của Q-OGA. Cần trình bày cả hai lựa chọn trong bài báo.

Nguồn: evaluation của checkpoint best-AP và cache prediction NPZ sau NMS, 3.769 frame; local KITTI-style rotated BEV AP R40; Car IoU 0,7, Pedestrian/Cyclist IoU 0,5.
Score threshold 0,05, NMS threshold 0,1; seed 42, cùng train 50 epoch. P11 chọn epoch 40 theo best-AP; các run còn lại ở bảng chọn epoch 50.
Latency lấy từ artifacts: NVIDIA L4, PyTorch FP32, từng frame; loại 10 warmup frame. Đây là phép đo đã lưu.
FP/TP theo evaluator local, sau khi xử lý ignored objects; precision vận hành = TP/(TP+FP). Micro recall = tổng TP / tổng GT 10.157, có trọng số theo lượng GT; macro recall là trung bình ba lớp.

## 1. So sánh chính

| Run | Cấu hình | mAP Mod | AP9 | FP | TP | Micro recall % | Precision % | FP/frame | Hz pipeline | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P01 | Baseline + focal | 67.379 | 69.460 | 106,231 | 8,924 | 87.861 | 7.750 | 28.185 | 21.31 | 815,743 |
| P02 | OGA + focal | 68.967 | 71.117 | 150,412 | 9,047 | 89.072 | 5.674 | 39.908 | 17.74 | 815,743 |
| P03 | Baseline + focal + IQA | 70.309 | 72.579 | 96,714 | 9,004 | 88.648 | 8.517 | 25.660 | 21.98 | 852,929 |
| P04 | OGA + focal + IQA | 70.579 | 72.660 | 140,340 | 9,081 | 89.406 | 6.077 | 37.235 | 18.77 | 852,929 |
| P11 | Q-OGA + focal | 69.201 | 71.274 | 88,206 | 9,066 | 89.259 | 9.320 | 23.403 | 23.17 | 815,743 |
| P16 | P04 + SimAM | 70.993 | 73.173 | 149,111 | 9,112 | 89.712 | 5.759 | 39.562 | 17.60 | 852,930 |

P11 so với P02: FP giảm 41,3571%, TP tăng 19; micro recall tăng 0,1871 điểm. mAP tăng 0,2336 điểm và AP9 tăng 0,1574 điểm.
Hz pipeline tăng 30,6261%; mean latency giảm 23,4456%. Cùng 815.743 detector parameters; các tham số criterion chỉ dùng lúc training.
Đây là lợi thế ở ngưỡng score đã lưu. Chênh AP nhỏ ở một seed chưa chứng minh tương đương thống kê; chưa có đối chứng FP ở cùng recall.

## 2. Từng lớp: P02 → P11

| Lớp | AP Mod OGA | AP Mod Q | Δ AP | FP OGA | FP Q | FP giảm % | Recall OGA % | Recall Q % | Δ TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Car | 84.087 | 84.893 | +0.806 | 70,102 | 45,725 | 34.77 | 92.710 | 93.015 | 24 |
| Pedestrian | 57.934 | 57.145 | -0.788 | 53,310 | 30,164 | 43.42 | 74.986 | 74.813 | -3 |
| Cyclist | 64.881 | 65.564 | +0.683 | 27,000 | 12,317 | 54.38 | 81.434 | 81.066 | -2 |

Car: Q-OGA tăng AP và recall, giảm FP. Pedestrian: giảm FP mạnh nhưng AP giảm 0,7884 điểm; giảm 3 TP. Cyclist: AP tăng 0,6827 điểm, FP giảm 54,38%, giảm 2 TP. Q-OGA có lợi thế rõ, nhưng không thắng mọi metric ở mọi lớp.

## 3. Tăng tốc nằm ở đâu?

| Thành phần (ms/frame) | P02 OGA | P11 Q-OGA | P04 OGA+IQA |
| --- | --- | --- | --- |
| Preprocess | 12.027 | 11.543 | 11.623 |
| H2D | 6.967 | 6.672 | 6.664 |
| Model | 7.827 | 7.464 | 8.186 |
| Decode/NMS | 29.243 | 17.234 | 26.553 |
| Toàn pipeline | 56.380 | 43.162 | 53.277 |
| Pipeline p95 | 95.821 | 64.817 | 89.585 |

Decode/NMS tiết kiệm 12,0092 ms, chiếm 90,85% của tổng 13,2188 ms giảm mean pipeline latency. Số detection sau NMS giảm 37,51%: 44,07 → 27,54/frame.
Điều này phù hợp với việc phân bố score thay đổi làm giảm chi phí decode/NMS; cache chỉ lưu kết quả sau NMS, nên chưa đo trực tiếp số candidate trước NMS.
Loss không được tính lúc inference. P02/P11 cùng graph: chênh model 7,827 → 7,464 ms trong một lần đo không chứng minh Q-OGA có graph nhanh hơn. Chênh preprocess/H2D không thể quy cho loss. Các stage không cộng đúng hoàn toàn với thời gian tổng vì còn overhead đo và điều phối.

## 4. IQA giúp gì, và có ít FP hơn Q-OGA không?

P02 → P04: mAP +1,6115 điểm; FP giảm 6,70% (150.412 → 140.340); TP +34. IQA giảm FP so với OGA không IQA ở ngưỡng 0,05.
P11 → P04: mAP +1,3779 điểm, nhưng FP tăng 59,10% (88.206 → 140.340), TP chỉ +15. Vì vậy, tổng FP thấp hơn không giải thích được lợi thế AP của P04 trước P11.

| Lớp | P11 AP Mod | P04 AP Mod | P04−P11 AP | P11 FP | P04 FP | P04−P11 TP |
| --- | --- | --- | --- | --- | --- | --- |
| Car | 84.893 | 84.866 | -0.027 | 45,725 | 63,679 | -17 |
| Pedestrian | 57.145 | 59.632 | +2.486 | 30,164 | 50,917 | 31 |
| Cyclist | 65.564 | 67.239 | +1.675 | 12,317 | 25,744 | 1 |

AP tính precision envelope theo recall qua thứ tự score; tổng FP tính tại ngưỡng score >0,05. AP có thể cao hơn dù nhiều detection sai nằm ở cuối bảng score. P04 hơn Q-OGA chủ yếu ở Pedestrian và Cyclist; Pedestrian tăng 31 TP. Chưa có match flags theo score để quy riêng lợi thế AP cho chất lượng box hay thứ tự score.
IQA vừa thêm head/supervision, vừa đổi score dùng chọn peak và NMS: `s = p_cls^(1-alpha) * p_iou^alpha`, alpha = 0,5. Candidate cần cả `s > threshold` và `p_cls > threshold`. Các run hiện tại chưa tách riêng supervision với scoring.
IQA thêm 37.186 params (+4,56% trên P02). Evaluate cùng checkpoint P04 với alpha = 0 và 0,5 sẽ tách tác động scoring lúc inference; đối chứng này không cần train lại.

## 5. Kiểm tra phân bố score từ cache

| Run | >0.05 | >0.10 | >0.20 | >0.30 | >0.50 | >0.70 |
| --- | --- | --- | --- | --- | --- | --- |
| P02 | 166,084 | 72,191 | 35,125 | 24,615 | 16,117 | 10,001 |
| P11 | 103,794 | 48,456 | 26,865 | 20,386 | 13,783 | 7,582 |
| P03 | 112,119 | 112,119 | 94,529 | 43,585 | 21,814 | 14,537 |
| P04 | 156,129 | 156,129 | 137,068 | 57,495 | 24,024 | 14,104 |
| P16 | 164,856 | 164,856 | 144,471 | 60,141 | 24,546 | 14,629 |

Các số này là detection count sau NMS, không phải FP count ở ngưỡng mới. NPZ không chứa TP/FP match flags; raw GT không có trong workspace để tái-match. Lọc cache không đo được tốc độ mới hoặc đổi NMS alpha/peaks; điều đó cần inference lại.
Score P04 nhỏ nhất là 0,13438, median 0,25875; P11 nhỏ nhất khoảng 0,05, median 0,09338. Lọc cache theo score đã lưu >0,10 không loại detection nào của P04, nhưng lọc gần một nửa detection P11. Đây không phải inference lại ở threshold 0,10: decoder IQA còn kiểm tra classification score riêng, trong khi cache chỉ lưu score kết hợp. Chênh lệch phân bố score phải được tính đến khi chọn ngưỡng vận hành.
Trong code, IQA dùng BCE trên các ô regression positive, không có supervision quality trực tiếp ở background. Score kết hợp có thể tăng so với classification score nếu `p_iou > p_cls`: ví dụ sqrt(0,06 × 0,6) ≈ 0,190. Điều này giải thích khả năng đổi thang score; cache chỉ lưu score kết hợp, nên chưa xác nhận riêng quality score ở background trong các run này.
Ngưỡng 0,05 mô tả đúng cấu hình đang dùng. Để kết luận FP suppression ở cùng khả năng phát hiện, cần đường PR/FROC hoặc FP/frame tại cùng recall từng lớp. Recall Pedestrian/Cyclist của Q-OGA hơi thấp hơn P02; giữ thông tin này khi báo cáo.

## 6. Tách từng thành phần hệ thống

| Cặp | Thành phần | Δ mAP | Δ TP | Δ FP % | Δ Hz % |
| --- | --- | --- | --- | --- | --- |
| B0→P01 | Focal / baseline loss, no IQA | -1.081 | -13 | -0.69 | -2.99 |
| P01→P02 | Whole OGA / focal, no IQA | +1.588 | +123 | +41.59 | -16.77 |
| P01→P03 | IQA training + scoring / baseline loss | +2.930 | +80 | -8.96 | +3.12 |
| P02→P04 | IQA training + scoring / OGA | +1.612 | +34 | -6.70 | +5.82 |
| P03→P04 | Whole OGA / focal + IQA | +0.269 | +77 | +45.11 | -14.59 |
| P02→P11 | Whole Q-OGA versus OGA / focal, no IQA | +0.234 | +19 | -41.36 | +30.63 |
| P01→P11 | Whole Q-OGA versus baseline / focal, no IQA | +1.822 | +142 | -16.97 | +8.71 |
| P05→P04 | Focal / OGA + IQA | +1.856 | -13 | -9.06 | +9.97 |
| P04→P14 | DETAIL_PATH / OGA + IQA + focal | -0.568 | +11 | +1.76 | +0.29 |
| P04→P15 | ECA / OGA + IQA + focal | -0.675 | +16 | +8.26 | -7.92 |
| P04→P16 | SimAM / OGA + IQA + focal | +0.414 | +31 | +6.25 | -6.25 |

OGA so baseline tăng AP/TP nhưng tăng FP khoảng 41,59% khi IQA off và 45,11% khi IQA on. Các run đo toàn bộ OGA (L1 → SmoothL1, geometry, adaptive weighting). IQA cải thiện AP mạnh, còn tổng FP của OGA + IQA vẫn cao hơn Q-OGA.
Q-OGA so với baseline cùng focal (P01 → P11) tăng AP 1,822 điểm, tăng 142 TP và giảm FP 16,97%. Đây là đối chứng thuận lợi cho proposed loss ngay cả khi không dùng OGA làm mốc.
Focal riêng làm AP baseline thấp hơn, nhưng focal + OGA + IQA cao hơn bản không focal 1,856 điểm, FP giảm 9,06%. Tác dụng focal phụ thuộc recipe.
DETAIL_PATH/ECA giảm AP và tăng FP trên P04. SimAM tăng AP 0,4142 điểm và TP 31, nhưng FP tăng 6,25% và pipeline Hz giảm 6,25%. Lợi ích quan sát được là AP/recall, chưa có lợi ích giảm FP.

## 7. Các thành phần bên trong Q-OGA: code khác gì?

- Quality-coupled focal classification: positive target `q^beta` (beta = 1 ở P11), q MGIoU detached; OGA thông thường dùng target 1. Nhánh negative giữ cùng dạng Gaussian modified focal. Đây là cơ chế phù hợp với việc gắn score vào chất lượng box, nhưng chưa có ablation QCFA riêng để quy toàn bộ giảm FP cho nó.
- Range-adaptive coordinate weighting: `w(r) = 1 + 1.5*(r/70.4)^2` (normalized r clamp tối đa 1,5); nhân SmoothL1 offset/size/yaw; không nhân geometry. Code dùng khoảng cách, không đo mật độ điểm trực tiếp.
- Smooth pi-symmetric corner distance: Q-OGA dùng soft-min logsumexp tau = 0,05; OGA dùng hard minimum hai đối xứng. Cùng chuẩn hóa theo kích thước và multi-axis projection GIoU.
- Adaptive task weighting: cả OGA (không IQA) và Q-OGA có 5 task và TemperatureSoftmaxUncertainty; OGA + IQA có 6 task. Đây là thành phần chung của hai loss.
- Chi tiết normalization: OGA SmoothL1 chia số positive-coordinate; Q-OGA cộng 2 tọa độ rồi trung bình theo positive, có thêm range weight. Khi weight = 1, raw coordinate loss Q-OGA ≈ 2× OGA cho cùng tensor. Sau đó còn EMA/task weighting, nên không suy ra đóng góp cuối của regression gấp đôi. P02/P11 đo toàn bộ objective, không riêng quality target.

P11 dùng `quality_target=mgiou` mặc định, không có quality curriculum. Các lợi thế đã đo thuộc toàn bộ Q-OGA. Chưa có internal ablation để định lượng riêng QCFA/RDA/soft-min.
Code đối chiếu: [quality focal](/home/duyennh/AI_projects/research_lidar/Lidar/detector/core/losses/quality_focal_loss.py:43), [Q-OGA](/home/duyennh/AI_projects/research_lidar/Lidar/detector/core/losses/strategies/q_oga.py:138), [OGA/IQA](/home/duyennh/AI_projects/research_lidar/Lidar/detector/core/losses/strategies/oga.py:99), [scoring](/home/duyennh/AI_projects/research_lidar/Lidar/detector/postprocess.py:169).

## 8. Theo khoảng cách: không claim RDA cải thiện mọi vùng xa

| Lớp | Vùng | GT | P02 AP | P11 AP | Δ AP |
| --- | --- | --- | --- | --- | --- |
| Pedestrian | 0_30m | 1589 | 63.004 | 61.696 | -1.307 |
| Pedestrian | 30_50m | 137 | 5.122 | 7.824 | +2.702 |
| Pedestrian | 50_70.4m | 13 | 0.322 | 0.769 | +0.448 |
| Cyclist | 0_30m | 346 | 74.414 | 73.879 | -0.535 |
| Cyclist | 30_50m | 181 | 46.164 | 48.706 | +2.542 |
| Cyclist | 50_70.4m | 17 | 14.724 | 9.387 | -5.337 |

Q-OGA cao hơn ở 30–50 m cho cả Pedestrian/Cyclist, nhưng Cyclist 50–70,4 m thấp hơn. Vùng xa có 13 Pedestrian và 17 Cyclist GT. Chưa tách RDA nên chưa quy gain vùng giữa cho RDA riêng.

## 9. Khuyến nghị cập nhật

- Ưu tiên AP: P16. Ưu tiên Q-OGA với FP thấp và pipeline nhanh trên recipe đã đo: P11. Giữ P03 làm đối chứng vì baseline + IQA đạt 70,309 AP / 21,976 Hz và FP 96.714.
- Q-OGA đáng giữ là candidate proposed loss về sự cân bằng AP, FP và tốc độ; OGA làm đối chứng. Trình bày IQA như một nhánh chất lượng tường minh, và Q-OGA như một nhánh gắn chất lượng vào classification, thay vì mặc định cả hai phải cùng bật trong một cấu hình.
- Trước khi thêm training, dùng cùng checkpoint và validation GT để sweep score threshold theo từng lớp, lập PR/FROC và so FP tại cùng recall; benchmark lại inference cùng môi trường. Với P04, thêm alpha = 0 / 0,5 để tách scoring IQA. Cache hiện có chưa đủ để thực hiện các phép đo này trong workspace.
- Nếu từng term QCFA/RDA/soft-min là contribution riêng, cần internal ablation; nếu claim toàn bộ Q-OGA, mức ablation phụ thuộc claim. Multi-seed tập trung P11/P02 và nhánh model cuối/đối chứng đã chọn.

Các delta là quan sát seed 42. [Biểu đồ](qoga_operating_metrics.png) và CSV lưu đầy đủ số chính xác cùng 9 AP từng lớp.
