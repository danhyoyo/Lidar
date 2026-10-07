# So sánh OGA + IQA và Q-OGA từ các run seed 42

Ngày đọc artifact: 2026-10-05. Đọc trực tiếp config.resolved.json, run.json, selected/selection.json, metrics.jsonl và evaluation_validation_best/last.json; không chạy thêm training hoặc evaluation.

Nguồn artifact: /home/duyennh/Downloads/lidar_training_artifacts-20261004T121553Z-1-001/lidar_training_artifacts

## Kết luận thực dụng

Trong các run đã có, chọn OGA + IQA nếu ưu tiên AP Moderate/Pedestrian; chọn Q-OGA thuần nếu ưu tiên model gọn và latency thấp. Cặp chính: OGA + IQA đạt 90.93698% so với Q-OGA 90.24013%, chênh +0.69685 điểm mAP ở epoch 100. Đây là kết quả quan sát với một seed, phù hợp để chọn run sử dụng; không tách được lợi ích nhân quả của từng thành phần.

## Điều kiện của cặp chính

Cùng MobilePixorNeXt, LiteMLA thường, SG-FPN, rich8, BN/SiLU, không reparam, seed 42, 100 epoch/warmup 8, AdamW, LR 0.0007, BF16, batch thực tế 16 và accumulation 1. Config ghi batch 32 nhưng run.json ghi batch thực tế 16. Cùng validation 1.497 frame và split SHA256 cae7f7b5a72f28af771ebe878faa97b81bdafbcecd2be51205fd8cb5ecd2733a.

Metric: local KITTI-style rotated BEV AP R40, Car IoU 0.7; Pedestrian/Cyclist IoU 0.5. Cùng score threshold 0.05, NMS threshold 0.1. IQA dùng nms_alpha=0.5; Q-OGA không dùng IQA. Latency evaluation: PyTorch FP32 trên NVIDIA L4.

Khác source commit (OGA + IQA: 19252fb; Q-OGA: 02de86f) và compile khi training (false/true). Git diff xác nhận các file loss strategy, quality focal loss, uncertainty weighting và postprocess không thay đổi giữa hai commit; các thay đổi model chủ yếu thêm tùy chọn RC neck và xử lý đầu vào. Cặp này vẫn không phải thí nghiệm chỉ đổi một biến.

## AP tại checkpoint last (epoch 100)

| Lớp | OGA + IQA AP | Q-OGA AP | Delta AP | IQA recall % | Q-OGA recall % | IQA TP | Q-OGA TP | IQA FP | Q-OGA FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Car | 96.8373 | 96.8889 | -0.0517 | 99.00 | 99.00 | 3167 | 3167 | 1296 | 1158 |
| Pedestrian | 84.9546 | 82.8850 | +2.0695 | 87.88 | 87.05 | 631 | 625 | 447 | 394 |
| Cyclist | 91.0191 | 90.9464 | +0.0727 | 93.98 | 93.98 | 203 | 203 | 179 | 160 |

Lợi thế AP tập trung ở Pedestrian (+2.0695 điểm), với thêm 6 TP; Car và Cyclist gần ngang nhau. Q-OGA có ít FP hơn ở ngưỡng đã lưu cho cả ba lớp. AP cao hơn không nhất thiết đồng nghĩa tổng FP thấp hơn: AP còn phụ thuộc thứ tự score và đường precision-recall. Dữ liệu tổng hợp không đủ để kết luận cơ chế cải thiện riêng là regression hay NMS ranking.

## AP theo khoảng cách ở checkpoint last

| Lớp | Khoảng cách | GT | OGA + IQA AP | Q-OGA AP | Delta AP | IQA TP | Q-OGA TP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pedestrian | 0_30m | 611 | 85.8840 | 85.5817 | +0.3022 | 543 | 540 |
| Pedestrian | 30_50m | 97 | 75.1008 | 71.1365 | +3.9643 | 81 | 77 |
| Pedestrian | 50_70.4m | 10 | 39.1429 | 63.4921 | -24.3492 | 5 | 7 |
| Cyclist | 0_30m | 149 | 91.6874 | 93.7456 | -2.0582 | 140 | 142 |
| Cyclist | 30_50m | 64 | 89.2897 | 86.8898 | +2.3999 | 60 | 58 |
| Cyclist | 50_70.4m | 3 | 100.0000 | 100.0000 | +0.0000 | 3 | 3 |

OGA + IQA tốt hơn ở 30–50 m: Pedestrian +3.9643 điểm và Cyclist +2.3999 điểm. Q-OGA tốt hơn ở Cyclist 0–30 m. Vùng 50–70.4 m chỉ có 10 Pedestrian và 3 Cyclist GT; ghi nhận Q-OGA tốt hơn với Pedestrian ở đây nhưng không dùng vùng ít mẫu làm kết luận chính.

## Latency và số tham số

| Thời gian (ms) | OGA + IQA | Q-OGA | Delta |
| --- | --- | --- | --- |
| Model forward | 7.4465 | 6.8344 | +0.6121 |
| Decode/NMS | 5.8005 | 5.3617 | +0.4388 |
| Toàn pipeline offline | 44.9890 | 42.9819 | +2.0071 |

Số tham số: 697,786 vs 693,097; IQA thêm 4,689 tham số (~0.68%). Model forward IQA cao hơn khoảng 9%; toàn pipeline đo được cao hơn khoảng 4.67%. Chênh pipeline gồm preprocess/H2D, không quy toàn bộ 2.007 ms cho head IQA. Đây là thời gian đã lưu, không phải benchmark lặp mới.

## Checkpoint best không phải best theo AP

| Phương án | Best epoch | Best AP | Last epoch | Last AP | Last - best |
| --- | --- | --- | --- | --- | --- |
| OGA thuần | 42 | 85.5248 | 100 | 88.7248 | +3.2000 |
| OGA + IQA | 26 | 86.1280 | 100 | 90.9370 | +4.8090 |
| Q-OGA thuần | 42 | 86.7456 | 100 | 90.2401 | +3.4946 |
| Baseline + IQA | 97 | 91.1473 | 100 | 91.1235 | -0.0238 |

selection.json dùng minimum mean validation loss. OGA + IQA chọn epoch 26; Q-OGA chọn epoch 42. Tại các run này last có AP cao hơn đáng kể. Nếu dùng checkpoint để đánh giá chất lượng cuối quá trình, dùng last hiện có thay vì mặc định best.pt. Không so trực tiếp trị số loss giữa các strategy do công thức/task/chuẩn hóa khác nhau; không gọi last là checkpoint AP tốt nhất toàn bộ 100 epoch vì chưa evaluate AP mọi epoch.

## Các biến thể khác đã chạy

| Phương án | mAP last | Pedestrian AP | Cyclist AP | Model ms | Commit |
| --- | --- | --- | --- | --- | --- |
| OGA thuần | 88.7248 | 79.6710 | 89.8043 | 6.9937 | 02de86f |
| OGA + IQA | 90.9370 | 84.9546 | 91.0191 | 7.4465 | 19252fb |
| Q-OGA thuần | 90.2401 | 82.8850 | 90.9464 | 6.8344 | 02de86f |
| Q-OGA + reparam | 89.8090 | 82.0181 | 90.5034 | 6.8006 | 19252fb |
| Q-OGA + MS LiteMLA | 89.7727 | 82.1522 | 90.3225 | 7.3294 | 19252fb |
| Q-OGA rich12 | 89.8850 | 83.1771 | 89.7588 | 6.8514 | ba1876b |
| OGA + IQA + MS LiteMLA + reparam | 91.0079 | 83.1463 | 92.9345 | 7.3598 | ba1876b |
| Baseline + IQA | 91.1235 | 85.4397 | 91.2070 | 7.2129 | 02de86f |
| GW-QAL thuần | 89.3978 | 78.7516 | 92.6832 | 7.3423 | 02de86f |

Tên run oga_loss-rich8-iqa-sgfpn-reparam không ghi MS LiteMLA nhưng config có scales [3,5], Q/K RMSNorm; vì vậy label ở bảng đã nêu đầy đủ. Không dùng nó làm cặp chỉ khác IQA với Q-OGA reparam dùng scales [5]/none. Baseline + IQA đạt 91.12345% last và 91.14729% best, cao nhất theo mAP Moderate trong bảng này; nhờ vậy nên giữ baseline + IQA làm mốc thực nghiệm, chưa có cơ sở mặc định geometry loss phức tạp hơn tốt hơn.

## Nguồn cặp chính

- OGA + IQA: [evaluation last](/home/duyennh/Downloads/lidar_training_artifacts-20261004T121553Z-1-001/lidar_training_artifacts/mobilepixornext-standard_aug-oga_loss-rich8-iqa-sgfpn-s42/evaluation_validation_last.json), [config](/home/duyennh/Downloads/lidar_training_artifacts-20261004T121553Z-1-001/lidar_training_artifacts/mobilepixornext-standard_aug-oga_loss-rich8-iqa-sgfpn-s42/config.resolved.json), [selection](/home/duyennh/Downloads/lidar_training_artifacts-20261004T121553Z-1-001/lidar_training_artifacts/mobilepixornext-standard_aug-oga_loss-rich8-iqa-sgfpn-s42/selected/selection.json).

- Q-OGA thuần: [evaluation last](/home/duyennh/Downloads/lidar_training_artifacts-20261004T121553Z-1-001/lidar_training_artifacts/mobilepixornext-standard_aug-q_oga_loss-rich8-baseline_iou-sgfpn-s42/evaluation_validation_last.json), [config](/home/duyennh/Downloads/lidar_training_artifacts-20261004T121553Z-1-001/lidar_training_artifacts/mobilepixornext-standard_aug-q_oga_loss-rich8-baseline_iou-sgfpn-s42/config.resolved.json), [selection](/home/duyennh/Downloads/lidar_training_artifacts-20261004T121553Z-1-001/lidar_training_artifacts/mobilepixornext-standard_aug-q_oga_loss-rich8-baseline_iou-sgfpn-s42/selected/selection.json).

Số liệu đầy đủ của 18 evaluation: [oga_iqa_vs_qoga_results.csv](oga_iqa_vs_qoga_results.csv).
