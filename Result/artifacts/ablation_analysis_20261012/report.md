# Phân tích ablation ngày 2026-10-12

**Model có AP tốt nhất hiện tại: P16 — focal + OGA + IQA + SimAM, không DETAIL_PATH/reparam.**
mAP Moderate 70.992834%; Mean AP-9 73.173378%; 852,930 params.
Checkpoint: [best_ap.pt](/home/duyennh/Downloads/drive-download-20261011T184106Z-1-001/mobilepixornext-hybrid_gt_aug-oga_loss-hist14-iqa-sgfpn-ctx_focal_local_simam_f32_detail0_8f04a8fa523a-grouped_e02c0e499671-s42-P16-h32/selected/best_ap.pt) — epoch 50.

**Bổ sung sau khi xét FP/recall và latency:** P11 (Q-OGA) là candidate có FP thấp hơn và pipeline nhanh hơn P02/P04/P16 tại cấu hình đang dùng. P11 giảm FP 41,36% so P02, với TP +19 và pipeline Hz +30,63%. Khuyến nghị P16 ở đây áp dụng cho mục tiêu AP cao nhất. Xem [phân tích Q-OGA/OGA/IQA](qoga_detailed_analysis.md) và [lựa chọn theo mục tiêu](selection_by_objective.json).

Đã kiểm tra 21 run (9 mới, 12 cũ), 41 evaluation. Các run mới đều đủ 50 epoch, seed 42, lịch AP mỗi 10 epoch, cùng augmentation/split/recipe.
Mỗi evaluation dùng đủ 3.769 frame và cùng 11.307 asset hashes. Hash config, checkpoint, prediction NPZ, AP macro và epoch chọn checkpoint đã đối chiếu.
P14–P16 lấy P04 làm anchor. Commit khác nhau chỉ bổ sung visualization; detector/training pipeline/master config/splits không đổi giữa các commit này.

| Run | Cấu hình | mAP Mod | Mean AP-9 | Δ B0 (điểm) | Hz model | Hz toàn pipeline | Params | Epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | Baseline | 68.460 | 70.283 | +0.000 | 144.16 | 21.97 | 790,623 | 50 |
| P01 | Baseline + focal | 67.379 | 69.460 | -1.081 | 128.66 | 21.31 | 815,743 | 50 |
| P02 | Focal + OGA | 68.967 | 71.117 | +0.507 | 127.76 | 17.74 | 815,743 | 50 |
| P03 | Focal + baseline + IQA | 70.309 | 72.579 | +1.849 | 121.06 | 21.98 | 852,929 | 50 |
| P04 | Focal + OGA + IQA | 70.579 | 72.660 | +2.119 | 122.16 | 18.77 | 852,929 | 50 |
| P05 | OGA + IQA, no focal | 68.723 | 70.647 | +0.263 | 127.99 | 17.07 | 827,809 | 50 |
| P11 | Focal + Q-OGA, no IQA | 69.201 | 71.274 | +0.741 | 133.97 | 23.17 | 815,743 | 40 |
| P14 | P04 + detail | 70.011 | 72.032 | +1.551 | 121.32 | 18.82 | 854,401 | 50 |
| P15 | P04 + ECA | 69.904 | 72.024 | +1.444 | 117.50 | 17.28 | 852,933 | 50 |
| P16 | P04 + SimAM | 70.993 | 73.173 | +2.533 | 116.90 | 17.60 | 852,930 | 50 |

## Đối chứng từng thành phần

| Đối chứng | Thành phần/chiến lược | Δ mAP Mod | Δ Car Mod | Δ Ped Mod | Δ Cyc Mod |
| --- | --- | ---: | ---: | ---: | ---: |
| B0 → P01 | Focal / baseline loss, IQA off | -1.081 | -0.337 | -1.336 | -1.571 |
| P01 → P02 | OGA / focal, IQA off | +1.588 | +1.116 | +0.942 | +2.707 |
| P01 → P03 | IQA training + score calibration / focal, baseline loss | +2.930 | +1.160 | +2.462 | +5.169 |
| P03 → P04 | OGA / focal, IQA on | +0.269 | +0.735 | +0.177 | -0.104 |
| P02 → P04 | IQA training + score calibration / focal, OGA | +1.612 | +0.779 | +1.698 | +2.357 |
| P05 → P04 | Focal / OGA + IQA | +1.856 | +0.767 | +1.319 | +3.482 |
| P02 → P11 | Whole Q-OGA objective / focal, IQA off | +0.234 | +0.806 | -0.788 | +0.683 |
| P04 → P14 | Detail path / focal + OGA + IQA | -0.568 | -0.571 | +0.064 | -1.196 |
| P04 → P15 | ECA / focal + OGA + IQA | -0.675 | -1.222 | +0.140 | -0.944 |
| P04 → P16 | SimAM / focal + OGA + IQA | +0.414 | -0.142 | +1.027 | +0.358 |

- Focal không cải thiện baseline (P01−B0=−1.081 điểm), nhưng giúp OGA+IQA (P04−P05=+1.856). Giữ focal trong cấu hình cuối có OGA+IQA; không claim focal cải thiện mọi baseline.
- OGA cải thiện1.588 điểm khi IQA tắt, chỉ0.269 điểm khi IQA bật. P04−P03 tăng Car0.735/Ped0.177 nhưng giảm Cyclist0.104. Gain Mean AP-9 chỉ0.0817 điểm; proposed loss chưa có evidence mạnh về độ ổn định từ một seed.
- IQA tăng2.930 điểm dưới baseline và1.612 dưới OGA. Difference-in-differences OGA×IQA là -1.318960 điểm; hai gain không cộng tuyến tính. Đây là quan sát seed 42, không phải kiểm định tương tác thống kê.
- Các cặp bật IQA đồng thời thêm supervision head và đổi NMS alpha0→0.5; chúng đo cả IQA module/calibration. Có thể evaluate checkpoint IQA ở alpha0 để phân biệt supervision và score calibration mà không train lại.
- Q-OGA tăng0.2336 điểm so OGA cùng focal/no-IQA: Car+0.8065/Cyclist+0.6827, Pedestrian−0.7884. Q-OGA có lợi thế về FP tại ngưỡng đã lưu và tốc độ pipeline, nên giữ làm candidate proposed loss; OGA+IQA có AP cao hơn. P11 dùng MGIoU mặc định, không phải exact rotated IoU.
- Q-OGA chọn epoch 40 (69.2007%), trong khi epoch 50 chỉ68.5288%, thấp hơn OGA epoch 50 (68.9671%). Best-AP là đối chứng đúng theo protocol; thêm so sánh cùng epoch khi phân tích hội tụ, tránh nói Q-OGA thắng ở mọi giai đoạn.
- DETAIL_PATH giảm0.5676 điểm; ECA giảm0.6751; không giữ hai module trong candidate chính ở seed này và chưa có lý do ưu tiên combo detail+local.
- SimAM tăng0.4142 điểm trên P04, tập trung Pedestrian+1.0265/Cyclist+0.3579, Car−0.1418. Tăng1 parameter (residual layer-scale) nhưng vẫn thêm tính toán; model latency tăng~4.50% so P04.
- P03 là lựa chọn baseline loss + focal + IQA có AP gần P04 và pipeline nhanh hơn trong lần đo này. Không gọi OGA là nguyên nhân của toàn bộ gain P16−B0.

## AP theo lớp của P16

| Lớp | IoU | Easy | Moderate | Hard |
| --- | ---: | ---: | ---: | ---: |
| Car | 0.7 | 89.748006 | 84.724001 | 83.404636 |
| Pedestrian | 0.5 | 67.799817 | 60.658065 | 55.855335 |
| Cyclist | 0.5 | 84.374934 | 67.596436 | 64.399174 |

## Checkpoint và tốc độ

Dùng selected/best_ap.pt. P16 best-AP epoch 50 có70.9928%; best-loss epoch 28 chỉ69.8262%. P04 best-loss epoch 25 chỉ68.9218%, thấp hơn best-AP1.6568 điểm. Các objective khác nhau nên không chọn model bằng so trị số validation loss giữa loss strategies.
Speed model và speed pipeline là hai đại lượng riêng: tất cả số ở bảng là NVIDIA L4, FP32, single-frame, loại10 warmup frames. Pipeline gồm bin load+voxelization+H2D+model+decode/NMS, không gồm ROS/tracking/HMI.
P16: model8.5544ms (116.898Hz), pipeline56.8277ms (17.597Hz), decode/NMS29.5133ms. NMS là phần lớn chi phí. CPU/GPU clock/load không được ghi lại, nên đây là thời gian đo quan sát, không phải benchmark lặp có kiểm soát.
P02 và P11 có cùng graph/815.743 params; loss không chạy khi inference. P11 có ít detection ở score threshold0.05 (27.54/frame so44.07/frame), decode/NMS17.23ms so29.24ms. Không quy chênh lệch model-only timing cho tính toán loss.
## Việc còn lại, theo ưu tiên

1. P16 là candidate AP tốt nhất; P11 là candidate Q-OGA với FP thấp và pipeline nhanh ở ngưỡng đang dùng. Giữ P04/P03 làm đối chứng nhánh IQA. Ưu tiên đánh giá FP tại cùng recall và benchmark lặp trước khi quyết định hướng bài báo.
2. Nếu chọn nhánh P16 làm model cuối: P22 = P16 chỉ đổi loss.name từ oga sang baseline, giữ focal + IQA + SimAM và toàn bộ recipe. P03 chưa thay được đối chứng loss trên kiến trúc cuối vì P03 chưa có SimAM. So P22/P16 để quyết định vai trò OGA trong nhánh này.
3. Sau P22, nếu chuẩn bị bài báo, ưu tiên thêm hai seed cho model cuối và đối chứng loss tương ứng; giữ cặp P04/P16 khi cần kiểm tra độ ổn định của SimAM. Không cần lặp toàn bộ các run.
4. Có thể re-evaluate P03/P04 với NMS alpha0 và alpha0.5 trên cùng checkpoint để tách scoring khỏi training supervision; không cần train mới.
5. Nếu claim geometry/corner/adaptive weighting là các contribution riêng trong OGA, cần ablation bên trong loss. P01–P05 hiện đo whole OGA strategy, chưa tách các term đó.
6. Chưa ưu tiên P12/P13/P17/P18 hay chạy lại P06–P10. P18 chỉ đáng làm nếu có claim reparam/deployment. Các đối chứng SGFPN/hybrid cũ có confound đã ghi ở audit trước; không dùng chúng để claim gain riêng đã được tách sạch.

## Giới hạn

Tất cả kết luận về gain là số quan sát của seed 42. Chưa đánh giá variance qua seed, chưa independent re-run AP từ raw KITTI, chưa có official KITTI test hay 3D AP. Các run cũ có encoder/neck/head/augmentation/schedule khác nhau nên chỉ xếp hạng chung; pairwise attribution chỉ dùng các cặp config đã kiểm tra.
R01 cũ mới38/100 epoch, không đưa vào ranking các run đã hoàn tất. B0=R07 vẫn là run cũ có Moderate mAP cao nhất.

## Xếp hạng toàn bộ run hoàn tất

| Run | mAP Mod | Mean AP-9 | Params |
| --- | ---: | ---: | ---: |
| P16 | 70.992834 | 73.173378 | 852,930 |
| P04 | 70.578624 | 72.660297 | 852,929 |
| P03 | 70.309153 | 72.578621 | 852,929 |
| P14 | 70.011065 | 72.031707 | 854,401 |
| P15 | 69.903559 | 72.023823 | 852,933 |
| P11 | 69.200712 | 71.274030 | 815,743 |
| P02 | 68.967108 | 71.116604 | 815,743 |
| P05 | 68.722546 | 70.647241 | 827,809 |
| R07 | 68.459941 | 70.283161 | 790,623 |
| R03 | 68.124178 | 70.051362 | 789,823 |
| R04 | 67.785462 | 69.916834 | 785,031 |
| P01 | 67.378677 | 69.460029 | 815,743 |
| R06 | 67.280263 | 69.607765 | 784,383 |
| R05 | 66.555981 | 68.976344 | 669,551 |
| R08 | 66.247315 | 68.453192 | 674,639 |
| R02 | 60.217937 | 61.109952 | 789,823 |
| R10 | 51.295928 | 53.955486 | 668,831 |
| R09 | 50.142749 | 52.002395 | 650,041 |
| R11 | 47.599536 | 49.081301 | 641,977 |
| R12 | 47.373931 | 48.950356 | 648,313 |

## Files đã tạo

- [Bảng CSV gồm đầy đủ 9 AP và hai loại tốc độ](paper_table.csv)
- [Biểu đồ ablation](ablation_summary.png)
- [Đường AP theo epoch](ap_learning_curves.png)
- [Tốc độ và AP](speed_accuracy.png)
- [Config model P16](recommended_model_config.json)
- [Config đối chứng P22, chưa train](configs/p22_focal-baseline-iqa-simam.json)
- [Lệnh chạy P22](p22_command.txt)
- [Kiểm tra hash và provenance](verification.json)
