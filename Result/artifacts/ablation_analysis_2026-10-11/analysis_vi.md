# Kiểm tra ablation BEV ngày 2026-10-11

Đã kiểm tra 12 run từ thư mục Downloads được cung cấp. Chưa đủ ablation hoàn chỉnh cho bài báo; đã có một số đối chứng encoder, head và width hữu ích. N6 là anchor có AP cao nhất hiện tại, chưa phải final proposed được xác nhận.

## Kết quả được ghi nhận

AP là local loader-aligned rotated BEV R40 Moderate, trung bình ba lớp. F là fusion width, H là số kênh đầu ra neck/đầu vào head. SG là scale gate bật. Stripped là hybrid_gt_sampling với các ENABLE_* tắt, không phải gt_sampling.

| ID | Cấu hình | AP trung bình | Car | Pedestrian | Cyclist | Params | Model ms |
|---|---|---:|---:|---:|---:|---:|---:|
| N0 | MobilePIXOR legacy35 | 41.3990 | 75.515 | 28.156 | 20.525 | 597817 | 11.041 |
| N1 | F32/H32 ungated, stripped aggressive | 60.2179 | 73.702 | 47.297 | 59.655 | 789823 | 6.705 |
| N2 | F32/H32 ungated, hybrid quotas6/8/10 | 68.1242 | 83.513 | 56.618 | 64.242 | 789823 | 6.719 |
| N3 | RC F24/H32, hybrid | 67.7855 | 82.874 | 54.458 | 66.024 | 785031 | 7.250 |
| N4 | SG F24/H16, stripped | 66.5560 | 82.370 | 55.120 | 62.179 | 669551 | 7.105 |
| N5 | SG F24/H32, stripped | 67.2803 | 84.881 | 54.882 | 62.078 | 784383 | 6.877 |
| N6 | SG F32/H32, hybrid | 68.4599 | 83.307 | 58.328 | 63.744 | 790623 | 6.937 |
| N7 | SG F32/H16, stripped | 66.2473 | 81.072 | 54.700 | 62.970 | 674639 | 7.449 |
| N8 | hist14 single standard | 50.1427 | 79.967 | 45.358 | 25.104 | 650041 | 5.917 |
| N9 | hist14 grouped standard | 51.2959 | 77.540 | 43.043 | 33.304 | 668831 | 6.916 |
| N10 | NeXt legacy35 | 47.5995 | 75.241 | 44.361 | 23.196 | 641977 | 5.883 |
| N11 | rich8 single standard | 47.3739 | 78.349 | 38.481 | 25.292 | 648313 | 6.060 |

## Kiểm chứng và giới hạn

- SHA256 của cả 12 file checkpoint được đánh giá khớp report. Epoch và điểm AP khớp selection/AP curve ở 10 run có AP validation. Hai run legacy chọn checkpoint theo loss; N0 chỉ có log đến epoch 38 dù cấu hình100 epoch, N10 có100 epoch.
- Cùng validation split SHA256, cùng input asset hashes, 3769 frame. Tất cả seed42; GPU đánh giá NVIDIA L4, FP 32. Việc này không xác nhận training manifest/GT DB giống nhau vì package không có đầy đủ audit DB/train assets.
- Các run mới N1–N7 và N9 ghi cùng evaluation source commit b635f0e. N8 ghi 9c453f1; N0/N10/N11 ghid3a46c9. Commit khác cần kiểm tra thay đổi liên quan, không mặc định kết quả mất giá trị. N8→N9 chỉ khác head ở config ngoài validation batch; source có tối ưu hist14 CHW/backend. Giữ kết quả AP như bằng chứng ban đầu, kiểm tra parity/đánh giá cùng code khi chốt claim.
- Latency e2e chứa đọc file và encoding; giữa các run đã có thay đổi backend, nên không dùng chênh lệch e2e để quy cho neck/head. Model ms cũng cần benchmark lại cùng môi trường khi báo cáo tốc độ.
- Mười run 50 epoch hoàn thành lịch đã cấu hình. Bảy run dùng GT sampling đạt AP tốt nhất ởepoch 50; đây là kết quả với budget50, chưa chứng minh đã hội tụ. Nếu mở rộng budget, mở rộng đồng nhất các cặp cần so sánh với cùng lịch LR.
- Báo cáo hiện tại là local ROI BEV, không phải kết quả official KITTI hidden test.

## Những kết luận hiện có

- N11→N8: rich8→hist14 tăng 2,7688 điểm AP dưới standard, single H16/F24. Khác source/backend nên cần kiểm chứng khi claim encoder; chưa có đối chứng encoder trên model mạnh N6.
- N8→N9: single→grouped tăng 1,1532 điểm AP. Cyclist tăng 8,2006; Car giảm 2,4266; Pedestrian giảm 2,3144. Đây là gói grouped head/target/loss aggregation, chưa tách kiến trúc head thuần. Val batch/source khác cần thống nhất đánh giá.
- N4→N5: H16→H32 ở SG F24, sampler stripped giữ nguyên, tăng 0,7243 điểm AP. N4→N7: F24→F32 ở H16 giảm 0,3087 điểm. Chưa đủ kết luận F32 tốt hơn; thiếu F32/H32 với sampler stripped tương ứng.
- N2→N6 tăng 0,3358 nhưng đồng thời bật gate và đổi quotas6/8/10→15/10/10. Chưa có gate-only control.
- N3 RC F24/H32 so N6 SG F32/H32 là so hai gói khác width; chênh−0,6745 không đủ chứng minh RC kém hơn cùng width.
- N1→N2 tăng 7,9062 nhưng đổi probability 1→0,5, quotas15/10/10→6/8/10, candidate multiplier1→3, attempts1→6, margin 0→0,1 và bật các kiểm tra. Không được quy toàn bộ mức tăng cho ground/static/density.
- N5→N6 cũng đổi cả fusion width và các kiểm tra hybrid; thiếu đối chứng chỉ thay kiểm tra.

## Augmentation còn thiếu

Tất cả bảy run có GT vẫn dùng NAME=hybrid_gt_sampling. Không có run gt_sampling thật, none hoặc openpcdet_global no-GT. Standard đã có nhưng chưa cùng kiến trúc mạnh N6. Cần so bốn recipe trên cùng kiến trúc; nếu muốn claim sampler, dùng cùng global transforms, quota, probability và DB.

N6 bật ground/static/density/visibility; LOS, shadow, radiometric đều tắt. Với shadow=False, extra_width=[0,0,0], margin 0,1, visibility protection không kích hoạt nhánh kiểm tra trong code. Do đó N6 không đại diện tất cả tính năng hybrid.

Preview N1 và N6 cùng frame000000/seed42 đều ghi added=27, forced preview. Hình minh họa khác vị trí; một frame forced preview không chứng minh số object trung bình hoặc phân bố training. Cần audit sampler trên cùng tập training frames: accepted theo lớp, rejected_by_reason, range bands, points/object. Cùng quota không bảo đảm cùng số object accepted; thêm đối chứng matched accepted count nếu claim chất lượng placement độc lập số lượng.

## Các control đã chuẩn bị

Full config được lấy từ N6; giữ50 epoch/warmup4, train batch16, val batch32, baseline loss và IQA off, seed42, AP every10/local_bev. Đường dẫn dữ liệu được chuyển sang data/kitti/processed và data/kitti/raw; cần sửa theo môi trường thật. Các config đã qua notebook configuration validator, chưa instantiate model hoặc train trong môi trường này.

| Config | Mục đích | So với |
|---|---|---|
| [C01_hybrid_constraints_off](configs/C01_hybrid_constraints_off.json) | Complete stripped F24/F32 x H16/H32 factorial; isolate active hybrid checks vs N6 | N4, N5, N7, N6 |
| [C02_hybrid_sg_f24_h32](configs/C02_hybrid_sg_f24_h32.json) | Matched SG-F24 control for RC N3 and fusion width N6 | N3, N6 |
| [C03_hybrid_gate_off](configs/C03_hybrid_gate_off.json) | Gate on/off with unchanged sampler quotas15/10/10 | N6 |
| [C04_true_gt_matched](configs/C04_true_gt_matched.json) | Real standard GT backend; matched quotas15/10/10 probability 0.5 and globals | N6 |
| [C05_standard_same_model](configs/C05_standard_same_model.json) | Same N6 architecture with standard augmentation | N6, C04_true_gt_matched |
| [C06_none_same_model](configs/C06_none_same_model.json) | No augmentation on N6 architecture | C05_standard_same_model, N6 |
| [C07_global_no_gt](configs/C07_global_no_gt.json) | Isolate effect of adding GT to identical global pipeline | C04_true_gt_matched, N6 |
| [C08_rich8_same_model](configs/C08_rich8_same_model.json) | Encoder under strong augmentation on N6 architecture | N6 |
| [C09_single_same_model](configs/C09_single_same_model.json) | Grouped vs single under same N6 augmentation and widths | N6 |
| [C10_focal_same_model](configs/C10_focal_same_model.json) | C4 Focal alone; retain baseline loss and IQA off | N6 |

## Thứ tự ưu tiên

1. C01: hoàn thiện factorial width của nhóm stripped và kiểm tra riêng gói ground/static/density trên N6. Tắt visibility trong C01 không tạo hiệu ứng riêng ở cấu hình này.
2. C04/C05/C06: true GT, standard và none trên cùng kiến trúc N6, hoàn thiện bốn mode augmentation. C04 là recipe GT matched probability 0,5 với N6, không phải preset mặc định probability 1. Dùng C07 nếu cần tách sampling khỏi global transforms.
3. C02/C03: sửa đối chứng RC/SG cùng F24 và gate on/off. Nếu muốn full width factorial cho hybrid có checks, cần thêm F24/H16 và F32/H16 có checks; không dùng nhóm stripped thay.
4. C08/C09: encoder rich8/hist14 và single/grouped dưới augmentation mạnh. Nếu C01–C07 dẫn đến chọn augmentation khác, sinh lại C08/C09 từ winner trước khi train.
5. C10: context none/focal, baseline loss/IQA off. Nếu architecture thay đổi sau các controls, sinh lại từ architecture winner.
6. Loss: giữ kiến trúc/context/augmentation winner, IQA off; baseline, UWAG, OGA và Q-OGA nếu cần so cả strategies. OGA cần term controls nếu claim loss mới: SmoothL1 only, +corner, +MGIoU, +both fixed weights, +adaptive weighting. Các controls này cần hỗ trợ implementation phù hợp, không chỉ đổi name.
7. IQA: baseline/OGA × IQA off/on. Off tái dùng bảng loss; thêm hai run on, cùng target nhất quán và alpha0 khi chọn checkpoint. Sweep alpha trên cùng checkpoint là evaluation, không phải training mới. Q-OGA classification quality target không đồng nghĩa IQA branch.
8. Baseline paper: chạy MobilePIXOR và NeXt legacy35 với cùng 50 epoch/warmup4/AP-selection/split/seed nếu dùng làm mốc incremental. Nếu claim backbone thuần, thêm control head BN/SiLU và depth để tránh trộn đổi head vào backbone.
9. Freeze final, xác nhận paired seeds 42/43/44 trên baseline/final và các claim chính; leave-one-out các contribution trên full model. So lại augmentation runner-up trên final để kiểm tra interaction.

## Ngân sách và phần tùy chọn

- Mười config C01–C10 là mười run mới nếu chạy toàn bộ trên anchor N6; C07 là bổ sung cho claim sampling. Bốn run ưu tiên C01+C04+C05+C06 đã xử lý phần lớn khoảng trống augmentation hiện tại. Không phải chỉ cần10 run là hoàn chỉnh bài báo.
- Loss strategy comparison thêm 2 run nếu chỉ baseline/UWAG/OGA, hoặc3 run nếu thêmQ-OGA; IQA thêm 2. Baseline50 epoch thêm 2. OGA-term controls, full hybrid-width factorial, multiple seeds và final leave-one-out có budget riêng.
- Nếu muốn dùng/claim LOS, shadow, radiometric: trên hybrid winner chạy +LOS, +shadow, +radiometric riêng và +cả ba; thêm 4 run. Nếu không đưa chúng vào proposed, không bắt buộc chạy chỉ vì code có option.
- Nếu claim riêng ground/static/density, cần bỏ từng tính năng khỏi hybrid winner với mọi tham số khác giữ nguyên. Visibility cần kiểm tra ở cấu hình mà nhánh thực sự hoạt động.

## Bản đồ run gốc

- N0: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixor-standard_aug-baseline_loss-legacy35-baseline_iou-s42`
- N1: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-hybrid_gt_aug-baseline_loss-hist14-baseline_iou-ctx_none_local_none_f32_detail0_766fef4e6653-grouped_e02c0e499671-s42-Fake_opengt-h32`
- N2: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-hybrid_gt_aug-baseline_loss-hist14-baseline_iou-ctx_none_local_none_f32_detail0_766fef4e6653-grouped_e02c0e499671-s42-h32`
- N3: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-hybrid_gt_aug-baseline_loss-hist14-baseline_iou-rc_sgfpn-ctx_none_local_none_f24_detail0_49723c0ff51f-grouped_e02c0e499671-s42-h32`
- N4: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-hybrid_gt_aug-baseline_loss-hist14-baseline_iou-sgfpn-ctx_none_local_none_f24_detail0_31da6d889331-grouped_ff947e4c9cc2-s42-Fake_opengt`
- N5: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-hybrid_gt_aug-baseline_loss-hist14-baseline_iou-sgfpn-ctx_none_local_none_f24_detail0_aa4491a30122-grouped_e02c0e499671-s42-Fake_opengt-h32`
- N6: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-hybrid_gt_aug-baseline_loss-hist14-baseline_iou-sgfpn-ctx_none_local_none_f32_detail0_a56fc07332a7-grouped_e02c0e499671-s42-h32`
- N7: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-hybrid_gt_aug-baseline_loss-hist14-baseline_iou-sgfpn-ctx_none_local_none_f32_detail0_c5b0155ec765-grouped_ff947e4c9cc2-s42-Fake_opengt`
- N8: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-standard_aug-baseline_loss-hist14-baseline_iou-ctx_none_local_none_f24_detail0_a94403319241-s42`
- N9: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-standard_aug-baseline_loss-hist14-baseline_iou-ctx_none_local_none_f24_detail0_bdf3f53ecbc7-grouped_ff947e4c9cc2-s42`
- N10: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-standard_aug-baseline_loss-legacy35-baseline_iou-s42`
- N11: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-standard_aug-baseline_loss-rich8-baseline_iou-ctx_none_local_none_f24_detail0_d7919995bdf4-s42`
