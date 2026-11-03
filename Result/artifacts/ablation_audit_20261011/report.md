# Kiểm tra ablation KITTI — 11/10/2026

Nguồn: `/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts`.
Phạm vi bài báo do người dùng xác nhận: toàn bộ hệ thống, gồm OGA/IQA và focal context.

**Kết luận: chưa đủ cho ablation của toàn bộ hệ thống.** Có các cặp đối chứng hữu ích cho encoding, grouped heads và độ rộng, nhưng chưa có run OGA, IQA hoặc focal. Một số so sánh neck/augmentation còn thay đổi nhiều yếu tố. Model tốt nhất hiện có là **R07**, chưa phải mô hình đã kiểm chứng đủ các đóng góp dự kiến.

## 1. Bằng chứng đã kiểm tra

- 12 thư mục run, 23 báo cáo evaluation. Kiểm tra SHA256 của 23 checkpoint được các báo cáo sử dụng và 23 file predictions NPZ: đều khớp.
- Cả 23 báo cáo đánh giá đủ 3.769 frame, cùng hash split validation và cùng bảng 11.307 hash tài sản đánh giá. Hash config cũng khớp các file `config.resolved.json` tải về.
- Tính lại Moderate macro mAP và AP9 từ AP từng lớp/difficulty: đều khớp. Epoch checkpoint khớp lịch sử chọn loss/AP được ghi lại.
- 11/12 run có lịch sử đủ số epoch cấu hình. R01 chỉ ghi hoàn thành epoch 1–38 trong lịch 100 epoch; log dừng khi đang bắt đầu epoch 39.
- Tất cả run: seed42, loss `baseline`, IQA tắt, C4 context `none`, local attention `none`, reparameterization tắt, box mode `bev`.
- Không có báo cáo audit dữ liệu huấn luyện/GT database trong archive. Các hash validation không chứng minh được train/val disjoint trên Colab hoặc GT database chỉ lấy từ train; cần ghi thêm audit tại nơi dữ liệu thật tồn tại.
- Trường `data.source` là source commit lúc evaluation, không đủ xác nhận source commit lúc training. Các báo cáo dùng ba commit khác nhau: `d3a46c9`, `9c453f1`, `b635f0e`.

Các kiểm tra trên xác nhận tính nhất quán của artifacts; không tải tensor checkpoint vào PyTorch, không chạy lại inference hoặc training. Môi trường Python hiện tại không có PyTorch. Mười config mới được kiểm tra bằng các contract thuần của repository, chưa kiểm tra GPU hoặc dữ liệu thật.

## 2. Tất cả kết quả hiện có

Metric: **local loader-aligned ROI BEV AP R40**. `mAP Mod` là trung bình đều của Car/Pedestrian/Cyclist ở Moderate. Không gộp nó với AP9.
F = `neck_fusion_channels`; H = `backbone_out_dim`, tức độ rộng feature cấp cho detection heads.
Trừ R01/R11, bảng dùng checkpoint chọn bằng maximum validation AP, đo mỗi 10 epoch và epoch cuối. R01/R11 dùng minimum validation loss.

| ID | Cấu hình rút gọn | Epoch hoàn thành/lịch | Epoch chọn | Car Mod | Ped Mod | Cyc Mod | mAP Mod | Params |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R01 | MobilePixor, binary35, single H16, OneOf | 38/100 | 30 loss | 75.515 | 28.156 | 20.525 | 41.399 | 597,817 |
| R02 | NeXt, hist14, ungated F32/H32, simplified hybrid P=1 | 50/50 | 50 AP | 73.702 | 47.297 | 59.655 | 60.218 | 789,823 |
| R03 | NeXt, hist14, ungated F32/H32, hybrid quota 6/8/10 | 50/50 | 50 AP | 83.513 | 56.618 | 64.242 | 68.124 | 789,823 |
| R04 | NeXt, hist14, RC-SGFPN mul F24/H32, hybrid | 50/50 | 50 AP | 82.874 | 54.458 | 66.024 | 67.785 | 785,031 |
| R05 | NeXt, hist14, SGFPN F24/H16, simplified hybrid | 50/50 | 50 AP | 82.370 | 55.120 | 62.179 | 66.556 | 669,551 |
| R06 | NeXt, hist14, SGFPN F24/H32, simplified hybrid | 50/50 | 50 AP | 84.881 | 54.882 | 62.078 | 67.280 | 784,383 |
| **R07** | **NeXt, hist14, SGFPN F32/H32, hybrid** | **50/50** | **50 AP** | **83.307** | **58.328** | **63.744** | **68.460** | **790,623** |
| R08 | NeXt, hist14, SGFPN F32/H16, simplified hybrid | 50/50 | 50 AP | 81.072 | 54.700 | 62.970 | 66.247 | 674,639 |
| R09 | NeXt, hist14, single H16, ungated F24, OneOf | 50/50 | 30 AP | 79.967 | 45.358 | 25.104 | 50.143 | 650,041 |
| R10 | NeXt, hist14, grouped H16, ungated F24, OneOf | 50/50 | 30 AP | 77.540 | 43.043 | 33.304 | 51.296 | 668,831 |
| R11 | NeXt, binary35, depths2/4/2, single H16, OneOf | 100/100 | 13 loss | 75.241 | 44.361 | 23.196 | 47.600 | 641,977 |
| R12 | NeXt, rich8, depths3/4/2, single H16, OneOf | 50/50 | 20 AP | 78.349 | 38.481 | 25.292 | 47.374 | 648,313 |

Tên thư mục đầy đủ và đường dẫn config/checkpoint của từng ID có trong [run_inventory.json](run_inventory.json); bảng CSV có trong [run_summary.csv](run_summary.csv).

## 3. Ours tốt nhất hiện tại

Run:

```text
mobilepixornext-hybrid_gt_aug-baseline_loss-hist14-baseline_iou-sgfpn-ctx_none_local_none_f32_detail0_a56fc07332a7-grouped_e02c0e499671-s42-h32
```

Chọn `selected/best_ap.pt`, epoch50, đi cùng `config.resolved.json` của chính run này.

| Lớp | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Car | 89.2563 | 83.3071 | 81.1476 |
| Pedestrian | 64.3840 | 58.3283 | 52.5905 |
| Cyclist | 78.9438 | 63.7444 | 60.8465 |

- Moderate mAP **68.4599405%**; AP9 **70.2831609%**.
- Backbone gồm neck: **641,648** params; heads: **148,975**; detector: **790,623**. Criterion baseline không có tham số học.
- Latency được ghi trên **NVIDIA L4, PyTorch FP32, từng frame**: model **6.9367 ms**, toàn luồng offline **45.5193 ms**. Tương ứng khoảng **144.16 FPS model** và **21.97 FPS toàn luồng**. Toàn luồng gồm đọc bin/encoding, transfer và decode/NMS.
- Config: hist14, stage depths3/4/2, grouped Car + Pedestrian/Cyclist, H32/F32, SGFPN, hybrid P=.5 và quotas15/10/10. Loss baseline, IQA/focal đều tắt.

R04 có checkpoint loss epoch33 đạt 68.258%, cao hơn checkpoint AP epoch50 của cùng run (67.785%). AP chỉ được đo theo lịch 10 epoch nên có thể bỏ lỡ đỉnh ở epoch33. Ghi cả hai như phân tích checkpoint, nhưng dùng một quy tắc chọn chung cho bảng ablation. R07 vẫn có điểm cao nhất trong toàn bộ các báo cáo, kể cả khi xét cả loss checkpoints.

Chênh R07–R03 chỉ **+0.336 điểm phần trăm**, chưa có nhiều seed và còn khác quotas. Chưa đủ bằng chứng kết luận SGFPN vượt ungated FPN. R06 có Car Moderate tốt nhất và R04 có Cyclist Moderate tốt nhất trong các AP-selected runs; chọn R07 theo mục tiêu macro mAP ba lớp.

## 4. Đã chứng minh được gì, còn hở ở đâu

| Cặp run | Chênh mAP Mod | Đánh giá |
|---|---:|---|
| R12 → R09: rich8 → hist14 | +2.769 | Cặp encoding tốt: kiến trúc/lịch train/augmentation trùng. Source evaluation và implementation preprocessing khác; cần đo lại tốc độ trên cùng code. |
| R09 → R10: single → grouped | +1.153 | Cặp cấu hình phù hợp; batch validation16→32 là khác biệt phụ. Cyc +8.201, nhưng Car −2.427, Ped −2.314: không được nói cả ba lớp đều tăng. Grouping còn đổi cách tổng hợp/chuẩn hóa loss theo nhóm. |
| R05 → R06: H16 → H32 | +0.724 | Chỉ `backbone_out_dim` đổi. Có thể dùng cho ablation width, nhưng mới một seed. |
| R05 → R08: F24 → F32, H16 giữ nguyên | −0.309 | Chỉ fusion width đổi. Không có bằng chứng F32 luôn tốt hơn; tác dụng có thể phụ thuộc H/augmentation. |
| R03 → R07: gate off → on | +0.336 | Đồng thời quotas Car6→15, Ped8→10 và batch validation16→32. Chưa tách được tác dụng gate. |
| R04 → R07: RC → SG | +0.674 | Đồng thời F24→32. Cần SGFPN F24 đối chứng; RC hiện không hỗ trợ F32. |
| R06 → R04 | +0.505 | Đồng thời đổi neck và bật bốn heuristic hybrid. Không dùng làm bằng chứng riêng RC hoặc augmentation. |
| R02 → R03 | +7.906 | Đổi protection/density, P1→.5, quotas và số candidate/attempts/margin. Đây là so sánh hai recipe, không phải ablation riêng physics. |
| R01 → R11 | +6.200 | R01 thiếu lịch sử train; header activation/BN khác. Cặp baseline chưa đạt điều kiện so sánh backbone riêng. |
| R11 → R12/R09 | — | Encoding, depth, epochs/warmup và selection đều khác; không phải ablation binary35 ↔ rich8/hist14. |

`Fake_opengt` là nhãn run do người chạy đặt, không chứng minh dữ liệu giả. Config của các run này vẫn gọi `hybrid_gt_sampling`, placement `source_relative`, có range/azimuth perturbation. Vì vậy không đặt tên dòng này là “standard OpenPCDet GT sampling”. Sampler chuẩn trong repo dùng `NAME="gt_sampling"`; cần run riêng nếu muốn so sánh.

Các hybrid runs hiện chỉ bật density subsampling, ground validation, static collision và visibility protection. Line of sight, shadow masking và radiometric calibration đều tắt. Không có bằng chứng cho “full physics/ray-consistent augmentation” hoặc cho lợi ích riêng từng heuristic. Visibility protection còn phụ thuộc điều kiện kích hoạt trong sampler; nên ghi số lần từng heuristic thực sự hoạt động khi ablate.

## 5. Mười run bổ sung có config sẵn

Giữ cố định từ R07: seed42, hist14, depths3/4/2, grouped H32, F32 trừ P06, BF16, train batch16/validation batch32, AdamW lr7e-4, weight decay1e-3, 50 epoch/4 warmup, cùng split và dữ liệu. AP selection mỗi 10 epoch + epoch cuối; score .05, NMS .10, cap500. Không tự chuyển sang preset có quotas8/6/6 hoặc fusion24 vì sẽ làm hỏng đối chứng R07.

### Ưu tiên 1 — năm run OGA/IQA/focal

| ID | Context | Loss | IQA | Cặp cần so sánh |
|---|---|---|---|---|
| R07 đã có | none | baseline | off | Điểm neo |
| **P01** | focal | baseline | off | R07 ↔ P01: focal trên baseline |
| **P02** | focal | OGA | off | P01 ↔ P02: OGA không có IQA |
| **P03** | focal | baseline | on | P01 ↔ P03: IQA không có OGA |
| **P04** | focal | OGA | on | P02/P03 ↔ P04: đóng góp và tương tác OGA/IQA |
| **P05** | none | OGA | on | P05 ↔ P04: focal trong mô hình có OGA/IQA |

Đây là factorial 2×2 OGA/IQA trên cùng focal architecture, cộng hai mức kiểm tra focal. Không chỉ chạy P04 rồi quy toàn bộ tăng AP cho ba thành phần.

IQA on phải đồng thời đặt `model.header_use_iou=true`, `loss.use_iou=true`; config đã đặt `data.kitti.nms_alpha=.5`. IQA off dùng alpha0. Target cố định `mgiou`, weight1. Target này là proxy multi-axis GIoU; không mô tả nó là exact rotated IoU. Nếu bài muốn kiểm tra exact rotated IoU, thêm một run chỉ đổi `iou_target_type="rotated_iou"` sau khi đã có IQA baseline.

### Ưu tiên 2 — năm run gỡ các so sánh đang lẫn yếu tố

| ID | Thay đổi so với R07 | Đối chứng / mục đích |
|---|---|---|
| **P06** | SGFPN F24/H32, giữ hybrid15/10/10 | P06 ↔ R04: SG vs RC cùng F24. P06 ↔ R07: F24 vs F32 với H32. |
| **P07** | Tắt scale gate, giữ F32/H32 và quotas15/10/10 | P07 ↔ R07: tác dụng gate, sửa lỗ hở R03. |
| **P08** | Bỏ GT sampler, giữ nguyên global flip/rotation/scaling/translation | P08 ↔ R07: GT augmentation trong cùng global pipeline. |
| **P09** | Simplified hybrid: chỉ tắt bốn heuristic đang bật | P09 ↔ R07: heuristic bundle với cùng P=.5, quota, candidate/attempts và placement. |
| **P10** | Sampler chuẩn `NAME="gt_sampling"`, P=.5, quotas15/10/10 | P10 ↔ R07: standard sampler vs hybrid. P08/P09/P10 phân biệt global transforms, sampling và heuristics. |

Mười config nằm trong [configs/](configs/); mapping, diff đầy đủ và lệnh train nằm trong [additional_runs.json](additional_runs.json) và [commands.txt](commands.txt). Đây là cấu hình đề xuất chưa train, không phải kết quả đã đo. Chúng giữ đường dẫn dữ liệu `/content/...` từ run thật; chỉnh đường dẫn khi chạy trên máy khác và đồng nhất giữa các đối chứng.

Lệnh ví dụ từ repository root:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config artifacts/ablation_audit_20261011/configs/p01_focal-baseline-noiqa.json \
  --detector-root detector --output-root artifacts/kitti \
  --run-name ablation-p01-focal-baseline-noiqa-s42
```

Dùng thư mục run mới. Log R07 ghi tổng train+validation+AP khoảng 3.04 giờ trên môi trường của run; đây chỉ là mốc lập ngân sách. Năm run đầu có thể vào khoảng 15 giờ GPU nếu điều kiện và tốc độ tương tự; OGA/focal có thể làm thời gian thay đổi.

## 6. Các run tiếp theo phụ thuộc claim và kết quả

**Nếu OGA là đóng góp mới của bài:** cần ba đối chứng nội bộ trên cùng P02 architecture, IQA off: (O1) SmoothL1 với trọng số cố định, không geometry; (O2) SmoothL1 + geometry, trọng số cố định; (O3) SmoothL1 + T-SBUW, không geometry. So với OGA đầy đủ P02 để tách hiệu ứng regression loss, geometry và weighting. Có thể thêm projection-only/corner-only nếu cả hai là đóng góp riêng. Code OGA hiện chưa có switch bỏ geometry hoặc chuyển weighting sang fixed; các run này cần triển khai switch trước, không thể giải quyết bằng config khóa không được code đọc. `corner_beta=0` chỉ bỏ corner term, vẫn còn projection GIoU. UWAG hiện đổi cả weighting lẫn loại geometry nên không thay thế được đối chứng riêng này.

**Sau khi chốt cấu hình full tốt nhất:** chạy thêm hai leave-one-out nếu muốn khẳng định hist14/grouped có ích trong hệ thống cuối: full đổi hist14→rich8; full đổi grouped→single. Giữ các thành phần khác, bỏ `data.head_groups` và `loss.group_weights` khi single. Cặp hiện có R12/R09/R10 là bằng chứng trong recipe đơn giản, không tự chứng minh tương tác với OGA/IQA/focal/hybrid.

**Kiểm tra độ ổn định:** chạy cấu hình đối chứng và full cuối cùng với seed43,44, cộng seed42 đã có: tối thiểu bốn run bổ sung cho một cặp ba seed. Nếu dùng chênh nhỏ SG/RC/gate làm claim chính, áp dụng ba seed cho chính cặp đó. Báo mean±std, giữ cùng chính sách chọn checkpoint, không chỉ giữ seed tốt nhất.

**Baseline và lịch train:** nếu MobilePixor/legacy NeXt xuất hiện trong bảng so sánh chính, retrain với cùng 50 epoch/4 warmup, AP every10, augmentation và điều kiện dữ liệu đã chốt. Nếu claim backbone riêng, còn phải ghép encoding/head/loss tương đương. Nếu chỉ so sánh toàn hệ thống, công bố đầy đủ các khác biệt kiến trúc/encoding. R01 chưa hoàn tất và R11 selection/lịch khác nên chưa dùng để quy gain cho backbone. Nếu binary35 là dòng ablation encoding, thêm binary35 depths3/4/2 cùng config R09; không dùng R11 thay nó. External/SOTA comparisons cần comparator phù hợp cùng task/protocol; archive này chỉ có baseline nội bộ.

**Nếu bài báo claim AP3D hoặc so sánh chuẩn KITTI 3D:** cần model full và baseline có z/height branches được train/fine-tune thật, cùng vertical objective và protocol; sau đó chạy reference AP3D và APBEV. Theo workflow hiện tại của repo, checkpoint BEV không được đổi evaluator flag để giả lập 3D. Nếu bài chỉ về BEV, không bắt buộc thêm 3D nhưng phải công bố đúng metric local ROI hoặc xây adapter/evaluator chuẩn thích hợp trước so sánh ngoài repo. KITTI công bố BEV và 3D là các benchmark và dùng 40 recall positions từ 2019; cùng nhãn R40 không bảo đảm evaluator local có cùng GT domain/matching. [KITTI benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=bev), [quy trình trong repo](../../docs/plans/lightweight_lidar_backbone_2026/benchmark_protocol.md).

ECA/SimAM/detail path, additive RC, bidirectional RC hoặc RepConv chỉ cần thêm nếu chúng là thành phần đưa vào bài hoặc ứng viên thực sự muốn chọn. Các kế hoạch LiteMLA cũ không phản ánh code hiện tại; không đưa vào bảng như đã được thử.

## 7. Việc cần đo lại, không cần training mới

- Evaluate các checkpoint dùng để viết bài trên cùng source commit, device, precision và pipeline. R09 hist14 cũ preprocessing52.4 ms trong khi R10 hist14 mới11.4 ms; phần transfer cũng khác. Chênh toàn luồng hiện tại không thể quy riêng cho grouped head.
- Với checkpoint IQA đã train, chạy inference alpha0/.25/.5/.75 để tách regularization/supervision khỏi reranking. Alpha .5 vẫn là giá trị đã chốt cho bảng chính; sweep là sensitivity analysis. Phải chạy lại decode/inference trước NMS, vì NPZ hiện lưu detections sau NMS và không giữ logits quality/raw candidates để tái dựng alpha sweep đúng.
- Ghi params, cùng quy ước MACs/FLOPs, latency model và toàn luồng riêng, p50/p95, peak memory, warmup, batch, hardware/backend. Số ~144 FPS của R07 là model-only; tốc độ toàn luồng đo được ~22 FPS.
- Giữ AP từng lớp Easy/Moderate/Hard, đặc biệt phân tích Ped/Cyc theo khoảng cách. Các báo cáo đã có distance bands; không cần train riêng. Vùng >50m chỉ có 13 Ped và 17 Cyc Moderate GT, nên kết quả vùng này biến động lớn và không phù hợp làm kết luận mạnh từ một seed.
- Chạy `tools/benchmarks/audit_kitti_assets.py` trên Colab với config thật để lưu bằng chứng train/val disjoint và GT database train-only; không giả định các audit hiện chưa có trong archive là đã thất bại.

## 8. Khuyến nghị thực thi

1. Chạy **P01–P05 trước**, vẫn giữ R07 làm best hiện có. Chọn thành phần theo bằng chứng thay vì mặc định full luôn tốt hơn.
2. Chạy **P06–P10** để hoàn thiện các đối chứng neck/augmentation. Các run này dùng baseline objective để nối với artifacts hiện có; nếu tương tác với full là claim chính, cần lặp đúng phép bỏ thành phần trên full đã chốt.
3. Hoàn thiện OGA nội bộ nếu là novelty, hai leave-one-out encoding/head trên full, và repeated seeds của cặp chính trước khi chốt bảng paper.
4. Thêm matched comparator/reference 3D khi claim bài yêu cầu. Không cần mở rộng toàn bộ optional modules chỉ vì chúng có trong code.

Chưa có kết quả mới trong báo cáo này. Chưa chạy training hoặc sửa code detector.
