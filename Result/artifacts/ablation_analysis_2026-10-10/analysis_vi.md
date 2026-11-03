# Phân tích ablation BEV và lựa chọn model ours — 10/10/2026

### Goal

Chốt một hướng nghiên cứu BEV có bằng chứng, chọn ứng viên ours và xác định những run thực sự cần bổ sung. Phân tích 9 run trong `/home/duyennh/Downloads/Runs_Lidar`; chưa có bản thảo bài báo trong thông tin được cung cấp, nên phần định hướng đóng góp là đề xuất dựa trên repo và kết quả.

### Constraints

- Người dùng xác nhận BEV là mục tiêu chính. Không cần chuyển toàn bộ thí nghiệm sang 3D.
- Số liệu hiện có: local ROI BEV AP R40, ba lớp, validation 3.769 frame; seed42; train BF16, evaluation PyTorch FP32 trên NVIDIA L4.
- Mọi nhận xét tăng/giảm AP bên dưới là điểm phần trăm, chưa có kiểm chứng độ biến thiên qua nhiều seed.
- Đánh giá chọn checkpoint và thời gian inference lấy từ artifact gốc; không chạy lại training hay inference trong lần phân tích này.
- Cấu hình sinh thêm đã qua `resolve_detection_config`; chưa dựng model hoặc smoke train vì interpreter hiện tại không có PyTorch. Cần profile và smoke trên môi trường train thực tế trước các run dài.

### Known context

**Bảng tổng hợp.** R0/R1 là checkpoint được chọn theo loss; các dòng còn lại chọn AP trên lịch đánh giá đã lưu. Với R0, tên file `evaluation_ap_...` không chứng minh checkpoint best-AP: history không có AP và selection gốc chọn loss. AP9 không được dùng để chọn ours trong phân tích này.

| ID | Model | Car Mod | Ped Mod | Cyc Mod | mAP Mod | Params detector | Model ms | Pipeline ms | Epoch checkpoint | Epoch log / cấu hình |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | MobilePIXOR, legacy35 | 75,52 | 28,16 | 20,53 | 41,40 | 597.817 | 11,04 | 88,02 | 30, loss | 38 / 100 |
| R1 | MobilePixorNeXt, legacy35 | 75,24 | 44,36 | 23,20 | 47,60 | 641.977 | 5,88 | 80,80 | 13, loss | 100 / 100 |
| R2 | NeXt, rich10, single | 77,30 | 35,61 | 27,54 | 46,82 | 648.889 | 6,07 | 59,30 | 30, AP | 50 / 50 |
| R3 | NeXt, rich12, single | 78,78 | 34,63 | 27,53 | 46,98 | 649.465 | 6,17 | 69,98 | 50, AP | 50 / 50 |
| R4 | NeXt, rich8, single | 78,35 | 38,48 | 25,29 | 47,37 | 648.313 | 6,06 | 40,67 | 20, AP | 50 / 50 |
| R5 | NeXt, rich8, grouped | 76,04 | 38,88 | 32,10 | 49,01 | 667.103 | 7,54 | 47,68 | 30, AP | 70 / 100 |
| R6 | NeXt, hist14, grouped, hybrid, F32/H32 | 83,74 | 54,50 | 63,16 | **67,13** | 789.823 | 6,84 | 92,83 | 50, AP | 50 / 50 |
| R7 | NeXt, rich8, grouped, hybrid, F32/H32 | 83,35 | 54,35 | 62,68 | **66,79** | 788.095 | 6,97 | 56,30 | 50, AP | 50 / 50 |
| R8 | NeXt, hist14, single | 79,97 | 45,36 | 25,10 | 50,14 | 650.041 | 5,92 | 77,39 | 30, AP | 50 / 50 |

F là `model.neck_fusion_channels`; H là `model.backbone_out_dim`, đầu vào của detection heads. Tăng H16→H32 cũng thay output projection của neck. Pipeline đo bin load + voxelization/encoding + H2D + model + decode/NMS; loại 10 warmup frame, không bao gồm ROS/tracking/HMI. Đây là các phép đo offline đã lưu, chưa phải benchmark lặp đồng điều kiện trên thiết bị edge.

**Kết luận từng nhóm.**

1. R1 so R0 tăng 6,20 mAP; model forward giảm khoảng 46,7%. Nhưng đổi cả backbone, BN/activation head và mức hoàn thành lịch train. Đây là so sánh hai hệ thống triển khai, chưa phải ablation cô lập backbone và chưa chứng minh vượt implementation gốc của một paper.
2. R2/R3/R4 có cùng NeXt depth3/4/2, lịch50/warmup4, loss baseline, single-head, standard augmentation. Rich10/rich12 không vượt rich8 về mAP. Validation batch của rich8 là16, hai run còn lại32: nên đồng nhất khi xác nhận lại.
3. R8 so R4: hist14 tăng **2,7688 mAP**, Ped +6,8762, Car +1,6181, Cyc −0,1879. Đây là đối chứng encoding tốt nhất hiện có (khác source commit cần kiểm tra code liên quan). Pipeline tăng40,67→77,39ms, chủ yếu ở preprocessing21,07→52,43ms.
4. R5 so R4: tăng1,6320 mAP, Cyc +6,8122 nhưng Car −2,3135. Cùng lúc đổi grouped head và lịch50/warmup4→100/warmup8, R5 chỉ có log tới70. Không gán toàn bộ gain cho grouped head. Grouped còn dùng chia nhóm và weighting loss0,5/0,5, nên đây là gói thiết kế head/objective; muốn tách tác dụng weighting cần đối chứng riêng.
5. R7 so R5: tăng17,7882 mAP (Car +7,3173; Ped +15,4692; Cyc +30,5781), nhưng đổi augmentation, F24→32, H16→32 và lịch train. Phần heads tăng37.631→148.975 weights. Không viết rằng hybrid sampler riêng lẻ tăng17,79 AP.
6. R6 so R7: hist14 chỉ thêm**0,3393 mAP**, Car +0,3899; Ped +0,1511; Cyc +0,4768; thêm1.728 weights. Đây là cặp encoding dưới recipe mạnh, có cùng config ngoài encoding nhưng khác commit nguồn đánh giá. Cần khóa source và kiểm tra nhiều seed trước kết luận thắng.

**Quy luật đáng quan tâm.** Hist14 hơn rich8 rõ hơn ở recipe standard (+2,77) nhưng lợi ích nhỏ dưới hybrid (+0,34). Giả thuyết cần kiểm chứng: augmentation giúp model học tín hiệu mà encoding histogram bổ sung ở recipe yếu. Chưa có dữ liệu đủ để kết luận cơ chế đó. Cyc30–50m của R6/R7 là48,03/42,05; Ped30–50m chỉ5,92/6,49. Nếu phát triển module mới, đây là vùng khó đáng khảo sát hơn việc thêm nhiều attention cùng lúc. Các band trên50m có rất ít GT (Ped13/Cyc17 trong báo cáo), nên không dùng chúng làm bằng chứng chính.

**Chọn checkpoint.** Rich10 best-AP46,82 so best-loss37,35: chênh9,4661 AP. Rich12 chênh4,2126; rich8 chênh2,1213. Khi objective khác nhau, so sánh raw loss giữa model không thích hợp. Tất cả run mới dùng Moderate macro AP R40 của ba lớp và cùng tần suất kiểm tra. Best-AP hiện nay nghĩa là tốt nhất trong các epoch đã đánh giá, thường10/20/30/40/50; không phải mọi epoch.

**Kiểm chứng dữ liệu.** 9 báo cáo có cùng split SHA256 `2f76fc0f69b5d9786f008694274baaa49d878ba076b96529c488939fd83564c0`, cùng11.307 entry hash input asset (3×3.769) và cùng Moderate GT Car7.874/Ped1.739/Cyc544. Điều này kiểm chứng validation được lưu; chưa kiểm chứng được GT database train-only và bản processed training data vì những dữ liệu đó không nằm trong thư mục kết quả đã xem.

### Risks

**Các module chưa được chứng minh.** Cả9 config đều `scale_gated_fpn=false`, `use_reparam=false`, `header_use_iou=false`, loss baseline/use_iou=false và không context/local attention. `neck_type=scale_gated_fpn` là tên lớp neck, không có nghĩa gate đang bật: forward thực tế dùng cộng feature khi flag false. Những sơ đồ hist14/focal/ECA hay roadmap OGA/IQA/Rep/LiteMLA không mô tả model đã đạt67,13. Chỉnh sơ đồ theo final resolved config hoặc chạy ablation cho các thành phần muốn claim. Hybrid cũng đang tắt shadow masking, radiometric calibration và line-of-sight; không ghi thành Full PCU-Aug bao gồm các tính năng đó.

**Protocol bài báo.** Các file hiện tại ghi `local loader-aligned KITTI-style rotated BEV AP R40`, GT lọc center theo ROI. Chúng không tương đương tự động với reference KITTI APBEV. [KITTI chính thức](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=bev) dùng IoU70% cho Car,50% cho Ped/Cyc, xếp hạng Moderate và40 recall positions; phải kiểm tra đủ quy tắc lọc/ignore khi so paper.

Repo có reference evaluator độc lập, nhưng `evaluate_kitti_3d.py:177` hiện từ chối checkpoint `box_mode=bev`, kể cả `metric_mode=bev`, vì đường checkpoint yêu cầu dự đoán vertical3D thật. Với mục tiêu BEV, việc cần làm là bổ sung/kiểm chứng đường reference APBEV cho output BEV, xử lý đúng hệ tọa độ, calibration và quy tắc detection filtering; không đổi flag metric rồi coi đã tương thích. Không dùng GT của validation để bổ sung z/h vào prediction. Trước khi đường này sẵn sàng, các bảng hiện có dùng nhãn **local ROI BEV R40**, so nội bộ cùng evaluator; không gộp với số paper/leaderboard.

**Lịch và source.** Source đánh giá gồm ba commit; snapshot config giống nhau không chứng minh implementation giống nhau. R0 chỉ38/100 và R5 chỉ70/100 trong dữ liệu được cung cấp. Không ghi chúng là run hoàn thành100epoch. Khi làm bảng paper, khóa source code, train/val manifests, database provenance, scheduler, seed, precision và policy chọn checkpoint. Không resume lịch50 thành100 để giả định là tương đương training100 từ đầu.

**Đóng góp khoa học.** Backbone/encoder/grouped head/GT sampling là các trục thiết kế, không tự động là bốn đóng góp mới. Cần xác định cơ chế cụ thể do bài đề xuất và so với đối chứng phổ biến; lợi ích hybrid phải so với standard GT có cùng quotas/probability/global transforms. Việc thêm nhiều module có sẵn không thay thế bằng chứng ablation.

### Options (2–4)

1. **Hiệu quả accuracy/latency: rich8 R7.** 0,788M params,66,79mAP, pipeline56,30ms (~17,76FPS offline); model6,97ms (~143,39FPS riêng forward). Gọn và gần best accuracy; chưa đủ điều kiện gọi thời gian này là realtime edge hay benchmark deployment.
2. **Ưu tiên accuracy: hist14 R6.** 0,790M params,67,13mAP; lợi ích histogram rõ hơn ở recipe yếu và Cyc30–50m. Pipeline92,83ms (~10,77FPS), preprocessing52,89ms là nút thắt. Chỉ nên chọn làm main nếu lợi ích được lặp lại hoặc tối ưu encoder làm chi phí chấp nhận được; không hứa tối ưu sẽ giữ nguyên AP trước khi kiểm chứng parity.
3. **Thêm module backbone/quality.** Focal, scale gate, OGA, IQA là ứng viên thí nghiệm tiếp theo; làm từng module trên anchor đã khóa. Hiện không có trained evidence cho chúng trong9run này. Rủi ro lớn hơn và tăng số training cần chạy; chưa ưu tiên trước khi lấp đối chứng thiếu.

### Recommendation

**Hướng chính đề xuất:** BEV detector nhẹ dưới1M **toàn detector**, cân bằng accuracy/latency trên ba lớp, tập trung cải thiện lớp nhỏ bằng gói head và augmentation được đối chứng rõ. Lấy **R7 rich8 làm ứng viên ours chính**, R6 hist14 là biến thể accuracy. Nếu luận điểm chính của bản thảo là encoding hist14, thứ tự này phải đảo: R6 là ứng viên main và cần chứng minh histogram vượt rich8 dưới recipe mạnh qua nhiều seed cùng benchmark latency. Số liệu hiện tại chưa đủ để tự động chốt một đóng góp mới.

Checkpoint ứng viên R7 là `selected/best_ap.pt`, epoch50; R6 cũng epoch50. Đường dẫn chính xác có trong `audit.json` và `results.csv`. Không dùng checkpoint từng lớp khác nhau để ghép một dòng kết quả.

**Bảng cumulative hợp lệ cần xây dựng (cùng50epoch/warmup4, seed42 trước).**

| Dòng | Encoding | Head | F/H | Augmentation | Recipe / hiện trạng |
|---|---|---|---|---|---|
| C0 | rich8 | single |24/16| standard OneOf | R4 đã có; đối chiếu source |
| C1 | rich8 | grouped |24/16| standard OneOf | **A9**, sửa lệch lịch của R5 |
| C2 | rich8 | grouped |32/32| standard OneOf | **A2**, đo gói tăng width |
| C3 | rich8 | grouped |32/32| global compose | **A3**, bỏ sampler, giữ transforms của hybrid |
| C4 | rich8 | grouped |32/32| standard GT + cùng global | **A4**, quotas/probability khớp hybrid |
| C5 | rich8 | grouped |32/32| hybrid GT + cùng global | **A0**, anchor R7 đã có |
| C6 | hist14 | grouped |32/32| cùng hybrid | **A1**, R6 đã có, optional accuracy |

C1−C0 đo grouped head/objective; C2−C1 đo width bundle; C3−C2 đo đổi global transforms; C4−C3 đo standard GT sampling; C5−C4 đo thay standard bằng hybrid; C6−C5 đo encoding. Không ép bảng phải tăng đều: một thành phần không giúp thì loại khỏi main và báo kết quả.

**Thứ tự chạy tiết kiệm.**

1. A3 trước: có GT sampling và không GT sampling trên đúng cùng backbone/head/global transforms. Đây là run phân giải gain lớn hiện nay.
2. A4: so hybrid với standard GT thực sự matched. File `openpcdet_gt.json` mặc định Car15/Ped10/Cyc10 và sampler probability mặc định1; hybrid hiện Car6/Ped8/Cyc10,probability0,5. Không lấy hai preset đó làm ablation thuần sampler. A4 đã khớp các giá trị này; khác biệt các cơ chế placement/collision/density thuộc gói hybrid, muốn attribution từng cơ chế phải ablate thêm sau.
3. A9: chạy grouped50/warmup4 thay cho R5 lịch100, so R4. Chỉ lấy checkpoint30 của R5 không sửa được khác scheduler.
4. A2: cô lập augmentation bundle trên model width32 và hoàn thiện bảng cumulative. Bốn run này lấp thiếu chính; A0/A1 đang có kết quả seed42 nhưng cần kiểm tra source/protocol trước tái sử dụng.
5. Nếu cần kiểm tra width trong recipe mạnh, chạy A6. Nếu claim riêng fusion width, chạy A7; nếu claim H32/head capacity, chạy A8. A0/A6/A7/A8 tạo2×2 factorial widths. A5 là single-head trong recipe mạnh để đo tương tác grouped×augmentation; useful nếu claim grouped là đóng góp chính.
6. A10 chỉ khi bài cần claim scale gate. Sau đó mới thử focal hoặc loss. Nếu thử OGA/IQA, dùng factorial baseline/IQA-off, OGA/IQA-off, baseline/IQA-on, OGA/IQA-on và khóa quality target/inference scoring. Không bật focal+ECA+OGA+IQA trong một run rồi gán gain cho từng thành phần.

**Freeze protocol:** AdamW lr0,0007, wd0,001, min_lr1e-6, gradclip10; batchtrain16/accumulation1, val16; BF16 train, encoding/target numba;50epoch,4warmup; AP10epoch và final; chọn `map_moderate_percent`, Car/Ped/Cyc equal macro. Giữ score0,05, NMS0,1,maxdet500,grid0,1m/ROI hiện có cho phase nội bộ. Nếu muốn policy dày hơn hay lịch100, phải áp dụng đồng nhất và dùng run names mới.

**Xác nhận ours:** sau screening, seed42/43/44 cho anchor và đối thủ quyết định. Pair same seeds; báo mean±sample std và từng lớp. Ngưỡng1AP hoặc1%latency là quy ước nghiên cứu phải đặt trước, không phải ngưỡng ý nghĩa thống kê mặc định. Có thể đặt quy tắc practical: chênh<0,5AP mà latency tăng rõ thì ưu tiên model nhanh hơn; ghi rõ đó là quy tắc lựa chọn, không khẳng định hai model tương đương thống kê. Benchmark latency lặp trên cùng máy/config/load, batch1, warmup cố định; tách encode/H2D/model/NMS và tổng pipeline, deployment/FP16 nếu có phải ghi riêng. Khoảng2–4giờ/run là tham khảo từ log L4 (rich8hybrid≈2,23giờ,hist14hybrid≈3,63giờ cho50epoch), không đảm bảo cho môi trường mới và chưa gồm preparation.

**Lệnh chạy.** Các recipe trong thư mục `configs/` bên cạnh báo cáo là fullJSON độc lập, đường data mặc định `data/kitti/processed` và raw`data/kitti/raw`. Nếu trên Colab, sửa hai trường tương ứng `/content/kitti_processed` và `/content/kitti_raw`; không truyền original run name. GT database phải thuộc training IDs của split đang dùng.

```bash
# Chạy từ repository root; đây là ví dụ run ưu tiên đầu tiên.
python3 tools/benchmarks/profile_detector.py \
  --config artifacts/ablation_analysis_2026-10-10/configs/A3_rich8_global_f32_h32.json \
  --output artifacts/ablation_analysis_2026-10-10/profile_A3.json

python3 tools/kitti_training_pipeline/train.py \
  --config artifacts/ablation_analysis_2026-10-10/configs/A3_rich8_global_f32_h32.json \
  --detector-root detector --output-root artifacts/paper_bev \
  --run-name paper_bev_A3_global_f32_h32_s42 --seed 42

python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
  --name paper_bev_A3_global_f32_h32_s42 --backend pytorch \
  --model artifacts/paper_bev/paper_bev_A3_global_f32_h32_s42/selected/best_ap.pt \
  --config artifacts/paper_bev/paper_bev_A3_global_f32_h32_s42/config.resolved.json \
  --detector-root detector --kitti-root data/kitti/raw --split splits/kitti/val.txt \
  --output artifacts/paper_bev/paper_bev_A3_global_f32_h32_s42/evaluation_ap_local_bev.json
```

Thay config/run-name theo A4,A9,A2; chạy từ đầu cho mỗi ablation, không warm-start chỉ variant có module mới. Đổi seed phải đổi run-name. Recipe A0/A1 cung cấp anchor để rerun và xác nhận seeds; không ghi đè các kết quả gốc trong Downloads.

### Acceptance criteria

- Các bảng paper dùng cùng evaluator và ghi đúng local/reference,R40/IoU/split; BEV reference path được kiểm chứng trước khi so số từ paper khác.
- Baseline và ours cùng schedule,policy chọn checkpoint,split và data provenance; hardware/precision benchmark ghi riêng.
- Có đối chứng A3/A4 cho sampler và A9 cho grouped; không attribution gain giữa các run đổi nhiều yếu tố.
- Ours được chọn theo metric đã khóa và accuracy/latency trên seeds phù hợp; các module trong schematic trùng final resolved config.
- Lưu full Easy/Moderate/Hard ba lớp, mAPMod,AP9, Params backbone/neck/heads/detector, pipeline breakdown, checkpointepoch/hash và source commit.

Tài liệu nguồn nội bộ: `docs/plans/lightweight_lidar_backbone_2026/reference_evaluation.md`; `detector/core/models/backbones/mobilepixornext.py`; `tools/kitti_training_pipeline/checkpoint_selection.py`. Bảng số liệu đầy đủ: `results.csv`; metadata và AP curves: `audit.json`; mô tả11recipe: `recipes.json`; biểu đồ: `accuracy_latency_and_curves.png`.
