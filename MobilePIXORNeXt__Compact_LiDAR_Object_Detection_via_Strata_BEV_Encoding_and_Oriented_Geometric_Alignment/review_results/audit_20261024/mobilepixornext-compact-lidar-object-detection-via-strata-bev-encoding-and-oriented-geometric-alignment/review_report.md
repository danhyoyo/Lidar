# Rà soát logic, số liệu và chỉnh sửa MobilePIXORNeXt

Ngày 24/10/2026. Execution: **sequential single-agent**. Đã đọc cả hai entry point, toàn bộ 8 section nội dung, 9 tệp bảng và bảng Strata trong Methods; đối chiếu nguồn Excel/CSV và một số code định nghĩa metric/loss. Đây là đánh giá nội bộ, không kiểm chứng tài liệu bên ngoài và không chạy lại huấn luyện/KITTI evaluation.

**Kết luận:** Có mâu thuẫn thực chất trong protocol, GPU, lịch huấn luyện và danh tính run. Một số hàng có AP trung bình đúng nhưng AP từng lớp hoặc tốc độ được ghép từ run khác. Bản thảo đã được sửa thành bản diễn giải trung thực với các nguồn đang có; chưa thể coi ablation đã được kiểm chứng đầy đủ hoặc bản thảo sẵn sàng nộp.

**Tổng hợp bản trước sửa:** 30 phát hiện (10 major, 16 moderate, 4 minor), đã xác minh 30/30 trích dẫn. Đây là phân loại vấn đề ban đầu, không phải 30 lỗi vẫn còn nguyên sau sửa.

## Các trường hợp “cùng model nhưng hai số”

| Vị trí | Số trong bản cũ | Đối chiếu nguồn và xử lý |
|---|---|---|
| A8 / Baseline loss | 67.38 / 69.46 nhưng 6.94 ms / 144.16 FPS | AP thuộc P1 focal; tốc độ thuộc S32. Tách A8=S32 và A8F=P1. |
| Loss Baseline P1 | 83.92 / 55.48 / 62.74 | Nguồn 82.97 / 56.99 / 62.17. Đã sửa; mAP vẫn 67.38. |
| Q-OGA P11 | 84.35 / 58.12 / 65.13; 0.735M | Nguồn 84.89 / 57.15 / 65.56; 815,743 params. Đã sửa; mAP vẫn69.20. |
| OGA+IQA 70.58 và final71.50 | Ghi chỉ thêm attention | 70.58 = P4 focal / independent heads; final=no-focal / shared heads, schedule khác. Không phải một model chỉ bật/tắt attention. |
| Final L4 và T4 | 71.50/73.45 ghép60.96 ms | L4:71.50/73.45; T4:71.46/73.42 và60.96 ms. Đã tách snapshot, chưa quy chênh lệch cho GPU. |
| Forward8.67 và17.88 ms | Ghi standalone/pipeline cùngT4 | 8.67 từL4,17.88 từT4. Đã sửa. |
| 66.25% | Ghi PC-GTA H32/F24 | Nguồn O32G16 simplified samplerV2 H16/F32/SG-FPN. Đã sửa nhãn và bỏ +6.03PC-GTA claim. |
| Block-SimAM gain | Methods+0.88, Results+0.92 pp | Hai phát biểu cũ mâu thuẫn; cả hai chưa có matched no-attention control trên final model. Đã bỏ causal gain. |

## Phạm vi sửa

Đã thống nhất entry point và abstract, chỉnh luồng train/inference và nghĩa yaw/quality, sửa các ô có nguồn rõ, thay bảng “tiến trình cộng dồn” bằng inventory có run ID, và giới hạn các kết luận nhân quả. Không tạo số thí nghiệm mới. Không sửa phương trình loss hoặc tự chọn một bộ kết quả thay thế khi nguồn chưa đủ.

Có ba hướng xử lý: chỉ trau chuốt câu sẽ để nguyên kết luận sai; giả định mọi run tương đương sẽ che mất confound; bản sửa hiện tại chọn **ghi đúng nguồn và giới hạn diễn giải**, đồng thời giữ các thiếu sót thực nghiệm thành việc cần làm. Bản trước sửa được lưu trong `artifacts/original/`; diff nằm ở `artifacts/data/source_changes.patch`.

## Phát hiện chi tiết

Các mục dưới đây là **[LLM]**: được suy luận từ văn bản và nguồn cục bộ. Các phép kiểm tự động được báo riêng ở cuối. Tọa độ và trích dẫn là bản trước sửa trong `artifacts/original/`.

### I01 — Hai entry point chứa abstract và tuyên bố khác nhau (major)

Vị trí bản trước sửa: [MobilePIXORNeXt.tex:56](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/MobilePIXORNeXt.tex:56).

> under the official 40-point interpolation protocol

main.tex mô tả local evaluator chưa xác minh tương đương; MobilePIXORNeXt.tex vẫn ghi official và highest Pedestrian AP. sections/abstract.tex lại có bản thứ ba. Đã thống nhất MobilePIXORNeXt.tex → main.tex → sections/abstract.tex.

**Trạng thái:** Đã sửa.

### I02 — Tuyên bố unified/official trái với evaluator thực tế (major)

Vị trí bản trước sửa: [sections/experiments.tex:40](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/experiments.tex:40).

> The evaluation code strictly adheres to the official KITTI devkit evaluation protocol

Workbook v7, sheet B05_Protocol_Audit: ours/MobilePIXOR dùng local loader-aligned, strict x/y ROI; PIXOR và năm baseline dùng reference evaluator, full-frame GT. Cùng R40 và IoU không chứng minh cùng ignore/matching/GT domain. Đã bỏ shared AP ranking, thêm L/R trong bảng. Chưa có đối chiếu evaluator trên cùng predictions.

**Trạng thái:** Đã sửa mô tả; thiếu thí nghiệm equivalence.

**Việc tiếp theo:** Chạy cùng predictions / GT qua hai evaluator, lưu class/difficulty ignore rules và matching outputs.

### I03 — 8.67 ms và 17.88 ms bị giải thích sai thành hai chế độ T4 (major)

Vị trí bản trước sửa: [sections/results.tex:45](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/results.tex:45).

> Standalone GPU Forward Latency

Seed workbook Theo seed!AF7: 8.671745659 ms trên L4; v7 B10_Speed_Detail!E14: 17.875186737 ms trên T4. Không có thí nghiệm cùng GPU chứng minh chênh lệch do synchronization/DataLoader. Đã ghi GPU và tách snapshot.

**Trạng thái:** Đã sửa.

### I04 — AP L4 và T4 của cùng tên model bị gộp (moderate)

Vị trí bản trước sửa: [tables/comparison.tex:14](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/tables/comparison.tex:14).

> 86.90 & \textbf{60.78} & \underline{66.81}

Seed workbook: L4 86.900441/60.778199/66.807415, mAP 71.495352, mAP9 73.445159. V7 T01_Benchmark row16: T4 86.903460/60.749361/66.732802, mAP 71.461874, mAP9 73.418711. Giữ cả hai snapshot với nhãn nguồn; chưa thể khẳng định khác biệt chỉ do GPU. Cần checkpoint hash/prediction hash.

**Trạng thái:** Đã tách; cần kiểm tra hash.

**Việc tiếp theo:** Đối chiếu SHA256 checkpoint, config, split và prediction files giữa báo cáo L4/T4.

### I05 — Lịch 50/100 epoch và epoch cuối không khớp metadata (major)

Vị trí bản trước sửa: [sections/experiments.tex:51](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/experiments.tex:51).

> all exploratory ablation experiments (A1--A10) were trained under a standardized 50-epoch schedule

Seed workbook Cau hinh: Final S42 completed/planned 160/160, selected 130; Final S43/S44 100/100, selected 100; B0 archived38/planned100 selected 30. runs_summary.csv: B1 budget 100 selected 13 (loss), E8 budget 50 selected 20, E14 budget 50 selected 30. Epoch47 là checkpoint loss-selected trong lịch sử final run, không chứng minh run độc lập50 epoch. Đã sửa toàn bộ khẳng định uniform schedule/asymptotic convergence.

**Trạng thái:** Đã sửa mô tả; chưa có thí nghiệm đồng budget.

**Việc tiếp theo:** Lập ma trận run/config/selected epoch/criterion; chỉ tính delta component khi các biến khác khớp.

### I06 — Bảng seed không chỉ thay seed (major)

Vị trí bản trước sửa: [tables/seed_robustness.tex:3](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/tables/seed_robustness.tex:3).

> under a 100-epoch training schedule

Cau hinh!T10=Binary35 của MobilePIXOR S42; T11/T12=hist14 của S43/S44. S42 archived38 epoch, các seed còn lại100; Final S42 160 epoch, S43/S44 100 epoch. Mean/std vẫn là phép tính mô tả hợp lệ nhưng không đo initialization variance hoặc chứng minh numerical stability. Đã đổi caption/note và bỏ suy luận nhân quả.

**Trạng thái:** Đã sửa diễn giải; cần matched reruns.

**Việc tiếp theo:** Giữ cùng encoding, model, sampler, optimizer schedule, budget và checkpoint selection cho mọi seed.

### I07 — A4→A5 không cô lập head width (major)

Vị trí bản trước sửa: [sections/results.tex:99](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/results.tex:99).

> Expanding head width to 32 channels (A5) adds only

A4 là G16 H16/F24/global-only, 51.30. A5=O32 H32/F32/SimplifiedGT V1 p=1,60.22. Cùng đổi head, neck, augmentation; +8.92 pp không phải head-only. Không có đo rank hoặc gradient chứng minh subspace collapse. Đã sửa cấu hình và bỏ cơ chế chưa đo.

**Trạng thái:** Đã sửa nhãn/diễn giải; chưa có isolated head run.

### I08 — 66.25% bị gắn nhầm PC-GTA H32/F24 (major)

Vị trí bản trước sửa: [tables/components.tex:17](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/tables/components.tex:17).

> A6 & MobilePIXORNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F24 / SumFPN

runs_summary.csv O32G16:66.247315/68.453192, H16/F32/SG-FPN/SimplifiedGT V2. A5=O32:H32/F32/SumFPN/SimplifiedGT V1. Vì vậy +6.03 pp không đo PC-GTA. H32 hybrid thực sự68.124178; O32→H32=+7.906241 pp, vẫn thay cả p/quota/placement recipe. Đã bỏ mọi +6.03 PC-GTA claim, ghi source-linked inventory.

**Trạng thái:** Đã sửa.

### I09 — A8 trộn AP P1 với tốc độ S32 (major)

Vị trí bản trước sửa: [tables/components.tex:39](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/tables/components.tex:39).

> 67.38 & 69.46 & 6.94 & 144.16

P1 focal context:67.378677/69.460029 và7.772452 ms/128.659530 FPS. S32 no-focal SG-FPN:68.459941/70.283161 và6.936693 ms/144.160918 FPS. Bản cũ ghép hai run vào một hàng. Đã giữ A8=S32 và thêm A8F=P1. A7H32→A8S32 còn đổi quota nên không cô lập neck.

**Trạng thái:** Đã sửa.

### I10 — AP từng lớp của Baseline và Q-OGA không khớp nguồn (moderate)

Vị trí bản trước sửa: [tables/loss.tex:10](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/tables/loss.tex:10).

> 83.92 & 55.48 & 62.74

P1 nguồn:82.970232/56.992056/62.173743 →82.97/56.99/62.17; P11 nguồn:84.893017/57.145242/65.563877 →84.89/57.15/65.56. Các bộ số cũ vẫn cho cùng mAP làm tròn nên chỉ kiểm mean sẽ bỏ sót. Đã sửa từ paper_06_Loss_IQA.csv và workbook v7 T02.

**Trạng thái:** Đã sửa.

### I11 — Ghép Q-OGA lịch sử với số tham số của model cuối (major)

Vị trí bản trước sửa: [tables/loss.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/tables/loss.tex:19).

> saving $18{,}626$ parameters

P1/P2/P11 dùng815743 params; P3/P4 dùng852929 params, chênh37186. Model cuối shared-head753569 khác kiến trúc. 69.20% là P11 lịch sử, không có bằng chứng ghép nó với734943 params. Đã sửa bảng0.816/0.853M và tách rõ mô hình cuối.

**Trạng thái:** Đã sửa.

### I12 — Factorial study không chứng minh positive synergy (moderate)

Vị trí bản trước sửa: [sections/results.tex:121](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/results.tex:121).

> These results demonstrate clear methodological synergy

Từ mAP chưa làm tròn: OGA khôngIQA +1.588431 pp; cóIQA +0.269471 pp. Interaction=P4−P3−P2+P1=−1.318960 pp. P4 cao nhất trong factorial không đồng nghĩa tương tác dương; chưa có uncertainty cho hiệu ứng nhỏ. Đã viết lại đúng main effects và interaction.

**Trạng thái:** Đã sửa.

### I13 — Attention table thay context, neck width và lịch sử run (major)

Vị trí bản trước sửa: [sections/results.tex:112](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/results.tex:112).

> strictly attributable to 3D spatial-energy attention

No-attention70.58 = P4 focal / independent towers852929; final71.50=no-focal / shared towers753569/selected 130. R25 middle stage row source F24, manuscriptF32. Lidar_result_final.xlsx còn có R37 F24 70.34 khác final. Không có đối chứng matched no-attention cho final. +0.88 Methods và+0.92 Results còn tự mâu thuẫn; đã bỏ isolated gain và ghi đúng phạm vi bảng.

**Trạng thái:** Đã sửa diễn giải/width; cần matched controls.

**Việc tiếp theo:** Chạy cùng final config, chỉ đổi block/stage attention, giữ width/depth/context/seed/schedule và rule chọn checkpoint.

### I14 — Related Work gán cơ chế LOS/shadow bị tắt cho kết quả chính (moderate)

Vị trí bản trước sửa: [sections/related_work.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/related_work.tex:11).

> integrating adaptive local ground plane fitting, static obstacle clearance filtering, line-of-sight ray checks, and shadow masking

Experiments và Results ghi LOS, shadow, radiometric disabled. Đã mô tả core enabled mechanisms và optional options, không quy AP gain cho cơ chế không bật.

**Trạng thái:** Đã sửa.

### I15 — Tổng task weights không phải tổng gradient budget (moderate)

Vị trí bản trước sửa: [sections/methodology.tex:416](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/methodology.tex:416).

> strictly conserves the total gradient budget across all tasks

Sum omega=J chỉ bảo toàn tổng trọng số. Gradient mỗi task còn phụ thuộc scale/direction và shared features; không suy ra gradient norm hoặc convergence. Đã sửa phát biểu, giữ nguyên công thức.

**Trạng thái:** Đã sửa.

### I16 — Detached surrogate không loại hết gradient competition (moderate)

Vị trí bản trước sửa: [sections/methodology.tex:383](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/methodology.tex:383).

> Detaching $q^*$ prevents gradient competition

q*=clipped mean projectionGIoU không phải exact polygon IoU. stopgrad chặn đường qua target, nhưng quality và geometry còn chung backbone/neck. Exact rotated IoU cũng không thể gọi là hoàn toàn nondifferentiable ở mọi cấu hình. Đã sửa mục tiêu surrogate và giới hạn gradient claim.

**Trạng thái:** Đã sửa.

### I17 — Doubled-angle không khôi phục directed heading liên tục (moderate)

Vị trí bản trước sửa: [sections/methodology.tex:475](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/methodology.tex:475).

> is fully recovered without trigonometric ambiguity

cos2psi/sin2psi biểu diễn orientation modulo pi, không có front/back. Scalar atan2 representative vẫn wrap ở boundary và vector0 không có direction. Đã nêu đúng contract, giữ nguyên phương trình.

**Trạng thái:** Đã sửa.

### I18 — BEV/edge latency bị suy rộng thành 3D và safety deployment (moderate)

Vị trí bản trước sửa: [sections/results.tex:190](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/results.tex:190).

> accurate 3D LiDAR detection was possible

Output chỉ (x,y,l,w,yaw modulo pi), khôngz/h/velocity/tracking. KITTI/T4/L4 không chứng minh embedded/closed-loop collision avoidance hoặc TTC; P99 offline<100 ms không phải worst-case guarantee. Đã giới hạn abstract/results/conclusion vào BEV và measured offline latency.

**Trạng thái:** Đã sửa.

### I19 — Cơ chế tối ưu hóa chưa có thí nghiệm phân biệt (moderate)

Vị trí bản trước sửa: [sections/results.tex:99](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/results.tex:99).

> induces severe subspace rank collapse

Không có feature singular values, gradient diagnostics hoặc matched width control. Các suy luận weak-return attenuation, synergy SG-FPN, T-SBUW suppresses spikes và epoch latency do sharpening đều vượt dữ liệu. Đã thay bằng quan sát AP/loss và ghi điều kiện kiểm chứng.

**Trạng thái:** Đã sửa.

### I20 — Monotonic descent mâu thuẫn validation spikes (moderate)

Vị trí bản trước sửa: [sections/results.tex:144](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/results.tex:144).

> stable monotonic descent

Results ghi classification spikes epochs18–25 rồi caption nói stable monotonic và rapidly stabilized byT-SBUW. Đã đổi thành generally decreasing trajectories; không quy nguyên nhân nếu không có weighting ablation.

**Trạng thái:** Đã sửa.

### I21 — Timing MobilePIXOR và một số percentile không khớp snapshot (moderate)

Vị trí bản trước sửa: [tables/efficiency.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/tables/efficiency.tex:11).

> 90.65 & 11.03

V7B10: MobilePIXOR historicalL4 model11.049958/E2E85.923112/P50 84.114180/P95 112.524565/P99 124.943548. PointRCNN P50/P99=194.769222/222.372998; PartA2=132.399653/169.129394; PVRCNN=187.348769/219.072451. Đã sửa các ô có nguồn xác định, bỏ same-T4 claim. Không suy speedup L4/T4.

**Trạng thái:** Đã sửa.

### I22 — Far-range Pedestrian regression bị bỏ qua trong diễn giải (moderate)

Vị trí bản trước sửa: [sections/results.tex:81](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/results.tex:81).

> These results are consistent with range-adaptive augmentation

Table distance:PedFar0.99→0.80,−0.19 pp; Results chỉ nhấn mạnhCyclistFar14.94 và VRUgain. Đã nêu regression và bỏ suy luận nhân quả sampler/kernel từ whole-model comparison.

**Trạng thái:** Đã sửa.

### I23 — Delta_z 0.1 m không mô tả Strata-BEV14 (minor)

Vị trí bản trước sửa: [tables/setup.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/tables/setup.tex:11).

> Grid resolution ($\Delta_x, \Delta_y, \Delta_z$)

Strata14 dùng4heightbins trên3.5m, còn dz0.1 thuộc legacy35slices. Đã tách horizontalgrid0.1x0.1 vớifourverticalstrata; không đổi công thức.

**Trạng thái:** Đã sửa.

### I24 — Serialized checkpoint không phải kích thước trọng số FP32 thuần (minor)

Vị trí bản trước sửa: [sections/methodology.tex:485](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/methodology.tex:485).

> with an FP32 checkpoint size of $9.34$

753569*4bytes=3.014276MB; serializedcheckpoint9.34MB có thể chứa thêmstate/metadata. Mainabstract cũ gọi753569trainable nhưng12T-SBUW logits nâng tổngtrainable753581. Đã thống nhất deployable/train-only và serializedsize, không khẳng định thành phần checkpoint khi chưa có inspection.

**Trạng thái:** Đã sửa.

### I25 — Hình vehicle được dùng xác nhận VRU AP (moderate)

Vị trí bản trước sửa: [sections/results.tex:166](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/results.tex:166).

> This visual evidence corroborates the substantial

Ví dụ vehicle heatmaps không xác nhận định lượngPed/Cyc30–50m. Whole-model visualization cũng không trực tiếpvalidate Block-SimAM/SGFPN effects hoặc NMSspeed. Đã giới hạn caption và observation vào selectedexamples.

**Trạng thái:** Đã sửa.

### I26 — Phép cộng CPU statistics không tự bảo đảm CUDA sync không đáng kể (moderate)

Vị trí bản trước sửa: [sections/methodology.tex:111](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/methodology.tex:111).

> negligible thread synchronization

Các cell độc lập sau group nhưng nhiều point có thể update cùngcell, cần grouping/reduction/atomic strategy. Chưa có đo CUDA hoặc asynchronouspipeline. Đã chuyển thành futureextension và nêu writeconflict.

**Trạng thái:** Đã sửa.

### I27 — Soft-min corner thiếu định nghĩa D và có thể âm (moderate)

Vị trí bản trước sửa: [sections/methodology.tex:399](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/methodology.tex:399).

> \mathcal{D}_{\mathrm{smooth}} = -\tau_{\mathrm{soft}}

D_direct/D_pi chưa được định nghĩa tường minh là tổng/trung bình khoảng cách theo mét trước normalize hay distance đã normalize. Code smooth_corner_loss.py dùng meancornerL1, softmin rồi chia diagonal. Unnormalized logsumexp-softmin có thể âm và không có exactzero baseline. Đây không tự động là lỗi optimizer, nhưng gọi distance/C-infinity là chưa chính xác khi D gồm abs. Giữ nguyên công thức/kết quả; cần làm rõ định nghĩa, đơn vị tau và intended behavior.

**Trạng thái:** Chưa sửa công thức; cần tác giả xác định.

**Việc tiếp theo:** Định nghĩa D_direct/D_pi, normalization order, tau units và việc cho phép negative loss; không đổi objective hồi tố nếu không chạy lại.

### I28 — Hình lưu sẵn còn legend/configuration cũ (moderate)

Vị trí bản trước sửa: [sections/results.tex:144](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/results.tex:144).

> Trajectories illustrate stable monotonic descent

training_curves PDF vẫn gắnA5=GroupedH32,A6=HybridGT/F24,A8=HybridGT/SGFPN trong khi đối chiếu source đã phát hiện mapping khác. Pareto PDF còn nối frontier qua khác evaluator/GPU. Đã sửa caption nhưng chưa tái tạo hình khi thiếu trace liên kết từng curve với run. Cần dựng lại từ per-run records trước nộp.

**Trạng thái:** Chưa giải quyết nội dung hình; caption đã giới hạn.

**Việc tiếp theo:** Khôi phục curve→run/checkpoint manifest, dựng lại legend và chia biểu đồ theo protocol/hardware.

### I29 — Data flow và index conventions chưa đủ rõ (minor)

Vị trí bản trước sửa: [sections/methodology.tex:20](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/methodology.tex:20).

> A point $\mathbf{p}_i$ maps to grid coordinate $(u_i, v_i)$ as:

u làx/column704, v lày/row800; shapeCHW dùngrowv,columnu. Methods trình bàyStrata trướcPCGTA dễ hiểu nhầm augmentation trên BEV. Đã nêu training points→PCGTA→rasterize và inference raw points→rasterize, cùng tensor indexing. Final heads share stem within eachgroup chứ không share initialgroupblock giữahai nhóm.

**Trạng thái:** Đã sửa.

### I30 — RDA được gọi density-adaptive nhưng công thức chỉ dùng range (minor)

Vị trí bản trước sửa: [sections/methodology.tex:402](/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment/artifacts/original/sections/methodology.tex:402).

> Range and Density-Adaptive (RDA) weighting

w(r) chỉ córange, không có local density input. Đã gọi range-adaptive weighting denotedRDA; chưa có ablation cho density adaptation.

**Trạng thái:** Đã sửa thuật ngữ.

## Nguồn đã đối chiếu

- `Result/mobilepixor_vs_ours_seed_summary.xlsx`: Theo seed và Cau hinh; epoch, encoding, GPU, per-class AP, params.
- `Result/benchmark_T4_20261018/Tables_bai_bao_LiDAR_v7_SpeedDetail.xlsx`: T01_Benchmark, B04_Ours_AP_vs_Loss, B05_Protocol_Audit, B10_Speed_Detail, B12_Timing_Scope, T02_Loss_IQA, S02_Augmentation.
- `Result/artifacts/ablation_analysis_20261012/`: runs_summary.csv, paper_01_Backbone.csv, paper_02_Encoding.csv, paper_03_Head_Width.csv, paper_04_Augmentation.csv, paper_05_Neck.csv, paper_06_Loss_IQA.csv.
- `Result/Lidar_result_final.xlsx`: bảng attention stage/neck width và các run lịch sử. Workbook này có một số ô bị Excel đổi thành date serial; không lấy các ô đó làm AP và không ghi ngược thay đổi vào Excel.
- Code local evaluator, T-SBUW và smooth-corner implementation chỉ được đọc để đối chiếu định nghĩa, không thay đổi.

Các workbook là nguồn lưu trữ cục bộ, không thay thế original JSON/config/checkpoint hashes. `workbooks.json` lưu trích xuất cả sheet/địa chỉ ô để kiểm lại; các giá trị date serial được giữ nguyên trong evidence.

## Những việc vẫn cần trước khi nộp

1. Xác nhận checkpoint/config/prediction hashes cho L4/T4 và mỗi hàng; khôi phục manifest curve→run.
2. So sánh trên cùng reference evaluator hoặc chứng minh local/reference equivalence bằng cùng predictions / GT.
3. Chạy lại các đối chứng chỉ đổi một yếu tố cho final architecture; không dùng legacy focal model đại diện final no-focal.
4. Chạy seed study cùng encoding, schedule, budget và selection rule.
5. Dựng lại hình Pareto/loss với nguồn và nhãn đúng; caption hiện đã nêu giới hạn nhưng nội dung vector cũ còn giữ.
6. Làm rõ D_direct/D_pi và normalization của soft-min. Chưa thay loss, vì đổi objective sẽ cần đánh giá lại.

## Kiểm chứng

- Biên dịch `main.tex` và compatibility entry `MobilePIXORNeXt.tex` thành công: 28 trang.
- Log cuối: 0 undefined references/citations, 0 overfull hbox. Còn các underfull hbox không chặn biên dịch.
- 41/41 kiểm tra nguồn và thực thi checker đạt; 30/30 quote gốc được xác minh. Đây không phải kết quả chạy lại thí nghiệm.
- Đã xem trực quan trang 1, 21, 23; các bảng chính không bị cắt hoặc chồng chữ.

Đọc `artifacts/data/verification.json` để xem kết quả máy chạy thực tế. Phase0 audit đã chạy nhưng text snapshot ban đầu chỉ chứa entry point; không coi output đó là bằng chứng đã kiểm semantic toàn bộ includes. Các phát hiện chi tiết trên đến từ việc đọc thủ công toàn bộ source và đối chiếu dữ liệu. Snapshot đầy đủ dùng kiểm quote đã được bổ sung bằng phép nối các source trước sửa. Đã chạy riêng check_tables, Methods logic và compile bằng script của skill. Không thực hiện online bibliography check. Môi trường không có `uv`/`python`; dùng `python3` tương đương cho script stdlib. Lệnh thử `python -B .../audit.py --help` trả127; fallback `python3` chạy được.
