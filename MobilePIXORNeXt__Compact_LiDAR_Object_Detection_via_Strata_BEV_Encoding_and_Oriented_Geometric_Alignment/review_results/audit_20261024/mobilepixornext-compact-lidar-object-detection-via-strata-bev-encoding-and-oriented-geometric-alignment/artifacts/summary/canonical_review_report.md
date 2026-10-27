# Deep Review Report

**Paper**: `/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/main.tex` | **Language**: EN | **Mode**: deep-review
**Generated**: 2026-10-24 21:49
**Artifacts**: `/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment/review_results/audit_20261024/mobilepixornext-compact-lidar-object-detection-via-strata-bev-encoding-and-oriented-geometric-alignment`

## Overall Assessment

Execution: sequential single-agent. Major Revision: source identities and descriptions corrected where evidenced; controlled component/seed/evaluator evidence and reconciled figures remain incomplete.

- **Major**: 10
- **Moderate**: 16
- **Minor**: 4

## Academic Pre-Review Committee

### Editor (Desk Reject Screen)

## Editor Pre-Screen

Execution: sequential single-agent.

Bài có câu hỏi nghiên cứu và mô tả kiến trúc rõ. Rủi ro chính ở abstract/introduction là chuyển BEV/T4 thành triển khai edge/safety và xếp hạng AP khác evaluator (I01,I02,I18). Pitch phù hợp hướng compact perception sau khi thu hẹp phạm vi.

### Reviewer 1 (Theory Contribution)

## Theory Contribution Review

Execution: sequential single-agent.

Đóng góp được đánh giá như thiết kế hệ thống thực nghiệm, không yêu cầu định lý mới. Những phát biểu cần sửa: conserved gradient budget, continuous directed heading, và surrogate≈exactIoU (I15–I17). Soft-min còn thiếu định nghĩa/normalization (I27).

### Reviewer 3 (Literature Dialogue)

## Literature Dialogue Review

Execution: sequential single-agent.

Related Work đã nhóm theo representation/augmentation/backbone/loss. Không xác minh ngoài kho. Cần phân biệt local TriBand adaptation và complete cited detector, cơ chếPCGTA thực sự bật; không coi prior-art absence là đã được kiểm chứng. I14 đã sửa.

### Reviewer 2 (Methodology & Transparency)

## Methodology Transparency Review

Execution: sequential single-agent.

MUST-FIX: evaluator identity, GPU/report identity, training/selection records, seed confounds, incorrect ablation mapping (I02–I13). Các dữ liệu mô tả được giữ với nhãn nguồn. Không có qualitative coding study; SRQR không áp dụng.

### Reviewer 4 (Logic Chain)

## Logic Chain Review

Execution: sequential single-agent.

Chuỗi hợp lệ: point statistics → compact BEV network → local AP + separately identified offline timing. Chuỗi đứt: heterogeneous historical runs → isolated component gains → causal stability/safety claims. I07–I13,I18–I22,I25 xử lý các bước nhảy này; dữ liệu đối chứng vẫn còn thiếu.

### Committee Consensus

## Cross-perspective assessment

Execution: sequential single-agent; this is cross-perspective agreement within one session.

Score under the skill arithmetic: 1.0/10. This is a count-based indicator, not a measured probability of acceptance.

Priority: repair run/evaluator/GPU identity; remove confounded component attribution; then regenerate source-linked figures and conduct matched comparisons. The prose revision is reviewable, but experimental blockers remain.

## Paper Summary

# Paper Summary: MobilePIXORNeXt: Compact LiDAR Object Detection via Strata-BEV Encoding and Oriented Geometric Alignment

## Research Question
- Accurate 3D object perception from Light Detection and Ranging (LiDAR) point clouds is an important challenge for autonomous vehicles, where volumetric and pillar networks frequently exceed edge computational budgets

## Core Thesis
- In this paper, we present MobilePIXORNeXt, an ultra-compact, proposal-free detector targeting oriented bounding footprints on the Special Euclidean (SE) manifold for real-time collision avoidance and path planning.

## Headline Claims
- In this paper, we present MobilePIXORNeXt, an ultra-compact, proposal-free detector targeting oriented bounding footprints on the Special Euclidean (SE) manifold for real-time collision avoidance and path planning.

## Section Map
- abstract (57-59): 261 words
- introduction (62-63): 2 words
- related (64-65): 3 words
- method (66-67): 3 words
- experiment (68-69): 2 words
- result (70-78): 27 words
- conclusion (79-88): 9 words

## Closure Targets
- No closure target was extracted automatically.

## Major Issues

### M1: Hai entry point chứa abstract và tuyên bố khác nhau
- **Type**: claim_accuracy
- **Source**: [LLM] via `self_standard_consistency`
- **Confidence**: high
- **Section**: MobilePIXORNeXt.tex:56
- **Related Sections**: MobilePIXORNeXt.tex:56, MobilePIXORNeXt.tex
- **Root Cause Key**: `i01`
- **Quote Verified**: yes
- **Quote**: `under the official 40-point interpolation protocol`
- **Explanation**: main.tex mô tả local evaluator chưa xác minh tương đương; MobilePIXORNeXt.tex vẫn ghi official và highest Pedestrian AP. sections/abstract.tex lại có bản thứ ba. Đã thống nhất MobilePIXORNeXt.tex → main.tex → sections/abstract.tex.

### M2: Tuyên bố unified/official trái với evaluator thực tế
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: high
- **Section**: sections/experiments.tex:40
- **Related Sections**: sections/experiments.tex:40, sections/experiments.tex
- **Root Cause Key**: `i02`
- **Quote Verified**: yes
- **Quote**: `The evaluation code strictly adheres to the official KITTI devkit evaluation protocol`
- **Explanation**: Workbook v7, sheet B05_Protocol_Audit: ours/MobilePIXOR dùng local loader-aligned, strict x/y ROI; PIXOR và năm baseline dùng reference evaluator, full-frame GT. Cùng R40 và IoU không chứng minh cùng ignore/matching/GT domain. Đã bỏ shared AP ranking, thêm L/R trong bảng. Chưa có đối chiếu evaluator trên cùng predictions.

### M3: Lịch 50/100 epoch và epoch cuối không khớp metadata
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: high
- **Section**: sections/experiments.tex:51
- **Related Sections**: sections/experiments.tex:51, sections/experiments.tex
- **Root Cause Key**: `i05`
- **Quote Verified**: yes
- **Quote**: `all exploratory ablation experiments (A1--A10) were trained under a standardized 50-epoch schedule`
- **Explanation**: Seed workbook Cau hinh: Final S42 completed/planned 160/160, selected130; Final S43/S44 100/100, selected100; B0 archived38/planned100 selected30. runs_summary.csv: B1 budget100 selected13 (loss), E8 budget50 selected20, E14 budget50 selected30. Epoch47 là checkpoint loss-selected trong lịch sử final run, không chứng minh run độc lập50epoch. Đã sửa toàn bộ khẳng định uniform schedule/asymptotic convergence.

### M4: Attention table thay context, neck width và lịch sử run
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: high
- **Section**: sections/results.tex:112
- **Related Sections**: sections/results.tex:112, sections/results.tex
- **Root Cause Key**: `i13`
- **Quote Verified**: yes
- **Quote**: `strictly attributable to 3D spatial-energy attention`
- **Explanation**: No-attention70.58=P4 focal/independent towers852929; final71.50=no-focal/shared towers753569/selected130. R25 middle stage row sourceF24, manuscriptF32. Lidar_result_final.xlsx còn có R37 F24 70.34 khác final. Không có đối chứng matched no-attention cho final. +0.88 Methods và+0.92 Results còn tự mâu thuẫn; đã bỏ isolated gain và ghi đúng phạm vi bảng.

### M5: 8.67 ms và 17.88 ms bị giải thích sai thành hai chế độ T4
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: high
- **Section**: sections/results.tex:45
- **Related Sections**: sections/results.tex:45, sections/results.tex
- **Root Cause Key**: `i03`
- **Quote Verified**: yes
- **Quote**: `Standalone GPU Forward Latency`
- **Explanation**: Seed workbook Theo seed!AF7: 8.671745659 ms trên L4; v7 B10_Speed_Detail!E14: 17.875186737 ms trên T4. Không có thí nghiệm cùng GPU chứng minh chênh lệch do synchronization/DataLoader. Đã ghi GPU và tách snapshot.

### M6: A4→A5 không cô lập head width
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_results`
- **Confidence**: high
- **Section**: sections/results.tex:99
- **Related Sections**: sections/results.tex:99, sections/results.tex
- **Root Cause Key**: `i07`
- **Quote Verified**: yes
- **Quote**: `Expanding head width to 32 channels (A5) adds only`
- **Explanation**: A4 là G16 H16/F24/global-only, 51.30. A5=O32 H32/F32/SimplifiedGT V1 p=1,60.22. Cùng đổi head, neck, augmentation; +8.92pp không phải head-only. Không có đo rank hoặc gradient chứng minh subspace collapse. Đã sửa cấu hình và bỏ cơ chế chưa đo.

### M7: 66.25% bị gắn nhầm PC-GTA H32/F24
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_results`
- **Confidence**: high
- **Section**: tables/components.tex:17
- **Related Sections**: tables/components.tex:17, tables/components.tex
- **Root Cause Key**: `i08`
- **Quote Verified**: yes
- **Quote**: `A6 & MobilePIXORNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F24 / SumFPN`
- **Explanation**: runs_summary.csv O32G16:66.247315/68.453192, H16/F32/SG-FPN/SimplifiedGT V2. A5=O32:H32/F32/SumFPN/SimplifiedGT V1. Vì vậy +6.03pp không đo PC-GTA. H32 hybrid thực sự68.124178; O32→H32=+7.906241pp, vẫn thay cả p/quota/placement recipe. Đã bỏ mọi +6.03 PC-GTA claim, ghi source-linked inventory.

### M8: A8 trộn AP P1 với tốc độ S32
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_results`
- **Confidence**: high
- **Section**: tables/components.tex:39
- **Related Sections**: tables/components.tex:39, tables/components.tex
- **Root Cause Key**: `i09`
- **Quote Verified**: yes
- **Quote**: `67.38 & 69.46 & 6.94 & 144.16`
- **Explanation**: P1 focal context:67.378677/69.460029 và7.772452ms/128.659530FPS. S32 no-focal SG-FPN:68.459941/70.283161 và6.936693ms/144.160918FPS. Bản cũ ghép hai run vào một hàng. Đã giữ A8=S32 và thêm A8F=P1. A7H32→A8S32 còn đổi quota nên không cô lập neck.

### M9: Ghép Q-OGA lịch sử với số tham số của model cuối
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: high
- **Section**: tables/loss.tex:19
- **Related Sections**: tables/loss.tex:19, tables/loss.tex
- **Root Cause Key**: `i11`
- **Quote Verified**: yes
- **Quote**: `saving $18{,}626$ parameters`
- **Explanation**: P1/P2/P11 dùng815743params; P3/P4 dùng852929params, chênh37186. Model cuối shared-head753569 khác kiến trúc. 69.20% là P11 lịch sử, không có bằng chứng ghép nó với734943params. Đã sửa bảng0.816/0.853M và tách rõ mô hình cuối.

### M10: Bảng seed không chỉ thay seed
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: high
- **Section**: tables/seed_robustness.tex:3
- **Related Sections**: tables/seed_robustness.tex:3, tables/seed_robustness.tex
- **Root Cause Key**: `i06`
- **Quote Verified**: yes
- **Quote**: `under a 100-epoch training schedule`
- **Explanation**: Cau hinh!T10=Binary35 của MobilePIXOR S42; T11/T12=hist14 của S43/S44. S42 archived38epoch, các seed còn lại100; Final S42 160epoch, S43/S44 100epoch. Mean/std vẫn là phép tính mô tả hợp lệ nhưng không đo initialization variance hoặc chứng minh numerical stability. Đã đổi caption/note và bỏ suy luận nhân quả.

## Moderate Issues

### M1: Phép cộng CPU statistics không tự bảo đảm CUDA sync không đáng kể
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_methods`
- **Confidence**: high
- **Section**: sections/methodology.tex:111
- **Related Sections**: sections/methodology.tex:111, sections/methodology.tex
- **Root Cause Key**: `i26`
- **Quote Verified**: yes
- **Quote**: `negligible thread synchronization`
- **Explanation**: Các cell độc lập sau group nhưng nhiều point có thể update cùngcell, cần grouping/reduction/atomic strategy. Chưa có đo CUDA hoặc asynchronouspipeline. Đã chuyển thành futureextension và nêu writeconflict.

### M2: Detached surrogate không loại hết gradient competition
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_methods`
- **Confidence**: high
- **Section**: sections/methodology.tex:383
- **Related Sections**: sections/methodology.tex:383, sections/methodology.tex
- **Root Cause Key**: `i16`
- **Quote Verified**: yes
- **Quote**: `Detaching $q^*$ prevents gradient competition`
- **Explanation**: q*=clipped mean projectionGIoU không phải exact polygon IoU. stopgrad chặn đường qua target, nhưng quality và geometry còn chung backbone/neck. Exact rotated IoU cũng không thể gọi là hoàn toàn nondifferentiable ở mọi cấu hình. Đã sửa mục tiêu surrogate và giới hạn gradient claim.

### M3: Soft-min corner thiếu định nghĩa D và có thể âm
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_methods`
- **Confidence**: high
- **Section**: sections/methodology.tex:399
- **Related Sections**: sections/methodology.tex:399, sections/methodology.tex
- **Root Cause Key**: `i27`
- **Quote Verified**: yes
- **Quote**: `\mathcal{D}_{\mathrm{smooth}} = -\tau_{\mathrm{soft}}`
- **Explanation**: D_direct/D_pi chưa được định nghĩa tường minh là tổng/trung bình khoảng cách theo mét trước normalize hay distance đã normalize. Code smooth_corner_loss.py dùng meancornerL1, softmin rồi chia diagonal. Unnormalized logsumexp-softmin có thể âm và không có exactzero baseline. Đây không tự động là lỗi optimizer, nhưng gọi distance/C-infinity là chưa chính xác khi D gồm abs. Giữ nguyên công thức/kết quả; cần làm rõ định nghĩa, đơn vị tau và intended behavior.

### M4: Tổng task weights không phải tổng gradient budget
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_methods`
- **Confidence**: high
- **Section**: sections/methodology.tex:416
- **Related Sections**: sections/methodology.tex:416, sections/methodology.tex
- **Root Cause Key**: `i15`
- **Quote Verified**: yes
- **Quote**: `strictly conserves the total gradient budget across all tasks`
- **Explanation**: Sum omega=J chỉ bảo toàn tổng trọng số. Gradient mỗi task còn phụ thuộc scale/direction và shared features; không suy ra gradient norm hoặc convergence. Đã sửa phát biểu, giữ nguyên công thức.

### M5: Doubled-angle không khôi phục directed heading liên tục
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_methods`
- **Confidence**: high
- **Section**: sections/methodology.tex:475
- **Related Sections**: sections/methodology.tex:475, sections/methodology.tex
- **Root Cause Key**: `i17`
- **Quote Verified**: yes
- **Quote**: `is fully recovered without trigonometric ambiguity`
- **Explanation**: cos2psi/sin2psi biểu diễn orientation modulo pi, không có front/back. Scalar atan2 representative vẫn wrap ở boundary và vector0 không có direction. Đã nêu đúng contract, giữ nguyên phương trình.

### M6: Related Work gán cơ chế LOS/shadow bị tắt cho kết quả chính
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_introduction_related`
- **Confidence**: high
- **Section**: sections/related_work.tex:11
- **Related Sections**: sections/related_work.tex:11, sections/related_work.tex
- **Root Cause Key**: `i14`
- **Quote Verified**: yes
- **Quote**: `integrating adaptive local ground plane fitting, static obstacle clearance filtering, line-of-sight ray checks, and shadow masking`
- **Explanation**: Experiments và Results ghi LOS, shadow, radiometric disabled. Đã mô tả core enabled mechanisms và optional options, không quy AP gain cho cơ chế không bật.

### M7: Factorial study không chứng minh positive synergy
- **Type**: claim_accuracy
- **Source**: [LLM] via `claims_vs_evidence`
- **Confidence**: high
- **Section**: sections/results.tex:121
- **Related Sections**: sections/results.tex:121, sections/results.tex
- **Root Cause Key**: `i12`
- **Quote Verified**: yes
- **Quote**: `These results demonstrate clear methodological synergy`
- **Explanation**: Từ mAP chưa làm tròn: OGA khôngIQA +1.588431pp; cóIQA +0.269471pp. Interaction=P4−P3−P2+P1=−1.318960pp. P4 cao nhất trong factorial không đồng nghĩa tương tác dương; chưa có uncertainty cho hiệu ứng nhỏ. Đã viết lại đúng main effects và interaction.

### M8: Hình lưu sẵn còn legend/configuration cũ
- **Type**: claim_accuracy
- **Source**: [LLM] via `pre_submission_readiness`
- **Confidence**: high
- **Section**: sections/results.tex:144
- **Related Sections**: sections/results.tex:144, sections/results.tex
- **Root Cause Key**: `i28`
- **Quote Verified**: yes
- **Quote**: `Trajectories illustrate stable monotonic descent`
- **Explanation**: training_curves PDF vẫn gắnA5=GroupedH32,A6=HybridGT/F24,A8=HybridGT/SGFPN trong khi đối chiếu source đã phát hiện mapping khác. Pareto PDF còn nối frontier qua khác evaluator/GPU. Đã sửa caption nhưng chưa tái tạo hình khi thiếu trace liên kết từng curve với run. Cần dựng lại từ per-run records trước nộp.

### M9: Monotonic descent mâu thuẫn validation spikes
- **Type**: claim_accuracy
- **Source**: [LLM] via `self_standard_consistency`
- **Confidence**: high
- **Section**: sections/results.tex:144
- **Related Sections**: sections/results.tex:144, sections/results.tex
- **Root Cause Key**: `i20`
- **Quote Verified**: yes
- **Quote**: `stable monotonic descent`
- **Explanation**: Results ghi classification spikes epochs18–25 rồi caption nói stable monotonic và rapidly stabilized byT-SBUW. Đã đổi thành generally decreasing trajectories; không quy nguyên nhân nếu không có weighting ablation.

### M10: Hình vehicle được dùng xác nhận VRU AP
- **Type**: claim_accuracy
- **Source**: [LLM] via `claims_vs_evidence`
- **Confidence**: high
- **Section**: sections/results.tex:166
- **Related Sections**: sections/results.tex:166, sections/results.tex
- **Root Cause Key**: `i25`
- **Quote Verified**: yes
- **Quote**: `This visual evidence corroborates the substantial`
- **Explanation**: Ví dụ vehicle heatmaps không xác nhận định lượngPed/Cyc30–50m. Whole-model visualization cũng không trực tiếpvalidate BlockSimAM/SGFPN effects hoặc NMSspeed. Đã giới hạn caption và observation vào selectedexamples.

### M11: BEV/edge latency bị suy rộng thành 3D và safety deployment
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_discussion_conclusion`
- **Confidence**: high
- **Section**: sections/results.tex:190
- **Related Sections**: sections/results.tex:190, sections/results.tex
- **Root Cause Key**: `i18`
- **Quote Verified**: yes
- **Quote**: `accurate 3D LiDAR detection was possible`
- **Explanation**: Output chỉ (x,y,l,w,yaw modulo pi), khôngz/h/velocity/tracking. KITTI/T4/L4 không chứng minh embedded/closed-loop collision avoidance hoặc TTC; P99 offline<100ms không phải worst-case guarantee. Đã giới hạn abstract/results/conclusion vào BEV và measured offline latency.

### M12: Far-range Pedestrian regression bị bỏ qua trong diễn giải
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_results`
- **Confidence**: high
- **Section**: sections/results.tex:81
- **Related Sections**: sections/results.tex:81, sections/results.tex
- **Root Cause Key**: `i22`
- **Quote Verified**: yes
- **Quote**: `These results are consistent with range-adaptive augmentation`
- **Explanation**: Table distance:PedFar0.99→0.80,−0.19pp; Results chỉ nhấn mạnhCyclistFar14.94 và VRUgain. Đã nêu regression và bỏ suy luận nhân quả sampler/kernel từ whole-model comparison.

### M13: Cơ chế tối ưu hóa chưa có thí nghiệm phân biệt
- **Type**: claim_accuracy
- **Source**: [LLM] via `claims_vs_evidence`
- **Confidence**: high
- **Section**: sections/results.tex:99
- **Related Sections**: sections/results.tex:99, sections/results.tex
- **Root Cause Key**: `i19`
- **Quote Verified**: yes
- **Quote**: `induces severe subspace rank collapse`
- **Explanation**: Không có feature singular values, gradient diagnostics hoặc matched width control. Các suy luận weak-return attenuation, synergy SG-FPN, T-SBUW suppresses spikes và epoch latency do sharpening đều vượt dữ liệu. Đã thay bằng quan sát AP/loss và ghi điều kiện kiểm chứng.

### M14: AP L4 và T4 của cùng tên model bị gộp
- **Type**: claim_accuracy
- **Source**: [LLM] via `notation_and_numeric_consistency`
- **Confidence**: high
- **Section**: tables/comparison.tex:14
- **Related Sections**: tables/comparison.tex:14, tables/comparison.tex
- **Root Cause Key**: `i04`
- **Quote Verified**: yes
- **Quote**: `86.90 & \textbf{60.78} & \underline{66.81}`
- **Explanation**: Seed workbook: L4 86.900441/60.778199/66.807415, mAP 71.495352, mAP9 73.445159. V7 T01_Benchmark row16: T4 86.903460/60.749361/66.732802, mAP 71.461874, mAP9 73.418711. Giữ cả hai snapshot với nhãn nguồn; chưa thể khẳng định khác biệt chỉ do GPU. Cần checkpoint hash/prediction hash.

### M15: Timing MobilePIXOR và một số percentile không khớp snapshot
- **Type**: claim_accuracy
- **Source**: [LLM] via `notation_and_numeric_consistency`
- **Confidence**: high
- **Section**: tables/efficiency.tex:11
- **Related Sections**: tables/efficiency.tex:11, tables/efficiency.tex
- **Root Cause Key**: `i21`
- **Quote Verified**: yes
- **Quote**: `90.65 & 11.03`
- **Explanation**: V7B10: MobilePIXOR historicalL4 model11.049958/E2E85.923112/P50 84.114180/P95 112.524565/P99 124.943548. PointRCNN P50/P99=194.769222/222.372998; PartA2=132.399653/169.129394; PVRCNN=187.348769/219.072451. Đã sửa các ô có nguồn xác định, bỏ same-T4 claim. Không suy speedup L4/T4.

### M16: AP từng lớp của Baseline và Q-OGA không khớp nguồn
- **Type**: claim_accuracy
- **Source**: [LLM] via `notation_and_numeric_consistency`
- **Confidence**: high
- **Section**: tables/loss.tex:10
- **Related Sections**: tables/loss.tex:10, tables/loss.tex
- **Root Cause Key**: `i10`
- **Quote Verified**: yes
- **Quote**: `83.92 & 55.48 & 62.74`
- **Explanation**: P1 nguồn:82.970232/56.992056/62.173743 →82.97/56.99/62.17; P11 nguồn:84.893017/57.145242/65.563877 →84.89/57.15/65.56. Các bộ số cũ vẫn cho cùng mAP làm tròn nên chỉ kiểm mean sẽ bỏ sót. Đã sửa từ paper_06_Loss_IQA.csv và workbook v7 T02.

## Minor Issues

### M1: Data flow và index conventions chưa đủ rõ
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_methods`
- **Confidence**: high
- **Section**: sections/methodology.tex:20
- **Related Sections**: sections/methodology.tex:20, sections/methodology.tex
- **Root Cause Key**: `i29`
- **Quote Verified**: yes
- **Quote**: `A point $\mathbf{p}_i$ maps to grid coordinate $(u_i, v_i)$ as:`
- **Explanation**: u làx/column704, v lày/row800; shapeCHW dùngrowv,columnu. Methods trình bàyStrata trướcPCGTA dễ hiểu nhầm augmentation trên BEV. Đã nêu trainpoints→PCGTA→rasterize và inferrawpoints→rasterize, cùng tensor indexing. Final heads share stem within eachgroup chứ không share initialgroupblock giữahai nhóm.

### M2: RDA được gọi density-adaptive nhưng công thức chỉ dùng range
- **Type**: claim_accuracy
- **Source**: [LLM] via `section_methods`
- **Confidence**: high
- **Section**: sections/methodology.tex:402
- **Related Sections**: sections/methodology.tex:402, sections/methodology.tex
- **Root Cause Key**: `i30`
- **Quote Verified**: yes
- **Quote**: `Range and Density-Adaptive (RDA) weighting`
- **Explanation**: w(r) chỉ córange, không có localdensityinput. Đã gọi range-adaptive weighting denotedRDA; chưa có ablation cho densityadaptation.

### M3: Serialized checkpoint không phải kích thước trọng số FP32 thuần
- **Type**: claim_accuracy
- **Source**: [LLM] via `notation_and_numeric_consistency`
- **Confidence**: high
- **Section**: sections/methodology.tex:485
- **Related Sections**: sections/methodology.tex:485, sections/methodology.tex
- **Root Cause Key**: `i24`
- **Quote Verified**: yes
- **Quote**: `with an FP32 checkpoint size of $9.34$`
- **Explanation**: 753569*4bytes=3.014276MB; serializedcheckpoint9.34MB có thể chứa thêmstate/metadata. Mainabstract cũ gọi753569trainable nhưng12T-SBUW logits nâng tổngtrainable753581. Đã thống nhất deployable/train-only và serializedsize, không khẳng định thành phần checkpoint khi chưa có inspection.

### M4: Delta_z 0.1 m không mô tả Strata-BEV14
- **Type**: claim_accuracy
- **Source**: [LLM] via `notation_and_numeric_consistency`
- **Confidence**: high
- **Section**: tables/setup.tex:11
- **Related Sections**: tables/setup.tex:11, tables/setup.tex
- **Root Cause Key**: `i23`
- **Quote Verified**: yes
- **Quote**: `Grid resolution ($\Delta_x, \Delta_y, \Delta_z$)`
- **Explanation**: Strata14 dùng4heightbins trên3.5m, còn dz0.1 thuộc legacy35slices. Đã tách horizontalgrid0.1x0.1 vớifourverticalstrata; không đổi công thức.

## Decision Signals

- **Reviewer Recommendation**: Reject
- **Issue Bundle**: 10 major / 16 moderate / 4 minor

## Revision Roadmap

### Priority 1 --- Must Address (Blocking)

- [ ] Abstract and conclusion claims need explicit evidence traceability ([LLM]; abstract)

### Priority 2 --- Strongly Recommended

- [ ] Novelty claim should be grounded against the closest prior work ([LLM]; abstract)
