# Phase 3 — Adversarial Claims-vs-Evidence Reconciliation

**Bản thảo:** MobilePIXORNeXt, target AJSE. **Ngày:** 2026-10-22. **Gate:** `FAIL_CLAIM_GATE / MAJOR_REVISION`. Hai chuỗi đóng góp có **CRITICAL BLOCKER** theo quy tắc bốn mắt xích của yêu cầu này: **T-SBUW (P9/C35)** và **Q-OGA (P10/C39)**.

Ma trận có **91 dòng**: **23 VERIFIED**, **38 METRIC_MISMATCH**, **16 OVERCLAIM**, **14 UNSUPPORTED**. Đây là số dòng đối soát, không phải số lỗi độc lập; những cách viết lặp lại được gom theo root cause trong JSON. Có 44 nhóm cần xử lý, bao gồm các lỗi tài liệu nguồn, đơn vị và các giới hạn bằng chứng ở nhiều mức độ.

Lượt review dùng **sequential single-agent**, theo [paper-audit](/home/duyennh/.agents/skills/paper-audit/SKILL.md), [Claim-Evidence Contract](/home/duyennh/.agents/skills/latex-paper-en/references/evidence/claim-evidence-contract.md), [Over-Claim Guard](/home/duyennh/.agents/skills/latex-paper-en/references/evidence/over-claim-guard.md) và quy tắc cùng tên của audit lane. Dùng thêm [pdf](/home/duyennh/.agents/skills/pdf_skill/SKILL.md) để đọc figure assets. Chỉ ghi báo cáo và artifact; các đoạn sửa mẫu bên dưới **chưa được áp dụng vào manuscript**.

## 1. Phạm vi và nguyên tắc phân xử

Không có `Paper/main.tex` trong workspace này. Entry thực tế là [MobilePIXORNeXt.tex](../MobilePIXORNeXt.tex); `Paper` là symlink tới `paper`. Đã mở toàn bộ entry và graph include **18 file**, gồm Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion, Declarations và các bảng. Đã đọc trọn [code_context.md](../code_context.md), và trích text từ **10/10 PDF figures được dùng**; xem trực tiếp hình Pareto, PR và loss để đối chiếu với caption. Các dòng nguồn trong ma trận là dòng thật trong file con, không phải số dòng ghép giả định của `main.tex`.

Audit giữ nguyên snapshot đã dùng ở Phase2; 20 file nguồn `.tex`/context/bibliography được hash trước–sau. Nguồn định lượng bổ sung là canonical benchmark logs và resolved final config đã có trong repo. Không huấn luyện, không chạy lại inference/timing, không tạo p-value/CI và không tra cứu external literature. Các statement về prior art thiếu original-paper verification được ghi `UNSUPPORTED` trong phạm vi local audit, không được kết luận là sai về nội dung.

**Định nghĩa status:**

- `VERIFIED`: claim được hỗ trợ trong đúng phạm vi dữ liệu/thiết kế được nêu; với design identity có thể dùng công thức/code, không bắt buộc bịa một bảng AP cho fact cấu trúc.
- `METRIC_MISMATCH`: không thống nhất số, đơn vị, metric/protocol, scope đo, backend/tensor metadata hoặc figure legend. Cột bằng chứng chỉ rõ bên nào lệch. Một số dòng là **context cũ trong khi paper đúng**, không phải đề nghị thay paper bằng số cũ.
- `OVERCLAIM`: bằng chứng có thật nhưng câu văn khẳng định rộng hoặc mạnh hơn nó cho phép.
- `UNSUPPORTED`: thiếu phép đo, control, kiểm tra nguồn trích dẫn hoặc mắt xích đóng góp cần thiết. Hedges như “may”/“consistent with” không tự tạo bằng chứng cho một giải thích cơ chế.

Nhãn `ORPHAN_CONTRIBUTION — CRITICAL BLOCKER` là lớp gate bổ sung trên status `UNSUPPORTED`, theo yêu cầu người dùng. JSON theo `ISSUE_SCHEMA.md` vẫn giữ severity chuẩn `major|moderate|minor`; hai orphan dùng `major`, `gate_blocker:true`, kèm `user_gate_label:CRITICAL BLOCKER`.

## 2. Các kết luận ảnh hưởng trực tiếp đến headline

1. **Phép trừ các headline đúng ở độ chính xác của bảng:** 71.46−70.13=1.33; 60.75−56.01=4.74; 60.75−56.56=4.19; 66.25−60.22=6.03; 70.58−67.38=3.20. Chúng là **điểm phần trăm (pp)**. Ví dụ1.33pp tương ứng khoảng1.89648% relative so với70.13, không phải1.33% relative. Không có cơ sở tố các phép trừ này tính sai; lỗi chính là đơn vị và trần diễn giải.
2. **Final-result snapshots không thống nhất:** context §6.1 ghi71.46/73.42, §6.6 ghi71.50/73.45. Final canonical log xác nhận71.46/73.42 sau rounding. Vì thế attention delta theo paper là.88/.76pp, còn dùng context R37 cũ sẽ thành.92/.79pp. Cần định danh run/checkpoint/protocol cho mỗi snapshot.
3. **SECOND reduction bị double rounding:** raw counts753569 và5325576 cho85.850000075…%, làm tròn trực tiếp thành85.9%(1dp) hoặc85.85%(2dp). Tính trên số đã làm tròn .754M/5.326M cho85.8430…%→85.8%. Mẫu sửa dùng85.85%, không đổi qua lại giữa hai cơ sở. PointPillars raw4834888 cho84.41393058…%→84.41% hoặc84.4%.
4. **“Matched SECOND” không được chứng minh:** own71.46 thấp hơn72.09 trong bảng .63pp; không có equivalence margin/test. CùngR40 và split chưa làm hai evaluator tương đương. Tuy nhiên PR asset **đã có shared-local-evaluator comparison**; audit này ghi nhận bằng chứng đó, và chỉ ra nó dùng baseline AP khác bảng chính, thay vì nói hoàn toàn không có matched-protocol evidence.
5. **PR figure/caption/table xung đột:** bảng dùng Pedestrian56.01/56.56; legends trong asset dùng55.83/56.47 cho PointPillars/SECOND. Cột/row và màu của caption cũng sai. Không thể dùng figure này để minh chứng trực tiếp margin4.74/4.19pp từ bảng mà không nêu khác protocol.
6. **Loss claims vượt quá hình:** asset có validation/classification spikes, và A9 thấp hơn A10 ở nhiều regression/quality panels. “No oscillation” và “lowest ... across tasks” không phù hợp với hình đang dùng.
7. **“Real-time” không tự sai với kết quả này:** mean16.41FPS>10Hz,60.96ms<100ms trênT4. Sai phạm vi xuất hiện khi suy thành embedded deployment, hard deadline hoặc robustness chưa đo. P99=95.91ms không chứng minh maximum latency≤100ms.

## 3. Ma trận đối soát tuyên bố – bằng chứng (đúng 6 cột)

Mỗi quote là một trích đoạn nguyên văn liên tục, được tìm lại bằng script trong assembled manuscript và đúng file/dòng. Các cách diễn đạt cùng claim ở section khác được chỉ vị trí ở cột2. Bảng bao phủ claim families của bốn phần được yêu cầu và mở rộng tới Results/caption/Related Work khi chúng chứa nguồn của headline hoặc bất đồng trực tiếp. Metadata hành chính, research questions và future-work được phân biệt với tuyên bố về kết quả đã đạt được.

| STT | Vị trí & Câu tuyên bố gốc (Claim in Abstract/Intro/Method/Conclusion) | Bằng chứng định lượng trong Bảng/Hình của Paper | Số liệu gốc trong `Paper/code_context.md` | Trạng thái | Cách sửa câu văn hoặc bổ sung số liệu |
|---|---|---|---|---|---|
| C01 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`a compact, proposal-free Bird's-Eye View (BEV) detector with 753,569 parameters.`<br>Cùng nhóm: Intro L11,15,19; Conclusion L7 | tab:comparison .754M; Methods L459–464:604864+36784+111921=753569; log confirms. | L23,229–236:753569 detector params +12 training-only logits. | `VERIFIED` | Giữ 753,569 detector parameters; phân biệt criterion training-only 12 logits. |
| C02 | [sections/conclusion.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:7)<br>`($9.34\,\text{MB}$ checkpoint)`<br>Cùng nhóm: Intro L19; Experiments L43 | tab:setup và Methods L459; canonical log:9338383bytes≈9.34MB. | L23 và §3.3:9.34MB,753569params. | `VERIFIED` | Giữ checkpoint file size 9.34MB; không gọi là raw FP32 weights hoặc training memory. |
| C03 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`a parameter-free deterministic NumPy/Numba CPU rasterizer.`<br>Cùng nhóm: Intro L11; Conclusion L5 | tab:strata_bev14 caption/Methods L96,122,tab:setup:CPU NumPy/Numba. | L59 nói parallel CUDA, trái CPU; §4.1 có14 deterministic formulas. | `METRIC_MISMATCH` | Giữ CPU theo implementation final; sửa context L59. Đây là implementation/backend mismatch ở context, không phải benchmark GPU raster đã được chứng minh. |
| C04 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`It extracted 14 statistical feature channels encoding stratified height distributions, elevation moments, surface reflectance, and continuous subcell centroid displacement offsets.`<br>Cùng nhóm: Intro L11,15; Conclusion L5 | tab:strata_bev14:4density+4height+2reflectance+logdensity+occupancy+2offset=14;fig:strata_bev14. | L248–262 đúng14features. | `VERIFIED` | Giữ mô tả feature statistics; không nâng thành khôi phục toàn bộ point geometry. |
| C05 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`Measured input preprocessing took 20.94 milliseconds, including point-cloud loading and rasterization.`<br>Cùng nhóm: Intro L11; Methods L122 | Results L50; log preprocess20.939096ms→20.94; đây là loading+rasterization. | §6.3 L446:20.94ms;L59 gọi raster riêng under21ms/CUDA là diễn giải sai. | `VERIFIED` | Giữ loading and CPU rasterization; không viết riêng raster kernel đạt20.94ms. |
| C06 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`71.46\% Moderate mean Average Precision (mAP)`<br>Cùng nhóm: Intro L19; Conclusion L7 | tab:comparison/tab:components A10/tab:attention:71.46; canonical raw71.461874…→71.46. | §1.1/§6.1 L28,413:71.46;§6.6 L497 và L517:71.50. | `METRIC_MISMATCH` | Giữ71.46 cho canonical final benchmark; gắn run/checkpoint/protocol cho71.50 hoặc bỏ số cũ khỏi context. Không đổi headline thành71.50. |
| C07 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`73.42\% overall 9-cell mAP`<br>Cùng nhóm: Intro L19; Conclusion L7 | tab:comparison/attention:73.42; canonical raw73.418711…→73.42. | §6.1 L413:73.42;§6.6 L497/L517:73.45. | `METRIC_MISMATCH` | Giữ73.42, tách snapshot73.45; mAP9 là9class×difficulty cells, không phảiIoU.9. |
| C08 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`Pedestrian Moderate AP reached 60.75\%.`<br>Cùng nhóm: Intro L19; Conclusion L7 | tab:comparison:60.75;PR figure own legend60.75;canonical log matches. | L413,417,470:60.75. | `VERIFIED` | Giữ giá trị cùng BEV/Moderate/R40/KITTI validation boundary. |
| C09 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`It outperformed PointPillars (70.13\% mAP)`<br>Cùng nhóm: Intro L19; Conclusion L7; Results L19 | tab:comparison:71.46 vs70.13;note:local vsreference evaluator. PR figure shared-local PP mean≈70.27,không70.13. | L411/413/416:70.13,71.46,+1.33;không cung cấp evaluator equivalence. | `OVERCLAIM` | The reported Moderate mAP values were 71.46% and70.13% under their respective evaluators; the1.33-point difference is descriptive. Muốn superiority:chung evaluator và seeds/uncertainty phù hợp. |
| C10 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`matched SECOND (72.09\% mAP)`<br>Cùng nhóm: Intro L19; Conclusion L7; Results caption L12 | tab:comparison:ours71.46<72.09,−.63pp;PR shared-local SECOND mean≈72.18;khôngequivalence test. | L416 thừa nhận trailing0.63. | `OVERCLAIM` | The reported score was0.63 percentage points below SECOND, with substantially fewer parameters; disclose protocol difference. Không dùng matched/equivalent. |
| C11 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`an 84.4\% parameter reduction`<br>Cùng nhóm: Intro L19; Conclusion L7 | tab:comparison .754M vs4.835M;raw753569/4834888→84.41393058%→84.4%. | L23,416:84.4%. | `VERIFIED` | 84.4% fewer detector parameters, hoặc84.41% nếu thống nhất2decimal. |
| C12 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`85.8\% fewer parameters`<br>Cùng nhóm: Intro L19; Conclusion L7 | Raw753569/5325576→85.850000075%→85.9%(1dp),85.85%(2dp). Rounded table .754/5.326→85.8430%→85.8%. | L23:85.8%;L412 dùng5.326M,khôngexact count. | `METRIC_MISMATCH` | Dùng85.85% từ raw counts, hoặc85.9% một chữ số; công bố cơ sở làm tròn.85.8% tái tạo từ bảng đã làm tròn nhưng không phải round-once raw count. |
| C13 | [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19)<br>`($7.1\times$ parameter reduction)`<br>Cùng nhóm: Conclusion L7; Results L19 | Raw5325576/753569=7.06713785→7.1;tab:comparison supports. | SECOND/ours counts ratio≈7.1 từL412–413. | `VERIFIED` | Use approximately one-seventh as many parameters (7.07× smaller parameter count); không viết7.1× fewer theo nghĩa phép trừ. |
| C14 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`The complete pipeline ran in 60.96 milliseconds (16.41 frames per second) on an NVIDIA Tesla T4 GPU,`<br>Cùng nhóm: Intro L19; Conclusion L7 | tab:efficiency:60.96ms/16.41FPS;log60.9563246ms và16.4051886FPS;3759timedframes,batch1FP32. | L27,413,446:khớp. | `VERIFIED` | Giữ sequential raw-file-input-to-detections mean latency trên T4;FPS lấy cùng unrounded mean,khôngmodel-only. |
| C15 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`the fastest end-to-end throughput among the evaluated real-time detectors.`<br>Cùng nhóm: Intro L19; Conclusion L7 | tab:efficiency:T4ours16.41>PP15.66>SECOND10.67;historicalMobilePIXOR11.64 trênL4. | L418 cùng nhận định,nhưngMobilePIXOR L4 ởL410. | `VERIFIED` | Giữ giới hạn among the evaluated detectors trong recorded benchmark; tốt hơn viết lowest measured mean E2E latency in the T4 cohort. Không suy universal SOTA/edge deployment. |
| C16 | [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)<br>`accurate LiDAR perception on resource-constrained embedded systems.`<br>Cùng nhóm: Methods L12; Intro L9,11; Conclusion L5 | tab:efficiency đoTeslaT4;khôngembedded target,watt/energy,thermal/INT8/ROS deployment test. | L386–389:T4/98.44MB,suitability làsuy rộng. | `OVERCLAIM` | These results motivate evaluation on embedded platforms; the present measurements are from a desktop/server GPU pipeline. |
| C17 | [sections/conclusion.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:5)<br>`a compact, proposal-free 3D LiDAR object detector`<br>Cùng nhóm: Intro L11; Methods L12,415; Results L192 | Methods Eq.target_tuple/Algorithm:chỉ x,y,l,w,yaw;tab:comparison BEV;khôngprimary3DAP. | L425–433:oursN/A3D,BEVonly;optionalvertical ởL121. | `OVERCLAIM` | a compact BEV object detector operating on3D LiDAR point clouds; giữ limitation full3D chưađánhgiá. |
| C18 | [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19)<br>`Empirical validation and leading edge Pareto efficiency.` | fig:pareto_tradeoff/tablecomparison:ours nondominated trong observed set;khácAPprotocol/hardware. Parameter-frontier line đi quaSECOND dùPointRCNN(4.039M,73.38)dominatesSECOND(5.326M,72.09). | §6.1 chỉ5models;khôngsearchtoànliterature. | `OVERCLAIM` | Empirical accuracy–parameter–latency trade-offs on KITTI validation. Có thể nói observed non-dominated point nếu giữ protocol caveats. |
| C19 | [sections/introduction.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:7)<br>`Neither combines continuous subcell displacements with continuous vertical spatial moments.`<br>Cùng nhóm: Related Work L6 | ChỉcitesPillarHist/TriBand;encodingtable đo3channelvariant,khôngxác minh nội dung từngpaper. | L503–508 soencoding variants;khôngprior-art claim verification. | `UNSUPPORTED` | Nêu thành phần mà paper này combines; kiểm tra original references trước khi khẳng định neither. Không tự thêm first/novel. |
| C20 | [sections/introduction.tex:15](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:15)<br>`without learned PointNet encoders or 3D sparse convolutions.`<br>Cùng nhóm: Abstract; Methods L12; Conclusion L5 | fig:r37_overview,tab:strata_bev14,backbone/tableconfig;codefinal2Dconv. | L24,58,220 xác nhận design. | `VERIFIED` | Giữ structural statement; không đồng nhất encoder-free với zero preprocessing cost. |
| C21 | [sections/methodology.tex:122](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:122)<br>`Both compute the same fourteen deterministic features without learned parameters.`<br>Cùng nhóm: Abstract; Intro L11; Conclusion L5 | tab:strata_bev14 vàrasterformulas define14ch;khôngquantlearner. | §4.1 L248–262:khớp. | `VERIFIED` | Giữ cùng feature definitions; không hàm ý bitwise-identical mọibackend nếu chưa differential test. |
| C22 | [sections/methodology.tex:261](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:261)<br>`Block-SimAM adds \textbf{zero parameters}`<br>Cùng nhóm: Abstract; Intro L15; Conclusion L5 | tab:attention caption;source SimAMCore khôngParameter;Phase2countsamearchitectureoff/on đều753569. | L71,281:0learnedparams. | `VERIFIED` | Giữ zero additional trainable attention parameters; có thêm operations/latency. |
| C23 | [sections/methodology.tex:235](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:235)<br>`for nine attention operators in total.`<br>Cùng nhóm: Intro L11; Experiments L43 | fig:backbone_f32/tab:setup depths3+4+2=9;resolved finalconfigmatches. | L70,159,215,517 nói8 hoặcdepth2/4/2. | `METRIC_MISMATCH` | Giữ9(final);context8 phảigắn lịch sử2/4/2. Không sửa final về8. |
| C24 | [sections/methodology.tex:291](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:291)<br>`\mathbf{F}_{\mathrm{out}} \in \mathbb{R}^{B \times 32 \times 200 \times 176}`<br>Cùng nhóm: Intro L11,15; Experiments L43 | fig:backbone_f32/tab:setup/head32;resolvedconfig F32,out32,H32. | L182–198,222–225 nóiout16/head16,trongL75 lạiF32. | `METRIC_MISMATCH` | Giữ final32; version context và channel contract. |
| C25 | [sections/methodology.tex:285](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:285)<br>`depthwise $3 \times 3$ refinement, BN, and SiLU produce $\mathbf{U}_4$ and $\mathbf{U}_3$`<br>Cùng nhóm: Intro L11; Methods caption L186,276 | fig:backbone_f32 thực ghiDW48→48 rồiConv48→32;codeproj_u3 thường,khôngDW. | L177–179 phân biệtRefine/Proj nhưngL73 diễnđạtchungDW. | `METRIC_MISMATCH` | Nêustride8 depthwise48→48;stride4 standardConv48→32. Sửa cùngcaption SG-FPN vàIntro. |
| C26 | [sections/introduction.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:11)<br>`avoiding deconvolution checkerboard artifacts.`<br>Cùng nhóm: Methods L276; Results L106 | tab:components A7→A8:68.12→67.38,−.74pp;fig:scale_gated_fusion làschematic,khôngartifacterror metric. | L73 nóieliminates;L515 gộpnhầmwidth/gategain. | `OVERCLAIM` | uses bilinear upsampling in place of transposed convolution; intended to avoid that particular upsampling mechanism. Chưaclaim artifact-free/suppressnoiseproved. |
| C27 | [sections/methodology.tex:192](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:192)<br>`a $7 \times 7$ depthwise receptive field spans $0.7 \times 0.7$\,m at stride 1.`<br>Cùng nhóm: Intro L9,11 | Kernel-footprint arithmetic7×.1=.7m;stride8 footprint7×8×.1=5.6m;fig:backbone. | L67,203–206 cùngvalues. | `VERIFIED` | Gọi nominal kernel footprint; không nhầm với accumulated effective receptive field của toàn network. |
| C28 | [sections/methodology.tex:207](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:207)<br>`The backbone achieved $55.94$\,FPS on Tesla T4` | tab:efficiency55.94 làwhole model+heads,17.88ms;backbone604864 chỉmộtcomponent. | L26,446 cũngwhole model. | `METRIC_MISMATCH` | The complete detector achieved55.94FPS for in-pipeline model forward; chưacóbackbone-onlytiming. |
| C29 | [sections/methodology.tex:466](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:466)<br>`standalone network throughput reached $55.94$\,FPS` | tab:components/efficiency note:standalone115.32FPS,8.67ms;pipelineforward55.94FPS,17.88ms. | L26 phânbiệtđúng2regimes. | `METRIC_MISMATCH` | Đổi55.94 thành in-pipeline model forward; giữ standalone riêng115.32 với raw timing protocol cần xác minh. |
| C30 | [sections/methodology.tex:128](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:128)<br>`It improved Moderate mAP by $+6.03\%$ over naive sampling in the ablations.`<br>Cùng nhóm: Intro L19; Results L102; Conclusion L7 | tab:componentsA5=60.22,A6=66.25:6.03pp;relative10.0132846%. | L64,486–487,514:+6.03 nhưnggọibằng%. | `METRIC_MISMATCH` | The reported progression increased Moderate mAP from60.22% to66.25%,a6.03-percentage-point change; controlled-effect qualification cần manifests. |
| C31 | [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19)<br>`OGA and IQA added \textbf{+3.20\% mAP}`<br>Cùng nhóm: Results L110; Conclusion L7 | tab:componentsA8=67.38,A9=70.58:3.20pp;lossfactorial full. | L493,516:70.58−67.38=3.20. | `METRIC_MISMATCH` | OGA+IQA row was3.20percentage points abovebaseline; package-level,khôngmỗimodule3.20. |
| C32 | [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19)<br>`Block-SimAM added \textbf{+0.88\% mAP} without additional parameters.`<br>Cùng nhóm: Methods L261; Results L115–117,134 | tab:attention70.58→71.46=.88pp;off/onparameter-freeclaim đúng;historicalrowidentity chưa khóa. | L493→497:70.58→71.50=.92pp,mAP9 .79 so paper .76. | `METRIC_MISMATCH` | The reported A9–A10 difference was.88percentage points; reconcile context snapshot and architecture/schedule before isolated causal claim. |
| C33 | [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19)<br>`In controlled ablations,`<br>Cùng nhóm: Conclusion L7; Tables encoding/loss | tab:loss noattention.721M vsfinal.753569M;zero-paramSimAM cannot explain delta;curves50 vsfinal160epochs. | §6.6 progression khôngmanifest từngrun;depth8 old vsfinal9. | `OVERCLAIM` | In the reported architectural progression; chỉkhôi phục controlled sau config/hash,budget,seed,selectionpolicy matched. |
| C34 | [sections/introduction.tex:17](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:17)<br>`Optional visibility, shadow-masking, and intensity-adjustment heuristics address additional synthesis artifacts.` | Khôngablation từngoptionalmechanism;Experiments L49:disableLOS/shadow/radiometry;visibilityflagguardbypassfinal. | L62,514 gángainchoLOS/shadowtoànpackage,khôngđúngfinaltoggles. | `UNSUPPORTED` | The sampler provides optional checks; their individual benefits were not evaluated in the reported final configuration. Giữ chúng làcapability,khôngvalidated contribution. |
| C35 | [sections/introduction.tex:17](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:17)<br>`12-parameter \textbf{T-SBUW} balances task objectives,`<br>Cùng nhóm: Conclusion L5; Methods L384 | MethodsL384–401 formula12logits;fig:loss_curves khônguniform/EMA-only/fullT-SBUW ablation. | L328–339 cóformulation nhưng§6.6khôngrowtáchweighting. | `UNSUPPORTED`<br>**ORPHAN_CONTRIBUTION — CRITICAL BLOCKER** | T-SBUW defines adaptive weights; its isolated contribution was not measured. Nếugiữđónggópchính,thêmmatched ablation. |
| C36 | [sections/methodology.tex:395](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:395)<br>`This construction strictly conserves the total weight.` | Eq.tsbuw_weights:sumω=J=6 bằngsoftmaxnormalization;proofalgebra,khôngcầnbảngAP. | L333:sumω=6. | `VERIFIED` | Giữ total task weight; không nâng thành conserved gradient norm hoặc guaranteed convergence. |
| C37 | [sections/methodology.tex:229](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:229)<br>`to ensure numerical stability during deep backpropagation.` | LayerScaleinit1e-5 ởMethods;khônggradientstats/nonfinite-rate/controlledstabilitytest. | L272 cũngensuringstableconvergence nhưngkhôngsupportingstats. | `OVERCLAIM` | to initialize residual updates at a small scale; stability làdesign motivation,khôngguarantee. |
| C38 | [sections/methodology.tex:368](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:368)<br>`Detaching the target limits gradient instability while retaining supervision for confidence calibration.` | Eq.iqa_loss stopgrad đúngchặngradientquaqualitytarget;khôngcalibration ECE/reliability/stability ablation. | L326–327:BCEwithdetachedproxy. | `UNSUPPORTED` | Detaching the target prevents gradients from propagating through the target-generation path. Chưađịnhlượnggradientstability/calibration improvement. |
| C39 | [sections/methodology.tex:371](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:371)<br>`We developed \textbf{Q-OGA} as a head-free alternative for resource-constrained platforms.`<br>Cùng nhóm: Related Work L24; Results L127 | tab:loss cóQOGA69.20/.702M/6.94ms;Methods cósectionnhưngIntroduction contributions/ConclusionkhôngtómQ-OGA. | §4.4.3 và§6.6 L491:69.20. | `UNSUPPORTED`<br>**ORPHAN_CONTRIBUTION — CRITICAL BLOCKER** | Nếu Q-OGA là đóng góp mới: thêm Intro và Conclusion cùng bounded result; hoặc ghi rõ auxiliary comparison, không phải main contribution. |
| C40 | [sections/methodology.tex:371](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:371)<br>`adding $18.59\,\text{K}$ parameters`<br>Cùng nhóm: Methods L381; Results L127; tab:loss note | tab:loss .721−.702=.019M saurounding,khôngsuyexact18.59K;final-source counterIQA=18626=18.626K. | §3.3heads111921 nhưngkhôngexactIQA-onlydelta;historicalconfigs chưakhóa. | `METRIC_MISMATCH` | Khôngkhẳngđịnh18.59K cho final. Côngbốexactper-run counts;finalsame-topologybranchdelta18,626,nhưngkhôngghiđèhistoricalrows chưaresolve. |
| C41 | [sections/methodology.tex:381](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:381)<br>`Q-OGA reduced forward latency by $1.25\,\text{ms}$`<br>Cùng nhóm: Results L127 | tab:loss8.19−6.94=1.25ms;đúngtableonly. | §6.6IQA122.16FPS→8.186ms,QOGA133.97FPS→7.465ms,delta≈.722ms;khác144.16FPSpaper. | `METRIC_MISMATCH` | The standalone table reports1.25ms; reconcile different run/device/timing snapshots in context before attributing saving solely to removing IQA. |
| C42 | [sections/results.tex:127](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:127)<br>`Q-OGA achieved $69.20\%$ Moderate mAP, exceeding OGA alone by $+0.23\%$ and baseline Smooth-L1 by $+1.82\%$.` | tab:loss69.20−68.97=.23pp;69.20−67.38=1.82pp. | L489–491:67.38,68.97,69.20. | `METRIC_MISMATCH` | Giữ69.20%;gains.23 và1.82percentage points. QOGA cònđổiRDA/softmin,khôngisolatequalitycoupling alone. |
| C43 | [sections/methodology.tex:371](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:371)<br>`The explicit IQA branch gave the highest ranking accuracy,` | tab:lossđoAP fullobjectives;khôngrankaccuracy/AP-atfixedranking/ECE/qualitycorrelationmetric. | §6.6IQA-only70.31,OGA+IQA70.58;khôngrankingaccuracymeasure. | `OVERCLAIM` | The OGA+IQA configuration had the highest Moderate mAP among the configurations in the loss-factorial table. |
| C44 | [sections/methodology.tex:360](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:360)<br>`Normalizing by the target bounding box diagonal ensures scale invariance across categories.` | Eq.corner_loss numerator/GTdiagonal scalescancel khi predicted/targetgeometry cùngscale,floorinactive. | L320–324 cùngformula. | `VERIFIED` | Giữ scale-normalized distance; qualify invariance under common uniform scaling away fromepsilon/logclamps. Khôngđồngnhấtclass-independentAP. |
| C45 | [sections/methodology.tex:371](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:371)<br>`continuous bounding box IoU quality $q^*$`<br>Cùng nhóm: IQA head descriptions/Conclusion L5 | Eq.iqa_loss targetclip(mean4axisGIoU,0,1),khôngexactrotatedareaIoU;Phase2squarewitnessproxy.5 vsIoU.25. | L319/327 gọiMGIoUsimilarity. | `OVERCLAIM` | detached projection-GIoU quality proxy; khôngclaim exactIoU calibration nếu chưađocorrelation/reliability. |
| C46 | [sections/results.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:19)<br>`a $+1.33\%$ margin`<br>Cùng nhóm: Conclusion L7 | tab:comparison71.46−70.13=1.33pp;relative1.896478%. | L416:+1.33%,arithmeticmatchnhưngunitsmơhồ. | `METRIC_MISMATCH` | a1.33-percentage-point reported difference; retain evaluator caveat. |
| C47 | [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19)<br>`(+4.74\% over PointPillars, +4.19\% over SECOND)`<br>Cùng nhóm: Conclusion L7; Results L30 | tab:comparison60.75−56.01=4.74pp;60.75−56.56=4.19pp;PRfigurebaselines55.83/56.47→4.92/4.28. | L411–417:referencevalues56.01/56.56. | `METRIC_MISMATCH` | 4.74 and4.19percentage points from reference-table values; đừngghép vớiPRshared-localnumbers. Chọnđồngnhấtprotocol trước infer superiority. |
| C48 | [sections/results.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:19)<br>`by $+30.06\%$ ($71.46\%$ vs. $41.40\%$) and by $+31.15\%$` | tab:comparison71.46−41.40=30.06pp;73.42−42.27=31.15pp. | §6.1L410,413:cùngvalueslocalR40. | `METRIC_MISMATCH` | 30.06percentage points inModerate mAP and31.15points inmAP9; khôngrelativepercent. |
| C49 | [sections/results.tex:86](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:86)<br>`(\textbf{+6.20\% mAP})` | tab:components47.60−41.40=6.20pp;throughput169.97/90.57≈1.8767 nên nearlydouble hợp lý nhưngGPUschưakhóa. | L481–482,511:+6.20. | `METRIC_MISMATCH` | a6.20-percentage-point increase in the reported progression; throughputprovenance phảiđồngnhất. |
| C50 | [sections/results.tex:91](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:91)<br>`($+5.72\%$ over TriBand-BEV and $+2.54\%$ over Binary35)` | tab:encoding50.14−44.42=5.72pp;50.14−47.60=2.54pp. | L505–508,512:khớpvalues. | `METRIC_MISMATCH` | 5.72 and2.54percentage points;góiencoding thay nhiều features,nênkhôngquyriêngoffsetgain. |
| C51 | [sections/results.tex:97](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:97)<br>`(\textbf{+8.92\% mAP})` | tab:componentsA5=60.22,A4=51.30:8.92pp;A3→A5=10.08ppcombinetwochanges. | L485–486,513:có50.14→60.22=10.08. | `METRIC_MISMATCH` | 8.92percentage points forH16→H32 row;10.08points appliesgrouping+widthcombined. |
| C52 | [sections/results.tex:102](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:102)<br>`(\textbf{+7.34\% mAP})` | tab:components68.45−61.11=7.34pp in mAP9,khôngModerate. | L64,486–487,514:7.34mAP9. | `METRIC_MISMATCH` | a7.34-percentage-point change inmAP9; labelmetricexplicitly. |
| C53 | [sections/results.tex:106](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:106)<br>`(\textbf{+1.87\% mAP})` | tab:componentsA6→A7:68.12−66.25=1.87pp(widthonly);A7→A8:−.74;A6→A8:+1.13. | L515 gộpaddinggatecũngthành68.12/+1.87;saiattribution. | `METRIC_MISMATCH` | 1.87points for fusion-width increase alone. SG-FPN row is.74points lower;updatecontext L515. |
| C54 | [sections/results.tex:125](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:125)<br>`(\textbf{+1.59\% mAP})`<br>Cùng nhóm: Results L125 | tab:loss68.97−67.38=1.59pp;Car.17,Ped2.45,Cyc2.14pp;IQAalone70.31−67.38=2.93pp. | L489–493:compositevaluesavailable;classdeltanottabulatedcontext. | `METRIC_MISMATCH` | Use1.59/2.93percentage points;fullfactorialinteraction3.20−1.59−2.93=−1.32points. Khônggọipositive synergisticgain. |
| C55 | [sections/results.tex:117](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:117)<br>`Block-SimAM gained 1.66--2.10 mAP points over stage-level variants` | tab:attention71.46−69.80=1.66,−69.44=2.02,−69.36=2.10pp;rangeđúng. | Contextstagevaluescùngnhưngfinal71.50→range1.70–2.14. | `METRIC_MISMATCH` | Giữ1.66–2.10points chofinal71.46;resolvecontext71.50. FPS144.05 thuộcstage3,khôngmọistage. |
| C56 | [sections/results.tex:139](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:139)<br>`The plotted losses weighted by T-SBUW show no divergence or oscillation.`<br>Cùng nhóm: Results L145–146 | fig:loss_curves actualPDF:validation/classificationA6pink spike≈epoch18→~100,otherfluctuations;chưadataexport. | Khôngloss-timeseries/rawspikestats ởcontext. | `OVERCLAIM` | Losses generally declined; visible validation/classification spikes occurred in some runs. Khôngclaimnooscillation. |
| C57 | [sections/results.tex:139](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:139)<br>`Configuration A10 has the lowest late-training validation losses across the continuous regression tasks.` | fig:loss_curves A9blue belowA10red ởGeometry/IoU vànhiềuđiểmOffset/Yaw;lossnormalizationkhácobjective. | Khôngper-tasklate-epochnumbers haycommon-normalizationcomparisons. | `OVERCLAIM` | The plotted regression losses decrease over training; no uniform lowest-loss claim forA10. Export comparable unweighted taskmetrics trước ranking. |
| C58 | [sections/results.tex:145](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:145)<br>`This behavior is consistent with regularization from physics-consistent point cloud augmentation (PC-GTA, A6) without apparent underfitting.` | fig:loss_curves givestrajectories;khôngmatchedregularization/underfittingdiagnostic;A6cóspike. | PC-GTA APpackageeffectkhôngđooverfitmechanism. | `UNSUPPORTED` | State observed training/validation trends; mechanism and degree of underfitting remain undetermined. Hedgeskhôngtạoevidence. |
| C59 | [sections/results.tex:24](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:24)<br>`compared to $56.01\%$ for PointPillars and $56.56\%$ for SECOND.` | PRasset legendsPedModeratePP55.83,SECOND56.47;tablemain56.01/56.56. Assettitle/footer:sharedlocalevaluator. | §6.1L411–412:56.01/56.56reference. Đây không chỉrounding. | `METRIC_MISMATCH` | Caption phảiphânbiệtreference-tablevaluesvslocalPRfigure. Regenerateallfigure/tablefromoneprotocol;khôngđổilegendthủcôngmàgiữcurvekhác. |
| C60 | [sections/results.tex:24](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:24)<br>`MobilePIXORNeXt consistently maintains superior recall and precision envelopes on vulnerable road users,` | tab:comparisonCyclist66.73<SECOND71.16;PRassetCyclistModerateours66.73<SECOND70.70;curvescross. | L413vsL412Cyclist66.73<71.16;farPed.80<.99. | `OVERCLAIM` | Higher Pedestrian AP is observed; Cyclist performance remains below SECOND at Moderate difficulty. KhôngclaimpointwisePRdominance/allVRUs. |
| C61 | [sections/results.tex:24](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:24)<br>`Car (top row,`<br>Cùng nhóm: Results L24 | PRactualcolumnsCar/Ped/Cyc,rowsEasy/Moderate/Hard;oursgold,PPgreen,SECONDblue,MobilePIXORred. | ContextkhôngPRlayout/color mapping. | `METRIC_MISMATCH` | Sửacaption theoactualrows/columns/colors;đồngthờisửareferenceAPs. Lỗi visualization/legendcontract,cùngroot vớiPRnumericmismatch. |
| C62 | [sections/results.tex:45](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:45)<br>`over $2.3\times$ faster than PIXOR` | tab:efficiency144.00/60.96≈2.3622;PIXOR/oursđềuT4. | L442,446:144.00,60.96. | `VERIFIED` | Giữ about2.36×pipeline throughput undermeasuredconfiguration;khôngmodel-onlyspeedup. |
| C63 | [sections/results.tex:51](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:51)<br>`reducing payload by $60.0\%$.`<br>Cùng nhóm: Related Work L7; tab:encoding note | tab:encoding35→14float32channels:(35−14)/35=.6;78.848MB→31.5392MB. | L507–508:78.85/31.54MB,512:60%. | `VERIFIED` | Giữ tensor-payload reduction60%;khôngđồngnhấtH2Dlatencyreduction60%. |
| C64 | [sections/results.tex:51](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:51)<br>`With pinned-memory buffers, H2D latency fell to $6.91\,\text{ms}$, compared with $40.63\,\text{ms}$ for MobilePIXOR.` | Tab:efficiency/rawbreakdownoursT4,MobilePIXORL4;channels/host/timerdiffer,khôngpinning-onlycontrol. | L443vs446:L4vsT4,40.63vs6.91. | `OVERCLAIM` | Recorded H2D times were6.91ms and40.63ms on the respective pipelines; no isolated attribution to pinning orchannelcount. |
| C65 | [sections/results.tex:70](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:70)<br>`(a $+34.78\%$ improvement)`<br>Cùng nhóm: Results L70 | tab:distance65.23−30.45=34.78pp;Cyclist76.32−25.55=50.77pp;ratios2.142/2.987. | L456,459:cùngvalues;L461doublenearlytripleđúng. | `METRIC_MISMATCH` | 34.78 and50.77percentage points; morethandouble/nearlytriple đúngAPratio,khôngsafetydemonstration. |
| C66 | [sections/results.tex:73](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:73)<br>`(vs. $12.61\%$, a $+31.69\%$ gain)`<br>Cùng nhóm: Results L73 | tab:distance44.30−12.61=31.69pp;11.49−3.75=7.74pp;farCyclist14.38pp. | §6.4L456,459:cùngvalues. | `METRIC_MISMATCH` | 31.69percentage points; explicitlyretainfarPedestrian.80vs.99,−.19points. |
| C67 | [sections/introduction.tex:9](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:9)<br>`Physics-consistent synthesis regularizes distant returns,`<br>Cùng nhóm: Results L70,73 | tab:distancefinalvsMobilePIXOR đổi nhiều modules;farPedours.80<.99;khôngrange-specificPC-GTAoff/on. | L461claimnear/mid tốt;L514 gáncơchếchưađượcisolate. | `UNSUPPORTED` | Range-adaptive subsampling is intended to model distance-dependent density; its isolated range-wise benefit was not established. |
| C68 | [sections/results.tex:192](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:192)<br>`Physics-consistent augmentation (\texttt{PC-GTA}) and decoupled grouped heads produced larger observed accuracy gains than model depth scaling alone.` | tab:components cópackage/widthchanges;khôngdepth-onlycontrol row. | §6.6 khôngdepth-onlyrun. | `UNSUPPORTED` | PC-GTA and grouped-head changes had large reported progression differences; remove comparison with untested depth-only scaling. |
| C69 | [sections/results.tex:149](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:149)<br>`from the extended run (Epoch 50)` | tab:ap_vs_lossL160/captionL153:130;setupfinalselected130;canonicalcheckpoint130. | L470:Epoch130;L378130extendedtraininglàcontextstaleschedule. | `METRIC_MISMATCH` | from the extended160-epoch run, selected at epoch130.50 chỉstandard schedule/illustratedcurves. |
| C70 | [sections/results.tex:153](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:153)<br>`validation AP maximization (Epoch 130)` | tab:ap_vs_lossfinalAP71.46<best-loss71.48;selectiontrainingpolicyrecordedmaximumvalAP,khôngallcheckpointselectionmetrics. | L470–471cũng71.46vs71.48. | `UNSUPPORTED` | recorded AP-selected checkpoint(epoch130);clarifytraining-selectionmetricvsfinalbenchmarkandresolvewhyfinalrescorebestlossishigher. |
| C71 | [sections/results.tex:166](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:166)<br>`had $20.27\,\text{ms}$ lower end-to-end latency` | tab:ap_vs_loss81.23−60.96=20.27ms;ours_best_loss/AP logsconfirm. | L470–471:81.23vs60.96. | `VERIFIED` | Giữobservedmeanlatencydifference;khônggánnguyênnhâncheckpointnếukhôngcontroltiming/candidatecounts. |
| C72 | [sections/results.tex:166](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:166)<br>`A possible explanation is that extended training sharpened classification heatmaps around positive centers,` | Khôngpeakcount,NMS-candidatehistograms,heatmapsharpness,h2d/model/decodecontrolledanalysis. | §6.5onlymetrics;khôngmechanismdiagnostics. | `UNSUPPORTED` | The mechanism behind the timing difference remains undetermined; testcandidatecounts andstage-levelrepeatedtimings. Giữhypothesisrõnếuđềxuấtfuturetest. |
| C73 | [sections/experiments.tex:52](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:52)<br>`Peak allocated GPU memory was $98.44\,\text{MB}$.` | rawfield98.439453125;source divides1024²→MiB,notdecimalMB;~103.22MBdecimal. | L389 gọi98.44MB,cùngunitlabelissue. | `METRIC_MISMATCH` | Peak allocated evaluation GPU memory was98.44MiB;khôngtrainingpeak/modeldiskbytes. |
| C74 | [sections/experiments.tex:46](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:46)<br>`The reported checkpoint came from an extended 160-epoch run with 10 warmup epochs and was selected at epoch 130.` | tab:setup đúng160/10/130;resolvedfinalconfig/checkpointidentityconfirms. | L377–378chỉstandard50/warmup4 hoặcextended130. | `METRIC_MISMATCH` | Giữpaper160/10/130;refreshcontext phânbiệtschedule length vàselectedepoch. |
| C75 | [sections/experiments.tex:52](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:52)<br>`Variance and standard deviation` | ExperimentsL52nóileftforfuture,ResultsL5cùngsingle-seed42;khôngstatisticalCItest. | Configseed42;contextkhôngmulti-seedstats. | `VERIFIED` | Giữsingle-seedlimitation;khôngtựthêmstatistically significant,robustacrossseeds ởrewrites. |
| C76 | [sections/methodology.tex:449](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:449)<br>`is fully recovered without trigonometric ambiguity.` | Eq.box_decodehalfatan2 choheadingcanonical[−π/2,π/2];notfull2πorientation;GTusesπsymmetry. | L119,139–140:cùngπsymmetriccontract. | `VERIFIED` | The canonical heading is recovered moduloπ; khôngclaimfront/backdisambiguation. |
| C77 | [sections/related_work.tex:6](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:6)<br>`($\mu_z, \sigma_z, z_{\min}, z_{\max}$)` | tab:strata_bev14actualmax,mean,span,std;khôngstandalonez_minchannel. | L250–254:max,mean,span,std. | `METRIC_MISMATCH` | ĐổiRelatedWorkthànhz_max,z_mean,z_span,z_std. Minimum cóthểsuy từmax−span,nhưngkhôngphảichannelriêng. |
| C78 | [sections/introduction.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:7)<br>`memory access latencies ($100\text{--}195\,\text{ms}$)`<br>Cùng nhóm: RelatedWork L11; Results L33 | tab:efficiency làE2E,khôngmemoryaccess;SECOND93.70ms<100;two-stage133.75–194.50. | §6.1SECOND93.70;§6.3stagebreakdownnotmemory-only. | `METRIC_MISMATCH` | Name E2E latencies and the exact detector subset; do not relabel whole-pipeline timing as memory access or infer embedded limits from T4. |
| C79 | [sections/methodology.tex:78](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:78)<br>`Calibrated surface returns capture material discrimination` | tab:strata_bev14ch8/9max/meanintensity;khôngmaterial-labelclassification/calibrationstudy. | L255–256onlyreflectanceformulas. | `UNSUPPORTED` | Reflectance statistics provide intensity cues; remove untested examples ofmaterialdiscrimination. |
| C80 | [sections/methodology.tex:140](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:140)<br>`anchored strictly flush with the terrain:` | Eq.ground_fitting setsbottomtoestimatedpercentileheight;khôngtruegrounderror/planeGTmetric. | L286–288sameheuristic. | `OVERCLAIM` | anchored to the estimated local ground elevation; do notpromiseexactphysicalterrainalignment. |
| C81 | [sections/methodology.tex:401](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:401)<br>`The twelve logit parameters are utilized strictly during training and discarded at inference.` | tab:setup/Methods12criterionparams;detectorforwardusesneithercriterionnorEMA. | L236,330:12trainingonly. | `VERIFIED` | Giữ12training-onlylogits,notpartof753569detectorweights. |
| C82 | [sections/experiments.tex:49](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:49)<br>`and visibility protection.` | tab:setupPCGTApackage;hybrid_sampler protect=flag&&(shadow&#124;&#124;extra_width>0&#124;&#124;margin<=1e-8). Finalflagtrue,shadowfalse,width0,margin.1→false. | Contextmô tảmechanisms,khôngruntimeguard distinction. | `METRIC_MISMATCH` | The visibility-protection flag was enabled, but the explicit check was bypassed under the final sampler guard; do notclaimitactivelytestedallplacements. |
| C83 | [sections/results.tex:91](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:91)<br>`($188.4\,\text{FPS}$)` | tab:encoding188.43→188.4(1dp),169.01→169.0;GPUL4caption. | L501–508:L4,188.43/169.01;componentsreuse169.97/165.02/169.01asT4. | `METRIC_MISMATCH` | Roundingvaluesđúng;reconcileL4vsT4forreusedrows. Do notcallforwardFPS E2E FPS. |
| C84 | [sections/results.tex:45](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:45)<br>`Percentile analysis showed low latency dispersion in the evaluated desktop GPU environment:` | tab:efficiencyp50=58.28,p95=80.85,p99=95.91;noquantitativecriterionforlowdispersion. | L446cùngpercentiles. | `OVERCLAIM` | Reportpercentilesdirectly;mean16.41FPSsupports10Hzaveragebutnotworst-caseharddeadline. |
| C85 | [sections/conclusion.tex:9](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:9)<br>`The primary detector estimates oriented 2D Bird's-Eye View bounding boxes.` | tab:comparisonBEV,Eq.target_tuple7values,fig:qualitativecaptionsprojectedfootprints. | L431–433:BEVonly. | `VERIFIED` | Giữlimitationnày vàđưaBEVscopeđếnAbstract/Intro;khôngxóachỉđểclaimmạnhhơn. |
| C86 | [sections/conclusion.tex:9](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:9)<br>`requires evaluation on dedicated off-road benchmarks.`<br>Cùng nhóm: Conclusion L9; Results L195 | Khôngoff-road/adverseweather/16beam/TensorRTINT8resulttrongtables;futuretenseproper. | ContextchỉKITTI64beam/fairweather;optional3Dheadnotbenchmarked. | `VERIFIED` | Giữaslimitation/futurework,khôngcoiđây làorphanvalidatedcontribution. Đồngthờikhôngclaimembeddedproven. |
| C87 | [sections/results.tex:30](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:30)<br>`The second is GroupedHeader, which separates pedestrian regression from vehicle footprints to reduce gradient interference between categories.`<br>Cùng nhóm: Results L97; Methods L307 | tab:componentsgrouping/widthgivesAProws;khônggradient-cosine/conflictstatistics;đồngthờiđổi nhiềufeatures trongcross-familycomparison. | L77,225,513 nóieliminatesconflict nhưngkhôngđogradients. | `UNSUPPORTED` | GroupedHeader separates task parameters. Its isolated effect on gradient interference was not measured; keep as motivation, not explanation established by AP. |
| C88 | [sections/methodology.tex:261](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:261)<br>`Its 3D spatial-and-channel weights modulate sparse LiDAR clusters relative to flat terrain backgrounds.` | Eq.simam_energy maskphụthuộcactivationdeviation;khôngsemanticLiDARcluster/terrainattentiongroundtruth. | L71 tươngtựamplifyingclusters;khôngclass-specificmaskmetrics. | `UNSUPPORTED` | The mask modulates feature activations according to their deviation from channel moments; tests would be needed to establish preferential amplification of object clusters. |
| C89 | [sections/methodology.tex:261](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:261)<br>`stage-level adapters add learned linear projections.` | Code ResidualLightAttention có một scalar beta vớiX+beta*(SimAM(X)−X),khônglearnedchannelprojection;tableattentionkhôngparamcounttừngadapter. | Context khôngexactadapterdelta;rawrevision củahistoricalruns chưađượcchỉrõ. | `METRIC_MISMATCH` | Describe the current adapter as a learned scalar residual interpolation, or cite the historical implementation actually benchmarked. |
| C90 | [sections/related_work.tex:13](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:13)<br>`inverted bottleneck expansions ($e=2.5$)` | RelatedWork gánchoConvNeXt citedpaper;Methods codefinalMobilePIXORNeXt-Sexpansion2.5 đúng nhưngkhôngxácminhoriginalConvNeXt. | L66,220 chỉmô tảprojectownblock2.5. | `UNSUPPORTED` | Attribute large-kernel design inspiration to ConvNeXt and state separately that the present block uses expansion2.5; verify original citation before assigning that ratio to it. |
| C91 | [sections/results.tex:33](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:33)<br>`by less than $1.7\%$` | tab:comparisonCar:86.90−88.02=−1.12pp,86.90−88.56=−1.66pp;vsPIXOR+14.88pp,MobilePIXOR+11.38pp. CyclistvsPP+.39pp,vsSECOND−4.43pp. | L409–413 cùngAPvalues. | `METRIC_MISMATCH` | less than1.7percentage points forCar;retainCyclist deficit4.43points relative toSECOND and evaluator boundary. |

## 4. Sổ phép tính và sự khác nhau giữa các protocol

### 4.1 AP deltas và ablation deltas

`[Script]` dùng `Decimal` cho44 phép trừ từ số đang hiển thị. Cột “khớp” chỉ xác nhận số học, không chứng nhận các thí nghiệm có cùng config, có statistical significance hoặc cơ chế nhân quả được kiểm soát. Gains đang viết bằng `%` nên đổi sang `percentage points`, giữ `%` cho AP level và relative parameter/payload reduction.

| Phép đối soát | Phép trừ đúng theo số đang hiển thị | Đơn vị | Kết luận arithmetic |
|---|---|---|---|
| ours-PointPillars Moderate | 71.46 − (70.13) = 1.33 | pp | Khớp exact Decimal |
| ours-SECOND Moderate | 71.46 − (72.09) = -0.63 | pp | Khớp exact Decimal |
| ours-MobilePIXOR Moderate | 71.46 − (41.40) = 30.06 | pp | Khớp exact Decimal |
| ours-MobilePIXOR mAP9 | 73.42 − (42.27) = 31.15 | pp | Khớp exact Decimal |
| Pedestrian vs PointPillars | 60.75 − (56.01) = 4.74 | pp | Khớp exact Decimal |
| Pedestrian vs SECOND | 60.75 − (56.56) = 4.19 | pp | Khớp exact Decimal |
| Pedestrian vs MobilePIXOR | 60.75 − (28.16) = 32.59 | pp | Khớp exact Decimal |
| Car vs PIXOR | 86.90 − (72.02) = 14.88 | pp | Khớp exact Decimal |
| Car vs MobilePIXOR | 86.90 − (75.52) = 11.38 | pp | Khớp exact Decimal |
| backbone A0-A1 | 47.60 − (41.40) = 6.20 | pp | Khớp exact Decimal |
| encoding vs TriBand | 50.14 − (44.42) = 5.72 | pp | Khớp exact Decimal |
| encoding vs Binary35 | 50.14 − (47.60) = 2.54 | pp | Khớp exact Decimal |
| group head width | 60.22 − (51.30) = 8.92 | pp | Khớp exact Decimal |
| PC-GTA Moderate | 66.25 − (60.22) = 6.03 | pp | Khớp exact Decimal |
| PC-GTA mAP9 | 68.45 − (61.11) = 7.34 | pp | Khớp exact Decimal |
| fusion width | 68.12 − (66.25) = 1.87 | pp | Khớp exact Decimal |
| SG-FPN isolated progression | 67.38 − (68.12) = -0.74 | pp | Khớp exact Decimal |
| width+neck combined | 67.38 − (66.25) = 1.13 | pp | Khớp exact Decimal |
| OGA+IQA | 70.58 − (67.38) = 3.20 | pp | Khớp exact Decimal |
| SimAM Moderate final | 71.46 − (70.58) = 0.88 | pp | Khớp exact Decimal |
| SimAM mAP9 final | 73.42 − (72.66) = 0.76 | pp | Khớp exact Decimal |
| SimAM vs stage2 | 71.46 − (69.44) = 2.02 | pp | Khớp exact Decimal |
| SimAM vs stage3 | 71.46 − (69.80) = 1.66 | pp | Khớp exact Decimal |
| SimAM vs stage4 | 71.46 − (69.36) = 2.10 | pp | Khớp exact Decimal |
| OGA alone | 68.97 − (67.38) = 1.59 | pp | Khớp exact Decimal |
| IQA alone | 70.31 − (67.38) = 2.93 | pp | Khớp exact Decimal |
| OGA added with IQA | 70.58 − (70.31) = 0.27 | pp | Khớp exact Decimal |
| IQA added with OGA | 70.58 − (68.97) = 1.61 | pp | Khớp exact Decimal |
| factorial interaction | 3.20 − (4.52) = -1.32 | pp | Khớp exact Decimal |
| Q-OGA vs OGA | 69.20 − (68.97) = 0.23 | pp | Khớp exact Decimal |
| Q-OGA vs baseline | 69.20 − (67.38) = 1.82 | pp | Khớp exact Decimal |
| quality head time | 8.19 − (6.94) = 1.25 | ms | Khớp exact Decimal |
| attention time | 8.67 − (8.19) = 0.48 | ms | Khớp exact Decimal |
| checkpoint E2E | 81.23 − (60.96) = 20.27 | ms | Khớp exact Decimal |
| near Pedestrian | 65.23 − (30.45) = 34.78 | pp | Khớp exact Decimal |
| near Cyclist | 76.32 − (25.55) = 50.77 | pp | Khớp exact Decimal |
| mid Pedestrian | 11.49 − (3.75) = 7.74 | pp | Khớp exact Decimal |
| mid Cyclist | 44.30 − (12.61) = 31.69 | pp | Khớp exact Decimal |
| far Pedestrian | 0.80 − (0.99) = -0.19 | pp | Khớp exact Decimal |
| far Cyclist | 14.94 − (0.56) = 14.38 | pp | Khớp exact Decimal |
| context R37 Moderate conflict | 71.50 − (71.46) = 0.04 | pp | Khớp exact Decimal |
| context R37 mAP9 conflict | 73.45 − (73.42) = 0.03 | pp | Khớp exact Decimal |
| context R37 attention delta | 71.50 − (70.58) = 0.92 | pp | Khớp exact Decimal |
| context R37 mAP9 attention delta | 73.45 − (72.66) = 0.79 | pp | Khớp exact Decimal |

### 4.2 Bảng chính và PR asset đang dùng hai bộ số baseline

| Nguồn | PointPillars Moderate (Car/Ped/Cyc) | SECOND Moderate (Car/Ped/Cyc) | Ours Moderate | Ý nghĩa |
|---|---|---|---|---|
| `tab:comparison`,context§6.1 |88.02/56.01/66.34|88.56/56.56/71.16|86.90/60.75/66.73|Reference baseline AP,local oursAP;reported macro70.13/72.09/71.46.|
| Legends `pr_all_difficulties.pdf` |88.60/55.83/66.38|89.36/56.47/70.70|86.90/60.75/66.73|Footer ghi shared local BEV evaluator;means từ displayedAPs≈70.27/72.18/71.46.|

Các mean70.27/72.18 là **reviewer arithmetic từ legend hiển thị**, không phải số benchmark mới hoặc đề nghị tự thay bảng chính. Cần raw prediction/evaluator provenance và precision gốc trước khi công bố lại. Nếu theo legends, Pedestrian margins là4.92/4.28pp, không phải4.74/4.19; Moderate mAP margins từ displayed panel means khoảng+1.19/−.72pp. Giữ riêng hai bộ số cho tới khi chọn protocol thống nhất.

Không flag tổng của ba AP đã làm tròn lệch.01 là lỗi tự động. Ví dụ(88.02+56.01+66.34)/3=70.1233…, nhưng mean của ba raw scores có thể round thành70.13 khi từng raw score đã bị round riêng ở bảng. Phải dùng raw metrics để xử lý rounding, không tự sửa70.13 thành70.12.

### 4.3 Pareto, latency và nhánh tính toán

`[Script]` từ bảng đang báo cáo, parameter-vs-accuracy non-dominated set gồm MobilePIXOR,ours,PointRCNN,PV-RCNN,Part-A2; PointRCNN(4.039M,73.38) dominates SECOND(5.326M,72.09) theo hai trục đó. Đường “Pareto Frontier” trong asset lại đi qua SECOND và bỏ PointRCNN. Với latency-vs-accuracy, observed non-dominated set gồm ours,SECOND,Part-A2. Vẽ lại frontier từ đúng tập points và nêu rõ mixed protocols/hardware; không suy global Pareto optimality.

Pipeline forward17.88ms/55.94FPS và standalone8.67ms/115.32FPS là hai regimes. Thời gian20.94ms gồm load+raster CPU. MobilePIXOR timing làL4; encoding ablation caption làL4 nhưng components/attention caption gọi timingT4. Cần raw timing manifest cho từng row. Checkpoint epoch50 trong ResultsL149 cần sửa thành130; schedule160 và50-epoch curves không nên được gộp làm một điều kiện huấn luyện.

### 4.4 Từ tuyệt đối hóa: kết quả scan có phân xử

Không tìm thấy các câu mẫu “dramatically outperforms all existing methods”, “completely solves the occlusion problem”, hay literal “state-of-the-art” trong section sources hiện tại. Không gán những câu này cho tác giả. Các từ/cụm thực sự cần sửa là “matched SECOND”, “leading edge Pareto efficiency”, “consistently ... superior ... on vulnerable road users”, “no divergence or oscillation”, “lowest ... across ... tasks”, và các guarantee về numerical stability/strict terrain alignment.

Các diễn đạt “zero additional attention parameters”, “fourteen deterministic features”, tổngweights6, và “lowest recorded mean pipeline latency among evaluated detectors” có trần bằng chứng phù hợp nếu giữ đúng scope. Không làm yếu chúng chỉ vì chứa zero/all/lowest. Không có multi-seed variance hoặc p-value trong bản hiện tại để thêm “statistically significant” hay “robust across seeds”.

## 5. Kiểm chuỗi đóng góp bốn mắt xích

Mức `CRITICAL BLOCKER` bên dưới áp dụng cho gate liên kết đóng góp do người dùng quy định, không phải một kết luận fabrication hay quyết định biên tập của AJSE. Ba bullets chính ở Introduction đều có nội dung tương ứng trong Method/Results/Conclusion ở mức tổng thể; review xuống cấp module để tránh che missing link của T-SBUW trong một bullet rất rộng.

| Đóng góp | Introduction | Methodology | Ablation / evidence row | Conclusion | Kết luận chuỗi |
|---|---|---|---|---|---|
| P0 — Hệ thống compact BEV | Intro L15 | Method L12–466 | comparison + efficiency + components A0/A10 | Conclusion L5,7 | ĐỦ; accuracy/latency scope cần sửa C09–C18. |
| P1 — Strata-BEV14 | Intro L15 | Method §Strata-BEV14 L42–122 | encoding:TriBand/Binary35/rich8/hist14;components A1–A3 | Conclusion L5 | ĐỦ ở mức encoding package; không isolate riêng centroid/moments. |
| P2 — MobilePIXORNeXt-S backbone | Intro L15 | Method §Backbone L176–229 | components A0→A1 | Conclusion L5 | ĐỦ; các manifests cần xác minh cho causal effect. |
| P3 — Block-SimAM placement | Intro L15,19 | Method L231–261 | attention rows;components A9→A10 | Conclusion L5 | ĐỦ mắt xích; matched topology/budget chưa xác nhận, C32–C33. |
| P4 — SG-FPN | Intro L15 | Method L264–291 | components A7→A8:−0.74pp | Conclusion L5 | ĐỦ; negative ablation vẫn là bằng chứng. Không suy gain hoặc artifact-free. |
| P5 — GroupedHeader | Intro L15 | Method L293–325 | components A3→A4→A5 | Conclusion L5 | ĐỦ; grouping và width là hai thay đổi, chưa đo gradient interference. |
| P6 — PC-GTA active package | Intro L17 | Method L124–174 | components A5→A6:+6.03pp | Conclusion L5,7 | ĐỦ package-level; optional disabled branches không có gain riêng. |
| P7 — Composite OGA | Intro L17 | Method L327–360 | loss:baseline/OGA × IQAoff/on | Conclusion L5,7 | ĐỦ composite objective; projection/corner chưa được isolate riêng. |
| P8 — IQA | Intro L17 | Method L362–368 | loss factorial explicit quality head | Conclusion L5,7 | ĐỦ; chưa có ranking/calibration metric riêng ngoài AP. |
| P9 — T-SBUW như đóng góp chính | Intro L17 | Method L383–401 | THIẾU weighting-off/uniform/EMA-only/full comparison | Conclusion L5 | CRITICAL BLOCKER — ORPHAN_CONTRIBUTION: thiếu mắt xích ablation. |
| P10 — Q-OGA được tuyên bố “We developed” | THIẾU trong contribution list | Method L370–381 | loss:Q-OGA row69.20%;Results L127 | THIẾU Q-OGA trong Conclusion | CRITICAL BLOCKER — ORPHAN_CONTRIBUTION: thiếu Intro và Conclusion nếu giữ là đóng góp mới. |
| P11 — Empirical accuracy/size/latency trade-off | Intro L19 | Method infer contract + Experiments metrics | comparison,efficiency,components,attention,loss | Conclusion L7 | ĐỦ; sửa protocol/rounding/edge-extrapolation trước khi dùng superiority wording. |

Mỗi hàng trên là một đơn vị đóng góp có ý nghĩa ở cấp module hoặc objective, không phải mỗi channel và mỗi toán tử con. Conclusion có thể tóm tắt một module bằng tên module; không bắt buộc lặp mọi công thức. Vì vậy không gán orphan cho từng feature F0–F13, từng primitive của composite OGA, hoặc từng nhánh sampler đã công khai là tùy chọn. Việc thiếu isolated ablation của những primitive đó vẫn giới hạn các suy luận về cơ chế (C34, C38, C45, C67, C87–C88).

**P9 / C35:** T-SBUW được đặt trong contribution bullet, có Methods và được nhắc lại ở Conclusion, nhưng không có hàng ablation kiểm tra riêng. Loss curves không thay thế can thiệp tắt/bật weighting. Đóng lỗi bằng matched runs uniform / EMA-only / full T-SBUW, hoặc chuyển weighting sang implementation detail và bỏ tuyên bố độc lập về gain/stability. Không được viết thêm một giá trị AP chưa chạy.

**P10 / C39:** Q-OGA có lời tuyên bố phát triển phương pháp, subsection và bảng kết quả nhưng thiếu ở Intro contribution list và Conclusion. Có thể thêm một câu ở mỗi nơi, với AP69.20% và phạm vi head-free alternative, hoặc xác định rõ đây là auxiliary comparison. Mẫu bên dưới chọn cách thêm cả hai mắt xích; chưa có chỉnh sửa nào được áp dụng vào manuscript.

Optional LOS/shadow/radiometry được trình bày là capability; final run tắt chúng. C34 yêu cầu thu hẹp câu “address additional synthesis artifacts” thành một mô tả tính năng chưa được đánh giá độc lập. Nếu tác giả tiếp tục trình bày từng nhánh này như **đóng góp mới đã được kiểm chứng**, chúng cũng phải có mắt xích ablation; chúng chưa đạt điều kiện đó trong bản hiện tại. Future-work INT8/RC-SGFPN/off-road là mục tiêu tương lai được ghi rõ, không phải kết quả đã tuyên bố đạt được.

## 6. Các đoạn văn sửa mẫu, nằm trong trần bằng chứng

Các đoạn dưới đây bằng tiếng Anh để dùng trong bản LaTeX. Chúng giữ canonical final benchmark71.46/73.42, thừa nhận evaluator khác nhau, dùng điểm phần trăm cho AP differences và giữ các kết quả bất lợi. Các mô tả “reported” không tự đóng lỗi provenance hoặc thay cho thí nghiệm còn thiếu. Không có số đo mới hoặc p-value được tạo ra.

### 6.1 Abstract thay thế

```latex
MobilePIXORNeXt is a compact, proposal-free bird's-eye-view (BEV) object
detector operating on three-dimensional LiDAR point clouds. It contains
753,569 detector parameters and uses Strata-BEV14, a deterministic,
parameter-free NumPy/Numba CPU rasterizer. Fourteen channels encode height
distributions, elevation statistics, reflectance and subcell centroid
displacements. A ConvNeXt-style backbone with Block-SimAM, a scale-gated
feature pyramid and grouped prediction towers process this representation.
Training combines physics-consistent ground-truth augmentation, oriented
geometric alignment and localization-quality supervision, with twelve
training-only task-weighting logits. On KITTI validation, the local BEV
evaluator reports 71.46\% Moderate mAP, 73.42\% nine-cell mAP and 60.75\%
Pedestrian Moderate AP. Reference-evaluator scores for PointPillars and
SECOND are 70.13\% and 72.09\% Moderate mAP; these cross-protocol comparisons
are descriptive. The detector uses 84.41\% and 85.85\% fewer parameters than
these two configurations, respectively. At batch size one in FP32 on Tesla
T4, mean input-to-detection latency is 60.96\,ms (16.41\,FPS), including
20.94\,ms for point-cloud loading and CPU rasterization. This is the lowest
recorded mean pipeline latency in the evaluated T4 cohort. These results
motivate further evaluation of compact BEV detection on embedded hardware.
```

Hai số reduction được tính **trực tiếp từ raw parameter counts**, rồi làm tròn một lần ở hai chữ số. Không đưa “matched SECOND”, “leading edge” hoặc kết quả embedded đã triển khai vào abstract này.

### 6.2 Introduction — thay phần contribution list

```latex
The principal contributions are:
\begin{itemize}
\item A compact BEV detection system combining the deterministic
Strata-BEV14 representation, a single-branch ConvNeXt-style backbone,
Block-SimAM, a scale-gated feature pyramid and grouped prediction towers.
The final detector contains 753,569 parameters.
\item A training design combining ground-aligned, obstacle-aware,
range-adaptive GT sampling with a composite oriented geometry objective
and an explicit quality head. We also examine Q-OGA as a head-free
alternative, reporting 69.20\% Moderate mAP in the loss comparison.
Adaptive task weighting is documented as an implementation component;
its independent effect has not been isolated experimentally.
\item An evaluation of accuracy, parameter count and sequential pipeline
latency on KITTI validation. The final local-evaluator results are
71.46\% Moderate BEV mAP and 73.42\% nine-cell mAP. Mean pipeline latency
is 60.96\,ms on Tesla T4 at batch size one in FP32. The reported progression
changes by 6.03 percentage points for PC-GTA, 3.20 points for OGA with IQA,
and 0.88 points from A9 to A10. An isolated Block-SimAM effect requires
confirmation of matched topology, training budget and checkpoint selection.
\end{itemize}
```

Mẫu này thu hẹp T-SBUW thành implementation component và thêm Q-OGA vào Intro. Nếu muốn giữ T-SBUW là đóng góp độc lập, cần bổ sung ablation thật rồi thay câu giới hạn tương ứng; sửa câu văn đơn thuần không tạo ra bằng chứng.

### 6.3 Method — scope, SG-FPN, PC-GTA, quality và stability

```latex
The detector predicts oriented BEV footprints $(x,y,l,w,\psi)$ from
three-dimensional LiDAR input. The reported primary configuration does
not estimate vertical position or height.

The stride-8 top-down refinement uses a $3\times3$ depthwise convolution
with 48 input and output channels. The stride-4 refinement uses a standard
$3\times3$ convolution to reduce 48 channels to 32. Both refinements follow
bilinear upsampling and use batch normalization and SiLU. The design avoids
using transposed convolution; artifact suppression was not measured as a
separate outcome. The A7--A8 progression changes Moderate mAP from 68.12\%
to 67.38\%.

PC-GTA anchors inserted objects to an estimated local ground elevation,
checks static clearance and subsamples points according to the source and
target ranges. LOS rejection, shadow masking and radiometric adjustment
are optional and are disabled in the final run. The visibility-protection
flag is enabled, but the implementation bypasses its explicit check when
shadow masking is disabled, removal widths are zero and the collision
margin is positive, as in this configuration. The reported package-level
change is 6.03 percentage points; the effects of the optional branches are
not isolated.

The quality head is trained against a detached, clipped projection-GIoU
similarity proxy, rather than exact rotated area IoU. Detachment prevents
gradients from passing through the target-generation path. LayerScale is
initialized to $10^{-5}$ to keep initial residual updates small. These
design choices do not establish a separate stability or calibration gain
without the corresponding controlled measurements.

SimAM uses channel-moment-normalized feature deviations to modulate the
residual features. It adds no trainable attention parameters. Its geometric
or semantic selectivity should be evaluated separately from the AP of the
complete detector.
```

Các loss primitives, normalization, corner ordering và Q-OGA soft-min/RDA vẫn cần bổ sung theo T03–T05 của Phase2; bản sửa claim này không thay thế đặc tả objective. Exact IQA parameter saving chỉ được công bố sau khi resolve topology từng row. Final-topology code count là18,626 nhưng không đủ để ghi đè bảng lịch sử0.702/0.721M.

### 6.4 Results — đối chiếu số chính và factorial interpretation

```latex
Table~\ref{tab:comparison} reports 71.46\% Moderate mAP for MobilePIXORNeXt,
70.13\% for PointPillars and 72.09\% for SECOND. These values differ by
$+1.33$ and $-0.63$ percentage points, respectively, under the separately
identified local and reference evaluators. The Pedestrian differences
computed from this table are $+4.74$ and $+4.19$ percentage points. They
should not be combined with the different baseline values shown in the
shared-local-evaluator PR figure. MobilePIXORNeXt is lower than SECOND on
Cyclist Moderate AP (66.73\% versus 71.16\%).

The loss-factorial table reports changes of 1.59 percentage points for
OGA alone, 2.93 points for IQA alone, and 3.20 points for their combination
relative to the baseline. The resulting interaction contrast is
$3.20-1.59-2.93=-1.32$ points. The combined score is the highest in this
factorial table, while the table does not establish a positive interaction
or directly measure ranking accuracy or calibration error.

At 50--70.4\,m, Cyclist AP rises from 0.56\% to 14.94\% relative to
MobilePIXOR, while Pedestrian AP falls from 0.99\% to 0.80\%.
These comparisons describe the complete configurations and do not isolate
the effects of range-adaptive augmentation or attention.
```

### 6.5 PR caption thay thế, theo asset hiện có

```latex
\caption{Precision--recall curves on 3,769 common KITTI validation frames
under the shared local BEV evaluator. Columns show Car, Pedestrian and
Cyclist; rows show Easy, Moderate and Hard difficulty. The gold curves
show MobilePIXORNeXt, green PointPillars, blue SECOND, red MobilePIXOR,
and pink PIXOR where supported. AP values are given in the panel legends.
These local-evaluator baseline AP values differ from the reference-evaluator
values in Table~\ref{tab:comparison}. The figure does not show uniform
precision--recall dominance across all vulnerable-road-user categories.}
```

Đây là caption cho **asset đang tồn tại**, không phải đề nghị tự đổi AP legends bằng số bảng. Nếu chọn dùng một evaluator chung cho cả bảng và hình, phải xuất lại curves/tables/deltas từ đúng nguồn đó. Caption đúng cũng không tự xác minh prediction provenance hay tương đương NMS của baseline.

### 6.6 Loss-curve caption và checkpoint paragraph thay thế

```latex
\caption{Training and validation loss trajectories over 50 epochs for
configurations A1--A10. The upper panels show the total objectives;
the remaining panels show the plotted task losses. Losses generally
decline, with transient spikes in some validation and classification
curves. These trajectories describe the reported runs; they do not isolate
the contribution of T-SBUW or establish uniformly lowest task losses for
A10. The final reported checkpoint is selected from a separate extended
160-epoch schedule.}

The recorded AP-selected checkpoint is at epoch 130, and the
minimum-validation-loss checkpoint is at epoch 47. Their final Moderate
mAP values are 71.46\% and 71.48\%, respectively. Mean E2E latencies are
60.96\,ms and 81.23\,ms, a difference of 20.27\,ms. The relationship between
the training-time selection metric and the final rescored AP requires
clarification. The cause of the timing difference remains undetermined;
candidate-count statistics and repeated stage-level timings would help
test a heatmap-sharpening explanation.
```

### 6.7 Conclusion thay thế, gồm Q-OGA và giới hạn ứng dụng

```latex
We presented MobilePIXORNeXt, a compact BEV object detector operating on
three-dimensional LiDAR point clouds. Strata-BEV14 computes deterministic
height, reflectance and subcell centroid statistics without a learned
point encoder. A single-branch backbone with Block-SimAM, a scale-gated
feature pyramid and grouped towers processes this representation. Training
combines PC-GTA, composite oriented geometric supervision and an explicit
quality head. Task weighting uses twelve training-only logits; its
independent empirical benefit remains unmeasured.

On KITTI validation, the final configuration reports 71.46\% Moderate
BEV mAP and 73.42\% nine-cell mAP under the local evaluator, with 753,569
detector parameters. Its parameter count is 84.41\% lower than PointPillars
and 85.85\% lower than SECOND in the evaluated configurations. The reported
reference-evaluator AP values are 70.13\% and 72.09\%; the corresponding
cross-protocol differences are descriptive. The auxiliary Q-OGA comparison
reports 69.20\% Moderate mAP without explicit quality heads.
The recorded sequential pipeline averages 60.96\,ms (16.41\,FPS) on Tesla T4
at batch size one in FP32.

The reported results concern oriented BEV footprints rather than full 3D
boxes. Embedded-device throughput, energy use, adverse-weather robustness,
ultra-sparse sensors and non-planar terrain remain to be evaluated.
Future work includes range-conditioned feature fusion and hardware-specific
quantization. The architectural progression motivates matched ablations
and multi-seed evaluation before assigning gains to individual mechanisms.
```

Conclusion này giữ các giới hạn chính của bản gốc; không biến planned INT8/embedded/off-road thành kết quả đạt được. Raw exact counts của reference detectors là số đo cấu hình trong repo, không phải tuyên bố mọi implementation của PointPillars/SECOND có đúng số tham số đó.

## 7. Điều kiện đóng gate và dấu vết kiểm chứng

Đóng P9/P10 bằng các hành động đã nêu ở mục5; đồng thời thống nhất final-result snapshot, protocol bảng/hình và quy ước làm tròn. Sửa đơn vị gain thành pp, thay các câu vượt bằng chứng, sửa PR/loss captions và scope BEV/embedded. Mẫu sửa chỉ là đề xuất; gate của manuscript hiện tại vẫn chưa đóng khi các sửa đổi chưa được áp dụng và những control cần thiết chưa có.

Artifacts:

- [claim_evidence_matrix.json](artifacts/phase3/claim_evidence_matrix.json):91claim records,exactquotes,status,trần câu văn và nguồn đối chiếu.
- [issues.json](artifacts/phase3/issues.json):gộp root cause theo `ISSUE_SCHEMA.md`,giữ orphan gate labels riêng.
- [arithmetic_checks.json](artifacts/phase3/arithmetic_checks.json):44subtractions,raw-vs-rounded parameter reduction,payload/FPS arithmetic.
- [pr_figure_checks.json](artifacts/phase3/pr_figure_checks.json),[pareto_checks.json](artifacts/phase3/pareto_checks.json):numericchecks từ bảng và readable figure legends.
- [include_inventory.json](artifacts/phase3/include_inventory.json),[source_candidate_inventory.json](artifacts/phase3/source_candidate_inventory.json),[skill_claim_candidates.json](artifacts/phase3/skill_claim_candidates.json):include graph vàcandidates để trace coverage; keyword/script candidates không tự quyết định lỗi.
- [figure_assets.json](artifacts/phase3/figure_assets.json):hash của10PDFassets; cùng thư mục có text extracts và3PNGpreviews đã xem.
- [source_integrity.json](artifacts/phase3/source_integrity.json),[verification.json](artifacts/phase3/verification.json):quotes/schema/table structure/sources unchanged.

Tái chạy từ repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B paper/audit/artifacts/phase3/collect_evidence.py
PYTHONDONTWRITEBYTECODE=1 python3 -B paper/audit/artifacts/phase3/build_report.py
```

Các text extracts của PDF đã được giữ làm artifact; `collect_evidence.py` không đổi figure gốc. Học thuyết/citation bên ngoài, raw matched-local predictions và control runs chưa có được ghi đúng là missing evidence; audit không điền chúng bằng suy đoán. Phát hiện PR asset dùng shared-local baseline và reduction85.85% bổ sung độ chi tiết so với Phase2; không coi nhận định Phase2 là nguồn sự thật cao hơn file gốc.
