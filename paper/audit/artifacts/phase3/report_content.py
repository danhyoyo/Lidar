"""Report-only revision examples and contribution-chain review."""

CHAINS = [
 ('P0','Hệ thống compact BEV','Intro L15','Method L12–466','comparison + efficiency + components A0/A10','Conclusion L5,7','ĐỦ; accuracy/latency scope cần sửa C09–C18.'),
 ('P1','Strata-BEV14','Intro L15','Method §Strata-BEV14 L42–122','encoding:TriBand/Binary35/rich8/hist14;components A1–A3','Conclusion L5','ĐỦ ở mức encoding package; không isolate riêng centroid/moments.'),
 ('P2','MobilePIXORNeXt-S backbone','Intro L15','Method §Backbone L176–229','components A0→A1','Conclusion L5','ĐỦ; các manifests cần xác minh cho causal effect.'),
 ('P3','Block-SimAM placement','Intro L15,19','Method L231–261','attention rows;components A9→A10','Conclusion L5','ĐỦ mắt xích; matched topology/budget chưa xác nhận, C32–C33.'),
 ('P4','SG-FPN','Intro L15','Method L264–291','components A7→A8:−0.74pp','Conclusion L5','ĐỦ; negative ablation vẫn là bằng chứng. Không suy gain hoặc artifact-free.'),
 ('P5','GroupedHeader','Intro L15','Method L293–325','components A3→A4→A5','Conclusion L5','ĐỦ; grouping và width là hai thay đổi, chưa đo gradient interference.'),
 ('P6','PC-GTA active package','Intro L17','Method L124–174','components A5→A6:+6.03pp','Conclusion L5,7','ĐỦ package-level; optional disabled branches không có gain riêng.'),
 ('P7','Composite OGA','Intro L17','Method L327–360','loss:baseline/OGA × IQAoff/on','Conclusion L5,7','ĐỦ composite objective; projection/corner chưa được isolate riêng.'),
 ('P8','IQA','Intro L17','Method L362–368','loss factorial explicit quality head','Conclusion L5,7','ĐỦ; chưa có ranking/calibration metric riêng ngoài AP.'),
 ('P9','T-SBUW như đóng góp chính','Intro L17','Method L383–401','THIẾU weighting-off/uniform/EMA-only/full comparison','Conclusion L5','CRITICAL BLOCKER — ORPHAN_CONTRIBUTION: thiếu mắt xích ablation.'),
 ('P10','Q-OGA được tuyên bố “We developed”','THIẾU trong contribution list','Method L370–381','loss:Q-OGA row69.20%;Results L127','THIẾU Q-OGA trong Conclusion','CRITICAL BLOCKER — ORPHAN_CONTRIBUTION: thiếu Intro và Conclusion nếu giữ là đóng góp mới.'),
 ('P11','Empirical accuracy/size/latency trade-off','Intro L19','Method infer contract + Experiments metrics','comparison,efficiency,components,attention,loss','Conclusion L7','ĐỦ; sửa protocol/rounding/edge-extrapolation trước khi dùng superiority wording.'),
]

ORPHAN_EXPLANATION = """
Mỗi hàng trên là một đơn vị đóng góp có ý nghĩa ở cấp module hoặc objective, không phải mỗi channel và mỗi toán tử con. Conclusion có thể tóm tắt một module bằng tên module; không bắt buộc lặp mọi công thức. Vì vậy không gán orphan cho từng feature F0–F13, từng primitive của composite OGA, hoặc từng nhánh sampler đã công khai là tùy chọn. Việc thiếu isolated ablation của những primitive đó vẫn giới hạn các suy luận về cơ chế (C34, C38, C45, C67, C87–C88).

**P9 / C35:** T-SBUW được đặt trong contribution bullet, có Methods và được nhắc lại ở Conclusion, nhưng không có hàng ablation kiểm tra riêng. Loss curves không thay thế can thiệp tắt/bật weighting. Đóng lỗi bằng matched runs uniform / EMA-only / full T-SBUW, hoặc chuyển weighting sang implementation detail và bỏ tuyên bố độc lập về gain/stability. Không được viết thêm một giá trị AP chưa chạy.

**P10 / C39:** Q-OGA có lời tuyên bố phát triển phương pháp, subsection và bảng kết quả nhưng thiếu ở Intro contribution list và Conclusion. Có thể thêm một câu ở mỗi nơi, với AP69.20% và phạm vi head-free alternative, hoặc xác định rõ đây là auxiliary comparison. Mẫu bên dưới chọn cách thêm cả hai mắt xích; chưa có chỉnh sửa nào được áp dụng vào manuscript.

Optional LOS/shadow/radiometry được trình bày là capability; final run tắt chúng. C34 yêu cầu thu hẹp câu “address additional synthesis artifacts” thành một mô tả tính năng chưa được đánh giá độc lập. Nếu tác giả tiếp tục trình bày từng nhánh này như **đóng góp mới đã được kiểm chứng**, chúng cũng phải có mắt xích ablation; chúng chưa đạt điều kiện đó trong bản hiện tại. Future-work INT8/RC-SGFPN/off-road là mục tiêu tương lai được ghi rõ, không phải kết quả đã tuyên bố đạt được.
"""

REVISIONS = r"""
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
"""
