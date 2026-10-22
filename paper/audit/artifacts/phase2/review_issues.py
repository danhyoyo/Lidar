"""Session-specific reviewer judgments; proposed LaTeX is report-only."""


def record(identifier, reviewer, title, quote, explanation, source, line, anchors,
           fix, latex=None, severity="moderate", blocker=False, root=None, previous=None,
           comment_type="methodology"):
    return {"id": identifier, "reviewer": reviewer, "title": title, "quote": quote,
            "explanation": explanation, "comment_type": comment_type,
            "severity": severity, "display_severity": "MINOR" if severity == "minor" else "MAJOR",
            "source_kind": "llm", "confidence": "high", "source_section": source,
            "source_file": source, "source_line": line,
            "review_lane": "notation_and_numeric_consistency" if reviewer == 1 else "evaluation_fairness_and_reproducibility",
            "root_cause_key": root or identifier.lower(), "related_previous_issues": previous or [],
            "evidence_anchor": [{"type": "analysis_artifact" if x.endswith(".json") else "section", "text": x} for x in anchors],
            "gate_blocker": blocker, "fix": fix, "latex_example": latex}


ISSUES = [
record("T01", 1, "SG-FPN mô tả sai phép giảm kênh và trộn hai hệ chỉ số pyramid/stride",
       r"depthwise $3 \times 3$ refinement, BN, and SiLU produce $\mathbf{U}_4$ and $\mathbf{U}_3$",
       "L5/L4 có 48 kênh, L3 có 32 kênh. Upsample P4 vẫn có 48 kênh; một depthwise convolution thuần túy không thực hiện phép 48→32 như mô tả. Code dùng DWConv48→48 cho U4 nhưng Conv2d thường 48→32 cho U3. Ngoài ra Eq. scale_gating dùng r∈{8,4} trong khi L3/L4/L5 được đặt theo pyramid level: L_r chưa được ánh xạ rõ, và L4 dễ bị hiểu vừa là lateral stride 8 vừa là lateral stride 4 trong công thức tổng quát. Phép cộng trong code đúng; mô tả chưa đủ để tái dựng đúng toán tử.",
       "sections/methodology.tex", 285,
       ["paper/code_context.md:173–180", "Code/detector/core/models/backbones/mobilepixornext.py:188–208,273–288", "parameter_and_mac_audit.json"],
       "Ghi riêng hai phép upsample/refine, sử dụng chỉ số stride nhất quán, nêu align_corners=False. Giữ head input 32 theo resolved config; không đổi về 16 chỉ để khớp context cũ.",
       r"""% Report-only proposal; preserve the existing figure/equation labels when editing.
\begin{align*}
\mathbf L^{(16)}&=\operatorname{Conv}_{1\times1}^{128\to48}(\mathbf F^{(16)}),\\
\mathbf L^{(8)}&=\operatorname{Conv}_{1\times1}^{96\to48}(\mathbf F^{(8)}),\\
\mathbf L^{(4)}&=\operatorname{Conv}_{1\times1}^{48\to32}(\mathbf F^{(4)}),\\
\mathbf U^{(8)}&=\operatorname{SiLU}\!\left(\operatorname{BN}\!\left(
 \operatorname{DWConv}_{3\times3}^{48\to48}(\operatorname{Up}_2(\mathbf L^{(16)}))\right)\right),\\
\mathbf P^{(8)}&=\mathbf U^{(8)}+\mathbf G^{(8)}\odot\mathbf L^{(8)},\\
\mathbf U^{(4)}&=\operatorname{SiLU}\!\left(\operatorname{BN}\!\left(
 \operatorname{Conv}_{3\times3}^{48\to32}(\operatorname{Up}_2(\mathbf P^{(8)}))\right)\right),\\
\mathbf G^{(r)}&=2\sigma\!\left(\operatorname{DWConv}_{3\times3}
 (\mathbf L^{(r)}+\mathbf U^{(r)})\right),\quad r\in\{8,4\},\\
\mathbf P^{(4)}&=\mathbf U^{(4)}+\mathbf G^{(4)}\odot\mathbf L^{(4)}.
\end{align*}
Here, $\operatorname{Up}_2$ uses bilinear interpolation with
\texttt{align\_corners=False}. Only the stride-8 refinement is depthwise;
the stride-4 projection uses a standard convolution to reduce 48 channels to 32."""),

record("T02", 1, "Tia che khuất dùng point có reflectance như vector hình học 3D",
       r"$\mathbf{r}(t) = t \cdot \mathbf{p}$",
       "Point p_i đã được định nghĩa trong R4 với thành phần reflectance. Trong ray–box intersection và khoảng cách shadow, r(t)=t p và norm(p) không chỉ rõ phép lấy XYZ; nếu dùng tuple đã định nghĩa thì ray là R4 và norm trộn mét với intensity. Code thực tế dùng norm(points[:, :3]). Context cũng lặp lại cách viết r(t)=t p nên không thể coi context là căn cứ để giữ nguyên sự nhập nhằng chiều này. LOS/shadow tắt ở final run: đây là lỗi đặc tả nhánh tùy chọn, không chứng minh kết quả final sai.",
       "sections/methodology.tex", 146,
       ["paper/sections/methodology.tex:19,146–155", "paper/code_context.md:94,293–295", "Code/detector/core/datasets/augmentor/hybrid_physics.py:49–55"],
       "Phân biệt point có thuộc tính với vector tọa độ XYZ. Chuẩn khoảng cách và ray chỉ tác động lên XYZ; intensity không bị nhân theo tham số ray.",
       r"""For an attributed point $\mathbf p=(x,y,z,\iota)$, define its geometric
projection $\mathbf p^{\mathrm{xyz}}=(x,y,z)^\top\in\mathbb R^3$.
The sensor ray and the distance behind the box exit are
\[
 \mathbf r_{\mathbf p}(t)=t\mathbf p^{\mathrm{xyz}},\qquad
 d_{\mathrm{behind}}=(1-t_{\mathrm{leave}})
       \lVert\mathbf p^{\mathrm{xyz}}\rVert_2.
\]
Shadow masking applies to ray--box hits with $d_{\mathrm{behind}}>0.05\,\mathrm m$.
Reflectance $\iota$ is an attribute and is excluded from these geometric operations."""),

record("T03", 1, "Hoán vị corner 5−m cần quy ước thứ tự đỉnh tường minh",
       r"\mathbf{c}_{5-m}^*",
       "Đảo thứ tự [4,3,2,1] biểu diễn quay π đúng với thứ tự code [++, +−, −+, −−], nhưng không đúng với thứ tự chạy quanh chu vi thông dụng [++, +−, −−, −+]. Bài chỉ nói bốn corners, chưa định nghĩa thứ tự trước Eq. corner_loss. Script witness cho thấy reverse-order error gần 0 với code order nhưng lớn với cyclic order. Công thức và code không sai nếu giữ đúng thứ tự; thiếu convention sẽ khiến người tái lập cài một loss khác.",
       "sections/methodology.tex", 355,
       ["paper/code_context.md:320–322", "Code/detector/core/losses/oriented_geometry_loss.py:22,44–62,81–103", "mathematical_witnesses.json"],
       "Định nghĩa local axes, corners và permutation trước loss; hoặc chuyển cả công thức lẫn code sang thứ tự cyclic với phép dịch hai đỉnh. Không chỉ đổi một vế công thức.",
       r"""Let $\mathbf e_l=(\cos\psi,\sin\psi)^\top$ and
$\mathbf e_w=(-\sin\psi,\cos\psi)^\top$. With center $\mathbf c$, define
\[
\begin{aligned}
\mathbf c_1&=\mathbf c+\tfrac l2\mathbf e_l+\tfrac w2\mathbf e_w,&
\mathbf c_2&=\mathbf c+\tfrac l2\mathbf e_l-\tfrac w2\mathbf e_w,\\
\mathbf c_3&=\mathbf c-\tfrac l2\mathbf e_l+\tfrac w2\mathbf e_w,&
\mathbf c_4&=\mathbf c-\tfrac l2\mathbf e_l-\tfrac w2\mathbf e_w.
\end{aligned}
\]
Under this ordering, the $\pi$-rotation correspondence is
$\pi(m)=5-m$. The predicted and target corners use the same ordering."""),

record("T04", 1, "Loss tổng có trọng số nhưng các loss cơ sở và normalization chưa đủ đặc tả",
       r"\mathcal{T} = \{\mathrm{cls}, \mathrm{off}, \mathrm{size}, \mathrm{yaw}, \mathrm{geo}, \mathrm{qual}\}",
       "Các trọng số không bị thiếu toàn bộ: group weights 0.5/0.5, beta_corner=1, T-SBUW b=3, tau=2, EMA=0.99 đã được nêu. Tuy nhiên Methods chưa viết đầy đủ classification focal, ba masked regression losses và phép giảm trên positive mask. Context mô tả L1/N_pos, còn code OGA dùng Smooth-L1 chia cho tổng mask đã expand qua 2 kênh (2N_pos+1e-4). BCE quality chỉ giảm trên reg_mask. EMA còn có first-batch initialization và chỉ update khi training. Hai người chỉ đọc paper/context có thể cài khác objective; không nên khắc phục bằng thêm các lambda tùy ý.",
       "sections/methodology.tex", 384,
       ["paper/code_context.md:307–314,324–339", "Code/detector/core/losses/strategies/oga.py:28–37,70–115", "Code/detector/core/losses/l1_loss.py:10–14", "Code/detector/core/losses/uncertainty_weighting.py:108–121", "Code/detector/core/losses/focal_loss.py:14–39"],
       "Bổ sung loss primitives, mask/reduction, trạng thái EMA và hệ số quality=1. Đối chiếu context để sửa chỗ L1 thành Smooth-L1 khi dùng OGA hiện tại. Ghi rõ weights là adaptive chứ không phải sáu hằng lambda cố định.",
       r"""For group $g$, let $\Omega_g$ be its regression-positive cells,
$N_g=|\Omega_g|$, and let $Y_{gic}\in[0,1]$ be the Gaussian class target.
For $t\in\{\mathrm{off},\mathrm{size},\mathrm{yaw}\}$,
\[
\mathcal L_{g,t}=\frac{\sum_{i\in\Omega_g}\sum_{d=1}^{2}
 \rho(\hat y_{g,i,d}^{(t)}-y_{g,i,d}^{(t)})}{2N_g+10^{-4}},\quad
\rho(e)=\begin{cases}e^2/2,&|e|<1,\\ |e|-1/2,&\text{otherwise.}\end{cases}
\]
With $p_{gic}=\operatorname{clip}(\sigma(z_{gic}),10^{-4},1-10^{-4})$
and $N_g^{\mathrm{peak}}=\#\{(i,c):Y_{gic}=1\}$,
\[
\mathcal L_{g,\mathrm{cls}}=-\frac{
 \sum_{Y_{gic}=1}(1-p_{gic})^2\log p_{gic}
 +\sum_{Y_{gic}<1}(1-Y_{gic})^4p_{gic}^{2}\log(1-p_{gic})}
 {\max(N_g^{\mathrm{peak}},1)}.
\]
The quality BCE is averaged over $\Omega_g$ only, with unit fixed coefficient;
the geometry loss is also averaged over regression-positive cells.
Empty regression sets contribute zero regression, geometry and quality loss.
For each group and task, the first training batch initializes the detached
EMA to $\max(\mathcal L_{g,t},10^{-4})$; later training batches update it with
momentum 0.99. The EMA is frozen during validation.
The effective task coefficient is
\[
 \lambda_{g,t}^{\mathrm{eff}}=
 \frac12\,\overline\mu_g\,
 \frac{\omega_{g,t}}{\max(\mu_{g,t},10^{-4})},\qquad
 \overline\mu_g=\frac16\sum_{t\in\mathcal T}\mu_{g,t}.
\]
The fixed corner coefficient and quality coefficient are both 1.0.""",
       severity="major", blocker=True, comment_type="missing_information"),

record("T05", 1, "Q-OGA không chỉ thay positive classification target như paper đang mô tả",
       "Q-OGA modulates positive classification targets",
       "Context mô tả thêm soft-min π-corner và range-adaptive regression weighting. Code q_oga thực hiện cả hai, trong khi Methods chỉ cho positive CQFL term. Negative Gaussian-neighbor term, normalization, J=5 không có IQA và range weights/soft-min temperature chưa được khai báo cho nhánh này. Vì vậy lợi ích Q-OGA không thể hiểu là chỉ bỏ IQA head. Có thêm một sai khác hiện tại trong code: grid_pos dùng +0.5 cell rồi cộng target offset theo grid origin, nên RDA dùng vị trí lệch +0.2 m mỗi trục; cần xác minh chủ đích trước khi ghi một công thức true-center như đã khớp triển khai.",
       "sections/methodology.tex", 371,
       ["paper/code_context.md:341–348", "Code/detector/core/losses/quality_focal_loss.py:38–62", "Code/detector/core/losses/strategies/q_oga.py:23,35–44,174–219", "Code/detector/core/losses/smooth_corner_loss.py:29–58", "mathematical_witnesses.json"],
       "Mô tả Q-OGA là biến thể objective đầy đủ, khai báo soft-min/RDA và negative CQFL. Nếu chỉ muốn so sánh head-free quality coupling, chạy thêm variant chỉ đổi coupling. Kiểm tra và sửa/giải thích offset origin của RDA ở lượt code riêng; không âm thầm viết lại code trong audit.",
       r"""% Complete objective description; the range-coordinate convention must first
% be reconciled with q_oga.py:177--182 (current implementation adds half a cell).
For positive peaks, use the quality-coupled term in Eq.~(\ref{eq:cqfl_loss}).
For $Y<1$, retain $(1-Y)^4p^2[-\log(1-p)]$ and normalize the sum of positive
and negative terms by $\max(N^{\mathrm{peak}},1)$.
Q-OGA additionally uses
\[
 d_{\mathrm{sm}}=-\tau_{\mathrm{sm}}\log
   \left(e^{-d_{\mathrm{dir}}/\tau_{\mathrm{sm}}}
        +e^{-d_{\pi}/\tau_{\mathrm{sm}}}\right),
 \qquad \tau_{\mathrm{sm}}=0.05,
\]
where $d_{\mathrm{dir}}$ and $d_\pi$ are mean L1 corner distances in metres.
Divide $d_{\mathrm{sm}}$ by the target diagonal before averaging.
The coordinate losses are weighted by
\[
 w_i=1+1.5\left[\operatorname{clip}
       (r_i/70.4,0,1.5)\right]^2.
\]
Five tasks $\{\mathrm{cls},\mathrm{off},\mathrm{size},\mathrm{yaw},\mathrm{geo}\}$
are balanced separately per group; no IQA branch is used.
% Author action: define r_i from the verified coordinate convention.
% With the current code, its coordinate is c_i^*+(0.2,0.2) m, not exactly c_i^*.
% Do not publish a true-center formula without reconciling that difference.""",
       severity="major", blocker=True, comment_type="missing_information"),

record("T06", 1, "Algorithm chọn peak trên classification nhưng code chọn peak trên joint score",
       r"Identify local $3 \times 3$ candidate peaks on classification heatmap",
       "Algorithm bước 8→9 và prose cuối Methods chọn local maxima của p_cls trước khi kết hợp q. Code tính ranking_scores trước rồi max_pool2d trên ranking_scores. Hai thứ tự không giao hoán: với hai cell kề nhau p=[0.9,0.8], q=[0.04,0.81], classification chọn cell 0, joint score=[0.1897,0.8050] chọn cell 1; cả hai cell đều vượt ngưỡng 0.05. Context cũng ghi thứ tự cũ như paper nên phải đối chiếu trực tiếp postprocess.py.",
       "sections/methodology.tex", 428,
       ["paper/code_context.md:129–139", "Code/detector/postprocess.py:175–178,223–229", "mathematical_witnesses.json"],
       "Tính joint score rồi lấy peak từng class trong group; giữ hai điều kiện threshold, decode và class-wise NMS. Thêm global sort/top-500 vào pseudocode để tự chứa; sửa cả câu prose ở dòng 441.",
       r"""% Replacement for peak/scoring/filtering steps, matching postprocess.py.
\begin{enumerate}
\item For each class $c$ in its task group, compute
 $S_c=P_{\mathrm{cls},c}^{1-\alpha}\hat q^{\alpha}$, with $\alpha=0.5$.
\item Form the candidate mask
\[
 M_c=\bigl[S_c=\operatorname{MaxPool}_{3\times3}(S_c)\bigr]
 \land[P_{\mathrm{cls},c}>0.05]\land[S_c>0.05].
\]
\item Decode cells selected by $M_c$ using Eq.~(\ref{eq:box_decode}).
\item Apply rotated NMS separately per class at IoU threshold 0.10.
\item Merge groups, sort by $S_c$, and retain at most 500 detections globally.
\end{enumerate}
The output consists of oriented BEV footprints; this configuration does not
predict the height and vertical position required for full 3D boxes.""",
       severity="major", blocker=True),

record("T07", 1, "F7 là population standard deviation, không phải sample standard deviation",
       "and the sample standard deviation:",
       "Eq. z_std_encoding dùng 1/N nên là độ lệch chuẩn của tập điểm trong cell (population moment), trong khi chữ sample thường chỉ estimator chia N−1. Context và rasterizer cùng dùng population moment. Với normalized heights [0,1], công thức cho 0.5 còn sample standard deviation là khoảng 0.7071. Không cần đổi công thức/code để sửa một tên gọi.",
       "sections/methodology.tex", 69,
       ["paper/code_context.md:253–254", "Code/detector/core/datasets/utils_1/bev_backend.py:168–176", "mathematical_witnesses.json"],
       "Đổi sample standard deviation ở prose và caption thành within-cell population standard deviation hoặc vertical dispersion; giữ phép tính hiện tại.", severity="minor", comment_type="presentation"),

record("T08", 1, "Đối chứng stage-level SimAM được mô tả có linear projections nhưng code chỉ thêm scalar beta",
       "stage-level adapters add learned linear projections",
       "ResidualLightAttention hiện có một beta scalar và một SimAMCore không tham số; không có learned channel-linear projection. Điều này không phủ định Block-SimAM có zero parameters, nhưng lý do so sánh chi phí adapter cần mô tả đúng. Nếu bảng attention dùng một implementation lịch sử khác, phải khóa revision thay vì gán code hiện tại cho bảng đó.",
       "sections/methodology.tex", 261,
       ["Code/detector/core/models/backbones/light_attention.py:40–55,73–74"],
       "Mô tả adapter hiện tại là X+beta(SimAM(X)−X) với một scalar trainable; hoặc dẫn đúng revision của adapter được dùng trong các run lịch sử.", severity="minor", comment_type="claim_accuracy"),

record("M01", 2, "PC-GTA còn thiếu recipe và ngưỡng quyết định đủ để tái lập",
       "GT sampling was applied with probability $0.5$",
       "Paper đã nêu bật/tắt physics, xác suất GT/global transforms và ground-radius. Tuy vậy N_thresh không được gán giá trị; thiếu quota mỗi class, source-relative placement, jitter, collision margin, visibility survival/minimum points và quy tắc hỗ trợ ground. LIMIT_WHOLE_SCENE=true trừ số object hiện có khỏi quota 15/10/10. Cần phân biệt visibility flag=true với guard thực thi: protect=flag AND (shadow OR extra_width>0 OR margin<=1e-8); final shadow=false, extra_width=0, margin=0.1 nên không chạy kiểm tra visibility riêng. Các biến này thay đổi cảnh training và là một phần của module có gain +6.03. Resolved config và sampler cho phép khôi phục cụ thể; master config mặc định không phải recipe final.",
       "sections/experiments.tex", 49,
       ["paper/sections/methodology.tex:142", "Result/iqa_comparison_20261019/config.inference.json:augmentation", "Code/detector/core/datasets/augmentor/hybrid_sampler.py:89–106,153–178,246–247", "Code/detector/core/datasets/augmentor/hybrid_physics.py:12–25,64–68"],
       "Công bố resolved augmentation dictionary hoặc bảng recipe và version sampler. Phân biệt cơ chế đã dùng với LOS/shadow/radiometric tắt; không gán gain package cho các nhánh bị tắt.",
       r"""PC-GTA uses target scene quotas of 15 Cars, 10 Pedestrians and 10 Cyclists,
with scene-level probability 0.5 and source-relative placement.
The requested count for each class is its quota minus the existing count,
clamped below at zero (sampling rate 1.0).
The radial scale is uniform on $[0.9,1.1]$ and azimuth jitter is uniform on
$[-5^\circ,5^\circ]$. We use a 0.1\,m box collision margin, six placement
attempts and a candidate multiplier of three. The static obstacle count
threshold is two, using the sampler's elevated clearance column.
When the visibility check is active, it requires at least
$\min(n_{\mathrm{before}},5)$ surviving points and 50\% of the original points.
Its configuration flag is enabled, but the current implementation bypasses
the explicit check when shadow masking is off, removal widths are zero,
and collision margin is positive, as in the final configuration.
Ground support requires 6, 4, or 2 nearby low returns at ranges
$d\le15$\,m, $15<d\le30$\,m, or $d>30$\,m, respectively.
Accepted ground elevations lie in $[-2.5,-0.8]$\,m, with support within
0.35\,m above the estimated ground. Density subsampling preserves a minimum
of five source points when available. LOS rejection, shadow masking and
radiometric calibration are disabled in the reported final run.""",
       severity="major", comment_type="missing_information"),

record("M02", 2, "Target assignment chưa công bố ownership cho regression head dùng chung",
       "Regression targets were assigned within half the Gaussian peak radius",
       "Paper cho radius nhưng chưa ghi gaussian_overlap=0.1, Gaussian sigma, quantized peak và nearest_center ownership khi nhiều GT dùng cùng cell. Pedestrian/Cyclist dùng chung offset/size/yaw nên một cell chỉ giữ được một GT regression target. Code dự trữ quantized peak cho owner gần nhất, rồi gán các cell khác theo nearest center; exact ties giữ GT đến trước. Cách diễn giải anchor center trong Fig. 6 caption cũng dễ gây thêm +0.2 m vì Eq. gt_targets/code dùng grid origin không có +0.5.",
       "sections/methodology.tex", 325,
       ["Result/iqa_comparison_20261019/config.inference.json:data.regression_assignment", "Code/detector/core/datasets/dataset.py:434–446", "Code/detector/core/datasets/utils_1/gaussian.py:37–39", "Code/detector/core/datasets/utils_1/target_backend.py:29–77", "paper/sections/methodology.tex:301,321"],
       "Nêu nearest_center và tie/peak reservation; phân biệt raster-cell centroid offset với regression anchor-grid offset. Không thêm +0.5 vào decode vì đó sẽ thay đổi code contract.",
       r"""The heatmap radius uses a minimum overlap of 0.1 and is bounded below
by four cells. A Gaussian of diameter $2r+1$ uses standard deviation
$(2r+1)/6$; the regression support radius is $r/2$.
The continuous projected center is quantized by integer truncation for its
classification peak. Each shared regression cell is assigned to the nearest
GT center within the task group. Quantized peak cells are reserved for their
nearest owner; exact ties preserve the canonical GT order.
The regression anchor is
\[
 \mathbf a_{uv}=(x_{\min}+u s_{\mathrm{out}}\Delta_x,
                 y_{\min}+v s_{\mathrm{out}}\Delta_y),\qquad
 \mathbf o^*=\mathbf c^*-\mathbf a_{uv}.
\]
This anchor is distinct from the input raster cell center used for
$F_{12}$ and $F_{13}$; no half-output-cell term is added at decoding.""",
       comment_type="missing_information"),

record("M03", 2, "Ablation chưa chứng minh kiểm soát architecture, schedule và checkpoint policy",
       "All variants share identical MobilePIXORNeXt backbone",
       "Table loss ghi OGA+IQA/no-attention 0.721M và AP 70.58; final là 0.753569M với gain +0.88, mặc dù chính SimAMCore không thêm parameter. Với đúng final shared H32/F32 topology, script đếm cả có/không SimAM đều 753569; hai IQA heads chiếm 18626 tham số, khác mô tả 18.59K. Bài cũng tách 50-epoch standard/curves khỏi final 160-epoch run chọn epoch 130 nhưng không cung cấp run manifest từng hàng ablation. Chưa thể tách gain attention khỏi thay đổi depth/head/schedule; không có log A9 đủ để kết luận chính xác biến nào đã thay đổi. Cùng split và seed 42 không đủ để khẳng định controlled ablation.",
       "tables/loss.tex", 19,
       ["paper/tables/loss.tex:12–19", "paper/tables/components.tex:40–45", "paper/sections/experiments.tex:43,46", "paper/sections/results.tex:139", "parameter_and_mac_audit.json", "Result/benchmark_T4_20261018/sources/ours_best_ap.json:model.parameter_counts"],
       "Xuất mỗi run's resolved config/hash, params, depth, head architecture, optimizer/schedule, selection epoch/policy và GPU. Chạy cặp final architecture attention off/on với cùng training budget hoặc giới hạn Table components là progression. Xác minh param delta của IQA thay vì sửa số làm tròn theo phỏng đoán.",
       r"""% Safe description until run manifests and matched controls are available.
Table~\ref{tab:components} records an architectural progression.
The final checkpoint was selected at epoch 130 from a 160-epoch schedule;
the illustrated training curves cover 50 epochs.
The currently reported rows do not establish an architecture- and
schedule-matched estimate of the isolated Block-SimAM effect.
Such an estimate requires identical stage depths, shared-head topology,
training budget and checkpoint selection, with only Block-SimAM toggled.
% For the verified final topology only: both attention settings have 753,569
% detector parameters; removing both IQA branches removes 18,626 parameters.
% Do not replace historical row counts until their actual configs are recovered.""",
       severity="major", blocker=True),

record("M04", 2, "T-SBUW và các thành phần OGA chưa có kiểm chứng tách riêng",
       "evaluates its empirical convergence behavior",
       "A0–A10 có backbone, encoding, grouped width, GT, fusion, SG-FPN, OGA+IQA, attention; factorial loss có OGA và IQA riêng. Nhưng không có hàng bật/tắt T-SBUW hoặc tách bounded softmax khỏi EMA; cũng chưa tách projection GIoU khỏi π-corner. Loss curves của các objective khác nhau không đo riêng lợi ích của weighting và không so được trực tiếp khi normalization khác. PC-GTA được kiểm chứng ở mức package; các nhánh physics tắt không cần một hàng hiệu quả được gán cho final run.",
       "sections/experiments.tex", 46,
       ["paper/sections/methodology.tex:360,383–401", "paper/tables/components.tex:11–21", "paper/tables/loss.tex:10–17", "paper/sections/results.tex:139–146"],
       "Bổ sung matched ablation uniform weighting / EMA-only / T-SBUW, và projection-only / corner-only / combined OGA nếu muốn quy kết contribution của từng cơ chế. Nếu chưa có các run này, giới hạn kết luận vào hệ thống kết hợp; không bịa AP mới.",
       r"""The loss-factorial study evaluates the composite OGA objective and IQA.
It does not separate the effects of projection GIoU and the symmetric corner
term. No isolated ablation of EMA calibration and bounded task weighting is
reported. The loss trajectories therefore describe convergence of the full
objectives, rather than establish the independent benefit of T-SBUW.
% Proposed additional controls, with AP to be filled only from real runs:
% uniform task weights; EMA-only calibration; full T-SBUW;
% projection-only geometry; corner-only geometry; combined OGA.""",
       severity="major", comment_type="missing_information"),

record("M05", 2, "Cùng R40/split chưa đủ chứng minh tương đương evaluator cho so sánh AP",
       r"Both implementations used $R_{40}$ interpolation on the same validation split",
       "Local evaluator thực hiện matching/ranking riêng trên BEV outputs và quy tắc ignore/ROI riêng; log nói rõ không phải official hidden test. Table comparison phân biệt local và reference, là công khai tốt, nhưng chưa có equivalence check để dùng chênh lệch +1.33 hoặc matched SECOND như bằng chứng chắc chắn hơn về detector. Cần tái hiện cùng evaluator hoặc đối chiếu semantics của ignore/matching/range/candidate filtering. Không khẳng định một quy tắc cụ thể như DontCare đã khác khi chưa thẩm định reference implementation tương ứng.",
       "sections/experiments.tex", 37,
       ["Code/tools/kitti_training_pipeline/evaluate_kitti_bev.py:164–224", "Result/benchmark_T4_20261018/sources/ours_best_ap.json:protocol", "paper/tables/comparison.tex:23"],
       "Giữ công khai hai protocol; đánh giá matched protocol trên cùng predictions/GT nếu khả thi và trình bày mức chênh theo từng class. Cho tới lúc đó coi bảng cross-family là descriptive comparison, không test tương đương/thống kê thắng thua.",
       r"""MobilePIXORNeXt and historical MobilePIXOR use the local loader-aligned
BEV evaluator; reference detectors use the reference KITTI evaluator.
Although both report $R_{40}$ on the same validation split, evaluator
equivalence has not been established here. Cross-family AP differences are
therefore descriptive. A common-evaluator comparison is required before
attributing small differences solely to detector quality.""",
       severity="major", blocker=True, root="e03-headline", previous=["Phase 1 E03"]),

record("M06", 2, "Hardware/protocol của standalone timing và chi phí preprocessing chưa đủ tái lập",
       r"standalone network throughput reached $55.94$\,FPS",
       "Methods cuối gọi 55.94 FPS là standalone, nhưng Results/Table components dùng standalone=115.32 FPS và pipeline-forward=55.94. Code device_timed đo CUDA events và sync; E2E wall-clock có I/O/preprocess. Context/table encoding ghi L4 cho các số 169.97/165.02/169.01, Table components lại gán T4, đúng vấn đề Phase 1 E07. Chưa có log standalone tương ứng và host CPU/threading/loop protocol đầy đủ; chỉ x86_64 không tái lập được 20.94 ms CPU preprocessing. Training GPU cũng chưa được định danh riêng, không thể suy từ T4 evaluation và bf16 training rằng train trên T4.",
       "sections/methodology.tex", 466,
       ["paper/sections/results.tex:53", "paper/tables/components.tex:3,32–34,45", "paper/tables/encoding.tex:3,17", "paper/code_context.md:501", "Code/tools/kitti_training_pipeline/evaluate_kitti_bev.py:276–288,554–588", "Result/benchmark_T4_20261018/sources/ours_best_ap.json:runtime,latency"],
       "Đổi tên 55.94 thành in-pipeline model forward; khóa GPU và timer cho từng regime. Ghi training GPU, CPU model, threads/workers, warmup/iterations, precision/batch/caching và synchronization. Chỉ gán T4 cho standalone sau xác minh log; giữ đầu vào .bin→detections của E2E riêng khỏi network-only.",
       r"""The sequential evaluation pipeline uses batch size one and FP32 on
Tesla T4. Network execution within this pipeline is measured with CUDA
events and end-event synchronization: 17.88\,ms (55.94\,FPS).
End-to-end wall time includes binary-file loading, CPU rasterization, H2D,
network execution, decoding and NMS: 60.96\,ms (16.41\,FPS).
The first 10 frames are excluded from timing, leaving 3,759 timed frames.
The separate optimized figure, 8.67\,ms (115.32\,FPS), belongs to a distinct
standalone regime and is not substituted for end-to-end performance.
% Author verification required before submission:
% add the standalone GPU, warmup/iteration count, timer and synchronization;
% add the host CPU model/thread count and the actual training GPU;
% reconcile T4/L4 provenance for every reused ablation timing row.""",
       severity="major", blocker=True, root="e07-nguồn", previous=["Phase 1 E07"]),

record("M07", 2, "Compute coverage còn thiếu FLOPs/MACs dù đã có params, memory và timing",
       r"Peak allocated GPU memory was $98.44\,\text{MB}$",
       "Bài không thiếu toàn bộ efficiency evidence: có Params, checkpoint size, peak evaluation GPU memory, forward/E2E FPS và latency percentiles. Tuy nhiên chưa có FLOPs/MACs hoặc scope đếm operations. Script source-based tính khoảng 11.4496 G Conv2d MACs cho batch=1, input14×800×704; đây chỉ là phép tính giải tích của reviewer, không phải profiler chạy model, không bao gồm SimAM/BN/activation/interpolation/decode/NMS. Không được trình bày số này như số đo tác giả đã báo cáo hoặc dùng để suy ra FPS.",
       "sections/experiments.tex", 52,
       ["paper/sections/methodology.tex:458–466", "paper/tables/efficiency.tex", "Code/tools/benchmarks/profile_detector.py:98–138", "parameter_and_mac_audit.json"],
       "Bổ sung compute count có profiler/version, batch/input shape, quy ước FLOP/MAC và danh sách operators bỏ qua. Tách peak inference allocation khỏi training memory. Thiếu MACs hiện là bổ sung nên làm, không tự nó là submission blocker.",
       severity="minor", comment_type="missing_information"),
]
