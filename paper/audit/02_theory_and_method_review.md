# Phase 2 — Specialist Deep Review: theory, method và khả năng tái lập

**Bản thảo:** `Paper/MobilePIXORNeXt.tex` và các file được include; target Arabian Journal for Science and Engineering (AJSE). **Ngày:** 2026-10-22. **Kết luận:** `MAJOR_REVISION` trước submission; **12 MAJOR, 3 MINOR, 0 CRITICAL được chứng minh**. Sáu điểm cần đóng trước khi coi phương pháp/kết luận chính đủ vững: T04–T06, M03, M05–M06. Không có bằng chứng để kết luận mọi kết quả vô hiệu hoặc phải loại ngay bài.

Đây là một lượt audit tuần tự dưới **hai vai reviewer chuyên môn**, không phải hai phản biện độc lập hay quyết định của AJSE. Phạm vi theo yêu cầu là Methods/Experiments, đọc kèm Results, bảng và code để đối chiếu; không lặp toàn bộ Phase 0.5/1 và không tra cứu external novelty. Toàn bộ nhận định chuyên môn mang nhãn `[LLM]`; phép đếm/check chạy thật mang nhãn `[Script]` ở phần bằng chứng. Theo `ISSUE_SCHEMA.md`, JSON giữ severity `major|moderate|minor`; để đáp ứng mức hiển thị yêu cầu, `major` và `moderate` đều hiển thị **MAJOR**, `minor` hiển thị **MINOR**. Canonical counts: 8 major, 4 moderate, 3 minor.

Đã áp dụng skill [paper-audit](/home/duyennh/.agents/skills/paper-audit/SKILL.md), [DEEP_REVIEW_CRITERIA.md](/home/duyennh/.agents/skills/paper-audit/references/DEEP_REVIEW_CRITERIA.md), [ISSUE_SCHEMA.md](/home/duyennh/.agents/skills/paper-audit/references/ISSUE_SCHEMA.md), kèm hướng dẫn theory/notation/methodology và method-interface. Chỉ ghi report/artifacts; **không chỉnh `.tex`, config hay code**. LaTeX dưới đây là mẫu sửa để tác giả review; các dữ kiện cần run/log mới được đánh dấu rõ, không điền AP giả.

Thang điểm: **1** = không đủ đặc tả/cơ sở; **2** = thiếu sót lớn ảnh hưởng tái lập/kết luận; **3** = nền tảng rõ nhưng cần major revision; **4** = phần lớn đầy đủ, thiếu sót nhỏ; **5** = đầy đủ và được kiểm chứng trong phạm vi bằng chứng hiện có. Điểm là reviewer judgment, không phải xác suất acceptance hay công thức từ số lỗi.

## 1. Nguồn, phiên bản và vị trí trích dẫn

Entry thực tế là [MobilePIXORNeXt.tex](../MobilePIXORNeXt.tex); không có `Paper/main.tex` trong snapshot này. Proposed Method được include ở entry L70, Experiments L72, Results L74. Mỗi finding dẫn **file con và dòng thật**, không gán số dòng của file ghép thành dòng của `main.tex`. `Paper` là symlink tới `paper`; các đường dẫn hai cách viết cùng trỏ một nguồn.

Đã đối chiếu trực tiếp [code_context.md](../code_context.md) với [resolved final config](../../Result/iqa_comparison_20261019/config.inference.json), [canonical best-AP benchmark log](../../Result/benchmark_T4_20261018/sources/ours_best_ap.json), architecture/loss/target/sampler/postprocess và evaluator trong `Code/`. Log final có detector 753.569 tham số, checkpoint epoch 130 của schedule 160 epoch; identity ghi stages 3/4/2, F32, H32, OGA+IQA, SimAM c3/c4/c5, seed 42. Các giá trị trong log là **bằng chứng được đọc lại**, không phải audit đã tự huấn luyện/chạy benchmark lại model.

### 1.1 Các bất đồng của context cần xử lý trước khi nói “khớp 100%”

| `code_context.md` | So với final source/config/log | Cách xử lý khi audit |
|---|---|---|
| L59: raster CUDA | Methods caption/actual backend CPU NumPy/Numba | Paper đúng final; context cần refresh. |
| L70,159,215:depth2/4/2,total8 | Resolved config3/4/2,total9;paper cũng9 | Không “sửa” paper thành8. |
| L182–198,222–225:output/head16 | Final F32/sharedH32;paper32 | Context stale; literal8-block/H16 topology tính ra651217params, không753569. |
| L278: gọi biểu thức inverse-energy là minimal energy | Paper tách e* và 1/e*,code importance là inverse | Paper/code khớp; context sai naming. |
| L313–314:L1/Npos | OGA code Smooth-L1/(2Npos+1e-4) | T04 là thiếu loss primitive trong paper; không lấy context L1 làm chuẩn. |
| L378:extended130epochs | Final schedule160,selectedcheckpoint130 | 130 là selected epoch; paper đã phân biệt đúng. |
| Physics package mô tả toàn bộ optional branches | Final LOS/shadow/radio=false;visibility flag=true nhưng explicit guard bypass | Toggle disclosure phần lớn đúng; kiểm tra runtime conditions M01. |
| Context inference:peak(cls) trước joint score | Code maxpool joint score | Cả context và paper cần cập nhật; T06. |
| Q-OGA có soft-min/RDA | Code xác nhận;Methods chỉ positive CQFL | Đây là context hữu ích, dẫn T05. |
| §6.6:R37 AP71.50/73.45 | Canonical final eval71.461874/73.418711, paper71.46/73.42 | Hai snapshot số liệu; không flag final table chỉ vì khác một bảng cũ. |
| Timing L4/T4 provenance | Encoding tableL4,componentsT4 reuse cùng số | Chưa thể chọn GPU đúng nếu thiếu raw standalone logs; M06. |

Nguồn ưu tiên khi có mâu thuẫn: **checkpoint identity + resolved config + benchmark log đúng model → implementation của nhánh được chọn → context tổng hợp → default config**. Default `Code/configs/config.json` là rich8/F16/noIQA/baseline/50epochs, không phải final hist14/F32/H32/OGA160. Context là tài liệu đối chiếu được yêu cầu, nhưng không đủ nhất quán để chứng nhận khớp100%.

## 2. Reviewer 1 — Theoretical Contribution & Mathematical Rigor

| Tiêu chí | Điểm /5 | Căn cứ |
|---|---:|---|
| Ký hiệu và convention | 3 | Ký hiệu phần lớn được định nghĩa; R4/ray và level/stride cần sửa; corner order thiếu. |
| Chiều tensor và khớp triển khai final | 3 | Numeric shape chain khớp; SG-FPN operator chưa đúng; context có nhiều snapshot cũ. |
| Loss, trọng số và numerical rigor | 2 | Adaptive/fixed weights đã có; primitive reductions, EMA lifecycle và Q-OGA còn thiếu. |
| Pseudocode và inference contract | 2 | Sai thứ tự peak/joint score tạo output khác trên counterexample cụ thể. |
| Động lực hình học/toán học | 3 | Moments,projection/corner có động lực; gate/attention và các heuristic cần evidence đúng mức. |
| Mức đóng góp lý thuyết | 3 | Có engineering/objective integration; không yêu cầu theorem, chưa xác minh novelty toàn literature. |

### 2.1 Bảng ký hiệu theo các tiểu mục Methodology

| Ký hiệu / họ ký hiệu | Định nghĩa, miền và lần xuất hiện | Kiểm tra / kết luận |
|---|---|---|
| $\mathcal P,\mathbf p_i,N_{\rm pts},i$ | L19: tập điểm, point $(x,y,z,\iota)\in\mathbb R^4$, số điểm, chỉ số | Được định nghĩa; không phải input chỉ có XYZ. |
| $x_i,y_i,z_i,\iota_i$ | L19: tọa độ mét và reflectance vô thứ nguyên | Đúng; geometric ray phải bỏ intensity: T02. |
| $x_{\min/\max},y_{\min/\max},z_{\min/\max}$ | L20–27: giới hạn ROI | Đúng; ký hiệu $z_{\max}$ trong tên channel là statistic, cần phân biệt với bound qua ngữ cảnh. |
| $\epsilon$ | L27: inward margin 0.001 m | Đừng dùng cùng giá trị này cho epsilon loss 1e-6. |
| $\Delta_x,\Delta_y,u_i,v_i$ | L27–30: resolution 0.1 m, floor indices | $u$ dọc x/W, $v$ dọc y/H; khớp code. |
| $H_{\rm in},W_{\rm in},H_{\rm out},W_{\rm out},s_{\rm out}$ | L27,34: 800,704,200,176,4 | H/W đúng; không đảo 704×800 khi viết BCHW. |
| $\mathbf b_k,k,c_k,x_k^*,y_k^*,l_k^*,w_k^*,\psi_k^*,s_k$ | L34–38: oriented BEV box và score | 7-tuple không có z/h, không phải full 3D box. Score là output, không phải GT geometry. |
| $\mathcal P_{u,v},N$ | L60: tập điểm và số điểm trong một cell | Rescope hợp lệ so với $N_{\rm pts}$. |
| $\zeta_i$ | L60: normalized height | Scalar [0,1]; input finite/ROI-filtered. |
| $b,N_b,N_{\rm norm},d_{\log}$ | L62–67: bin 0–3, count,32, clipped log mapping | Bin width 0.875 m; không phải 35 z-slices của binary35. |
| $F_0,F_1,F_2,F_3$ | L63: density từng stratum | Scalar/cell, vô thứ nguyên, đúng. |
| $F_4,F_5,F_6,F_7$ | L69–76: max/mean/span/std normalized height | Công thức đúng; tên sample std sai convention: T07. |
| $F_8,F_9$ | L79–82: max/mean intensity | Đúng scalar; calibration/material semantics chưa chứng minh robustness riêng. |
| $F_{10},F_{11}$ | L84: total log density, occupancy | Đúng; empty cell toàn 0. |
| $x_u^{\rm cell},y_v^{\rm cell},F_{12},F_{13}$ | L85–91: raster centroid anchor, offsets normalized | Có +0.5 ở raster cell; khác regression anchor origin. |
| $\mathbf X,B$ | L92: $B\times14\times800\times704$ | B dùng theo convention batch trước khi gloss tường minh; thêm glossary sẽ rõ hơn. |
| $(x,y,z,l,w,h,\psi),d$ | L131: inserted object tuple và planar range | z geometric center; code lưu bottom internally nhưng chuyển đổi được, không phải lỗi. |
| $R(d),\hat z_{\rm ground},\mathcal N(x,y),z_{insert}$ | L134–140: support radius,5th percentile ground,neighborhood,center height | Units radius m; support counts/z bounds thiếu: M01. |
| $N_{\rm static},N_{\rm thresh}$ | L142: static point count/rejection threshold | Threshold chưa gán số; sampler default2 trên clearance column: M01. |
| $\mathbf p,\mathbf r(t),t,t_{\rm leave}$ | L146–155: ray-box intersection | Thiếu projection $\mathbf p^{xyz}$; T02. |
| $\mathcal P_{augmented},\mathcal P_{target},\mathcal P_{object},\operatorname{Mask}_{shadow}$ | L147–156: augmented scene,target scene,inserted object,mask | Tập membership rõ; khoảng cách phải là norm XYZ. |
| $P_{\rm keep},r_{\rm src},r_{\rm tgt}$ | L159–162: keep probability và source/target range | Vô thứ nguyên; code có minimum surviving points, M01. |
| $\eta_{int},I_{src},I_{tgt},\gamma$ (radiometry) | L164–172: intensity ratio compensation,$\gamma=1.7$ | I khác notation intensity $\iota$ nhưng vai trò rõ; tắt final. $\gamma$ CQFL=2 là local rescope, nên đặt subscript. |
| $C_2,C_3,C_4,\mathbf F^{(2)},\mathbf F^{(4)},\mathbf F^{(8)},\mathbf F^{(16)}$ | L197–202: widths48/96/128 và features theo stride | Code gọi stages c3/c4/c5 theo stride4/8/16; cần ánh xạ, không phải thiếu stage. |
| $C,H,W,\mathbf Z,\mathbf V,\mathbf R,\mathbf Z',e$ | L222–244: generic tensor axes,input/intermediate/residual/output,expansion2.5 | B/C/H/W được khai báo qua tensor shape; gloss thêm bằng lời sẽ rõ hơn. H/W local theo stage; không có D voxel axis. |
| $\operatorname{DWConv},\operatorname{BN},\operatorname{SiLU},\operatorname{PWConv}$ | L223–229: Next block operators | 7×7 depthwise và 1×1 expansion2.5C đúng, no change spatial shape. |
| $W_{\rm exp},W_{\rm red}$ | L229: $(2.5C,C,1,1)$ và $(C,2.5C,1,1)$ | Hidden120/240/320 integer cho stages final. |
| $\boldsymbol\gamma_{\rm blk}$ | L229: channel-wise LayerScale, init1e-5 | Broadcast $(1,C,1,1)$ cần hiểu; không phải CQFL gamma. |
| $M,\mu_c,\nu_c,t,\lambda_{sim},e_t^\star,s_t$ | L238–252: spatial count HW,mean,sample variance,feature element,1e-4,energy,inverse energy | Code match; max(M−1,1) guarded. t feature không phải ray parameter trong subsection trước. |
| $\mathbf A,\mathcal S,\mathbf R$ | L254–257: sigmoid mask,SimAM operator,residual feature | Shape BCHW; “3D attention” ở đây là C/H/W, không phải voxel XYZ. |
| $\mathbf L_5,\mathbf L_4,\mathbf L_3,\mathbf U_4,\mathbf U_3,\mathbf P$ | L283–291: lateral/upsample/pyramid features | Numeric shapes đúng nhưng operator/channel transition chưa đúng: T01. |
| $r,\mathbf G_r,\mathbf L_r,\mathbf U_r,\mathbf P^{(r)}$ | L286–290: r∈{8,4} stride,gates | Trộn stride với level indices3/4/5; cần binding T01. |
| $\mathbf F_{\rm out}$ | L291: $(B,32,200,176)$ | Đúng final; context output16 đã cũ. |
| $g,K_1,K_2$ | L310–316: task group và class counts1/2 | Car riêng; Ped/Cyc shared regression head. |
| $p_{cls},\hat o_x,\hat o_y,\hat d_1,\hat d_2,\hat a_1,\hat a_2,\hat q$ | L312–316: head outputs | Channels K/2/2/2/1; raw logit vs sigmoid cần ghi rõ khi dùng loss/inference. Yaw outputs không bị hard-constrain unit circle. |
| $\mathbf o^*,\mathbf d^*,\mathbf a^*,\mathbf M_{reg}$ | L320–325: metric offset,log(w,l),cos2ψ/sin2ψ,mask | Targets match code; offset không chia output stride và không +0.5. |
| $P$ trong Fig. head geometry | L301: gọi anchor center | Plain point P khác $\mathcal P$ point cloud, không collision formal; anchor origin wording dễ sai M02. |
| $C,\Delta x,\Delta y,t_w,t_l,c_2,s_2,\theta$ (Fig. head) | L300–301: object center,metric displacements,log sizes,doubled-angle components,heading | C là point trong figure nhưng channel count trong tensor context; local rescope rõ. θ≡ψ; Δx figure displacement khác resolution $\Delta_x$ có subscript. |
| $\widehat{\mathbf c}_m,\mathbf c_m^*,m,\mathbf n_k,\mathbf n_{len/wid}^{pred/gt}$ | L334–339: 4 corners và axes R2 | Thứ tự corner chưa định nghĩa; axes và projections dimensionally valid. |
| $I_k,U_k,E_k,g_k,\bar g$ | L336–348: interval lengths,1D GIoU,mean4 | I/U/E là mét; g vô thứ nguyên. Proxy không đồng nhất rotated area IoU. |
| $\ell_{\rm proj},\ell_{\rm corner},\mathcal L_{\rm geo},\beta_{\rm corner}$ | L341–360 | Loss scalar; corner L1 distance/GT diagonal dimensionless; beta1 clear. T03. |
| $\hat q,\hat Z_{qual},q^*,\operatorname{stopgrad},\mathcal L_{qual}$ | L364–367: sigmoid quality,logit,target clipped proxy,BCE | Need positive-mask reduction; T04. $q^*$ dùng proxy không phải exact IoU. |
| $y^*,\beta,\hat p,\gamma,\mathcal L_{\rm CQFL}$ | L372–380: quality target exponent1,pred classprob,focal2 | $\hat p$ và $y^*$ implied classification role; should gloss. Missing negatives/full objective T05. |
| $\mathcal T,J,k,j,a_k,\bar a_k,b_{uw},\tau_{uw},Z_{uw},\omega_k$ | L384–394:6 tasks,indices,logits,bound3,temp2,normalizer,weights | k reused object/task locally; acceptable. Add g-index for 2×6 independent logits. |
| $\beta_{EMA},\mu_k,\mathcal L_k,\mathcal L_g,\mathcal L_{total}$ | L396–401: momentum.99,EMA means,task/group/total losses | $\mu_c$ SimAM vs $\mu_k$ EMA disambiguated by context; initialization/freeze absent T04. |
| $\tau_{score},\tau_{nms},\alpha,\mathcal B,M,\mathbf P_{cls},\widehat{\mathbf o},\widehat{\mathbf d},\widehat{\mathbf a},\widehat{\mathbf q}$ | L413–434: infer thresholds,output set,count,head maps | α trong algorithm bằng αscore trong prose. M là detection count, không phải HW SimAM; local rescope hợp lệ. Peak ordering sai T06. |
| $\hat x,\hat y,\hat w,\hat l,\hat\psi$ | L442–448: metric decode,exp sizes,half-atan2 | Consistent size order log(w,l), return format l,w explicit. Finite guards not fully described. |
| $s_{det},p_{cls},\alpha_{score}$ | L452–455: probabilistic score with exponent.5 | Dimensionless [0,1]; should be used before peak selection per code. |

Không tìm thấy collision nghiêm trọng chỉ do dùng lại chữ F hoặc P: $F_j$ scalar encoding, $\mathbf F$ feature tensor và $\mathcal P$ cloud là ba đối tượng có typography khác nhau. Các rủi ro thực tế là **chỉ số level/stride (T01), point R4/ray R3 (T02), corner order (T03)**. Nên định nghĩa B/C/H/W, gloss $\hat p$ và thêm group-index ở weighting để giảm suy diễn; các điểm convention này không được nhân thành nhiều lỗi MAJOR.

### 2.2 Hợp đồng chiều tensor

| Bước | Tensor / kích thước final | Kết quả đối chiếu |
|---|---|---|
| Points một frame | $(N_{pts},4)$, XYZ+reflectance | Không có arbitrary C extra features trong final input. |
| Geometry metadata | $(W,H,D)=(704,800,35)$ nếu xét z_res0.1 | D35 là geometry/binary slicing; hist14 không đưa D axis vào CNN. |
| Hist14 raster | YXC=(800,704,14)→BCHW=$(B,14,800,704)$ | CPU/Numba; 4 height strata rộng.875m. |
| Stem stride2 | $(B,32,400,352)$ | 3×3 stride2 rồi 3×3 stride1. |
| Stage2 stride4 | $(B,48,200,176)$,3 blocks | Code key c3. |
| Stage3 stride8 | $(B,96,100,88)$,4 blocks | Code key c4. |
| Stage4 stride16 | $(B,128,50,44)$,2 blocks | Code key c5. |
| PW expansion mỗi stage | 48→120→48;96→240→96;128→320→128 | 2.5C integer; residual addition hợp lệ. |
| SimAM | mean/var $(B,C,1,1)$,mask $(B,C,H,W)$ | 9 blocks,zero added trainable parameters. |
| Laterals | L5:48×50×44,L4:48×100×88,L3:32×200×176 | Batch dimension chung B. |
| SG-FPN stride8 | upsample→48×100×88,DW48→48,gated lateral48 | Đúng code. |
| SG-FPN stride4 | upsample→48×200×176,**regularConv48→32**,gate32 | T01; pure depthwise như prose không mô tả đúng operator. |
| Neck/head shared input | $(B,32,200,176)$ | Context16 cũ; Methods final32 đúng. |
| Group Car | cls1,offset2,size2,yaw2,quality1 | Mỗi map chung B,200,176. |
| Group Ped/Cyc | cls2,offset2,size2,yaw2,quality1 | Regression/quality shared giữa2classes: cần ownership M02. |
| Positive geometry | centers$(N_+,2)$,corners$(N_+,4,2)$,axes$(N_+,4,2)$ | T03 corner order; positives là regression cells, không luôn số object. |
| Loss tasks | scalar×6/group,2groups,12 learned logits | 12 chỉ training, không cộng vào detector753569. |
| Final detections | $(N_{det},7)$: class,score,x,y,l,w,yaw | Ndet≤500global; oriented BEV, không full3D. |

### 2.3 Kiểm từng nhóm phương trình

| Nhãn phương trình, dòng Methodology | Kiểm tra đại số/đơn vị và đối chiếu triển khai |
|---|---|
| `eq:spatial_roi`,20–25 | Khoảng scalar mét đúng; prose ngay sau bổ sung open ε nên không coi ký hiệu interval đóng riêng lẻ là lỗi code. |
| `eq:grid_assignment`,28–30 | Floor của tỉ số m/m; x→u∈[0,703], y→v∈[0,799] sau ROI. |
| `eq:target_tuple`,35–37 | BEV state7 gồm class/confidence; không có prediction z,h. |
| `eq:density_encoding`,63–65 | F0–3 và F10 scalar clipped log; density_norm32 đúng. |
| `eq:z_std_encoding`,70–75 | Vô thứ nguyên; population moment đúng code, tên sample sai T07. |
| `eq:intensity_encoding`,79–81 | Scalar [0,1]; matching clamp/scale1. |
| `eq:offset_encoding`,85–90 | Mean displacement/(N×cell width) dimensionless; đúng +0.5 ở raster. |
| `eq:ground_fitting`,134–137 | Radius2.5–4m, slope0.03; hợp lệ engineering heuristic, thiếu support recipe M01. |
| `eq:shadow_mask`,147–155 | Set operation đúng nếu ray chỉ XYZ; T02. Optional branch tắt final. |
| `eq:inverse_square`,159–161 | Probability min1(range ratio²); khớp raw rule, minimum-points guard chưa nói M01. |
| `eq:next_block`,223–227 | C→2.5C→C, spatial HW giữ nguyên, LayerScale broadcast đúng. |
| `eq:spatial_moments`,238–243 | Reduce HW per B,C, sample variance/(M−1); khớp SimAM code. |
| `eq:simam_energy`,246–251 | e* và nghịch đảo đúng (context từng gọi nhầm nghịch đảo là energy); λ1e-4 cùng scale squared activation. |
| `eq:simam_modulation`,254–256 | A và R cùng BCHW, Hadamard hợp lệ. |
| `eq:scale_gating`,286–289 | Gate 2σ range(0,2), addition hợp lệ sau explicit48→32; operator/index mismatch T01. |
| `eq:gt_targets`,320–323 | Offset2m,size2 log metre-values,yaw2 unitless; origin contract đúng, ownership thiếu M02. |
| `eq:projection_axes`,336–338 | Four R2 unit directions; duplicate axes được phép khi boxes aligned. |
| `eq:mgiou_loss`,341–347 | I/U/E interval lengths, average4 dimensionless, projection loss=(1−g)/2; là proxy không areaIoU. |
| `eq:corner_loss`,351–358 | Sum L1/4/GT diagonal dimensionless; π permutation đúng chỉ với code corner order T03. |
| `eq:iqa_loss`,364–366 | stopgrad clipped proxy và BCE scalar đúng; mask/reduce thiếu T04. |
| `eq:cqfl_target`,372–374 | q^β,β1 đúng positive target. |
| `eq:cqfl_loss`,377–379 | Positive formula đúng; full negatives, RDA,soft-min và J5 chưa đủ T05. |
| `eq:tsbuw_weights`,386–393 | Softmax weights sumJ, bounded logits đúng, b3/τ2 khớp; sumweights không có nghĩa gradient norm conserved. |
| `eq:balanced_loss`,397–399 | µbar ΣωL/maxµ scalar cùng loss scale; detached EMA init/update còn thiếu T04. |
| `eq:box_decode`,442–447 | Offset cộng grid origin m; exp log(w,l); half atan2 đúng symmetryπ. |
| `eq:joint_score`,452–454 | p^(1−α)q^α,α.5 đúng; thứ tự thuật toán sai T06. |
| Không đánh số: radiometric η,164–172 | Ratio exponent max(0,2−1.7)=.3 đúng; disabled, không gán gain final. |
| Không đánh số: Lgeo,360; Ltotal,401 | βcorner1 và group weights.5/.5 đã có; objective primitives cần bổ sung. |

Inventory có **26 equation/align blocks đánh số** và **337 inline math spans** (kể cả lần lặp/caption, không phải 337 biến khác nhau), lưu tại `artifacts/phase2/equation_inventory.json`. Bảng trên bao phủ cả công thức inline trọng yếu; không chứng nhận một formal proof hay mọi trạng thái số học trên dữ liệu thực.

### 2.4 Trọng số loss: phần nào đã có, phần nào chưa đủ

Không có căn cứ phê bình “Ltotal thiếu mọi λ”. Methods đã khai báo $\mathcal L_{total}=0.5\mathcal L_1+0.5\mathcal L_2$, $\mathcal L_{geo}=\ell_{proj}+1.0\ell_{corner}$, sáu adaptive weights mỗi group, b=3, τ=2, EMA=.99. Config final đặt `iou_loss_weight=1.0`, target `mgiou`; code áp dụng quality coefficient trước weighting. Với cùng notation, hệ số hữu hiệu của task k/group g là $\tfrac12\bar\mu_g\omega_{gk}/\max(\mu_{gk},10^{-4})$; không cần thêm sáu λ cố định không có trong triển khai.

Điểm thiếu là **định nghĩa/reduction của task losses và lifecycle state**, T04: OGA dùng Smooth-L1 qua hai kênh/(2Npositive+1e-4), không phải L1/Npositive trong context. EMA khởi tạo bằng detached first-batch losses, chỉ update lúc train; val freeze. Q-OGA là biến thể khác, năm tasks, RDA-weighted regression sum2/Npositive, khác normalization2N của OGA, cùng soft-min corner; không được tái lập chỉ bằng positive CQFL formula (T05).

T-SBUW bảo toàn tổng weights bằng 6. Với logits bị chặn ±3 và τ=2, mỗi weight nằm trong giới hạn khoảng **0.05916 đến 4.80409** (giới hạn tiệm cận); đây không phải gradient norm bound hay proof convergence. Detector 753.569 + 12 criterion logits = 753.581 trainable parameters khi tính cả criterion, còn inference detector giữ 753.569. SimAM không thêm parameters nhưng vẫn thêm operations.

### 2.5 Đối chiếu Algorithm 1 theo thứ tự thực thi

| Bước | Paper vs code | Kết luận |
|---|---|---|
| ROI→raster→stem→3stages→neck→groupheads | Shapes khớp final | Hợp lệ, ngoại trừ operator description T01. |
| Head raw outputs→probabilities | Code sigmoid cls/IQA trước ranking | Paper nên ghi rõ raw logits vs probabilities. |
| Peak và joint score | Paper peak(cls)→joint;code joint→peak(joint) | **Không tương đương: T06.** |
| Threshold/decode | pcls>.05 và joint>.05,metric offsets,exp sizes,halfatan2 | Khớp contract; decode có numerical clamps/fallback cần kèm released implementation. |
| Rotated class-wise NMS→merge/sort | NMS.10 | Khớp; thêm global top500 vào pseudocode cho tự chứa. |
| Output | 7-value BEV footprint | Nên bỏ gọi “3D boxes” nếu không dự đoán z/h. |

Counterexample `[Script]`: hai cell kề nhau có cls=[.9,.8], q=[.04,.81] cho joint=[.18974,.80498]. Peak(cls) chọn cell 0, peak(joint) chọn cell 1; cả hai vượt thresholds. Vì vậy đây là sai khác có thể đổi detections, không chỉ thứ tự trình bày. Corner permutation cũng có witness: reverse-order π error≈2.22e-16 với order code, nhưng 2.0 với cyclic order trên box 4×2 m; cần explicit convention T03.

### 2.6 Động lực toán học và các giới hạn cần giữ khi diễn giải

| Cơ chế | Động lực hình học/kỹ thuật đã có | Phần còn thiếu / giới hạn lập luận |
|---|---|---|
| Strata-BEV14 | Moments/strata lưu vertical distribution; subcell centroid giảm quantization; deterministic encoder giảm learned-encoder cost | Log compression làm dịu dynamic range, **không bù đầy đủ r^-2**, không khôi phục point/shape đã mất; centroid channels không đảo ngược discretization. |
| Ground/PC-GTA | Ground-aligned placement,static/visibility giữ scene consistency | R(d) slope/radii và percentile5 là heuristic; cần giải thích reliability theo density/range và công bố thresholds M01. Không đòi theorem. |
| Inverse-square density | r^-2 hợp lý như approximation của angular sampling trên surfaces | Phụ thuộc angle/occlusion/shape; không đảm bảo mọi object có density đúng vật lý; near/far validation sẽ củng cố. |
| Radiometric η | Range-dependent intensity compensation có mô hình đơn giản | γ1.7 cần căn cứ calibration; nhánh tắt final nên không kết luận nó tăng AP. |
| Next block/SimAM | Local neighborhood và mean/variance saliency,zero added attention parameters | Attention hoạt động trong BEV feature C/H/W; chưa chứng minh riêng bảo toàn geometry3D hay ưu tiên vật xa. High-deviation cũng có thể là noise; cần qualitative/matched evidence cho claim mạnh. |
| SG-FPN gate | Fusion multi-resolution và factor2σ cho phép attenuation/amplification;zero-init cho gate1 | Gate không tự cung cấp guarantee chống interpolation artifact; hiện A7→A8 AP giảm. T01 làm rõ operator trước khi bàn hiệu quả. |
| Four-axis GIoU proxy | Projection onto predicted/GT axes cung cấp oriented interval overlap tín hiệu; corner term đối xứngπ hợp orientation BEV | Không phải rotated IoU chính xác: centered square sides2 và4 cho proxy.5 nhưng areaIoU.25. Paper gọi proxy là đúng; cần giữ ranh giới này khi giải thích IQA. |
| T-SBUW | Bounded logits tránh task weight collapse;EMA calibration giảm loss-scale imbalance | Cần init/freeze T04 và isolated ablation M04; tổngweightJ không chứng minh gradient norm conserved hay convergence theorem. |
| Q-OGA | Quality target và RDA có động lực ranking/range | Mô tả hiện che soft-min/RDA/normalization; T05. Không đồng nhất gain biến thể với một phép thay positive target. |

Đóng góp phù hợp với **thiết kế hệ thống BEV gọn và geometry-aware objective**. Không có cơ sở trong phạm vi đọc local này để tuyên bố thuật toán attention mới, formal convergence guarantee, hay novelty tuyệt đối so với toàn bộ literature. SimAM là module kế thừa; theoretical increment cần đặt ở integration/objective và bằng chứng có kiểm soát. Một bài engineering không bắt buộc có theorem nếu claim và kiểm chứng tương xứng.

### 2.7 Findings của Reviewer 1 theo ISSUE_SCHEMA

### T01 — [LLM] MAJOR — SG-FPN mô tả sai phép giảm kênh và trộn hai hệ chỉ số pyramid/stride
- **Vị trí:** [sections/methodology.tex:285](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:285).
- **Schema:** `comment_type: methodology`; `severity: moderate`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: t01`; `review_lane: notation_and_numeric_consistency`; `gate_blocker: false`.

**Trích nguyên văn (`quote`):**

> depthwise $3 \times 3$ refinement, BN, and SiLU produce $\mathbf{U}_4$ and $\mathbf{U}_3$

**Phân tích (`explanation`):** L5/L4 có 48 kênh, L3 có 32 kênh. Upsample P4 vẫn có 48 kênh; một depthwise convolution thuần túy không thực hiện phép 48→32 như mô tả. Code dùng DWConv48→48 cho U4 nhưng Conv2d thường 48→32 cho U3. Ngoài ra Eq. scale_gating dùng r∈{8,4} trong khi L3/L4/L5 được đặt theo pyramid level: L_r chưa được ánh xạ rõ, và L4 dễ bị hiểu vừa là lateral stride 8 vừa là lateral stride 4 trong công thức tổng quát. Phép cộng trong code đúng; mô tả chưa đủ để tái dựng đúng toán tử.

**Bằng chứng (`evidence_anchor`):** `paper/code_context.md:173–180`; `Code/detector/core/models/backbones/mobilepixornext.py:188–208,273–288`; `parameter_and_mac_audit.json`.

**Hướng sửa:** Ghi riêng hai phép upsample/refine, sử dụng chỉ số stride nhất quán, nêu align_corners=False. Giữ head input 32 theo resolved config; không đổi về 16 chỉ để khớp context cũ.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
% Report-only proposal; preserve the existing figure/equation labels when editing.
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
the stride-4 projection uses a standard convolution to reduce 48 channels to 32.
```

### T02 — [LLM] MAJOR — Tia che khuất dùng point có reflectance như vector hình học 3D
- **Vị trí:** [sections/methodology.tex:146](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:146).
- **Schema:** `comment_type: methodology`; `severity: moderate`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: t02`; `review_lane: notation_and_numeric_consistency`; `gate_blocker: false`.

**Trích nguyên văn (`quote`):**

> $\mathbf{r}(t) = t \cdot \mathbf{p}$

**Phân tích (`explanation`):** Point p_i đã được định nghĩa trong R4 với thành phần reflectance. Trong ray–box intersection và khoảng cách shadow, r(t)=t p và norm(p) không chỉ rõ phép lấy XYZ; nếu dùng tuple đã định nghĩa thì ray là R4 và norm trộn mét với intensity. Code thực tế dùng norm(points[:, :3]). Context cũng lặp lại cách viết r(t)=t p nên không thể coi context là căn cứ để giữ nguyên sự nhập nhằng chiều này. LOS/shadow tắt ở final run: đây là lỗi đặc tả nhánh tùy chọn, không chứng minh kết quả final sai.

**Bằng chứng (`evidence_anchor`):** `paper/sections/methodology.tex:19,146–155`; `paper/code_context.md:94,293–295`; `Code/detector/core/datasets/augmentor/hybrid_physics.py:49–55`.

**Hướng sửa:** Phân biệt point có thuộc tính với vector tọa độ XYZ. Chuẩn khoảng cách và ray chỉ tác động lên XYZ; intensity không bị nhân theo tham số ray.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
For an attributed point $\mathbf p=(x,y,z,\iota)$, define its geometric
projection $\mathbf p^{\mathrm{xyz}}=(x,y,z)^\top\in\mathbb R^3$.
The sensor ray and the distance behind the box exit are
\[
 \mathbf r_{\mathbf p}(t)=t\mathbf p^{\mathrm{xyz}},\qquad
 d_{\mathrm{behind}}=(1-t_{\mathrm{leave}})
       \lVert\mathbf p^{\mathrm{xyz}}\rVert_2.
\]
Shadow masking applies to ray--box hits with $d_{\mathrm{behind}}>0.05\,\mathrm m$.
Reflectance $\iota$ is an attribute and is excluded from these geometric operations.
```

### T03 — [LLM] MAJOR — Hoán vị corner 5−m cần quy ước thứ tự đỉnh tường minh
- **Vị trí:** [sections/methodology.tex:355](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:355).
- **Schema:** `comment_type: methodology`; `severity: moderate`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: t03`; `review_lane: notation_and_numeric_consistency`; `gate_blocker: false`.

**Trích nguyên văn (`quote`):**

> \mathbf{c}_{5-m}^*

**Phân tích (`explanation`):** Đảo thứ tự [4,3,2,1] biểu diễn quay π đúng với thứ tự code [++, +−, −+, −−], nhưng không đúng với thứ tự chạy quanh chu vi thông dụng [++, +−, −−, −+]. Bài chỉ nói bốn corners, chưa định nghĩa thứ tự trước Eq. corner_loss. Script witness cho thấy reverse-order error gần 0 với code order nhưng lớn với cyclic order. Công thức và code không sai nếu giữ đúng thứ tự; thiếu convention sẽ khiến người tái lập cài một loss khác.

**Bằng chứng (`evidence_anchor`):** `paper/code_context.md:320–322`; `Code/detector/core/losses/oriented_geometry_loss.py:22,44–62,81–103`; `mathematical_witnesses.json`.

**Hướng sửa:** Định nghĩa local axes, corners và permutation trước loss; hoặc chuyển cả công thức lẫn code sang thứ tự cyclic với phép dịch hai đỉnh. Không chỉ đổi một vế công thức.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
Let $\mathbf e_l=(\cos\psi,\sin\psi)^\top$ and
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
$\pi(m)=5-m$. The predicted and target corners use the same ordering.
```

### T04 — [LLM] MAJOR — Loss tổng có trọng số nhưng các loss cơ sở và normalization chưa đủ đặc tả
- **Vị trí:** [sections/methodology.tex:384](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:384).
- **Schema:** `comment_type: missing_information`; `severity: major`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: t04`; `review_lane: notation_and_numeric_consistency`; `gate_blocker: true`.

**Trích nguyên văn (`quote`):**

> \mathcal{T} = \{\mathrm{cls}, \mathrm{off}, \mathrm{size}, \mathrm{yaw}, \mathrm{geo}, \mathrm{qual}\}

**Phân tích (`explanation`):** Các trọng số không bị thiếu toàn bộ: group weights 0.5/0.5, beta_corner=1, T-SBUW b=3, tau=2, EMA=0.99 đã được nêu. Tuy nhiên Methods chưa viết đầy đủ classification focal, ba masked regression losses và phép giảm trên positive mask. Context mô tả L1/N_pos, còn code OGA dùng Smooth-L1 chia cho tổng mask đã expand qua 2 kênh (2N_pos+1e-4). BCE quality chỉ giảm trên reg_mask. EMA còn có first-batch initialization và chỉ update khi training. Hai người chỉ đọc paper/context có thể cài khác objective; không nên khắc phục bằng thêm các lambda tùy ý.

**Bằng chứng (`evidence_anchor`):** `paper/code_context.md:307–314,324–339`; `Code/detector/core/losses/strategies/oga.py:28–37,70–115`; `Code/detector/core/losses/l1_loss.py:10–14`; `Code/detector/core/losses/uncertainty_weighting.py:108–121`; `Code/detector/core/losses/focal_loss.py:14–39`.

**Hướng sửa:** Bổ sung loss primitives, mask/reduction, trạng thái EMA và hệ số quality=1. Đối chiếu context để sửa chỗ L1 thành Smooth-L1 khi dùng OGA hiện tại. Ghi rõ weights là adaptive chứ không phải sáu hằng lambda cố định.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
For group $g$, let $\Omega_g$ be its regression-positive cells,
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
The fixed corner coefficient and quality coefficient are both 1.0.
```

### T05 — [LLM] MAJOR — Q-OGA không chỉ thay positive classification target như paper đang mô tả
- **Vị trí:** [sections/methodology.tex:371](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:371).
- **Schema:** `comment_type: missing_information`; `severity: major`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: t05`; `review_lane: notation_and_numeric_consistency`; `gate_blocker: true`.

**Trích nguyên văn (`quote`):**

> Q-OGA modulates positive classification targets

**Phân tích (`explanation`):** Context mô tả thêm soft-min π-corner và range-adaptive regression weighting. Code q_oga thực hiện cả hai, trong khi Methods chỉ cho positive CQFL term. Negative Gaussian-neighbor term, normalization, J=5 không có IQA và range weights/soft-min temperature chưa được khai báo cho nhánh này. Vì vậy lợi ích Q-OGA không thể hiểu là chỉ bỏ IQA head. Có thêm một sai khác hiện tại trong code: grid_pos dùng +0.5 cell rồi cộng target offset theo grid origin, nên RDA dùng vị trí lệch +0.2 m mỗi trục; cần xác minh chủ đích trước khi ghi một công thức true-center như đã khớp triển khai.

**Bằng chứng (`evidence_anchor`):** `paper/code_context.md:341–348`; `Code/detector/core/losses/quality_focal_loss.py:38–62`; `Code/detector/core/losses/strategies/q_oga.py:23,35–44,174–219`; `Code/detector/core/losses/smooth_corner_loss.py:29–58`; `mathematical_witnesses.json`.

**Hướng sửa:** Mô tả Q-OGA là biến thể objective đầy đủ, khai báo soft-min/RDA và negative CQFL. Nếu chỉ muốn so sánh head-free quality coupling, chạy thêm variant chỉ đổi coupling. Kiểm tra và sửa/giải thích offset origin của RDA ở lượt code riêng; không âm thầm viết lại code trong audit.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
% Complete objective description; the range-coordinate convention must first
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
% Do not publish a true-center formula without reconciling that difference.
```

### T06 — [LLM] MAJOR — Algorithm chọn peak trên classification nhưng code chọn peak trên joint score
- **Vị trí:** [sections/methodology.tex:428](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:428).
- **Schema:** `comment_type: methodology`; `severity: major`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: t06`; `review_lane: notation_and_numeric_consistency`; `gate_blocker: true`.

**Trích nguyên văn (`quote`):**

> Identify local $3 \times 3$ candidate peaks on classification heatmap

**Phân tích (`explanation`):** Algorithm bước 8→9 và prose cuối Methods chọn local maxima của p_cls trước khi kết hợp q. Code tính ranking_scores trước rồi max_pool2d trên ranking_scores. Hai thứ tự không giao hoán: với hai cell kề nhau p=[0.9,0.8], q=[0.04,0.81], classification chọn cell 0, joint score=[0.1897,0.8050] chọn cell 1; cả hai cell đều vượt ngưỡng 0.05. Context cũng ghi thứ tự cũ như paper nên phải đối chiếu trực tiếp postprocess.py.

**Bằng chứng (`evidence_anchor`):** `paper/code_context.md:129–139`; `Code/detector/postprocess.py:175–178,223–229`; `mathematical_witnesses.json`.

**Hướng sửa:** Tính joint score rồi lấy peak từng class trong group; giữ hai điều kiện threshold, decode và class-wise NMS. Thêm global sort/top-500 vào pseudocode để tự chứa; sửa cả câu prose ở dòng 441.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
% Replacement for peak/scoring/filtering steps, matching postprocess.py.
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
predict the height and vertical position required for full 3D boxes.
```

### T07 — [LLM] MINOR — F7 là population standard deviation, không phải sample standard deviation
- **Vị trí:** [sections/methodology.tex:69](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:69).
- **Schema:** `comment_type: presentation`; `severity: minor`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: t07`; `review_lane: notation_and_numeric_consistency`; `gate_blocker: false`.

**Trích nguyên văn (`quote`):**

> and the sample standard deviation:

**Phân tích (`explanation`):** Eq. z_std_encoding dùng 1/N nên là độ lệch chuẩn của tập điểm trong cell (population moment), trong khi chữ sample thường chỉ estimator chia N−1. Context và rasterizer cùng dùng population moment. Với normalized heights [0,1], công thức cho 0.5 còn sample standard deviation là khoảng 0.7071. Không cần đổi công thức/code để sửa một tên gọi.

**Bằng chứng (`evidence_anchor`):** `paper/code_context.md:253–254`; `Code/detector/core/datasets/utils_1/bev_backend.py:168–176`; `mathematical_witnesses.json`.

**Hướng sửa:** Đổi sample standard deviation ở prose và caption thành within-cell population standard deviation hoặc vertical dispersion; giữ phép tính hiện tại.

### T08 — [LLM] MINOR — Đối chứng stage-level SimAM được mô tả có linear projections nhưng code chỉ thêm scalar beta
- **Vị trí:** [sections/methodology.tex:261](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:261).
- **Schema:** `comment_type: claim_accuracy`; `severity: minor`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: t08`; `review_lane: notation_and_numeric_consistency`; `gate_blocker: false`.

**Trích nguyên văn (`quote`):**

> stage-level adapters add learned linear projections

**Phân tích (`explanation`):** ResidualLightAttention hiện có một beta scalar và một SimAMCore không tham số; không có learned channel-linear projection. Điều này không phủ định Block-SimAM có zero parameters, nhưng lý do so sánh chi phí adapter cần mô tả đúng. Nếu bảng attention dùng một implementation lịch sử khác, phải khóa revision thay vì gán code hiện tại cho bảng đó.

**Bằng chứng (`evidence_anchor`):** `Code/detector/core/models/backbones/light_attention.py:40–55,73–74`.

**Hướng sửa:** Mô tả adapter hiện tại là X+beta(SimAM(X)−X) với một scalar trainable; hoặc dẫn đúng revision của adapter được dùng trong các run lịch sử.

## 3. Reviewer 2 — Methodology Transparency & Experimental Reproducibility

| Tiêu chí | Điểm /5 | Căn cứ |
|---|---:|---|
| Reproducibility checklist | 3 | Geometry/optimizer/augmentation chính rõ; recipe/ownership/training GPU còn thiếu. |
| Ablation và kiểm soát thực nghiệm | 2 | Có nhiều rows và factorial loss; thiếu isolated weighting/geometry và manifests cho SimAM. |
| Complexity, memory và latency | 3 | Params/memory/pipeline timing tốt; standalone provenance và CPU/training hardware chưa đủ; thiếu MAC count. |

### 3.1 Checklist đối chiếu Experimental Setup

Trong bảng này, L là dòng trong `sections/experiments.tex`, trừ khi ghi Methods hoặc file khác.

| Hạng mục | Giá trị thực final / bằng chứng | Paper đã khai báo? |
|---|---|---|
| Split | 3712/3769; script check ID overlap0 | Có,L5; không thay cho audit toàn bộ contents GT database. |
| Range | x[0,70.4],y[−40,40],z[−2.5,1],inwardε.001m | Có,L16; khớp final. |
| Grid / encoding | .1m XY,800×704,14ch,output stride4,200×176 | Có,L16 và Methods; không learned pillar encoder. |
| Channels/backbone | 3/4/2blocks,width48/96/128,F32,H32 | Có,L43; context8blocks/H16 cũ. |
| Eval IoU | Car.7,Ped/Cyc.5,BEV,R40,difficulty criteria | Có,L19–37; equivalence reference chưa established M05. |
| Decode | score.05,NMS.1,alpha.5,cap500 | Có Methods; peak ordering sai T06. |
| Gaussian/assignment | overlap.1,minradius4,sigma(2r+1)/6,regRadiusr/2,nearest_center | Minradius/regRadius có; ownership/overlap thiếu M02. |
| GT sampling | p.5,quota15/10/10,source-relative,.9–1.1,±5° | Probability có; quotas/placement/jitter thiếu M01. |
| Physics | flags ground/static/density/visibility on,LOS/shadow/radio off | Toggles có,L49; visibility runtime guard không active ở final; threshold/guard recipe thiếu M01. |
| Global augmentation | flipx p.5,rot±π/4,scale.95–1.05,translation p.5,std(.2,.2,.1)m | Có,L49; khớp resolved config. |
| Optimizer | AdamW,wd.001,norm/bias wd0,clip10 | Có,L46; khớp. |
| LR schedule | base7e-4,min1e-6,cosine;final160epochs,warmup10 | Có,L46; standard50/warmup4 cũng có, cần mapping từng ablation M03. |
| Batch/workers | train16,val32,accum1,bf16,workers6,pin/persistent,prefetch2 | Batch/precision/workers có,L52; accum1 nằm config, không lỗ hổng chính. |
| Checkpoint selection | AP mỗi10epochs,maximum val Moderate mAP,selected130 | Có core policy; cần hash/config từng row M03. |
| Seed/repeats | seed42,single run; variance deferred | Có,L52; không tự coi single-seed là sai nếu không tuyên bố significance. |
| Training GPU | Chưa tìm thấy identity riêng cho training final | Thiếu M06; không suy diễn từ evalT4/bf16. |
| Evaluation GPU/software | TeslaT4,FP32,batch1,PyTorch2.11/CUDA13 | Có,L52/results; khớp canonical eval log. |
| Host CPU / threads | x86_64 được nêu,CPU model/thread budget chưa đủ | Thiếu để tái lập preprocess timing M06. |
| Loss coefficients/state | group.5/.5,beta_corner1,quality1,b3,tau2,EMA.99 | Phần lớn có; primitive reduction/EMA lifecycle thiếu T04. |
| Q-OGA alternative | beta1,gamma2,alphaGaussian4,softmin.05,RDA1.5/2/70.4,J5 | Methods mới có positive CQFL subset; T05. |

### 3.2 Module–ablation matrix

| Đóng góp / nhánh | Hàng hiện có | Có isolate riêng? / giới hạn |
|---|---|---|
| Next-style backbone | components A0→A1 | Table khai báo giữ input/head/neck/sampling; cần resolved manifest để xác minh ngoài khai báo. |
| Strata-BEV14 | A1/A2/A3;encoding binary35/rich8/hist14 | Có encoding-level control; không bắt buộc ablate từng14channels cho paper system-level. |
| Grouped head / H32 | A3→A4→A5 | Có riêng grouping/width; recipe từng row chưa khóa. |
| PC-GTA | A5→A6,+6.03points | Có package-level; ground/static/density chưa tách từng branch, visibility chỉ bật flag nhưng guard final bypass; nên giới hạn claim vào package. |
| LOS/shadow/radiometric | Formulas Methodology nhưng final tắt | Không gán gain final; không yêu cầu kết quả positive cho nhánh không dùng. |
| Fusion width32 | A6→A7 | Có progression; cần cùng budget. |
| SG-FPN | A7→A8,68.12→67.38 | Có riêng row,không cho thấy AP gain; lợi ích artifact suppression chưa đo riêng. |
| OGA và IQA | loss factorial 4rows:baseline/OGA ×noIQA/IQA;A8→A9 là combined | Có tách package OGA/IQA; không thể chỉ gọi A8→A9 là gain geometry đơn thuần. |
| Projection GIoU / π-corner | Chỉ composite OGA | **Chưa có projection-only/corner-only**: M04. |
| T-SBUW / EMA / bounded weighting | Loss curves,không riêng row | **Chưa có uniform/EMA-only/full controls**: M04. |
| Block-SimAM / placements | attention none/stage2/stage3/stage4/block;A9→A10 | Có row,nhưng .721M→.754M dù SimAM zero-param; depth/schedule matching chưa xác minh: M03. |
| Q-OGA head-free | loss Q-OGA row | Có package-level; thay nhiều objective mechanisms T05 nên không isolate quality coupling alone. |

Tất cả được mô tả trên KITTI validation cùng split, nhưng **không đủ dữ liệu để xác nhận mọi row có cùng architecture và điều kiện huấn luyện**. Các row progression chủ đích đổi module; điều cần giữ là mọi biến còn lại. Cụ thể A9→A10 cần cùng depth,head,160/50-epoch budget,seed và checkpoint policy. Chưa có manifest A9 nên không kết luận chắc chắn biến nào đã thay đổi. Không tính package-level evidence thành chứng minh riêng từng nhánh.

### 3.3 Những gì chưa thể suy từ bảng ablation

Loss table no-attention `.721M` và final `.753569M` không thể khác chỉ do parameter-free SimAM. Counter giữ đúng final architecture cho cả off/on đều ra753569; bỏ hai IQAheads còn734943,delta18626. Đây là yêu cầu truy nguyên cấu hình lịch sử, chưa phải bằng chứng A9 đã dùng chính xác2/4/2blocks. Cùng split không thay cho cùng schedule/selection; các50-epoch curves không chứng minh mọiablation theo budget final160epochs.

### 3.4 Chi phí tính toán và latency: bằng chứng thực có

| Đại lượng | Paper / log final | Kết quả audit |
|---|---|---|
| Params | 753569≈.754M | `[Script]` tính từ các lớp khớp:body604864+neck36784+heads111921. |
| Checkpoint | 9338383bytes≈9.34MB | File checkpoint,không đồng nhất số byte raw weights hay peak GPU memory. |
| Peak eval allocation | 98.43945 theo field `torch_peak_memory_mb` | Có; code chia1024² nên giá trị theo MiB; nên thống nhất nhãn, không coi là blocker. Không phải training memory. |
| Standalone forward | 8.67ms,115.32FPS | Paper có; thiếu corresponding raw timing metadata; T4/L4 provenance M06. |
| In-pipeline model | 17.87519ms,55.94347FPS | CUDA-event timing có sync; không phải backbone-only và không phải standalone8.67ms. |
| CPU preprocess | 20.93910ms | Bao gồm file/data loading+raster preprocessing; không riêng kernel raster. |
| H2D / decode-NMS | 6.91278 /14.94373ms | Chi phí đáng kể; không suy E2E từ standalone FPS. |
| E2E | 60.95632ms,16.40519FPS | Sequential bin→detections,FP32,batch1,T4. Có3759frames sau10warmup. |
| E2E percentiles | p50=58.27517,p95=80.85136,p99=95.91263ms | Có tail-latency evidence; chưa suy thành guarantee mọi frame≤100ms. |
| FLOPs/MACs | Paper chưa báo cáo | Reviewer static count:11.4495744 **G Conv2d MAC/frame**,22.8991488GFLOPs theo2FLOP/MAC. Không phải profiler/model execution. |

Phép đếm MAC lấy batch 1, input 14×800×704, architecture 3/4/2, F32, shared H32, quality heads on. Tổng Conv2d MAC không gồm BN, SiLU, SimAM, bilinear interpolation, gating elementwise, raster, decode, NMS; không dùng số này để suy FPS. M07 là bổ sung MINOR vì bài **đã báo cáo params, memory, FPS và ms**. M06 là MAJOR vì naming/hardware protocol hiện làm suy yếu khả năng tái lập so sánh tốc độ. Không có torch trong môi trường audit nên không chạy forward/profile hay retrain; không khẳng định throughput đã được xác minh thực nghiệm lại.

### 3.5 Findings của Reviewer 2 theo ISSUE_SCHEMA

### M01 — [LLM] MAJOR — PC-GTA còn thiếu recipe và ngưỡng quyết định đủ để tái lập
- **Vị trí:** [sections/experiments.tex:49](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:49).
- **Schema:** `comment_type: missing_information`; `severity: major`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: m01`; `review_lane: evaluation_fairness_and_reproducibility`; `gate_blocker: false`.

**Trích nguyên văn (`quote`):**

> GT sampling was applied with probability $0.5$

**Phân tích (`explanation`):** Paper đã nêu bật/tắt physics, xác suất GT/global transforms và ground-radius. Tuy vậy N_thresh không được gán giá trị; thiếu quota mỗi class, source-relative placement, jitter, collision margin, visibility survival/minimum points và quy tắc hỗ trợ ground. LIMIT_WHOLE_SCENE=true trừ số object hiện có khỏi quota 15/10/10. Cần phân biệt visibility flag=true với guard thực thi: protect=flag AND (shadow OR extra_width>0 OR margin<=1e-8); final shadow=false, extra_width=0, margin=0.1 nên không chạy kiểm tra visibility riêng. Các biến này thay đổi cảnh training và là một phần của module có gain +6.03. Resolved config và sampler cho phép khôi phục cụ thể; master config mặc định không phải recipe final.

**Bằng chứng (`evidence_anchor`):** `paper/sections/methodology.tex:142`; `Result/iqa_comparison_20261019/config.inference.json:augmentation`; `Code/detector/core/datasets/augmentor/hybrid_sampler.py:89–106,153–178,246–247`; `Code/detector/core/datasets/augmentor/hybrid_physics.py:12–25,64–68`.

**Hướng sửa:** Công bố resolved augmentation dictionary hoặc bảng recipe và version sampler. Phân biệt cơ chế đã dùng với LOS/shadow/radiometric tắt; không gán gain package cho các nhánh bị tắt.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
PC-GTA uses target scene quotas of 15 Cars, 10 Pedestrians and 10 Cyclists,
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
radiometric calibration are disabled in the reported final run.
```

### M02 — [LLM] MAJOR — Target assignment chưa công bố ownership cho regression head dùng chung
- **Vị trí:** [sections/methodology.tex:325](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:325).
- **Schema:** `comment_type: missing_information`; `severity: moderate`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: m02`; `review_lane: evaluation_fairness_and_reproducibility`; `gate_blocker: false`.

**Trích nguyên văn (`quote`):**

> Regression targets were assigned within half the Gaussian peak radius

**Phân tích (`explanation`):** Paper cho radius nhưng chưa ghi gaussian_overlap=0.1, Gaussian sigma, quantized peak và nearest_center ownership khi nhiều GT dùng cùng cell. Pedestrian/Cyclist dùng chung offset/size/yaw nên một cell chỉ giữ được một GT regression target. Code dự trữ quantized peak cho owner gần nhất, rồi gán các cell khác theo nearest center; exact ties giữ GT đến trước. Cách diễn giải anchor center trong Fig. 6 caption cũng dễ gây thêm +0.2 m vì Eq. gt_targets/code dùng grid origin không có +0.5.

**Bằng chứng (`evidence_anchor`):** `Result/iqa_comparison_20261019/config.inference.json:data.regression_assignment`; `Code/detector/core/datasets/dataset.py:434–446`; `Code/detector/core/datasets/utils_1/gaussian.py:37–39`; `Code/detector/core/datasets/utils_1/target_backend.py:29–77`; `paper/sections/methodology.tex:301,321`.

**Hướng sửa:** Nêu nearest_center và tie/peak reservation; phân biệt raster-cell centroid offset với regression anchor-grid offset. Không thêm +0.5 vào decode vì đó sẽ thay đổi code contract.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
The heatmap radius uses a minimum overlap of 0.1 and is bounded below
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
$F_{12}$ and $F_{13}$; no half-output-cell term is added at decoding.
```

### M03 — [LLM] MAJOR — Ablation chưa chứng minh kiểm soát architecture, schedule và checkpoint policy
- **Vị trí:** [tables/loss.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/loss.tex:19).
- **Schema:** `comment_type: methodology`; `severity: major`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: m03`; `review_lane: evaluation_fairness_and_reproducibility`; `gate_blocker: true`.

**Trích nguyên văn (`quote`):**

> All variants share identical MobilePIXORNeXt backbone

**Phân tích (`explanation`):** Table loss ghi OGA+IQA/no-attention 0.721M và AP 70.58; final là 0.753569M với gain +0.88, mặc dù chính SimAMCore không thêm parameter. Với đúng final shared H32/F32 topology, script đếm cả có/không SimAM đều 753569; hai IQA heads chiếm 18626 tham số, khác mô tả 18.59K. Bài cũng tách 50-epoch standard/curves khỏi final 160-epoch run chọn epoch 130 nhưng không cung cấp run manifest từng hàng ablation. Chưa thể tách gain attention khỏi thay đổi depth/head/schedule; không có log A9 đủ để kết luận chính xác biến nào đã thay đổi. Cùng split và seed 42 không đủ để khẳng định controlled ablation.

**Bằng chứng (`evidence_anchor`):** `paper/tables/loss.tex:12–19`; `paper/tables/components.tex:40–45`; `paper/sections/experiments.tex:43,46`; `paper/sections/results.tex:139`; `parameter_and_mac_audit.json`; `Result/benchmark_T4_20261018/sources/ours_best_ap.json:model.parameter_counts`.

**Hướng sửa:** Xuất mỗi run's resolved config/hash, params, depth, head architecture, optimizer/schedule, selection epoch/policy và GPU. Chạy cặp final architecture attention off/on với cùng training budget hoặc giới hạn Table components là progression. Xác minh param delta của IQA thay vì sửa số làm tròn theo phỏng đoán.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
% Safe description until run manifests and matched controls are available.
Table~\ref{tab:components} records an architectural progression.
The final checkpoint was selected at epoch 130 from a 160-epoch schedule;
the illustrated training curves cover 50 epochs.
The currently reported rows do not establish an architecture- and
schedule-matched estimate of the isolated Block-SimAM effect.
Such an estimate requires identical stage depths, shared-head topology,
training budget and checkpoint selection, with only Block-SimAM toggled.
% For the verified final topology only: both attention settings have 753,569
% detector parameters; removing both IQA branches removes 18,626 parameters.
% Do not replace historical row counts until their actual configs are recovered.
```

### M04 — [LLM] MAJOR — T-SBUW và các thành phần OGA chưa có kiểm chứng tách riêng
- **Vị trí:** [sections/experiments.tex:46](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:46).
- **Schema:** `comment_type: missing_information`; `severity: major`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: m04`; `review_lane: evaluation_fairness_and_reproducibility`; `gate_blocker: false`.

**Trích nguyên văn (`quote`):**

> evaluates its empirical convergence behavior

**Phân tích (`explanation`):** A0–A10 có backbone, encoding, grouped width, GT, fusion, SG-FPN, OGA+IQA, attention; factorial loss có OGA và IQA riêng. Nhưng không có hàng bật/tắt T-SBUW hoặc tách bounded softmax khỏi EMA; cũng chưa tách projection GIoU khỏi π-corner. Loss curves của các objective khác nhau không đo riêng lợi ích của weighting và không so được trực tiếp khi normalization khác. PC-GTA được kiểm chứng ở mức package; các nhánh physics tắt không cần một hàng hiệu quả được gán cho final run.

**Bằng chứng (`evidence_anchor`):** `paper/sections/methodology.tex:360,383–401`; `paper/tables/components.tex:11–21`; `paper/tables/loss.tex:10–17`; `paper/sections/results.tex:139–146`.

**Hướng sửa:** Bổ sung matched ablation uniform weighting / EMA-only / T-SBUW, và projection-only / corner-only / combined OGA nếu muốn quy kết contribution của từng cơ chế. Nếu chưa có các run này, giới hạn kết luận vào hệ thống kết hợp; không bịa AP mới.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
The loss-factorial study evaluates the composite OGA objective and IQA.
It does not separate the effects of projection GIoU and the symmetric corner
term. No isolated ablation of EMA calibration and bounded task weighting is
reported. The loss trajectories therefore describe convergence of the full
objectives, rather than establish the independent benefit of T-SBUW.
% Proposed additional controls, with AP to be filled only from real runs:
% uniform task weights; EMA-only calibration; full T-SBUW;
% projection-only geometry; corner-only geometry; combined OGA.
```

### M05 — [LLM] MAJOR — Cùng R40/split chưa đủ chứng minh tương đương evaluator cho so sánh AP
- **Vị trí:** [sections/experiments.tex:37](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:37).
- **Schema:** `comment_type: methodology`; `severity: major`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: e03-headline`; `review_lane: evaluation_fairness_and_reproducibility`; `gate_blocker: true`.
- **Liên hệ lần trước:** Phase 1 E03; đây là phân tích sâu tiếp nối cùng root cause, không tính là lỗi mới độc lập của Phase 1.

**Trích nguyên văn (`quote`):**

> Both implementations used $R_{40}$ interpolation on the same validation split

**Phân tích (`explanation`):** Local evaluator thực hiện matching/ranking riêng trên BEV outputs và quy tắc ignore/ROI riêng; log nói rõ không phải official hidden test. Table comparison phân biệt local và reference, là công khai tốt, nhưng chưa có equivalence check để dùng chênh lệch +1.33 hoặc matched SECOND như bằng chứng chắc chắn hơn về detector. Cần tái hiện cùng evaluator hoặc đối chiếu semantics của ignore/matching/range/candidate filtering. Không khẳng định một quy tắc cụ thể như DontCare đã khác khi chưa thẩm định reference implementation tương ứng.

**Bằng chứng (`evidence_anchor`):** `Code/tools/kitti_training_pipeline/evaluate_kitti_bev.py:164–224`; `Result/benchmark_T4_20261018/sources/ours_best_ap.json:protocol`; `paper/tables/comparison.tex:23`.

**Hướng sửa:** Giữ công khai hai protocol; đánh giá matched protocol trên cùng predictions/GT nếu khả thi và trình bày mức chênh theo từng class. Cho tới lúc đó coi bảng cross-family là descriptive comparison, không test tương đương/thống kê thắng thua.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
MobilePIXORNeXt and historical MobilePIXOR use the local loader-aligned
BEV evaluator; reference detectors use the reference KITTI evaluator.
Although both report $R_{40}$ on the same validation split, evaluator
equivalence has not been established here. Cross-family AP differences are
therefore descriptive. A common-evaluator comparison is required before
attributing small differences solely to detector quality.
```

### M06 — [LLM] MAJOR — Hardware/protocol của standalone timing và chi phí preprocessing chưa đủ tái lập
- **Vị trí:** [sections/methodology.tex:466](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:466).
- **Schema:** `comment_type: methodology`; `severity: major`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: e07-nguồn`; `review_lane: evaluation_fairness_and_reproducibility`; `gate_blocker: true`.
- **Liên hệ lần trước:** Phase 1 E07; đây là phân tích sâu tiếp nối cùng root cause, không tính là lỗi mới độc lập của Phase 1.

**Trích nguyên văn (`quote`):**

> standalone network throughput reached $55.94$\,FPS

**Phân tích (`explanation`):** Methods cuối gọi 55.94 FPS là standalone, nhưng Results/Table components dùng standalone=115.32 FPS và pipeline-forward=55.94. Code device_timed đo CUDA events và sync; E2E wall-clock có I/O/preprocess. Context/table encoding ghi L4 cho các số 169.97/165.02/169.01, Table components lại gán T4, đúng vấn đề Phase 1 E07. Chưa có log standalone tương ứng và host CPU/threading/loop protocol đầy đủ; chỉ x86_64 không tái lập được 20.94 ms CPU preprocessing. Training GPU cũng chưa được định danh riêng, không thể suy từ T4 evaluation và bf16 training rằng train trên T4.

**Bằng chứng (`evidence_anchor`):** `paper/sections/results.tex:53`; `paper/tables/components.tex:3,32–34,45`; `paper/tables/encoding.tex:3,17`; `paper/code_context.md:501`; `Code/tools/kitti_training_pipeline/evaluate_kitti_bev.py:276–288,554–588`; `Result/benchmark_T4_20261018/sources/ours_best_ap.json:runtime,latency`.

**Hướng sửa:** Đổi tên 55.94 thành in-pipeline model forward; khóa GPU và timer cho từng regime. Ghi training GPU, CPU model, threads/workers, warmup/iterations, precision/batch/caching và synchronization. Chỉ gán T4 cho standalone sau xác minh log; giữ đầu vào .bin→detections của E2E riêng khỏi network-only.

**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**

```latex
The sequential evaluation pipeline uses batch size one and FP32 on
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
% reconcile T4/L4 provenance for every reused ablation timing row.
```

### M07 — [LLM] MINOR — Compute coverage còn thiếu FLOPs/MACs dù đã có params, memory và timing
- **Vị trí:** [sections/experiments.tex:52](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:52).
- **Schema:** `comment_type: missing_information`; `severity: minor`; `source_kind: llm`; `confidence: high`; `quote_verified: true`.
- **Theo dõi:** `root_cause_key: m07`; `review_lane: evaluation_fairness_and_reproducibility`; `gate_blocker: false`.

**Trích nguyên văn (`quote`):**

> Peak allocated GPU memory was $98.44\,\text{MB}$

**Phân tích (`explanation`):** Bài không thiếu toàn bộ efficiency evidence: có Params, checkpoint size, peak evaluation GPU memory, forward/E2E FPS và latency percentiles. Tuy nhiên chưa có FLOPs/MACs hoặc scope đếm operations. Script source-based tính khoảng 11.4496 G Conv2d MACs cho batch=1, input14×800×704; đây chỉ là phép tính giải tích của reviewer, không phải profiler chạy model, không bao gồm SimAM/BN/activation/interpolation/decode/NMS. Không được trình bày số này như số đo tác giả đã báo cáo hoặc dùng để suy ra FPS.

**Bằng chứng (`evidence_anchor`):** `paper/sections/methodology.tex:458–466`; `paper/tables/efficiency.tex`; `Code/tools/benchmarks/profile_detector.py:98–138`; `parameter_and_mac_audit.json`.

**Hướng sửa:** Bổ sung compute count có profiler/version, batch/input shape, quy ước FLOP/MAC và danh sách operators bỏ qua. Tách peak inference allocation khỏi training memory. Thiếu MACs hiện là bổ sung nên làm, không tự nó là submission blocker.

## 4. Thứ tự xử lý và tiêu chí đóng lỗi

1. **Khóa bản triển khai final.** Gắn checkpoint/config/source revision; refresh context từ 8→9 blocks, H16→H32 và CPU raster. Đây là đồng bộ tài liệu, không đổi final model theo default config.
2. **Đặc tả lại giao diện toán học.** Đóng T01–T06 và M01–M02: đúng SG-FPN 48→32, XYZ ray, corner order, full OGA/Q-OGA objectives, EMA lifecycle, assignment ownership và joint-score peak. Một người đọc paper+released config phải dựng lại được các bước và reductions.
3. **Khôi phục provenance từng ablation.** Đóng M03 bằng manifests và pair SimAM off/on thật sự cùng topology/budget/selection; nếu chưa có, ghi table là progression và giới hạn causal claim. Đóng M04 bằng controls thật hoặc thu hẹp kết luận về contribution riêng T-SBUW/projection/corner.
4. **Củng cố so sánh AP/tốc độ.** Đóng M05 bằng common-evaluator validation hoặc giới hạn rõ cross-protocol AP; đóng M06 bằng GPU/timer/host/training metadata và sửa tên latencyregimes. Không ghép số từ các thiết bị/phép đo khác nhau thành benchmark thống nhất.
5. **Hoàn thiện nhỏ.** T07 populationstd,T08scalaradapter,M07MAC count có scope/profiler. Các snippet có comment “author verification” chỉ là mẫu an toàn; xóa TODO mà không bổ sung bằng chứng không đóng lỗi.

Phase 2 không thay thế verdict screening trước đó và không chứng nhận journal acceptance. M05/M06 tiếp nối E03/E07 trong Phase 1 theo cùng root cause; không cộng gộp 15 findings này với 12 findings cũ như 27 lỗi độc lập.

## 5. Dấu vết kiểm chứng và giới hạn

Các artifact ở [artifacts/phase2/](artifacts/phase2/) gồm:

- [issues.json](artifacts/phase2/issues.json): 15 records đúng canonical schema, 15/15 quotes được skill `verify_quotes` tìm thấy trong assembled manuscript và được kiểm tra lại đúng file/dòng; 12/12 MAJOR có mẫu LaTeX.
- [scores.json](artifacts/phase2/scores.json): 9 criterion scores của hai vai reviewer.
- [equation_inventory.json](artifacts/phase2/equation_inventory.json): 26 numbered blocks và 337 inline spans; bảng ký hiệu phía trên do reviewer đọc và phân loại, không phải parser tự chứng minh mọi nghĩa.
- [numerical_checks.py](artifacts/phase2/numerical_checks.py), [parameter_and_mac_audit.json](artifacts/phase2/parameter_and_mac_audit.json), [mathematical_witnesses.json](artifacts/phase2/mathematical_witnesses.json):static parameter/MAC arithmetic và witnesses peak/corner/std/proxy/RDA/weight bounds.
- [dataset_split_check.json](artifacts/phase2/dataset_split_check.json): 3712 train/3769 val, intersection 0; không audit lại mọi raw training sample hay chứng minh mọi database entry train-only bằng chạy training.
- [resolved_evidence.json](artifacts/phase2/resolved_evidence.json): cấu hình/log đã chọn để đối chiếu final, bao gồm checkpoint SHA256.
- [source_integrity.json](artifacts/phase2/source_integrity.json): 70 source files được so hash trước/sau, 0 file thay đổi trong lượt audit.
- [verification.json](artifacts/phase2/verification.json):schema,quotes,source-line checks,snippet coverage,hash integrity và equation coverage.

Có thể tái chạy phép tính và dựng báo cáo từ repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B paper/audit/artifacts/phase2/numerical_checks.py
PYTHONDONTWRITEBYTECODE=1 python3 -B paper/audit/artifacts/phase2/build_review.py
```

Không chạy lại training/accuracy/timing; không xác minh external prior art hay reference-evaluator equivalence. Các chỗ thiếu raw run manifests được ghi là **chưa xác minh**, không đổi thành cáo buộc fabrication hay kết luận ablation chắc chắn dùng sai điều kiện.
