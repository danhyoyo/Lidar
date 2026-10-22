"""Human-reviewed coverage tables for the Phase 2 report."""

SYMBOLS = r"""
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
"""

EQUATIONS = r"""
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
"""

TENSORS = r"""
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
"""

REPRO = r"""
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
"""

ABLATIONS = r"""
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
"""

MOTIVATION = r"""
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
"""

CONTEXT = r"""
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
"""
