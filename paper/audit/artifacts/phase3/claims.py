"""Claim-level judgments, with original wording preserved as exact substrings."""
CLAIMS=[]
FILES={'A':'sections/abstract.tex','I':'sections/introduction.tex','M':'sections/methodology.tex',
       'C':'sections/conclusion.tex','R':'sections/results.tex','E':'sections/experiments.tex',
       'W':'sections/related_work.tex','T':'tables/encoding.tex'}


def add(src,quote,evidence,context,status,fix,also='',root=None,orphan=False):
    number=len(CLAIMS)+1
    CLAIMS.append({'id':f'C{number:02}','source_file':FILES[src], 'quote':quote,
       'paper_evidence':evidence,'context_evidence':context,'status':status,'allowed_wording':fix,
       'related_locations':also,'root_cause_key':root or f'claim-{number:02}',
       'orphan_contribution':orphan,'critical_blocker':orphan,
       'claim_strength':'strong' if status=='VERIFIED' else 'observed' if status=='METRIC_MISMATCH' else 'unsupported',
       'source_kind':'llm','comment_type':'claim_accuracy'})


add('A','a compact, proposal-free Bird\'s-Eye View (BEV) detector with 753,569 parameters.',
    'tab:comparison .754M; Methods L459–464:604864+36784+111921=753569; log confirms.',
    'L23,229–236:753569 detector params +12 training-only logits.', 'VERIFIED',
    'Giữ 753,569 detector parameters; phân biệt criterion training-only 12 logits.', 'Intro L11,15,19; Conclusion L7')
add('C',r'($9.34\,\text{MB}$ checkpoint)',
    'tab:setup và Methods L459; canonical log:9338383bytes≈9.34MB.',
    'L23 và §3.3:9.34MB,753569params.', 'VERIFIED',
    'Giữ checkpoint file size 9.34MB; không gọi là raw FP32 weights hoặc training memory.', 'Intro L19; Experiments L43')
add('A','a parameter-free deterministic NumPy/Numba CPU rasterizer.',
    'tab:strata_bev14 caption/Methods L96,122,tab:setup:CPU NumPy/Numba.',
    'L59 nói parallel CUDA, trái CPU; §4.1 có14 deterministic formulas.', 'METRIC_MISMATCH',
    'Giữ CPU theo implementation final; sửa context L59. Đây là implementation/backend mismatch ở context, không phải benchmark GPU raster đã được chứng minh.', 'Intro L11; Conclusion L5',root='context-raster-backend')
add('A','It extracted 14 statistical feature channels encoding stratified height distributions, elevation moments, surface reflectance, and continuous subcell centroid displacement offsets.',
    'tab:strata_bev14:4density+4height+2reflectance+logdensity+occupancy+2offset=14;fig:strata_bev14.',
    'L248–262 đúng14features.', 'VERIFIED',
    'Giữ mô tả feature statistics; không nâng thành khôi phục toàn bộ point geometry.', 'Intro L11,15; Conclusion L5')
add('A','Measured input preprocessing took 20.94 milliseconds, including point-cloud loading and rasterization.',
    'Results L50; log preprocess20.939096ms→20.94; đây là loading+rasterization.',
    '§6.3 L446:20.94ms;L59 gọi raster riêng under21ms/CUDA là diễn giải sai.', 'VERIFIED',
    'Giữ loading and CPU rasterization; không viết riêng raster kernel đạt20.94ms.', 'Intro L11; Methods L122')
add('A',r'71.46\% Moderate mean Average Precision (mAP)',
    'tab:comparison/tab:components A10/tab:attention:71.46; canonical raw71.461874…→71.46.',
    '§1.1/§6.1 L28,413:71.46;§6.6 L497 và L517:71.50.', 'METRIC_MISMATCH',
    'Giữ71.46 cho canonical final benchmark; gắn run/checkpoint/protocol cho71.50 hoặc bỏ số cũ khỏi context. Không đổi headline thành71.50.', 'Intro L19; Conclusion L7',root='context-final-result-snapshot')
add('A',r'73.42\% overall 9-cell mAP',
    'tab:comparison/attention:73.42; canonical raw73.418711…→73.42.',
    '§6.1 L413:73.42;§6.6 L497/L517:73.45.', 'METRIC_MISMATCH',
    'Giữ73.42, tách snapshot73.45; mAP9 là9class×difficulty cells, không phảiIoU.9.', 'Intro L19; Conclusion L7',root='context-final-result-snapshot')
add('A',r'Pedestrian Moderate AP reached 60.75\%.',
    'tab:comparison:60.75;PR figure own legend60.75;canonical log matches.',
    'L413,417,470:60.75.', 'VERIFIED', 'Giữ giá trị cùng BEV/Moderate/R40/KITTI validation boundary.', 'Intro L19; Conclusion L7')
add('A',r'It outperformed PointPillars (70.13\% mAP)',
    'tab:comparison:71.46 vs70.13;note:local vsreference evaluator. PR figure shared-local PP mean≈70.27,không70.13.',
    'L411/413/416:70.13,71.46,+1.33;không cung cấp evaluator equivalence.', 'OVERCLAIM',
    'The reported Moderate mAP values were 71.46% and70.13% under their respective evaluators; the1.33-point difference is descriptive. Muốn superiority:chung evaluator và seeds/uncertainty phù hợp.', 'Intro L19; Conclusion L7; Results L19',root='e03-headline')
add('A',r'matched SECOND (72.09\% mAP)',
    'tab:comparison:ours71.46<72.09,−.63pp;PR shared-local SECOND mean≈72.18;khôngequivalence test.',
    'L416 thừa nhận trailing0.63.', 'OVERCLAIM',
    'The reported score was0.63 percentage points below SECOND, with substantially fewer parameters; disclose protocol difference. Không dùng matched/equivalent.', 'Intro L19; Conclusion L7; Results caption L12',root='e03-headline')
add('A',r'an 84.4\% parameter reduction',
    'tab:comparison .754M vs4.835M;raw753569/4834888→84.41393058%→84.4%.',
    'L23,416:84.4%.', 'VERIFIED',
    '84.4% fewer detector parameters, hoặc84.41% nếu thống nhất2decimal.', 'Intro L19; Conclusion L7')
add('A',r'85.8\% fewer parameters',
    'Raw753569/5325576→85.850000075%→85.9%(1dp),85.85%(2dp). Rounded table .754/5.326→85.8430%→85.8%.',
    'L23:85.8%;L412 dùng5.326M,khôngexact count.', 'METRIC_MISMATCH',
    'Dùng85.85% từ raw counts, hoặc85.9% một chữ số; công bố cơ sở làm tròn.85.8% tái tạo từ bảng đã làm tròn nhưng không phải round-once raw count.', 'Intro L19; Conclusion L7',root='parameter-reduction-double-rounding')
add('I',r'($7.1\times$ parameter reduction)',
    'Raw5325576/753569=7.06713785→7.1;tab:comparison supports.',
    'SECOND/ours counts ratio≈7.1 từL412–413.', 'VERIFIED',
    'Use approximately one-seventh as many parameters (7.07× smaller parameter count); không viết7.1× fewer theo nghĩa phép trừ.', 'Conclusion L7; Results L19')
add('A','The complete pipeline ran in 60.96 milliseconds (16.41 frames per second) on an NVIDIA Tesla T4 GPU,',
    'tab:efficiency:60.96ms/16.41FPS;log60.9563246ms và16.4051886FPS;3759timedframes,batch1FP32.',
    'L27,413,446:khớp.', 'VERIFIED',
    'Giữ sequential raw-file-input-to-detections mean latency trên T4;FPS lấy cùng unrounded mean,khôngmodel-only.', 'Intro L19; Conclusion L7')
add('A','the fastest end-to-end throughput among the evaluated real-time detectors.',
    'tab:efficiency:T4ours16.41>PP15.66>SECOND10.67;historicalMobilePIXOR11.64 trênL4.',
    'L418 cùng nhận định,nhưngMobilePIXOR L4 ởL410.', 'VERIFIED',
    'Giữ giới hạn among the evaluated detectors trong recorded benchmark; tốt hơn viết lowest measured mean E2E latency in the T4 cohort. Không suy universal SOTA/edge deployment.', 'Intro L19; Conclusion L7')
add('A','accurate LiDAR perception on resource-constrained embedded systems.',
    'tab:efficiency đoTeslaT4;khôngembedded target,watt/energy,thermal/INT8/ROS deployment test.',
    'L386–389:T4/98.44MB,suitability làsuy rộng.', 'OVERCLAIM',
    'These results motivate evaluation on embedded platforms; the present measurements are from a desktop/server GPU pipeline.', 'Methods L12; Intro L9,11; Conclusion L5',root='edge-deployment-extrapolation')
add('C','a compact, proposal-free 3D LiDAR object detector',
    'Methods Eq.target_tuple/Algorithm:chỉ x,y,l,w,yaw;tab:comparison BEV;khôngprimary3DAP.',
    'L425–433:oursN/A3D,BEVonly;optionalvertical ởL121.', 'OVERCLAIM',
    'a compact BEV object detector operating on3D LiDAR point clouds; giữ limitation full3D chưađánhgiá.', 'Intro L11; Methods L12,415; Results L192',root='bev-versus-full-3d')
add('I','Empirical validation and leading edge Pareto efficiency.',
    'fig:pareto_tradeoff/tablecomparison:ours nondominated trong observed set;khácAPprotocol/hardware. Parameter-frontier line đi quaSECOND dùPointRCNN(4.039M,73.38)dominatesSECOND(5.326M,72.09).',
    '§6.1 chỉ5models;khôngsearchtoànliterature.', 'OVERCLAIM',
    'Empirical accuracy–parameter–latency trade-offs on KITTI validation. Có thể nói observed non-dominated point nếu giữ protocol caveats.',root='pareto-scope')
add('I','Neither combines continuous subcell displacements with continuous vertical spatial moments.',
    'ChỉcitesPillarHist/TriBand;encodingtable đo3channelvariant,khôngxác minh nội dung từngpaper.',
    'L503–508 soencoding variants;khôngprior-art claim verification.', 'UNSUPPORTED',
    'Nêu thành phần mà paper này combines; kiểm tra original references trước khi khẳng định neither. Không tự thêm first/novel.', 'Related Work L6',root='prior-art-claim-not-verified')
add('I','without learned PointNet encoders or 3D sparse convolutions.',
    'fig:r37_overview,tab:strata_bev14,backbone/tableconfig;codefinal2Dconv.',
    'L24,58,220 xác nhận design.', 'VERIFIED',
    'Giữ structural statement; không đồng nhất encoder-free với zero preprocessing cost.', 'Abstract; Methods L12; Conclusion L5')
add('M','Both compute the same fourteen deterministic features without learned parameters.',
    'tab:strata_bev14 vàrasterformulas define14ch;khôngquantlearner.',
    '§4.1 L248–262:khớp.', 'VERIFIED',
    'Giữ cùng feature definitions; không hàm ý bitwise-identical mọibackend nếu chưa differential test.', 'Abstract; Intro L11; Conclusion L5')
add('M',r'Block-SimAM adds \textbf{zero parameters}',
    'tab:attention caption;source SimAMCore khôngParameter;Phase2countsamearchitectureoff/on đều753569.',
    'L71,281:0learnedparams.', 'VERIFIED',
    'Giữ zero additional trainable attention parameters; có thêm operations/latency.', 'Abstract; Intro L15; Conclusion L5')
add('M','for nine attention operators in total.',
    'fig:backbone_f32/tab:setup depths3+4+2=9;resolved finalconfigmatches.',
    'L70,159,215,517 nói8 hoặcdepth2/4/2.', 'METRIC_MISMATCH',
    'Giữ9(final);context8 phảigắn lịch sử2/4/2. Không sửa final về8.', 'Intro L11; Experiments L43',root='context-architecture-snapshot')
add('M',r'\mathbf{F}_{\mathrm{out}} \in \mathbb{R}^{B \times 32 \times 200 \times 176}',
    'fig:backbone_f32/tab:setup/head32;resolvedconfig F32,out32,H32.',
    'L182–198,222–225 nóiout16/head16,trongL75 lạiF32.', 'METRIC_MISMATCH',
    'Giữ final32; version context và channel contract.', 'Intro L11,15; Experiments L43',root='context-architecture-snapshot')
add('M',r'depthwise $3 \times 3$ refinement, BN, and SiLU produce $\mathbf{U}_4$ and $\mathbf{U}_3$',
    'fig:backbone_f32 thực ghiDW48→48 rồiConv48→32;codeproj_u3 thường,khôngDW.',
    'L177–179 phân biệtRefine/Proj nhưngL73 diễnđạtchungDW.', 'METRIC_MISMATCH',
    'Nêustride8 depthwise48→48;stride4 standardConv48→32. Sửa cùngcaption SG-FPN vàIntro.', 'Intro L11; Methods caption L186,276',root='t01')
add('I','avoiding deconvolution checkerboard artifacts.',
    'tab:components A7→A8:68.12→67.38,−.74pp;fig:scale_gated_fusion làschematic,khôngartifacterror metric.',
    'L73 nóieliminates;L515 gộpnhầmwidth/gategain.', 'OVERCLAIM',
    'uses bilinear upsampling in place of transposed convolution; intended to avoid that particular upsampling mechanism. Chưaclaim artifact-free/suppressnoiseproved.', 'Methods L276; Results L106',root='gating-artifact-claim')
add('M',r'a $7 \times 7$ depthwise receptive field spans $0.7 \times 0.7$\,m at stride 1.',
    'Kernel-footprint arithmetic7×.1=.7m;stride8 footprint7×8×.1=5.6m;fig:backbone.',
    'L67,203–206 cùngvalues.', 'VERIFIED',
    'Gọi nominal kernel footprint; không nhầm với accumulated effective receptive field của toàn network.', 'Intro L9,11')
add('M',r'The backbone achieved $55.94$\,FPS on Tesla T4',
    'tab:efficiency55.94 làwhole model+heads,17.88ms;backbone604864 chỉmộtcomponent.',
    'L26,446 cũngwhole model.', 'METRIC_MISMATCH',
    'The complete detector achieved55.94FPS for in-pipeline model forward; chưacóbackbone-onlytiming.',root='latency-scope')
add('M',r'standalone network throughput reached $55.94$\,FPS',
    'tab:components/efficiency note:standalone115.32FPS,8.67ms;pipelineforward55.94FPS,17.88ms.',
    'L26 phânbiệtđúng2regimes.', 'METRIC_MISMATCH',
    'Đổi55.94 thành in-pipeline model forward; giữ standalone riêng115.32 với raw timing protocol cần xác minh.',root='latency-scope')
add('M',r'It improved Moderate mAP by $+6.03\%$ over naive sampling in the ablations.',
    'tab:componentsA5=60.22,A6=66.25:6.03pp;relative10.0132846%.',
    'L64,486–487,514:+6.03 nhưnggọibằng%.', 'METRIC_MISMATCH',
    'The reported progression increased Moderate mAP from60.22% to66.25%,a6.03-percentage-point change; controlled-effect qualification cần manifests.', 'Intro L19; Results L102; Conclusion L7',root='percentage-points-units')
add('I',r'OGA and IQA added \textbf{+3.20\% mAP}',
    'tab:componentsA8=67.38,A9=70.58:3.20pp;lossfactorial full.',
    'L493,516:70.58−67.38=3.20.', 'METRIC_MISMATCH',
    'OGA+IQA row was3.20percentage points abovebaseline; package-level,khôngmỗimodule3.20.', 'Results L110; Conclusion L7',root='percentage-points-units')
add('I',r'Block-SimAM added \textbf{+0.88\% mAP} without additional parameters.',
    'tab:attention70.58→71.46=.88pp;off/onparameter-freeclaim đúng;historicalrowidentity chưa khóa.',
    'L493→497:70.58→71.50=.92pp,mAP9 .79 so paper .76.', 'METRIC_MISMATCH',
    'The reported A9–A10 difference was.88percentage points; reconcile context snapshot and architecture/schedule before isolated causal claim.', 'Methods L261; Results L115–117,134',root='context-final-result-snapshot')
add('I','In controlled ablations,',
    'tab:loss noattention.721M vsfinal.753569M;zero-paramSimAM cannot explain delta;curves50 vsfinal160epochs.',
    '§6.6 progression khôngmanifest từngrun;depth8 old vsfinal9.', 'OVERCLAIM',
    'In the reported architectural progression; chỉkhôi phục controlled sau config/hash,budget,seed,selectionpolicy matched.', 'Conclusion L7; Tables encoding/loss',root='m03')
add('I','Optional visibility, shadow-masking, and intensity-adjustment heuristics address additional synthesis artifacts.',
    'Khôngablation từngoptionalmechanism;Experiments L49:disableLOS/shadow/radiometry;visibilityflagguardbypassfinal.',
    'L62,514 gángainchoLOS/shadowtoànpackage,khôngđúngfinaltoggles.', 'UNSUPPORTED',
    'The sampler provides optional checks; their individual benefits were not evaluated in the reported final configuration. Giữ chúng làcapability,khôngvalidated contribution.',root='optional-branch-evidence')
add('I',r'12-parameter \textbf{T-SBUW} balances task objectives,',
    'MethodsL384–401 formula12logits;fig:loss_curves khônguniform/EMA-only/fullT-SBUW ablation.',
    'L328–339 cóformulation nhưng§6.6khôngrowtáchweighting.', 'UNSUPPORTED',
    'T-SBUW defines adaptive weights; its isolated contribution was not measured. Nếugiữđónggópchính,thêmmatched ablation.', 'Conclusion L5; Methods L384',root='orphan-tsbuw',orphan=True)
add('M','This construction strictly conserves the total weight.',
    'Eq.tsbuw_weights:sumω=J=6 bằngsoftmaxnormalization;proofalgebra,khôngcầnbảngAP.',
    'L333:sumω=6.', 'VERIFIED',
    'Giữ total task weight; không nâng thành conserved gradient norm hoặc guaranteed convergence.')
add('M','to ensure numerical stability during deep backpropagation.',
    'LayerScaleinit1e-5 ởMethods;khônggradientstats/nonfinite-rate/controlledstabilitytest.',
    'L272 cũngensuringstableconvergence nhưngkhôngsupportingstats.', 'OVERCLAIM',
    'to initialize residual updates at a small scale; stability làdesign motivation,khôngguarantee.',root='stability-guarantee')
add('M','Detaching the target limits gradient instability while retaining supervision for confidence calibration.',
    'Eq.iqa_loss stopgrad đúngchặngradientquaqualitytarget;khôngcalibration ECE/reliability/stability ablation.',
    'L326–327:BCEwithdetachedproxy.', 'UNSUPPORTED',
    'Detaching the target prevents gradients from propagating through the target-generation path. Chưađịnhlượnggradientstability/calibration improvement.',root='quality-calibration-mechanism')
add('M',r'We developed \textbf{Q-OGA} as a head-free alternative for resource-constrained platforms.',
    'tab:loss cóQOGA69.20/.702M/6.94ms;Methods cósectionnhưngIntroduction contributions/ConclusionkhôngtómQ-OGA.',
    '§4.4.3 và§6.6 L491:69.20.', 'UNSUPPORTED',
    'Nếu Q-OGA là đóng góp mới: thêm Intro và Conclusion cùng bounded result; hoặc ghi rõ auxiliary comparison, không phải main contribution.', 'Related Work L24; Results L127',root='orphan-qoga',orphan=True)
add('M',r'adding $18.59\,\text{K}$ parameters',
    'tab:loss .721−.702=.019M saurounding,khôngsuyexact18.59K;final-source counterIQA=18626=18.626K.',
    '§3.3heads111921 nhưngkhôngexactIQA-onlydelta;historicalconfigs chưakhóa.', 'METRIC_MISMATCH',
    'Khôngkhẳngđịnh18.59K cho final. Côngbốexactper-run counts;finalsame-topologybranchdelta18,626,nhưngkhôngghiđèhistoricalrows chưaresolve.', 'Methods L381; Results L127; tab:loss note',root='iqa-parameter-delta')
add('M',r'Q-OGA reduced forward latency by $1.25\,\text{ms}$',
    'tab:loss8.19−6.94=1.25ms;đúngtableonly.',
    '§6.6IQA122.16FPS→8.186ms,QOGA133.97FPS→7.465ms,delta≈.722ms;khác144.16FPSpaper.', 'METRIC_MISMATCH',
    'The standalone table reports1.25ms; reconcile different run/device/timing snapshots in context before attributing saving solely to removing IQA.', 'Results L127',root='standalone-run-provenance')
add('R',r'Q-OGA achieved $69.20\%$ Moderate mAP, exceeding OGA alone by $+0.23\%$ and baseline Smooth-L1 by $+1.82\%$.',
    'tab:loss69.20−68.97=.23pp;69.20−67.38=1.82pp.',
    'L489–491:67.38,68.97,69.20.', 'METRIC_MISMATCH',
    'Giữ69.20%;gains.23 và1.82percentage points. QOGA cònđổiRDA/softmin,khôngisolatequalitycoupling alone.',root='percentage-points-units')
add('M','The explicit IQA branch gave the highest ranking accuracy,',
    'tab:lossđoAP fullobjectives;khôngrankaccuracy/AP-atfixedranking/ECE/qualitycorrelationmetric.',
    '§6.6IQA-only70.31,OGA+IQA70.58;khôngrankingaccuracymeasure.', 'OVERCLAIM',
    'The OGA+IQA configuration had the highest Moderate mAP among the configurations in the loss-factorial table.',root='quality-calibration-mechanism')
add('M','Normalizing by the target bounding box diagonal ensures scale invariance across categories.',
    'Eq.corner_loss numerator/GTdiagonal scalescancel khi predicted/targetgeometry cùngscale,floorinactive.',
    'L320–324 cùngformula.', 'VERIFIED',
    'Giữ scale-normalized distance; qualify invariance under common uniform scaling away fromepsilon/logclamps. Khôngđồngnhấtclass-independentAP.')
add('M',r'continuous bounding box IoU quality $q^*$',
    'Eq.iqa_loss targetclip(mean4axisGIoU,0,1),khôngexactrotatedareaIoU;Phase2squarewitnessproxy.5 vsIoU.25.',
    'L319/327 gọiMGIoUsimilarity.', 'OVERCLAIM',
    'detached projection-GIoU quality proxy; khôngclaim exactIoU calibration nếu chưađocorrelation/reliability.', 'IQA head descriptions/Conclusion L5',root='proxy-versus-iou')
add('R',r'a $+1.33\%$ margin',
    'tab:comparison71.46−70.13=1.33pp;relative1.896478%.',
    'L416:+1.33%,arithmeticmatchnhưngunitsmơhồ.', 'METRIC_MISMATCH',
    'a1.33-percentage-point reported difference; retain evaluator caveat.', 'Conclusion L7',root='percentage-points-units')
add('I',r'(+4.74\% over PointPillars, +4.19\% over SECOND)',
    'tab:comparison60.75−56.01=4.74pp;60.75−56.56=4.19pp;PRfigurebaselines55.83/56.47→4.92/4.28.',
    'L411–417:referencevalues56.01/56.56.', 'METRIC_MISMATCH',
    '4.74 and4.19percentage points from reference-table values; đừngghép vớiPRshared-localnumbers. Chọnđồngnhấtprotocol trước infer superiority.', 'Conclusion L7; Results L30',root='percentage-points-units')
add('R',r'by $+30.06\%$ ($71.46\%$ vs. $41.40\%$) and by $+31.15\%$',
    'tab:comparison71.46−41.40=30.06pp;73.42−42.27=31.15pp.',
    '§6.1L410,413:cùngvalueslocalR40.', 'METRIC_MISMATCH',
    '30.06percentage points inModerate mAP and31.15points inmAP9; khôngrelativepercent.',root='percentage-points-units')
add('R',r'(\textbf{+6.20\% mAP})',
    'tab:components47.60−41.40=6.20pp;throughput169.97/90.57≈1.8767 nên nearlydouble hợp lý nhưngGPUschưakhóa.',
    'L481–482,511:+6.20.', 'METRIC_MISMATCH',
    'a6.20-percentage-point increase in the reported progression; throughputprovenance phảiđồngnhất.',root='percentage-points-units')
add('R',r'($+5.72\%$ over TriBand-BEV and $+2.54\%$ over Binary35)',
    'tab:encoding50.14−44.42=5.72pp;50.14−47.60=2.54pp.',
    'L505–508,512:khớpvalues.', 'METRIC_MISMATCH',
    '5.72 and2.54percentage points;góiencoding thay nhiều features,nênkhôngquyriêngoffsetgain.',root='percentage-points-units')
add('R',r'(\textbf{+8.92\% mAP})',
    'tab:componentsA5=60.22,A4=51.30:8.92pp;A3→A5=10.08ppcombinetwochanges.',
    'L485–486,513:có50.14→60.22=10.08.', 'METRIC_MISMATCH',
    '8.92percentage points forH16→H32 row;10.08points appliesgrouping+widthcombined.',root='percentage-points-units')
add('R',r'(\textbf{+7.34\% mAP})',
    'tab:components68.45−61.11=7.34pp in mAP9,khôngModerate.',
    'L64,486–487,514:7.34mAP9.', 'METRIC_MISMATCH',
    'a7.34-percentage-point change inmAP9; labelmetricexplicitly.',root='percentage-points-units')
add('R',r'(\textbf{+1.87\% mAP})',
    'tab:componentsA6→A7:68.12−66.25=1.87pp(widthonly);A7→A8:−.74;A6→A8:+1.13.',
    'L515 gộpaddinggatecũngthành68.12/+1.87;saiattribution.', 'METRIC_MISMATCH',
    '1.87points for fusion-width increase alone. SG-FPN row is.74points lower;updatecontext L515.',root='context-gating-attribution')
add('R',r'(\textbf{+1.59\% mAP})',
    'tab:loss68.97−67.38=1.59pp;Car.17,Ped2.45,Cyc2.14pp;IQAalone70.31−67.38=2.93pp.',
    'L489–493:compositevaluesavailable;classdeltanottabulatedcontext.', 'METRIC_MISMATCH',
    'Use1.59/2.93percentage points;fullfactorialinteraction3.20−1.59−2.93=−1.32points. Khônggọipositive synergisticgain.', 'Results L125',root='percentage-points-units')
add('R','Block-SimAM gained 1.66--2.10 mAP points over stage-level variants',
    'tab:attention71.46−69.80=1.66,−69.44=2.02,−69.36=2.10pp;rangeđúng.',
    'Contextstagevaluescùngnhưngfinal71.50→range1.70–2.14.', 'METRIC_MISMATCH',
    'Giữ1.66–2.10points chofinal71.46;resolvecontext71.50. FPS144.05 thuộcstage3,khôngmọistage.',root='context-final-result-snapshot')
add('R','The plotted losses weighted by T-SBUW show no divergence or oscillation.',
    'fig:loss_curves actualPDF:validation/classificationA6pink spike≈epoch18→~100,otherfluctuations;chưadataexport.',
    'Khôngloss-timeseries/rawspikestats ởcontext.', 'OVERCLAIM',
    'Losses generally declined; visible validation/classification spikes occurred in some runs. Khôngclaimnooscillation.', 'Results L145–146',root='loss-curve-overclaim')
add('R','Configuration A10 has the lowest late-training validation losses across the continuous regression tasks.',
    'fig:loss_curves A9blue belowA10red ởGeometry/IoU vànhiềuđiểmOffset/Yaw;lossnormalizationkhácobjective.',
    'Khôngper-tasklate-epochnumbers haycommon-normalizationcomparisons.', 'OVERCLAIM',
    'The plotted regression losses decrease over training; no uniform lowest-loss claim forA10. Export comparable unweighted taskmetrics trước ranking.',root='loss-curve-overclaim')
add('R','This behavior is consistent with regularization from physics-consistent point cloud augmentation (PC-GTA, A6) without apparent underfitting.',
    'fig:loss_curves givestrajectories;khôngmatchedregularization/underfittingdiagnostic;A6cóspike.',
    'PC-GTA APpackageeffectkhôngđooverfitmechanism.', 'UNSUPPORTED',
    'State observed training/validation trends; mechanism and degree of underfitting remain undetermined. Hedgeskhôngtạoevidence.',root='loss-mechanism-speculation')
add('R',r'compared to $56.01\%$ for PointPillars and $56.56\%$ for SECOND.',
    'PRasset legendsPedModeratePP55.83,SECOND56.47;tablemain56.01/56.56. Assettitle/footer:sharedlocalevaluator.',
    '§6.1L411–412:56.01/56.56reference. Đây không chỉrounding.', 'METRIC_MISMATCH',
    'Caption phảiphânbiệtreference-tablevaluesvslocalPRfigure. Regenerateallfigure/tablefromoneprotocol;khôngđổilegendthủcôngmàgiữcurvekhác.',root='pr-table-protocol-mismatch')
add('R','MobilePIXORNeXt consistently maintains superior recall and precision envelopes on vulnerable road users,',
    'tab:comparisonCyclist66.73<SECOND71.16;PRassetCyclistModerateours66.73<SECOND70.70;curvescross.',
    'L413vsL412Cyclist66.73<71.16;farPed.80<.99.', 'OVERCLAIM',
    'Higher Pedestrian AP is observed; Cyclist performance remains below SECOND at Moderate difficulty. KhôngclaimpointwisePRdominance/allVRUs.',root='pr-vru-dominance')
add('R','Car (top row,',
    'PRactualcolumnsCar/Ped/Cyc,rowsEasy/Moderate/Hard;oursgold,PPgreen,SECONDblue,MobilePIXORred.',
    'ContextkhôngPRlayout/color mapping.', 'METRIC_MISMATCH',
    'Sửacaption theoactualrows/columns/colors;đồngthờisửareferenceAPs. Lỗi visualization/legendcontract,cùngroot vớiPRnumericmismatch.', 'Results L24',root='pr-table-protocol-mismatch')
add('R',r'over $2.3\times$ faster than PIXOR',
    'tab:efficiency144.00/60.96≈2.3622;PIXOR/oursđềuT4.',
    'L442,446:144.00,60.96.', 'VERIFIED',
    'Giữ about2.36×pipeline throughput undermeasuredconfiguration;khôngmodel-onlyspeedup.')
add('R',r'reducing payload by $60.0\%$.',
    'tab:encoding35→14float32channels:(35−14)/35=.6;78.848MB→31.5392MB.',
    'L507–508:78.85/31.54MB,512:60%.', 'VERIFIED',
    'Giữ tensor-payload reduction60%;khôngđồngnhấtH2Dlatencyreduction60%.', 'Related Work L7; tab:encoding note')
add('R',r'With pinned-memory buffers, H2D latency fell to $6.91\,\text{ms}$, compared with $40.63\,\text{ms}$ for MobilePIXOR.',
    'Tab:efficiency/rawbreakdownoursT4,MobilePIXORL4;channels/host/timerdiffer,khôngpinning-onlycontrol.',
    'L443vs446:L4vsT4,40.63vs6.91.', 'OVERCLAIM',
    'Recorded H2D times were6.91ms and40.63ms on the respective pipelines; no isolated attribution to pinning orchannelcount.',root='h2d-causal-attribution')
add('R',r'(a $+34.78\%$ improvement)',
    'tab:distance65.23−30.45=34.78pp;Cyclist76.32−25.55=50.77pp;ratios2.142/2.987.',
    'L456,459:cùngvalues;L461doublenearlytripleđúng.', 'METRIC_MISMATCH',
    '34.78 and50.77percentage points; morethandouble/nearlytriple đúngAPratio,khôngsafetydemonstration.', 'Results L70',root='percentage-points-units')
add('R',r'(vs. $12.61\%$, a $+31.69\%$ gain)',
    'tab:distance44.30−12.61=31.69pp;11.49−3.75=7.74pp;farCyclist14.38pp.',
    '§6.4L456,459:cùngvalues.', 'METRIC_MISMATCH',
    '31.69percentage points; explicitlyretainfarPedestrian.80vs.99,−.19points.', 'Results L73',root='percentage-points-units')
add('I','Physics-consistent synthesis regularizes distant returns,',
    'tab:distancefinalvsMobilePIXOR đổi nhiều modules;farPedours.80<.99;khôngrange-specificPC-GTAoff/on.',
    'L461claimnear/mid tốt;L514 gáncơchếchưađượcisolate.', 'UNSUPPORTED',
    'Range-adaptive subsampling is intended to model distance-dependent density; its isolated range-wise benefit was not established.', 'Results L70,73',root='range-mechanism-speculation')
add('R',r'Physics-consistent augmentation (\texttt{PC-GTA}) and decoupled grouped heads produced larger observed accuracy gains than model depth scaling alone.',
    'tab:components cópackage/widthchanges;khôngdepth-onlycontrol row.',
    '§6.6 khôngdepth-onlyrun.', 'UNSUPPORTED',
    'PC-GTA and grouped-head changes had large reported progression differences; remove comparison with untested depth-only scaling.',root='untested-depth-comparison')
add('R','from the extended run (Epoch 50)',
    'tab:ap_vs_lossL160/captionL153:130;setupfinalselected130;canonicalcheckpoint130.',
    'L470:Epoch130;L378130extendedtraininglàcontextstaleschedule.', 'METRIC_MISMATCH',
    'from the extended160-epoch run, selected at epoch130.50 chỉstandard schedule/illustratedcurves.',root='checkpoint-epoch')
add('R','validation AP maximization (Epoch 130)',
    'tab:ap_vs_lossfinalAP71.46<best-loss71.48;selectiontrainingpolicyrecordedmaximumvalAP,khôngallcheckpointselectionmetrics.',
    'L470–471cũng71.46vs71.48.', 'UNSUPPORTED',
    'recorded AP-selected checkpoint(epoch130);clarifytraining-selectionmetricvsfinalbenchmarkandresolvewhyfinalrescorebestlossishigher.',root='checkpoint-selection-provenance')
add('R',r'had $20.27\,\text{ms}$ lower end-to-end latency',
    'tab:ap_vs_loss81.23−60.96=20.27ms;ours_best_loss/AP logsconfirm.',
    'L470–471:81.23vs60.96.', 'VERIFIED',
    'Giữobservedmeanlatencydifference;khônggánnguyênnhâncheckpointnếukhôngcontroltiming/candidatecounts.')
add('R','A possible explanation is that extended training sharpened classification heatmaps around positive centers,',
    'Khôngpeakcount,NMS-candidatehistograms,heatmapsharpness,h2d/model/decodecontrolledanalysis.',
    '§6.5onlymetrics;khôngmechanismdiagnostics.', 'UNSUPPORTED',
    'The mechanism behind the timing difference remains undetermined; testcandidatecounts andstage-levelrepeatedtimings. Giữhypothesisrõnếuđềxuấtfuturetest.',root='checkpoint-latency-speculation')
add('E',r'Peak allocated GPU memory was $98.44\,\text{MB}$.',
    'rawfield98.439453125;source divides1024²→MiB,notdecimalMB;~103.22MBdecimal.',
    'L389 gọi98.44MB,cùngunitlabelissue.', 'METRIC_MISMATCH',
    'Peak allocated evaluation GPU memory was98.44MiB;khôngtrainingpeak/modeldiskbytes.',root='memory-unit')
add('E','The reported checkpoint came from an extended 160-epoch run with 10 warmup epochs and was selected at epoch 130.',
    'tab:setup đúng160/10/130;resolvedfinalconfig/checkpointidentityconfirms.',
    'L377–378chỉstandard50/warmup4 hoặcextended130.', 'METRIC_MISMATCH',
    'Giữpaper160/10/130;refreshcontext phânbiệtschedule length vàselectedepoch.',root='context-training-snapshot')
add('E','Variance and standard deviation',
    'ExperimentsL52nóileftforfuture,ResultsL5cùngsingle-seed42;khôngstatisticalCItest.',
    'Configseed42;contextkhôngmulti-seedstats.', 'VERIFIED',
    'Giữsingle-seedlimitation;khôngtựthêmstatistically significant,robustacrossseeds ởrewrites.')
add('M',r'is fully recovered without trigonometric ambiguity.',
    'Eq.box_decodehalfatan2 choheadingcanonical[−π/2,π/2];notfull2πorientation;GTusesπsymmetry.',
    'L119,139–140:cùngπsymmetriccontract.', 'VERIFIED',
    'The canonical heading is recovered moduloπ; khôngclaimfront/backdisambiguation.')
add('W',r'($\mu_z, \sigma_z, z_{\min}, z_{\max}$)',
    'tab:strata_bev14actualmax,mean,span,std;khôngstandalonez_minchannel.',
    'L250–254:max,mean,span,std.', 'METRIC_MISMATCH',
    'ĐổiRelatedWorkthànhz_max,z_mean,z_span,z_std. Minimum cóthểsuy từmax−span,nhưngkhôngphảichannelriêng.',root='encoding-feature-list')
add('I',r'memory access latencies ($100\text{--}195\,\text{ms}$)',
    'tab:efficiency làE2E,khôngmemoryaccess;SECOND93.70ms<100;two-stage133.75–194.50.',
    '§6.1SECOND93.70;§6.3stagebreakdownnotmemory-only.', 'METRIC_MISMATCH',
    'Name E2E latencies and the exact detector subset; do not relabel whole-pipeline timing as memory access or infer embedded limits from T4.', 'RelatedWork L11; Results L33',root='latency-scope')
add('M','Calibrated surface returns capture material discrimination',
    'tab:strata_bev14ch8/9max/meanintensity;khôngmaterial-labelclassification/calibrationstudy.',
    'L255–256onlyreflectanceformulas.', 'UNSUPPORTED',
    'Reflectance statistics provide intensity cues; remove untested examples ofmaterialdiscrimination.',root='reflectance-semantics')
add('M','anchored strictly flush with the terrain:',
    'Eq.ground_fitting setsbottomtoestimatedpercentileheight;khôngtruegrounderror/planeGTmetric.',
    'L286–288sameheuristic.', 'OVERCLAIM',
    'anchored to the estimated local ground elevation; do notpromiseexactphysicalterrainalignment.',root='ground-estimate-versus-truth')
add('M','The twelve logit parameters are utilized strictly during training and discarded at inference.',
    'tab:setup/Methods12criterionparams;detectorforwardusesneithercriterionnorEMA.',
    'L236,330:12trainingonly.', 'VERIFIED',
    'Giữ12training-onlylogits,notpartof753569detectorweights.')
add('E','and visibility protection.',
    'tab:setupPCGTApackage;hybrid_sampler protect=flag&&(shadow||extra_width>0||margin<=1e-8). Finalflagtrue,shadowfalse,width0,margin.1→false.',
    'Contextmô tảmechanisms,khôngruntimeguard distinction.', 'METRIC_MISMATCH',
    'The visibility-protection flag was enabled, but the explicit check was bypassed under the final sampler guard; do notclaimitactivelytestedallplacements.',root='visibility-guard')
add('R',r'($188.4\,\text{FPS}$)',
    'tab:encoding188.43→188.4(1dp),169.01→169.0;GPUL4caption.',
    'L501–508:L4,188.43/169.01;componentsreuse169.97/165.02/169.01asT4.', 'METRIC_MISMATCH',
    'Roundingvaluesđúng;reconcileL4vsT4forreusedrows. Do notcallforwardFPS E2E FPS.',root='e07-nguồn')
add('R','Percentile analysis showed low latency dispersion in the evaluated desktop GPU environment:',
    'tab:efficiencyp50=58.28,p95=80.85,p99=95.91;noquantitativecriterionforlowdispersion.',
    'L446cùngpercentiles.', 'OVERCLAIM',
    'Reportpercentilesdirectly;mean16.41FPSsupports10Hzaveragebutnotworst-caseharddeadline.',root='latency-variability-wording')
add('C','The primary detector estimates oriented 2D Bird\'s-Eye View bounding boxes.',
    'tab:comparisonBEV,Eq.target_tuple7values,fig:qualitativecaptionsprojectedfootprints.',
    'L431–433:BEVonly.', 'VERIFIED',
    'Giữlimitationnày vàđưaBEVscopeđếnAbstract/Intro;khôngxóachỉđểclaimmạnhhơn.')
add('C','requires evaluation on dedicated off-road benchmarks.',
    'Khôngoff-road/adverseweather/16beam/TensorRTINT8resulttrongtables;futuretenseproper.',
    'ContextchỉKITTI64beam/fairweather;optional3Dheadnotbenchmarked.', 'VERIFIED',
    'Giữaslimitation/futurework,khôngcoiđây làorphanvalidatedcontribution. Đồngthờikhôngclaimembeddedproven.', 'Conclusion L9; Results L195')
add('R','The second is GroupedHeader, which separates pedestrian regression from vehicle footprints to reduce gradient interference between categories.',
    'tab:componentsgrouping/widthgivesAProws;khônggradient-cosine/conflictstatistics;đồngthờiđổi nhiềufeatures trongcross-familycomparison.',
    'L77,225,513 nóieliminatesconflict nhưngkhôngđogradients.', 'UNSUPPORTED',
    'GroupedHeader separates task parameters. Its isolated effect on gradient interference was not measured; keep as motivation, not explanation established by AP.', 'Results L97; Methods L307',root='gradient-mechanism-speculation')
add('M','Its 3D spatial-and-channel weights modulate sparse LiDAR clusters relative to flat terrain backgrounds.',
    'Eq.simam_energy maskphụthuộcactivationdeviation;khôngsemanticLiDARcluster/terrainattentiongroundtruth.',
    'L71 tươngtựamplifyingclusters;khôngclass-specificmaskmetrics.', 'UNSUPPORTED',
    'The mask modulates feature activations according to their deviation from channel moments; tests would be needed to establish preferential amplification of object clusters.',root='attention-semantic-extrapolation')
add('M','stage-level adapters add learned linear projections.',
    'Code ResidualLightAttention có một scalar beta vớiX+beta*(SimAM(X)−X),khônglearnedchannelprojection;tableattentionkhôngparamcounttừngadapter.',
    'Context khôngexactadapterdelta;rawrevision củahistoricalruns chưađượcchỉrõ.', 'METRIC_MISMATCH',
    'Describe the current adapter as a learned scalar residual interpolation, or cite the historical implementation actually benchmarked.',root='t08')
add('W',r'inverted bottleneck expansions ($e=2.5$)',
    'RelatedWork gánchoConvNeXt citedpaper;Methods codefinalMobilePIXORNeXt-Sexpansion2.5 đúng nhưngkhôngxácminhoriginalConvNeXt.',
    'L66,220 chỉmô tảprojectownblock2.5.', 'UNSUPPORTED',
    'Attribute large-kernel design inspiration to ConvNeXt and state separately that the present block uses expansion2.5; verify original citation before assigning that ratio to it.',root='prior-art-claim-not-verified')
add('R',r'by less than $1.7\%$',
    'tab:comparisonCar:86.90−88.02=−1.12pp,86.90−88.56=−1.66pp;vsPIXOR+14.88pp,MobilePIXOR+11.38pp. CyclistvsPP+.39pp,vsSECOND−4.43pp.',
    'L409–413 cùngAPvalues.', 'METRIC_MISMATCH',
    'less than1.7percentage points forCar;retainCyclist deficit4.43points relative toSECOND and evaluator boundary.',root='percentage-points-units')
