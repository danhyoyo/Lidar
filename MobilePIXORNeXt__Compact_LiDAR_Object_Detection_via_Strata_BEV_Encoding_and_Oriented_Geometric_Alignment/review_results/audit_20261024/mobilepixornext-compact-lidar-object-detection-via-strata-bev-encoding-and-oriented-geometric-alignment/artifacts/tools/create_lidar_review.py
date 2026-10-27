from pathlib import Path
import json,re,difflib,collections,html,shutil
P=Path('/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment')
W=next((P/'review_results/audit_20261024').iterdir());A=W/'artifacts';O=A/'original'
issues=[]
def issue(id,title,file,needle,explanation,lane='notation_and_numeric_consistency',severity='moderate',status='Đã sửa diễn giải',action=''):
    source=(O/file).read_text();assert needle in source,(id,file,needle)
    line=source[:source.index(needle)].count('\n')+1
    issues.append(dict(id=id,title=title,quote=needle,explanation=explanation,comment_type='methodology' if lane=='evaluation_fairness_and_reproducibility' else 'claim_accuracy',severity=severity,confidence='high',source_kind='llm',source_section=f'{file}:{line}',related_sections=[file],root_cause_key=id.lower(),review_lane=lane,gate_blocker=severity=='major',quote_verified=False,status=status,action=action))
issue('I01','Hai entry point chứa abstract và tuyên bố khác nhau','MobilePIXORNeXt.tex','under the official 40-point interpolation protocol','main.tex mô tả local evaluator chưa xác minh tương đương; MobilePIXORNeXt.tex vẫn ghi official và highest Pedestrian AP. sections/abstract.tex lại có bản thứ ba. Đã thống nhất MobilePIXORNeXt.tex → main.tex → sections/abstract.tex.',lane='self_standard_consistency',severity='major',status='Đã sửa')
issue('I02','Tuyên bố unified/official trái với evaluator thực tế','sections/experiments.tex','The evaluation code strictly adheres to the official KITTI devkit evaluation protocol','Workbook v7, sheet B05_Protocol_Audit: ours/MobilePIXOR dùng local loader-aligned, strict x/y ROI; PIXOR và năm baseline dùng reference evaluator, full-frame GT. Cùng R40 và IoU không chứng minh cùng ignore/matching/GT domain. Đã bỏ shared AP ranking, thêm L/R trong bảng. Chưa có đối chiếu evaluator trên cùng predictions.',lane='evaluation_fairness_and_reproducibility',severity='major',status='Đã sửa mô tả; thiếu thí nghiệm equivalence',action='Chạy cùng predictions/GT qua hai evaluator, lưu class/difficulty ignore rules và matching outputs.')
issue('I03','8.67 ms và 17.88 ms bị giải thích sai thành hai chế độ T4','sections/results.tex','Standalone GPU Forward Latency','Seed workbook Theo seed!AF7: 8.671745659 ms trên L4; v7 B10_Speed_Detail!E14: 17.875186737 ms trên T4. Không có thí nghiệm cùng GPU chứng minh chênh lệch do synchronization/DataLoader. Đã ghi GPU và tách snapshot.',lane='evaluation_fairness_and_reproducibility',severity='major',status='Đã sửa')
issue('I04','AP L4 và T4 của cùng tên model bị gộp','tables/comparison.tex','86.90 & \\textbf{60.78} & \\underline{66.81}','Seed workbook: L4 86.900441/60.778199/66.807415, mAP 71.495352, mAP9 73.445159. V7 T01_Benchmark row16: T4 86.903460/60.749361/66.732802, mAP 71.461874, mAP9 73.418711. Giữ cả hai snapshot với nhãn nguồn; chưa thể khẳng định khác biệt chỉ do GPU. Cần checkpoint hash/prediction hash.',severity='moderate',status='Đã tách; cần kiểm tra hash',action='Đối chiếu SHA256 checkpoint, config, split và prediction files giữa báo cáo L4/T4.')
issue('I05','Lịch 50/100 epoch và epoch cuối không khớp metadata','sections/experiments.tex','all exploratory ablation experiments (A1--A10) were trained under a standardized 50-epoch schedule','Seed workbook Cau hinh: Final S42 completed/planned 160/160, selected130; Final S43/S44 100/100, selected100; B0 archived38/planned100 selected30. runs_summary.csv: B1 budget100 selected13 (loss), E8 budget50 selected20, E14 budget50 selected30. Epoch47 là checkpoint loss-selected trong lịch sử final run, không chứng minh run độc lập50epoch. Đã sửa toàn bộ khẳng định uniform schedule/asymptotic convergence.',lane='evaluation_fairness_and_reproducibility',severity='major',status='Đã sửa mô tả; chưa có thí nghiệm đồng budget',action='Lập ma trận run/config/selected epoch/criterion; chỉ tính delta component khi các biến khác khớp.')
issue('I06','Bảng seed không chỉ thay seed','tables/seed_robustness.tex','under a 100-epoch training schedule','Cau hinh!T10=Binary35 của MobilePIXOR S42; T11/T12=hist14 của S43/S44. S42 archived38epoch, các seed còn lại100; Final S42 160epoch, S43/S44 100epoch. Mean/std vẫn là phép tính mô tả hợp lệ nhưng không đo initialization variance hoặc chứng minh numerical stability. Đã đổi caption/note và bỏ suy luận nhân quả.',lane='evaluation_fairness_and_reproducibility',severity='major',status='Đã sửa diễn giải; cần matched reruns',action='Giữ cùng encoding, model, sampler, optimizer schedule, budget và checkpoint selection cho mọi seed.')
issue('I07','A4→A5 không cô lập head width','sections/results.tex','Expanding head width to 32 channels (A5) adds only','A4 là G16 H16/F24/global-only, 51.30. A5=O32 H32/F32/SimplifiedGT V1 p=1,60.22. Cùng đổi head, neck, augmentation; +8.92pp không phải head-only. Không có đo rank hoặc gradient chứng minh subspace collapse. Đã sửa cấu hình và bỏ cơ chế chưa đo.',lane='section_results',severity='major',status='Đã sửa nhãn/diễn giải; chưa có isolated head run')
issue('I08','66.25% bị gắn nhầm PC-GTA H32/F24','tables/components.tex','A6 & MobilePIXORNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F24 / SumFPN','runs_summary.csv O32G16:66.247315/68.453192, H16/F32/SG-FPN/SimplifiedGT V2. A5=O32:H32/F32/SumFPN/SimplifiedGT V1. Vì vậy +6.03pp không đo PC-GTA. H32 hybrid thực sự68.124178; O32→H32=+7.906241pp, vẫn thay cả p/quota/placement recipe. Đã bỏ mọi +6.03 PC-GTA claim, ghi source-linked inventory.',lane='section_results',severity='major',status='Đã sửa')
issue('I09','A8 trộn AP P1 với tốc độ S32','tables/components.tex','67.38 & 69.46 & 6.94 & 144.16','P1 focal context:67.378677/69.460029 và7.772452ms/128.659530FPS. S32 no-focal SG-FPN:68.459941/70.283161 và6.936693ms/144.160918FPS. Bản cũ ghép hai run vào một hàng. Đã giữ A8=S32 và thêm A8F=P1. A7H32→A8S32 còn đổi quota nên không cô lập neck.',lane='section_results',severity='major',status='Đã sửa')
issue('I10','AP từng lớp của Baseline và Q-OGA không khớp nguồn','tables/loss.tex','83.92 & 55.48 & 62.74','P1 nguồn:82.970232/56.992056/62.173743 →82.97/56.99/62.17; P11 nguồn:84.893017/57.145242/65.563877 →84.89/57.15/65.56. Các bộ số cũ vẫn cho cùng mAP làm tròn nên chỉ kiểm mean sẽ bỏ sót. Đã sửa từ paper_06_Loss_IQA.csv và workbook v7 T02.',status='Đã sửa')
issue('I11','Ghép Q-OGA lịch sử với số tham số của model cuối','tables/loss.tex','saving $18{,}626$ parameters','P1/P2/P11 dùng815743params; P3/P4 dùng852929params, chênh37186. Model cuối shared-head753569 khác kiến trúc. 69.20% là P11 lịch sử, không có bằng chứng ghép nó với734943params. Đã sửa bảng0.816/0.853M và tách rõ mô hình cuối.',lane='evaluation_fairness_and_reproducibility',severity='major',status='Đã sửa')
issue('I12','Factorial study không chứng minh positive synergy','sections/results.tex','These results demonstrate clear methodological synergy','Từ mAP chưa làm tròn: OGA khôngIQA +1.588431pp; cóIQA +0.269471pp. Interaction=P4−P3−P2+P1=−1.318960pp. P4 cao nhất trong factorial không đồng nghĩa tương tác dương; chưa có uncertainty cho hiệu ứng nhỏ. Đã viết lại đúng main effects và interaction.',lane='claims_vs_evidence',status='Đã sửa')
issue('I13','Attention table thay context, neck width và lịch sử run','sections/results.tex','strictly attributable to 3D spatial-energy attention','No-attention70.58=P4 focal/independent towers852929; final71.50=no-focal/shared towers753569/selected130. R25 middle stage row sourceF24, manuscriptF32. Lidar_result_final.xlsx còn có R37 F24 70.34 khác final. Không có đối chứng matched no-attention cho final. +0.88 Methods và+0.92 Results còn tự mâu thuẫn; đã bỏ isolated gain và ghi đúng phạm vi bảng.',lane='evaluation_fairness_and_reproducibility',severity='major',status='Đã sửa diễn giải/width; cần matched controls',action='Chạy cùng final config, chỉ đổi block/stage attention, giữ width/depth/context/seed/schedule và rule chọn checkpoint.')
issue('I14','Related Work gán cơ chế LOS/shadow bị tắt cho kết quả chính','sections/related_work.tex','integrating adaptive local ground plane fitting, static obstacle clearance filtering, line-of-sight ray checks, and shadow masking','Experiments và Results ghi LOS, shadow, radiometric disabled. Đã mô tả core enabled mechanisms và optional options, không quy AP gain cho cơ chế không bật.',lane='section_introduction_related',status='Đã sửa')
issue('I15','Tổng task weights không phải tổng gradient budget','sections/methodology.tex','strictly conserves the total gradient budget across all tasks','Sum omega=J chỉ bảo toàn tổng trọng số. Gradient mỗi task còn phụ thuộc scale/direction và shared features; không suy ra gradient norm hoặc convergence. Đã sửa phát biểu, giữ nguyên công thức.',lane='section_methods',status='Đã sửa')
issue('I16','Detached surrogate không loại hết gradient competition','sections/methodology.tex','Detaching $q^*$ prevents gradient competition','q*=clipped mean projectionGIoU không phải exact polygon IoU. stopgrad chặn đường qua target, nhưng quality và geometry còn chung backbone/neck. Exact rotated IoU cũng không thể gọi là hoàn toàn nondifferentiable ở mọi cấu hình. Đã sửa mục tiêu surrogate và giới hạn gradient claim.',lane='section_methods',status='Đã sửa')
issue('I17','Doubled-angle không khôi phục directed heading liên tục','sections/methodology.tex','is fully recovered without trigonometric ambiguity','cos2psi/sin2psi biểu diễn orientation modulo pi, không có front/back. Scalar atan2 representative vẫn wrap ở boundary và vector0 không có direction. Đã nêu đúng contract, giữ nguyên phương trình.',lane='section_methods',status='Đã sửa')
issue('I18','BEV/edge latency bị suy rộng thành 3D và safety deployment','sections/results.tex','accurate 3D LiDAR detection was possible','Output chỉ (x,y,l,w,yaw modulo pi), khôngz/h/velocity/tracking. KITTI/T4/L4 không chứng minh embedded/closed-loop collision avoidance hoặc TTC; P99 offline<100ms không phải worst-case guarantee. Đã giới hạn abstract/results/conclusion vào BEV và measured offline latency.',lane='section_discussion_conclusion',status='Đã sửa')
issue('I19','Cơ chế tối ưu hóa chưa có thí nghiệm phân biệt','sections/results.tex','induces severe subspace rank collapse','Không có feature singular values, gradient diagnostics hoặc matched width control. Các suy luận weak-return attenuation, synergy SG-FPN, T-SBUW suppresses spikes và epoch latency do sharpening đều vượt dữ liệu. Đã thay bằng quan sát AP/loss và ghi điều kiện kiểm chứng.',lane='claims_vs_evidence',status='Đã sửa')
issue('I20','Monotonic descent mâu thuẫn validation spikes','sections/results.tex','stable monotonic descent','Results ghi classification spikes epochs18–25 rồi caption nói stable monotonic và rapidly stabilized byT-SBUW. Đã đổi thành generally decreasing trajectories; không quy nguyên nhân nếu không có weighting ablation.',lane='self_standard_consistency',status='Đã sửa')
issue('I21','Timing MobilePIXOR và một số percentile không khớp snapshot','tables/efficiency.tex','90.65 & 11.03','V7B10: MobilePIXOR historicalL4 model11.049958/E2E85.923112/P50 84.114180/P95 112.524565/P99 124.943548. PointRCNN P50/P99=194.769222/222.372998; PartA2=132.399653/169.129394; PVRCNN=187.348769/219.072451. Đã sửa các ô có nguồn xác định, bỏ same-T4 claim. Không suy speedup L4/T4.',status='Đã sửa')
issue('I22','Far-range Pedestrian regression bị bỏ qua trong diễn giải','sections/results.tex','These results are consistent with range-adaptive augmentation','Table distance:PedFar0.99→0.80,−0.19pp; Results chỉ nhấn mạnhCyclistFar14.94 và VRUgain. Đã nêu regression và bỏ suy luận nhân quả sampler/kernel từ whole-model comparison.',lane='section_results',status='Đã sửa')
issue('I23','Delta_z 0.1 m không mô tả Strata-BEV14','tables/setup.tex','Grid resolution ($\\Delta_x, \\Delta_y, \\Delta_z$)','Strata14 dùng4heightbins trên3.5m, còn dz0.1 thuộc legacy35slices. Đã tách horizontalgrid0.1x0.1 vớifourverticalstrata; không đổi công thức.',severity='minor',status='Đã sửa')
issue('I24','Serialized checkpoint không phải kích thước trọng số FP32 thuần','sections/methodology.tex','with an FP32 checkpoint size of $9.34$','753569*4bytes=3.014276MB; serializedcheckpoint9.34MB có thể chứa thêmstate/metadata. Mainabstract cũ gọi753569trainable nhưng12T-SBUW logits nâng tổngtrainable753581. Đã thống nhất deployable/train-only và serializedsize, không khẳng định thành phần checkpoint khi chưa có inspection.',severity='minor',status='Đã sửa')
issue('I25','Hình vehicle được dùng xác nhận VRU AP','sections/results.tex','This visual evidence corroborates the substantial','Ví dụ vehicle heatmaps không xác nhận định lượngPed/Cyc30–50m. Whole-model visualization cũng không trực tiếpvalidate BlockSimAM/SGFPN effects hoặc NMSspeed. Đã giới hạn caption và observation vào selectedexamples.',lane='claims_vs_evidence',status='Đã sửa')
issue('I26','Phép cộng CPU statistics không tự bảo đảm CUDA sync không đáng kể','sections/methodology.tex','negligible thread synchronization','Các cell độc lập sau group nhưng nhiều point có thể update cùngcell, cần grouping/reduction/atomic strategy. Chưa có đo CUDA hoặc asynchronouspipeline. Đã chuyển thành futureextension và nêu writeconflict.',lane='section_methods',status='Đã sửa')
issue('I27','Soft-min corner thiếu định nghĩa D và có thể âm','sections/methodology.tex','\\mathcal{D}_{\\mathrm{smooth}} = -\\tau_{\\mathrm{soft}}','D_direct/D_pi chưa được định nghĩa tường minh là tổng/trung bình khoảng cách theo mét trước normalize hay distance đã normalize. Code smooth_corner_loss.py dùng meancornerL1, softmin rồi chia diagonal. Unnormalized logsumexp-softmin có thể âm và không có exactzero baseline. Đây không tự động là lỗi optimizer, nhưng gọi distance/C-infinity là chưa chính xác khi D gồm abs. Giữ nguyên công thức/kết quả; cần làm rõ định nghĩa, đơn vị tau và intended behavior.',lane='section_methods',status='Chưa sửa công thức; cần tác giả xác định',action='Định nghĩa D_direct/D_pi, normalization order, tau units và việc cho phép negative loss; không đổi objective hồi tố nếu không chạy lại.')
issue('I28','Hình lưu sẵn còn legend/configuration cũ','sections/results.tex','Trajectories illustrate stable monotonic descent','training_curves PDF vẫn gắnA5=GroupedH32,A6=HybridGT/F24,A8=HybridGT/SGFPN trong khi đối chiếu source đã phát hiện mapping khác. Pareto PDF còn nối frontier qua khác evaluator/GPU. Đã sửa caption nhưng chưa tái tạo hình khi thiếu trace liên kết từng curve với run. Cần dựng lại từ per-run records trước nộp.',lane='pre_submission_readiness',status='Chưa giải quyết nội dung hình; caption đã giới hạn',action='Khôi phục curve→run/checkpoint manifest, dựng lại legend và chia biểu đồ theo protocol/hardware.')
issue('I29','Data flow và index conventions chưa đủ rõ','sections/methodology.tex','A point $\\mathbf{p}_i$ maps to grid coordinate $(u_i, v_i)$ as:','u làx/column704, v lày/row800; shapeCHW dùngrowv,columnu. Methods trình bàyStrata trướcPCGTA dễ hiểu nhầm augmentation trên BEV. Đã nêu trainpoints→PCGTA→rasterize và inferrawpoints→rasterize, cùng tensor indexing. Final heads share stem within eachgroup chứ không share initialgroupblock giữahai nhóm.',lane='section_methods',severity='minor',status='Đã sửa')
issue('I30','RDA được gọi density-adaptive nhưng công thức chỉ dùng range','sections/methodology.tex','Range and Density-Adaptive (RDA) weighting','w(r) chỉ córange, không có localdensityinput. Đã gọi range-adaptive weighting denotedRDA; chưa có ablation cho densityadaptation.',lane='section_methods',severity='minor',status='Đã sửa thuật ngữ')

# Preserve the canonical script output, then supply an explicit expanded source snapshot.
shutil.copy2(A/'meta/full_text.md',A/'meta/phase0_entry_text.md')
snapshot='\n\n'.join('% SOURCE: '+str(p.relative_to(O))+'\n'+p.read_text() for p in sorted(O.rglob('*.tex')))
(A/'meta/full_text.md').write_text('# Original source snapshot (explicit concatenation for exact quote verification)\n\n'+snapshot)
(A/'data/reviewer_issues_with_status.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2))
# Replace only artifacts generated in this run, avoiding generic deterministic fallback claims.
for path in (A/'comments').glob('*.json'):path.write_text('[]\n')
bylane=collections.defaultdict(list)
for i in issues:bylane[i['review_lane']].append(i)
for name,items in bylane.items():(A/'comments'/f'{name}.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
# The method edge table records the actual training/inference graph.
(A/'data/method_interfaces.md').write_text('''Execution: sequential single-agent.

| Producer | Output | Transform / consumer |
|---|---|---|
| Raw LiDAR | xyz/intensity points | ROI filter; PC-GTA before rasterization during training only |
| PC-GTA | augmented points + adjusted labels | Strata-BEV14; raw points bypass this step at inference |
| Strata-BEV14 | B×14×800×704, row=v(y), col=u(x) | ConvNeXt stem and backbone |
| Backbone including Block-SimAM | stride 4/8/16 features | SG-FPN lateral projections and gated top-down fusion |
| SG-FPN | B×32×200×176 | independent class-group stems, shared tasks within each group |
| Grouped heads | classification / offset / size / yaw / quality | train: decoded geometry and detached projection-quality targets; infer: score fusion, filtering, decoding, NMS |
| OGA/IQA/T-SBUW | group objectives and task weights | shared-backbone gradient sum, not a conserved gradient norm |
| Decoding / NMS | class/score/x/y/l/w/orientation modulo π | final BEV footprints; no vertical extent or velocity |

The document contains depth-2 subsections but no depth-3 numbered units; subsection_context_polish is not applicable. Adjacent paragraphs were reviewed directly.
''')

committee={
'editor':('Editor Pre-Screen','Bài có câu hỏi nghiên cứu và mô tả kiến trúc rõ. Rủi ro chính ở abstract/introduction là chuyển BEV/T4 thành triển khai edge/safety và xếp hạng AP khác evaluator (I01,I02,I18). Pitch phù hợp hướng compact perception sau khi thu hẹp phạm vi.'),
'theory':('Theory Contribution Review','Đóng góp được đánh giá như thiết kế hệ thống thực nghiệm, không yêu cầu định lý mới. Những phát biểu cần sửa: conserved gradient budget, continuous directed heading, và surrogate≈exactIoU (I15–I17). Soft-min còn thiếu định nghĩa/normalization (I27).'),
'literature':('Literature Dialogue Review','Related Work đã nhóm theo representation/augmentation/backbone/loss. Không xác minh ngoài kho. Cần phân biệt local TriBand adaptation và complete cited detector, cơ chếPCGTA thực sự bật; không coi prior-art absence là đã được kiểm chứng. I14 đã sửa.'),
'methodology':('Methodology Transparency Review','MUST-FIX: evaluator identity, GPU/report identity, training/selection records, seed confounds, incorrect ablation mapping (I02–I13). Các dữ liệu mô tả được giữ với nhãn nguồn. Không có qualitative coding study; SRQR không áp dụng.'),
'logic':('Logic Chain Review','Chuỗi hợp lệ: point statistics → compact BEV network → local AP + separately identified offline timing. Chuỗi đứt: heterogeneous historical runs → isolated component gains → causal stability/safety claims. I07–I13,I18–I22,I25 xử lý các bước nhảy này; dữ liệu đối chứng vẫn còn thiếu.')}
for name,(title,body) in committee.items():
    (A/'committee'/f'{name}.md').write_text(f'## {title}\n\nExecution: sequential single-agent.\n\n{body}\n')
counts=collections.Counter(i['severity'] for i in issues);score=max(1,9-1.5*counts['major']-.7*counts['moderate']-.2*counts['minor'])
(A/'committee/consensus.md').write_text(f'''## Cross-perspective assessment

Execution: sequential single-agent; this is cross-perspective agreement within one session.

Score under the skill arithmetic: {score:.1f}/10. This is a count-based indicator, not a measured probability of acceptance.

Priority: repair run/evaluator/GPU identity; remove confounded component attribution; then regenerate source-linked figures and conduct matched comparisons. The prose revision is reviewable, but experimental blockers remain.
''')
(A/'summary/overall_assessment.txt').write_text('Execution: sequential single-agent. Major Revision: source identities and descriptions corrected where evidenced; controlled component/seed/evaluator evidence and reconciled figures remain incomplete.\n')

# Full surgical diffs and per-rewrite metadata, kept outside the manuscript.
diffs=[];blocks=[];changed=[]
for old in sorted(O.rglob('*.tex')):
    rel=old.relative_to(O);new=P/rel
    if not new.exists():continue
    a=old.read_text();b=new.read_text()
    if a==b:continue
    changed.append(str(rel));diffs.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='before/'+str(rel),tofile='after/'+str(rel)))
    sm=difflib.SequenceMatcher(None,a.splitlines(),b.splitlines())
    for tag,i,j,k,l in sm.get_opcodes():
        if tag=='equal':continue
        oldtext='\n'.join(a.splitlines()[i:j]);newtext='\n'.join(b.splitlines()[k:l]);
        token_changes=re.findall(r'\\(?:cite|ref|label)\{[^}]+\}',oldtext+newtext)
        blocks.append(f'% COHERENCE [LLM] ({rel}: before {i+1}, after {k+1})\n% Changed: source-aware coherence/provenance correction; see unified diff.\n% Protected: math environments and established labels retained; citation/reference tokens: '+(', '.join(sorted(set(token_changes))) or 'none in this block')+'\n% Meaning-Check: NEEDS-LLM\n% Risk-Flags: terminology-drift\n% Evidence: reviewer_issues_with_status.json, workbooks.json, source-linked CSVs.\n')
(A/'data/source_changes.patch').write_text(''.join(diffs))
(A/'data/rewrite_comments.tex').write_text('\n'.join(blocks))
(A/'data/changed_files.json').write_text(json.dumps(changed,indent=2))

# Vietnamese primary report; canonical renderer output is saved separately later.
intro='''# Rà soát logic, số liệu và chỉnh sửa MobilePIXORNeXt

Ngày 24/10/2026. Execution: **sequential single-agent**. Đã đọc cả hai entry point, toàn bộ 8 section nội dung, 9 tệp bảng và bảng Strata trong Methods; đối chiếu nguồn Excel/CSV và một số code định nghĩa metric/loss. Đây là đánh giá nội bộ, không kiểm chứng tài liệu bên ngoài và không chạy lại huấn luyện/KITTI evaluation.

**Kết luận:** Có mâu thuẫn thực chất trong protocol, GPU, lịch huấn luyện và danh tính run. Một số hàng có AP trung bình đúng nhưng AP từng lớp hoặc tốc độ được ghép từ run khác. Bản thảo đã được sửa thành bản diễn giải trung thực với các nguồn đang có; chưa thể coi ablation đã được kiểm chứng đầy đủ hoặc bản thảo sẵn sàng nộp.

## Các trường hợp “cùng model nhưng hai số”

| Vị trí | Số trong bản cũ | Đối chiếu nguồn và xử lý |
|---|---|---|
| A8 / Baseline loss | 67.38 / 69.46 nhưng 6.94 ms / 144.16 FPS | AP thuộc P1 focal; tốc độ thuộc S32. Tách A8=S32 và A8F=P1. |
| Loss Baseline P1 | 83.92 / 55.48 / 62.74 | Nguồn 82.97 / 56.99 / 62.17. Đã sửa; mAP vẫn 67.38. |
| Q-OGA P11 | 84.35 / 58.12 / 65.13; 0.735M | Nguồn 84.89 / 57.15 / 65.56; 815,743params. Đã sửa; mAP vẫn69.20. |
| OGA+IQA 70.58 và final71.50 | Ghi chỉ thêm attention | 70.58=P4 focal/independent heads; final=no-focal/shared heads, schedule khác. Không phải một model chỉ bật/tắt attention. |
| Final L4 và T4 | 71.50/73.45 ghép60.96ms | L4:71.50/73.45; T4:71.46/73.42 và60.96ms. Đã tách snapshot, chưa quy chênh lệch cho GPU. |
| Forward8.67 và17.88ms | Ghi standalone/pipeline cùngT4 | 8.67 từL4,17.88 từT4. Đã sửa. |
| 66.25% | Ghi PC-GTA H32/F24 | Nguồn O32G16 simplified samplerV2 H16/F32/SG-FPN. Đã sửa nhãn và bỏ +6.03PC-GTA claim. |
| Block-SimAM gain | Methods+0.88, Results+0.92pp | Hai phát biểu cũ mâu thuẫn; cả hai chưa có matched no-attention control trên final model. Đã bỏ causal gain. |

## Phạm vi sửa

Đã thống nhất entry point và abstract, chỉnh luồng train/inference và nghĩa yaw/quality, sửa các ô có nguồn rõ, thay bảng “tiến trình cộng dồn” bằng inventory có run ID, và giới hạn các kết luận nhân quả. Không tạo số thí nghiệm mới. Không sửa phương trình loss hoặc tự chọn một bộ kết quả thay thế khi nguồn chưa đủ.

Có ba hướng xử lý: chỉ trau chuốt câu sẽ để nguyên kết luận sai; giả định mọi run tương đương sẽ che mất confound; bản sửa hiện tại chọn **ghi đúng nguồn và giới hạn diễn giải**, đồng thời giữ các thiếu sót thực nghiệm thành việc cần làm. Bản trước sửa được lưu trong `artifacts/original/`; diff nằm ở `artifacts/data/source_changes.patch`.

## Phát hiện chi tiết

Các mục dưới đây là **[LLM]**: được suy luận từ văn bản và nguồn cục bộ. Các phép kiểm tự động được báo riêng ở cuối. Tọa độ và trích dẫn là bản trước sửa trong `artifacts/original/`.

'''
parts=[intro]
for i in issues:
    parts.append(f"### {i['id']} — {i['title']} ({i['severity']})\n\nVị trí: `{i['source_section']}`.\n\n> {i['quote']}\n\n{i['explanation']}\n\n**Trạng thái:** {i['status']}.\n"+(f"\n**Việc tiếp theo:** {i['action']}\n" if i['action'] else '')+'\n')
parts.append('''## Nguồn đã đối chiếu

- `Result/mobilepixor_vs_ours_seed_summary.xlsx`: Theo seed và Cau hinh; epoch, encoding, GPU, per-class AP, params.
- `Result/benchmark_T4_20261018/Tables_bai_bao_LiDAR_v7_SpeedDetail.xlsx`: T01_Benchmark, B04_Ours_AP_vs_Loss, B05_Protocol_Audit, B10_Speed_Detail, B12_Timing_Scope, T02_Loss_IQA, S02_Augmentation.
- `Result/artifacts/ablation_analysis_20261012/`: runs_summary.csv, paper_01_Backbone.csv, paper_02_Encoding.csv, paper_03_Head_Width.csv, paper_04_Augmentation.csv, paper_05_Neck.csv, paper_06_Loss_IQA.csv.
- `Result/Lidar_result_final.xlsx`: bảng attention stage/neck width và các run lịch sử. Workbook này có một số ô bị Excel đổi thành date serial; không lấy các ô đó làm AP và không ghi ngược thay đổi vào Excel.
- Code local evaluator, T-SBUW và smooth-corner implementation chỉ được đọc để đối chiếu định nghĩa, không thay đổi.

Các workbook là nguồn lưu trữ cục bộ, không thay thế original JSON/config/checkpoint hashes. `workbooks.json` lưu trích xuất cả sheet/địa chỉ ô để kiểm lại; các giá trị date serial được giữ nguyên trong evidence.

## Những việc vẫn cần trước khi nộp

1. Xác nhận checkpoint/config/prediction hashes cho L4/T4 và mỗi hàng; khôi phục manifest curve→run.
2. So sánh trên cùng reference evaluator hoặc chứng minh local/reference equivalence bằng cùng predictions/GT.
3. Chạy lại các đối chứng chỉ đổi một yếu tố cho final architecture; không dùng legacy focal model đại diện final no-focal.
4. Chạy seed study cùng encoding, schedule, budget và selection rule.
5. Dựng lại hình Pareto/loss với nguồn và nhãn đúng; caption hiện đã nêu giới hạn nhưng nội dung vector cũ còn giữ.
6. Làm rõ D_direct/D_pi và normalization của soft-min. Chưa thay loss, vì đổi objective sẽ cần đánh giá lại.

## Kiểm chứng

Đọc `artifacts/data/verification.json` để xem kết quả máy chạy thực tế. Phase0 audit đã chạy nhưng text snapshot ban đầu chỉ chứa entry point; không coi output đó là bằng chứng đã kiểm semantic toàn bộ includes. Các phát hiện chi tiết trên đến từ việc đọc thủ công toàn bộ source và đối chiếu dữ liệu. Snapshot đầy đủ dùng kiểm quote đã được bổ sung bằng phép nối các source trước sửa. Đã chạy riêng check_tables, Methods logic và compile bằng script của skill. Không thực hiện online bibliography check. Môi trường không có `uv`/`python`; dùng `python3` tương đương cho script stdlib. Lệnh thử `python -B .../audit.py --help` trả127; fallback `python3` chạy được.
''')
(A/'summary/review_vi.md').write_text(''.join(parts))
(A/'summary/revision_vi.md').write_text('''# Việc đã sửa và việc còn thiếu

Đã chỉnh source đúng folder được yêu cầu; thư mục `paper/` không được đồng bộ. Bản gốc của phiên này và diff được giữ trong artifacts.

| Nhóm | Đã sửa | Còn cần bằng chứng |
|---|---|---|
| Entry/abstract | Một entry source, một abstract; số deployable/train-only nhất quán | Không |
| Protocol | Local/reference theo từng nhóm, bỏ winner claim | Evaluator equivalence hoặc common reference reevaluation |
| Run identity | Sửa sampler/head/context/epoch/selection labels | Original report/config/checkpoint/prediction hashes |
| Bảng loss | Sửa per-class P1/P11 và params; phân biệt legacy focal | Đối chứng loss trên final no-focal/shared heads |
| Ablation | Inventory có runID; tách P1/S32; bỏ causal gain không đủ control | Matched component runs |
| Seed | Giữ mean/std mô tả, ghi encoding/budget khác | Fixed-recipe seed study |
| Timing | L4/T4 riêng; sửa percentile có nguồn, phân biệt offline boundary | Common hardware/CPU/pinning/postprocess setup |
| Methods | Modulo-pi, surrogate quality, shared gradient, tensor index | Soft-min D/units/normalization clarification |
| Figures | Caption giới hạn, không dùng ảnh vehicle chứng minh VRU AP | Regenerate Pareto/loss legends from run manifests |

Các sửa đổi câu chi tiết ở `artifacts/data/source_changes.patch`; metadata bảo vệ nội dung ở `artifacts/data/rewrite_comments.tex`. Không có số liệu thực nghiệm mới được tạo.
''')
print('Recorded',len(issues),'anchored findings;',dict(counts),'; changed source files',len(changed))
