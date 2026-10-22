"""Render and validate the explicitly requested Phase 2 report, without source edits."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PAPER = ROOT / 'paper'
sys.path.insert(0, '/home/duyennh/.agents/skills/paper-audit/scripts')
from tex_loader import assemble
from verify_quotes import verify_quotes
from review_issues import ISSUES
from report_tables import SYMBOLS, EQUATIONS, TENSORS, REPRO, ABLATIONS, MOTIVATION, CONTEXT


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


doc = assemble(PAPER / 'MobilePIXORNeXt.tex')
assert not doc.missing, doc.missing
issues = verify_quotes(doc.content, ISSUES)
for issue in issues:
    assert issue['quote_verified'], issue['id']
    txt = (PAPER / issue['source_file']).read_text()
    idx = txt.find(issue['quote'])
    assert idx >= 0, issue['id']
    line = txt[:idx].count('\n') + 1
    assert line == issue['source_line'], (issue['id'], line)
    issue['quote_source_verified'] = True
    assert issue['comment_type'] in {'methodology','claim_accuracy','presentation','missing_information'}
    assert issue['severity'] in {'major','moderate','minor'}
    assert issue['source_kind'] in {'script','llm'}
    if issue['display_severity'] == 'MAJOR':
        assert issue['latex_example']
save('issues.json', issues)

scores = [
    (1, 'Ký hiệu và convention', 3, 'Ký hiệu phần lớn được định nghĩa; R4/ray và level/stride cần sửa; corner order thiếu.'),
    (1, 'Chiều tensor và khớp triển khai final', 3, 'Numeric shape chain khớp; SG-FPN operator chưa đúng; context có nhiều snapshot cũ.'),
    (1, 'Loss, trọng số và numerical rigor', 2, 'Adaptive/fixed weights đã có; primitive reductions, EMA lifecycle và Q-OGA còn thiếu.'),
    (1, 'Pseudocode và inference contract', 2, 'Sai thứ tự peak/joint score tạo output khác trên counterexample cụ thể.'),
    (1, 'Động lực hình học/toán học', 3, 'Moments,projection/corner có động lực; gate/attention và các heuristic cần evidence đúng mức.'),
    (1, 'Mức đóng góp lý thuyết', 3, 'Có engineering/objective integration; không yêu cầu theorem, chưa xác minh novelty toàn literature.'),
    (2, 'Reproducibility checklist', 3, 'Geometry/optimizer/augmentation chính rõ; recipe/ownership/training GPU còn thiếu.'),
    (2, 'Ablation và kiểm soát thực nghiệm', 2, 'Có nhiều rows và factorial loss; thiếu isolated weighting/geometry và manifests cho SimAM.'),
    (2, 'Complexity, memory và latency', 3, 'Params/memory/pipeline timing tốt; standalone provenance và CPU/training hardware chưa đủ; thiếu MAC count.'),
]
for _,_,score,_ in scores:
    assert 1 <= score <= 5
save('scores.json', [{'reviewer':r,'criterion':c,'score':s,'reason':why} for r,c,s,why in scores])


def score_table(reviewer):
    result = '| Tiêu chí | Điểm /5 | Căn cứ |\n|---|---:|---|\n'
    return result + '\n'.join(f'| {c} | {s} | {why} |' for r,c,s,why in scores if r == reviewer)


def render_issue(issue):
    x = issue
    file = PAPER / x['source_file']
    result = [f"### {x['id']} — [{x['source_kind'].upper()}] {x['display_severity']} — {x['title']}",
              f"- **Vị trí:** [{x['source_file']}:{x['source_line']}]({file}:{x['source_line']}).",
              f"- **Schema:** `comment_type: {x['comment_type']}`; `severity: {x['severity']}`; `source_kind: {x['source_kind']}`; `confidence: {x['confidence']}`; `quote_verified: true`.",
              f"- **Theo dõi:** `root_cause_key: {x['root_cause_key']}`; `review_lane: {x['review_lane']}`; `gate_blocker: {str(x['gate_blocker']).lower()}`."]
    if x['related_previous_issues']:
        result.append('- **Liên hệ lần trước:** ' + ', '.join(x['related_previous_issues']) + '; đây là phân tích sâu tiếp nối cùng root cause, không tính là lỗi mới độc lập của Phase 1.')
    result += ['', '**Trích nguyên văn (`quote`):**', '', '> ' + x['quote'], '',
               '**Phân tích (`explanation`):** ' + x['explanation'], '',
               '**Bằng chứng (`evidence_anchor`):** ' + '; '.join('`' + a['text'] + '`' for a in x['evidence_anchor']) + '.', '',
               '**Hướng sửa:** ' + x['fix']]
    if x['latex_example']:
        result += ['', '**LaTeX sửa mẫu — đề xuất trong báo cáo, chưa áp dụng vào manuscript:**', '', '```latex', x['latex_example'], '```']
    return '\n'.join(result)


header = r"""# Phase 2 — Specialist Deep Review: theory, method và khả năng tái lập

**Bản thảo:** `Paper/MobilePIXORNeXt.tex` và các file được include; target Arabian Journal for Science and Engineering (AJSE). **Ngày:** 2026-10-22. **Kết luận:** `MAJOR_REVISION` trước submission; **12 MAJOR, 3 MINOR, 0 CRITICAL được chứng minh**. Sáu điểm cần đóng trước khi coi phương pháp/kết luận chính đủ vững: T04–T06, M03, M05–M06. Không có bằng chứng để kết luận mọi kết quả vô hiệu hoặc phải loại ngay bài.

Đây là một lượt audit tuần tự dưới **hai vai reviewer chuyên môn**, không phải hai phản biện độc lập hay quyết định của AJSE. Phạm vi theo yêu cầu là Methods/Experiments, đọc kèm Results, bảng và code để đối chiếu; không lặp toàn bộ Phase 0.5/1 và không tra cứu external novelty. Toàn bộ nhận định chuyên môn mang nhãn `[LLM]`; phép đếm/check chạy thật mang nhãn `[Script]` ở phần bằng chứng. Theo `ISSUE_SCHEMA.md`, JSON giữ severity `major|moderate|minor`; để đáp ứng mức hiển thị yêu cầu, `major` và `moderate` đều hiển thị **MAJOR**, `minor` hiển thị **MINOR**. Canonical counts: 8 major, 4 moderate, 3 minor.

Đã áp dụng skill [paper-audit](/home/duyennh/.agents/skills/paper-audit/SKILL.md), [DEEP_REVIEW_CRITERIA.md](/home/duyennh/.agents/skills/paper-audit/references/DEEP_REVIEW_CRITERIA.md), [ISSUE_SCHEMA.md](/home/duyennh/.agents/skills/paper-audit/references/ISSUE_SCHEMA.md), kèm hướng dẫn theory/notation/methodology và method-interface. Chỉ ghi report/artifacts; **không chỉnh `.tex`, config hay code**. LaTeX dưới đây là mẫu sửa để tác giả review; các dữ kiện cần run/log mới được đánh dấu rõ, không điền AP giả.

Thang điểm: **1** = không đủ đặc tả/cơ sở; **2** = thiếu sót lớn ảnh hưởng tái lập/kết luận; **3** = nền tảng rõ nhưng cần major revision; **4** = phần lớn đầy đủ, thiếu sót nhỏ; **5** = đầy đủ và được kiểm chứng trong phạm vi bằng chứng hiện có. Điểm là reviewer judgment, không phải xác suất acceptance hay công thức từ số lỗi.

## 1. Nguồn, phiên bản và vị trí trích dẫn

Entry thực tế là [MobilePIXORNeXt.tex](../MobilePIXORNeXt.tex); không có `Paper/main.tex` trong snapshot này. Proposed Method được include ở entry L70, Experiments L72, Results L74. Mỗi finding dẫn **file con và dòng thật**, không gán số dòng của file ghép thành dòng của `main.tex`. `Paper` là symlink tới `paper`; các đường dẫn hai cách viết cùng trỏ một nguồn.

Đã đối chiếu trực tiếp [code_context.md](../code_context.md) với [resolved final config](../../Result/iqa_comparison_20261019/config.inference.json), [canonical best-AP benchmark log](../../Result/benchmark_T4_20261018/sources/ours_best_ap.json), architecture/loss/target/sampler/postprocess và evaluator trong `Code/`. Log final có detector 753.569 tham số, checkpoint epoch 130 của schedule 160 epoch; identity ghi stages 3/4/2, F32, H32, OGA+IQA, SimAM c3/c4/c5, seed 42. Các giá trị trong log là **bằng chứng được đọc lại**, không phải audit đã tự huấn luyện/chạy benchmark lại model.

### 1.1 Các bất đồng của context cần xử lý trước khi nói “khớp 100%”
"""

loss_summary = r"""
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
"""

efficiency = r"""
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
"""

ending = r"""
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
"""

report = [header, CONTEXT, '## 2. Reviewer 1 — Theoretical Contribution & Mathematical Rigor', score_table(1),
          '### 2.1 Bảng ký hiệu theo các tiểu mục Methodology', SYMBOLS,
          '### 2.2 Hợp đồng chiều tensor', TENSORS,
          '### 2.3 Kiểm từng nhóm phương trình', EQUATIONS, loss_summary,
          '### 2.6 Động lực toán học và các giới hạn cần giữ khi diễn giải', MOTIVATION,
          '### 2.7 Findings của Reviewer 1 theo ISSUE_SCHEMA',
          '\n\n'.join(render_issue(x) for x in issues if x['reviewer']==1),
          '## 3. Reviewer 2 — Methodology Transparency & Experimental Reproducibility', score_table(2),
          '### 3.1 Checklist đối chiếu Experimental Setup',
          'Trong bảng này, L là dòng trong `sections/experiments.tex`, trừ khi ghi Methods hoặc file khác.', REPRO,
          '### 3.2 Module–ablation matrix', ABLATIONS,
          '### 3.3 Những gì chưa thể suy từ bảng ablation',
          'Loss table no-attention `.721M` và final `.753569M` không thể khác chỉ do parameter-free SimAM. Counter giữ đúng final architecture cho cả off/on đều ra753569; bỏ hai IQAheads còn734943,delta18626. Đây là yêu cầu truy nguyên cấu hình lịch sử, chưa phải bằng chứng A9 đã dùng chính xác2/4/2blocks. Cùng split không thay cho cùng schedule/selection; các50-epoch curves không chứng minh mọiablation theo budget final160epochs.',
          efficiency, '### 3.5 Findings của Reviewer 2 theo ISSUE_SCHEMA',
          '\n\n'.join(render_issue(x) for x in issues if x['reviewer']==2), ending]
result = '\n\n'.join(part.strip() for part in report) + '\n'

before = json.loads((OUT/'source_hashes_before.json').read_text())
changed = [path for path,digest in before.items()
           if not (ROOT/path).exists() or hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest]
assert not changed, changed
save('source_integrity.json', {'checked_files':len(before),'changed_files':changed})
equations = json.loads((OUT/'equation_inventory.json').read_text())
missing_labels = [x['label'] for x in equations['numbered_blocks'] if '`'+x['label']+'`' not in EQUATIONS]
assert not missing_labels, missing_labels
verification = {'issues':len(issues),'canonical_severities':dict(Counter(x['severity'] for x in issues)),
                'display_severities':dict(Counter(x['display_severity'] for x in issues)),
                'quotes_verified':sum(x['quote_verified'] for x in issues),
                'source_file_and_line_verified':sum(x['quote_source_verified'] for x in issues),
                'major_latex_examples':sum(bool(x['latex_example']) for x in issues if x['display_severity']=='MAJOR'),
                'gate_blockers':[x['id'] for x in issues if x['gate_blocker']],
                'scores_in_range':True,'equation_labels_covered':len(equations['numbered_blocks']),
                'source_files_checked':len(before),'source_files_changed':changed,
                'manuscript_edited':False,'model_executed':False,'new_training_or_benchmark':False}
save('verification.json', verification)
(PAPER/'audit/02_theory_and_method_review.md').write_text(result)
print(json.dumps(verification, ensure_ascii=False, indent=2))
print('Report:', PAPER/'audit/02_theory_and_method_review.md')
print('Report lines:', len(result.splitlines()))
