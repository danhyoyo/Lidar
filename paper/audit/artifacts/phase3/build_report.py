"""Validate source anchors and render the user-requested six-column claim matrix."""
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
PAPER=ROOT/'paper'
sys.path.insert(0,'/home/duyennh/.agents/skills/paper-audit/scripts')
from tex_loader import assemble
from verify_quotes import verify_quotes
from claims import CLAIMS
from report_content import CHAINS, ORPHAN_EXPLANATION, REVISIONS


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


doc=assemble(PAPER/'MobilePIXORNeXt.tex')
assert not doc.missing
claims=verify_quotes(doc.content,CLAIMS)
for x in claims:
    txt=(PAPER/x['source_file']).read_text()
    pos=txt.find(x['quote'])
    assert x['quote_verified'] and pos>=0,x['id']
    x['source_line']=txt[:pos].count('\n')+1
    x['quote_source_verified']=True
    assert x['status'] in {'VERIFIED','METRIC_MISMATCH','OVERCLAIM','UNSUPPORTED'}
    x['evidence_anchor']=[{'type':'figure_or_table','text':x['paper_evidence']},
                          {'type':'section','text':'paper/code_context.md: '+x['context_evidence']}]
    x['missing_evidence']=[] if x['status']=='VERIFIED' else [x['allowed_wording']]
    x['forbidden_wording']=[] if x['status']=='VERIFIED' else [x['quote']]
save('claim_evidence_matrix.json',claims)

# Canonical issue records group repeated wording by the same root cause.
groups=defaultdict(list)
for x in claims:
    if x['status']!='VERIFIED':groups[x['root_cause_key']].append(x)
issues=[]
for key,rows in groups.items():
    x=rows[0]
    orphan=any(row['orphan_contribution'] for row in rows)
    issues.append({'title':f"{x['status']}: {key}",'quote':x['quote'],
        'explanation':'\n'.join(f"{r['id']}: {r['paper_evidence']} Context: {r['context_evidence']}" for r in rows),
        'comment_type':'claim_accuracy','severity':'major' if orphan or any(r['status'] in {'OVERCLAIM','UNSUPPORTED'} for r in rows) else 'moderate',
        'confidence':'high' if key!='prior-art-claim-not-verified' else 'medium',
        'source_kind':'llm','source_section':x['source_file'],'source_line':x['source_line'],
        'root_cause_key':key,'review_lane':'claims_vs_evidence','evidence_anchor':x['evidence_anchor'],
        'claim_strength':x['claim_strength'],'missing_evidence':[r['allowed_wording'] for r in rows],
        'allowed_wording':x['allowed_wording'],'forbidden_wording':[r['quote'] for r in rows],
        'gate_blocker':orphan or key in {'e03-headline','pr-table-protocol-mismatch','m03','loss-curve-overclaim'},
        'quote_verified':True,'claim_ids':[r['id'] for r in rows],
        'user_gate_label':'CRITICAL BLOCKER' if orphan else None})
save('issues.json',issues)

assets=[]
for name in sorted(set(re.findall(r'\\paperfigure(?:\[[^\]]*\])?\{([^}]+\.pdf)\}',doc.content))):
    p=PAPER/'figures'/name
    assert p.exists(),name
    assets.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                   'extracted_text':name.replace('.pdf','.txt')})
assert len(assets)==10,len(assets)
for x in assets:assert (OUT/x['extracted_text']).exists(),x
save('figure_assets.json',assets)

counts=Counter(x['status'] for x in claims)


def cell(value):
    return str(value).replace('|','&#124;').replace('\n','<br>')


def source_link(x):
    return f"[{x['source_file']}:{x['source_line']}]({PAPER/x['source_file']}:{x['source_line']})"


table=['| STT | Vị trí & Câu tuyên bố gốc (Claim in Abstract/Intro/Method/Conclusion) | Bằng chứng định lượng trong Bảng/Hình của Paper | Số liệu gốc trong `Paper/code_context.md` | Trạng thái | Cách sửa câu văn hoặc bổ sung số liệu |',
       '|---|---|---|---|---|---|']
for x in claims:
    location=source_link(x)+'<br>`'+cell(x['quote'])+'`'
    if x['related_locations']:location+='<br>Cùng nhóm: '+cell(x['related_locations'])
    status='`'+x['status']+'`'
    if x['orphan_contribution']:status+='<br>**ORPHAN_CONTRIBUTION — CRITICAL BLOCKER**'
    table.append('| '+' | '.join([x['id'],location,cell(x['paper_evidence']),cell(x['context_evidence']),status,cell(x['allowed_wording'])])+' |')

chains=['| Đóng góp | Introduction | Methodology | Ablation / evidence row | Conclusion | Kết luận chuỗi |',
        '|---|---|---|---|---|---|']
for number,name,intro,method,ablation,conclusion,decision in CHAINS:
    chains.append('| '+' | '.join(map(cell,[number+' — '+name,intro,method,ablation,conclusion,decision]))+' |')
arithmetic=json.loads((OUT/'arithmetic_checks.json').read_text())
ledger=['| Phép đối soát | Phép trừ đúng theo số đang hiển thị | Đơn vị | Kết luận arithmetic |',
        '|---|---|---|---|']
for row in arithmetic['subtractions']:
    ledger.append(f"| {row['name']} | {row['a']} − ({row['b']}) = {row['exact']} | {row['unit']} | Khớp exact Decimal |")

header=f"""# Phase 3 — Adversarial Claims-vs-Evidence Reconciliation

**Bản thảo:** MobilePIXORNeXt, target AJSE. **Ngày:** 2026-10-22. **Gate:** `FAIL_CLAIM_GATE / MAJOR_REVISION`. Hai chuỗi đóng góp có **CRITICAL BLOCKER** theo quy tắc bốn mắt xích của yêu cầu này: **T-SBUW (P9/C35)** và **Q-OGA (P10/C39)**.

Ma trận có **{len(claims)} dòng**: **{counts['VERIFIED']} VERIFIED**, **{counts['METRIC_MISMATCH']} METRIC_MISMATCH**, **{counts['OVERCLAIM']} OVERCLAIM**, **{counts['UNSUPPORTED']} UNSUPPORTED**. Đây là số dòng đối soát, không phải số lỗi độc lập; những cách viết lặp lại được gom theo root cause trong JSON. Có {len(issues)} nhóm cần xử lý, bao gồm các lỗi tài liệu nguồn, đơn vị và các giới hạn bằng chứng ở nhiều mức độ.

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
"""

quant=r"""
## 4. Sổ phép tính và sự khác nhau giữa các protocol

### 4.1 AP deltas và ablation deltas

`[Script]` dùng `Decimal` cho44 phép trừ từ số đang hiển thị. Cột “khớp” chỉ xác nhận số học, không chứng nhận các thí nghiệm có cùng config, có statistical significance hoặc cơ chế nhân quả được kiểm soát. Gains đang viết bằng `%` nên đổi sang `percentage points`, giữ `%` cho AP level và relative parameter/payload reduction.
"""

protocol=r"""
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
"""

footer=r"""
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
"""

report='\n\n'.join([header.strip(),'\n'.join(table),quant.strip(),'\n'.join(ledger),protocol.strip(),
                      '\n'.join(chains),ORPHAN_EXPLANATION.strip(),REVISIONS.strip(),footer.strip()])+'\n'
assert report.count('```')%2==0
latex_examples=re.findall(r'```latex\n(.*?)\n```',report,re.S)
assert len(latex_examples)==7
for snippet in latex_examples:
    assert not re.search(r'(?<!\\)%',snippet), 'Unescaped percent sign in LaTeX example'
    assert snippet.count('{')==snippet.count('}'), 'Unbalanced braces in LaTeX example'
assert len(table)-2==len(claims)
for line in table:
    assert line.count('|')==7, line[:100]
before=json.loads((OUT/'source_hashes_before.json').read_text())
changed=[path for path,digest in before.items() if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest]
assert not changed,changed
save('source_integrity.json',{'checked_files':len(before),'changed_files':changed})
abstract_sample=re.search(r'### 6\.1 Abstract.*?```latex\n(.*?)\n```',REVISIONS,re.S).group(1)
abstract_words=len(abstract_sample.split())
assert 150<=abstract_words<=250,abstract_words
for path in re.findall(r'\]\(([^)]+)\)',report):
    if '://' in path:continue
    p=re.sub(r':\d+$','',path)
    if p.endswith('verification.json'):continue # written immediately below
    target=Path(p) if p.startswith('/') else PAPER/'audit'/p
    assert target.exists(),path
verify={'matrix_rows':len(claims),'status_counts':dict(counts),'canonical_issue_groups':len(issues),
        'quotes_verified':sum(x['quote_verified'] for x in claims),'source_lines_verified':len(claims),
        'matrix_columns':6,'orphan_critical_blockers':[x['id'] for x in claims if x['orphan_contribution']],
        'contribution_chains':len(CHAINS),'figure_assets_read':len(assets),'arithmetic_subtractions':len(arithmetic['subtractions']),
        'latex_examples':len(latex_examples),'latex_percent_and_brace_checks':True,
        'abstract_example_words_whitespace':abstract_words,'sources_checked':len(before),'source_changes':changed,
        'source_edited':False,'retraining_or_benchmark_executed':False,'external_literature_verified':False}
save('verification.json',verify)
target=PAPER/'audit/04_claims_vs_evidence_matrix.md'
target.write_text(report)
print(json.dumps(verify,ensure_ascii=False,indent=2))
print('Report:',target)
