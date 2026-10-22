# Phase 0.5 + Phase 1 — AJSE desk screening and automated audit

**Phán quyết: `BORDERLINE_DESK_REJECT`** — tương ứng **Conditional Pass**, chưa nên nộp nguyên trạng. Không có cơ sở cho `IMMEDIATE_DESK_REJECT` từ phạm vi kiểm tra này.

Ngày audit: 2026-10-22T14:22:13+07:00. Target: **Arabian Journal for Science and Engineering (AJSE), Research Article / Computer Science and Engineering**.

## 1. Phạm vi, đường dẫn và khả năng tái lập

- Entrypoint được khóa là `paper/MobilePIXORNeXt.tex`. `Paper/` là symlink tới `paper/`; ký tự `@` là cú pháp đính kèm của yêu cầu, không thuộc tên file. **Không tồn tại `Paper/main.tex`**. Báo cáo dùng số dòng thật trong entrypoint và đường dẫn include đến section/table; không bịa số dòng main.tex.
- `[LLM]`: chỉ dùng Title, Abstract, Keywords, Introduction, Declarations, các heading và hình/bảng/caption trong vai EIC. `[Script]`: checker quét toàn bộ cây include. Đây không phải Phase 2/full peer review; không thẩm định phương trình, thực nghiệm mới hoặc prior-art đầy đủ.
- Hình thức thực hiện: **sequential single-agent**. Các tiến trình checker được chạy đồng thời; không có hội đồng hay subagent độc lập.
- Chỉ tạo báo cáo/artifacts trong `paper/audit/`; không sửa `.tex`, `.bib`, `.pdf` của bản thảo. Không bật `--online` hoặc `--literature-search`; không gửi bản thảo đến API bên ngoài. Chỉ tra cứu trang quy định/scope công khai của AJSE.
- Manifest SHA-256 của snapshot nguồn: `a2a629d913a7637acce559808e944669bdd0ea5cc0d8e84f7ee93b13a79bdf1c`. Danh sách hash từng file: [source_hashes_before.json](artifacts/source_hashes_before.json).
- Đã đọc `paper-audit/SKILL.md`, `agents/editor_in_chief_agent.md`, `references/ISSUE_SCHEMA.md`, các review criteria/checklist, consolidation và presubmission rules; đọc thêm hướng dẫn `latex-paper-en` cho các checker hỗ trợ. Không thay đổi skill cài đặt.

### Lệnh yêu cầu và thay thế môi trường

Lệnh `uv run python -B .../paper-audit/scripts/audit.py .../paper/MobilePIXORNeXt.tex --mode self-check --lang en` đã được thử: **exit 127**, `uv` không có trong môi trường. Dùng Python chuẩn để chạy chính checker đó, không cài lại dependency hay tạo main.tex giả:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B /home/duyennh/.agents/skills/paper-audit/scripts/audit.py paper/MobilePIXORNeXt.tex --mode self-check --lang en --venue ajse --format json
```

`self-check` được skill chuyển về `quick-audit`. `ajse` không có profile kiểm tra venue riêng trong bộ script này: không được diễn giải exit 0 thành chứng nhận đạt AJSE. Các giới hạn riêng của AJSE được kiểm tra bổ sung có dẫn nguồn.

| Checker | Exit | Kết quả thô / giới hạn |
|---|---:|---|
| `audit_self_check` | 0 | 536 bản ghi issue thô; có cả header/status/help text do lỗi ingest, không phải 536 lỗi bản thảo. |
| `references` | 0 | 31 cảnh báo: 19 unreferenced equation labels + 12 forward references; không có undefined ref. |
| `format` | 0 | 341 cảnh báo ChkTeX; status WARNING, không phải 341 lỗi đã xác nhận. |
| `deai` | 2 | Lần đầu dùng --json không được hỗ trợ; lỗi CLI đã được giữ trong log và chạy lại đúng cú pháp. |
| `tables` | 0 | 10 bảng; 0 structural issues theo checker. |
| `figures` | 1 | Báo thiếu figures/#2: false positive trên tham số macro; chưa nhận ra 10 lời gọi paperfigure thực tế. |
| `deai_corrected` | 0 | --analyze: 16 trace thô, max 2.67% trace/source-line; không phải lexical density hoặc xác suất AI. |

Toàn bộ lệnh, stdout/stderr và exit code được giữ ở [commands.json](artifacts/commands.json), [requested_uv_command.json](artifacts/requested_uv_command.json), [deai_corrected.command.json](artifacts/deai_corrected.command.json). Kiểm tra bổ sung có thể chạy lại bằng:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B paper/audit/artifacts/supplementary_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 -B paper/audit/artifacts/build_report.py
```

## 2. Phase 0.5 — EIC desk-reject screening

### Quy chuẩn AJSE và front matter

Các mốc 150–250 từ, 4–6 keywords, tự giải thích viết tắt và heading khai báo được đối chiếu từ [AJSE submission guidelines](https://link.springer.com/journal/13369/submission-guidelines), truy cập trong phiên audit này. Springer khuyến nghị template hiện hành; việc dùng `svjour3` không tự động là lỗi chí mạng.

| Hạng mục | Kết quả | Đánh giá |
|---|---|---|
| Abstract word count | **223 từ** theo whitespace sau chuẩn hóa math/escaped percent; phép đếm lexical thay thế **222** | **PASS**, cả hai trong 150–250 |
| Citation trong abstract | **0** lệnh cite | **PASS** |
| Viết tắt | LiDAR/BEV/mAP có gloss; CPU/GPU/AP cần mở rộng, R40 cần giải thích | **NEEDS FIX — E01** |
| Keywords | **5** mục, phân cách bằng `\and` | **PASS**, trong 4–6 |
| Declarations hiện diện | Có `\section*{Declarations}`; đủ các nhóm khai báo liên quan đã nêu | **PASS về hiện diện**; tên heading cần đồng bộ, E02 |
| Funding / interests / ethics / consent | Có phát biểu cụ thể, không chỉ heading trống | **PASS về hiện diện**; rà consent wording, E09 |
| Data / code / contributions | Có cả ba | **PASS về hiện diện**, không yêu cầu tự phát hành repo ngay chỉ vì audit |
| Abstract năm thành phần | Bối cảnh: edge budget; mục tiêu: compact detector; phương pháp; kết quả số; kết luận ở câu cuối | **PASS qua đọc EIC**; raw script báo thiếu cả năm là false positive |

Đếm abstract giữ từ ghép có dấu gạch nối làm một từ. Khác biệt 223/222 là quy ước token hóa, không ảnh hưởng phán quyết độ dài. Toàn bộ abstract nằm trên **sections/abstract.tex:1**, được include tại **MobilePIXORNeXt.tex:53**.

### Scope, engineering relevance và novelty hook

**Scope: phù hợp.** AJSE nêu Computer Science and Engineering trong [aims and scope](https://link.springer.com/journal/13369/aims-and-scope). Introduction liên hệ LiDAR với autonomous navigation, mobile robotics, giới hạn latency dưới 100 ms/scan, băng thông bộ nhớ và ngân sách edge. Giá trị kỹ thuật được lượng hóa bằng số tham số, AP và độ trễ toàn pipeline; không phải một bài chỉ bàn kiến trúc trừu tượng.

**Novelty hook: đủ động lực để cân nhắc review, chưa xác nhận novelty tuyệt đối.** Introduction:7–9 liên kết quantization/subcell geometry, density/augmentation, context và ranking; câu hỏi nghiên cứu ở dòng 9 rõ. Các contribution:15–19 là systems contribution có chuỗi lý do kỹ thuật, không đủ bằng chứng để kết luận chỉ ghép A+B cơ học. Tuy vậy cần phân biệt thành phần kế thừa (ConvNeXt, SimAM, MGIoU có citation) với thiết kế/tích hợp cụ thể và bằng chứng ablation. Không dùng desk screen này để tuyên bố prior art chưa từng có.

**Fatal flaw screen:** chưa phát hiện đạo văn, gian lận, thiếu toàn bộ evaluation hay bất khả thi phương pháp từ các phần được phép đọc. Những điều này cũng chưa được kiểm chứng để loại trừ tuyệt đối. Điểm rủi ro có thật là headline comparison và nguồn GPU của các bảng, E03/E07; ngôn ngữ nhìn chung đủ rõ để review.

### Visual first impression

- Fig. 1 có caption giàu thông tin nhưng không hoàn toàn khớp mức chi tiết/kiểu đường trong artwork; quan trọng hơn, hình ở trang 4 và caption ở trang 5. Xem E05/E06.
- Table 3 tự giải thích được metric, đơn vị, class, difficulty và local/reference distinction trong caption/note. Table 4 ghi batch size, precision, warmup và phạm vi E2E rõ. Điểm này được ghi nhận là đạt, không gán thiếu caption hàng loạt.
- Table 6/7 đọc độc lập được nhưng metadata GPU không nhất quán khi đặt cạnh nhau. Table 10 thiếu đơn vị ở header/caption. Fig. 8 có câu kết luận rộng hơn dữ liệu Cyclist của Table 3. Xem E07/E08/E10.
- Đã xem PDF hiện có, không recompile: trang 1, 4–5, 16, 19–20, 23 được render thành ảnh trong artifacts. Không suy ra compile-pass từ việc có PDF; PDF parser phát một warning số PDF, chưa có bằng chứng đó là lỗi nội dung của tác giả.

### Điểm và phán quyết

| Dimension | Điểm / 10 | Trọng số |
|---|---:|---:|
| Pitch quality | 7.5 | 30% |
| Venue fit | 9.0 | 20% |
| Fatal flaw detection / độ yên tâm sau screening | 5.5 | 30% |
| Presentation baseline | 5.5 | 20% |
| **Weighted** | **6.80** | **100%** |

Theo ngưỡng skill, 5.0 ≤ score < 7.0 là Conditional Pass. Ánh xạ ba nhãn bạn yêu cầu: Pass to Review → PASS_TO_REVIEW; Conditional Pass → BORDERLINE_DESK_REJECT; Desk Reject → IMMEDIATE_DESK_REJECT. Điểm là phán đoán EIC mô phỏng, không phải xác suất bị AJSE từ chối.

Bài toán perception nhỏ gọn có động lực kỹ thuật, câu hỏi nghiên cứu và kết quả định lượng rõ, phù hợp Computer Science and Engineering của AJSE. Chưa có lỗi chí mạng được chứng minh ở mức desk screen. Tuy nhiên provenance GPU của các bảng chưa thống nhất, headline so sánh chưa thể hiện ranh giới evaluator, abstract còn viết tắt chưa định nghĩa và Fig. 1 bị tách khỏi caption. Cần xử lý các điểm này trước khi chuyển hồ sơ đi review.

## 3. Phase 1 — dữ liệu định lượng đã làm sạch

### Citation, reference, label, bảng và asset

| Chỉ số | Kết quả kiểm tra |
|---|---:|
| Nhãn duy nhất | 73 |
| Lượt ref/eqref/autoref/cref/pageref | 63 |
| Undefined references / duplicate labels | **0 / 0** |
| Nhãn không được viện dẫn, tất cả loại | **31** |
| Trong đó nhãn phương trình | **19** |
| Trong đó nhãn section | **12** — không coi là lỗi |
| Forward references | **12** — hợp lệ trong LaTeX, không coi là lỗi |
| Bib entries / unique cited keys | **20 / 20** |
| Undefined cite keys / uncited bib entries / duplicate bib keys | **0 / 0 / 0** |
| Bảng được checker nhận diện / structural issues | **10 / 0** |
| Lời gọi paperfigure thực tế / asset thiếu | **10 / 0** |

Các số 31 của raw references và 31 unused-label inventory là hai tập khác nhau: raw checker = 19 equations + 12 forward refs; inventory bổ sung = 19 equations + 12 section labels. Không cộng hai số này thành 62 lỗi.

### AI-tone density theo từng mục

Hai đại lượng khác nhau được báo riêng. **Lexical density** = 100 × số token khớp 14 nhóm từ trong PRE_SUBMISSION_RULES / số từ tiếng Anh trong prose đã lọc. **Raw trace score** của deai_check = 100 × trace_count / số dòng source của section đã assemble; không phải phần trăm từ hoặc xác suất AI viết. Các số 0 không chứng minh văn bản do con người hay AI tạo.

| Section | Từ prose (denominator lexical) | Lexicon hits | Lexical density | Raw traces / source lines | Raw trace score |
|---|---:|---:|---:|---:|---:|
| abstract | 209 | 0 | 0.0000% | 0 / 4 | 0.00% |
| introduction | 946 | 1 | 0.1057% | 0 / 24 | 0.00% |
| related_work | 696 | 0 | 0.0000% | 0 / 26 | 0.00% |
| methodology | 2507 | 0 | 0.0000% | 6 / 468 | 1.28% |
| experiments | 700 | 0 | 0.0000% | 0 / 87 | 0.00% |
| results | 1750 | 0 | 0.0000% | 10 / 374 | 2.67% |
| conclusion | 303 | 0 | 0.0000% | 0 / 19 | 0.00% |
| declarations | 251 | 0 | 0.0000% | N/A | N/A |

Lexicon chỉ khớp **superior** tại Introduction:9 trong prose đã lọc, có nêu ngữ cảnh mục tiêu throughput. Không có căn cứ đưa một từ đơn lẻ thành lỗi AI-tone. Caption/tables, math, citation, macro, headings và thuật toán được loại khỏi denominator lexical; vì vậy số từ này không dùng để kiểm tra giới hạn abstract.

**Giới hạn parser được phát hiện:** adapter `_iter_visible_lines` của deai hiểu `\\[5pt]` trong macro helper là mở display math `\[`, làm mất phần đầu tài liệu khi quét từ vựng toàn văn. Supplementary adapter bỏ macro definitions, nhận đúng delimiter và tính lại density theo source. 16 raw traces vẫn được giữ nguyên, nhưng phần lớn là cảnh báo `\item \textit`, `\par\vspace`, caption/figure tense hoặc mô tả kỹ thuật thì hiện tại; không nâng thành 16 lỗi ngôn ngữ.

Dữ liệu đầy đủ: [lexical_density.json](artifacts/lexical_density.json), [deai_api_analysis.json](artifacts/deai_api_analysis.json), [deai_corrected.stdout.txt](artifacts/deai_corrected.stdout.txt). Không chạy chỉnh văn phong và không dùng thống kê để kết luận tác quyền AI.

### Định dạng số/đơn vị và ChkTeX

- ChkTeX sinh **341 warning**: spacing 300; other 18; punctuation 11; parentheses 10; math 2. Chưa có warning nào được phép tự chuyển thành lỗi nghiêm trọng mà không đọc ngữ cảnh.
- Scanner số dính unit nhận 11 match, tất cả nằm ở tham số độ dài TeX như `32mm`, `60mm`, `75mm`; đây là cú pháp chiều dài hợp lệ, **0 lỗi number–unit adjacency được xác nhận** trong prose từ phép kiểm tra giới hạn này.
- Lỗi đơn vị có ý nghĩa là **điểm phần trăm vs %** (E04) và **thiếu đơn vị cột trong Table 10** (E10). Kiểm tra cấu trúc bảng không phát hiện hai vấn đề ngữ nghĩa này.
- Một số warning `\log`, `\exp`, `\det` xuất hiện ở tên subscript, nhãn score hoặc nội dung mô tả; không sửa máy móc thành operator. `7x7`/`2x2` trong mô tả macro/artwork và khoảng trắng lệnh không được coi là fatal flaw.

## 4. Issue bundle — các phát hiện được giữ lại

Mỗi mục có đủ trường bắt buộc của ISSUE_SCHEMA: title, quote, explanation, comment_type, severity, source_kind. `severity` canonical dùng major/moderate/minor; nhãn hiển thị theo yêu cầu: MAJOR cho major/moderate, MINOR cho minor; chỉ dùng CRITICAL khi có lỗi chí mạng được chứng minh. **Không có CRITICAL được xác nhận.** `gate_blocker` chỉ đánh dấu điểm phải làm rõ trước khi khẳng định readiness; không đồng nghĩa mọi MAJOR là immediate desk reject.

Có **12 issue đã hợp nhất theo nguyên nhân**, gồm 2 `[Script]` và 10 `[LLM]`; một issue chứa toàn bộ 19 nhãn phương trình để tránh nhân số lỗi. Quote verification: **12/12** bằng hàm từ `paper-audit/scripts/verify_quotes.py`. Bundle machine-readable: [issues.json](artifacts/issues.json).

### E01 — [LLM] [MAJOR] Abstract còn các viết tắt chưa tự giải thích

- **title:** Abstract còn các viết tắt chưa tự giải thích
- **quote:** `NumPy/Numba CPU rasterizer`
- **explanation:** CPU và GPU chưa được viết đầy đủ; AP xuất hiện trong BEV AP trước khi được giải thích riêng. Cụm mean Average Precision (mAP) không định nghĩa rõ AP ở lần xuất hiện trước đó; R40 cũng chưa được giải thích là nội suy 40 điểm. LiDAR và BEV đã được định nghĩa. KITTI là tên benchmark, không cần tự bịa một tên mở rộng. Đây là lỗi tự chứa của abstract, không phải thiếu toàn bộ cấu trúc abstract.
- **location:** [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1); include chain: `MobilePIXORNeXt.tex:53 → sections/abstract.tex:1`.
- **comment_type:** `presentation`; **severity:** `moderate`; **source_kind:** `llm`; **confidence:** `high`.
- **root_cause_key:** `e01-abstract`; **review_lane:** `editor_in_chief`; **gate_blocker:** `false`; **quote_verified:** `true`.
- **evidence_anchor:** sections/abstract.tex:1; CPU / BEV AP R40 / NVIDIA Tesla T4 GPU.
- **Hướng sửa cụ thể:** Viết đầy đủ central processing unit (CPU), graphics processing unit (GPU), Average Precision (AP) ở lần đầu hoặc bỏ viết tắt không cần thiết; diễn giải R40 bằng 40-point interpolated AP. Giữ abstract trong 150–250 từ sau sửa.

### E02 — [LLM] [MINOR] Tên heading khai báo khác hướng dẫn AJSE hiện hành

- **title:** Tên heading khai báo khác hướng dẫn AJSE hiện hành
- **quote:** `\section*{Declarations}`
- **explanation:** Mục Declarations thực sự có mặt, gồm Funding, Competing Interests, Ethics/Consent, Data Availability, Code Availability và Author Contributions. Vì vậy không được báo thiếu declarations. Tuy nhiên hướng dẫn AJSE hiện hành dùng heading Statements and Declarations. Đây là khác biệt hình thức nhỏ; không có bằng chứng các khai báo liên quan bị bỏ trống.
- **location:** [sections/declarations.tex:8](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/declarations.tex:8); include chain: `MobilePIXORNeXt.tex:76 → sections/declarations.tex:8`.
- **comment_type:** `presentation`; **severity:** `minor`; **source_kind:** `llm`; **confidence:** `high`.
- **root_cause_key:** `e02-tên`; **review_lane:** `editor_in_chief`; **gate_blocker:** `false`; **quote_verified:** `true`.
- **evidence_anchor:** sections/declarations.tex:8.
- **Hướng sửa cụ thể:** Đồng bộ heading thành \section*{Statements and Declarations} khi chuẩn bị gói nộp; giữ các nội dung áp dụng hiện có. Không tự thêm tuyên bố về tài trợ/đạo đức chưa được tác giả xác nhận.

### E03 — [LLM] [MAJOR] Headline so sánh chưa phản ánh ranh giới evaluator và mức chênh lệch

- **title:** Headline so sánh chưa phản ánh ranh giới evaluator và mức chênh lệch
- **quote:** `and matched SECOND (72.09\% mAP)`
- **explanation:** Abstract và contribution cuối Introduction dùng outperformed/matched. Note của Table 3 công khai rằng MobilePIXORNeXt dùng local loader-aligned BEV AP, còn PointPillars/SECOND dùng reference KITTI AP. Cùng R40 và cùng split chưa chứng minh hai evaluator tương đương mọi quy tắc. 71.46 cũng thấp hơn 72.09 là 0.63 điểm; matched dễ bị hiểu là tương đương đã được chứng minh. Note đã có là điểm tích cực, nên chưa có cơ sở kết luận số AP sai hoặc so sánh bị ngụy tạo.
- **location:** [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1); include chain: `MobilePIXORNeXt.tex:53,58 → abstract.tex:1 / introduction.tex:19; main:66 → tables/comparison.tex:23`.
- **comment_type:** `claim_accuracy`; **severity:** `major`; **source_kind:** `llm`; **confidence:** `high`.
- **root_cause_key:** `e03-headline`; **review_lane:** `editor_in_chief`; **gate_blocker:** `false`; **quote_verified:** `true`.
- **evidence_anchor:** sections/introduction.tex:19; Table 3, tables/comparison.tex:23: local vs reference evaluator; 71.46 vs 72.09; 71.46 vs 70.13.
- **Hướng sửa cụ thể:** Nêu rõ local/reference khi trình bày so sánh headline và dùng approached/trailed by 0.63 percentage points cho SECOND. Trước một khẳng định vượt trội theo benchmark chung, đánh giá bằng cùng evaluator hoặc cung cấp đối chiếu tương đương giữa hai evaluator.

### E04 — [LLM] [MAJOR] Độ tăng AP/mAP đang viết bằng phần trăm thay vì điểm phần trăm

- **title:** Độ tăng AP/mAP đang viết bằng phần trăm thay vì điểm phần trăm
- **quote:** `+6.03\% Moderate mAP`
- **explanation:** Contribution ghi tăng từ 60.22% lên 66.25% là +6.03%. Đây là chênh lệch tuyệt đối 6.03 điểm phần trăm; tăng tương đối là khoảng 10.01%. Tương tự +3.20, +0.88, +4.74 và +4.19 trong cùng contribution là điểm AP/mAP. Lỗi này làm sai đơn vị của độ cải thiện, dù các giá trị AP gốc vẫn hợp lệ.
- **location:** [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19); include chain: `MobilePIXORNeXt.tex:58 → sections/introduction.tex:19`.
- **comment_type:** `claim_accuracy`; **severity:** `moderate`; **source_kind:** `llm`; **confidence:** `high`.
- **root_cause_key:** `e04-độ`; **review_lane:** `editor_in_chief`; **gate_blocker:** `false`; **quote_verified:** `true`.
- **evidence_anchor:** 66.25 − 60.22 = 6.03 points; 100 × (66.25 − 60.22)/60.22 ≈ 10.01%.
- **Hướng sửa cụ thể:** Ghi percentage points hoặc AP/mAP points cho hiệu AP. Rà các mức tăng/giảm AP trong Results/captions theo cùng quy tắc. Giữ % cho các tỷ lệ tương đối đúng, như 84.4% parameter reduction; không thay toàn bộ ký hiệu % một cách máy móc.

### E05 — [LLM] [MAJOR] Hình pipeline và caption bị tách sang hai trang trong PDF

- **title:** Hình pipeline và caption bị tách sang hai trang trong PDF
- **quote:** `\captionof{figure}{End-to-end architecture and tensor flow of MobilePIXORNeXt.`
- **explanation:** PDF hiện tại đặt sơ đồ Fig. 1 ở cuối trang 4 nhưng caption bắt đầu ở đầu trang 5. Người đọc không xem được hình và chú giải trong một đơn vị trình bày. Đây là vấn đề quan sát trực tiếp trên PDF, không phải suy đoán từ cảnh báo LaTeX. Hình vẫn tồn tại và không bị coi là asset thiếu.
- **location:** [sections/methodology.tex:8](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:8); include chain: `MobilePIXORNeXt.tex:62 → sections/methodology.tex:5–10`.
- **comment_type:** `presentation`; **severity:** `moderate`; **source_kind:** `llm`; **confidence:** `high`.
- **root_cause_key:** `e05-hình`; **review_lane:** `editor_in_chief`; **gate_blocker:** `false`; **quote_verified:** `true`.
- **evidence_anchor:** page04_pipeline_image.png and page05_pipeline.png.
- **Hướng sửa cụ thể:** Giữ sơ đồ và caption trong cùng một khối không bị ngắt trang; điều chỉnh vị trí/kích thước figure hoặc cách dùng strip/float. Sau khi sửa ở lượt biên tập riêng, kiểm tra PDF để bảo đảm cả hai cùng một trang.

### E06 — [LLM] [MINOR] Caption pipeline mô tả các thành phần chưa được phân biệt rõ trong sơ đồ

- **title:** Caption pipeline mô tả các thành phần chưa được phân biệt rõ trong sơ đồ
- **quote:** `Dashed arrows show training-only Oriented Geometric Alignment (OGA)`
- **explanation:** Caption nói dashed arrows cho đường training, đồng thời mô tả ConvNeXt stages, SG-FPN và hai GroupedHeader. Trong artwork pipeline, detector chủ yếu là một hộp Proposed Detector; mũi tên từ các output xuống FL + Smooth L1 + OGA + BCE + T-SBUW là nét liền, trong khi nét đứt nổi bật ở khung và một đường hồi tiếp. Không có legend giúp đọc đúng toàn bộ mô tả. Đây là vấn đề caption–artwork correspondence, không phải kết luận thuật toán sai.
- **location:** [sections/methodology.tex:8](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:8); include chain: `MobilePIXORNeXt.tex:62 → sections/methodology.tex:7–8; figures/pipeline_overview.pdf`.
- **comment_type:** `presentation`; **severity:** `minor`; **source_kind:** `llm`; **confidence:** `high`.
- **root_cause_key:** `e06-caption`; **review_lane:** `editor_in_chief`; **gate_blocker:** `false`; **quote_verified:** `true`.
- **evidence_anchor:** Fig. 1, figures/pipeline_overview.pdf; PDF pages 4–5.
- **Hướng sửa cụ thể:** Thêm legend và dùng nhất quán kiểu đường training/inference; thể hiện hoặc đánh dấu khối backbone–SG-FPN–grouped heads, hoặc thu hẹp caption theo đúng mức chi tiết đang vẽ. Mở rộng FL/BCE trong caption nếu tiếp tục dùng các nhãn này.

### E07 — [LLM] [MAJOR] Nguồn GPU của cùng các số standalone FPS chưa thống nhất giữa hai bảng

- **title:** Nguồn GPU của cùng các số standalone FPS chưa thống nhất giữa hai bảng
- **quote:** `on an NVIDIA Tesla T4 GPU.`
- **explanation:** Table 6 caption và note gán standalone timing cho Tesla T4. Table 7 caption và note gán timing cho L4. Ba cặp encoding tương ứng lại trùng FPS đến hai chữ số thập phân: Binary35/A1 = 169.97; Strata-BEV8/A2 = 165.02; Strata-BEV14/A3 = 169.01. Đây là mâu thuẫn provenance có thể thấy ngay từ các bảng, không chứng minh hai GPU không thể tình cờ cho cùng kết quả. Không được tự chọn T4 hoặc L4 làm giá trị đúng khi chưa đối chiếu log.
- **location:** [tables/components.tex:3](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/components.tex:3); include chain: `MobilePIXORNeXt.tex:66 → sections/results.tex:83 → tables/components.tex:3,32–34,45; results.tex:93 → tables/encoding.tex:3,11–13,17`.
- **comment_type:** `claim_accuracy`; **severity:** `major`; **source_kind:** `llm`; **confidence:** `high`.
- **root_cause_key:** `e07-nguồn`; **review_lane:** `editor_in_chief`; **gate_blocker:** `true`; **quote_verified:** `true`.
- **evidence_anchor:** Table 6 (T4) vs Table 7 (L4); 169.97 / 165.02 / 169.01 FPS repeated across hardware labels.
- **Hướng sửa cụ thể:** Đối chiếu log profiling và cấu hình của từng hàng; bổ sung GPU, precision, batch size và điều kiện standalone thực tế. Nếu Table 6 có số đo từ nhiều GPU, đánh dấu theo hàng; nếu chép lại số từ Table 7 thì giữ đúng provenance L4. Không sửa chỉ bằng đổi tên GPU khi chưa xác minh log.

### E08 — [LLM] [MAJOR] Caption PR khái quát ưu thế Pedestrian thành toàn bộ vulnerable road users

- **title:** Caption PR khái quát ưu thế Pedestrian thành toàn bộ vulnerable road users
- **quote:** `consistently maintains superior recall and precision envelopes on vulnerable road users`
- **explanation:** Caption Fig. 8 dùng vulnerable road users như một kết luận chung, nhưng Table 3 cho Cyclist Moderate của mô hình là 66.73%, thấp hơn SECOND 71.16%. Pedestrian Moderate 60.75% đúng là cao hơn cả hai baseline. Một ví dụ Pedestrian không hỗ trợ phát biểu nhất quán cho cả Pedestrian và Cyclist, và thứ hạng AP cũng không tự chứng minh thống trị từng điểm trên PR curve.
- **location:** [sections/results.tex:24](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:24); include chain: `MobilePIXORNeXt.tex:66 → sections/results.tex:24; tables/comparison.tex:14–15`.
- **comment_type:** `claim_accuracy`; **severity:** `moderate`; **source_kind:** `llm`; **confidence:** `high`.
- **root_cause_key:** `e08-caption`; **review_lane:** `editor_in_chief`; **gate_blocker:** `false`; **quote_verified:** `true`.
- **evidence_anchor:** Fig. 8 caption and Table 3; Cyclist Moderate: ours 66.73%, SECOND 71.16%.
- **Hướng sửa cụ thể:** Giới hạn kết luận caption vào Pedestrian và các difficulty được figure chứng minh; nêu riêng trade-off Cyclist. Bỏ consistently superior nếu không có bằng chứng thống trị PR theo điểm.

### E09 — [LLM] [MINOR] Tuyên bố không có ảnh cá nhân cần được giới hạn rõ với ảnh đường phố minh họa

- **title:** Tuyên bố không có ảnh cá nhân cần được giới hạn rõ với ảnh đường phố minh họa
- **quote:** `This manuscript does not contain any personal identifying information, individual patient records, or individual photographs.`
- **explanation:** Fig. 1 có ảnh đường phố với người đi bộ/đi xe; Fig. 10 cũng chứa camera scenes. Cụm individual photographs có thể được hiểu rộng là không có hình ảnh con người, trong khi ảnh thực sự hiện diện. Việc dùng benchmark công khai không tự chứng minh có vi phạm đồng thuận; audit này không đưa ra kết luận như vậy và không suy đoán danh tính người trong ảnh.
- **location:** [sections/declarations.tex:20](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/declarations.tex:20); include chain: `MobilePIXORNeXt.tex:76 → sections/declarations.tex:19–20; main:62,66 → Figs. 1,10`.
- **comment_type:** `presentation`; **severity:** `minor`; **source_kind:** `llm`; **confidence:** `medium`.
- **root_cause_key:** `e09-tuyên`; **review_lane:** `editor_in_chief`; **gate_blocker:** `false`; **quote_verified:** `true`.
- **evidence_anchor:** Fig. 1 street photo; Fig. 10 camera overlays, PDF page 23.
- **Hướng sửa cụ thể:** Tác giả kiểm tra phạm vi phát biểu consent với các hình được đưa vào bài; mô tả đúng việc sử dụng ảnh benchmark đã công bố và điều kiện nguồn áp dụng. Giữ phần patient records/clinical data nếu phù hợp, tránh khẳng định rộng hơn nội dung đã xác minh.

### E10 — [LLM] [MINOR] Bảng checkpoint thiếu đơn vị ở các cột số

- **title:** Bảng checkpoint thiếu đơn vị ở các cột số
- **quote:** `\textbf{E2E}`
- **explanation:** Table 10 dùng header Car, Ped., Cyc., mAP_M và E2E mà caption không gán đơn vị %/ms. Caption nêu R40 Moderate nhưng không làm rõ 60.96/81.23 là milliseconds. Các bảng khác có đơn vị ở header; riêng bảng này chưa tự chứa khi đọc độc lập.
- **location:** [sections/results.tex:158](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:158); include chain: `MobilePIXORNeXt.tex:66 → sections/results.tex:153,158–161`.
- **comment_type:** `presentation`; **severity:** `minor`; **source_kind:** `llm`; **confidence:** `high`.
- **root_cause_key:** `e10-bảng`; **review_lane:** `editor_in_chief`; **gate_blocker:** `false`; **quote_verified:** `true`.
- **evidence_anchor:** Table 10 caption/header, sections/results.tex:153,158.
- **Hướng sửa cụ thể:** Đổi header thành Car AP_M (%), Ped. AP_M (%), Cyc. AP_M (%), mAP_M (%) và E2E (ms), hoặc khai báo các đơn vị này rõ trong caption. Kiểm tra lại chiều rộng cột sau bổ sung.

### S01 — [Script] [MINOR] 19 nhãn phương trình không được viện dẫn trong văn bản

- **title:** 19 nhãn phương trình không được viện dẫn trong văn bản
- **quote:** `\label{eq:spatial_roi}`
- **explanation:** check_references.py báo 19 nhãn eq: chưa có ref/eqref; supplementary_audit.py xác nhận cùng tập. Tổng nhãn không được viện dẫn là 31 nếu kể cả 12 nhãn section. Nhãn không được dùng không phải undefined reference và không tự làm hỏng biên dịch. Đây là gợi ý vệ sinh đánh số, không phải lý do desk reject.
- **location:** [sections/methodology.tex:25](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:25); include chain: `MobilePIXORNeXt.tex:62 → sections/methodology.tex; xem Appendix A cho đủ 19 dòng`.
- **comment_type:** `presentation`; **severity:** `minor`; **source_kind:** `script`; **confidence:** `high`.
- **root_cause_key:** `s01-19`; **review_lane:** `pre_submission_readiness`; **gate_blocker:** `false`; **quote_verified:** `true`.
- **evidence_anchor:** references.json; reference_metrics.json.
- **Hướng sửa cụ thể:** Giữ đánh số cho công thức cần dẫn lại và bổ sung viện dẫn hữu ích; với công thức không cần tham chiếu, cân nhắc môi trường không đánh số. Không xóa nhãn section chỉ vì hiện chưa dùng.

### S02 — [Script] [MAJOR] Epoch của checkpoint Best AP không khớp giữa prose, caption và hàng bảng

- **title:** Epoch của checkpoint Best AP không khớp giữa prose, caption và hàng bảng
- **quote:** `extended run (Epoch 50)`
- **explanation:** Supplementary script trích xuất Epoch 50 ở Results:149, nhưng Epoch 130 ở caption:153 và hàng Best AP:160. So sánh số nguyên trả về consistent=false. Đây là kiểm tra nhất quán số tự động; không phải reviewer suy luận về lịch huấn luyện. Không dùng phát hiện này để mở rộng thành đánh giá độ đúng của phương pháp.
- **location:** [sections/results.tex:149](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:149); include chain: `MobilePIXORNeXt.tex:66 → sections/results.tex:149,153,160`.
- **comment_type:** `presentation`; **severity:** `moderate`; **source_kind:** `script`; **confidence:** `high`.
- **root_cause_key:** `s02-epoch`; **review_lane:** `pre_submission_readiness`; **gate_blocker:** `false`; **quote_verified:** `true`.
- **evidence_anchor:** checkpoint_consistency.json.
- **Hướng sửa cụ thể:** Xác nhận epoch từ metadata checkpoint và đồng bộ ba vị trí; nếu caption/hàng bảng đúng thì sửa Epoch 50 ở câu dẫn thành Epoch 130. Không đổi checkpoint hoặc số AP dựa chỉ trên lỗi mô tả này.

## 5. Cảnh báo không nâng thành lỗi và missing evidence

| Output thô | Xử lý sau kiểm chứng |
|---|---|
| Audit overall 1.95/10, raw verdict và 536 issues | Không dùng làm EIC score: ingest đưa separator/status/help text thành lỗi, riêng SENTENCES có 441 bản ghi không tương đương 441 finding đã xác minh. |
| Abstract thiếu cả năm thành phần | False positive do checker đọc entry skeleton/không xử lý include đúng trong nhánh này. Abstract thực có phương pháp, số kết quả và kết luận. |
| Missing figures/#2 | False positive trên tham số macro; 10 asset thực đều tồn tại. Không báo thiếu toàn bộ hình. |
| Anonymous submission failed vì có tên tác giả | Không áp cứng checklist blind review của venue khác cho AJSE; không có bằng chứng phải xóa author block ở submission này. |
| Acronyms checklist passed nhưng details Potentially undefined FPT | Warning nội bộ không nhất quán, không dùng để chứng nhận abstract đạt. FPT University là tên tổ chức; E01 chỉ nêu viết tắt thật trong abstract. |
| 12 reference-before-definition | Forward reference là thông lệ LaTeX; không tự động là lỗi. |
| 12 section labels không được gọi | Không phải lỗi cross-reference; giữ chúng cho cấu trúc/revision là hợp lệ. |
| 7 cảnh báo bibliography theo IEEE | Không dùng quy tắc số tác giả/truncation của IEEE làm yêu cầu AJSE. Missing DOI hoặc volume/pages là lời nhắc kiểm tra metadata; không chứng minh citation giả. |
| No citation stacking issues found / No rule-based issues detected | Dòng status, không phải finding. |
| Logic: thiếu research gap / thiếu conclusion summary | Heuristic keyword không phải phán quyết EIC; introduction có research question và conclusion tồn tại. Các chi tiết Methods ngoài phạm vi được giữ ở raw context, chưa nâng thành review issue. |
| ChkTeX spacing, dash, det/log/exp | Giữ nguyên 341 vị trí trong raw registry; chỉ E04/E10 là lỗi đơn vị đã xác nhận qua nội dung. |
| 16 deai traces | Cần kiểm tra ngữ cảnh; không có căn cứ xác nhận tác giả dùng AI từ score này. |

**Missing evidence:** chưa compile mới; chưa đối chiếu timing logs hoặc chứng minh evaluator-equivalence; chưa kiểm tra DOI/prior-art qua dịch vụ ngoài; chưa xác minh giấy phép/consent của từng hình benchmark. Các mục này không được ghi là đã PASS. `visual` checker tự động cho PDF bị skip trong `.tex` mode, nhưng đã xem các trang render bằng mắt. PyMuPDF không có trong môi trường; việc trích/raster PDF dùng Poppler, không cần cài thêm.

`uv` và lần gọi deai `--json` không chạy thành công nhưng đã có đường thay thế cùng checker và có log; không gọi đó là missing scientific evidence. Không tạo deep-review workspace hoặc report HTML vì không nằm trong hai phase được yêu cầu.

## 6. Thứ tự xử lý đề nghị

1. **Trước khi nộp:** xác minh provenance GPU E07; giới hạn headline comparison E03; sửa viết tắt abstract E01; giữ Fig. 1 cùng caption E05.
2. **Đồng bộ độ chính xác trình bày:** E04, E08, S02 và E10; chỉnh caption/legend E06.
3. **Preflight hình thức:** đồng bộ heading E02, rà wording consent E09, xử lý có chọn lọc 19 numbered equations S01.
4. Chạy lại các checker sau khi tác giả sửa ở lượt biên tập riêng; đối chiếu PDF render và giữ log thay vì dựa vào overall score của parser hiện tại.

Không có chỉnh sửa nguồn nào được thực hiện trong audit này. Phán quyết áp dụng cho snapshot hiện tại; không phải quyết định thật của tòa soạn AJSE.

## Appendix A — đủ 19 nhãn phương trình chưa được viện dẫn

Tất cả nằm dưới include `MobilePIXORNeXt.tex:62`; đây là toàn bộ tập S01, không cộng thành 19 blocker.

| Label | Dòng nguồn thật |
|---|---|
| `eq:spatial_roi` | [sections/methodology.tex:25](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:25) |
| `eq:grid_assignment` | [sections/methodology.tex:30](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:30) |
| `eq:target_tuple` | [sections/methodology.tex:37](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:37) |
| `eq:z_std_encoding` | [sections/methodology.tex:75](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:75) |
| `eq:intensity_encoding` | [sections/methodology.tex:81](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:81) |
| `eq:ground_fitting` | [sections/methodology.tex:137](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:137) |
| `eq:shadow_mask` | [sections/methodology.tex:155](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:155) |
| `eq:inverse_square` | [sections/methodology.tex:161](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:161) |
| `eq:next_block` | [sections/methodology.tex:227](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:227) |
| `eq:spatial_moments` | [sections/methodology.tex:243](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:243) |
| `eq:simam_energy` | [sections/methodology.tex:251](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:251) |
| `eq:simam_modulation` | [sections/methodology.tex:256](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:256) |
| `eq:scale_gating` | [sections/methodology.tex:289](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:289) |
| `eq:gt_targets` | [sections/methodology.tex:323](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:323) |
| `eq:projection_axes` | [sections/methodology.tex:338](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:338) |
| `eq:mgiou_loss` | [sections/methodology.tex:347](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:347) |
| `eq:corner_loss` | [sections/methodology.tex:358](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:358) |
| `eq:iqa_loss` | [sections/methodology.tex:366](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:366) |
| `eq:joint_score` | [sections/methodology.tex:454](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:454) |

## Appendix B — registry toàn bộ output thô

- [audit_self_check.json](artifacts/audit_self_check.json): đủ 536 candidate records, không loại bỏ âm thầm.
- [references.json](artifacts/references.json): đủ 31 cảnh báo raw có source line.
- [format.json](artifacts/format.json): đủ 341 ChkTeX warnings, file/line/column/code/message.
- [tables.json](artifacts/tables.json), [figures.stdout.txt](artifacts/figures.stdout.txt), [figure_metrics.json](artifacts/figure_metrics.json): structural/asset evidence.
- [deai_api_analysis.json](artifacts/deai_api_analysis.json): đầy đủ từng trace và vị trí source.
- [source_map.json](artifacts/source_map.json): ánh xạ 1.110 assembled lines sang file/dòng thật. Assembled lines không được trình bày nhầm là dòng main.tex.
- [editor_in_chief.json](artifacts/editor_in_chief.json), [quote_verification.json](artifacts/quote_verification.json), [source_integrity.json](artifacts/source_integrity.json): decision và kiểm chứng.

### ChkTeX: phân bố warning code (không phải mức độ nghiêm trọng của bản thảo)

| Code | Số warning | Message |
|---|---:|---|
| 1 | 235 | Command terminated with space. |
| 3 | 10 | You should enclose the previous parenthesis with `{}'. |
| 8 | 11 | Wrong length of dash may have been used. |
| 13 | 8 | Intersentence spacing (`\@') should perhaps be used. |
| 24 | 8 | Delete this space to maintain correct pagereferences. |
| 25 | 10 | You might wish to put this between a pair of `{}' |
| 29 | 2 | $\times$ may look prettier here. |
| 35 | 4 | You should perhaps use `\det' instead. |
| 35 | 1 | You should perhaps use `\exp' instead. |
| 35 | 3 | You should perhaps use `\log' instead. |
| 36 | 49 | You should put a space in front of parenthesis. |

Báo cáo giữ đầy đủ raw evidence nhưng chỉ issue bundle ở mục 4 là danh sách lỗi/điểm cần sửa được xác nhận trong phạm vi hai phase.
