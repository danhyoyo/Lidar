# Báo cáo deai và grammar — MobilePIXORNeXt

Đã đọc các tài liệu được yêu cầu trong skill `latex-paper-en`, chạy năm script trên toàn bộ **17 file `.tex`** của `paper/`, và áp dụng **107 bản sửa đoạn/chú thích vào 13 file**. Các file chính, helper hình, và những bảng còn lại vẫn được kiểm tra dù không cần sửa nguồn.

## Đọc kết quả

- [Danh sách 192 câu cần đối chiếu](findings.md): nguyên văn LaTeX, vị trí nguồn, từ/cấu trúc liên quan, số từ, cảnh báo thì và cách xử lý. Đây là **các ứng viên đã đọc theo ngữ cảnh**, không phải 192 lỗi độc lập đã được chứng minh.
- [Toàn bộ 107 bản trước–sau đã áp dụng](rewrites.md): mỗi mục có `Changed`, `Protected`, `Meaning-Check`, `Risk-Flags` và token gốc để kiểm tra.
- [Chỉ mục toàn bộ 41 dòng có từ/cụm cần xem ngữ cảnh](lexical_index.md), gồm cả tiêu đề, vai trò tác giả và nội dung fallback của macro.
- [Mọi cảnh báo câu dài của script, trước và sau](script_sentence_findings.md): giữ đủ lượt chạy file chính và từng file con; có giải thích cảnh báo giả và mục lặp.
- [Đối chiếu toàn bộ cảnh báo deai/thì](deai_trace_triage.md).
- [Kiểm tra protected tokens](protected_tokens.json), [bằng chứng xác minh](verification.json), [lệnh đã chạy](commands.json), [lệnh kiểm tra bổ sung](additional_commands.json), [log biên dịch](compile.txt).
- [PDF sau sửa](/tmp/lidar_language_build_20261022/MobilePIXORNeXt.pdf).

## Từ vựng và cấu trúc

Các ví dụ tác giả nêu — `delve into`, `pivotal`, `testament`, `tapestry`, `underscore`, `crucial`, `comprehensive` — **không xuất hiện trong nguồn**. Những vấn đề thực tế gồm `severe`, `essential`, `fundamental`, `robust`, `significantly`, `substantial`, `synergistic`, `prohibitive`, `decisive`, cùng các câu như “key scientific insight”, “Guided by this design philosophy”, “fundamental operational dichotomy”, và “guarantees stable convergence”. Chỉ mục từ vựng ghi đủ từng dòng và quyết định xử lý.

`references/deai/forbidden-terms.md` thực tế là tài liệu về **thuật ngữ/ký hiệu phải bảo vệ**, không phải từ điển cấm các từ được tác giả nêu. Vì vậy, không xóa thuật ngữ kỹ thuật chỉ dựa vào một danh sách từ. Ví dụ, “critical revision” trong đóng góp tác giả có nghĩa cụ thể và được giữ nguyên. Các tên và tiêu đề kỹ thuật được bảo vệ cũng không bị đổi tên để giảm số lần xuất hiện một từ.

Các câu quảng bá hoặc gán nguyên nhân quá chắc đã được viết lại bằng quan sát định lượng sẵn có. “guarantees stable convergence” trở thành mô tả các đường loss không có phân kỳ quan sát được; không biến đồ thị thành bảo đảm toán học. Các giải thích cơ chế chưa được tách riêng bằng thí nghiệm được ghi dưới dạng giải thích có điều kiện. Các mục đó mang `Risk-Flags: overstatement` để tác giả kiểm tra việc hạ mức khẳng định.

Không kết luận một câu do AI viết từ danh sách từ, dấu câu hoặc điểm mật độ. Các nhóm H-* trong tài liệu pattern-clusters được dùng để đọc ngữ cảnh; không được biến thành bộ dò từ khóa.

## Câu dài

Ngưỡng dùng khi chạy `analyze_sentences.py`: **hơn 50 từ hoặc bộ đếm mệnh đề lớn hơn 3**. Bộ đếm của script cộng dấu phẩy và các từ nối như `which`, `that`, `while`; nó không phân tích cú pháp ngữ pháp thực sự.

Ba câu văn xuôi rõ ràng vượt 50 từ sau khi tách ranh giới LaTeX và loại toán khỏi bộ đếm nằm tại Introduction dòng 11, Methodology dòng 50, và Related Work dòng 24. Chúng đã được tách thành các câu ngắn hơn, giữ nguyên thứ tự số liệu, ký hiệu và tham chiếu. Những câu dưới 50 từ nhưng chứa quá nhiều cơ chế cũng được biên tập trong bản trước–sau.

Script phát **208 cảnh báo thô trước sửa và 118 sau sửa**, tính cả lượt chạy file chính và các file con nên có mục lặp. Các cảnh báo còn lại thuộc danh sách, công thức/bảng bị gộp vào văn xuôi, hoặc ranh giới `?}`/caption/macro. Không tách công thức, danh sách tác giả hay hàng bảng thành câu rời để làm bộ đếm về 0. Danh sách đầy đủ và quyết định cho từng cảnh báo nằm trong báo cáo liên kết ở trên.

## Thì và ngữ pháp

Introduction dùng hiện tại cho nền tảng; Related Work dùng quá khứ khi thuật lại một nghiên cứu cụ thể. Các thao tác cấu hình, huấn luyện, đánh giá và kết quả đã thực hiện được chuyển sang quá khứ. Định nghĩa, công thức, khả năng của phần mềm và mô tả trực tiếp hình/bảng vẫn dùng hiện tại theo `tense-guide.md`.

`deai_check.py --analyze` báo **23 dấu vết trước sửa và 14 sau sửa** trong lượt duyệt file chính. Báo cáo mặc định chỉ in tối đa 10 mục mỗi section, nên đã lấy thêm JSON `--fix-suggestions` để không bỏ sót các mục bị ẩn. Cả 14 mục sau sửa đã được đối chiếu: 6 cảnh báo lặp markup (`\item \textit`, `\par \vspace`), 8 cảnh báo thì thuộc caption, định nghĩa hoặc mô tả nội dung hình hợp lệ.

`deai_batch.py` và `analyze_grammar.py` không phát hiện rule hit tương ứng. `improve_expression.py` không đưa ra bản thay thế tự động, nhưng sau sửa có hai ứng viên weak verb `make` (lặp ở lượt chạy file chính và file con). Cả hai được giữ vì `make` đang dùng đúng nghĩa gây ra một trạng thái, không phải cụm vòng như `make use of`; xem [đối chiếu expression](expression_triage.md). Điều đó phản ánh phạm vi rule của script, không chứng nhận toàn bộ bài không có lỗi. Một số file con không có section heading còn không được `deai_batch.py` phân đoạn; lượt duyệt toàn văn từ file chính và việc đọc từng nguồn bổ sung cho giới hạn này. Một số lượt chạy riêng `sections/experiments.tex` hoặc `sections/results.tex` còn cảnh báo không tìm thấy `tables/...` theo đường dẫn tương đối của file con; lượt duyệt file chính resolve được các include, từng file bảng cũng được chạy trực tiếp, và biên dịch đầy đủ xác nhận tất cả include tồn tại. Không áp dụng các đề xuất tách mọi dấu phẩy thành dấu chấm của script.

## Bảo toàn nội dung kỹ thuật

So sánh toàn bộ 17 file với snapshot ngay trước lượt sửa ngôn ngữ, không so với Git HEAD đang có thay đổi từ trước:

| Nhóm kiểm tra | Kết quả |
|---|---|
| Cite/ref/label và các biến thể | 203 mục, nguyên vẹn và giữ thứ tự |
| Toán inline/display/môi trường phương trình | 805 đoạn, byte-identical và giữ thứ tự |
| Giá trị số và định danh có số | 1.860 token, giữ nguyên giá trị và thứ tự; dấu câu kết thúc được phân biệt với chữ số |
| Chuỗi lệnh LaTeX | 3.104 lệnh, giữ nguyên thứ tự |
| Dòng dữ liệu bảng chứa `&` | 136 dòng, byte-identical |
| Tên kỹ thuật và acronym gốc | Không mất lần xuất hiện nào trong danh mục đã kiểm tra |
| Custom macro hình, thuật toán | Giữ nguyên |
| Khối hình PR tác giả tự xử lý | Giữ nguyên toàn bộ |

Các nhóm đếm có giao nhau. Việc bảo toàn token được máy kiểm tra; sự tương đương về ý nghĩa và mức độ khẳng định là đánh giá biên tập có ghi rõ trong từng mục. Không thêm số liệu, citation, công thức hay kết quả thí nghiệm.

## Xác minh

- 85 lượt script trước sửa và 85 lượt sau sửa theo đúng thứ tự năm script, tất cả exit code 0.
- Biên dịch bằng `scripts/compile.py --recipe latexmk`: exit code 0, PDF **25 trang**, không có citation/reference chưa giải quyết hoặc nhãn trùng.
- Đã xem trang 1 (abstract), 14 (thiết lập thí nghiệm/bảng), 22 (qualitative/discussion); không thấy chữ/hình chồng lên nhau hoặc bị cắt ở các trang này.
- Log còn một overfull hbox rất nhỏ 0.12177 pt và các underfull hbox; có ghi trong verification.json. `pdftotext` trích đủ 25 trang với một cảnh báo cấu trúc PDF “Badly formatted number”; ba trang kiểm tra vẫn render thành công. Lượt này không thay đổi các PDF hình nguồn.

Theo hướng dẫn của skill về việc có hỗ trợ biên tập bằng LLM, tác giả có thể đối chiếu chính sách khai báo của venue trong [AI-disclosure matrix](/home/duyennh/.agents/skills/latex-paper-en/references/venues/ai-disclosure.md).
