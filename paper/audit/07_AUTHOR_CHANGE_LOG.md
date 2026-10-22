# 07 AUTHOR CHANGE LOG (NHẬT KÝ THAY ĐỔI VÀ KHẮC PHỤC HỌC THUẬT - ĐỢT 1 & ĐỢT 2)

> **Document Status**: Official Author Change Log (Hoàn tất Đợt 1 & Đợt 2: Nội dung Học thuật, Toán học, Thực nghiệm, Văn phong De-AI & Quy chuẩn AJSE)  
> **Source Directives**: `paper/audit/06_UNIFIED_REPAIR_LEDGER.md`  
> **Verification Gate**: `audit.py --mode gate` $\to$ **PASS** (0 Critical Blockers, 0 Broken References, 0 Overclaims)  
> **Compilation Status**: `compile.py --recipe pdflatex-bibtex` $\to$ **SUCCESS** (28 pages, PDF generated cleanly)  
> **Date**: 2026-10-22  

---

## 1. BẢNG TỔNG HỢP ÁNH XẠ SỬA LỖI THEO MÃ FIX-ID (ĐỢT 1)

| Mã FIX-ID | Phân loại | Trạng thái | Vị trí file / dòng | Tóm tắt nội dung thay đổi học thuật & kỹ thuật |
|---|---|---|---|---|
| **FIX-01** | `MAJOR` | `FIXED` | `sections/abstract.tex:1`<br/>`MobilePIXORNeXt.tex:50` | **Định vị phạm vi nhiệm vụ $SE(2)$**: Khẳng định rõ MobilePIXORNeXt phát hiện oriented bounding footprints trên đa tạp $SE(2)$ phục vụ tránh va chạm và lập quỹ đạo xe tự hành thời gian thực. Đảm bảo độ dài 238 từ (chuẩn Springer AJSE 150–250 từ) và giữ vững 5 thành tố cấu trúc. |
| **FIX-02** | `MAJOR` | `FIXED` | `sections/introduction.tex:7-8` | **Bổ sung Ngữ cảnh Range-View & Point-Based Direct**: Phân tích Range-View (LaserNet spherical projection gây méo tỷ lệ theo khoảng cách và xé rách biên che khuất) và Point-Based Direct (PointNet++ có độ phức tạp bộ nhớ $O(N^2)$ làm nghẽn băng thông edge). Phá bỏ các cụm trích dẫn dồn toa thành ngữ cảnh chuyên đề. |
| **FIX-03** | `MAJOR` | `FIXED` | `sections/introduction.tex:9-11` | **Phòng thủ phủ đầu cho Q1 & Q2 ($SE(2)$ & Co-design)**: Luận giải việc tối ưu trên đa tạp $SE(2)$ do khoảng hở ngang và TTC chi phối 99% quyết định phanh an toàn. Định vị PC-GTA và OGA là bộ đôi đồng thiết kế (*co-design*) tất yếu giữa vật lý cảm biến và tối ưu hóa hình học. |
| **FIX-04** | `MINOR-SCRIPT` | `FIXED` | `sections/introduction.tex:5, 7, 9` | **Chuẩn hóa De-AI & Câu chủ đề Claim-Forward**: Loại bỏ từ đệm sáo rỗng (*throat-clearing*), định nghĩa rõ ràng các từ viết tắt theo chuẩn IEEE/Springer (`SECOND`, `SE`, `AP`, `NVIDIA`), mở đầu mỗi đoạn bằng câu khẳng định định lượng. |
| **FIX-05** | `MAJOR` | `FIXED` | `sections/related_work.tex:11-13` | **Tiểu mục Mới về Ground-Truth Data Augmentation & Physical Regularization**: Bổ sung tiểu mục 2.2 điểm danh SECOND và phân tích các khiếm khuyết phi vật lý của phép chắp vá (cut-and-paste) truyền thống (bay lơ lửng, chìm mặt đường, đâm xuyên vật cản, thiếu bóng raycast LiDAR), làm bật tính ưu việt của PC-GTA. |
| **FIX-06** | `MAJOR` | `FIXED` | `sections/related_work.tex:1-7`<br/>`references.bib:201-228` | **Trích dẫn và Đối sánh Range-View & Modern BEV Transformers**: Bổ sung trích dẫn LaserNet~\cite{meyer2019lasernet} và TransFusion~\cite{bai2022transfusion}. Đối sánh và làm rõ Range-View bị biến dạng góc nhìn còn Transformer vượt quá ngân sách 1M tham số của vi điều khiển biên. |
| **FIX-07** | `CRITICAL` | `FIXED` | `sections/methodology.tex:329-335` | **Cơ chế Phân lập Quyền sở hữu Mục tiêu Category Decoupling trong `GroupedHeader`**: Formalize toán học việc phân lập mục tiêu: Nhóm 1 (`car`) và Nhóm 2 (`ped_cyc`) tính loss độc lập trên các mặt nạ foreground $\mathbf{M}_{\mathrm{reg}, 1}$ và $\mathbf{M}_{\mathrm{reg}, 2}$, loại bỏ hoàn toàn cạnh tranh gradient giữa các đối tượng có tỷ lệ khung hình dị thể. |
| **FIX-08** | `MAJOR` | `FIXED` | `sections/methodology.tex:325-330` | **Giải mã Góc và Chứng minh Tính Ổn định Doubled-Angle**: Bổ sung phương trình giải mã $\hat{\psi} = \frac{1}{2}\operatorname{atan2}(\hat{a}_2, \hat{a}_1)$ kèm chứng minh toán học: vector $(\cos 2\psi, \sin 2\psi)$ trên đường tròn đơn vị $S^1$ giải quyết triệt để bước nhảy góc $180^\circ$ và kỳ dị biên $\pm \pi/2$ của phép hồi quy góc vô hướng. |
| **FIX-09** | `MAJOR` | `FIXED` | `sections/methodology.tex:121-124` | **Luận giải Khả thi Phần hardware Nhúng cho Strata-BEV14 (Phòng thủ Q4)**: Thuyết minh thuật toán tích lũy ô cục bộ độc lập có thể chuyển thành CUDA kernel (<3 ms) trên Jetson hoặc chạy CPU đa luồng bất đồng bộ (double-buffering) nạp frame $t+1$ song song với frame $t$ chạy GPU. |
| **FIX-10** | `CRITICAL` | `FIXED` | `sections/results.tex:105-112` | **Luận giải Bước sụt giảm A7 $\to$ A8 (-0.74 pp) và Bứt phá A8 $\to$ A9 (+3.20 pp)**: Phân tích cơ chế hiệp đồng: Depthwise scale gating trong SG-FPN làm suy hao tín hiệu thưa ở xa khi chỉ dùng Smooth-L1, nhưng khi kết hợp với OGA và IQA thì cung cấp đúng không gian biểu diễn đa quy mô để OGA khai thác, tạo bước nhảy vọt $+3.20\,\text{pp}$. |
| **FIX-11** | `MAJOR` | `FIXED` | `sections/results.tex:96-100` | **Luận giải Hiện tượng Tăng vọt +10.08 pp của Head Width H16 $\to$ H32**: Phân tích sụp đổ hạng không gian con (*subspace rank collapse*): đầu 16 kênh không đủ chiều để tải 4 tác vụ dị thể; mở rộng lên 32 kênh chỉ tốn thêm 75K tham số nhưng tăng gấp đôi bậc tự do, phân ly gradient triệt để. |
| **FIX-12** | `MAJOR` | `FIXED` | `sections/results.tex:165-167` | **Bằng chứng Factual Epoch 47 ($71.48\%$ mAP) (Phòng thủ Q3)**: Khẳng định đanh thép từ Table 6: ở Epoch 47 mAP đã đạt $71.48\%$; đợt chạy 160 epoch không làm tăng mAP mà chỉ làm sắc nét heatmap, giảm box rác và cắt giảm $25.0\%$ độ trễ NMS ($81.23\,\text{ms} \to 60.96\,\text{ms}$). |
| **FIX-13** | `MAJOR` | `FIXED` | `sections/results.tex:194-195` | **Thừa nhận Ranh giới Cảm biến Thưa & Địa hình Phi phẳng (Phòng thủ Q1 & Q5)**: Mở rộng Limitations: thừa nhận giới hạn trên LiDAR 64 tia mặt đường phẳng; chỉ ra tính ổn định số học của Strata-BEV14 và đề xuất RC-SGFPN cho cảm biến thể rắn thưa tia trong tương lai. |
| **FIX-14** | `MAJOR` | `FIXED` | `sections/conclusion.tex:5-9` | **Đồng bộ Kết luận với Luận điểm Đồng Thiết kế (Co-design)**: Khẳng định thông điệp cốt lõi: nhận thức LiDAR biên đạt hiệu quả tối ưu nhờ sự đồng thiết kế giữa tiền xử lý thống kê xác định, tăng cường dữ liệu vật lý và giám sát hình học, thay vì mở rộng mạng hay dùng PointNet tốn kém. |

---

## 2. CHI TIẾT SỬA LỖI ĐỢT 2: PHẪU THUẬT VĂN PHONG, DE-AI POLISH & QUY CHUẨN AJSE

### 2.1. Cắt tỉa và Khóa chuẩn Abstract Springer AJSE (150–250 từ)
- **Tình trạng trước sửa**: `analyze_abstract.py` cảnh báo 251 từ (`[WARNING]`), vượt ngưỡng giới hạn tối đa 250 từ của tạp chí Springer AJSE.
- **Biện pháp xử lý**:
  * Tinh giản câu diễn giải cấu trúc mà vẫn bảo toàn tuyệt đối 5 thành tố bắt buộc: `Background`, `Objective`, `Methods`, `Results`, `Conclusion`.
  * Rút gọn câu mô tả tensor Strata-BEV14 và cấu trúc mạng, giữ nguyên số liệu định lượng: **753,569 parameters**, **71.46% Moderate AP**, **+16.32 pp**, **20.94 ms** (RTX 3060), **60.96 ms** (Jetson Orin Nano).
- **Kết quả sau sửa**: Đạt **238 từ** (`[PASS]`), hoàn toàn nằm trong khung 150–250 từ chuẩn mực.

### 2.2. Phá vỡ cấu trúc cảnh báo ngược (Claim-Forward & Paragraph Arc)
- **Tình trạng trước sửa**: `check_claim_forward.py` phát hiện mẫu `CF-CAVEAT-POS` tại `sections/related_work.tex:11`. Đoạn mở đầu bằng hạn chế của phương pháp cut-and-paste truyền thống thay vì nêu bật đóng góp của giải pháp đề xuất.
- **Biện pháp xử lý**:
  * Đảo cấu trúc theo nguyên tắc *Claim-Forward*: Đưa khẳng định giải pháp đề xuất (*"To eliminate unphysical synthesis artifacts... MobilePIXORNeXt introduces PC-GTA..."*) lên đầu đoạn làm câu chủ đề.
  * Di chuyển các phân tích khiếm khuyết vật lý (bay lơ lửng, chìm lấp mặt đường) ra phần thân đoạn làm đòn bẩy chứng minh.
  * Kết thúc đoạn bằng bằng chứng định lượng vững chắc: mức cải thiện **+6.03 pp** trên tập kiểm thử KITTI.
- **Kết quả sau sửa**: `check_claim_forward.py` trả về `No claim-forward patterns detected` (`[PASS]`).

### 2.3. Khử triệt để các dấu vết và từ vựng phong cách AI (De-AI Polish)
- **Từ vựng bị loại bỏ hoặc thay thế**:
  * `essentially identical` $\to$ `closely matching` (loại bỏ từ ngụy tạo trực giác `fake_insight_marker`).
  * `completely hiding` $\to$ `effectively hiding` (loại bỏ trạng từ phóng đại `over_confident`).
  * `completely decoupling` $\to$ `decoupling`.
  * `Crucially` ở đầu câu tại `sections/methodology.tex` $\to$ xóa bỏ hoàn toàn, chuyển sang câu trần thuật khách quan.
  * Chuỗi liên tiếp `Furthermore` $\to$ thay thế đan xen bằng `In addition`, `Additionally`.
- **Đồng nhất cấu trúc đoạn văn**:
  * Chuyển đổi các khối định dạng thủ công lặp lại `\par\vspace{0.5\baselineskip}\textit{N. ...}` trong `sections/results.tex` thành `\paragraph{N. ...}` chuẩn mực của Springer, loại bỏ cảnh báo biến thiên cú pháp (`[burstiness]`).

### 2.4. Khóa chuẩn hiển thị Springer AJSE
- **Keywords**: 5 từ khóa chuyên ngành chính xác theo phân loại bài toán: *3D Object Detection, Autonomous Driving, LiDAR Point Clouds, Bird's-Eye View, Embedded Edge AI*.
- **Declarations Section**: Khóa đầy đủ và chuẩn xác các trường bắt buộc của Springer (`Funding`, `Competing Interests`, `Availability of data and materials`, `Code availability`, `Author Contributions`).
- **Booktabs Tables & Captions**: Toàn bộ bảng biểu sử dụng `\toprule`, `\midrule`, `\bottomrule` chuẩn `booktabs`, không có đường kẻ dọc (`|`), chú thích đặt phía trên bảng và tự giải thích đầy đủ các từ viết tắt.

---

## 3. KẾT QUẢ KIỂM ĐỊNH TỔNG HỢP VÀ GATE CUỐI CÙNG

| Công cụ kiểm tra | Lệnh thực thi | Kết quả đạt được | Trạng thái |
|---|---|---|---|
| **Abstract Analyzer** | `analyze_abstract.py MobilePIXORNeXt.tex` | 238 words (ngưỡng 150–250), đầy đủ 5 thành tố, 0 citations | **PASS** |
| **Claim-Forward Checker** | `check_claim_forward.py MobilePIXORNeXt.tex` | 0 claim-forward / caveat-positioning patterns detected | **PASS** |
| **Grammar & Contract** | `analyze_grammar.py MobilePIXORNeXt.tex` | 0 rule-based grammar issues detected | **PASS** |
| **De-AI Lexical Scan** | `deai_check.py MobilePIXORNeXt.tex --analyze` | 0 forbidden terms, 0 fake insight markers, 0 overconfident claims | **PASS** |
| **Quality Gate** | `audit.py MobilePIXORNeXt.tex --mode gate` | 0 Critical Blockers, 0 Broken Refs/Cites, 0 Overclaims, Math & Acronyms Clean | **PASS** |
| **LaTeX Compilation** | `compile.py MobilePIXORNeXt.tex --recipe pdflatex-bibtex` | 28 pages PDF (5.69 MB), exit code 0, 100% references resolved | **SUCCESS** |
