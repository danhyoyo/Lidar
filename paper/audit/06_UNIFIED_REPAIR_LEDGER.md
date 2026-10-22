# 06 UNIFIED REPAIR LEDGER (SỔ CÁI ĐIỀU PHỐI VÀ KHỬ XUNG ĐỘT SỬA LỖI)

> **Document Status**: Definitive Pre-Edit Dispatch Ledger (Bảng Điều Phối Khử Xung Đột)  
> **Source Inputs**:  
> - `paper/audit/00_MASTER_AUDIT_REPORT.md` (Consolidated Audit Phase 1–3)  
> - `academic-paper-reviewer` (Devil's Advocate Seat 5 Adversarial Analysis - Round 1.5 Task 1)  
> - `academic-paper-review` (`bytedance/deer-flow` Literature Positioning & 5 Questions for Authors - Round 1.5 Task 2)  
> - `paper/code_context.md` (Ground Truth Codebase Specification & Experimental Logs)  
> - `latex-paper-en/references/deai/forbidden-terms.md` (De-AI Style & Protected Terms Guide)  
> **Operational Constraint**: STRICT READ-ONLY on `.tex` files until user approval.  

---

## 1. QUY TẮC GIẢI QUYẾT XUNG ĐỘT (CONFLICT RESOLUTION RULES)

1. **Quy tắc 1 (Ưu tiên Tuyệt đối Tiêu chuẩn Springer AJSE & Ground Truth Codebase)**:
   - Mọi số liệu thực nghiệm, độ trễ, số lượng tham số, kích thước checkpoint và tên hàm/lớp phải ánh xạ chính xác 100% với [`paper/code_context.md`](file:///home/duyennh/AI_projects/research_lidar/Lidar/paper/code_context.md).
   - Nếu có mâu thuẫn giữa đề xuất "giữ phong cách tổng quát" và quy định cứng của tạp chí AJSE (Abstract 150–250 từ, Declarations đầy đủ theo chuẩn Springer, đơn vị `percentage points (pp)` cho chênh lệch), quy chuẩn của Springer AJSE là ưu tiên tối thượng.
2. **Quy tắc 2 (Lọc Bỏ Góp ý Từ vựng Chủ quan - Non-Substantive Filter)**:
   - Loại bỏ toàn bộ các góp ý `MINOR` mang tính thay đổi từ vựng chủ quan nếu câu gốc chuẩn ngữ pháp tiếng Anh học thuật và không vi phạm danh mục từ cấm trong `latex-paper-en/references/deai/forbidden-terms.md`.
   - Giữ nguyên các thuật ngữ kỹ thuật chuẩn (LiDAR, BEV, PointNet, IoU, NMS, ConvNeXt, SimAM, OGA, PC-GTA, T-SBUW).
3. **Quy tắc 3 (Bảo toàn 100% Protected Tokens)**:
   - Tuyệt đối không xóa, đổi tên, hoặc làm hỏng bất kỳ token nào thuộc nhóm: `\cite{...}`, `\ref{...}`, `\label{...}`, và các môi trường toán học.
4. **Quy tắc 4 (Phòng thủ Phủ đầu - Pre-emptive Defense)**:
   - Mọi câu hỏi chất vấn sát sườn từ Devil's Advocate và Deer-Flow phải được trả lời trực tiếp trong nội dung bài báo dưới dạng lập luận khoa học chặt chẽ hoặc bổ sung vào mục Limitations, không né tránh.

---

## 2. BẢNG TỔNG HỢP 5 CÂU HỎI CHẤT VẤN SÁT SƯỜN ("QUESTIONS FOR AUTHORS")

| # | Câu hỏi chất vấn từ Reviewer (Deer-Flow / DA) | Rủi ro học thuật | Chiến lược trả lời phủ đầu trong văn bản (Pre-emptive Defense) |
|---|---|---|---|
| **Q1** | **Bất đối xứng bài toán (2D BEV vs. 3D Box)**: Tại sao so sánh với SECOND và PointPillars khi MobilePIXORNeXt chỉ dự đoán footprint 2D $(x,y,l,w,\psi)$ mà bỏ qua cao độ $z$ và chiều cao $h$? | Bị nghi ngờ hạ thấp độ khó của bài toán để "thổi phồng" mAP. | Khẳng định rõ miền ứng dụng của xe tự hành trên đa tạp $SE(2)$: khoảng hở ngang và nón va chạm quyết định 99% thời gian phản ứng phanh (TTC). Nêu rõ đầu hồi quy vertical 3D là nhánh phụ trợ, thừa nhận địa hình dốc là hướng nghiên cứu tương lai trong Limitations. |
| **Q2** | **Nghi ngờ Đột phá Kiến trúc vs. Mở rộng Dung lượng**: Hơn 53% mức tăng (+16.11 pp) đến từ tăng kênh head (H16 $\to$ H32) và dữ liệu PC-GTA (+6.03 pp), trong khi SG-FPN giảm -0.74 pp và SimAM chỉ tăng +0.88 pp. Đâu là tính mới kiến trúc thực sự? | Bị đánh trượt vì coi bài báo chỉ là "tập hợp thủ thuật kỹ thuật" (engineering bag-of-tricks). | Luận giải: Đầu H16 bị sụp đổ hạng không gian con (*subspace rank collapse*) khi phải gánh 4 tác vụ dị thể. Mở rộng lên H32 chỉ tốn thêm 75K tham số nhưng phân ly gradient triệt để. SG-FPN giảm nhẹ dưới Smooth-L1 do depthwise gating làm yếu tín hiệu xa, nhưng khi kết hợp với OGA và IQA thì bứt phá +3.20 pp lên 70.58%, chứng minh sự đồng thiết kế (*co-design*) tất yếu. |
| **Q3** | **Tính Công bằng của Lịch trình Huấn luyện (160 vs. 50 Epochs)**: Checkpoint chính được chọn ở Epoch 130 của đợt chạy 160 epoch, trong khi baselines chỉ chạy 50 epoch. Có phải mAP cao là do luyện lâu? | Bị coi là unfair comparison / cherry-picking. | Bằng chứng đanh thép từ dữ liệu thực tế Table 6 (`code_context.md:468-471`): Ở **Epoch 47 của đợt 50 epoch, mAP đã đạt 71.48%** (cao hơn cả Epoch 130 là 71.46%). Chạy 160 epoch không làm tăng mAP mà chỉ làm sắc nét heatmap, giảm box rác và hạ độ trễ NMS từ 81.23 ms xuống 60.96 ms. |
| **Q4** | **Nút thắt Tiền xử lý CPU (20.94 ms)**: 20.94 ms rasterize trên CPU chiếm 34% tổng thời gian pipeline. Triển khai trên chip nhúng ARM có làm tê liệt hệ thống? | Nghi ngờ tính khả thi thực tế trên nền tảng nhúng (edge feasibility). | Chứng minh: Strata-BEV14 dùng các bộ tích lũy ô cục bộ song song và bảng tra logarit, có thể viết thành CUDA kernel (<3 ms trên Jetson) hoặc chạy đa luồng bất đồng bộ (asynchronous double-buffering) nạp frame $t+1$ song song với frame $t$ chạy GPU. |
| **Q5** | **Khả năng Tổng quát hóa Cảm biến Thưa**: Mô hình chỉ thử nghiệm trên LiDAR 64 tia của KITTI. Nếu chạy trên cảm biến 16 tia hoặc LiDAR thể rắn quét ngẫu nhiên thì Strata-BEV14 có sụp đổ? | Giảm phạm vi đóng góp của biểu diễn 14 kênh. | Luận giải: Chuẩn hóa logarit và moment liên tục ($z_{\mathrm{span}}, z_{\mathrm{std}}$) duy trì tính ổn định số học khi ô thưa điểm. Thừa nhận trung thực giới hạn và đề xuất điều biến tần số liên tục (RC-SGFPN) cho LiDAR thể rắn trong Limitations. |

---

## 3. SỔ CÁI ĐIỀU PHỐI SỬA LỖI CHI TIẾT (ACTIONABLE REPAIR LEDGER)

### Mục 1: Abstract (`paper/sections/abstract.tex` & `paper/MobilePIXORNeXt.tex`)

| FIX-ID | Mức độ | Vị trí dòng trong `main.tex` | Mô tả lỗi / Căn cứ phát hiện | Chỉ thị sửa chính xác cho `latex-paper-en` |
|---|---|---|---|---|
| **FIX-01** | `MAJOR` | `MobilePIXORNeXt.tex:50-60` / `abstract.tex:1` | **Định vị phạm vi nhiệm vụ $SE(2)$**: Abstract hiện tại so sánh trực tiếp với PointPillars và SECOND nhưng chưa làm rõ phạm vi phát hiện footprint 2D BEV có hướng cho bài toán tránh va chạm xe tự hành. | Khẳng định rõ: MobilePIXORNeXt là bộ phát hiện footprint có hướng trên mặt phẳng $SE(2)$ phục vụ tránh va chạm và lập quỹ đạo thời gian thực. Giữ nguyên độ dài 219 từ và đầy đủ 5 thành tố cấu trúc. |

---

### Mục 2: Introduction (`paper/sections/introduction.tex`)

| FIX-ID | Mức độ | Vị trí dòng trong `main.tex` | Mô tả lỗi / Căn cứ phát hiện | Chỉ thị sửa chính xác cho `latex-paper-en` |
|---|---|---|---|---|
| **FIX-02** | `MAJOR` | `sections/introduction.tex:7-8` | **Thiếu ngữ cảnh Range-View và Point-Based Direct** (Phát hiện từ Deer-Flow): Đoạn 2 phân loại 3 nhánh chính nhưng bỏ sót Range-View (LaserNet, RangeDet) và Point-based (PointNet++, 3D-SSD). | Bổ sung 2 câu phân tích: Range-view chiếu cầu dùng được 2D CNN nhưng bị méo tỷ lệ vật thể theo khoảng cách và che khuất chồng chéo; Point-based direct giữ độ chính xác điểm thô nhưng thuật toán Ball Query / FPS có độ phức tạp bộ nhớ $O(N^2)$, làm nghẽn bộ nhớ đệm biên. |
| **FIX-03** | `MAJOR` | `sections/introduction.tex:9-11` | **Phòng thủ phủ đầu cho Q1 & Q2 ($SE(2)$ Manifold & Co-design)**: Cần luận giải lý do tập trung vào mặt phẳng BEV và định vị PC-GTA như một trụ cột đồng thiết kế tất yếu. | Thêm lập luận: Khoảng hở ngang và thời gian va chạm (TTC) trên mặt phẳng $SE(2)$ quyết định an toàn động lực học của xe tự hành. Khẳng định PC-GTA và OGA là bộ đôi đồng thiết kế giữa vật lý cảm biến và tối ưu hóa hình học. |
| **FIX-04** | `MINOR-SCRIPT` | `sections/introduction.tex:5, 9` | **Lọc từ ngữ đệm AI (De-AI)**: Rà soát loại bỏ các cụm từ đệm sáo rỗng (*throat-clearing*) theo `forbidden-terms.md`. | Kiểm tra và tinh gọn văn phong, đảm bảo câu mở đầu mỗi đoạn đạt chuẩn *Claim-Forward*. |

---

### Mục 3: Related Work (`paper/sections/related_work.tex`)

| FIX-ID | Mức độ | Vị trí dòng trong `main.tex` | Mô tả lỗi / Căn cứ phát hiện | Chỉ thị sửa chính xác cho `latex-paper-en` |
|---|---|---|---|---|
| **FIX-05** | `MAJOR` | `sections/related_work.tex:7-8` | **Thiếu tiểu mục / đoạn phân tích Ground-Truth Data Augmentation** (Phát hiện từ Deer-Flow): PC-GTA là đóng góp lớn (+6.03 pp) nhưng Related Work chưa điểm danh các nghiên cứu về GT-Augmentation và vật lý LiDAR. | Thêm 1 đoạn văn phân tích: Phép chắp vá (cut-and-paste) truyền thống từ SECOND đưa vật thể vào bừa bãi gây lỗi phi vật lý (bay lơ lửng, chìm dưới đường, đâm xuyên vật cản tĩnh, sai lệch mật độ tia $1/r^2$). Nêu bật vai trò của các bộ lọc kiểm chứng vật lý. |
| **FIX-06** | `MAJOR` | `sections/related_work.tex:1-6, 11-13` | **Trích dẫn bổ sung Range-View và Modern BEV Transformers**: Thiếu đối sánh với LaserNet và các mô hình Transformer hiện đại (TransFusion, DSVT). | Trích dẫn LaserNet~\cite{meyer2019lasernet} và TransFusion~\cite{bai2022transfusion}. Giải thích rõ Range-View bị biến dạng tỷ lệ còn Transformer vượt quá ngân sách 1M tham số của vi điều khiển nhúng. |

---

### Mục 4: Methodology (`paper/sections/methodology.tex`)

| FIX-ID | Mức độ | Vị trí dòng trong `main.tex` | Mô tả lỗi / Căn cứ phát hiện | Chỉ thị sửa chính xác cho `latex-paper-en` |
|---|---|---|---|---|
| **FIX-07** | `CRITICAL` | `sections/methodology.tex:310-325` | **Cơ chế gán nhãn mục tiêu Category Decoupling trong `GroupedHeader`** (Chất vấn từ DA Reviewer): Cần làm rõ cách gán mục tiêu hồi quy khi các đối tượng khác lớp nằm gần nhau. | Khẳng định rõ: Quyền sở hữu mục tiêu hồi quy được phân lập tuyệt đối theo nhãn lớp mặt đất ở cấp độ tập dữ liệu; các đầu ra cho Nhóm 1 (`car`) và Nhóm 2 (`ped_cyc`) được tính toán độc lập trên các mặt nạ foreground $\mathbf{M}_{\mathrm{reg}, 1}$ và $\mathbf{M}_{\mathrm{reg}, 2}$, loại bỏ hoàn toàn cạnh tranh gradient giữa các lớp. |
| **FIX-08** | `MAJOR` | `sections/methodology.tex:330-345` | **Tính ổn định của hàm góc xoay Doubled-Angle**: Cần hoàn thiện công thức giải mã góc và chứng minh loại bỏ kỳ dị biên $\pm \pi/2$. | Bổ sung phương trình giải mã góc $\hat{\psi} = \frac{1}{2}\operatorname{atan2}(\hat{a}_2, \hat{a}_1)$ kèm chứng minh toán học: vector $(\cos 2\psi, \sin 2\psi)$ ánh xạ liên tục trên đường tròn đơn vị $S^1$, giải quyết triệt để bước nhảy góc $180^\circ$ của phép hồi quy góc vô hướng. |
| **FIX-09** | `MAJOR` | `sections/methodology.tex:120-135` | **Khả thi phần cứng nhúng cho Strata-BEV14 (Pre-emptive defense cho Q4)**: Cần giải thích cách triệt tiêu nút thắt 20.94 ms CPU khi triển khai SoC thực tế. | Bổ sung thuyết minh: Thuật toán rasterize sử dụng các phép tích lũy cục bộ có thể song song hóa hoàn toàn thành CUDA kernel (<3 ms) hoặc chạy đa luồng CPU bất đồng bộ (double-buffering) nạp frame $t+1$ song song với quá trình GPU suy luận frame $t$. |

---

### Mục 5: Experiments & Discussion (`paper/sections/results.tex` & `experiments.tex`)

| FIX-ID | Mức độ | Vị trí dòng trong `main.tex` | Mô tả lỗi / Căn cứ phát hiện | Chỉ thị sửa chính xác cho `latex-paper-en` |
|---|---|---|---|---|
| **FIX-10** | `CRITICAL` | `sections/results.tex:105-112` | **Luận giải bước sụt giảm A7 $\to$ A8 (-0.74 pp) và bứt phá A8 $\to$ A9 (+3.20 pp)** (Chất vấn cốt tử từ DA & Deer-Flow Q2): Cần giải thích tại sao SG-FPN giảm điểm dưới Smooth-L1 nhưng lại tăng vọt khi có OGA. | Bổ sung phân tích hiệp đồng (*co-design synergy*): Dưới Smooth-L1 thông thường, gating chiều sâu làm suy giảm tín hiệu gradient yếu từ các điểm thưa ở xa (khiến mAP giảm nhẹ từ $68.12\%$ xuống $67.38\%$). Tuy nhiên, khi kết hợp với giám sát hình học xoay OGA và xếp hạng chất lượng IQA, SG-FPN cung cấp đúng không gian biểu diễn đa quy mô để OGA khai thác triệt để, tạo ra bước nhảy vọt $+3.20\,\text{pp}$ lên $70.58\%$. |
| **FIX-11** | `MAJOR` | `sections/results.tex:96-100` | **Luận giải hiện tượng tăng vọt +10.08 pp của Head Width H16 $\to$ H32** (Chất vấn từ DA Q2): Reviewer nghi ngờ baseline quá yếu và chỉ tăng kênh để lấy điểm. | Bổ sung phân tích: Đầu H16 bị sụp đổ hạng không gian con (*subspace rank collapse*) khi gánh 4 tác vụ dị thể. Mở rộng lên H32 chỉ tốn thêm 75K tham số nhưng cung cấp đủ chiều không gian con để phân ly gradient triệt để. |
| **FIX-12** | `MAJOR` | `sections/results.tex:148-166` | **Đưa bằng chứng Epoch 47 ($71.48\%$ mAP) vào văn bản** (Phòng thủ phủ đầu đanh thép cho Q3): Phá tan nghi ngờ "cherry-pick đợt chạy 160 epoch". | Đưa số liệu thực nghiệm Table 6 (`code_context.md`) vào bài: Ở Epoch 47 của lịch trình 50 epoch, mAP đã đạt $71.48\%$ và $\text{mAP}_9$ đạt $73.57\%$. Lịch trình 160 epoch không tăng mAP mà chỉ làm sắc nét heatmap, giúp giảm box rác và hạ độ trễ E2E từ $81.23\,\text{ms}$ xuống $60.96\,\text{ms}$. |
| **FIX-13** | `MAJOR` | `sections/results.tex:194-195` | **Thừa nhận giới hạn cảm biến thưa và địa hình phi phẳng (Phòng thủ phủ đầu cho Q1 & Q5)**: Cần trung thực về ranh giới ứng dụng. | Mở rộng mục Limitations: Nêu rõ đánh giá hiện tại dựa trên LiDAR 64 tia mặt đường phẳng; việc thích ứng với LiDAR 16 tia, LiDAR thể rắn quét ngẫu nhiên và dốc nghiêng lớn là bài toán mở đòi hỏi điều biến tần số liên tục (RC-SGFPN). |

---

### Mục 6: Conclusion & Declarations (`paper/sections/conclusion.tex` & `declarations.tex`)

| FIX-ID | Mức độ | Vị trí dòng trong `main.tex` | Mô tả lỗi / Căn cứ phát hiện | Chỉ thị sửa chính xác cho `latex-paper-en` |
|---|---|---|---|---|
| **FIX-14** | `MAJOR` | `sections/conclusion.tex:5-9` | **Đồng bộ hóa kết luận và thông điệp cốt lõi**: Kết luận cần phản ánh đúng sự đồng thiết kế giữa tiền xử lý thống kê, tăng cường dữ liệu vật lý và giám sát hình học. | Khẳng định: Hiệu năng vượt trội trong tầm ngân sách dưới 0.76M tham số là minh chứng cho thấy nhận thức LiDAR biên đạt hiệu quả tối đa thông qua sự đồng thiết kế giữa tiền xử lý thống kê xác định, tăng cường dữ liệu vật lý và giám sát hình học, thay vì mở rộng độ sâu mạng hay dùng PointNet tốn kém. |

---

## 4. MA TRẬN PHÂN LOẠI & ĐỘ ƯU TIÊN THI HÀNH

```
        CRITICAL (Blockers học thuật & toán học)
       ┌────────────────────────────────────────────────────────┐
       │ FIX-07 (Category Target Decoupling)                    │
       │ FIX-10 (Ablation Synergy: SG-FPN Drop & OGA Surge)     │
       └────────────────────────────────────────────────────────┘
                                 │
                                 ▼
        MAJOR (Phòng thủ phủ đầu & Ngữ cảnh tài liệu)
       ┌────────────────────────────────────────────────────────┐
       │ FIX-01 (Abstract SE(2) Scope Qualification)             │
       │ FIX-02 (Intro: Range-View & Point-Based Context)       │
       │ FIX-03 (Intro: Pre-emptive Defense Q1 & Q2)            │
       │ FIX-05 (Related Work: GT Augmentation & Physics Lit.)  │
       │ FIX-06 (Related Work: Citations LaserNet & TransFusion)│
       │ FIX-08 (Methodology: Doubled-Angle Orientation Proof)  │
       │ FIX-09 (Methodology: Embedded SoC Hardware Defense)    │
       │ FIX-11 (Results: Head Capacity Rank Bottleneck)        │
       │ FIX-12 (Results: Epoch 47 mAP Factual Defense)         │
       │ FIX-13 (Results: Limitations on Sparsity & Slopes)     │
       │ FIX-14 (Conclusion: Synthesis of Co-design Pillars)    │
       └────────────────────────────────────────────────────────┘
                                 │
                                 ▼
        MINOR-SCRIPT (Chuẩn hóa De-AI & Tinh gọn văn phong)
       ┌────────────────────────────────────────────────────────┐
       │ FIX-04 (Intro: De-AI & Claim-Forward Polish)           │
       └────────────────────────────────────────────────────────┘
```

---
*Ghi chú*: Bảng điều phối này đã được khóa và sẵn sàng đưa vào áp dụng trực tiếp lên các file mã nguồn LaTeX của bài báo ngay sau khi người dùng phê duyệt kế hoạch.
