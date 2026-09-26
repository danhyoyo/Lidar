# Đặc Tả Kỹ Thuật & Cơ Sở Lý Thuyết: Hàm Loss OGA (Oriented Geometric Alignment & Adaptive Loss)

Trạng thái: **Đề xuất nghiên cứu đóng góp mới (Novel Research Contribution)**  
Ngày: **2026-09-26**  
Tác giả / Nhóm nghiên cứu: Antigravity AI Assistant & DuyenNH  
Áp dụng cho: Mạng phát hiện vật thể 3D LiDAR trên Bird's Eye View (BEVNeXt / MobilePixor)

---

## 1. Bối cảnh & Động lực nghiên cứu (Research Background & Motivation)

### 1.1. Hiện trạng codebase & Giới hạn của UWAG
Trong bài toán phát hiện vật thể 3D định hướng từ đám mây điểm LiDAR (KITTI 3D BEV), mạng dự đoán 4 đầu ra chính:
1. `cls`: Gaussian heatmap biểu diễn tâm vật thể (3 lớp: Car, Pedestrian, Cyclist).
2. `offset`: vector dịch chuyển $(dx, dy)$ (mét) từ tâm ô lưới BEV.
3. `size`: log-kích thước $(\log w, \log l)$ của bounding box.
4. `yaw`: vector góc quay nhân đôi $(\cos 2\theta, \sin 2\theta)$ để đảm bảo tính đối xứng chu kỳ $\pi$ (hướng trước/sau của hộp chữ nhật xoay).

Codebase trước đây có triển khai loss lịch sử mang tên **UWAG** (Uncertainty-Weighted Loss with Adaptive Geometry). Tuy nhiên:
- **Bản quyền / Tính độc lập:** UWAG là ý tưởng và mã nguồn từ nghiên cứu/luận văn trước của người khác, **không được phép sử dụng trực tiếp**, chỉ có thể kế thừa ý tưởng tổng quát.
- **Lỗ hổng hình học cốt lõi (Geometric Flaw):** Phần hình học của UWAG (`_yaw_aware_bev_iou_loss`) tính IoU dựa trên **axis-aligned bounding box** (hộp song song trục toạ độ) rồi nhân với hệ số heuristic góc $\frac{1 + \cos(2\Delta\theta)}{2}$.
  - Đối với các hộp xoay góc $45^\circ$ hoặc $90^\circ$, bounding box song song trục bị biến dạng diện tích nghiêm trọng.
  - Khi hai hộp không giao nhau (disjoint), IoU = 0 dẫn đến gradient triệt tiêu hoàn toàn ($\nabla = 0$).
- **Lỗ hổng tối ưu đa nhiệm (Unbounded Uncertainty Instability):** UWAG sử dụng công thức bất định đồng phương sai không bị chặn của Kendall et al. (CVPR 2018):
  $$\mathcal{L} = \sum_{t} \left( e^{-s_t} \mathcal{L}_t + s_t \right)$$
  Công thức này không có ràng buộc bảo toàn tổng trọng số ($\sum w_t$ không cố định). Trong huấn luyện mixed-precision (BF16/FP16), khi $s_t$ trôi dần, hàm loss có thể bị sụp đổ (collapse) hoặc bùng nổ gradient, như lịch sử repository từng ghi nhận sự cố non-finite ở epoch 81.

### 1.2. Bài học rút ra từ nhánh `feature/loss-function` (B-Series Ablations)
Trên nhánh `feature/loss-function`, các tác giả đã thử nghiệm thay thế UWAG bằng các hàm loss hình học xoay đơn lẻ (single-objective):
- **B0 (Legacy UWAG anchor):** mAP Moderate = **78.20%** (Car: 92.53%, Ped: 64.08%, Cyc: 77.98%).
- **B2 (Full KFIoU - ICLR 2023):** mAP Moderate = **65.49%** (giảm -12.71%).
- **B4 (KLD - NeurIPS 2021):** mAP Moderate = **66.05%** (giảm -12.15%).
- **B3 (ProbIoU - ICCV 2021):** mAP Moderate = **63.37%** (giảm -14.83%).
- **B5 (MGIoU - AAAI 2026):** mAP Moderate = **56.19%** (giảm -22.01%).

#### Tại sao các hàm loss hình học SOTA đơn lẻ (B2-B5) lại sụt giảm mạnh so với UWAG?
1. **Nghịch lý Giám sát Tham số Độc lập vs. Ghép cặp Hình học (The Decoupled vs. Coupled Dilemma):**
   Trong B2-B5, toàn bộ các hàm loss tham số độc lập ($L_{\text{offset}}, L_{\text{size}}, L_{\text{yaw}}$) bị loại bỏ hoàn toàn, chỉ giữ lại duy nhất 1 scalar geometry loss.
   - Khi mạng bắt đầu huấn luyện hoặc khi dự đoán còn cách xa mục tiêu, việc tối ưu đồng thời cả $(x, y, w, l, \theta)$ qua 1 hàm IoU/Gaussian phi lồi là bài toán cực kỳ nghịch lý (ill-conditioned inverse problem).
   - Ngược lại, $L_1$ trên từng tham số riêng rẽ cung cấp vector gradient lồi, trực tiếp và độc lập hướng về ground truth.
2. **Mất cân bằng Gradient Phân loại và Hồi quy (Task Scale Mismatch):**
   B2-B5 cố định trọng số regression = 1.0 và classification = 1.0. Gradient từ Gaussian focal loss trên heatmap $704 \times 800$ và gradient từ geometry loss trên các ô dương tính có biên độ chênh lệch rất lớn, gây triệt tiêu lẫn nhau nếu không có cơ chế thích ứng động.
3. **Bệnh lý Ma trận Hiệp phương sai Gaussian (Covariance Degeneracy):**
   KFIoU, KLD, ProbIoU cần nghịch đảo/cholesky solve trên ma trận $2 \times 2$. Với các đối tượng có tỷ lệ cạnh lớn (như người đi bộ $0.8 \times 0.6\text{m}$, xe đạp), ma trận dễ bị suy biến, đòi hỏi regularizer nhân tạo và buộc phải tắt BF16 autocast.

---

## 2. Thiết Kế Hàm Loss Mới: OGA-Loss (Oriented Geometric Alignment & Adaptive Loss)

Để giải quyết triệt để các hạn chế trên và tạo ra một **đóng góp khoa học lớn, mới lạ và vượt trội** cho đề tài, chúng tôi đề xuất **OGA-Loss** dựa trên 3 trụ cột toán học vững chắc:

```
                                  OGA-Loss Architecture
                                  
                       ┌──────────────────────────────────────────────┐
                       │           Dự đoán từ Detector Head           │
                       │   cls [B,3,H,W]     offset [B,2,H,W]         │
                       │   size [B,2,H,W]    yaw [B,2,H,W]            │
                       └──────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │                                               │
                      ▼                                               ▼
         [Stream 1: Decoupled Params]                    [Stream 2: Rotated Geometry]
         - Focal Loss (cls)                              - Pi-Symmetric Normalized
         - Smooth-L1 (offset)                              Corner Distance (L_NCD)
         - Smooth-L1 (size)                              - Multi-Axis Projection
         - Smooth-L1 (yaw)                                 GIoU (L_proj)
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │   Trụ cột 3: Temperature-Softmax Bounded  │
                        │      Uncertainty Weighting (T-SBUW)       │
                        │    w_t = M * exp(s_t / tau) / sum(exp)    │
                        └─────────────────────┬─────────────────────┘
                                              │
                                              ▼
                                   L_total = sum(w_t * L_t)
```

---

### Trụ Cột 1: Phối Hợp Hình Học Đẳng Cự Đa Trục & Chu Kỳ $\pi$ (Rotated Geometric Alignment - RGA)

Thay vì xấp xỉ bounding box song song trục (như UWAG) và thay vì giải ma trận Gaussian bất ổn (như KFIoU/KLD), OGA-Loss kết hợp hai thành phần hình học xoay chính xác:

#### 1.1. Khoảng cách Đỉnh Hộp Chu Kỳ $\pi$ Chuẩn Hoá Kích Thước (Scale-Normalized $\pi$-Symmetric Corner Distance - $L_{\text{NCD}}$)
Một bounding box xoay 2D được xác định bởi tâm $(x, y)$, chiều dài $l = \exp(\text{size}_1)$, chiều rộng $w = \exp(\text{size}_0)$, và góc xoay $\theta = 0.5 \text{atan2}(\sin 2\theta, \cos 2\theta)$.
4 đỉnh của hộp trong toạ độ BEV là:
$$\mathbf{p}_k = (x, y) + \mathbf{R}(\theta) \begin{pmatrix} \pm l/2 \\ \pm w/2 \end{pmatrix}, \quad k \in \{0, 1, 2, 3\}$$

Vì mạng biểu diễn góc xoay qua vector góc nhân đôi $(\cos 2\theta, \sin 2\theta)$, bounding box có tính đối xứng $\pi$ (quay $180^\circ$ thì hình học không đổi). Do đó, sự tương ứng giữa 4 đỉnh dự đoán $\mathbf{P} = \{\mathbf{p}_k\}$ và 4 đỉnh mục tiêu $\mathbf{G} = \{\mathbf{g}_k\}$ có đúng 2 hoán vị chu kỳ hợp lệ:
- Hoán vị trực tiếp: $\pi_0 = (0, 1, 2, 3)$
- Hoán vị lệch $\pi$: $\pi_1 = (2, 3, 0, 1)$

Khoảng cách đỉnh đối xứng $\pi$ được tính là:
$$D_{\text{corner}}(\mathbf{P}, \mathbf{G}) = \min \left( \frac{1}{4} \sum_{k=0}^3 \|\mathbf{p}_k - \mathbf{g}_k\|_1, \; \frac{1}{4} \sum_{k=0}^3 \|\mathbf{p}_k - \mathbf{g}_{(k+2)\%4}\|_1 \right)$$

Để đối xử công bằng giữa vật thể kích thước nhỏ (Pedestrian $0.8\text{m} \times 0.6\text{m}$) và vật thể lớn (Car $4.0\text{m} \times 1.6\text{m}$), khoảng cách này được chuẩn hoá theo độ dài đường chéo của hộp mục tiêu:
$$\text{diag}_{\text{tgt}} = \sqrt{w_{\text{tgt}}^2 + l_{\text{tgt}}^2}$$
$$\mathcal{L}_{\text{NCD}} = \frac{D_{\text{corner}}(\mathbf{P}, \mathbf{G})}{\text{diag}_{\text{tgt}}}$$

**Ưu điểm vượt trội của $\mathcal{L}_{\text{NCD}}$:**
- Bằng đúng 0 khi và chỉ khi 2 hộp trùng khớp hoàn toàn.
- Gradient tuyến tính, liên tục và khác 0 ở mọi khoảng cách (kể cả khi 2 hộp cách nhau hàng chục mét, không bao giờ bị vanishing gradient).
- Ràng buộc trực tiếp cả tâm $(x, y)$, kích thước $(w, l)$ và hướng xoay $\theta$ trong cùng một đại lượng metric (mét/tỉ lệ).
- Hoàn toàn không cần nghịch đảo ma trận hay slogdet.

#### 1.2. GIoU Chiếu Đa Trục Pháp Tuyến (Multi-Axis Projection GIoU - $\mathcal{L}_{\text{proj}}$)
Để bổ sung thước đo tương quan diện tích giao nhau có chặn $[0, 1]$:
Chiếu 4 đỉnh của prediction và target lên 4 trục pháp tuyến của cả hai hộp (2 trục từ prediction, 2 trục từ target). Trên mỗi trục chiếu $a \in \{1, 2, 3, 4\}$, thu được hai đoạn 1D $[p_{\min}, p_{\max}]$ và $[t_{\min}, t_{\max}]$.
1D Generalized IoU trên trục $a$ được tính:
$$\text{GIoU}_a = \frac{\text{Inter}_a}{\text{Union}_a} - \frac{\text{Hull}_a - \text{Union}_a}{\text{Hull}_a}$$
$$\mathcal{L}_{\text{proj}} = \frac{1 - \frac{1}{4} \sum_{a=1}^4 \text{GIoU}_a}{2}$$

#### 1.3. Tổng Hợp Hình Học Xoay ($\mathcal{L}_{\text{geo}}$):
$$\mathcal{L}_{\text{geo}} = \mathcal{L}_{\text{proj}} + \beta \cdot \mathcal{L}_{\text{NCD}}$$
(với hyperparameter mặc định $\beta = 1.0$).

---

### Trụ Cột 2: Kiến Trúc Giám Sát Kép Độc Lập - Ghép Cặp (Dual-Stream Supervision)

Khắc phục triệt để sự sụt giảm của nhánh B-series bằng cấu trúc 2 luồng:
1. **Luồng Cương Toạ Độ Độc Lập (Decoupled Parameter Stream):**
   - $\mathcal{L}_{\text{cls}}$: Modified Focal Loss trên toàn bộ Gaussian heatmap.
   - $\mathcal{L}_{\text{offset}}$: Smooth-$L_1$ trên $(dx, dy)$ tại các pixel positive.
   - $\mathcal{L}_{\text{size}}$: Smooth-$L_1$ trên $(\log w, \log l)$ tại các pixel positive.
   - $\mathcal{L}_{\text{yaw}}$: Smooth-$L_1$ trên $(\cos 2\theta, \sin 2\theta)$ tại các pixel positive.
2. **Luồng Khớp Hình Học Toàn Diện (Holistic Geometric Stream):**
   - $\mathcal{L}_{\text{geo}}$: Tích hợp đồng thời cả $\mathcal{L}_{\text{proj}}$ và $\mathcal{L}_{\text{NCD}}$.

Nhờ đó, mô hình vừa có lực kéo convex mạnh mẽ từ từng toạ độ riêng biệt ở giai đoạn đầu, vừa được uốn nắn chính xác theo diện tích và góc xoay BEV thực tế.

---

### Trụ Cột 3: Cân Bằng Đa Nhiệm Bất Định Chuẩn Hoá Softmax Nhiệt Độ (Temperature-Softmax Bounded Uncertainty Balancing - T-SBUW)

Khắc phục lỗi trôi bất định và sụp đổ số của Kendall UWAG:
Thay vì trọng số tự do không bảo toàn $e^{-s_t}$, ta định nghĩa vector tham số học $\mathbf{s} = (s_{\text{cls}}, s_{\text{offset}}, s_{\text{size}}, s_{\text{yaw}}, s_{\text{geo}}) \in \mathbb{R}^5$ với khởi tạo $\mathbf{s} = \mathbf{0}$.
Các trọng số tác vụ được chuẩn hoá qua hàm Softmax có nhiệt độ (Temperature $\tau \ge 1.0$):
$$w_t = M \cdot \frac{\exp(s_t / \tau)}{\sum_{j=1}^M \exp(s_j / \tau)}, \quad M = 5$$
với giới hạn biên (soft clamp) $s_t \in [-c, c]$ (ví dụ $c = 3.0$).

**Tính chất toán học ưu việt:**
1. **Bảo Toàn Ngân Sách Gradient (Gradient Scale Conservation):**
   $$\sum_{t=1}^M w_t = M$$
   Mạng không thể "gian lận" bằng cách đẩy mọi $s_t \to \infty$ để hạ thấp hàm loss, và tổng năng lượng gradient truyền về backbone luôn ổn định.
2. **Đảm Bảo Không Bỏ Rơi Tác Vụ (Anti-Task Starvation):**
   Vì $w_t > 0$ và bị chặn dưới bởi $M \frac{e^{-c/\tau}}{(M-1)e^{c/\tau} + e^{-c/\tau}} > 0$, không tác vụ nào bị gán trọng số 0.
3. **Miễn Nhiễm Với Lỗi Số Học BF16/FP32:**
   Softmax luôn hữu hạn và khả vi trơn tru. Không bao giờ xảy ra tình trạng overflow luỹ thừa âm $\exp(-s_t)$ khi $s_t \to -\infty$.

---

## 3. Tổng Hợp Hàm Mục Tiêu OGA (Final Mathematical Formulation)

$$\mathcal{L}_{\text{total}} = \sum_{t \in \{\text{cls}, \text{offset}, \text{size}, \text{yaw}, \text{geo}\}} w_t \cdot \mathcal{L}_t$$

Trong đó:
- $w_t$ được tính động từ module `TemperatureSoftmaxUncertainty`.
- $\mathcal{L}_{\text{cls}}$ tính qua `modified_focal_loss`.
- $\mathcal{L}_{\text{offset}}, \mathcal{L}_{\text{size}}, \mathcal{L}_{\text{yaw}}$ tính qua `smooth_l1_loss` tại mask $reg\_mask = 1$.
- $\mathcal{L}_{\text{geo}} = \mathcal{L}_{\text{proj}} + \beta \mathcal{L}_{\text{NCD}}$ tính qua `OrientedGeometryLoss`.

---

## 4. Bảng So Sánh Đóng Góp Khoa Học (Scientific Contribution Comparison)

| Tiêu chí | UWAG Lịch Sử | Nhánh B-Series (B2-B5) | **Hàm Loss Đề Xuất (OGA-Loss)** |
|---|---|---|---|
| **Bản quyền / Nguồn gốc** | Mã nguồn đồ án cũ của người khác | Thử nghiệm cô lập KFIoU/KLD/MGIoU | **Tác phẩm mới tự thiết kế, 100% độc lập** |
| **Bản chất hình học góc xoay** | Hộp song song trục (Axis-aligned) $\times$ cos heuristic | Gaussian covariance hoặc 1D Projection đơn lẻ | **Chuẩn hoá đỉnh chu kỳ $\pi$ ($L_{\text{NCD}}$) + Chiếu 4 trục ($L_{\text{proj}}$)** |
| **Gradient khi hộp không giao nhau** | Bằng 0 (triệt tiêu gradient) | KFIoU: suy giảm $1/r$; MGIoU: có; KLD: suy biến | **Luôn hữu hạn, tuyến tính, khác 0 ở mọi cự ly** |
| **Cơ chế ổn định số (Stability)** | Dễ overflow/underflow ở BF16 (lỗi epoch 81) | Cần tắt autocast, regularize covariance | **Vô điều kiện hữu hạn (Unconditionally Finite), tương thích 100% BF16** |
| **Cơ chế học đa nhiệm (Multi-Task)** | Kendall $e^{-s} L + s$ không bảo toàn | Trọng số cố định 1.0 (gây tụt 12-22% AP) | **Softmax nhiệt độ chuẩn hoá có chặn (T-SBUW, $\sum w_i = M$)** |
| **Giám sát toạ độ độc lập** | Có ($L_1$) | Bị loại bỏ hoàn toàn | **Giữ vững nguyên lý Giám sát Kép (Dual-Stream)** |
| **Khả năng triển khai suy luận** | Giữ nguyên head contract | Giữ nguyên head contract | **Hoàn toàn giữ nguyên `cls, offset, size, yaw` (Zero Inference Cost)** |

---

## 5. Hợp Đồng Kiểm Thử & Kiểm Chứng (Verification Contracts)

1. **Tính Bất Biến Góc $\pi$ (Pi-Symmetry):**
   Dự đoán góc $\theta$ và $\theta + \pi$ phải sinh ra cùng một giá trị loss $L_{\text{geo}}$ và gradient giống hệt nhau.
2. **Hộp Trùng Khớp Tuyệt Đối (Identical Boxes):**
   Khi prediction trùng target: $L_{\text{NCD}} = 0$, $L_{\text{proj}} = 0 \implies L_{\text{geo}} = 0$.
3. **Tính Dịch Chuyển Không Giao Nhau (Non-Overlap Translation):**
   Khi 2 hộp cách nhau $10\text{m}$, loss hữu hạn và gradient toạ độ khác 0, hướng về mục tiêu.
4. **Bảo Toàn Trọng Số (Weight Conservation):**
   $\sum_{t=1}^5 w_t = 5.0$ trong mọi epoch và mọi bước huấn luyện.
5. **Không Có Điểm Kỳ Dị Trong BF16/FP32:**
   Không có `NaN`, `Inf`, hay exception trên các trường hợp hộp cực dẹt, hộp gần vuông, hoặc toạ độ âm.
6. **Mask Rỗng (Empty Mask):**
   Khi batch không có object (`reg_mask` toàn 0), trả về scalar zero có gắn graph autograd cho toàn bộ prediction heads.
