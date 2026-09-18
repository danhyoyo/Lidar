# Geometry-Guided Anisotropic Pillar Propagation

## 1. Ý tưởng chính

Thay vì học toàn bộ covariance của Gaussian từ pillar feature, phương pháp sử dụng **hình học cục bộ của LiDAR làm prior** để định hướng Gaussian propagation, sau đó chỉ học một **residual correction**.

Mục tiêu là kiểm tra giả thuyết:

> **Local LiDAR geometry có thể cung cấp inductive bias hữu ích cho directional sparse feature propagation, giúp ổn định và cải thiện learned Gaussian propagation.**

---

## 2. Pipeline đề xuất

Với mỗi pillar \(i\):

1. Lấy neighborhood cục bộ, ví dụ **3×3 neighboring pillars** hoặc radius search.
2. Gom các LiDAR points trong neighborhood.
3. Tính local covariance trên mặt phẳng \(xy\):

\[
C_i^{local}
=
\frac{1}{N}
\sum_k
(q_k-\bar q_i)(q_k-\bar q_i)^T
\]

4. Phân rã covariance để lấy:
   - principal orientation \(\theta_i^{geo}\),
   - mức anisotropy,
   - geometric scale prior.

5. Pillar feature \(f_i\) chỉ học phần residual:

\[
\theta_i = \theta_i^{geo}+\Delta\theta_i
\]

\[
\sigma_{x,i}=\sigma_{x,i}^{geo}e^{\Delta s_{x,i}},
\qquad
\sigma_{y,i}=\sigma_{y,i}^{geo}e^{\Delta s_{y,i}}
\]

6. Tạo covariance cuối:

\[
\Sigma_i
=
R(\theta_i)
\begin{bmatrix}
\sigma_{x,i}^2 & 0\\
0 & \sigma_{y,i}^2
\end{bmatrix}
R(\theta_i)^T
\]

7. Gaussian propagation:

\[
G_{ij}
=
\exp\left(
-\frac12
\Delta x_{ij}^T
\Sigma_i^{-1}
\Delta x_{ij}
\right)
\]

8. Chỉ giữ **Top-K neighbors** để duy trì sparsity.

---

## 3. Vì sao không nên dùng covariance chỉ trong một pillar?

Một pillar có footprint rất nhỏ nên covariance bên trong chính pillar thường chỉ phản ánh point distribution cục bộ trong một vùng hẹp, chưa đủ để biểu diễn hướng của object hoặc surface.

Vì vậy nên dùng:

- 3×3 neighboring pillars,
- radius neighborhood,
- hoặc weighted local neighborhood.

Neighborhood lớn hơn giúp principal direction phản ánh local surface tốt hơn.

---

## 4. Vì sao nên học residual thay vì dùng trực tiếp

Không nên dùng trực tiếp:

\[
\Sigma_i=C_i^{local}+\Delta\Sigma_i
\]

vì covariance cuối có thể không còn positive definite.

Học residual trên **angle và scale** an toàn hơn và dễ kiểm soát hơn.

Geometry cung cấp:

- orientation prior,
- anisotropy prior.

Neural network học:

- correction của orientation,
- propagation scale,
- mức tin cậy vào geometry.

---

## 5. Gated fusion tùy chọn

Có thể thêm một geometry confidence:

\[
\lambda_i
=
\sigma(\mathrm{MLP}[f_i,N_i,d_i,a_i])
\]

và dùng:

\[
\Sigma_i
=
\lambda_i\Sigma_i^{geo}
+
(1-\lambda_i)\Sigma_i^{learned}
\]

Trong đó:

- \(N_i\): số points local,
- \(d_i\): khoảng cách tới ego vehicle,
- \(a_i\): anisotropy score.

Ý tưởng là khi point cloud quá sparse hoặc geometry nhiễu, model có thể giảm mức tin cậy vào geometric prior.

---

## 6. Rủi ro chính

### Geometry không đồng nghĩa object geometry

Local covariance có thể bị ảnh hưởng bởi:

- road points,
- background,
- object lân cận,
- quá ít points ở far range.

Giải pháp đơn giản:

- distance-weighted covariance,
- geometry confidence,
- giới hạn neighborhood,
- learned residual để sửa geometric prior.

---

## 7. So sánh với các hướng gần nhất

### SAFDNet

SAFDNet đã dùng adaptive sparse feature diffusion để mở rộng sparse features sang vùng lân cận.

Khác biệt chính:

- SAFDNet chủ yếu học **bao xa cần diffuse**.
- Phương pháp này học thêm **hướng và hình dạng spatial support** thông qua anisotropic Gaussian được dẫn hướng bởi geometry.

### RadarGaussianDet3D

RadarGaussianDet3D dùng Gaussian ở mức radar point để splat feature sang BEV.

Khác biệt:

- radar point-level Gaussian,
- phương pháp này dùng **LiDAR pillar-level Gaussian**,
- Gaussian shape được anchored vào local LiDAR geometry.

### Local PCA / geometry-based methods

Các phương pháp dùng PCA hoặc local covariance thường sử dụng geometry như một **feature prior**.

Ở đây geometry được dùng để điều khiển trực tiếp **spatial propagation kernel**, tức geometry đóng vai trò một operator thay vì chỉ là input feature.

---

## 8. Ablation tối thiểu

Nên so sánh 4 biến thể:

1. **Isotropic Gaussian**
2. **Learned anisotropic Gaussian**
3. **Geometry-only Gaussian**
4. **Geometry + learned residual**

Tất cả nên dùng cùng một Top-K budget để comparison công bằng.

Metric nên gồm:

- mAP / NDS,
- AP theo distance,
- AP theo số LiDAR points/object,
- số active cells,
- latency.

---

## 9. Đánh giá tính khả thi

- **Implementation:** cao
- **Compute overhead:** thấp đến trung bình
- **Ablation clarity:** rất cao
- **Novelty:** trung bình
- **Khả năng tạo improvement:** khá, nếu local geometry có correlation tốt với object/surface orientation

Phần tốn compute chủ yếu không nằm ở covariance \(2\times2\), mà ở sparse scatter và neighbor aggregation.

---

## 10. Phiên bản MVP đề xuất

Bắt đầu với:

\[
\boxed{
\text{Baseline}
\rightarrow
\text{Learned Anisotropic Gaussian}
\rightarrow
\text{Geometry-Only Gaussian}
\rightarrow
\text{Geometry + Learned Residual}
}
\]

Dùng neighborhood 3×3 và Top-K cố định, ví dụ \(K=4\) hoặc \(K=8\).

Nếu geometry + residual tốt hơn purely learned Gaussian với cùng sparsity budget, hypothesis được hỗ trợ khá mạnh.

---

## 11. Framing contribution

Không nên mô tả đơn giản là “dùng covariance để tạo Gaussian”.

Framing tốt hơn:

> **We use measured local LiDAR geometry as an explicit prior for learning anisotropic sparse feature propagation, rather than learning the spatial propagation kernel entirely from pillar features.**

