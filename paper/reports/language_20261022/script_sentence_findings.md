# Mọi cảnh báo câu dài của script và quyết định xử lý

Giữ đủ các lượt chạy file gốc và từng file con, vì hai cách parse có thể tạo ranh giới khác nhau. Các mục lặp là lượt kiểm tra khác nhau, không phải các lỗi độc lập. `Original` của script có thể đã lược bỏ math/cite, nên không dùng làm văn bản thay thế nguồn.

## S001 — before, sections/introduction.tex:5

Script: 35 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Unlike passive optical cameras, which remain vulnerable to ambient illumination fluctuations and projective scale ambiguities, LiDAR sensors actively emit pulsed laser beams to measure direct metric ranges, capturing high-precision geometric boundaries under arbitrary day-and-night conditions~ .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U028](rewrites.md#u028).

## S002 — before, sections/introduction.tex:5

Script: 37 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
In time-critical vehicular environments, perception systems must accurately detect, orient, and classify surrounding objects (from passenger vehicles to compact vulnerable road users such as pedestrians and cyclists) within strict latency envelopes, typically requiring sub-  execution per scan~ .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U028](rewrites.md#u028).

## S003 — before, sections/introduction.tex:7

Script: 30 từ; 7 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
First, volumetric voxel-based architectures, exemplified by SECOND~ , and two-stage point-voxel frameworks, such as PointRCNN~ , Part- ~ , and PV-RCNN~ , achieve competitive detection accuracy by deploying 3D sparse convolutions or localized point-cloud abstractions.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U029](rewrites.md#u029).

## S004 — before, sections/introduction.tex:7

Script: 22 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
While faster than volumetric methods, non-deterministic point gathering, dynamic memory allocation, and sparse scattering introduce substantial runtime overhead and memory transfer latency.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U029](rewrites.md#u029).

## S005 — before, sections/introduction.tex:7

Script: 41 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Although conceptually efficient, existing rasterization methods suffer from severe spatial quantization errors on standard   grids; by relying on crude binary occupancy slices, they discard continuous subcell offsets and high-order vertical dispersion statistics, inducing spatial staircasing and localization jitter on small objects.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U029](rewrites.md#u029).

## S006 — before, sections/introduction.tex:9

Script: 78 từ; 4 bộ đếm mệnh đề. Nhóm: `question-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Grounded in this principle, we investigate the central research question: \textit{Can an ultra-compact LiDAR detector operating on a parameter-free BEV tensor rival the localization accuracy of heavy volumetric and pillar architectures while achieving superior end-to-end throughput on commodity edge hardware?} To realize this objective, our framework couples this deterministic representation with a single-branch ConvNeXt-style backbone~  utilizing large   depthwise kernels and parameter-free 3D spatial-energy attention (SimAM)~ , expanding physical receptive fields to envelop entire vehicles without multi-branch training overhead.
```

Script gộp câu hỏi kết thúc bằng ?} với câu tiếp theo; kiểm tra riêng từng câu. [Đối chiếu U030](rewrites.md#u030).

## S007 — before, sections/introduction.tex:11

Script: 21 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Guided by this design philosophy, we propose \textbf{MobilePIXORNeXt}, an ultra-compact, proposal-free, multi-class 3D LiDAR object detector containing exactly   trainable parameters ( ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S008 — before, sections/introduction.tex:11

Script: 57 từ; 5 bộ đếm mệnh đề. Nhóm: `prose-reviewed`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
An unordered LiDAR point cloud is deterministically transformed via the parameter-free \textbf{Strata-BEV14} rasterizer into a 14-channel statistical BEV tensor (  at   resolution) using NumPy or Numba on CPU, with point-cloud loading and rasterization taking   in the measured pipeline, capturing height-band densities, vertical spatial moments ( ), surface reflectance statistics, and continuous subcell centroid displacement offsets   to eliminate quantization error.
```

Đã biên tập văn xuôi ở đơn vị liên kết; báo cáo gốc được giữ để đối chiếu. [Đối chiếu U031](rewrites.md#u031).

## S009 — before, sections/introduction.tex:11

Script: 36 từ; 8 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
To regularize training without unphysical artifacts, we formulate \textbf{Physics-Consistent Ground-Truth Augmentation (PC-GTA)}, which combines local ground elevation fitting, static obstacle clearance, and range-adaptive point subsampling, with optional line-of-sight checks, raycast shadow masking, and intensity attenuation.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S010 — before, sections/introduction.tex:11

Script: 36 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Multi-scale feature extraction is performed by the \textbf{MobilePixorNeXt-S} backbone hierarchy across stages  , utilizing streamlined   depthwise inverted bottleneck blocks ( ), SiLU activations, LayerScale stabilization, and embedded parameter-free \textbf{Block-SimAM} energy attention operating inside all nine residual blocks.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S011 — before, sections/introduction.tex:11

Script: 33 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Multi-scale semantic context and fine coordinates are balanced by the top-down \textbf{Scale-Gated Feature Pyramid Network (SG-FPN Neck,  )}, which employs bilinear upsampling, depthwise refinement, and learnable scale gates to eliminate deconvolution checkerboard artifacts.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S012 — before, sections/introduction.tex:11

Script: 32 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
By deterministically capturing continuous geometry at the input, expanding receptive fields via single-branch large-kernel convolution, and supervising rotated overlap directly, this coherent pipeline resolves each aforementioned bottleneck within a unified, edge-deployable framework.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S013 — before, sections/introduction.tex:17

Script: 36 từ; 7 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\item \textbf{Physics-consistent data augmentation and oriented geometric supervision.} We formulate \textbf{PC-GTA}, which combines local terrain alignment, obstacle clearance, and range-dependent point subsampling, and supports optional visibility, shadow-masking, and intensity-adjustment heuristics to reduce synthesis artifacts.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U033](rewrites.md#u033).

## S014 — before, sections/introduction.tex:19

Script: 27 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
In complete end-to-end pipeline speed, it operates at \textbf{16.41\,FPS} (\textbf{60.96\,ms} on NVIDIA Tesla T4), running faster than all evaluated real-time detectors.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U034](rewrites.md#u034).

## S015 — before, sections/related_work.tex:6

Script: 31 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
On one hand, PointNet-based pillar encoders~  achieve expressive geometric learning, yet the non-deterministic point gathering, dynamic memory allocation, and sparse scatter operations incur prohibitive latency and energy penalties on edge accelerators.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U114](rewrites.md#u114).

## S016 — before, sections/related_work.tex:6

Script: 38 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
By deterministically integrating four stratified vertical density bins, four continuous elevation moments ( ), and two continuous subcell centroid displacement offsets ( ) into a 14-channel layout, Strata-BEV14 preserves sub-voxel geometric fidelity while reducing Host-to-Device transfer payloads by   without learned encoder latency.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U114](rewrites.md#u114).

## S017 — before, sections/related_work.tex:13

Script: 29 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
The ConvNeXt paradigm~  demonstrated that conventional convolutions modernized with   depthwise kernels, inverted bottleneck expansions ( ), and LayerScale can achieve representational power competitive with Vision Transformers while preserving convolutional hardware efficiency.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U116](rewrites.md#u116).

## S018 — before, sections/related_work.tex:24

Script: 26 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Third, unconstrained multi-task uncertainty weighting allows task variance parameters to diverge, leading to training instability when balancing disparate loss scales across classification, geometric overlap, and orientation.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U120](rewrites.md#u120).

## S019 — before, sections/related_work.tex:24

Script: 73 từ; 5 bộ đếm mệnh đề. Nhóm: `prose-reviewed`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Specifically, OGA unifies multi-axis MGIoU projections with scale-normalized  -symmetric corner supervision to eliminate boundary discontinuities; Q-OGA couples continuous IoU quality directly into classification supervision to achieve quality calibration with zero head overhead; IQA provides explicit multi-branch quality alignment scored via   to suppress miscalibrated false positives; and Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW) guarantees stable convergence by constraining multi-task weights within a normalized, bounded simplex, simultaneously resolving orientation discontinuities, quality miscalibration, and loss scale divergence.
```

Đã biên tập văn xuôi ở đơn vị liên kết; báo cáo gốc được giữ để đối chiếu. [Đối chiếu U120](rewrites.md#u120).

## S020 — before, sections/methodology.tex:5

Script: 15 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Decoupled grouped prediction towers output heatmaps, subcell offsets, log-sizes, doubled yaw angles, and localization quality.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S021 — before, sections/methodology.tex:5

Script: 32 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Solid arrows trace the feedforward inference path, whereas dashed arrows indicate training-only supervision incorporating Oriented Geometric Alignment (OGA), IoU-Aware Quality Alignment (IQA), and dynamic loss balancing via Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW).}
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U036](rewrites.md#u036).

## S022 — before, sections/methodology.tex:18

Script: 45 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{LiDAR Coordinate Frame and Spatial Region of Interest.} An unordered LiDAR point cloud is acquired via a rotating scanner and defined as a set  , where each point   encapsulates longitudinal distance   (forward positive), lateral distance   (leftward positive), vertical elevation   (upward positive), and calibrated reflectance intensity  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U038](rewrites.md#u038).

## S023 — before, sections/methodology.tex:18

Script: 33 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Perception is constrained to a fixed Cartesian Region of Interest (ROI): \begin{equation} \begin{aligned} x_i &\in [x_{\min}, x_{\max}], \\ y_i &\in [y_{\min}, y_{\max}], \\  z_i &\in [z_{\min}, z_{\max}].
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U038](rewrites.md#u038).

## S024 — before, sections/methodology.tex:26

Script: 14 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{equation} configured with longitudinal bounds  \,m, lateral bounds  \,m, and vertical bounds  \,m.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U039](rewrites.md#u039).

## S025 — before, sections/methodology.tex:33

Script: 25 từ; 8 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
For each detected object instance  , the target state is parameterized by the 7-tuple: \begin{equation} \mathbf{b}_k = (c_k, x_k^*, y_k^*, l_k^*, w_k^*, \psi_k^*, s_k),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S026 — before, sections/methodology.tex:38

Script: 27 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Operating on this contract requires a structured, dense BEV representation that captures fine 3D geometric distributions without the latency of learned point encoders, which we formulate next.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U041](rewrites.md#u041).

## S027 — before, sections/methodology.tex:45

Script: 70 từ; 7 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Channels 0–3 capture stratified height-band densities, Channels 4–7 encode vertical elevation statistics (peak, mean, span, deviation), Channels 8–9 capture material surface reflectance, Channels 10–11 represent global density and occupancy, and Channels 12–13 encode continuous subcell horizontal offsets to eliminate grid discretization error.} \captionof{figure}{Visual breakdown of the fourteen deterministic feature channels in the Strata-BEV14 representation for a representative KITTI driving scene (  grid at   resolution).
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S028 — before, sections/methodology.tex:45

Script: 68 từ; 3 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Channels 0--3 capture logarithmic point densities across four uniform vertical elevation strata ( ); Channels 4--7 encode vertical spatial moments (normalized maximum elevation  , mean elevation  , elevation span  , and sample standard deviation  ); Channels 8--9 capture calibrated surface reflectance statistics (maximum and mean intensity); Channel 10 records total cell logarithmic density; Channel 11 provides a binary occupancy indicator; and Channels 12--13 preserve continuous subcell horizontal centroid displacements   to mitigate spatial discretization errors.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U042](rewrites.md#u042).

## S029 — before, sections/methodology.tex:69

Script: 52 từ; 9 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textit{Vertical Spatial Moments ( -- )}: To encode obstacle geometry without explicit voxelization, we extract peak elevation  , mean elevation  , vertical span  , and the sample standard deviation: \begin{equation} F_7 = \begin{cases} \sqrt{\max\left(0, \frac{1}{N} \sum_{i=1}^N \zeta_i^2 - F_5^2\right)}, & \text{if } N > 1, \\ 0, & \text{otherwise}.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S030 — before, sections/methodology.tex:78

Script: 39 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textit{Reflectance Statistics ( -- )}: Calibrated surface returns capture material discrimination (e.g., distinguishing retro-reflective license plates, pedestrian clothing, or road signs): \begin{equation} F_8 = \max_{i=1,\dots,N} \iota_i, \qquad F_9 = \frac{1}{N} \sum_{i=1}^N \iota_i.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S031 — before, sections/methodology.tex:84

Script: 82 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
To prevent quantization error from coarse  \,m cell discretization, channels   and   preserve continuous subcell horizontal centroid displacements: \begin{align} x_u^{\mathrm{cell}} &= x_{\min} + (u + 0.5)\Delta_x, \nonumber\\ y_v^{\mathrm{cell}} &= y_{\min} + (v + 0.5)\Delta_y, \nonumber\\ F_{12} &= \operatorname{clip}\left(\frac{\sum_{i=1}^N (x_i - x_u^{\mathrm{cell}})}{N\Delta_x}, -0.5, 0.5\right), \nonumber\\ F_{13} &= \operatorname{clip}\left(\frac{\sum_{i=1}^N (y_i - y_v^{\mathrm{cell}})}{N\Delta_y}, -0.5, 0.5\right).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U049](rewrites.md#u049).

## S032 — before, sections/methodology.tex:127

Script: 31 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{Motivation.} Cut-and-paste ground-truth (GT) augmentation balances rare LiDAR categories, but naive insertion ignores scene physics, producing floating or submerged objects, background collisions, missing raycast shadows, and unrealistic range-dependent point densities.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S033 — before, sections/methodology.tex:130

Script: 83 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
The mechanisms can be enabled independently; the settings used for the reported final run are given in Section~ : \begin{enumerate} \item \textit{Adaptive Ground Plane Fitting}: To prevent floating or sunken objects, local road elevation is estimated within a range-dependent search radius: \begin{align} R(d) &= \operatorname{clip}(2.5 + 0.03\max(0, d - 15), 2.5, 4.0)\,\mathrm{m},\nonumber\\[0.8em] \hat{z}_{\mathrm{ground}} &= \operatorname{percentile}(\{z_j \mid \mathbf{p}_j \in \mathcal{N}(x, y)\}, 5),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S034 — before, sections/methodology.tex:146

Script: 53 từ; 1 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Points residing behind the exit face ( ) with radial distance  \,m are masked out: \begin{equation} \begin{aligned} \mathcal{P}_{\mathrm{augmented}} &= \mathcal{P}_{\mathrm{target}} \setminus \{\mathbf{p} \in \mathcal{P}_{\mathrm{target}} \mid \\ &\qquad \operatorname{Mask}_{\mathrm{shadow}}(\mathbf{p}) = \mathrm{True}\} \cup \mathcal{P}_{\mathrm{object}}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U060](rewrites.md#u060).

## S035 — before, sections/methodology.tex:158

Script: 50 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\item \textit{Range-Adaptive Point and Intensity Attenuation}: If an object captured at source distance   is placed at target distance  , its point cloud is randomly subsampled according to the inverse-square divergence law: \begin{equation} P_{\mathrm{keep}} = \min\left(1, \left(\frac{r_{\mathrm{src}}}{r_{\mathrm{tgt}}}\right)^2\right),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S036 — before, sections/methodology.tex:162

Script: 47 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{equation} and the optional radiometric adjustment attenuates reflectance intensity according to: \[ \begin{aligned} \eta_{\mathrm{int}} &= \left(\frac{r_{\mathrm{src}}}{r_{\mathrm{tgt}}}\right)^{\max(0, 2 - \gamma)}, \qquad \gamma = 1.7, \\ I_{\mathrm{tgt}} &= \operatorname{clip}(I_{\mathrm{src}}\eta_{\mathrm{int}}, 0, 1).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S037 — before, sections/methodology.tex:180

Script: 38 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
A two-layer convolutional stem ingests 14-channel input at stride 2 ( ), followed by three hierarchical stages (C2: 48 channels, depth 3; C3: 96 channels, depth 4; C4: 128 channels, depth 2) composed of \texttt{MobilePixorNeXt-S} blocks with embedded Block-SimAM.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U063](rewrites.md#u063).

## S038 — before, sections/methodology.tex:191

Script: 48 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
In contrast, at our input grid resolution of  \,m, a single   depthwise receptive field directly spans  \,m of physical terrain at stride 1, and rapidly expands to over  \,m by Stage 3 (stride 8)---sufficient to envelop an entire passenger vehicle in a single layer without dilation artifacts.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U064](rewrites.md#u064).

## S039 — before, sections/methodology.tex:194

Script: 49 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{Backbone Topology and Multi-Stage Hierarchy.} As depicted in Figure~ , the network topology comprises: \begin{enumerate} \item \textit{Convolutional Stem}: Input   is ingested by two consecutive   convolutions (stride 2 followed by stride 1) with 32 output channels, Batch Normalization (BN), and Sigmoid Linear Unit (SiLU) activations, yielding stride-2 representation  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U065](rewrites.md#u065).

## S040 — before, sections/methodology.tex:214

Script: 29 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
(Top) Complete feedforward path from input X to output Y through the 7x7 depthwise convolution, inverted bottleneck expansion (e=2.5), SimAM modulation, LayerScale scaling, and identity residual addition.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S041 — before, sections/methodology.tex:214

Script: 39 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
(Bottom) Zoom-in structure of the parameter-free SimAM module, detailing spatial moment estimation, inverse-energy importance calculation, sigmoid saliency mapping, and element-wise feature recalibration.} \captionof{figure}{Micro-architecture of the \texttt{MobilePixorNeXt-S} residual block and the embedded parameter-free Block-SimAM energy attention module.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S042 — before, sections/methodology.tex:214

Script: 36 từ; 7 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textbf{Top}: Complete residual transformation path from input   to output  , featuring a large-kernel   depthwise convolution, inverted bottleneck expansion with ratio  , SiLU activation, pointwise projection, parameter-free 3D spatial-energy attention  , LayerScale scaling   (initialized to  ), and an identity shortcut.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U071](rewrites.md#u071).

## S043 — before, sections/methodology.tex:214

Script: 37 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textbf{Bottom}: Internal operational mechanism of the SimAM operator, which calculates channel-wise spatial moments ( ) in FP32 precision, evaluates visual saliency via inverse-energy importance  , and produces a 3D modulation tensor   via sigmoid scaling without introducing any learnable parameters.}
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U071](rewrites.md#u071).

## S044 — before, sections/methodology.tex:221

Script: 57 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Given input tensor  , each block applies an inverted bottleneck with expansion ratio  : \begin{align} \mathbf{V} &= \operatorname{BN}(\operatorname{DWConv}_{7\times7}(\mathbf{Z})),\nonumber\\ \mathbf{R} &= \operatorname{BN}\left(W_{\mathrm{red}} \operatorname{SiLU}(W_{\mathrm{exp}}\mathbf{V})\right),\nonumber\\ \mathbf{Z}' &= \mathbf{Z} + \boldsymbol{\gamma}_{\mathrm{blk}} \odot \mathcal{S}(\mathbf{R}).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S045 — before, sections/methodology.tex:234

Script: 31 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{Design and Placement.} As illustrated in Figure~ , SimAM is embedded inside each \texttt{MobilePixorNeXt-\allowbreak S} block across all three stages (C2, C3, C4), for nine attention operators in total.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S046 — before, sections/methodology.tex:237

Script: 44 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
For each channel slice with   spatial locations, spatial mean   and sample variance   are computed: \begin{equation} \begin{aligned} \mu_c &= \frac{1}{M}\sum_{u, v} R_{c, u, v}, \\ \nu_c &= \frac{1}{\max(M-1, 1)}\sum_{u, v} (R_{c, u, v} - \mu_c)^2.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S047 — before, sections/methodology.tex:244

Script: 56 từ; 2 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{equation} Following the energy-minimization formulation of SimAM~ , the minimized energy   and its inverse-energy importance score   for neuron   are: \begin{equation} \begin{aligned} e_t^\star &= \frac{4(\nu_c + \lambda_{\mathrm{sim}})}{(t-\mu_c)^2 + 2(\nu_c + \lambda_{\mathrm{sim}})}, \\ s_t &= \frac{1}{e_t^\star} = \frac{(t - \mu_c)^2}{4(\nu_c + \lambda_{\mathrm{sim}})} + \frac{1}{2}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S048 — before, sections/methodology.tex:282

Script: 27 từ; 8 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{Mathematical Formulation.} As shown in Figure~ , lateral features from Stages C4 ( \,ch), C3 ( \,ch), and C2 ( \,ch) are projected via   convolutions into matching lateral dimensions  ,  , and  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S049 — before, sections/methodology.tex:285

Script: 51 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
At each fusion stride  , lateral and top-down representations are adaptively balanced via learnable depthwise scale gating: \begin{align} \mathbf{G}_r &= 2\sigma\left(\operatorname{DWConv}_{3\times3, r}(\mathbf{L}_r + \mathbf{U}_r)\right),\nonumber\\ \mathbf{P}^{(r)} &= \mathbf{U}_r + \mathbf{G}_r \odot \mathbf{L}_r.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U084](rewrites.md#u084).

## S050 — before, sections/methodology.tex:296

Script: 81 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\par\vspace{-0.5\baselineskip} \noindent \begin{minipage}{\columnwidth} \centering \paperfigure{geometry_output_2x2.pdf}{75mm}{Geometric parameterization, orientation encoding, and quality-aware score calibration across prediction heads: (a) Continuous subcell coordinate offsets   from anchor point   to object center  ; (b) Log-transformed bounding box dimension targets  ; (c) Doubled-angle trigonometric yaw representation   on the unit circle resolving  -heading ambiguity; and (d) IoU-quality-aware composite score ranking   calibrated for post-processing peak selection and rotated NMS.} \captionof{figure}{Geometric parameterization, orientation formulation, and quality-aware score calibration across prediction heads.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U086](rewrites.md#u086).

## S051 — before, sections/methodology.tex:306

Script: 20 từ; 11 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Cars exhibit rigid rectangular footprints ( \,m,  \,m), whereas Pedestrians ( \,m,  \,m) and Cyclists ( \,m,  \,m) are compact and highly deformable.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U087](rewrites.md#u087).

## S052 — before, sections/methodology.tex:319

Script: 30 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
For output cell  , the ground-truth geometric targets are formulated as: \begin{align} \mathbf{o}^* &=  ,\nonumber\\ \mathbf{d}^* &= (\log w^*, \log l^*), \qquad \mathbf{a}^* = (\cos 2\psi^*, \sin 2\psi^*).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S053 — before, sections/methodology.tex:333

Script: 62 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textit{Multi-Axis Projection GIoU ( )}: Projected bounding box overlap is evaluated following Marginalized Generalized IoU~  across four unit normal projection axes defining the prediction and ground-truth rectangles: \begin{equation} \mathbf{n}_k \in \left\{ \mathbf{n}_{\mathrm{len}}^{\mathrm{pred}}, \mathbf{n}_{\mathrm{wid}}^{\mathrm{pred}}, \mathbf{n}_{\mathrm{len}}^{\mathrm{gt}}, \mathbf{n}_{\mathrm{wid}}^{\mathrm{gt}} \right\}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U096](rewrites.md#u096).

## S054 — before, sections/methodology.tex:339

Script: 52 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{equation} For each axis  , 1D intervals of intersection length  , union length  , and smallest enclosing hull   are determined, yielding 1D GIoU~ : \begin{equation} \begin{aligned} g_k &= \frac{I_k}{U_k} - \frac{E_k - U_k}{E_k}, \\ \bar{g} &= \frac{1}{4}\sum_{k=1}^4 g_k, \qquad \ell_{\mathrm{proj}} = \frac{1 - \bar{g}}{2}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S055 — before, sections/methodology.tex:348

Script: 78 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Scale-Normalized  -Symmetric Corner Distance ( )}:\par}\noindent To ensure accurate vertex localization under doubled-angle symmetry, we evaluate the minimum L1 distance across direct and  -permuted corner correspondences: \begin{equation} \ell_{\mathrm{corner}} = \frac{\min\left\{ \begin{array}{l} \displaystyle\sum_{m=1}^4 \|\widehat{\mathbf{c}}_m - \mathbf{c}_m^*\|_1, \\ \displaystyle\sum_{m=1}^4 \|\widehat{\mathbf{c}}_m - \mathbf{c}_{5-m}^*\|_1 \end{array} \right.}{4\max\left(\sqrt{(w^*)^2 + (l^*)^2}, 10^{-6}\right)}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S056 — before, sections/methodology.tex:362

Script: 38 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
It is supervised via Binary Cross-Entropy (BCE) using the detached MGIoU similarity as the ground-truth target: \begin{equation} q^* = \operatorname{stopgrad}(\operatorname{clip}(\bar{g}, 0, 1)), \qquad \mathcal{L}_{\mathrm{qual}} = \operatorname{BCE}(\hat{q}, q^*).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S057 — before, sections/methodology.tex:383

Script: 74 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textit{Bounded Softmax Weights}: Learnable logit parameters   are soft-clamped via hyperbolic tangent with bound  , and converted to positive normalized weights via temperature-controlled softmax ( ): \begin{equation} \begin{aligned} \bar{a}_k &= b_{\mathrm{uw}} \tanh(a_k / b_{\mathrm{uw}}), \\ Z_{\mathrm{uw}} &= \sum_{j \in \mathcal{T}} \exp(\bar{a}_j / \tau_{\mathrm{uw}}), \\ \omega_k &= J \cdot \frac{\exp(\bar{a}_k / \tau_{\mathrm{uw}})}{Z_{\mathrm{uw}}}, \\ \sum_{k \in \mathcal{T}} \omega_k &= J.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S058 — before, sections/methodology.tex:411

Script: 61 từ; 4 bộ đếm mệnh đề. Nhóm: `algorithm`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\vspace{2pt} {\small \noindent\textbf{Require:} Raw LiDAR point cloud  , ROI bounds  \par \noindent\textbf{Require:} Score threshold  , Rotated NMS threshold  , quality exponent  \par \noindent\textbf{Ensure:} Final decoded 3D oriented bounding boxes  \par \begin{enumerate} \setlength{\topsep}{2pt} \setlength{\itemsep}{0pt} \setlength{\parsep}{0pt} \renewcommand{\labelenumi}{\textbf{\arabic{enumi}:}} \item Filter raw points   within inward margin  \,m of spatial ROI.
```

Giữ nguyên thuật toán; danh sách đầu vào/đầu ra không phải câu văn cần tách.

## S059 — before, sections/methodology.tex:411

Script: 20 từ; 5 bộ đếm mệnh đề. Nhóm: `algorithm`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\item Evaluate Grouped Prediction Towers for Car ( ) and Pedestrian/Cyclist ( ), yielding classification maps  , subcell offsets  , log-sizes  , doubled yaws  , and quality  .
```

Giữ nguyên thuật toán; danh sách đầu vào/đầu ra không phải câu văn cần tách.

## S060 — before, sections/methodology.tex:441

Script: 75 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
For each candidate cell   where predicted classification probability   forms a   local spatial peak, metric continuous bounding boxes are decoded via: \begin{align} \hat{x} &= x_{\min} + u s_{\mathrm{out}}\Delta_x + \hat{o}_x, \nonumber\\ \hat{y} &= y_{\min} + v s_{\mathrm{out}}\Delta_y + \hat{o}_y, \nonumber\\ \hat{w} &= \exp(\hat{d}_1), \qquad \hat{l} = \exp(\hat{d}_2), \nonumber\\ \hat{\psi} &= \frac{1}{2}\operatorname{atan2}(\hat{a}_2, \hat{a}_1).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U106](rewrites.md#u106).

## S061 — before, sections/methodology.tex:458

Script: 32 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{Parameter Distribution Breakdown.} The complete MobilePIXORNeXt detector contains exactly \textbf{753,569 trainable parameters} ( \,M), corresponding to an ultra-compact FP32 checkpoint size of  \,MB: \begin{itemize} \item \textbf{Backbone Body}:   parameters ( ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U110](rewrites.md#u110).

## S062 — before, sections/methodology.tex:458

Script: 25 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{itemize} This compact footprint enables full end-to-end inference at  \,FPS ( \,ms) and network forward throughput of  \,FPS ( \,ms) on commodity NVIDIA Tesla T4 GPUs.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U112](rewrites.md#u112).

## S063 — before, sections/experiments.tex:7

Script: 37 từ; 8 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Perception is bounded within the Cartesian Region of Interest: \begin{equation} \begin{aligned} x &\in [0.0, 70.4]\,\text{m}, \\ y &\in [-40.0, 40.0]\,\text{m}, \\ z &\in [-2.5, 1.0]\,\text{m}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U014](rewrites.md#u014).

## S064 — before, sections/experiments.tex:18

Script: 60 từ; 2 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{Evaluation Metrics.} Detection performance is benchmarked on oriented 2D Bird's-Eye View (BEV) footprints using 40-point interpolated Average Precision (AP  ) on the KITTI validation set, which calculates the area under the precision--recall curve over 40 recall positions  : \begin{equation} \mathrm{AP}|_{R_{40}} = \frac{1}{40} \sum_{r \in \mathcal{R}_{40}} \max_{r' \ge r} p(r').
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U016](rewrites.md#u016).

## S065 — before, sections/experiments.tex:18

Script: 43 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{itemize} Following autonomous driving literature, \textbf{Moderate macro mAP} ( ) was adopted as the primary benchmark ranking criterion: \begin{equation} \mathrm{mAP}_{\mathrm{M}} = \frac{1}{3} \sum_{c \in \{\mathrm{Car}, \mathrm{Ped}, \mathrm{Cyc}\}} \mathrm{AP}_{c, \mathrm{M}}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S066 — before, sections/experiments.tex:18

Script: 11 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Here, E, M, and H denote Easy, Moderate, and Hard regimes.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U023](rewrites.md#u023).

## S067 — before, sections/experiments.tex:42

Script: 39 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
The network comprises a two-layer   convolutional stem (  channels, stride 2), three hierarchical backbone stages with channel widths   and block depths   based on \texttt{MobilePixorNeXt-S} blocks with large   depthwise convolutions, inverted expansion ratio  , SiLU activations, and LayerScale parameters initialized to  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U024](rewrites.md#u024).

## S068 — before, sections/experiments.tex:48

Script: 30 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
In the final reported run, local ground fitting, static obstacle clearance, range-adaptive point subsampling, and visibility protection were enabled; LOS rejection, shadow masking, and radiometric calibration were available but disabled.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U026](rewrites.md#u026).

## S069 — before, sections/experiments.tex:51

Script: 26 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{Batching, Precision, and Hardware Environment.} Training used a physical batch size of 16 on a single GPU, while validation used a batch size of 32.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S070 — before, sections/experiments.tex:51

Script: 32 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Numerical forward and backward passes utilized Automatic Mixed Precision (AMP) in Bfloat16 (\texttt{bf16}), while loss evaluation, coordinate regression, and rotated NMS were conducted in full FP32 precision to maintain numerical stability.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U027](rewrites.md#u027).

## S071 — before, sections/experiments.tex:51

Script: 40 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
The final MobilePIXORNeXt timing and accuracy evaluation was performed on a server equipped with an NVIDIA Tesla T4 GPU (16\,GB GDDR6 VRAM, Turing architecture), host x86\_64 CPU, Linux OS, and PyTorch 2.11.0 with CUDA 13.0.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U027](rewrites.md#u027).

## S072 — before, tables/setup.tex:5

Script: 171 từ; 12 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{.38\columnwidth}>{\raggedright\arraybackslash}p{.56\columnwidth}@{}} \toprule \textbf{Configuration Element} & \textbf{Specification / Value} \\ \midrule LiDAR Input Representation & Strata-BEV14 (14 channels; NumPy/Numba CPU) \\ Spatial ROI bounds ( ) &   \\ Grid resolution ( ) &   \\ Input / Output Grid Dimensions &   /   ( ) \\ Detector Parameters & 753,569 trainable ( ); 9.34\,MB FP32 \\ Backbone Stages ( ) & Widths: (48, 96, 128); Depths: (3, 4, 2) \\ Inverted Bottleneck Expansion &  ;   Depthwise Convolution \\ Attention Mechanism & Parameter-free Block-SimAM (9 residual blocks) \\ Multiscale Neck & Scale-Gated FPN (SG-FPN,   fusion width) \\ Prediction Towers & GroupedHeader: Car ( ), Ped/Cyc ( ) \\ Loss Formulation & OGA (MGIoU +  -Corner) + IQA + T-SBUW \\ Data Augmentation & Physics-Consistent GT (PC-GTA) + Global \\ Optimizer / Base Learning Rate & AdamW (fused) /   \\ Minimum LR / Standard Schedule &   / 50 epochs: 4 warmup + 46 cosine \\ Final Reported Checkpoint & Extended run: 160 epochs, 10 warmup; selected at epoch 130 \\ Weight Decay &   (2D+ tensors only; biases/norm: 0.00) \\ Gradient Clipping Norm & Maximum   norm = 10.00 \\ Batch Size (Train / Val) & 16 / 32 \\ Numerical Precision & Automatic Mixed Precision (Bfloat16 / FP32) \\ Hardware Environment & NVIDIA Tesla T4 GPU (16\,GB GDDR6, 98.44\,MB peak) \\ \bottomrule \end{tabular} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S073 — before, tables/comparison.tex:5

Script: 176 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular}{@{}lrrrrrr@{}} \toprule \textbf{Method} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Car AP \\(\%)  }} & \textbf{\shortstack{Pedestrian\\AP  (\%)  }} & \textbf{\shortstack{Cyclist\\AP  (\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} \\ \midrule PIXOR (Car only)~  & 2.135 & 72.02 & -- & -- & -- & -- \\ MobilePIXOR~  & \textbf{0.598} & 75.52 & 28.16 & 20.53 & 41.40 & 42.27 \\ PointPillars~  & 4.835 & \underline{88.02} & 56.01 & 66.34 & 70.13 & 72.25 \\ SECOND~  & 5.326 & \textbf{88.56} & \underline{56.56} & \textbf{71.16} & \textbf{72.09} & \textbf{73.79} \\ \textbf{MobilePIXORNeXt (Ours)} & \underline{0.754} & 86.90 & \textbf{60.75} & \underline{66.73} & \underline{71.46} & \underline{73.42} \\ \midrule \multicolumn{7}{@{}l}{\textit{Heavyweight 3D Anchor-Based Reference Detectors}} \\ PointRCNN~  & 4.039 & 88.42 & 57.84 & 73.87 & 73.38 & 75.31 \\ Part-A -Anchor~  & 63.815 & 90.05 & 64.20 & 74.60 & 76.28 & 78.04 \\ PV-RCNN~  & 13.116 & 90.29 & 58.48 & 74.54 & 74.44 & 76.49 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.}   is the unweighted mean across the three target classes at Moderate difficulty.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S074 — before, sections/results.tex:9

Script: 46 từ; 6 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\begin{figure*}[t] \centering \paperfigure[width=\textwidth]{pareto_tradeoff.pdf}{48mm}{Pareto trade-off efficiency on the KITTI validation benchmark.} Detection accuracy ( , \%) versus parameter footprint (M) on a logarithmic scale, highlighting that MobilePIXORNeXt (red star) achieves   mAP with only   parameters, matching SECOND ( ,  ) and outperforming PointPillars ( ,  ) with   fewer parameters.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U122](rewrites.md#u122).

## S075 — before, sections/results.tex:9

Script: 36 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textbf{(b)} Detection accuracy ( , \%) versus end-to-end pipeline latency (ms) at batch size 1 (Tesla T4, except the historical MobilePIXOR result on L4), demonstrating that MobilePIXORNeXt achieves the fastest complete perception pipeline ( ,  ) among all evaluated real-time detectors.}
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U122](rewrites.md#u122).

## S076 — before, sections/results.tex:29

Script: 36 từ; 7 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
As evidenced in Table~  and the Precision--Recall curves in Figure~ , MobilePIXORNeXt achieved \textbf{60.75\% Moderate AP} on Pedestrians, outperforming PointPillars ( , a   gain) and SECOND ( , a   gain), while more than doubling the baseline MobilePIXOR ( , a   gain).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U125](rewrites.md#u125).

## S077 — before, sections/results.tex:29

Script: 50 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
We attribute this performance advantage primarily to two architectural mechanisms: (i) Strata-BEV14's continuous subcell displacement offsets  , which mitigate the   grid quantization error that inherently impairs small object localization; and (ii) the decoupled GroupedHeader architecture, which isolates pedestrian regression from large vehicle footprints to alleviate gradient interference across morphologically disparate categories.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U125](rewrites.md#u125).

## S078 — before, sections/results.tex:32

Script: 28 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Balanced Multi-Class Detection.} For passenger vehicles (Car), MobilePIXORNeXt achieved   Moderate AP, closely trailing PointPillars ( ) and SECOND ( ) by less than  , while outperforming the original PIXOR ( ) by   and MobilePIXOR ( ) by  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S079 — before, sections/results.tex:32

Script: 33 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
While two-stage 3D anchor-based detectors (PointRCNN, Part-A , PV-RCNN) achieve higher absolute accuracy on rigid vehicles, they require between   and   parameters and operate at higher latencies ( ), exceeding the real-time constraints of embedded edge processors.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U126](rewrites.md#u126).

## S080 — before, tables/efficiency.tex:5

Script: 231 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular}{@{}lrrrrrrrr@{}} \toprule \textbf{Method} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Model\\(ms)  }} & \textbf{\shortstack{Model\\FPS  }} & \textbf{\shortstack{E2E\\(ms)  }} & \textbf{\shortstack{E2E\\FPS  }} & \textbf{\shortstack{P50\\(ms)  }} & \textbf{\shortstack{P95\\(ms)  }} & \textbf{\shortstack{P99\\(ms)  }} \\ \midrule PIXOR~  & 2.135 & 70.79 & 14.13 & 144.00 & 6.94 & 142.29 & 159.38 & 175.43 \\ MobilePIXOR (L4)~  & \textbf{0.598} & \textbf{11.05} & \textbf{90.50} & 85.92 & 11.64 & 84.11 & 112.52 & 124.94 \\ PointPillars~  & 4.835 & 32.13 & 31.12 & \underline{63.84} & \underline{15.66} & \underline{62.10} & \textbf{75.88} & \textbf{92.73} \\ SECOND~  & 5.326 & 62.64 & 15.96 & 93.70 & 10.67 & 92.02 & 107.42 & 126.35 \\ \textbf{MobilePIXORNeXt (Ours)} & \underline{0.754} & \underline{17.88} & \underline{55.94} & \textbf{60.96} & \textbf{16.41} & \textbf{58.28} & \underline{80.85} & \underline{95.91} \\ \midrule \multicolumn{9}{@{}l}{\textit{Heavyweight 3D Anchor-Based Reference Detectors}} \\ PointRCNN~  & 4.039 & 169.36 & 5.90 & 194.50 & 5.14 & 191.24 & 213.48 & 235.10 \\ Part-A -Anchor~  & 63.815 & 102.57 & 9.75 & 133.75 & 7.48 & 131.50 & 150.49 & 172.80 \\ PV-RCNN~  & 13.116 & 160.79 & 6.22 & 188.60 & 5.30 & 186.40 & 202.62 & 224.15 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.} Model latency measures neural network execution time within the evaluation pipeline on Tesla T4.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S081 — before, tables/efficiency.tex:5

Script: 27 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
End-to-end (E2E) pipeline latency spans raw LiDAR point cloud loading, deterministic statistical rasterization, Host-to-Device (H2D) memory transfer, forward propagation, subcell coordinate decoding, and class-wise rotated NMS filtering.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U175](rewrites.md#u175).

## S082 — before, sections/results.tex:39

Script: 25 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
A complete perception system must account for point cloud loading, input feature tensorization, Host-to-Device (H2D) memory transfer, forward execution, coordinate decoding, and Non-Maximum Suppression (NMS).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U127](rewrites.md#u127).

## S083 — before, sections/results.tex:44

Script: 34 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
End-to-End Pipeline Latency Comparison.} As detailed in Table~ , MobilePIXORNeXt achieved an End-to-End (E2E) pipeline latency of \textbf{60.96\,ms} (\textbf{16.41\,FPS}), delivering the lowest full-system latency among all evaluated real-time detectors.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U128](rewrites.md#u128).

## S084 — before, sections/results.tex:44

Script: 16 từ; 7 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
It operated faster than PointPillars ( ,  ), MobilePIXOR ( ,  ), and SECOND ( ,  ), and ran over   faster than the original PIXOR ( ,  ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U128](rewrites.md#u128).

## S085 — before, sections/results.tex:44

Script: 34 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Percentile analysis indicates consistent latency bounds in the evaluated environment: MobilePIXORNeXt exhibited a median P50 latency of \textbf{58.28\,ms}, P95 of  , and P99 of  , demonstrating low latency dispersion under desktop GPU benchmarking.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U128](rewrites.md#u128).

## S086 — before, sections/results.tex:47

Script: 42 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\item \textbf{Host-to-Device Memory Transfer ( )}: Unlike 35-channel binary occupancy tensors that require transferring   per scan, Strata-BEV14's compact 14-channel layout requires transferring only   (a   payload reduction), which, in conjunction with pinned-memory buffer optimization, reduces H2D latency to   compared with   for the MobilePIXOR baseline.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U131](rewrites.md#u131).

## S087 — before, tables/distance_our_vs_mbpixor.tex:5

Script: 95 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{1.5pt} \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrr@{}} \toprule \textbf{Category} & \textbf{Model} & \textbf{\shortstack{0--30\,m}} & \textbf{\shortstack{30--50\,m}} & \textbf{\shortstack{50--70.4\,m}} \\ \midrule Pedestrian & MobilePIXOR~  & 30.45 & 3.75 & \textbf{0.99} \\ & \textbf{Ours} & \textbf{65.23} & \textbf{11.49} & 0.80 \\ & \textit{Diff.} & \textit{+34.78} & \textit{+7.74} & \textit{-0.19} \\ \midrule Cyclist & MobilePIXOR~  & 25.55 & 12.61 & 0.56 \\ & \textbf{Ours} & \textbf{76.32} & \textbf{44.30} & \textbf{14.94} \\ & \textit{Diff.} & \textit{+50.77} & \textit{+31.69} & \textit{+14.38} \\ \bottomrule \end{tabular*} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S088 — before, sections/results.tex:69

Script: 40 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Near-Range Safety Envelope ( ).} Within the near-range   evaluation zone, MobilePIXORNeXt delivered substantial accuracy gains over baseline MobilePIXOR: Pedestrian AP reached \textbf{65.23\%} (more than double MobilePIXOR's  , a   improvement), while Cyclist AP reached \textbf{76.32\%} (nearly triple MobilePIXOR's  , a   improvement).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U135](rewrites.md#u135).

## S089 — before, tables/components.tex:5

Script: 293 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textbf{(a) Architectural configurations and component progression}\par\smallskip \begingroup\footnotesize\setlength{\tabcolsep}{4pt} \begin{tabular}{@{}lllllll@{}} \toprule \textbf{Step} & \textbf{Backbone} & \shortstack[l]{\textbf{BEV}\\ \textbf{Input}} & \shortstack[l]{\textbf{Head}\\ \textbf{Architecture}} & \shortstack[l]{\textbf{GT}\\ \textbf{Sampling}} & \shortstack[l]{\textbf{Fusion /}\\ \textbf{Neck}} & \shortstack[l]{\textbf{Supervision /}\\ \textbf{Attention}} \\ \midrule A0 & MobilePIXOR & Binary35  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A1 & MobilePixorNeXt & Binary35  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A2 & MobilePixorNeXt & Strata-BEV8  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A3 & MobilePixorNeXt & Strata-BEV14 & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A4 & MobilePixorNeXt & Strata-BEV14 & Grouped H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A5 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A6 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F24 / SumFPN & Baseline Smooth-L1 \\ A7 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SumFPN & Baseline Smooth-L1 \\ A8 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & Baseline Smooth-L1 \\ A9 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & OGA + IQA Head \\ A10 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & OGA + IQA + Block-SimAM \\ \bottomrule \end{tabular}\endgroup \par\medskip \textbf{(b) Quantitative detection accuracy and standalone model inference speed}\par\smallskip \begingroup\footnotesize\setlength{\tabcolsep}{3pt} \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l>{\raggedright\arraybackslash}p{0.42\textwidth}rrrr@{}} \toprule \textbf{Step} & \textbf{\shortstack[l]{Key Modification /\\Investigated Factor}} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{Standalone Forward\\(ms)  }} & \textbf{\shortstack{Standalone FPS\\ }} \\ \midrule A0 & Baseline MobilePIXOR starting checkpoint & 41.40 & 42.27 & 11.05 & 90.57 \\ A1 & + ConvNeXt-style backbone (  DW, SiLU, Inverted Bottleneck) & 47.60 & 49.08 & \textbf{5.88} & \textbf{169.97} \\ A2 & + 8-channel RichBEV statistical representation & 47.37 & 48.95 & 6.06 & 165.02 \\ A3 & + 14-channel Strata-BEV14 (subcell offsets, height moments) & 50.14 & 52.00 & \underline{5.92} & \underline{169.01} \\ A4 & + Decoupled Grouped Header (Car vs.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S090 — before, tables/components.tex:5

Script: 130 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Ped/Cyc, 16 channels) & 51.30 & 53.96 & 6.92 & 144.58 \\ A5 & + Expanded Grouped Header capacity (H32, 32 channels) & 60.22 & 61.11 & 6.70 & 149.15 \\ A6 & + Physics-Aware Ground-Truth Augmentation (PC-GTA) & 66.25 & 68.45 & 7.45 & 134.25 \\ A7 & + Expanded neck fusion layer ( , 32 channels) & 68.12 & 70.05 & 6.72 & 148.82 \\ A8 & + Bilinear Depthwise Scale-Gated Neck (SG-FPN) & 67.38 & 69.46 & 6.94 & 144.16 \\ A9 & + Oriented Geometric Alignment (OGA) and IQA supervision & \underline{70.58} & \underline{72.66} & 8.19 & 122.16 \\ A10 & + Parameter-Free Block-SimAM Energy Attention inside all blocks & \textbf{71.46} & \textbf{73.42} & 8.67 & 115.32 \\ \bottomrule \end{tabular*}\endgroup \par\vspace{2pt} {\footnotesize \textit{Note.} Step identifiers A0--A10 correspond to the experimental progression from baseline R01 to peak configuration.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S091 — before, sections/results.tex:85

Script: 42 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Backbone Architecture (A0   A1).} Replacing MobilePIXOR's residual blocks with our ConvNeXt-style backbone (  depthwise convolutions, inverted bottleneck expansion  , SiLU activations, and LayerScale) elevated Moderate mAP from   to   (\textbf{+6.20\% mAP}), while nearly doubling standalone forward throughput from   to \textbf{169.97\,FPS}.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U138](rewrites.md#u138).

## S092 — before, tables/loss.tex:5

Script: 168 từ; 4 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{0.26\textwidth}ccrrrrr@{}} \toprule \textbf{Training Objective} & \textbf{\shortstack{IQA\\Head}} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Forward\\(ms)  }} & \textbf{\shortstack{Car AP \\(\%)  }} & \textbf{\shortstack{Pedestrian\\AP  (\%)  }} & \textbf{\shortstack{Cyclist\\AP  (\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} \\ \midrule Baseline Smooth-L1 & No  & \textbf{0.702} & \textbf{6.94} & 83.92 & 55.48 & 62.74 & 67.38 \\ OGA (MGIoU +  -Corner) & No  & \textbf{0.702} & \textbf{6.94} & 84.09 & 57.93 & 64.88 & 68.97 \\ Baseline Smooth-L1 & Yes & 0.721 & 8.19 & 84.13 & \underline{59.45} & \textbf{67.34} & 70.31 \\ \textbf{OGA (MGIoU +  -Corner)} & \textbf{Yes} & 0.721 & 8.19 & \textbf{84.87} & \textbf{59.63} & \underline{67.24} & \textbf{70.58} \\ \midrule Q-OGA (Quality-Coupled Focal Loss) & No & \textbf{0.702} & \textbf{6.94} & \underline{84.35} & 58.12 & 65.13 & 69.20 \\ \bottomrule \end{tabular*} \par\vspace{2pt} {\footnotesize \textit{Note.} All variants share identical MobilePixorNeXt backbone, Strata-BEV14 input, Grouped H32 heads, F32 SG-FPN neck, and PC-GTA augmentation.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S093 — before, sections/results.tex:123

Script: 27 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
As shown in Table~ , introducing OGA alone into baseline training increased Moderate mAP from   to   (\textbf{+1.59\% mAP}), with consistent gains across Car ( ), Pedestrian ( ), and Cyclist ( ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U148](rewrites.md#u148).

## S094 — before, sections/results.tex:123

Script: 40 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
These findings demonstrate that OGA and IQA provide complementary benefits: OGA refines continuous corner and boundary alignment during backpropagation, whereas IQA calibrates classification scores with true geometric overlap during inference ( ), preventing misaligned boxes from suppressing high-quality candidates during rotated NMS.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U148](rewrites.md#u148).

## S095 — before, sections/results.tex:125

Script: 33 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
These findings confirm that Q-OGA provides a parsimonious edge-deployment configuration when memory and latency budgets are strictly capped, whereas the explicit IQA configuration offers peak accuracy ( ) when resources accommodate a dedicated quality branch.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U149](rewrites.md#u149).

## S096 — before, tables/attention.tex:5

Script: 116 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{5pt} \begin{tabular}{@{}llrrrr@{}} \toprule \textbf{\shortstack[l]{SimAM Attention\\Placement}} & \textbf{\shortstack[l]{Neck Fusion\\Width}} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{Standalone\\Forward (ms)  }} & \textbf{\shortstack{Standalone\\FPS  }} \\ \midrule None (No attention) &   & \underline{70.58} & \underline{72.66} & \underline{8.19} & \underline{122.16} \\ Stage-2 ( ) &   & 69.44 & 71.49 & 7.30 & 136.91 \\ Stage-3 ( ) &   & 69.80 & 71.98 & \textbf{6.94} & \textbf{144.05} \\ Stage-4 ( ) &   & 69.36 & 71.34 & 7.44 & 134.43 \\ \textbf{Block-level ( )} & \textbf{ } & \textbf{71.46} & \textbf{73.42} & 8.67 & 115.32 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.} All configurations share the OGA loss, IQA quality head, Strata-BEV14 input, and PC-GTA data augmentation.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S097 — before, sections/results.tex:141

Script: 54 từ; 7 bộ đếm mệnh đề. Nhóm: `prose-reviewed`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{Multi-Task Optimization Dynamics and Convergence.} To analyze empirical training stability and convergence behavior across the ablation sequence, Figure~  illustrates the training and validation loss trajectories across 50 epochs for the overall composite objective alongside the six constituent multi-task components (Classification, Geometry, IoU Quality, Subcell Offset, Size, and Yaw), tracking configurations A1 through A10.
```

Đã biên tập văn xuôi ở đơn vị liên kết; báo cáo gốc được giữ để đối chiếu. [Đối chiếu U153](rewrites.md#u153).

## S098 — before, sections/results.tex:153

Script: 65 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{1.5pt} \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcrrrrr@{}} \toprule \textbf{Selection} & \textbf{Epoch} & \textbf{Car} & \textbf{Ped.} & \textbf{Cyc.} & \textbf{mAP } & \textbf{E2E} \\ \midrule \textbf{Best AP} & \textbf{130} & \textbf{86.90} & 60.75 & 66.73 & \textbf{71.46} & \textbf{60.96} \\ Best loss & 47 & 82.71 & \textbf{62.30} & \textbf{69.43} & \textbf{71.48} & 81.23 \\ \bottomrule \end{tabular*} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S099 — before, sections/results.tex:172

Script: 37 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\begin{figure}[t] \centering \paperfigure[width=\columnwidth,height=0.38\textheight,keepaspectratio]{qualitative_detections.pdf}{65mm}{Qualitative BEV detection visualizations on KITTI validation scenes.} ) on KITTI validation frames \texttt{007038}, \texttt{005571}, and \texttt{006761} (top to bottom).
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S100 — before, sections/conclusion.tex:5

Script: 35 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
By introducing the deterministic Strata-BEV14 rasterizer, the architecture eliminates computationally intensive PointNet pillar encoders and 3D sparse convolutions, preserving continuous vertical distribution moments, material surface reflectivity, and horizontal subcell centroid displacements with zero learned parameters.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U002](rewrites.md#u002).

## S101 — before, sections/conclusion.tex:9

Script: 30 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
While vertical elevation and height regression are supported via an auxiliary branch, localization fidelity on steep grades, multi-level interchanges, and irregular non-planar terrains requires further investigation on dedicated off-road benchmarks.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U004](rewrites.md#u004).

## S102 — before, sections/declarations.tex:4

Script: 24 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\begin{acknowledgements} The authors would like to acknowledge FPT University, Vietnam, for providing computational infrastructure, GPU workstation clusters, and laboratory facilities supporting this research.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U005](rewrites.md#u005).

## S103 — before, MobilePIXORNeXt.tex:52

Script: 48 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
The framework replaces learned point encoders with Strata-BEV14, a parameter-free deterministic NumPy/Numba CPU rasterizer that extracts 14 statistical feature channels---capturing stratified height distributions, elevation moments, surface reflectance, and continuous subcell centroid displacement offsets---with a measured input preprocessing time of 20.94 milliseconds, including point-cloud loading and rasterization.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S104 — before, sections/abstract.tex:1

Script: 48 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__abstract.tex.txt).

```latex
The framework replaces learned point encoders with Strata-BEV14, a parameter-free deterministic NumPy/Numba CPU rasterizer that extracts 14 statistical feature channels---capturing stratified height distributions, elevation moments, surface reflectance, and continuous subcell centroid displacement offsets---with a measured input preprocessing time of 20.94 milliseconds, including point-cloud loading and rasterization.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U001](rewrites.md#u001).

## S105 — before, sections/conclusion.tex:5

Script: 35 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__conclusion.tex.txt).

```latex
By introducing the deterministic Strata-BEV14 rasterizer, the architecture eliminates computationally intensive PointNet pillar encoders and 3D sparse convolutions, preserving continuous vertical distribution moments, material surface reflectivity, and horizontal subcell centroid displacements with zero learned parameters.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U002](rewrites.md#u002).

## S106 — before, sections/conclusion.tex:9

Script: 30 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__conclusion.tex.txt).

```latex
While vertical elevation and height regression are supported via an auxiliary branch, localization fidelity on steep grades, multi-level interchanges, and irregular non-planar terrains requires further investigation on dedicated off-road benchmarks.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U004](rewrites.md#u004).

## S107 — before, sections/declarations.tex:4

Script: 24 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__declarations.tex.txt).

```latex
\begin{acknowledgements} The authors would like to acknowledge FPT University, Vietnam, for providing computational infrastructure, GPU workstation clusters, and laboratory facilities supporting this research.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U005](rewrites.md#u005).

## S108 — before, sections/declarations.tex:28

Script: 19 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__declarations.tex.txt).

```latex
\textbf{Nhan Trong Phan}: Formal analysis, loss ablation studies (OGA, IQA, T-SBUW), quantitative visualization, and writing the original draft.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U012](rewrites.md#u012).

## S109 — before, sections/declarations.tex:28

Script: 18 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__declarations.tex.txt).

```latex
\textbf{Hoang Ngoc Tran}: Research supervision, conceptual validation, project administration, critical revision, and final review of the manuscript.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U012](rewrites.md#u012).

## S110 — before, sections/experiments.tex:7

Script: 37 từ; 8 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__experiments.tex.txt).

```latex
Perception is bounded within the Cartesian Region of Interest: \begin{equation} \begin{aligned} x &\in [0.0, 70.4]\,\text{m}, \\ y &\in [-40.0, 40.0]\,\text{m}, \\ z &\in [-2.5, 1.0]\,\text{m}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U014](rewrites.md#u014).

## S111 — before, sections/experiments.tex:18

Script: 60 từ; 2 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__experiments.tex.txt).

```latex
\paragraph{Evaluation Metrics.} Detection performance is benchmarked on oriented 2D Bird's-Eye View (BEV) footprints using 40-point interpolated Average Precision (AP  ) on the KITTI validation set, which calculates the area under the precision--recall curve over 40 recall positions  : \begin{equation} \mathrm{AP}|_{R_{40}} = \frac{1}{40} \sum_{r \in \mathcal{R}_{40}} \max_{r' \ge r} p(r').
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U016](rewrites.md#u016).

## S112 — before, sections/experiments.tex:18

Script: 43 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__experiments.tex.txt).

```latex
\end{itemize} Following autonomous driving literature, \textbf{Moderate macro mAP} ( ) was adopted as the primary benchmark ranking criterion: \begin{equation} \mathrm{mAP}_{\mathrm{M}} = \frac{1}{3} \sum_{c \in \{\mathrm{Car}, \mathrm{Ped}, \mathrm{Cyc}\}} \mathrm{AP}_{c, \mathrm{M}}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S113 — before, sections/experiments.tex:18

Script: 11 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__experiments.tex.txt).

```latex
Here, E, M, and H denote Easy, Moderate, and Hard regimes.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U023](rewrites.md#u023).

## S114 — before, sections/experiments.tex:42

Script: 39 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__experiments.tex.txt).

```latex
The network comprises a two-layer   convolutional stem (  channels, stride 2), three hierarchical backbone stages with channel widths   and block depths   based on \texttt{MobilePixorNeXt-S} blocks with large   depthwise convolutions, inverted expansion ratio  , SiLU activations, and LayerScale parameters initialized to  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U024](rewrites.md#u024).

## S115 — before, sections/experiments.tex:48

Script: 30 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__experiments.tex.txt).

```latex
In the final reported run, local ground fitting, static obstacle clearance, range-adaptive point subsampling, and visibility protection were enabled; LOS rejection, shadow masking, and radiometric calibration were available but disabled.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U026](rewrites.md#u026).

## S116 — before, sections/experiments.tex:51

Script: 26 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__experiments.tex.txt).

```latex
\paragraph{Batching, Precision, and Hardware Environment.} Training used a physical batch size of 16 on a single GPU, while validation used a batch size of 32.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S117 — before, sections/experiments.tex:51

Script: 32 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__experiments.tex.txt).

```latex
Numerical forward and backward passes utilized Automatic Mixed Precision (AMP) in Bfloat16 (\texttt{bf16}), while loss evaluation, coordinate regression, and rotated NMS were conducted in full FP32 precision to maintain numerical stability.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U027](rewrites.md#u027).

## S118 — before, sections/experiments.tex:51

Script: 40 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__experiments.tex.txt).

```latex
The final MobilePIXORNeXt timing and accuracy evaluation was performed on a server equipped with an NVIDIA Tesla T4 GPU (16\,GB GDDR6 VRAM, Turing architecture), host x86\_64 CPU, Linux OS, and PyTorch 2.11.0 with CUDA 13.0.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U027](rewrites.md#u027).

## S119 — before, sections/introduction.tex:5

Script: 35 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
Unlike passive optical cameras, which remain vulnerable to ambient illumination fluctuations and projective scale ambiguities, LiDAR sensors actively emit pulsed laser beams to measure direct metric ranges, capturing high-precision geometric boundaries under arbitrary day-and-night conditions~ .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U028](rewrites.md#u028).

## S120 — before, sections/introduction.tex:5

Script: 37 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
In time-critical vehicular environments, perception systems must accurately detect, orient, and classify surrounding objects (from passenger vehicles to compact vulnerable road users such as pedestrians and cyclists) within strict latency envelopes, typically requiring sub-  execution per scan~ .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U028](rewrites.md#u028).

## S121 — before, sections/introduction.tex:7

Script: 30 từ; 7 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
First, volumetric voxel-based architectures, exemplified by SECOND~ , and two-stage point-voxel frameworks, such as PointRCNN~ , Part- ~ , and PV-RCNN~ , achieve competitive detection accuracy by deploying 3D sparse convolutions or localized point-cloud abstractions.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U029](rewrites.md#u029).

## S122 — before, sections/introduction.tex:7

Script: 22 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
While faster than volumetric methods, non-deterministic point gathering, dynamic memory allocation, and sparse scattering introduce substantial runtime overhead and memory transfer latency.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U029](rewrites.md#u029).

## S123 — before, sections/introduction.tex:7

Script: 41 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
Although conceptually efficient, existing rasterization methods suffer from severe spatial quantization errors on standard   grids; by relying on crude binary occupancy slices, they discard continuous subcell offsets and high-order vertical dispersion statistics, inducing spatial staircasing and localization jitter on small objects.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U029](rewrites.md#u029).

## S124 — before, sections/introduction.tex:9

Script: 78 từ; 4 bộ đếm mệnh đề. Nhóm: `question-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
Grounded in this principle, we investigate the central research question: \textit{Can an ultra-compact LiDAR detector operating on a parameter-free BEV tensor rival the localization accuracy of heavy volumetric and pillar architectures while achieving superior end-to-end throughput on commodity edge hardware?} To realize this objective, our framework couples this deterministic representation with a single-branch ConvNeXt-style backbone~  utilizing large   depthwise kernels and parameter-free 3D spatial-energy attention (SimAM)~ , expanding physical receptive fields to envelop entire vehicles without multi-branch training overhead.
```

Script gộp câu hỏi kết thúc bằng ?} với câu tiếp theo; kiểm tra riêng từng câu. [Đối chiếu U030](rewrites.md#u030).

## S125 — before, sections/introduction.tex:11

Script: 21 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
Guided by this design philosophy, we propose \textbf{MobilePIXORNeXt}, an ultra-compact, proposal-free, multi-class 3D LiDAR object detector containing exactly   trainable parameters ( ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S126 — before, sections/introduction.tex:11

Script: 57 từ; 5 bộ đếm mệnh đề. Nhóm: `prose-reviewed`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
An unordered LiDAR point cloud is deterministically transformed via the parameter-free \textbf{Strata-BEV14} rasterizer into a 14-channel statistical BEV tensor (  at   resolution) using NumPy or Numba on CPU, with point-cloud loading and rasterization taking   in the measured pipeline, capturing height-band densities, vertical spatial moments ( ), surface reflectance statistics, and continuous subcell centroid displacement offsets   to eliminate quantization error.
```

Đã biên tập văn xuôi ở đơn vị liên kết; báo cáo gốc được giữ để đối chiếu. [Đối chiếu U031](rewrites.md#u031).

## S127 — before, sections/introduction.tex:11

Script: 36 từ; 8 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
To regularize training without unphysical artifacts, we formulate \textbf{Physics-Consistent Ground-Truth Augmentation (PC-GTA)}, which combines local ground elevation fitting, static obstacle clearance, and range-adaptive point subsampling, with optional line-of-sight checks, raycast shadow masking, and intensity attenuation.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S128 — before, sections/introduction.tex:11

Script: 36 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
Multi-scale feature extraction is performed by the \textbf{MobilePixorNeXt-S} backbone hierarchy across stages  , utilizing streamlined   depthwise inverted bottleneck blocks ( ), SiLU activations, LayerScale stabilization, and embedded parameter-free \textbf{Block-SimAM} energy attention operating inside all nine residual blocks.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S129 — before, sections/introduction.tex:11

Script: 33 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
Multi-scale semantic context and fine coordinates are balanced by the top-down \textbf{Scale-Gated Feature Pyramid Network (SG-FPN Neck,  )}, which employs bilinear upsampling, depthwise refinement, and learnable scale gates to eliminate deconvolution checkerboard artifacts.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S130 — before, sections/introduction.tex:11

Script: 32 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
By deterministically capturing continuous geometry at the input, expanding receptive fields via single-branch large-kernel convolution, and supervising rotated overlap directly, this coherent pipeline resolves each aforementioned bottleneck within a unified, edge-deployable framework.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S131 — before, sections/introduction.tex:17

Script: 36 từ; 7 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
\item \textbf{Physics-consistent data augmentation and oriented geometric supervision.} We formulate \textbf{PC-GTA}, which combines local terrain alignment, obstacle clearance, and range-dependent point subsampling, and supports optional visibility, shadow-masking, and intensity-adjustment heuristics to reduce synthesis artifacts.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U033](rewrites.md#u033).

## S132 — before, sections/introduction.tex:19

Script: 27 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__introduction.tex.txt).

```latex
In complete end-to-end pipeline speed, it operates at \textbf{16.41\,FPS} (\textbf{60.96\,ms} on NVIDIA Tesla T4), running faster than all evaluated real-time detectors.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U034](rewrites.md#u034).

## S133 — before, sections/methodology.tex:5

Script: 15 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
Decoupled grouped prediction towers output heatmaps, subcell offsets, log-sizes, doubled yaw angles, and localization quality.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S134 — before, sections/methodology.tex:5

Script: 32 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
Solid arrows trace the feedforward inference path, whereas dashed arrows indicate training-only supervision incorporating Oriented Geometric Alignment (OGA), IoU-Aware Quality Alignment (IQA), and dynamic loss balancing via Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW).}
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U036](rewrites.md#u036).

## S135 — before, sections/methodology.tex:18

Script: 45 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\paragraph{LiDAR Coordinate Frame and Spatial Region of Interest.} An unordered LiDAR point cloud is acquired via a rotating scanner and defined as a set  , where each point   encapsulates longitudinal distance   (forward positive), lateral distance   (leftward positive), vertical elevation   (upward positive), and calibrated reflectance intensity  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U038](rewrites.md#u038).

## S136 — before, sections/methodology.tex:18

Script: 33 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
Perception is constrained to a fixed Cartesian Region of Interest (ROI): \begin{equation} \begin{aligned} x_i &\in [x_{\min}, x_{\max}], \\ y_i &\in [y_{\min}, y_{\max}], \\  z_i &\in [z_{\min}, z_{\max}].
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U038](rewrites.md#u038).

## S137 — before, sections/methodology.tex:26

Script: 14 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\end{equation} configured with longitudinal bounds  \,m, lateral bounds  \,m, and vertical bounds  \,m.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U039](rewrites.md#u039).

## S138 — before, sections/methodology.tex:33

Script: 25 từ; 8 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
For each detected object instance  , the target state is parameterized by the 7-tuple: \begin{equation} \mathbf{b}_k = (c_k, x_k^*, y_k^*, l_k^*, w_k^*, \psi_k^*, s_k),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S139 — before, sections/methodology.tex:38

Script: 27 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
Operating on this contract requires a structured, dense BEV representation that captures fine 3D geometric distributions without the latency of learned point encoders, which we formulate next.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U041](rewrites.md#u041).

## S140 — before, sections/methodology.tex:45

Script: 70 từ; 7 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
Channels 0–3 capture stratified height-band densities, Channels 4–7 encode vertical elevation statistics (peak, mean, span, deviation), Channels 8–9 capture material surface reflectance, Channels 10–11 represent global density and occupancy, and Channels 12–13 encode continuous subcell horizontal offsets to eliminate grid discretization error.} \captionof{figure}{Visual breakdown of the fourteen deterministic feature channels in the Strata-BEV14 representation for a representative KITTI driving scene (  grid at   resolution).
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S141 — before, sections/methodology.tex:45

Script: 68 từ; 3 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
Channels 0--3 capture logarithmic point densities across four uniform vertical elevation strata ( ); Channels 4--7 encode vertical spatial moments (normalized maximum elevation  , mean elevation  , elevation span  , and sample standard deviation  ); Channels 8--9 capture calibrated surface reflectance statistics (maximum and mean intensity); Channel 10 records total cell logarithmic density; Channel 11 provides a binary occupancy indicator; and Channels 12--13 preserve continuous subcell horizontal centroid displacements   to mitigate spatial discretization errors.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U042](rewrites.md#u042).

## S142 — before, sections/methodology.tex:69

Script: 52 từ; 9 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\textit{Vertical Spatial Moments ( -- )}: To encode obstacle geometry without explicit voxelization, we extract peak elevation  , mean elevation  , vertical span  , and the sample standard deviation: \begin{equation} F_7 = \begin{cases} \sqrt{\max\left(0, \frac{1}{N} \sum_{i=1}^N \zeta_i^2 - F_5^2\right)}, & \text{if } N > 1, \\ 0, & \text{otherwise}.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S143 — before, sections/methodology.tex:78

Script: 39 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\textit{Reflectance Statistics ( -- )}: Calibrated surface returns capture material discrimination (e.g., distinguishing retro-reflective license plates, pedestrian clothing, or road signs): \begin{equation} F_8 = \max_{i=1,\dots,N} \iota_i, \qquad F_9 = \frac{1}{N} \sum_{i=1}^N \iota_i.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S144 — before, sections/methodology.tex:84

Script: 82 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
To prevent quantization error from coarse  \,m cell discretization, channels   and   preserve continuous subcell horizontal centroid displacements: \begin{align} x_u^{\mathrm{cell}} &= x_{\min} + (u + 0.5)\Delta_x, \nonumber\\ y_v^{\mathrm{cell}} &= y_{\min} + (v + 0.5)\Delta_y, \nonumber\\ F_{12} &= \operatorname{clip}\left(\frac{\sum_{i=1}^N (x_i - x_u^{\mathrm{cell}})}{N\Delta_x}, -0.5, 0.5\right), \nonumber\\ F_{13} &= \operatorname{clip}\left(\frac{\sum_{i=1}^N (y_i - y_v^{\mathrm{cell}})}{N\Delta_y}, -0.5, 0.5\right).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U049](rewrites.md#u049).

## S145 — before, sections/methodology.tex:127

Script: 31 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\paragraph{Motivation.} Cut-and-paste ground-truth (GT) augmentation balances rare LiDAR categories, but naive insertion ignores scene physics, producing floating or submerged objects, background collisions, missing raycast shadows, and unrealistic range-dependent point densities.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S146 — before, sections/methodology.tex:130

Script: 83 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
The mechanisms can be enabled independently; the settings used for the reported final run are given in Section~ : \begin{enumerate} \item \textit{Adaptive Ground Plane Fitting}: To prevent floating or sunken objects, local road elevation is estimated within a range-dependent search radius: \begin{align} R(d) &= \operatorname{clip}(2.5 + 0.03\max(0, d - 15), 2.5, 4.0)\,\mathrm{m},\nonumber\\[0.8em] \hat{z}_{\mathrm{ground}} &= \operatorname{percentile}(\{z_j \mid \mathbf{p}_j \in \mathcal{N}(x, y)\}, 5),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S147 — before, sections/methodology.tex:146

Script: 53 từ; 1 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
Points residing behind the exit face ( ) with radial distance  \,m are masked out: \begin{equation} \begin{aligned} \mathcal{P}_{\mathrm{augmented}} &= \mathcal{P}_{\mathrm{target}} \setminus \{\mathbf{p} \in \mathcal{P}_{\mathrm{target}} \mid \\ &\qquad \operatorname{Mask}_{\mathrm{shadow}}(\mathbf{p}) = \mathrm{True}\} \cup \mathcal{P}_{\mathrm{object}}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U060](rewrites.md#u060).

## S148 — before, sections/methodology.tex:158

Script: 50 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\item \textit{Range-Adaptive Point and Intensity Attenuation}: If an object captured at source distance   is placed at target distance  , its point cloud is randomly subsampled according to the inverse-square divergence law: \begin{equation} P_{\mathrm{keep}} = \min\left(1, \left(\frac{r_{\mathrm{src}}}{r_{\mathrm{tgt}}}\right)^2\right),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S149 — before, sections/methodology.tex:162

Script: 47 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\end{equation} and the optional radiometric adjustment attenuates reflectance intensity according to: \[ \begin{aligned} \eta_{\mathrm{int}} &= \left(\frac{r_{\mathrm{src}}}{r_{\mathrm{tgt}}}\right)^{\max(0, 2 - \gamma)}, \qquad \gamma = 1.7, \\ I_{\mathrm{tgt}} &= \operatorname{clip}(I_{\mathrm{src}}\eta_{\mathrm{int}}, 0, 1).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S150 — before, sections/methodology.tex:180

Script: 38 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
A two-layer convolutional stem ingests 14-channel input at stride 2 ( ), followed by three hierarchical stages (C2: 48 channels, depth 3; C3: 96 channels, depth 4; C4: 128 channels, depth 2) composed of \texttt{MobilePixorNeXt-S} blocks with embedded Block-SimAM.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U063](rewrites.md#u063).

## S151 — before, sections/methodology.tex:191

Script: 48 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
In contrast, at our input grid resolution of  \,m, a single   depthwise receptive field directly spans  \,m of physical terrain at stride 1, and rapidly expands to over  \,m by Stage 3 (stride 8)---sufficient to envelop an entire passenger vehicle in a single layer without dilation artifacts.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U064](rewrites.md#u064).

## S152 — before, sections/methodology.tex:194

Script: 49 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\paragraph{Backbone Topology and Multi-Stage Hierarchy.} As depicted in Figure~ , the network topology comprises: \begin{enumerate} \item \textit{Convolutional Stem}: Input   is ingested by two consecutive   convolutions (stride 2 followed by stride 1) with 32 output channels, Batch Normalization (BN), and Sigmoid Linear Unit (SiLU) activations, yielding stride-2 representation  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U065](rewrites.md#u065).

## S153 — before, sections/methodology.tex:214

Script: 29 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
(Top) Complete feedforward path from input X to output Y through the 7x7 depthwise convolution, inverted bottleneck expansion (e=2.5), SimAM modulation, LayerScale scaling, and identity residual addition.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S154 — before, sections/methodology.tex:214

Script: 39 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
(Bottom) Zoom-in structure of the parameter-free SimAM module, detailing spatial moment estimation, inverse-energy importance calculation, sigmoid saliency mapping, and element-wise feature recalibration.} \captionof{figure}{Micro-architecture of the \texttt{MobilePixorNeXt-S} residual block and the embedded parameter-free Block-SimAM energy attention module.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S155 — before, sections/methodology.tex:214

Script: 36 từ; 7 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\textbf{Top}: Complete residual transformation path from input   to output  , featuring a large-kernel   depthwise convolution, inverted bottleneck expansion with ratio  , SiLU activation, pointwise projection, parameter-free 3D spatial-energy attention  , LayerScale scaling   (initialized to  ), and an identity shortcut.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U071](rewrites.md#u071).

## S156 — before, sections/methodology.tex:214

Script: 37 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\textbf{Bottom}: Internal operational mechanism of the SimAM operator, which calculates channel-wise spatial moments ( ) in FP32 precision, evaluates visual saliency via inverse-energy importance  , and produces a 3D modulation tensor   via sigmoid scaling without introducing any learnable parameters.}
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U071](rewrites.md#u071).

## S157 — before, sections/methodology.tex:221

Script: 57 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
Given input tensor  , each block applies an inverted bottleneck with expansion ratio  : \begin{align} \mathbf{V} &= \operatorname{BN}(\operatorname{DWConv}_{7\times7}(\mathbf{Z})),\nonumber\\ \mathbf{R} &= \operatorname{BN}\left(W_{\mathrm{red}} \operatorname{SiLU}(W_{\mathrm{exp}}\mathbf{V})\right),\nonumber\\ \mathbf{Z}' &= \mathbf{Z} + \boldsymbol{\gamma}_{\mathrm{blk}} \odot \mathcal{S}(\mathbf{R}).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S158 — before, sections/methodology.tex:234

Script: 31 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\paragraph{Design and Placement.} As illustrated in Figure~ , SimAM is embedded inside each \texttt{MobilePixorNeXt-\allowbreak S} block across all three stages (C2, C3, C4), for nine attention operators in total.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S159 — before, sections/methodology.tex:237

Script: 44 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
For each channel slice with   spatial locations, spatial mean   and sample variance   are computed: \begin{equation} \begin{aligned} \mu_c &= \frac{1}{M}\sum_{u, v} R_{c, u, v}, \\ \nu_c &= \frac{1}{\max(M-1, 1)}\sum_{u, v} (R_{c, u, v} - \mu_c)^2.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S160 — before, sections/methodology.tex:244

Script: 56 từ; 2 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\end{equation} Following the energy-minimization formulation of SimAM~ , the minimized energy   and its inverse-energy importance score   for neuron   are: \begin{equation} \begin{aligned} e_t^\star &= \frac{4(\nu_c + \lambda_{\mathrm{sim}})}{(t-\mu_c)^2 + 2(\nu_c + \lambda_{\mathrm{sim}})}, \\ s_t &= \frac{1}{e_t^\star} = \frac{(t - \mu_c)^2}{4(\nu_c + \lambda_{\mathrm{sim}})} + \frac{1}{2}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S161 — before, sections/methodology.tex:282

Script: 27 từ; 8 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\paragraph{Mathematical Formulation.} As shown in Figure~ , lateral features from Stages C4 ( \,ch), C3 ( \,ch), and C2 ( \,ch) are projected via   convolutions into matching lateral dimensions  ,  , and  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S162 — before, sections/methodology.tex:285

Script: 51 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
At each fusion stride  , lateral and top-down representations are adaptively balanced via learnable depthwise scale gating: \begin{align} \mathbf{G}_r &= 2\sigma\left(\operatorname{DWConv}_{3\times3, r}(\mathbf{L}_r + \mathbf{U}_r)\right),\nonumber\\ \mathbf{P}^{(r)} &= \mathbf{U}_r + \mathbf{G}_r \odot \mathbf{L}_r.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U084](rewrites.md#u084).

## S163 — before, sections/methodology.tex:296

Script: 81 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\par\vspace{-0.5\baselineskip} \noindent \begin{minipage}{\columnwidth} \centering \paperfigure{geometry_output_2x2.pdf}{75mm}{Geometric parameterization, orientation encoding, and quality-aware score calibration across prediction heads: (a) Continuous subcell coordinate offsets   from anchor point   to object center  ; (b) Log-transformed bounding box dimension targets  ; (c) Doubled-angle trigonometric yaw representation   on the unit circle resolving  -heading ambiguity; and (d) IoU-quality-aware composite score ranking   calibrated for post-processing peak selection and rotated NMS.} \captionof{figure}{Geometric parameterization, orientation formulation, and quality-aware score calibration across prediction heads.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U086](rewrites.md#u086).

## S164 — before, sections/methodology.tex:306

Script: 20 từ; 11 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
Cars exhibit rigid rectangular footprints ( \,m,  \,m), whereas Pedestrians ( \,m,  \,m) and Cyclists ( \,m,  \,m) are compact and highly deformable.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U087](rewrites.md#u087).

## S165 — before, sections/methodology.tex:319

Script: 30 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
For output cell  , the ground-truth geometric targets are formulated as: \begin{align} \mathbf{o}^* &=  ,\nonumber\\ \mathbf{d}^* &= (\log w^*, \log l^*), \qquad \mathbf{a}^* = (\cos 2\psi^*, \sin 2\psi^*).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S166 — before, sections/methodology.tex:333

Script: 62 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\textit{Multi-Axis Projection GIoU ( )}: Projected bounding box overlap is evaluated following Marginalized Generalized IoU~  across four unit normal projection axes defining the prediction and ground-truth rectangles: \begin{equation} \mathbf{n}_k \in \left\{ \mathbf{n}_{\mathrm{len}}^{\mathrm{pred}}, \mathbf{n}_{\mathrm{wid}}^{\mathrm{pred}}, \mathbf{n}_{\mathrm{len}}^{\mathrm{gt}}, \mathbf{n}_{\mathrm{wid}}^{\mathrm{gt}} \right\}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U096](rewrites.md#u096).

## S167 — before, sections/methodology.tex:339

Script: 52 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\end{equation} For each axis  , 1D intervals of intersection length  , union length  , and smallest enclosing hull   are determined, yielding 1D GIoU~ : \begin{equation} \begin{aligned} g_k &= \frac{I_k}{U_k} - \frac{E_k - U_k}{E_k}, \\ \bar{g} &= \frac{1}{4}\sum_{k=1}^4 g_k, \qquad \ell_{\mathrm{proj}} = \frac{1 - \bar{g}}{2}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S168 — before, sections/methodology.tex:348

Script: 78 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
Scale-Normalized  -Symmetric Corner Distance ( )}:\par}\noindent To ensure accurate vertex localization under doubled-angle symmetry, we evaluate the minimum L1 distance across direct and  -permuted corner correspondences: \begin{equation} \ell_{\mathrm{corner}} = \frac{\min\left\{ \begin{array}{l} \displaystyle\sum_{m=1}^4 \|\widehat{\mathbf{c}}_m - \mathbf{c}_m^*\|_1, \\ \displaystyle\sum_{m=1}^4 \|\widehat{\mathbf{c}}_m - \mathbf{c}_{5-m}^*\|_1 \end{array} \right.}{4\max\left(\sqrt{(w^*)^2 + (l^*)^2}, 10^{-6}\right)}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S169 — before, sections/methodology.tex:362

Script: 38 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
It is supervised via Binary Cross-Entropy (BCE) using the detached MGIoU similarity as the ground-truth target: \begin{equation} q^* = \operatorname{stopgrad}(\operatorname{clip}(\bar{g}, 0, 1)), \qquad \mathcal{L}_{\mathrm{qual}} = \operatorname{BCE}(\hat{q}, q^*).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S170 — before, sections/methodology.tex:383

Script: 74 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\textit{Bounded Softmax Weights}: Learnable logit parameters   are soft-clamped via hyperbolic tangent with bound  , and converted to positive normalized weights via temperature-controlled softmax ( ): \begin{equation} \begin{aligned} \bar{a}_k &= b_{\mathrm{uw}} \tanh(a_k / b_{\mathrm{uw}}), \\ Z_{\mathrm{uw}} &= \sum_{j \in \mathcal{T}} \exp(\bar{a}_j / \tau_{\mathrm{uw}}), \\ \omega_k &= J \cdot \frac{\exp(\bar{a}_k / \tau_{\mathrm{uw}})}{Z_{\mathrm{uw}}}, \\ \sum_{k \in \mathcal{T}} \omega_k &= J.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S171 — before, sections/methodology.tex:411

Script: 61 từ; 4 bộ đếm mệnh đề. Nhóm: `algorithm`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\vspace{2pt} {\small \noindent\textbf{Require:} Raw LiDAR point cloud  , ROI bounds  \par \noindent\textbf{Require:} Score threshold  , Rotated NMS threshold  , quality exponent  \par \noindent\textbf{Ensure:} Final decoded 3D oriented bounding boxes  \par \begin{enumerate} \setlength{\topsep}{2pt} \setlength{\itemsep}{0pt} \setlength{\parsep}{0pt} \renewcommand{\labelenumi}{\textbf{\arabic{enumi}:}} \item Filter raw points   within inward margin  \,m of spatial ROI.
```

Giữ nguyên thuật toán; danh sách đầu vào/đầu ra không phải câu văn cần tách.

## S172 — before, sections/methodology.tex:411

Script: 20 từ; 5 bộ đếm mệnh đề. Nhóm: `algorithm`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\item Evaluate Grouped Prediction Towers for Car ( ) and Pedestrian/Cyclist ( ), yielding classification maps  , subcell offsets  , log-sizes  , doubled yaws  , and quality  .
```

Giữ nguyên thuật toán; danh sách đầu vào/đầu ra không phải câu văn cần tách.

## S173 — before, sections/methodology.tex:441

Script: 75 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
For each candidate cell   where predicted classification probability   forms a   local spatial peak, metric continuous bounding boxes are decoded via: \begin{align} \hat{x} &= x_{\min} + u s_{\mathrm{out}}\Delta_x + \hat{o}_x, \nonumber\\ \hat{y} &= y_{\min} + v s_{\mathrm{out}}\Delta_y + \hat{o}_y, \nonumber\\ \hat{w} &= \exp(\hat{d}_1), \qquad \hat{l} = \exp(\hat{d}_2), \nonumber\\ \hat{\psi} &= \frac{1}{2}\operatorname{atan2}(\hat{a}_2, \hat{a}_1).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U106](rewrites.md#u106).

## S174 — before, sections/methodology.tex:458

Script: 32 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\paragraph{Parameter Distribution Breakdown.} The complete MobilePIXORNeXt detector contains exactly \textbf{753,569 trainable parameters} ( \,M), corresponding to an ultra-compact FP32 checkpoint size of  \,MB: \begin{itemize} \item \textbf{Backbone Body}:   parameters ( ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U110](rewrites.md#u110).

## S175 — before, sections/methodology.tex:458

Script: 25 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__methodology.tex.txt).

```latex
\end{itemize} This compact footprint enables full end-to-end inference at  \,FPS ( \,ms) and network forward throughput of  \,FPS ( \,ms) on commodity NVIDIA Tesla T4 GPUs.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U112](rewrites.md#u112).

## S176 — before, sections/related_work.tex:6

Script: 31 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__related_work.tex.txt).

```latex
On one hand, PointNet-based pillar encoders~  achieve expressive geometric learning, yet the non-deterministic point gathering, dynamic memory allocation, and sparse scatter operations incur prohibitive latency and energy penalties on edge accelerators.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U114](rewrites.md#u114).

## S177 — before, sections/related_work.tex:6

Script: 38 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__related_work.tex.txt).

```latex
By deterministically integrating four stratified vertical density bins, four continuous elevation moments ( ), and two continuous subcell centroid displacement offsets ( ) into a 14-channel layout, Strata-BEV14 preserves sub-voxel geometric fidelity while reducing Host-to-Device transfer payloads by   without learned encoder latency.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U114](rewrites.md#u114).

## S178 — before, sections/related_work.tex:13

Script: 29 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__related_work.tex.txt).

```latex
The ConvNeXt paradigm~  demonstrated that conventional convolutions modernized with   depthwise kernels, inverted bottleneck expansions ( ), and LayerScale can achieve representational power competitive with Vision Transformers while preserving convolutional hardware efficiency.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U116](rewrites.md#u116).

## S179 — before, sections/related_work.tex:24

Script: 26 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__related_work.tex.txt).

```latex
Third, unconstrained multi-task uncertainty weighting allows task variance parameters to diverge, leading to training instability when balancing disparate loss scales across classification, geometric overlap, and orientation.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U120](rewrites.md#u120).

## S180 — before, sections/related_work.tex:24

Script: 73 từ; 5 bộ đếm mệnh đề. Nhóm: `prose-reviewed`. [Báo cáo gốc](before/analyze_sentences/sections__related_work.tex.txt).

```latex
Specifically, OGA unifies multi-axis MGIoU projections with scale-normalized  -symmetric corner supervision to eliminate boundary discontinuities; Q-OGA couples continuous IoU quality directly into classification supervision to achieve quality calibration with zero head overhead; IQA provides explicit multi-branch quality alignment scored via   to suppress miscalibrated false positives; and Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW) guarantees stable convergence by constraining multi-task weights within a normalized, bounded simplex, simultaneously resolving orientation discontinuities, quality miscalibration, and loss scale divergence.
```

Đã biên tập văn xuôi ở đơn vị liên kết; báo cáo gốc được giữ để đối chiếu. [Đối chiếu U120](rewrites.md#u120).

## S181 — before, sections/results.tex:8

Script: 46 từ; 6 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
\begin{figure*}[t] \centering \paperfigure[width=\textwidth]{pareto_tradeoff.pdf}{48mm}{Pareto trade-off efficiency on the KITTI validation benchmark.} Detection accuracy ( , \%) versus parameter footprint (M) on a logarithmic scale, highlighting that MobilePIXORNeXt (red star) achieves   mAP with only   parameters, matching SECOND ( ,  ) and outperforming PointPillars ( ,  ) with   fewer parameters.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị. [Đối chiếu U122](rewrites.md#u122).

## S182 — before, sections/results.tex:8

Script: 36 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
\textbf{(b)} Detection accuracy ( , \%) versus end-to-end pipeline latency (ms) at batch size 1 (Tesla T4, except the historical MobilePIXOR result on L4), demonstrating that MobilePIXORNeXt achieves the fastest complete perception pipeline ( ,  ) among all evaluated real-time detectors.}
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U122](rewrites.md#u122).

## S183 — before, sections/results.tex:28

Script: 36 từ; 7 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
As evidenced in Table~  and the Precision--Recall curves in Figure~ , MobilePIXORNeXt achieved \textbf{60.75\% Moderate AP} on Pedestrians, outperforming PointPillars ( , a   gain) and SECOND ( , a   gain), while more than doubling the baseline MobilePIXOR ( , a   gain).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U125](rewrites.md#u125).

## S184 — before, sections/results.tex:28

Script: 50 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
We attribute this performance advantage primarily to two architectural mechanisms: (i) Strata-BEV14's continuous subcell displacement offsets  , which mitigate the   grid quantization error that inherently impairs small object localization; and (ii) the decoupled GroupedHeader architecture, which isolates pedestrian regression from large vehicle footprints to alleviate gradient interference across morphologically disparate categories.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U125](rewrites.md#u125).

## S185 — before, sections/results.tex:31

Script: 28 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
Balanced Multi-Class Detection.} For passenger vehicles (Car), MobilePIXORNeXt achieved   Moderate AP, closely trailing PointPillars ( ) and SECOND ( ) by less than  , while outperforming the original PIXOR ( ) by   and MobilePIXOR ( ) by  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S186 — before, sections/results.tex:31

Script: 33 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
While two-stage 3D anchor-based detectors (PointRCNN, Part-A , PV-RCNN) achieve higher absolute accuracy on rigid vehicles, they require between   and   parameters and operate at higher latencies ( ), exceeding the real-time constraints of embedded edge processors.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U126](rewrites.md#u126).

## S187 — before, sections/results.tex:37

Script: 25 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
A complete perception system must account for point cloud loading, input feature tensorization, Host-to-Device (H2D) memory transfer, forward execution, coordinate decoding, and Non-Maximum Suppression (NMS).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U127](rewrites.md#u127).

## S188 — before, sections/results.tex:42

Script: 34 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
End-to-End Pipeline Latency Comparison.} As detailed in Table~ , MobilePIXORNeXt achieved an End-to-End (E2E) pipeline latency of \textbf{60.96\,ms} (\textbf{16.41\,FPS}), delivering the lowest full-system latency among all evaluated real-time detectors.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U128](rewrites.md#u128).

## S189 — before, sections/results.tex:42

Script: 16 từ; 7 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
It operated faster than PointPillars ( ,  ), MobilePIXOR ( ,  ), and SECOND ( ,  ), and ran over   faster than the original PIXOR ( ,  ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U128](rewrites.md#u128).

## S190 — before, sections/results.tex:42

Script: 34 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
Percentile analysis indicates consistent latency bounds in the evaluated environment: MobilePIXORNeXt exhibited a median P50 latency of \textbf{58.28\,ms}, P95 of  , and P99 of  , demonstrating low latency dispersion under desktop GPU benchmarking.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U128](rewrites.md#u128).

## S191 — before, sections/results.tex:45

Script: 42 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
\item \textbf{Host-to-Device Memory Transfer ( )}: Unlike 35-channel binary occupancy tensors that require transferring   per scan, Strata-BEV14's compact 14-channel layout requires transferring only   (a   payload reduction), which, in conjunction with pinned-memory buffer optimization, reduces H2D latency to   compared with   for the MobilePIXOR baseline.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U131](rewrites.md#u131).

## S192 — before, sections/results.tex:66

Script: 40 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
Near-Range Safety Envelope ( ).} Within the near-range   evaluation zone, MobilePIXORNeXt delivered substantial accuracy gains over baseline MobilePIXOR: Pedestrian AP reached \textbf{65.23\%} (more than double MobilePIXOR's  , a   improvement), while Cyclist AP reached \textbf{76.32\%} (nearly triple MobilePIXOR's  , a   improvement).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U135](rewrites.md#u135).

## S193 — before, sections/results.tex:81

Script: 42 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
Backbone Architecture (A0   A1).} Replacing MobilePIXOR's residual blocks with our ConvNeXt-style backbone (  depthwise convolutions, inverted bottleneck expansion  , SiLU activations, and LayerScale) elevated Moderate mAP from   to   (\textbf{+6.20\% mAP}), while nearly doubling standalone forward throughput from   to \textbf{169.97\,FPS}.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U138](rewrites.md#u138).

## S194 — before, sections/results.tex:118

Script: 27 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
As shown in Table~ , introducing OGA alone into baseline training increased Moderate mAP from   to   (\textbf{+1.59\% mAP}), with consistent gains across Car ( ), Pedestrian ( ), and Cyclist ( ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U148](rewrites.md#u148).

## S195 — before, sections/results.tex:118

Script: 40 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
These findings demonstrate that OGA and IQA provide complementary benefits: OGA refines continuous corner and boundary alignment during backpropagation, whereas IQA calibrates classification scores with true geometric overlap during inference ( ), preventing misaligned boxes from suppressing high-quality candidates during rotated NMS.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U148](rewrites.md#u148).

## S196 — before, sections/results.tex:120

Script: 33 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
These findings confirm that Q-OGA provides a parsimonious edge-deployment configuration when memory and latency budgets are strictly capped, whereas the explicit IQA configuration offers peak accuracy ( ) when resources accommodate a dedicated quality branch.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U149](rewrites.md#u149).

## S197 — before, sections/results.tex:135

Script: 54 từ; 7 bộ đếm mệnh đề. Nhóm: `prose-reviewed`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
\paragraph{Multi-Task Optimization Dynamics and Convergence.} To analyze empirical training stability and convergence behavior across the ablation sequence, Figure~  illustrates the training and validation loss trajectories across 50 epochs for the overall composite objective alongside the six constituent multi-task components (Classification, Geometry, IoU Quality, Subcell Offset, Size, and Yaw), tracking configurations A1 through A10.
```

Đã biên tập văn xuôi ở đơn vị liên kết; báo cáo gốc được giữ để đối chiếu. [Đối chiếu U153](rewrites.md#u153).

## S198 — before, sections/results.tex:147

Script: 65 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
\setlength{\tabcolsep}{1.5pt} \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcrrrrr@{}} \toprule \textbf{Selection} & \textbf{Epoch} & \textbf{Car} & \textbf{Ped.} & \textbf{Cyc.} & \textbf{mAP } & \textbf{E2E} \\ \midrule \textbf{Best AP} & \textbf{130} & \textbf{86.90} & 60.75 & 66.73 & \textbf{71.46} & \textbf{60.96} \\ Best loss & 47 & 82.71 & \textbf{62.30} & \textbf{69.43} & \textbf{71.48} & 81.23 \\ \bottomrule \end{tabular*} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S199 — before, sections/results.tex:166

Script: 37 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](before/analyze_sentences/sections__results.tex.txt).

```latex
\begin{figure}[t] \centering \paperfigure[width=\columnwidth,height=0.38\textheight,keepaspectratio]{qualitative_detections.pdf}{65mm}{Qualitative BEV detection visualizations on KITTI validation scenes.} ) on KITTI validation frames \texttt{007038}, \texttt{005571}, and \texttt{006761} (top to bottom).
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S200 — before, tables/attention.tex:5

Script: 116 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/tables__attention.tex.txt).

```latex
\setlength{\tabcolsep}{5pt} \begin{tabular}{@{}llrrrr@{}} \toprule \textbf{\shortstack[l]{SimAM Attention\\Placement}} & \textbf{\shortstack[l]{Neck Fusion\\Width}} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{Standalone\\Forward (ms)  }} & \textbf{\shortstack{Standalone\\FPS  }} \\ \midrule None (No attention) &   & \underline{70.58} & \underline{72.66} & \underline{8.19} & \underline{122.16} \\ Stage-2 ( ) &   & 69.44 & 71.49 & 7.30 & 136.91 \\ Stage-3 ( ) &   & 69.80 & 71.98 & \textbf{6.94} & \textbf{144.05} \\ Stage-4 ( ) &   & 69.36 & 71.34 & 7.44 & 134.43 \\ \textbf{Block-level ( )} & \textbf{ } & \textbf{71.46} & \textbf{73.42} & 8.67 & 115.32 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.} All configurations share the OGA loss, IQA quality head, Strata-BEV14 input, and PC-GTA data augmentation.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S201 — before, tables/comparison.tex:5

Script: 176 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/tables__comparison.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular}{@{}lrrrrrr@{}} \toprule \textbf{Method} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Car AP \\(\%)  }} & \textbf{\shortstack{Pedestrian\\AP  (\%)  }} & \textbf{\shortstack{Cyclist\\AP  (\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} \\ \midrule PIXOR (Car only)~  & 2.135 & 72.02 & -- & -- & -- & -- \\ MobilePIXOR~  & \textbf{0.598} & 75.52 & 28.16 & 20.53 & 41.40 & 42.27 \\ PointPillars~  & 4.835 & \underline{88.02} & 56.01 & 66.34 & 70.13 & 72.25 \\ SECOND~  & 5.326 & \textbf{88.56} & \underline{56.56} & \textbf{71.16} & \textbf{72.09} & \textbf{73.79} \\ \textbf{MobilePIXORNeXt (Ours)} & \underline{0.754} & 86.90 & \textbf{60.75} & \underline{66.73} & \underline{71.46} & \underline{73.42} \\ \midrule \multicolumn{7}{@{}l}{\textit{Heavyweight 3D Anchor-Based Reference Detectors}} \\ PointRCNN~  & 4.039 & 88.42 & 57.84 & 73.87 & 73.38 & 75.31 \\ Part-A -Anchor~  & 63.815 & 90.05 & 64.20 & 74.60 & 76.28 & 78.04 \\ PV-RCNN~  & 13.116 & 90.29 & 58.48 & 74.54 & 74.44 & 76.49 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.}   is the unweighted mean across the three target classes at Moderate difficulty.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S202 — before, tables/components.tex:5

Script: 293 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/tables__components.tex.txt).

```latex
\textbf{(a) Architectural configurations and component progression}\par\smallskip \begingroup\footnotesize\setlength{\tabcolsep}{4pt} \begin{tabular}{@{}lllllll@{}} \toprule \textbf{Step} & \textbf{Backbone} & \shortstack[l]{\textbf{BEV}\\ \textbf{Input}} & \shortstack[l]{\textbf{Head}\\ \textbf{Architecture}} & \shortstack[l]{\textbf{GT}\\ \textbf{Sampling}} & \shortstack[l]{\textbf{Fusion /}\\ \textbf{Neck}} & \shortstack[l]{\textbf{Supervision /}\\ \textbf{Attention}} \\ \midrule A0 & MobilePIXOR & Binary35  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A1 & MobilePixorNeXt & Binary35  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A2 & MobilePixorNeXt & Strata-BEV8  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A3 & MobilePixorNeXt & Strata-BEV14 & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A4 & MobilePixorNeXt & Strata-BEV14 & Grouped H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A5 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A6 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F24 / SumFPN & Baseline Smooth-L1 \\ A7 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SumFPN & Baseline Smooth-L1 \\ A8 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & Baseline Smooth-L1 \\ A9 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & OGA + IQA Head \\ A10 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & OGA + IQA + Block-SimAM \\ \bottomrule \end{tabular}\endgroup \par\medskip \textbf{(b) Quantitative detection accuracy and standalone model inference speed}\par\smallskip \begingroup\footnotesize\setlength{\tabcolsep}{3pt} \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l>{\raggedright\arraybackslash}p{0.42\textwidth}rrrr@{}} \toprule \textbf{Step} & \textbf{\shortstack[l]{Key Modification /\\Investigated Factor}} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{Standalone Forward\\(ms)  }} & \textbf{\shortstack{Standalone FPS\\ }} \\ \midrule A0 & Baseline MobilePIXOR starting checkpoint & 41.40 & 42.27 & 11.05 & 90.57 \\ A1 & + ConvNeXt-style backbone (  DW, SiLU, Inverted Bottleneck) & 47.60 & 49.08 & \textbf{5.88} & \textbf{169.97} \\ A2 & + 8-channel RichBEV statistical representation & 47.37 & 48.95 & 6.06 & 165.02 \\ A3 & + 14-channel Strata-BEV14 (subcell offsets, height moments) & 50.14 & 52.00 & \underline{5.92} & \underline{169.01} \\ A4 & + Decoupled Grouped Header (Car vs.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S203 — before, tables/components.tex:5

Script: 130 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/tables__components.tex.txt).

```latex
Ped/Cyc, 16 channels) & 51.30 & 53.96 & 6.92 & 144.58 \\ A5 & + Expanded Grouped Header capacity (H32, 32 channels) & 60.22 & 61.11 & 6.70 & 149.15 \\ A6 & + Physics-Aware Ground-Truth Augmentation (PC-GTA) & 66.25 & 68.45 & 7.45 & 134.25 \\ A7 & + Expanded neck fusion layer ( , 32 channels) & 68.12 & 70.05 & 6.72 & 148.82 \\ A8 & + Bilinear Depthwise Scale-Gated Neck (SG-FPN) & 67.38 & 69.46 & 6.94 & 144.16 \\ A9 & + Oriented Geometric Alignment (OGA) and IQA supervision & \underline{70.58} & \underline{72.66} & 8.19 & 122.16 \\ A10 & + Parameter-Free Block-SimAM Energy Attention inside all blocks & \textbf{71.46} & \textbf{73.42} & 8.67 & 115.32 \\ \bottomrule \end{tabular*}\endgroup \par\vspace{2pt} {\footnotesize \textit{Note.} Step identifiers A0--A10 correspond to the experimental progression from baseline R01 to peak configuration.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S204 — before, tables/distance_our_vs_mbpixor.tex:5

Script: 95 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/tables__distance_our_vs_mbpixor.tex.txt).

```latex
\setlength{\tabcolsep}{1.5pt} \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrr@{}} \toprule \textbf{Category} & \textbf{Model} & \textbf{\shortstack{0--30\,m}} & \textbf{\shortstack{30--50\,m}} & \textbf{\shortstack{50--70.4\,m}} \\ \midrule Pedestrian & MobilePIXOR~  & 30.45 & 3.75 & \textbf{0.99} \\ & \textbf{Ours} & \textbf{65.23} & \textbf{11.49} & 0.80 \\ & \textit{Diff.} & \textit{+34.78} & \textit{+7.74} & \textit{-0.19} \\ \midrule Cyclist & MobilePIXOR~  & 25.55 & 12.61 & 0.56 \\ & \textbf{Ours} & \textbf{76.32} & \textbf{44.30} & \textbf{14.94} \\ & \textit{Diff.} & \textit{+50.77} & \textit{+31.69} & \textit{+14.38} \\ \bottomrule \end{tabular*} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S205 — before, tables/efficiency.tex:5

Script: 231 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/tables__efficiency.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular}{@{}lrrrrrrrr@{}} \toprule \textbf{Method} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Model\\(ms)  }} & \textbf{\shortstack{Model\\FPS  }} & \textbf{\shortstack{E2E\\(ms)  }} & \textbf{\shortstack{E2E\\FPS  }} & \textbf{\shortstack{P50\\(ms)  }} & \textbf{\shortstack{P95\\(ms)  }} & \textbf{\shortstack{P99\\(ms)  }} \\ \midrule PIXOR~  & 2.135 & 70.79 & 14.13 & 144.00 & 6.94 & 142.29 & 159.38 & 175.43 \\ MobilePIXOR (L4)~  & \textbf{0.598} & \textbf{11.05} & \textbf{90.50} & 85.92 & 11.64 & 84.11 & 112.52 & 124.94 \\ PointPillars~  & 4.835 & 32.13 & 31.12 & \underline{63.84} & \underline{15.66} & \underline{62.10} & \textbf{75.88} & \textbf{92.73} \\ SECOND~  & 5.326 & 62.64 & 15.96 & 93.70 & 10.67 & 92.02 & 107.42 & 126.35 \\ \textbf{MobilePIXORNeXt (Ours)} & \underline{0.754} & \underline{17.88} & \underline{55.94} & \textbf{60.96} & \textbf{16.41} & \textbf{58.28} & \underline{80.85} & \underline{95.91} \\ \midrule \multicolumn{9}{@{}l}{\textit{Heavyweight 3D Anchor-Based Reference Detectors}} \\ PointRCNN~  & 4.039 & 169.36 & 5.90 & 194.50 & 5.14 & 191.24 & 213.48 & 235.10 \\ Part-A -Anchor~  & 63.815 & 102.57 & 9.75 & 133.75 & 7.48 & 131.50 & 150.49 & 172.80 \\ PV-RCNN~  & 13.116 & 160.79 & 6.22 & 188.60 & 5.30 & 186.40 & 202.62 & 224.15 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.} Model latency measures neural network execution time within the evaluation pipeline on Tesla T4.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S206 — before, tables/efficiency.tex:5

Script: 27 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](before/analyze_sentences/tables__efficiency.tex.txt).

```latex
End-to-end (E2E) pipeline latency spans raw LiDAR point cloud loading, deterministic statistical rasterization, Host-to-Device (H2D) memory transfer, forward propagation, subcell coordinate decoding, and class-wise rotated NMS filtering.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U175](rewrites.md#u175).

## S207 — before, tables/loss.tex:5

Script: 168 từ; 4 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/tables__loss.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{0.26\textwidth}ccrrrrr@{}} \toprule \textbf{Training Objective} & \textbf{\shortstack{IQA\\Head}} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Forward\\(ms)  }} & \textbf{\shortstack{Car AP \\(\%)  }} & \textbf{\shortstack{Pedestrian\\AP  (\%)  }} & \textbf{\shortstack{Cyclist\\AP  (\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} \\ \midrule Baseline Smooth-L1 & No  & \textbf{0.702} & \textbf{6.94} & 83.92 & 55.48 & 62.74 & 67.38 \\ OGA (MGIoU +  -Corner) & No  & \textbf{0.702} & \textbf{6.94} & 84.09 & 57.93 & 64.88 & 68.97 \\ Baseline Smooth-L1 & Yes & 0.721 & 8.19 & 84.13 & \underline{59.45} & \textbf{67.34} & 70.31 \\ \textbf{OGA (MGIoU +  -Corner)} & \textbf{Yes} & 0.721 & 8.19 & \textbf{84.87} & \textbf{59.63} & \underline{67.24} & \textbf{70.58} \\ \midrule Q-OGA (Quality-Coupled Focal Loss) & No & \textbf{0.702} & \textbf{6.94} & \underline{84.35} & 58.12 & 65.13 & 69.20 \\ \bottomrule \end{tabular*} \par\vspace{2pt} {\footnotesize \textit{Note.} All variants share identical MobilePixorNeXt backbone, Strata-BEV14 input, Grouped H32 heads, F32 SG-FPN neck, and PC-GTA augmentation.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S208 — before, tables/setup.tex:5

Script: 171 từ; 12 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](before/analyze_sentences/tables__setup.tex.txt).

```latex
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{.38\columnwidth}>{\raggedright\arraybackslash}p{.56\columnwidth}@{}} \toprule \textbf{Configuration Element} & \textbf{Specification / Value} \\ \midrule LiDAR Input Representation & Strata-BEV14 (14 channels; NumPy/Numba CPU) \\ Spatial ROI bounds ( ) &   \\ Grid resolution ( ) &   \\ Input / Output Grid Dimensions &   /   ( ) \\ Detector Parameters & 753,569 trainable ( ); 9.34\,MB FP32 \\ Backbone Stages ( ) & Widths: (48, 96, 128); Depths: (3, 4, 2) \\ Inverted Bottleneck Expansion &  ;   Depthwise Convolution \\ Attention Mechanism & Parameter-free Block-SimAM (9 residual blocks) \\ Multiscale Neck & Scale-Gated FPN (SG-FPN,   fusion width) \\ Prediction Towers & GroupedHeader: Car ( ), Ped/Cyc ( ) \\ Loss Formulation & OGA (MGIoU +  -Corner) + IQA + T-SBUW \\ Data Augmentation & Physics-Consistent GT (PC-GTA) + Global \\ Optimizer / Base Learning Rate & AdamW (fused) /   \\ Minimum LR / Standard Schedule &   / 50 epochs: 4 warmup + 46 cosine \\ Final Reported Checkpoint & Extended run: 160 epochs, 10 warmup; selected at epoch 130 \\ Weight Decay &   (2D+ tensors only; biases/norm: 0.00) \\ Gradient Clipping Norm & Maximum   norm = 10.00 \\ Batch Size (Train / Val) & 16 / 32 \\ Numerical Precision & Automatic Mixed Precision (Bfloat16 / FP32) \\ Hardware Environment & NVIDIA Tesla T4 GPU (16\,GB GDDR6, 98.44\,MB peak) \\ \bottomrule \end{tabular} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S209 — after, sections/introduction.tex:5

Script: 20 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Vehicle perception systems must detect, orient, and classify nearby objects, including passenger vehicles, pedestrians, and cyclists, within strict latency limits.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U028](rewrites.md#u028).

## S210 — after, sections/introduction.tex:7

Script: 21 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
They are faster than volumetric methods, although point gathering, dynamic memory allocation, and sparse scattering add runtime and memory transfer overhead.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U029](rewrites.md#u029).

## S211 — after, sections/introduction.tex:9

Script: 54 từ; 1 bộ đếm mệnh đề. Nhóm: `question-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Our research question was: \textit{Can an ultra-compact LiDAR detector operating on a parameter-free BEV tensor rival the localization accuracy of heavy volumetric and pillar architectures while achieving superior end-to-end throughput on commodity edge hardware?} We coupled this representation with a single-branch ConvNeXt-style backbone~  using large   depthwise kernels and parameter-free 3D spatial-energy attention (SimAM)~ .
```

Script gộp câu hỏi kết thúc bằng ?} với câu tiếp theo; kiểm tra riêng từng câu. [Đối chiếu U030](rewrites.md#u030).

## S212 — after, sections/introduction.tex:11

Script: 28 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
The top-down \textbf{Scale-Gated Feature Pyramid Network (SG-FPN Neck,  )} balances semantic context and fine coordinates through bilinear upsampling, depthwise refinement, and learnable scale gates, avoiding deconvolution checkerboard artifacts.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S213 — after, sections/related_work.tex:6

Script: 24 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Pillar encoders~  learn geometric features but require point gathering, dynamic memory allocation, and sparse scattering, which add latency and energy costs on edge accelerators.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U114](rewrites.md#u114).

## S214 — after, sections/related_work.tex:11

Script: 13 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
PointRCNN~ , Part- ~ , and PV-RCNN~  developed two-stage point processing, part-aware aggregation, and voxel-set abstractions.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U115](rewrites.md#u115).

## S215 — after, sections/methodology.tex:5

Script: 15 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Decoupled grouped prediction towers output heatmaps, subcell offsets, log-sizes, doubled yaw angles, and localization quality.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S216 — after, sections/methodology.tex:18

Script: 32 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Perception is restricted to a Cartesian Region of Interest (ROI): \begin{equation} \begin{aligned} x_i &\in [x_{\min}, x_{\max}], \\ y_i &\in [y_{\min}, y_{\max}], \\  z_i &\in [z_{\min}, z_{\max}].
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U038](rewrites.md#u038).

## S217 — after, sections/methodology.tex:26

Script: 14 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{equation} We used longitudinal bounds  \,m, lateral bounds  \,m, and vertical bounds  \,m.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U039](rewrites.md#u039).

## S218 — after, sections/methodology.tex:33

Script: 25 từ; 8 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
For each detected object instance  , the target state is parameterized by the 7-tuple: \begin{equation} \mathbf{b}_k = (c_k, x_k^*, y_k^*, l_k^*, w_k^*, \psi_k^*, s_k),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S219 — after, sections/methodology.tex:45

Script: 65 từ; 7 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Channels 0–3 capture stratified height-band densities, Channels 4–7 encode vertical elevation statistics (peak, mean, span, deviation), Channels 8–9 capture material surface reflectance, Channels 10–11 represent global density and occupancy, and Channels 12–13 encode continuous subcell horizontal offsets to eliminate grid discretization error.} \captionof{figure}{Fourteen deterministic feature channels in the Strata-BEV14 representation for a KITTI driving scene (  grid at   resolution).
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S220 — after, sections/methodology.tex:69

Script: 52 từ; 9 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textit{Vertical Spatial Moments ( -- )}: To encode obstacle geometry without explicit voxelization, we extract peak elevation  , mean elevation  , vertical span  , and the sample standard deviation: \begin{equation} F_7 = \begin{cases} \sqrt{\max\left(0, \frac{1}{N} \sum_{i=1}^N \zeta_i^2 - F_5^2\right)}, & \text{if } N > 1, \\ 0, & \text{otherwise}.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S221 — after, sections/methodology.tex:78

Script: 39 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textit{Reflectance Statistics ( -- )}: Calibrated surface returns capture material discrimination (e.g., distinguishing retro-reflective license plates, pedestrian clothing, or road signs): \begin{equation} F_8 = \max_{i=1,\dots,N} \iota_i, \qquad F_9 = \frac{1}{N} \sum_{i=1}^N \iota_i.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S222 — after, sections/methodology.tex:84

Script: 80 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
For  \,m cells, channels   and   retain continuous horizontal subcell centroid displacements to reduce quantization error: \begin{align} x_u^{\mathrm{cell}} &= x_{\min} + (u + 0.5)\Delta_x, \nonumber\\ y_v^{\mathrm{cell}} &= y_{\min} + (v + 0.5)\Delta_y, \nonumber\\ F_{12} &= \operatorname{clip}\left(\frac{\sum_{i=1}^N (x_i - x_u^{\mathrm{cell}})}{N\Delta_x}, -0.5, 0.5\right), \nonumber\\ F_{13} &= \operatorname{clip}\left(\frac{\sum_{i=1}^N (y_i - y_v^{\mathrm{cell}})}{N\Delta_y}, -0.5, 0.5\right).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U049](rewrites.md#u049).

## S223 — after, sections/methodology.tex:130

Script: 83 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
The mechanisms can be enabled independently; the settings used for the reported final run are given in Section~ : \begin{enumerate} \item \textit{Adaptive Ground Plane Fitting}: To prevent floating or sunken objects, local road elevation is estimated within a range-dependent search radius: \begin{align} R(d) &= \operatorname{clip}(2.5 + 0.03\max(0, d - 15), 2.5, 4.0)\,\mathrm{m},\nonumber\\[0.8em] \hat{z}_{\mathrm{ground}} &= \operatorname{percentile}(\{z_j \mid \mathbf{p}_j \in \mathcal{N}(x, y)\}, 5),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S224 — after, sections/methodology.tex:146

Script: 52 từ; 1 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
It masks points behind the exit face ( ) whose radial distance satisfies  \,m: \begin{equation} \begin{aligned} \mathcal{P}_{\mathrm{augmented}} &= \mathcal{P}_{\mathrm{target}} \setminus \{\mathbf{p} \in \mathcal{P}_{\mathrm{target}} \mid \\ &\qquad \operatorname{Mask}_{\mathrm{shadow}}(\mathbf{p}) = \mathrm{True}\} \cup \mathcal{P}_{\mathrm{object}}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U060](rewrites.md#u060).

## S225 — after, sections/methodology.tex:158

Script: 50 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\item \textit{Range-Adaptive Point and Intensity Attenuation}: If an object captured at source distance   is placed at target distance  , its point cloud is randomly subsampled according to the inverse-square divergence law: \begin{equation} P_{\mathrm{keep}} = \min\left(1, \left(\frac{r_{\mathrm{src}}}{r_{\mathrm{tgt}}}\right)^2\right),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S226 — after, sections/methodology.tex:162

Script: 47 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{equation} and the optional radiometric adjustment attenuates reflectance intensity according to: \[ \begin{aligned} \eta_{\mathrm{int}} &= \left(\frac{r_{\mathrm{src}}}{r_{\mathrm{tgt}}}\right)^{\max(0, 2 - \gamma)}, \qquad \gamma = 1.7, \\ I_{\mathrm{tgt}} &= \operatorname{clip}(I_{\mathrm{src}}\eta_{\mathrm{int}}, 0, 1).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S227 — after, sections/methodology.tex:214

Script: 29 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
(Top) Complete feedforward path from input X to output Y through the 7x7 depthwise convolution, inverted bottleneck expansion (e=2.5), SimAM modulation, LayerScale scaling, and identity residual addition.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S228 — after, sections/methodology.tex:214

Script: 33 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
(Bottom) Zoom-in structure of the parameter-free SimAM module, detailing spatial moment estimation, inverse-energy importance calculation, sigmoid saliency mapping, and element-wise feature recalibration.} \captionof{figure}{The \texttt{MobilePixorNeXt-S} residual block and parameter-free Block-SimAM attention.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S229 — after, sections/methodology.tex:221

Script: 57 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Given input tensor  , each block applies an inverted bottleneck with expansion ratio  : \begin{align} \mathbf{V} &= \operatorname{BN}(\operatorname{DWConv}_{7\times7}(\mathbf{Z})),\nonumber\\ \mathbf{R} &= \operatorname{BN}\left(W_{\mathrm{red}} \operatorname{SiLU}(W_{\mathrm{exp}}\mathbf{V})\right),\nonumber\\ \mathbf{Z}' &= \mathbf{Z} + \boldsymbol{\gamma}_{\mathrm{blk}} \odot \mathcal{S}(\mathbf{R}).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S230 — after, sections/methodology.tex:234

Script: 31 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{Design and Placement.} As illustrated in Figure~ , SimAM is embedded inside each \texttt{MobilePixorNeXt-\allowbreak S} block across all three stages (C2, C3, C4), for nine attention operators in total.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S231 — after, sections/methodology.tex:237

Script: 44 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
For each channel slice with   spatial locations, spatial mean   and sample variance   are computed: \begin{equation} \begin{aligned} \mu_c &= \frac{1}{M}\sum_{u, v} R_{c, u, v}, \\ \nu_c &= \frac{1}{\max(M-1, 1)}\sum_{u, v} (R_{c, u, v} - \mu_c)^2.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S232 — after, sections/methodology.tex:244

Script: 56 từ; 2 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{equation} Following the energy-minimization formulation of SimAM~ , the minimized energy   and its inverse-energy importance score   for neuron   are: \begin{equation} \begin{aligned} e_t^\star &= \frac{4(\nu_c + \lambda_{\mathrm{sim}})}{(t-\mu_c)^2 + 2(\nu_c + \lambda_{\mathrm{sim}})}, \\ s_t &= \frac{1}{e_t^\star} = \frac{(t - \mu_c)^2}{4(\nu_c + \lambda_{\mathrm{sim}})} + \frac{1}{2}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S233 — after, sections/methodology.tex:282

Script: 27 từ; 8 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{Mathematical Formulation.} As shown in Figure~ , lateral features from Stages C4 ( \,ch), C3 ( \,ch), and C2 ( \,ch) are projected via   convolutions into matching lateral dimensions  ,  , and  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S234 — after, sections/methodology.tex:296

Script: 76 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\par\vspace{-0.5\baselineskip} \noindent \begin{minipage}{\columnwidth} \centering \paperfigure{geometry_output_2x2.pdf}{75mm}{Geometric parameterization, orientation encoding, and quality-aware score calibration across prediction heads: (a) Continuous subcell coordinate offsets   from anchor point   to object center  ; (b) Log-transformed bounding box dimension targets  ; (c) Doubled-angle trigonometric yaw representation   on the unit circle resolving  -heading ambiguity; and (d) IoU-quality-aware composite score ranking   calibrated for post-processing peak selection and rotated NMS.} \captionof{figure}{Prediction-head geometry, orientation, and quality-aware scoring.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S235 — after, sections/methodology.tex:306

Script: 18 từ; 11 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Cars are rectangular ( \,m,  \,m), while Pedestrians ( \,m,  \,m) and Cyclists ( \,m,  \,m) are smaller and more deformable.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U087](rewrites.md#u087).

## S236 — after, sections/methodology.tex:319

Script: 30 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
For output cell  , the ground-truth geometric targets are formulated as: \begin{align} \mathbf{o}^* &=  ,\nonumber\\ \mathbf{d}^* &= (\log w^*, \log l^*), \qquad \mathbf{a}^* = (\cos 2\psi^*, \sin 2\psi^*).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S237 — after, sections/methodology.tex:333

Script: 62 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textit{Multi-Axis Projection GIoU ( )}: Projected bounding box overlap is evaluated following Marginalized Generalized IoU~  across four unit normal projection axes defining the prediction and ground-truth rectangles: \begin{equation} \mathbf{n}_k \in \left\{ \mathbf{n}_{\mathrm{len}}^{\mathrm{pred}}, \mathbf{n}_{\mathrm{wid}}^{\mathrm{pred}}, \mathbf{n}_{\mathrm{len}}^{\mathrm{gt}}, \mathbf{n}_{\mathrm{wid}}^{\mathrm{gt}} \right\}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U096](rewrites.md#u096).

## S238 — after, sections/methodology.tex:339

Script: 52 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{equation} For each axis  , 1D intervals of intersection length  , union length  , and smallest enclosing hull   are determined, yielding 1D GIoU~ : \begin{equation} \begin{aligned} g_k &= \frac{I_k}{U_k} - \frac{E_k - U_k}{E_k}, \\ \bar{g} &= \frac{1}{4}\sum_{k=1}^4 g_k, \qquad \ell_{\mathrm{proj}} = \frac{1 - \bar{g}}{2}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S239 — after, sections/methodology.tex:348

Script: 78 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Scale-Normalized  -Symmetric Corner Distance ( )}:\par}\noindent To ensure accurate vertex localization under doubled-angle symmetry, we evaluate the minimum L1 distance across direct and  -permuted corner correspondences: \begin{equation} \ell_{\mathrm{corner}} = \frac{\min\left\{ \begin{array}{l} \displaystyle\sum_{m=1}^4 \|\widehat{\mathbf{c}}_m - \mathbf{c}_m^*\|_1, \\ \displaystyle\sum_{m=1}^4 \|\widehat{\mathbf{c}}_m - \mathbf{c}_{5-m}^*\|_1 \end{array} \right.}{4\max\left(\sqrt{(w^*)^2 + (l^*)^2}, 10^{-6}\right)}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S240 — after, sections/methodology.tex:362

Script: 38 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
It is supervised via Binary Cross-Entropy (BCE) using the detached MGIoU similarity as the ground-truth target: \begin{equation} q^* = \operatorname{stopgrad}(\operatorname{clip}(\bar{g}, 0, 1)), \qquad \mathcal{L}_{\mathrm{qual}} = \operatorname{BCE}(\hat{q}, q^*).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S241 — after, sections/methodology.tex:383

Script: 74 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textit{Bounded Softmax Weights}: Learnable logit parameters   are soft-clamped via hyperbolic tangent with bound  , and converted to positive normalized weights via temperature-controlled softmax ( ): \begin{equation} \begin{aligned} \bar{a}_k &= b_{\mathrm{uw}} \tanh(a_k / b_{\mathrm{uw}}), \\ Z_{\mathrm{uw}} &= \sum_{j \in \mathcal{T}} \exp(\bar{a}_j / \tau_{\mathrm{uw}}), \\ \omega_k &= J \cdot \frac{\exp(\bar{a}_k / \tau_{\mathrm{uw}})}{Z_{\mathrm{uw}}}, \\ \sum_{k \in \mathcal{T}} \omega_k &= J.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U104](rewrites.md#u104).

## S242 — after, sections/methodology.tex:411

Script: 61 từ; 4 bộ đếm mệnh đề. Nhóm: `algorithm`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\vspace{2pt} {\small \noindent\textbf{Require:} Raw LiDAR point cloud  , ROI bounds  \par \noindent\textbf{Require:} Score threshold  , Rotated NMS threshold  , quality exponent  \par \noindent\textbf{Ensure:} Final decoded 3D oriented bounding boxes  \par \begin{enumerate} \setlength{\topsep}{2pt} \setlength{\itemsep}{0pt} \setlength{\parsep}{0pt} \renewcommand{\labelenumi}{\textbf{\arabic{enumi}:}} \item Filter raw points   within inward margin  \,m of spatial ROI.
```

Giữ nguyên thuật toán; danh sách đầu vào/đầu ra không phải câu văn cần tách.

## S243 — after, sections/methodology.tex:411

Script: 20 từ; 5 bộ đếm mệnh đề. Nhóm: `algorithm`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\item Evaluate Grouped Prediction Towers for Car ( ) and Pedestrian/Cyclist ( ), yielding classification maps  , subcell offsets  , log-sizes  , doubled yaws  , and quality  .
```

Giữ nguyên thuật toán; danh sách đầu vào/đầu ra không phải câu văn cần tách.

## S244 — after, sections/methodology.tex:441

Script: 75 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
For each candidate cell   whose classification probability   is a   local spatial maximum, bounding boxes are decoded in metric coordinates as: \begin{align} \hat{x} &= x_{\min} + u s_{\mathrm{out}}\Delta_x + \hat{o}_x, \nonumber\\ \hat{y} &= y_{\min} + v s_{\mathrm{out}}\Delta_y + \hat{o}_y, \nonumber\\ \hat{w} &= \exp(\hat{d}_1), \qquad \hat{l} = \exp(\hat{d}_2), \nonumber\\ \hat{\psi} &= \frac{1}{2}\operatorname{atan2}(\hat{a}_2, \hat{a}_1).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U106](rewrites.md#u106).

## S245 — after, sections/methodology.tex:458

Script: 26 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\paragraph{Parameter Distribution Breakdown.} MobilePIXORNeXt contains \textbf{753,569 trainable parameters} ( \,M), with an FP32 checkpoint size of  \,MB: \begin{itemize} \item \textbf{Backbone Body}:   parameters ( ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U110](rewrites.md#u110).

## S246 — after, sections/methodology.tex:458

Script: 20 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{itemize} The complete pipeline achieved  \,FPS ( \,ms), while standalone network throughput reached  \,FPS ( \,ms) on NVIDIA Tesla T4 GPUs.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U112](rewrites.md#u112).

## S247 — after, sections/experiments.tex:7

Script: 37 từ; 8 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
We restricted perception to the Cartesian Region of Interest: \begin{equation} \begin{aligned} x &\in [0.0, 70.4]\,\text{m}, \\ y &\in [-40.0, 40.0]\,\text{m}, \\ z &\in [-2.5, 1.0]\,\text{m}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U014](rewrites.md#u014).

## S248 — after, sections/experiments.tex:18

Script: 43 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\end{itemize} Following autonomous driving literature, \textbf{Moderate macro mAP} ( ) was adopted as the primary benchmark ranking criterion: \begin{equation} \mathrm{mAP}_{\mathrm{M}} = \frac{1}{3} \sum_{c \in \{\mathrm{Car}, \mathrm{Ped}, \mathrm{Cyc}\}} \mathrm{AP}_{c, \mathrm{M}}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S249 — after, sections/experiments.tex:18

Script: 9 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
E, M, and H denote Easy, Moderate, and Hard.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U023](rewrites.md#u023).

## S250 — after, sections/experiments.tex:51

Script: 34 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Final MobilePIXORNeXt timing and accuracy evaluation used an NVIDIA Tesla T4 GPU (16\,GB GDDR6 VRAM, Turing architecture), an x86\_64 host CPU, Linux OS, and PyTorch 2.11.0 with CUDA 13.0.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U027](rewrites.md#u027).

## S251 — after, tables/setup.tex:5

Script: 171 từ; 12 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{.38\columnwidth}>{\raggedright\arraybackslash}p{.56\columnwidth}@{}} \toprule \textbf{Configuration Element} & \textbf{Specification / Value} \\ \midrule LiDAR Input Representation & Strata-BEV14 (14 channels; NumPy/Numba CPU) \\ Spatial ROI bounds ( ) &   \\ Grid resolution ( ) &   \\ Input / Output Grid Dimensions &   /   ( ) \\ Detector Parameters & 753,569 trainable ( ); 9.34\,MB FP32 \\ Backbone Stages ( ) & Widths: (48, 96, 128); Depths: (3, 4, 2) \\ Inverted Bottleneck Expansion &  ;   Depthwise Convolution \\ Attention Mechanism & Parameter-free Block-SimAM (9 residual blocks) \\ Multiscale Neck & Scale-Gated FPN (SG-FPN,   fusion width) \\ Prediction Towers & GroupedHeader: Car ( ), Ped/Cyc ( ) \\ Loss Formulation & OGA (MGIoU +  -Corner) + IQA + T-SBUW \\ Data Augmentation & Physics-Consistent GT (PC-GTA) + Global \\ Optimizer / Base Learning Rate & AdamW (fused) /   \\ Minimum LR / Standard Schedule &   / 50 epochs: 4 warmup + 46 cosine \\ Final Reported Checkpoint & Extended run: 160 epochs, 10 warmup; selected at epoch 130 \\ Weight Decay &   (2D+ tensors only; biases/norm: 0.00) \\ Gradient Clipping Norm & Maximum   norm = 10.00 \\ Batch Size (Train / Val) & 16 / 32 \\ Numerical Precision & Automatic Mixed Precision (Bfloat16 / FP32) \\ Hardware Environment & NVIDIA Tesla T4 GPU (16\,GB GDDR6, 98.44\,MB peak) \\ \bottomrule \end{tabular} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S252 — after, tables/comparison.tex:5

Script: 176 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular}{@{}lrrrrrr@{}} \toprule \textbf{Method} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Car AP \\(\%)  }} & \textbf{\shortstack{Pedestrian\\AP  (\%)  }} & \textbf{\shortstack{Cyclist\\AP  (\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} \\ \midrule PIXOR (Car only)~  & 2.135 & 72.02 & -- & -- & -- & -- \\ MobilePIXOR~  & \textbf{0.598} & 75.52 & 28.16 & 20.53 & 41.40 & 42.27 \\ PointPillars~  & 4.835 & \underline{88.02} & 56.01 & 66.34 & 70.13 & 72.25 \\ SECOND~  & 5.326 & \textbf{88.56} & \underline{56.56} & \textbf{71.16} & \textbf{72.09} & \textbf{73.79} \\ \textbf{MobilePIXORNeXt (Ours)} & \underline{0.754} & 86.90 & \textbf{60.75} & \underline{66.73} & \underline{71.46} & \underline{73.42} \\ \midrule \multicolumn{7}{@{}l}{\textit{Heavyweight 3D Anchor-Based Reference Detectors}} \\ PointRCNN~  & 4.039 & 88.42 & 57.84 & 73.87 & 73.38 & 75.31 \\ Part-A -Anchor~  & 63.815 & 90.05 & 64.20 & 74.60 & 76.28 & 78.04 \\ PV-RCNN~  & 13.116 & 90.29 & 58.48 & 74.54 & 74.44 & 76.49 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.}   is the unweighted mean across the three target classes at Moderate difficulty.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S253 — after, tables/efficiency.tex:5

Script: 229 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular}{@{}lrrrrrrrr@{}} \toprule \textbf{Method} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Model\\(ms)  }} & \textbf{\shortstack{Model\\FPS  }} & \textbf{\shortstack{E2E\\(ms)  }} & \textbf{\shortstack{E2E\\FPS  }} & \textbf{\shortstack{P50\\(ms)  }} & \textbf{\shortstack{P95\\(ms)  }} & \textbf{\shortstack{P99\\(ms)  }} \\ \midrule PIXOR~  & 2.135 & 70.79 & 14.13 & 144.00 & 6.94 & 142.29 & 159.38 & 175.43 \\ MobilePIXOR (L4)~  & \textbf{0.598} & \textbf{11.05} & \textbf{90.50} & 85.92 & 11.64 & 84.11 & 112.52 & 124.94 \\ PointPillars~  & 4.835 & 32.13 & 31.12 & \underline{63.84} & \underline{15.66} & \underline{62.10} & \textbf{75.88} & \textbf{92.73} \\ SECOND~  & 5.326 & 62.64 & 15.96 & 93.70 & 10.67 & 92.02 & 107.42 & 126.35 \\ \textbf{MobilePIXORNeXt (Ours)} & \underline{0.754} & \underline{17.88} & \underline{55.94} & \textbf{60.96} & \textbf{16.41} & \textbf{58.28} & \underline{80.85} & \underline{95.91} \\ \midrule \multicolumn{9}{@{}l}{\textit{Heavyweight 3D Anchor-Based Reference Detectors}} \\ PointRCNN~  & 4.039 & 169.36 & 5.90 & 194.50 & 5.14 & 191.24 & 213.48 & 235.10 \\ Part-A -Anchor~  & 63.815 & 102.57 & 9.75 & 133.75 & 7.48 & 131.50 & 150.49 & 172.80 \\ PV-RCNN~  & 13.116 & 160.79 & 6.22 & 188.60 & 5.30 & 186.40 & 202.62 & 224.15 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.} Model latency measures network execution in the evaluation pipeline on Tesla T4.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S254 — after, tables/efficiency.tex:5

Script: 19 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
End-to-end (E2E) latency includes LiDAR loading, statistical rasterization, Host-to-Device (H2D) transfer, network execution, coordinate decoding, and class-wise rotated NMS.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U175](rewrites.md#u175).

## S255 — after, sections/results.tex:39

Script: 18 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Full-system latency includes point cloud loading, tensorization, Host-to-Device (H2D) transfer, network execution, coordinate decoding, and Non-Maximum Suppression (NMS).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U127](rewrites.md#u127).

## S256 — after, sections/results.tex:44

Script: 8 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
MobilePIXORNeXt ran faster than PointPillars ( ,  ), MobilePIXOR ( ,  ), and SECOND ( ,  ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U128](rewrites.md#u128).

## S257 — after, tables/distance_our_vs_mbpixor.tex:5

Script: 95 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{1.5pt} \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrr@{}} \toprule \textbf{Category} & \textbf{Model} & \textbf{\shortstack{0--30\,m}} & \textbf{\shortstack{30--50\,m}} & \textbf{\shortstack{50--70.4\,m}} \\ \midrule Pedestrian & MobilePIXOR~  & 30.45 & 3.75 & \textbf{0.99} \\ & \textbf{Ours} & \textbf{65.23} & \textbf{11.49} & 0.80 \\ & \textit{Diff.} & \textit{+34.78} & \textit{+7.74} & \textit{-0.19} \\ \midrule Cyclist & MobilePIXOR~  & 25.55 & 12.61 & 0.56 \\ & \textbf{Ours} & \textbf{76.32} & \textbf{44.30} & \textbf{14.94} \\ & \textit{Diff.} & \textit{+50.77} & \textit{+31.69} & \textit{+14.38} \\ \bottomrule \end{tabular*} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S258 — after, tables/components.tex:5

Script: 293 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\textbf{(a) Architectural configurations and component progression}\par\smallskip \begingroup\footnotesize\setlength{\tabcolsep}{4pt} \begin{tabular}{@{}lllllll@{}} \toprule \textbf{Step} & \textbf{Backbone} & \shortstack[l]{\textbf{BEV}\\ \textbf{Input}} & \shortstack[l]{\textbf{Head}\\ \textbf{Architecture}} & \shortstack[l]{\textbf{GT}\\ \textbf{Sampling}} & \shortstack[l]{\textbf{Fusion /}\\ \textbf{Neck}} & \shortstack[l]{\textbf{Supervision /}\\ \textbf{Attention}} \\ \midrule A0 & MobilePIXOR & Binary35  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A1 & MobilePixorNeXt & Binary35  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A2 & MobilePixorNeXt & Strata-BEV8  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A3 & MobilePixorNeXt & Strata-BEV14 & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A4 & MobilePixorNeXt & Strata-BEV14 & Grouped H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A5 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A6 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F24 / SumFPN & Baseline Smooth-L1 \\ A7 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SumFPN & Baseline Smooth-L1 \\ A8 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & Baseline Smooth-L1 \\ A9 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & OGA + IQA Head \\ A10 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & OGA + IQA + Block-SimAM \\ \bottomrule \end{tabular}\endgroup \par\medskip \textbf{(b) Quantitative detection accuracy and standalone model inference speed}\par\smallskip \begingroup\footnotesize\setlength{\tabcolsep}{3pt} \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l>{\raggedright\arraybackslash}p{0.42\textwidth}rrrr@{}} \toprule \textbf{Step} & \textbf{\shortstack[l]{Key Modification /\\Investigated Factor}} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{Standalone Forward\\(ms)  }} & \textbf{\shortstack{Standalone FPS\\ }} \\ \midrule A0 & Baseline MobilePIXOR starting checkpoint & 41.40 & 42.27 & 11.05 & 90.57 \\ A1 & + ConvNeXt-style backbone (  DW, SiLU, Inverted Bottleneck) & 47.60 & 49.08 & \textbf{5.88} & \textbf{169.97} \\ A2 & + 8-channel RichBEV statistical representation & 47.37 & 48.95 & 6.06 & 165.02 \\ A3 & + 14-channel Strata-BEV14 (subcell offsets, height moments) & 50.14 & 52.00 & \underline{5.92} & \underline{169.01} \\ A4 & + Decoupled Grouped Header (Car vs.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S259 — after, tables/components.tex:5

Script: 128 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
Ped/Cyc, 16 channels) & 51.30 & 53.96 & 6.92 & 144.58 \\ A5 & + Expanded Grouped Header capacity (H32, 32 channels) & 60.22 & 61.11 & 6.70 & 149.15 \\ A6 & + Physics-Aware Ground-Truth Augmentation (PC-GTA) & 66.25 & 68.45 & 7.45 & 134.25 \\ A7 & + Expanded neck fusion layer ( , 32 channels) & 68.12 & 70.05 & 6.72 & 148.82 \\ A8 & + Bilinear Depthwise Scale-Gated Neck (SG-FPN) & 67.38 & 69.46 & 6.94 & 144.16 \\ A9 & + Oriented Geometric Alignment (OGA) and IQA supervision & \underline{70.58} & \underline{72.66} & 8.19 & 122.16 \\ A10 & + Parameter-Free Block-SimAM Energy Attention inside all blocks & \textbf{71.46} & \textbf{73.42} & 8.67 & 115.32 \\ \bottomrule \end{tabular*}\endgroup \par\vspace{2pt} {\footnotesize \textit{Note.} Step identifiers A0--A10 trace additions from baseline R01 to the final configuration.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S260 — after, tables/loss.tex:5

Script: 168 từ; 4 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{0.26\textwidth}ccrrrrr@{}} \toprule \textbf{Training Objective} & \textbf{\shortstack{IQA\\Head}} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Forward\\(ms)  }} & \textbf{\shortstack{Car AP \\(\%)  }} & \textbf{\shortstack{Pedestrian\\AP  (\%)  }} & \textbf{\shortstack{Cyclist\\AP  (\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} \\ \midrule Baseline Smooth-L1 & No  & \textbf{0.702} & \textbf{6.94} & 83.92 & 55.48 & 62.74 & 67.38 \\ OGA (MGIoU +  -Corner) & No  & \textbf{0.702} & \textbf{6.94} & 84.09 & 57.93 & 64.88 & 68.97 \\ Baseline Smooth-L1 & Yes & 0.721 & 8.19 & 84.13 & \underline{59.45} & \textbf{67.34} & 70.31 \\ \textbf{OGA (MGIoU +  -Corner)} & \textbf{Yes} & 0.721 & 8.19 & \textbf{84.87} & \textbf{59.63} & \underline{67.24} & \textbf{70.58} \\ \midrule Q-OGA (Quality-Coupled Focal Loss) & No & \textbf{0.702} & \textbf{6.94} & \underline{84.35} & 58.12 & 65.13 & 69.20 \\ \bottomrule \end{tabular*} \par\vspace{2pt} {\footnotesize \textit{Note.} All variants share identical MobilePixorNeXt backbone, Strata-BEV14 input, Grouped H32 heads, F32 SG-FPN neck, and PC-GTA augmentation.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S261 — after, sections/results.tex:123

Script: 18 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
With OGA, Moderate mAP rose from   to   (\textbf{+1.59\% mAP}), with gains for Car ( ), Pedestrian ( ), and Cyclist ( ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U148](rewrites.md#u148).

## S262 — after, tables/attention.tex:5

Script: 114 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{5pt} \begin{tabular}{@{}llrrrr@{}} \toprule \textbf{\shortstack[l]{SimAM Attention\\Placement}} & \textbf{\shortstack[l]{Neck Fusion\\Width}} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{Standalone\\Forward (ms)  }} & \textbf{\shortstack{Standalone\\FPS  }} \\ \midrule None (No attention) &   & \underline{70.58} & \underline{72.66} & \underline{8.19} & \underline{122.16} \\ Stage-2 ( ) &   & 69.44 & 71.49 & 7.30 & 136.91 \\ Stage-3 ( ) &   & 69.80 & 71.98 & \textbf{6.94} & \textbf{144.05} \\ Stage-4 ( ) &   & 69.36 & 71.34 & 7.44 & 134.43 \\ \textbf{Block-level ( )} & \textbf{ } & \textbf{71.46} & \textbf{73.42} & 8.67 & 115.32 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.} All configurations share OGA, the IQA quality head, Strata-BEV14 input, and PC-GTA augmentation.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S263 — after, sections/results.tex:141

Script: 17 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
It includes the total objective and six tasks: Classification, Geometry, IoU Quality, Subcell Offset, Size, and Yaw.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U153](rewrites.md#u153).

## S264 — after, sections/results.tex:153

Script: 65 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\setlength{\tabcolsep}{1.5pt} \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcrrrrr@{}} \toprule \textbf{Selection} & \textbf{Epoch} & \textbf{Car} & \textbf{Ped.} & \textbf{Cyc.} & \textbf{mAP } & \textbf{E2E} \\ \midrule \textbf{Best AP} & \textbf{130} & \textbf{86.90} & 60.75 & 66.73 & \textbf{71.46} & \textbf{60.96} \\ Best loss & 47 & 82.71 & \textbf{62.30} & \textbf{69.43} & \textbf{71.48} & 81.23 \\ \bottomrule \end{tabular*} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S265 — after, sections/results.tex:172

Script: 37 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\begin{figure}[t] \centering \paperfigure[width=\columnwidth,height=0.38\textheight,keepaspectratio]{qualitative_detections.pdf}{65mm}{Qualitative BEV detection visualizations on KITTI validation scenes.} ) on KITTI validation frames \texttt{007038}, \texttt{005571}, and \texttt{006761} (top to bottom).
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S266 — after, sections/declarations.tex:4

Script: 23 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/MobilePIXORNeXt.tex.txt).

```latex
\begin{acknowledgements} The authors acknowledge FPT University, Vietnam, for providing the computational infrastructure, GPU workstation clusters, and laboratory facilities used in this research.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U005](rewrites.md#u005).

## S267 — after, sections/declarations.tex:4

Script: 23 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__declarations.tex.txt).

```latex
\begin{acknowledgements} The authors acknowledge FPT University, Vietnam, for providing the computational infrastructure, GPU workstation clusters, and laboratory facilities used in this research.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U005](rewrites.md#u005).

## S268 — after, sections/declarations.tex:28

Script: 19 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__declarations.tex.txt).

```latex
\textbf{Nhan Trong Phan}: Formal analysis, loss ablation studies (OGA, IQA, T-SBUW), quantitative visualization, and writing the original draft.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U012](rewrites.md#u012).

## S269 — after, sections/declarations.tex:28

Script: 18 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__declarations.tex.txt).

```latex
\textbf{Hoang Ngoc Tran}: Research supervision, conceptual validation, project administration, critical revision, and final review of the manuscript.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U012](rewrites.md#u012).

## S270 — after, sections/experiments.tex:7

Script: 37 từ; 8 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__experiments.tex.txt).

```latex
We restricted perception to the Cartesian Region of Interest: \begin{equation} \begin{aligned} x &\in [0.0, 70.4]\,\text{m}, \\ y &\in [-40.0, 40.0]\,\text{m}, \\ z &\in [-2.5, 1.0]\,\text{m}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U014](rewrites.md#u014).

## S271 — after, sections/experiments.tex:18

Script: 43 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__experiments.tex.txt).

```latex
\end{itemize} Following autonomous driving literature, \textbf{Moderate macro mAP} ( ) was adopted as the primary benchmark ranking criterion: \begin{equation} \mathrm{mAP}_{\mathrm{M}} = \frac{1}{3} \sum_{c \in \{\mathrm{Car}, \mathrm{Ped}, \mathrm{Cyc}\}} \mathrm{AP}_{c, \mathrm{M}}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S272 — after, sections/experiments.tex:18

Script: 9 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__experiments.tex.txt).

```latex
E, M, and H denote Easy, Moderate, and Hard.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U023](rewrites.md#u023).

## S273 — after, sections/experiments.tex:51

Script: 34 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__experiments.tex.txt).

```latex
Final MobilePIXORNeXt timing and accuracy evaluation used an NVIDIA Tesla T4 GPU (16\,GB GDDR6 VRAM, Turing architecture), an x86\_64 host CPU, Linux OS, and PyTorch 2.11.0 with CUDA 13.0.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U027](rewrites.md#u027).

## S274 — after, sections/introduction.tex:5

Script: 20 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__introduction.tex.txt).

```latex
Vehicle perception systems must detect, orient, and classify nearby objects, including passenger vehicles, pedestrians, and cyclists, within strict latency limits.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U028](rewrites.md#u028).

## S275 — after, sections/introduction.tex:7

Script: 21 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__introduction.tex.txt).

```latex
They are faster than volumetric methods, although point gathering, dynamic memory allocation, and sparse scattering add runtime and memory transfer overhead.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U029](rewrites.md#u029).

## S276 — after, sections/introduction.tex:9

Script: 54 từ; 1 bộ đếm mệnh đề. Nhóm: `question-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__introduction.tex.txt).

```latex
Our research question was: \textit{Can an ultra-compact LiDAR detector operating on a parameter-free BEV tensor rival the localization accuracy of heavy volumetric and pillar architectures while achieving superior end-to-end throughput on commodity edge hardware?} We coupled this representation with a single-branch ConvNeXt-style backbone~  using large   depthwise kernels and parameter-free 3D spatial-energy attention (SimAM)~ .
```

Script gộp câu hỏi kết thúc bằng ?} với câu tiếp theo; kiểm tra riêng từng câu. [Đối chiếu U030](rewrites.md#u030).

## S277 — after, sections/introduction.tex:11

Script: 28 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__introduction.tex.txt).

```latex
The top-down \textbf{Scale-Gated Feature Pyramid Network (SG-FPN Neck,  )} balances semantic context and fine coordinates through bilinear upsampling, depthwise refinement, and learnable scale gates, avoiding deconvolution checkerboard artifacts.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U031](rewrites.md#u031).

## S278 — after, sections/methodology.tex:5

Script: 15 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
Decoupled grouped prediction towers output heatmaps, subcell offsets, log-sizes, doubled yaw angles, and localization quality.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S279 — after, sections/methodology.tex:18

Script: 32 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
Perception is restricted to a Cartesian Region of Interest (ROI): \begin{equation} \begin{aligned} x_i &\in [x_{\min}, x_{\max}], \\ y_i &\in [y_{\min}, y_{\max}], \\  z_i &\in [z_{\min}, z_{\max}].
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U038](rewrites.md#u038).

## S280 — after, sections/methodology.tex:26

Script: 14 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\end{equation} We used longitudinal bounds  \,m, lateral bounds  \,m, and vertical bounds  \,m.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U039](rewrites.md#u039).

## S281 — after, sections/methodology.tex:33

Script: 25 từ; 8 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
For each detected object instance  , the target state is parameterized by the 7-tuple: \begin{equation} \mathbf{b}_k = (c_k, x_k^*, y_k^*, l_k^*, w_k^*, \psi_k^*, s_k),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S282 — after, sections/methodology.tex:45

Script: 65 từ; 7 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
Channels 0–3 capture stratified height-band densities, Channels 4–7 encode vertical elevation statistics (peak, mean, span, deviation), Channels 8–9 capture material surface reflectance, Channels 10–11 represent global density and occupancy, and Channels 12–13 encode continuous subcell horizontal offsets to eliminate grid discretization error.} \captionof{figure}{Fourteen deterministic feature channels in the Strata-BEV14 representation for a KITTI driving scene (  grid at   resolution).
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S283 — after, sections/methodology.tex:69

Script: 52 từ; 9 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\textit{Vertical Spatial Moments ( -- )}: To encode obstacle geometry without explicit voxelization, we extract peak elevation  , mean elevation  , vertical span  , and the sample standard deviation: \begin{equation} F_7 = \begin{cases} \sqrt{\max\left(0, \frac{1}{N} \sum_{i=1}^N \zeta_i^2 - F_5^2\right)}, & \text{if } N > 1, \\ 0, & \text{otherwise}.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S284 — after, sections/methodology.tex:78

Script: 39 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\textit{Reflectance Statistics ( -- )}: Calibrated surface returns capture material discrimination (e.g., distinguishing retro-reflective license plates, pedestrian clothing, or road signs): \begin{equation} F_8 = \max_{i=1,\dots,N} \iota_i, \qquad F_9 = \frac{1}{N} \sum_{i=1}^N \iota_i.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S285 — after, sections/methodology.tex:84

Script: 80 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
For  \,m cells, channels   and   retain continuous horizontal subcell centroid displacements to reduce quantization error: \begin{align} x_u^{\mathrm{cell}} &= x_{\min} + (u + 0.5)\Delta_x, \nonumber\\ y_v^{\mathrm{cell}} &= y_{\min} + (v + 0.5)\Delta_y, \nonumber\\ F_{12} &= \operatorname{clip}\left(\frac{\sum_{i=1}^N (x_i - x_u^{\mathrm{cell}})}{N\Delta_x}, -0.5, 0.5\right), \nonumber\\ F_{13} &= \operatorname{clip}\left(\frac{\sum_{i=1}^N (y_i - y_v^{\mathrm{cell}})}{N\Delta_y}, -0.5, 0.5\right).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U049](rewrites.md#u049).

## S286 — after, sections/methodology.tex:130

Script: 83 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
The mechanisms can be enabled independently; the settings used for the reported final run are given in Section~ : \begin{enumerate} \item \textit{Adaptive Ground Plane Fitting}: To prevent floating or sunken objects, local road elevation is estimated within a range-dependent search radius: \begin{align} R(d) &= \operatorname{clip}(2.5 + 0.03\max(0, d - 15), 2.5, 4.0)\,\mathrm{m},\nonumber\\[0.8em] \hat{z}_{\mathrm{ground}} &= \operatorname{percentile}(\{z_j \mid \mathbf{p}_j \in \mathcal{N}(x, y)\}, 5),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S287 — after, sections/methodology.tex:146

Script: 52 từ; 1 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
It masks points behind the exit face ( ) whose radial distance satisfies  \,m: \begin{equation} \begin{aligned} \mathcal{P}_{\mathrm{augmented}} &= \mathcal{P}_{\mathrm{target}} \setminus \{\mathbf{p} \in \mathcal{P}_{\mathrm{target}} \mid \\ &\qquad \operatorname{Mask}_{\mathrm{shadow}}(\mathbf{p}) = \mathrm{True}\} \cup \mathcal{P}_{\mathrm{object}}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U060](rewrites.md#u060).

## S288 — after, sections/methodology.tex:158

Script: 50 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\item \textit{Range-Adaptive Point and Intensity Attenuation}: If an object captured at source distance   is placed at target distance  , its point cloud is randomly subsampled according to the inverse-square divergence law: \begin{equation} P_{\mathrm{keep}} = \min\left(1, \left(\frac{r_{\mathrm{src}}}{r_{\mathrm{tgt}}}\right)^2\right),
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S289 — after, sections/methodology.tex:162

Script: 47 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\end{equation} and the optional radiometric adjustment attenuates reflectance intensity according to: \[ \begin{aligned} \eta_{\mathrm{int}} &= \left(\frac{r_{\mathrm{src}}}{r_{\mathrm{tgt}}}\right)^{\max(0, 2 - \gamma)}, \qquad \gamma = 1.7, \\ I_{\mathrm{tgt}} &= \operatorname{clip}(I_{\mathrm{src}}\eta_{\mathrm{int}}, 0, 1).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S290 — after, sections/methodology.tex:214

Script: 29 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
(Top) Complete feedforward path from input X to output Y through the 7x7 depthwise convolution, inverted bottleneck expansion (e=2.5), SimAM modulation, LayerScale scaling, and identity residual addition.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S291 — after, sections/methodology.tex:214

Script: 33 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
(Bottom) Zoom-in structure of the parameter-free SimAM module, detailing spatial moment estimation, inverse-energy importance calculation, sigmoid saliency mapping, and element-wise feature recalibration.} \captionof{figure}{The \texttt{MobilePixorNeXt-S} residual block and parameter-free Block-SimAM attention.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S292 — after, sections/methodology.tex:221

Script: 57 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
Given input tensor  , each block applies an inverted bottleneck with expansion ratio  : \begin{align} \mathbf{V} &= \operatorname{BN}(\operatorname{DWConv}_{7\times7}(\mathbf{Z})),\nonumber\\ \mathbf{R} &= \operatorname{BN}\left(W_{\mathrm{red}} \operatorname{SiLU}(W_{\mathrm{exp}}\mathbf{V})\right),\nonumber\\ \mathbf{Z}' &= \mathbf{Z} + \boldsymbol{\gamma}_{\mathrm{blk}} \odot \mathcal{S}(\mathbf{R}).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S293 — after, sections/methodology.tex:234

Script: 31 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\paragraph{Design and Placement.} As illustrated in Figure~ , SimAM is embedded inside each \texttt{MobilePixorNeXt-\allowbreak S} block across all three stages (C2, C3, C4), for nine attention operators in total.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S294 — after, sections/methodology.tex:237

Script: 44 từ; 9 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
For each channel slice with   spatial locations, spatial mean   and sample variance   are computed: \begin{equation} \begin{aligned} \mu_c &= \frac{1}{M}\sum_{u, v} R_{c, u, v}, \\ \nu_c &= \frac{1}{\max(M-1, 1)}\sum_{u, v} (R_{c, u, v} - \mu_c)^2.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S295 — after, sections/methodology.tex:244

Script: 56 từ; 2 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\end{equation} Following the energy-minimization formulation of SimAM~ , the minimized energy   and its inverse-energy importance score   for neuron   are: \begin{equation} \begin{aligned} e_t^\star &= \frac{4(\nu_c + \lambda_{\mathrm{sim}})}{(t-\mu_c)^2 + 2(\nu_c + \lambda_{\mathrm{sim}})}, \\ s_t &= \frac{1}{e_t^\star} = \frac{(t - \mu_c)^2}{4(\nu_c + \lambda_{\mathrm{sim}})} + \frac{1}{2}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S296 — after, sections/methodology.tex:282

Script: 27 từ; 8 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\paragraph{Mathematical Formulation.} As shown in Figure~ , lateral features from Stages C4 ( \,ch), C3 ( \,ch), and C2 ( \,ch) are projected via   convolutions into matching lateral dimensions  ,  , and  .
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết.

## S297 — after, sections/methodology.tex:296

Script: 76 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\par\vspace{-0.5\baselineskip} \noindent \begin{minipage}{\columnwidth} \centering \paperfigure{geometry_output_2x2.pdf}{75mm}{Geometric parameterization, orientation encoding, and quality-aware score calibration across prediction heads: (a) Continuous subcell coordinate offsets   from anchor point   to object center  ; (b) Log-transformed bounding box dimension targets  ; (c) Doubled-angle trigonometric yaw representation   on the unit circle resolving  -heading ambiguity; and (d) IoU-quality-aware composite score ranking   calibrated for post-processing peak selection and rotated NMS.} \captionof{figure}{Prediction-head geometry, orientation, and quality-aware scoring.
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S298 — after, sections/methodology.tex:306

Script: 18 từ; 11 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
Cars are rectangular ( \,m,  \,m), while Pedestrians ( \,m,  \,m) and Cyclists ( \,m,  \,m) are smaller and more deformable.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U087](rewrites.md#u087).

## S299 — after, sections/methodology.tex:319

Script: 30 từ; 5 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
For output cell  , the ground-truth geometric targets are formulated as: \begin{align} \mathbf{o}^* &=  ,\nonumber\\ \mathbf{d}^* &= (\log w^*, \log l^*), \qquad \mathbf{a}^* = (\cos 2\psi^*, \sin 2\psi^*).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S300 — after, sections/methodology.tex:333

Script: 62 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\textit{Multi-Axis Projection GIoU ( )}: Projected bounding box overlap is evaluated following Marginalized Generalized IoU~  across four unit normal projection axes defining the prediction and ground-truth rectangles: \begin{equation} \mathbf{n}_k \in \left\{ \mathbf{n}_{\mathrm{len}}^{\mathrm{pred}}, \mathbf{n}_{\mathrm{wid}}^{\mathrm{pred}}, \mathbf{n}_{\mathrm{len}}^{\mathrm{gt}}, \mathbf{n}_{\mathrm{wid}}^{\mathrm{gt}} \right\}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U096](rewrites.md#u096).

## S301 — after, sections/methodology.tex:339

Script: 52 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\end{equation} For each axis  , 1D intervals of intersection length  , union length  , and smallest enclosing hull   are determined, yielding 1D GIoU~ : \begin{equation} \begin{aligned} g_k &= \frac{I_k}{U_k} - \frac{E_k - U_k}{E_k}, \\ \bar{g} &= \frac{1}{4}\sum_{k=1}^4 g_k, \qquad \ell_{\mathrm{proj}} = \frac{1 - \bar{g}}{2}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S302 — after, sections/methodology.tex:348

Script: 78 từ; 3 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
Scale-Normalized  -Symmetric Corner Distance ( )}:\par}\noindent To ensure accurate vertex localization under doubled-angle symmetry, we evaluate the minimum L1 distance across direct and  -permuted corner correspondences: \begin{equation} \ell_{\mathrm{corner}} = \frac{\min\left\{ \begin{array}{l} \displaystyle\sum_{m=1}^4 \|\widehat{\mathbf{c}}_m - \mathbf{c}_m^*\|_1, \\ \displaystyle\sum_{m=1}^4 \|\widehat{\mathbf{c}}_m - \mathbf{c}_{5-m}^*\|_1 \end{array} \right.}{4\max\left(\sqrt{(w^*)^2 + (l^*)^2}, 10^{-6}\right)}.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S303 — after, sections/methodology.tex:362

Script: 38 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
It is supervised via Binary Cross-Entropy (BCE) using the detached MGIoU similarity as the ground-truth target: \begin{equation} q^* = \operatorname{stopgrad}(\operatorname{clip}(\bar{g}, 0, 1)), \qquad \mathcal{L}_{\mathrm{qual}} = \operatorname{BCE}(\hat{q}, q^*).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán.

## S304 — after, sections/methodology.tex:383

Script: 74 từ; 4 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\textit{Bounded Softmax Weights}: Learnable logit parameters   are soft-clamped via hyperbolic tangent with bound  , and converted to positive normalized weights via temperature-controlled softmax ( ): \begin{equation} \begin{aligned} \bar{a}_k &= b_{\mathrm{uw}} \tanh(a_k / b_{\mathrm{uw}}), \\ Z_{\mathrm{uw}} &= \sum_{j \in \mathcal{T}} \exp(\bar{a}_j / \tau_{\mathrm{uw}}), \\ \omega_k &= J \cdot \frac{\exp(\bar{a}_k / \tau_{\mathrm{uw}})}{Z_{\mathrm{uw}}}, \\ \sum_{k \in \mathcal{T}} \omega_k &= J.
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U104](rewrites.md#u104).

## S305 — after, sections/methodology.tex:411

Script: 61 từ; 4 bộ đếm mệnh đề. Nhóm: `algorithm`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\vspace{2pt} {\small \noindent\textbf{Require:} Raw LiDAR point cloud  , ROI bounds  \par \noindent\textbf{Require:} Score threshold  , Rotated NMS threshold  , quality exponent  \par \noindent\textbf{Ensure:} Final decoded 3D oriented bounding boxes  \par \begin{enumerate} \setlength{\topsep}{2pt} \setlength{\itemsep}{0pt} \setlength{\parsep}{0pt} \renewcommand{\labelenumi}{\textbf{\arabic{enumi}:}} \item Filter raw points   within inward margin  \,m of spatial ROI.
```

Giữ nguyên thuật toán; danh sách đầu vào/đầu ra không phải câu văn cần tách.

## S306 — after, sections/methodology.tex:411

Script: 20 từ; 5 bộ đếm mệnh đề. Nhóm: `algorithm`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\item Evaluate Grouped Prediction Towers for Car ( ) and Pedestrian/Cyclist ( ), yielding classification maps  , subcell offsets  , log-sizes  , doubled yaws  , and quality  .
```

Giữ nguyên thuật toán; danh sách đầu vào/đầu ra không phải câu văn cần tách.

## S307 — after, sections/methodology.tex:441

Script: 75 từ; 6 bộ đếm mệnh đề. Nhóm: `math-boundary`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
For each candidate cell   whose classification probability   is a   local spatial maximum, bounding boxes are decoded in metric coordinates as: \begin{align} \hat{x} &= x_{\min} + u s_{\mathrm{out}}\Delta_x + \hat{o}_x, \nonumber\\ \hat{y} &= y_{\min} + v s_{\mathrm{out}}\Delta_y + \hat{o}_y, \nonumber\\ \hat{w} &= \exp(\hat{d}_1), \qquad \hat{l} = \exp(\hat{d}_2), \nonumber\\ \hat{\psi} &= \frac{1}{2}\operatorname{atan2}(\hat{a}_2, \hat{a}_1).
```

Script gộp văn xuôi với công thức; chỉ sửa phần văn xuôi nếu cần, giữ nguyên toàn bộ toán. [Đối chiếu U106](rewrites.md#u106).

## S308 — after, sections/methodology.tex:458

Script: 26 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\paragraph{Parameter Distribution Breakdown.} MobilePIXORNeXt contains \textbf{753,569 trainable parameters} ( \,M), with an FP32 checkpoint size of  \,MB: \begin{itemize} \item \textbf{Backbone Body}:   parameters ( ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U110](rewrites.md#u110).

## S309 — after, sections/methodology.tex:458

Script: 20 từ; 6 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__methodology.tex.txt).

```latex
\end{itemize} The complete pipeline achieved  \,FPS ( \,ms), while standalone network throughput reached  \,FPS ( \,ms) on NVIDIA Tesla T4 GPUs.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U112](rewrites.md#u112).

## S310 — after, sections/related_work.tex:6

Script: 24 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__related_work.tex.txt).

```latex
Pillar encoders~  learn geometric features but require point gathering, dynamic memory allocation, and sparse scattering, which add latency and energy costs on edge accelerators.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U114](rewrites.md#u114).

## S311 — after, sections/related_work.tex:11

Script: 13 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__related_work.tex.txt).

```latex
PointRCNN~ , Part- ~ , and PV-RCNN~  developed two-stage point processing, part-aware aggregation, and voxel-set abstractions.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U115](rewrites.md#u115).

## S312 — after, sections/results.tex:37

Script: 18 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__results.tex.txt).

```latex
Full-system latency includes point cloud loading, tensorization, Host-to-Device (H2D) transfer, network execution, coordinate decoding, and Non-Maximum Suppression (NMS).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U127](rewrites.md#u127).

## S313 — after, sections/results.tex:42

Script: 8 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__results.tex.txt).

```latex
MobilePIXORNeXt ran faster than PointPillars ( ,  ), MobilePIXOR ( ,  ), and SECOND ( ,  ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U128](rewrites.md#u128).

## S314 — after, sections/results.tex:118

Script: 18 từ; 4 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__results.tex.txt).

```latex
With OGA, Moderate mAP rose from   to   (\textbf{+1.59\% mAP}), with gains for Car ( ), Pedestrian ( ), and Cyclist ( ).
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U148](rewrites.md#u148).

## S315 — after, sections/results.tex:135

Script: 17 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/sections__results.tex.txt).

```latex
It includes the total objective and six tasks: Classification, Geometry, IoU Quality, Subcell Offset, Size, and Yaw.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U153](rewrites.md#u153).

## S316 — after, sections/results.tex:147

Script: 65 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/sections__results.tex.txt).

```latex
\setlength{\tabcolsep}{1.5pt} \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcrrrrr@{}} \toprule \textbf{Selection} & \textbf{Epoch} & \textbf{Car} & \textbf{Ped.} & \textbf{Cyc.} & \textbf{mAP } & \textbf{E2E} \\ \midrule \textbf{Best AP} & \textbf{130} & \textbf{86.90} & 60.75 & 66.73 & \textbf{71.46} & \textbf{60.96} \\ Best loss & 47 & 82.71 & \textbf{62.30} & \textbf{69.43} & \textbf{71.48} & 81.23 \\ \bottomrule \end{tabular*} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S317 — after, sections/results.tex:166

Script: 37 từ; 4 bộ đếm mệnh đề. Nhóm: `protected-figure-macro`. [Báo cáo gốc](after/analyze_sentences/sections__results.tex.txt).

```latex
\begin{figure}[t] \centering \paperfigure[width=\columnwidth,height=0.38\textheight,keepaspectratio]{qualitative_detections.pdf}{65mm}{Qualitative BEV detection visualizations on KITTI validation scenes.} ) on KITTI validation frames \texttt{007038}, \texttt{005571}, and \texttt{006761} (top to bottom).
```

Có nội dung fallback của macro hình hoặc ranh giới caption; giữ nguyên macro và đối chiếu caption hiển thị.

## S318 — after, tables/attention.tex:5

Script: 114 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/tables__attention.tex.txt).

```latex
\setlength{\tabcolsep}{5pt} \begin{tabular}{@{}llrrrr@{}} \toprule \textbf{\shortstack[l]{SimAM Attention\\Placement}} & \textbf{\shortstack[l]{Neck Fusion\\Width}} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{Standalone\\Forward (ms)  }} & \textbf{\shortstack{Standalone\\FPS  }} \\ \midrule None (No attention) &   & \underline{70.58} & \underline{72.66} & \underline{8.19} & \underline{122.16} \\ Stage-2 ( ) &   & 69.44 & 71.49 & 7.30 & 136.91 \\ Stage-3 ( ) &   & 69.80 & 71.98 & \textbf{6.94} & \textbf{144.05} \\ Stage-4 ( ) &   & 69.36 & 71.34 & 7.44 & 134.43 \\ \textbf{Block-level ( )} & \textbf{ } & \textbf{71.46} & \textbf{73.42} & 8.67 & 115.32 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.} All configurations share OGA, the IQA quality head, Strata-BEV14 input, and PC-GTA augmentation.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S319 — after, tables/comparison.tex:5

Script: 176 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/tables__comparison.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular}{@{}lrrrrrr@{}} \toprule \textbf{Method} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Car AP \\(\%)  }} & \textbf{\shortstack{Pedestrian\\AP  (\%)  }} & \textbf{\shortstack{Cyclist\\AP  (\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} \\ \midrule PIXOR (Car only)~  & 2.135 & 72.02 & -- & -- & -- & -- \\ MobilePIXOR~  & \textbf{0.598} & 75.52 & 28.16 & 20.53 & 41.40 & 42.27 \\ PointPillars~  & 4.835 & \underline{88.02} & 56.01 & 66.34 & 70.13 & 72.25 \\ SECOND~  & 5.326 & \textbf{88.56} & \underline{56.56} & \textbf{71.16} & \textbf{72.09} & \textbf{73.79} \\ \textbf{MobilePIXORNeXt (Ours)} & \underline{0.754} & 86.90 & \textbf{60.75} & \underline{66.73} & \underline{71.46} & \underline{73.42} \\ \midrule \multicolumn{7}{@{}l}{\textit{Heavyweight 3D Anchor-Based Reference Detectors}} \\ PointRCNN~  & 4.039 & 88.42 & 57.84 & 73.87 & 73.38 & 75.31 \\ Part-A -Anchor~  & 63.815 & 90.05 & 64.20 & 74.60 & 76.28 & 78.04 \\ PV-RCNN~  & 13.116 & 90.29 & 58.48 & 74.54 & 74.44 & 76.49 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.}   is the unweighted mean across the three target classes at Moderate difficulty.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S320 — after, tables/components.tex:5

Script: 293 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/tables__components.tex.txt).

```latex
\textbf{(a) Architectural configurations and component progression}\par\smallskip \begingroup\footnotesize\setlength{\tabcolsep}{4pt} \begin{tabular}{@{}lllllll@{}} \toprule \textbf{Step} & \textbf{Backbone} & \shortstack[l]{\textbf{BEV}\\ \textbf{Input}} & \shortstack[l]{\textbf{Head}\\ \textbf{Architecture}} & \shortstack[l]{\textbf{GT}\\ \textbf{Sampling}} & \shortstack[l]{\textbf{Fusion /}\\ \textbf{Neck}} & \shortstack[l]{\textbf{Supervision /}\\ \textbf{Attention}} \\ \midrule A0 & MobilePIXOR & Binary35  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A1 & MobilePixorNeXt & Binary35  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A2 & MobilePixorNeXt & Strata-BEV8  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A3 & MobilePixorNeXt & Strata-BEV14 & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A4 & MobilePixorNeXt & Strata-BEV14 & Grouped H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A5 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\ A6 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F24 / SumFPN & Baseline Smooth-L1 \\ A7 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SumFPN & Baseline Smooth-L1 \\ A8 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & Baseline Smooth-L1 \\ A9 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & OGA + IQA Head \\ A10 & MobilePixorNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & OGA + IQA + Block-SimAM \\ \bottomrule \end{tabular}\endgroup \par\medskip \textbf{(b) Quantitative detection accuracy and standalone model inference speed}\par\smallskip \begingroup\footnotesize\setlength{\tabcolsep}{3pt} \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l>{\raggedright\arraybackslash}p{0.42\textwidth}rrrr@{}} \toprule \textbf{Step} & \textbf{\shortstack[l]{Key Modification /\\Investigated Factor}} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} & \textbf{\shortstack{Standalone Forward\\(ms)  }} & \textbf{\shortstack{Standalone FPS\\ }} \\ \midrule A0 & Baseline MobilePIXOR starting checkpoint & 41.40 & 42.27 & 11.05 & 90.57 \\ A1 & + ConvNeXt-style backbone (  DW, SiLU, Inverted Bottleneck) & 47.60 & 49.08 & \textbf{5.88} & \textbf{169.97} \\ A2 & + 8-channel RichBEV statistical representation & 47.37 & 48.95 & 6.06 & 165.02 \\ A3 & + 14-channel Strata-BEV14 (subcell offsets, height moments) & 50.14 & 52.00 & \underline{5.92} & \underline{169.01} \\ A4 & + Decoupled Grouped Header (Car vs.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S321 — after, tables/components.tex:5

Script: 128 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/tables__components.tex.txt).

```latex
Ped/Cyc, 16 channels) & 51.30 & 53.96 & 6.92 & 144.58 \\ A5 & + Expanded Grouped Header capacity (H32, 32 channels) & 60.22 & 61.11 & 6.70 & 149.15 \\ A6 & + Physics-Aware Ground-Truth Augmentation (PC-GTA) & 66.25 & 68.45 & 7.45 & 134.25 \\ A7 & + Expanded neck fusion layer ( , 32 channels) & 68.12 & 70.05 & 6.72 & 148.82 \\ A8 & + Bilinear Depthwise Scale-Gated Neck (SG-FPN) & 67.38 & 69.46 & 6.94 & 144.16 \\ A9 & + Oriented Geometric Alignment (OGA) and IQA supervision & \underline{70.58} & \underline{72.66} & 8.19 & 122.16 \\ A10 & + Parameter-Free Block-SimAM Energy Attention inside all blocks & \textbf{71.46} & \textbf{73.42} & 8.67 & 115.32 \\ \bottomrule \end{tabular*}\endgroup \par\vspace{2pt} {\footnotesize \textit{Note.} Step identifiers A0--A10 trace additions from baseline R01 to the final configuration.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S322 — after, tables/distance_our_vs_mbpixor.tex:5

Script: 95 từ; 3 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/tables__distance_our_vs_mbpixor.tex.txt).

```latex
\setlength{\tabcolsep}{1.5pt} \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrr@{}} \toprule \textbf{Category} & \textbf{Model} & \textbf{\shortstack{0--30\,m}} & \textbf{\shortstack{30--50\,m}} & \textbf{\shortstack{50--70.4\,m}} \\ \midrule Pedestrian & MobilePIXOR~  & 30.45 & 3.75 & \textbf{0.99} \\ & \textbf{Ours} & \textbf{65.23} & \textbf{11.49} & 0.80 \\ & \textit{Diff.} & \textit{+34.78} & \textit{+7.74} & \textit{-0.19} \\ \midrule Cyclist & MobilePIXOR~  & 25.55 & 12.61 & 0.56 \\ & \textbf{Ours} & \textbf{76.32} & \textbf{44.30} & \textbf{14.94} \\ & \textit{Diff.} & \textit{+50.77} & \textit{+31.69} & \textit{+14.38} \\ \bottomrule \end{tabular*} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S323 — after, tables/efficiency.tex:5

Script: 229 từ; 0 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/tables__efficiency.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular}{@{}lrrrrrrrr@{}} \toprule \textbf{Method} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Model\\(ms)  }} & \textbf{\shortstack{Model\\FPS  }} & \textbf{\shortstack{E2E\\(ms)  }} & \textbf{\shortstack{E2E\\FPS  }} & \textbf{\shortstack{P50\\(ms)  }} & \textbf{\shortstack{P95\\(ms)  }} & \textbf{\shortstack{P99\\(ms)  }} \\ \midrule PIXOR~  & 2.135 & 70.79 & 14.13 & 144.00 & 6.94 & 142.29 & 159.38 & 175.43 \\ MobilePIXOR (L4)~  & \textbf{0.598} & \textbf{11.05} & \textbf{90.50} & 85.92 & 11.64 & 84.11 & 112.52 & 124.94 \\ PointPillars~  & 4.835 & 32.13 & 31.12 & \underline{63.84} & \underline{15.66} & \underline{62.10} & \textbf{75.88} & \textbf{92.73} \\ SECOND~  & 5.326 & 62.64 & 15.96 & 93.70 & 10.67 & 92.02 & 107.42 & 126.35 \\ \textbf{MobilePIXORNeXt (Ours)} & \underline{0.754} & \underline{17.88} & \underline{55.94} & \textbf{60.96} & \textbf{16.41} & \textbf{58.28} & \underline{80.85} & \underline{95.91} \\ \midrule \multicolumn{9}{@{}l}{\textit{Heavyweight 3D Anchor-Based Reference Detectors}} \\ PointRCNN~  & 4.039 & 169.36 & 5.90 & 194.50 & 5.14 & 191.24 & 213.48 & 235.10 \\ Part-A -Anchor~  & 63.815 & 102.57 & 9.75 & 133.75 & 7.48 & 131.50 & 150.49 & 172.80 \\ PV-RCNN~  & 13.116 & 160.79 & 6.22 & 188.60 & 5.30 & 186.40 & 202.62 & 224.15 \\ \bottomrule \end{tabular} \par\vspace{2pt} {\footnotesize \textit{Note.} Model latency measures network execution in the evaluation pipeline on Tesla T4.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S324 — after, tables/efficiency.tex:5

Script: 19 từ; 5 bộ đếm mệnh đề. Nhóm: `comma-marker-proxy`. [Báo cáo gốc](after/analyze_sentences/tables__efficiency.tex.txt).

```latex
End-to-end (E2E) latency includes LiDAR loading, statistical rasterization, Host-to-Device (H2D) transfer, network execution, coordinate decoding, and class-wise rotated NMS.
```

Dấu phẩy và từ nối là bộ đếm gần đúng, không phải số mệnh đề ngữ pháp. Giữ danh sách kỹ thuật rõ ràng; sửa câu ghép cần thiết trong đơn vị liên kết. [Đối chiếu U175](rewrites.md#u175).

## S325 — after, tables/loss.tex:5

Script: 168 từ; 4 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/tables__loss.tex.txt).

```latex
\setlength{\tabcolsep}{3pt} \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{0.26\textwidth}ccrrrrr@{}} \toprule \textbf{Training Objective} & \textbf{\shortstack{IQA\\Head}} & \textbf{\shortstack{Params\\(M)  }} & \textbf{\shortstack{Forward\\(ms)  }} & \textbf{\shortstack{Car AP \\(\%)  }} & \textbf{\shortstack{Pedestrian\\AP  (\%)  }} & \textbf{\shortstack{Cyclist\\AP  (\%)  }} & \textbf{\shortstack{mAP \\(\%)  }} \\ \midrule Baseline Smooth-L1 & No  & \textbf{0.702} & \textbf{6.94} & 83.92 & 55.48 & 62.74 & 67.38 \\ OGA (MGIoU +  -Corner) & No  & \textbf{0.702} & \textbf{6.94} & 84.09 & 57.93 & 64.88 & 68.97 \\ Baseline Smooth-L1 & Yes & 0.721 & 8.19 & 84.13 & \underline{59.45} & \textbf{67.34} & 70.31 \\ \textbf{OGA (MGIoU +  -Corner)} & \textbf{Yes} & 0.721 & 8.19 & \textbf{84.87} & \textbf{59.63} & \underline{67.24} & \textbf{70.58} \\ \midrule Q-OGA (Quality-Coupled Focal Loss) & No & \textbf{0.702} & \textbf{6.94} & \underline{84.35} & 58.12 & 65.13 & 69.20 \\ \bottomrule \end{tabular*} \par\vspace{2pt} {\footnotesize \textit{Note.} All variants share identical MobilePixorNeXt backbone, Strata-BEV14 input, Grouped H32 heads, F32 SG-FPN neck, and PC-GTA augmentation.
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.

## S326 — after, tables/setup.tex:5

Script: 171 từ; 12 bộ đếm mệnh đề. Nhóm: `table-syntax`. [Báo cáo gốc](after/analyze_sentences/tables__setup.tex.txt).

```latex
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{.38\columnwidth}>{\raggedright\arraybackslash}p{.56\columnwidth}@{}} \toprule \textbf{Configuration Element} & \textbf{Specification / Value} \\ \midrule LiDAR Input Representation & Strata-BEV14 (14 channels; NumPy/Numba CPU) \\ Spatial ROI bounds ( ) &   \\ Grid resolution ( ) &   \\ Input / Output Grid Dimensions &   /   ( ) \\ Detector Parameters & 753,569 trainable ( ); 9.34\,MB FP32 \\ Backbone Stages ( ) & Widths: (48, 96, 128); Depths: (3, 4, 2) \\ Inverted Bottleneck Expansion &  ;   Depthwise Convolution \\ Attention Mechanism & Parameter-free Block-SimAM (9 residual blocks) \\ Multiscale Neck & Scale-Gated FPN (SG-FPN,   fusion width) \\ Prediction Towers & GroupedHeader: Car ( ), Ped/Cyc ( ) \\ Loss Formulation & OGA (MGIoU +  -Corner) + IQA + T-SBUW \\ Data Augmentation & Physics-Consistent GT (PC-GTA) + Global \\ Optimizer / Base Learning Rate & AdamW (fused) /   \\ Minimum LR / Standard Schedule &   / 50 epochs: 4 warmup + 46 cosine \\ Final Reported Checkpoint & Extended run: 160 epochs, 10 warmup; selected at epoch 130 \\ Weight Decay &   (2D+ tensors only; biases/norm: 0.00) \\ Gradient Clipping Norm & Maximum   norm = 10.00 \\ Batch Size (Train / Val) & 16 / 32 \\ Numerical Precision & Automatic Mixed Precision (Bfloat16 / FP32) \\ Hardware Environment & NVIDIA Tesla T4 GPU (16\,GB GDDR6, 98.44\,MB peak) \\ \bottomrule \end{tabular} \end{table}
```

Không tách bảng thành câu; giữ nguyên các hàng dữ liệu.
