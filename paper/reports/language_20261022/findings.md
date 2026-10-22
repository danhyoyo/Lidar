# Tất cả câu cần đối chiếu trong văn xuôi

Danh sách này dùng nguyên văn nguồn, giữ công thức và tham chiếu. Các cảnh báo `comma-marker-proxy>3` là tín hiệu đọc lại, không tự động được coi là lỗi ngữ pháp. Câu được sửa trong cùng đoạn có thể giữ một phần nguyên văn khi đã đúng. Các vị trí từ ngữ nằm ngoài đơn vị văn xuôi (ví dụ vai trò tác giả, tiêu đề, macro) được liệt kê riêng ở [lexical_index.md](lexical_index.md).

Ngưỡng mặc định: hơn 50 từ hoặc bộ đếm dấu phẩy/từ nối hơn 3. Không có ngưỡng xác suất tác giả AI.

## M001 — U001, [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1), câu 1

Phân loại: vocabulary-context.
Từ hiển thị: 29; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: severe, critical.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Three-dimensional object detection from Light Detection and Ranging (LiDAR) point clouds is critical for autonomous navigation; however, conventional volumetric and pillar-based architectures impose severe computational challenges on edge accelerators.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U001](rewrites.md#u001).

## M002 — U001, [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1), câu 2

Phân loại: vocabulary-context, tense-operation-or-finding.
Từ hiển thị: 15; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: ultra-compact.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: We propose.

```latex
We propose MobilePIXORNeXt, an ultra-compact, proposal-free Bird's-Eye View (BEV) detector operating under 753,569 parameters.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U001](rewrites.md#u001).

## M003 — U001, [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1), câu 3

Phân loại: comma-marker-proxy>3, tense-operation-or-finding.
Từ hiển thị: 48; bộ đếm dấu phẩy/từ nối: 6.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: The framework replaces; extracts 14.

```latex
The framework replaces learned point encoders with Strata-BEV14, a parameter-free deterministic NumPy/Numba CPU rasterizer that extracts 14 statistical feature channels---capturing stratified height distributions, elevation moments, surface reflectance, and continuous subcell centroid displacement offsets---with a measured input preprocessing time of 20.94 milliseconds, including point-cloud loading and rasterization.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U001](rewrites.md#u001).

## M004 — U001, [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1), câu 4

Phân loại: tense-operation-or-finding.
Từ hiển thị: 29; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Spatial features are extracted.

```latex
Spatial features are extracted by a single-branch ConvNeXt-style backbone with embedded parameter-free 3D spatial-energy attention (Block-SimAM), fused through a Scale-Gated Feature Pyramid Network, and predicted by decoupled grouped heads.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U001](rewrites.md#u001).

## M005 — U001, [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1), câu 5

Phân loại: tense-operation-or-finding.
Từ hiển thị: 19; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Training is supervised.

```latex
Training is supervised by Physics-Consistent Ground-Truth Augmentation and Oriented Geometric Alignment, dynamically balanced via 12-parameter Temperature-Softmax Bounded Uncertainty Weighting.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U001](rewrites.md#u001).

## M006 — U001, [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1), câu 6

Phân loại: tense-operation-or-finding.
Từ hiển thị: 29; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: achieves.

```latex
On the KITTI validation set using BEV AP $R_{40}$, MobilePIXORNeXt achieves 71.46\% Moderate mean Average Precision (mAP) and 73.42\% overall 9-cell mAP across Car, Pedestrian, and Cyclist categories.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U001](rewrites.md#u001).

## M007 — U001, [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1), câu 7

Phân loại: tense-operation-or-finding.
Từ hiển thị: 31; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: It outperforms; matches SECOND.

```latex
It outperforms PointPillars (70.13\% mAP) with an 84.4\% parameter reduction and matches SECOND (72.09\% mAP) with 85.8\% fewer parameters, while reaching 60.75\% Moderate AP on pedestrians.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U001](rewrites.md#u001).

## M008 — U001, [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1), câu 8

Phân loại: tense-operation-or-finding.
Từ hiển thị: 28; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: The complete pipeline executes.

```latex
The complete pipeline executes in 60.96 milliseconds (16.41 frames per second) on an NVIDIA Tesla T4 GPU, delivering the fastest end-to-end throughput among evaluated real-time detectors.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U001](rewrites.md#u001).

## M009 — U002, [sections/conclusion.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:5), câu 1

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 21; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: ultra-compact.
Cụm cấu trúc đã đọc theo ngữ cảnh: In this work, we.
Cụm thì cần sửa: không.

```latex
In this work, we presented MobilePIXORNeXt, an ultra-compact, proposal-free 3D LiDAR object detector designed for real-time edge perception on resource-constrained platforms.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U002](rewrites.md#u002).

## M010 — U002, [sections/conclusion.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:5), câu 2

Phân loại: vocabulary-context, comma-marker-proxy>3.
Từ hiển thị: 35; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: eliminates.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
By introducing the deterministic Strata-BEV14 rasterizer, the architecture eliminates computationally intensive PointNet pillar encoders and 3D sparse convolutions, preserving continuous vertical distribution moments, material surface reflectivity, and horizontal subcell centroid displacements with zero learned parameters.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U002](rewrites.md#u002).

## M011 — U002, [sections/conclusion.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:5), câu 3

Phân loại: tense-operation-or-finding.
Từ hiển thị: 30; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Multi-scale spatial features are processed.

```latex
Multi-scale spatial features are processed via a single-branch ConvNeXt-style backbone hierarchy with embedded parameter-free Block-SimAM 3D energy attention, unified by a Scale-Gated Feature Pyramid Network into decoupled grouped prediction towers.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U002](rewrites.md#u002).

## M012 — U002, [sections/conclusion.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:5), câu 4

Phân loại: tense-operation-or-finding.
Từ hiển thị: 27; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Training is governed.

```latex
Training is governed by Physics-Consistent Ground-Truth Augmentation (PC-GTA) alongside Oriented Geometric Alignment (OGA) and IoU-Aware Quality Alignment (IQA), dynamically balanced via 12-parameter Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U002](rewrites.md#u002).

## M013 — U003, [sections/conclusion.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:7), câu 1

Phân loại: vocabulary-context, LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 26; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: leading.
Cụm cấu trúc đã đọc theo ngữ cảnh: establishes a leading Pareto trade-off.
Cụm thì cần sửa: Evaluations on.

```latex
Evaluations on the KITTI validation benchmark using Moderate BEV AP $R_{40}$ demonstrate that MobilePIXORNeXt establishes a leading Pareto trade-off between detection accuracy, model compactness, and operational latency.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U003](rewrites.md#u003).

## M014 — U003, [sections/conclusion.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:7), câu 2

Phân loại: tense-operation-or-finding.
Từ hiển thị: 15; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: the model achieves.

```latex
Containing exactly $753{,}569$ parameters ($9.34\,\text{MB}$ checkpoint), the model achieves $71.46\%$ Moderate mAP and $73.42\%$ $\text{mAP}_9$ across Car, Pedestrian, and Cyclist.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U003](rewrites.md#u003).

## M015 — U003, [sections/conclusion.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:7), câu 3

Phân loại: tense-operation-or-finding.
Từ hiển thị: 25; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: These results surpass.

```latex
These results surpass PointPillars by $1.33\%$ Moderate mAP with an $84.4\%$ parameter reduction, while matching the detection performance of SECOND ($72.09\%$ mAP) with an $85.8\%$ parameter reduction ($7.1\times$ fewer parameters).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U003](rewrites.md#u003).

## M016 — U003, [sections/conclusion.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:7), câu 4

Phân loại: tense-operation-or-finding.
Từ hiển thị: 17; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: the network achieves.

```latex
On vulnerable road users, the network achieves $60.75\%$ Moderate AP on pedestrians, outperforming PointPillars by $4.74\%$ and SECOND by $4.19\%$.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U003](rewrites.md#u003).

## M017 — U003, [sections/conclusion.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:7), câu 5

Phân loại: tense-operation-or-finding.
Từ hiển thị: 25; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: MobilePIXORNeXt executes.

```latex
In full-system benchmarking on an NVIDIA Tesla T4 GPU, MobilePIXORNeXt executes the complete input-to-detection pipeline in $60.96\,\text{ms}$ ($16.41\,\text{FPS}$), achieving the lowest operational latency among evaluated real-time detectors.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U003](rewrites.md#u003).

## M018 — U003, [sections/conclusion.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:7), câu 6

Phân loại: LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 33; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: confirm that.
Cụm thì cần sửa: confirm that; PC-GTA yields; OGA and IQA deliver.

```latex
Controlled ablation experiments confirm that PC-GTA yields a $+6.03\%$ mAP improvement by mitigating unphysical synthesis artifacts, while OGA and IQA deliver an additional $+3.20\%$ mAP gain through rotated geometric overlap alignment and confidence score calibration.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U003](rewrites.md#u003).

## M019 — U004, [sections/conclusion.tex:9](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:9), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 11; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: Despite these advantages, several practical limitations outline directions for future research.
Cụm thì cần sửa: không.

```latex
Despite these advantages, several practical limitations outline directions for future research.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U004](rewrites.md#u004).

## M020 — U004, [sections/conclusion.tex:9](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:9), câu 3

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 30; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
While vertical elevation and height regression are supported via an auxiliary branch, localization fidelity on steep grades, multi-level interchanges, and irregular non-planar terrains requires further investigation on dedicated off-road benchmarks.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U004](rewrites.md#u004).

## M021 — U005, [sections/declarations.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/declarations.tex:5), câu 1

Phân loại: LLM-contextual-pattern, comma-marker-proxy>3.
Từ hiển thị: 22; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: would like to acknowledge.
Cụm thì cần sửa: không.

```latex
The authors would like to acknowledge FPT University, Vietnam, for providing computational infrastructure, GPU workstation clusters, and laboratory facilities supporting this research.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U005](rewrites.md#u005).

## M022 — U012, [sections/declarations.tex:29](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/declarations.tex:29), câu 1

Phân loại: vocabulary-context.
Từ hiển thị: 13; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: significantly.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
All authors contributed significantly to the study conception, architecture design, and empirical evaluation.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U012](rewrites.md#u012).

## M023 — U013, [sections/experiments.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:5), câu 3

Phân loại: tense-operation-or-finding.
Từ hiển thị: 22; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: We adopt.

```latex
We adopt the canonical 50/50 split introduced by Chen et al.~\cite{yang2018pixor}, which partitions the benchmark into $3{,}712$ training frames and $3{,}769$ validation frames.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U013](rewrites.md#u013).

## M024 — U013, [sections/experiments.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:5), câu 4

Phân loại: tense-operation-or-finding.
Từ hiển thị: 25; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: are generated exclusively.

```latex
To prevent data leakage and ensure strict experimental isolation, all ground-truth sampling databases utilized by data augmentation pipelines are generated exclusively from the $3{,}712$ training frames.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U013](rewrites.md#u013).

## M025 — U014, [sections/experiments.tex:8](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:8), câu 1

Phân loại: tense-operation-or-finding.
Từ hiển thị: 20; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: is aligned.

```latex
Following the geometric contract defined in Section~\ref{sec:problem_formulation}, the LiDAR coordinate frame is aligned with the vehicle body (forward $X$, leftward $Y$, upward $Z$).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U014](rewrites.md#u014).

## M026 — U014, [sections/experiments.tex:8](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:8), câu 2

Phân loại: tense-operation-or-finding.
Từ hiển thị: 9; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Perception is bounded.

```latex
Perception is bounded within the Cartesian Region of Interest:
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U014](rewrites.md#u014).

## M027 — U015, [sections/experiments.tex:16](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:16), câu 1

Phân loại: tense-operation-or-finding.
Từ hiển thị: 11; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: is enforced.

```latex
An open-boundary inward margin of $\epsilon = 0.001\,\text{m}$ is enforced along all three axes.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U015](rewrites.md#u015).

## M028 — U015, [sections/experiments.tex:16](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:16), câu 4

Phân loại: tense-operation-or-finding.
Từ hiển thị: 12; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Evaluation focuses.

```latex
Evaluation focuses on the three principal road categories: Car, Pedestrian, and Cyclist.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U015](rewrites.md#u015).

## M029 — U016, [sections/experiments.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:19), câu 1

Phân loại: tense-operation-or-finding.
Từ hiển thị: 34; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: is benchmarked.

```latex
Detection performance is benchmarked on oriented 2D Bird's-Eye View (BEV) footprints using 40-point interpolated Average Precision (AP $R_{40}$) on the KITTI validation set, which calculates the area under the precision--recall curve over 40 recall positions $\mathcal{R}_{40} = \{1/40, 2/40, \dots, 1.0\}$:
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U016](rewrites.md#u016).

## M030 — U017, [sections/experiments.tex:23](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:23), câu 1

Phân loại: tense-operation-or-finding.
Từ hiển thị: 22; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: are set.

```latex
Following KITTI matching conventions, class-specific oriented bounding box Intersection-over-Union (IoU) overlap thresholds are set to $\tau_{\mathrm{IoU}} = 0.70$ for Car, and $\tau_{\mathrm{IoU}} = 0.50$ for Pedestrian and Cyclist.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U017](rewrites.md#u017).

## M031 — U017, [sections/experiments.tex:23](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:23), câu 2

Phân loại: tense-operation-or-finding.
Từ hiển thị: 7; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: are categorized.

```latex
Detections are categorized into three difficulty regimes:
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U017](rewrites.md#u017).

## M032 — U022, [sections/experiments.tex:33](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:33), câu 1

Phân loại: tense-operation-or-finding.
Từ hiển thị: 11; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: We additionally report.

```latex
We additionally report the unweighted mean across all nine class-difficulty combinations:
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U022](rewrites.md#u022).

## M033 — U023, [sections/experiments.tex:37](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:37), câu 1

Phân loại: tense-operation-or-finding.
Từ hiển thị: 24; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: results use.

```latex
The MobilePIXORNeXt and historical MobilePIXOR results use the local loader-aligned BEV evaluator, while the reference detector results in Table~\ref{tab:comparison} use the reference KITTI evaluator.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U023](rewrites.md#u023).

## M034 — U023, [sections/experiments.tex:37](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:37), câu 2

Phân loại: tense-operation-or-finding.
Từ hiển thị: 17; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: All report.

```latex
All report $R_{40}$ interpolation on the same validation split; the evaluator implementations are distinguished in the comparison table.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U023](rewrites.md#u023).

## M035 — U023, [sections/experiments.tex:37](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:37), câu 3

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 11; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Here, E, M, and H denote Easy, Moderate, and Hard regimes.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U023](rewrites.md#u023).

## M036 — U024, [sections/experiments.tex:43](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:43), câu 2

Phân loại: comma-marker-proxy>3, tense-operation-or-finding.
Từ hiển thị: 37; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: The network comprises.

```latex
The network comprises a two-layer $3 \times 3$ convolutional stem ($32$ channels, stride 2), three hierarchical backbone stages with channel widths $(48, 96, 128)$ and block depths $(3, 4, 2)$ based on \texttt{MobilePixorNeXt-S} blocks with large $7 \times 7$ depthwise convolutions, inverted expansion ratio $e=2.5$, SiLU activations, and LayerScale parameters initialized to $10^{-5}$.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U024](rewrites.md#u024).

## M037 — U024, [sections/experiments.tex:43](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:43), câu 3

Phân loại: tense-operation-or-finding.
Từ hiển thị: 10; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: attention is embedded.

```latex
Parameter-free Block-SimAM attention is embedded inside all nine residual blocks.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U024](rewrites.md#u024).

## M038 — U024, [sections/experiments.tex:43](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:43), câu 4

Phân loại: vocabulary-context, tense-operation-or-finding.
Từ hiển thị: 21; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: optimal.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Network (SG-FPN) fuses.

```latex
The multiscale Scale-Gated Feature Pyramid Network (SG-FPN) fuses stages C4, C3, and C2 into an optimal $F_{32}$ (32-channel) representation at stride 4 ($200 \times 176$).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U024](rewrites.md#u024).

## M039 — U024, [sections/experiments.tex:43](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:43), câu 5

Phân loại: tense-operation-or-finding.
Từ hiển thị: 17; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: towers process.

```latex
Decoupled GroupedHeader towers process Car ($K_1=1$) and Pedestrian/Cyclist ($K_2=2$) independently, utilizing a shared initial $3 \times 3$ Conv-BN-SiLU transformation per group.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U024](rewrites.md#u024).

## M040 — U025, [sections/experiments.tex:46](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:46), câu 5

Phân loại: tense-operation-or-finding.
Từ hiển thị: 25; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: schedule spans.

```latex
The standard schedule spans 50 epochs and ends at epoch 50, comprising 4 linear warmup epochs followed by 46 epochs of cosine annealing down to $\eta_{\min}$.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U025](rewrites.md#u025).

## M041 — U025, [sections/experiments.tex:46](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:46), câu 6

Phân loại: tense-operation-or-finding.
Từ hiển thị: 28; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: checkpoint comes.

```latex
The final reported checkpoint comes from an extended 160-epoch run with 10 warmup epochs and was selected at epoch 130; the checkpoint-selection comparison retains these recorded epoch indices.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U025](rewrites.md#u025).

## M042 — U026, [sections/experiments.tex:49](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:49), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 30; bộ đếm dấu phẩy/từ nối: 6.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
In the final reported run, local ground fitting, static obstacle clearance, range-adaptive point subsampling, and visibility protection were enabled; LOS rejection, shadow masking, and radiometric calibration were available but disabled.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U026](rewrites.md#u026).

## M043 — U027, [sections/experiments.tex:52](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:52), câu 3

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 30; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Numerical forward and backward passes utilized Automatic Mixed Precision (AMP) in Bfloat16 (\texttt{bf16}), while loss evaluation, coordinate regression, and rotated NMS were conducted in full FP32 precision to maintain numerical stability.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U027](rewrites.md#u027).

## M044 — U027, [sections/experiments.tex:52](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:52), câu 4

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 40; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
The final MobilePIXORNeXt timing and accuracy evaluation was performed on a server equipped with an NVIDIA Tesla T4 GPU (16\,GB GDDR6 VRAM, Turing architecture), host x86\_64 CPU, Linux OS, and PyTorch 2.11.0 with CUDA 13.0.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U027](rewrites.md#u027).

## M045 — U027, [sections/experiments.tex:52](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:52), câu 7

Phân loại: tense-operation-or-finding.
Từ hiển thị: 42; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: all reported ablation and comparative metrics reflect.

```latex
Following standard conventions on the competitive KITTI benchmark, all reported ablation and comparative metrics reflect controlled deterministic evaluation with a fixed random seed ($\text{seed} = 42$) under identical validation split conditions, where variance and standard deviation ($\pm\,\text{std}$) across multi-seed runs are demarcated for future multi-fold benchmarks.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U027](rewrites.md#u027).

## M046 — U028, [sections/introduction.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:5), câu 1

Phân loại: vocabulary-context.
Từ hiển thị: 26; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: essential.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Three-dimensional (3D) spatial perception from Light Detection and Ranging (LiDAR) point clouds provides essential sensory grounding for safe autonomous navigation, mobile robotics, and intelligent transportation systems.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U028](rewrites.md#u028).

## M047 — U028, [sections/introduction.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:5), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 35; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Unlike passive optical cameras, which remain vulnerable to ambient illumination fluctuations and projective scale ambiguities, LiDAR sensors actively emit pulsed laser beams to measure direct metric ranges, capturing high-precision geometric boundaries under arbitrary day-and-night conditions~\cite{geiger2012kitti}.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U028](rewrites.md#u028).

## M048 — U028, [sections/introduction.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:5), câu 3

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 37; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
In time-critical vehicular environments, perception systems must accurately detect, orient, and classify surrounding objects (from passenger vehicles to compact vulnerable road users such as pedestrians and cyclists) within strict latency envelopes, typically requiring sub-$100\,\text{ms}$ execution per scan~\cite{yang2018pixor}.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U028](rewrites.md#u028).

## M049 — U028, [sections/introduction.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:5), câu 4

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 23; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: acute operational tension.
Cụm thì cần sửa: không.

```latex
However, deploying high-accuracy 3D object detection models on embedded vehicular platforms faces an acute operational tension between detection accuracy and hardware resource consumption.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U028](rewrites.md#u028).

## M050 — U028, [sections/introduction.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:5), câu 6

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 27; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: essential, ultra-compact, highly discriminative.
Cụm cấu trúc đã đọc theo ngữ cảnh: essential prerequisite.
Cụm thì cần sửa: không.

```latex
Consequently, developing ultra-compact yet highly discriminative LiDAR object detection architectures that sustain real-time execution without forfeiting spatial localization fidelity remains an essential prerequisite for scalable autonomous driving.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U028](rewrites.md#u028).

## M051 — U029, [sections/introduction.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:7), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 30; bộ đếm dấu phẩy/từ nối: 7.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
First, volumetric voxel-based architectures, exemplified by SECOND~\cite{yan2018second}, and two-stage point-voxel frameworks, such as PointRCNN~\cite{shi2019pointrcnn}, Part-$A^2$~\cite{shi2020part}, and PV-RCNN~\cite{shi2020pvrcnn}, achieve competitive detection accuracy by deploying 3D sparse convolutions or localized point-cloud abstractions.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U029](rewrites.md#u029).

## M052 — U029, [sections/introduction.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:7), câu 5

Phân loại: vocabulary-context, comma-marker-proxy>3.
Từ hiển thị: 22; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: substantial.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
While faster than volumetric methods, non-deterministic point gathering, dynamic memory allocation, and sparse scattering introduce substantial runtime overhead and memory transfer latency.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U029](rewrites.md#u029).

## M053 — U029, [sections/introduction.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:7), câu 6

Phân loại: vocabulary-context.
Từ hiển thị: 23; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: eliminate.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Third, proposal-free BEV rasterization methods, such as PIXOR~\cite{yang2018pixor} and MobilePIXOR~\cite{nguyen2024indoor}, eliminate learned encoders entirely by discretizing point clouds into fixed vertical occupancy grids.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U029](rewrites.md#u029).

## M054 — U029, [sections/introduction.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:7), câu 7

Phân loại: vocabulary-context, comma-marker-proxy>3.
Từ hiển thị: 41; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: severe.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Although conceptually efficient, existing rasterization methods suffer from severe spatial quantization errors on standard $0.1\,\text{m}$ grids; by relying on crude binary occupancy slices, they discard continuous subcell offsets and high-order vertical dispersion statistics, inducing spatial staircasing and localization jitter on small objects.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U029](rewrites.md#u029).

## M055 — U029, [sections/introduction.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:7), câu 10

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 32; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: constitutes the core bottleneck impeding.
Cụm thì cần sửa: không.

```latex
The failure of existing compact detectors to simultaneously resolve fine subcell quantization errors, unphysical augmentation artifacts, multi-scale receptive field coverage, and geometric score misalignment constitutes the core bottleneck impeding sub-million-parameter LiDAR perception.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U029](rewrites.md#u029).

## M056 — U030, [sections/introduction.tex:9](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:9), câu 1

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 45; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: fundamental.
Cụm cấu trúc đã đọc theo ngữ cảnh: key scientific insight.
Cụm thì cần sửa: không.

```latex
We address this fundamental challenge through a key scientific insight: sub-voxel geometric precision can be preserved deterministically at the representation level by coupling multi-band vertical densities and continuous elevation moments with horizontal subcell centroid displacements, bypassing learned PointNet encoders entirely without forfeiting fine spatial boundaries.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U030](rewrites.md#u030).

## M057 — U030, [sections/introduction.tex:9](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:9), câu 2

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 40; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: ultra-compact, superior.
Cụm cấu trúc đã đọc theo ngữ cảnh: Grounded in this principle.
Cụm thì cần sửa: không.

```latex
Grounded in this principle, we investigate the central research question: \textit{Can an ultra-compact LiDAR detector operating on a parameter-free BEV tensor rival the localization accuracy of heavy volumetric and pillar architectures while achieving superior end-to-end throughput on commodity edge hardware?}
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. Câu hỏi nghiên cứu trong \textit{} được giữ nguyên như một giả thuyết so sánh, không coi là kết luận đã chứng minh. [Đối chiếu U030](rewrites.md#u030).

## M058 — U030, [sections/introduction.tex:9](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:9), câu 3

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 37; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: To realize this objective.
Cụm thì cần sửa: không.

```latex
To realize this objective, our framework couples this deterministic representation with a single-branch ConvNeXt-style backbone~\cite{liu2022convnext} utilizing large $7 \times 7$ depthwise kernels and parameter-free 3D spatial-energy attention (SimAM)~\cite{yang2021simam}, expanding physical receptive fields to envelop entire vehicles without multi-branch training overhead.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U030](rewrites.md#u030).

## M059 — U030, [sections/introduction.tex:9](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:9), câu 5

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 26; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: bridge the performance divide.
Cụm thì cần sửa: không.

```latex
Together, these complementary mechanisms demonstrate that disciplined geometric encoding and calibrated multi-task loss formulation bridge the performance divide between compact 2D representations and multi-million-parameter 3D networks.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U030](rewrites.md#u030).

## M060 — U031, [sections/introduction.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:11), câu 1

Phân loại: vocabulary-context, LLM-contextual-pattern, comma-marker-proxy>3.
Từ hiển thị: 20; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: ultra-compact.
Cụm cấu trúc đã đọc theo ngữ cảnh: Guided by this design philosophy.
Cụm thì cần sửa: không.

```latex
Guided by this design philosophy, we propose \textbf{MobilePIXORNeXt}, an ultra-compact, proposal-free, multi-class 3D LiDAR object detector containing exactly $753{,}569$ trainable parameters ($<0.76\,\text{M}$).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U031](rewrites.md#u031).

## M061 — U031, [sections/introduction.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:11), câu 3

Phân loại: vocabulary-context, words>50, comma-marker-proxy>3.
Từ hiển thị: 56; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: eliminate.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
An unordered LiDAR point cloud is deterministically transformed via the parameter-free \textbf{Strata-BEV14} rasterizer into a 14-channel statistical BEV tensor ($800 \times 704$ at $0.1\,\text{m}$ resolution) using NumPy or Numba on CPU, with point-cloud loading and rasterization taking $20.94\,\text{ms}$ in the measured pipeline, capturing height-band densities, vertical spatial moments ($z_{\max}, z_{\mathrm{mean}}, z_{\mathrm{span}}, z_{\mathrm{std}}$), surface reflectance statistics, and continuous subcell centroid displacement offsets $(\Delta\bar{x}, \Delta\bar{y})$ to eliminate quantization error.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U031](rewrites.md#u031).

## M062 — U031, [sections/introduction.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:11), câu 4

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 35; bộ đếm dấu phẩy/từ nối: 8.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
To regularize training without unphysical artifacts, we formulate \textbf{Physics-Consistent Ground-Truth Augmentation (PC-GTA)}, which combines local ground elevation fitting, static obstacle clearance, and range-adaptive point subsampling, with optional line-of-sight checks, raycast shadow masking, and intensity attenuation.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U031](rewrites.md#u031).

## M063 — U031, [sections/introduction.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:11), câu 5

Phân loại: vocabulary-context, comma-marker-proxy>3.
Từ hiển thị: 34; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: streamlined.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Multi-scale feature extraction is performed by the \textbf{MobilePixorNeXt-S} backbone hierarchy across stages $C_2, C_3, C_4$, utilizing streamlined $7 \times 7$ depthwise inverted bottleneck blocks ($e=2.5$), SiLU activations, LayerScale stabilization, and embedded parameter-free \textbf{Block-SimAM} energy attention operating inside all nine residual blocks.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U031](rewrites.md#u031).

## M064 — U031, [sections/introduction.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:11), câu 6

Phân loại: vocabulary-context, comma-marker-proxy>3.
Từ hiển thị: 32; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: eliminate.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Multi-scale semantic context and fine coordinates are balanced by the top-down \textbf{Scale-Gated Feature Pyramid Network (SG-FPN Neck, $F_{32}$)}, which employs bilinear upsampling, depthwise refinement, and learnable scale gates to eliminate deconvolution checkerboard artifacts.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U031](rewrites.md#u031).

## M065 — U031, [sections/introduction.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:11), câu 10

Phân loại: LLM-contextual-pattern, comma-marker-proxy>3.
Từ hiển thị: 32; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: this coherent pipeline resolves each aforementioned bottleneck.
Cụm thì cần sửa: không.

```latex
By deterministically capturing continuous geometry at the input, expanding receptive fields via single-branch large-kernel convolution, and supervising rotated overlap directly, this coherent pipeline resolves each aforementioned bottleneck within a unified, edge-deployable framework.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U031](rewrites.md#u031).

## M066 — U032, [sections/introduction.tex:15](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:15), câu 1

Phân loại: vocabulary-context.
Từ hiển thị: 4; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: ultra-compact.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
\item \textbf{Ultra-compact proposal-free BEV detection.}
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U032](rewrites.md#u032).

## M067 — U032, [sections/introduction.tex:15](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:15), câu 2

Phân loại: vocabulary-context.
Từ hiển thị: 18; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: ultra-compact, eliminates.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
We develop \textbf{MobilePIXORNeXt}, an ultra-compact ($753{,}569$ trainable parameters) LiDAR detector that eliminates learned PointNet encoders and 3D sparse convolutions.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U032](rewrites.md#u032).

## M068 — U032, [sections/introduction.tex:15](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:15), câu 3

Phân loại: vocabulary-context.
Từ hiển thị: 31; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: streamlined.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
The architecture integrates the deterministic 14-channel \textbf{Strata-BEV14} representation, the streamlined \textbf{MobilePixorNeXt-S} backbone hierarchy with embedded parameter-free \textbf{Block-SimAM} energy attention, the \textbf{SG-FPN} neck ($F_{32}$), and decoupled \textbf{GroupedHeader} towers to balance multi-scale road categories.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U032](rewrites.md#u032).

## M069 — U033, [sections/introduction.tex:17](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:17), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 26; bộ đếm dấu phẩy/từ nối: 7.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
We formulate \textbf{PC-GTA}, which combines local terrain alignment, obstacle clearance, and range-dependent point subsampling, and supports optional visibility, shadow-masking, and intensity-adjustment heuristics to reduce synthesis artifacts.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U033](rewrites.md#u033).

## M070 — U034, [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19), câu 1

Phân loại: vocabulary-context.
Từ hiển thị: 7; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: leading.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
\item \textbf{Empirical validation and leading edge Pareto efficiency.}
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U034](rewrites.md#u034).

## M071 — U034, [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19), câu 2

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 17; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: leading.
Cụm cấu trúc đã đọc theo ngữ cảnh: leading accuracy-to-parameter.
Cụm thì cần sửa: không.

```latex
Evaluated using BEV AP $R_{40}$ on the KITTI validation set, MobilePIXORNeXt establishes a leading accuracy-to-parameter and accuracy-to-latency trade-off.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U034](rewrites.md#u034).

## M072 — U034, [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19), câu 4

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 25; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
In complete end-to-end pipeline speed, it operates at \textbf{16.41\,FPS} (\textbf{60.96\,ms} on NVIDIA Tesla T4), running faster than all evaluated real-time detectors.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U034](rewrites.md#u034).

## M073 — U034, [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19), câu 6

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 34; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: decisive.
Cụm cấu trúc đã đọc theo ngữ cảnh: decisive.
Cụm thì cần sửa: không.

```latex
Controlled ablations demonstrate that PC-GTA yields a decisive \textbf{+6.03\% Moderate mAP} gain ($60.22\%$ to $66.25\%$), OGA and IQA deliver an additional \textbf{+3.20\% mAP} gain, and Block-SimAM provides \textbf{+0.88\% mAP} at zero parameter overhead.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U034](rewrites.md#u034).

## M074 — U035, [sections/introduction.tex:22](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:22), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 9; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: The remainder of this paper is structured as follows.
Cụm thì cần sửa: không.

```latex
The remainder of this paper is structured as follows.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U035](rewrites.md#u035).

## M075 — U036, [sections/methodology.tex:8](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:8), câu 3

Phân loại: vocabulary-context.
Từ hiển thị: 31; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: streamlined.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
The streamlined backbone extracts multi-scale features through ConvNeXt-style stages with embedded parameter-free Block-SimAM attention, fused by a top-down Scale-Gated Feature Pyramid Network (SG-FPN) into a 32-channel shared representation at stride 4 ($200 \times 176$).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U036](rewrites.md#u036).

## M076 — U036, [sections/methodology.tex:8](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:8), câu 5

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 32; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Solid arrows trace the feedforward inference path, whereas dashed arrows indicate training-only supervision incorporating Oriented Geometric Alignment (OGA), IoU-Aware Quality Alignment (IQA), and dynamic loss balancing via Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW).}
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U036](rewrites.md#u036).

## M077 — U037, [sections/methodology.tex:12](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:12), câu 2

Phân loại: vocabulary-context.
Từ hiển thị: 22; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: eliminates.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Operating strictly on an efficient Bird's-Eye View (BEV) representation, the framework eliminates computationally expensive learned point-cloud feature encoders and three-dimensional sparse convolutions.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U037](rewrites.md#u037).

## M078 — U038, [sections/methodology.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:19), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 36; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
An unordered LiDAR point cloud is acquired via a rotating scanner and defined as a set $\mathcal{P} = \{\mathbf{p}_i\}_{i=1}^{N_{\mathrm{pts}}}$, where each point $\mathbf{p}_i = (x_i, y_i, z_i, \iota_i) \in \mathbb{R}^4$ encapsulates longitudinal distance $x_i$ (forward positive), lateral distance $y_i$ (leftward positive), vertical elevation $z_i$ (upward positive), and calibrated reflectance intensity $\iota_i \in [0, 1]$.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U038](rewrites.md#u038).

## M079 — U039, [sections/methodology.tex:27](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:27), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 12; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
configured with longitudinal bounds $[0.0, 70.4]$\,m, lateral bounds $[-40.0, 40.0]$\,m, and vertical bounds $[-2.5, 1.0]$\,m.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U039](rewrites.md#u039).

## M080 — U039, [sections/methodology.tex:27](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:27), câu 2

Phân loại: vocabulary-context, tense-operation-or-finding.
Từ hiển thị: 12; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: eliminate.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: is enforced.

```latex
An open-boundary inward margin of $\epsilon = 0.001$\,m is enforced to eliminate boundary artifacts.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U039](rewrites.md#u039).

## M081 — U041, [sections/methodology.tex:39](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:39), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 27; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Operating on this contract requires a structured, dense BEV representation that captures fine 3D geometric distributions without the latency of learned point encoders, which we formulate next.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U041](rewrites.md#u041).

## M082 — U042, [sections/methodology.tex:50](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:50), câu 2

Phân loại: words>50.
Từ hiển thị: 68; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Channels 0--3 capture logarithmic point densities across four uniform vertical elevation strata ($[-2.5, 1.0]\,\text{m}$); Channels 4--7 encode vertical spatial moments (normalized maximum elevation $z_{\max}$, mean elevation $z_{\mathrm{mean}}$, elevation span $z_{\mathrm{span}}$, and sample standard deviation $z_{\mathrm{std}}$); Channels 8--9 capture calibrated surface reflectance statistics (maximum and mean intensity); Channel 10 records total cell logarithmic density; Channel 11 provides a binary occupancy indicator; and Channels 12--13 preserve continuous subcell horizontal centroid displacements $(\Delta\bar{x}, \Delta\bar{y}) \in [-0.5, 0.5]$ to mitigate spatial discretization errors.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U042](rewrites.md#u042).

## M083 — U043, [sections/methodology.tex:56](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:56), câu 3

Phân loại: tense-operation-or-finding.
Từ hiển thị: 14; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: We therefore propose.

```latex
We therefore propose \textbf{Strata-BEV14} (Figure~\ref{fig:strata_bev14}), a deterministic, parameter-free rasterizer that balances detail and efficiency.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U043](rewrites.md#u043).

## M084 — U044, [sections/methodology.tex:60](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:60), câu 1

Phân loại: tense-operation-or-finding.
Từ hiển thị: 20; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: are normalized; are clamped.

```latex
For each occupied cell $(u, v)$ containing $N$ points $\mathcal{P}_{u, v} = \{\mathbf{p}_i\}_{i=1}^N$, vertical elevations are normalized linearly to unit interval $\zeta_i = (z_i - z_{\min}) / (z_{\max} - z_{\min}) \in [0, 1]$, and reflectance values $\iota_i$ are clamped to $[0, 1]$.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U044](rewrites.md#u044).

## M085 — U047, [sections/methodology.tex:69](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:69), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 23; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
\textit{Vertical Spatial Moments ($F_4$--$F_7$)}: To encode obstacle geometry without explicit voxelization, we extract peak elevation $F_4 = \max_i \zeta_i$, mean elevation $F_5 = \frac{1}{N}\sum_i \zeta_i$, vertical span $F_6 = \max_i \zeta_i - \min_i \zeta_i$, and the sample standard deviation:
```

**Xử lý:** Giữ nguyên: cần đọc theo vai trò định nghĩa, danh sách hoặc mô tả hình; không tự suy ra lỗi từ bộ đếm dấu phẩy. [Đối chiếu U047](rewrites.md#u047).

## M086 — U052, [sections/methodology.tex:122](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:122), câu 4

Phân loại: tense-operation-or-finding.
Từ hiển thị: 20; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: together take.

```latex
Point-cloud loading and rasterization together take $20.94$\,ms in the measured evaluation pipeline; this preprocessing measurement is not an encoder-only timing.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U052](rewrites.md#u052).

## M087 — U053, [sections/methodology.tex:128](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:128), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 29; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Cut-and-paste ground-truth (GT) augmentation balances rare LiDAR categories, but naive insertion ignores scene physics, producing floating or submerged objects, background collisions, missing raycast shadows, and unrealistic range-dependent point densities.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U053](rewrites.md#u053).

## M088 — U053, [sections/methodology.tex:128](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:128), câu 2

Phân loại: tense-operation-or-finding.
Từ hiển thị: 18; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: improves Moderate mAP.

```latex
Our \textbf{Physics-Consistent Ground-Truth Augmentation} (\texttt{PC-GTA}) addresses these artifacts and improves Moderate mAP by $+6.03\%$ over naive sampling in our ablations.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U053](rewrites.md#u053).

## M089 — U063, [sections/methodology.tex:186](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:186), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 36; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
A two-layer convolutional stem ingests 14-channel input at stride 2 ($400 \times 352$), followed by three hierarchical stages (C2: 48 channels, depth 3; C3: 96 channels, depth 4; C4: 128 channels, depth 2) composed of \texttt{MobilePixorNeXt-S} blocks with embedded Block-SimAM.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U063](rewrites.md#u063).

## M090 — U063, [sections/methodology.tex:186](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:186), câu 4

Phân loại: vocabulary-context.
Từ hiển thị: 22; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: eliminate.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Independent grouped prediction towers decouple Car from Pedestrian and Cyclist to eliminate aspect-ratio conflicts while conserving model capacity (753,569 total parameters).}
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U063](rewrites.md#u063).

## M091 — U064, [sections/methodology.tex:192](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:192), câu 3

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 48; bộ đếm dấu phẩy/từ nối: 6.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
In contrast, at our input grid resolution of $\Delta_x = \Delta_y = 0.1$\,m, a single $7 \times 7$ depthwise receptive field directly spans $0.7 \times 0.7$\,m of physical terrain at stride 1, and rapidly expands to over $5.6 \times 5.6$\,m by Stage 3 (stride 8)---sufficient to envelop an entire passenger vehicle in a single layer without dilation artifacts.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U064](rewrites.md#u064).

## M092 — U064, [sections/methodology.tex:192](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:192), câu 4

Phân loại: tense-operation-or-finding.
Từ hiển thị: 30; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: we design.

```latex
To achieve this wide context under strict parameter parsimony, we design the \textbf{MobilePixorNeXt Backbone} hierarchy featuring large $7 \times 7$ depthwise convolutions, inverted bottleneck expansions, and LayerScale residual stabilization organized into repeated \texttt{MobilePixorNeXt-S} blocks.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U064](rewrites.md#u064).

## M093 — U070, [sections/methodology.tex:207](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:207), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 25; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: clean single-branch.
Cụm thì cần sửa: không.

```latex
Unlike multi-branch reparameterization paradigms that require complex multi-path training topologies and post-hoc weight folding, MobilePIXORNeXt directly trains and deploys a clean single-branch large-kernel ($7 \times 7$) depthwise convolution.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U070](rewrites.md#u070).

## M094 — U070, [sections/methodology.tex:207](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:207), câu 2

Phân loại: LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 34; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: rapid optimization convergence.
Cụm thì cần sửa: the backbone achieves.

```latex
By pairing wide spatial receptive fields with inverted bottleneck expansion ($e=2.5$) and LayerScale stabilization, the backbone achieves rapid optimization convergence and fast throughput ($55.94$\,FPS on Tesla T4) without incurring auxiliary branch memory overhead during backpropagation.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U070](rewrites.md#u070).

## M095 — U071, [sections/methodology.tex:217](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:217), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 35; bộ đếm dấu phẩy/từ nối: 7.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
\textbf{Top}: Complete residual transformation path from input $\mathbf{Z}$ to output $\mathbf{Z}'$, featuring a large-kernel $7 \times 7$ depthwise convolution, inverted bottleneck expansion with ratio $e = 2.5$, SiLU activation, pointwise projection, parameter-free 3D spatial-energy attention $\mathcal{S}(\mathbf{R})$, LayerScale scaling $\boldsymbol{\gamma}_{\mathrm{blk}}$ (initialized to $10^{-5}$), and an identity shortcut.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U071](rewrites.md#u071).

## M096 — U071, [sections/methodology.tex:217](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:217), câu 3

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 36; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
\textbf{Bottom}: Internal operational mechanism of the SimAM operator, which calculates channel-wise spatial moments ($\mu_c, \nu_c$) in FP32 precision, evaluates visual saliency via inverse-energy importance $s_t = (t - \mu_c)^2 / (4(\nu_c + \lambda_{\mathrm{sim}})) + 0.5$, and produces a 3D modulation tensor $\mathbf{A} \in [0, 1]^{B \times C \times H \times W}$ via sigmoid scaling without introducing any learnable parameters.}
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U071](rewrites.md#u071).

## M097 — U074, [sections/methodology.tex:232](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:232), câu 2

Phân loại: vocabulary-context.
Từ hiển thị: 16; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: prohibitive.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Conversely, non-local spatial attention requires quadratic computational complexity $\mathcal{O}((HW)^2)$, rendering it prohibitive for real-time dense BEV grids ($200 \times 176$).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U074](rewrites.md#u074).

## M098 — U074, [sections/methodology.tex:232](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:232), câu 3

Phân loại: LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 19; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: To address this trade-off.
Cụm thì cần sửa: we integrate.

```latex
To address this trade-off, we integrate \textbf{Block-SimAM}~\cite{yang2021simam}---a 3D spatial-energy attention module operating without adding a single learned parameter.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U074](rewrites.md#u074).

## M099 — U075, [sections/methodology.tex:235](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:235), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 23; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
As illustrated in Figure~\ref{fig:next_s_block}, SimAM is embedded inside each \texttt{MobilePixorNeXt-\allowbreak S} block across all three stages (C2, C3, C4), for nine attention operators in total.
```

**Xử lý:** Giữ nguyên: cần đọc theo vai trò định nghĩa, danh sách hoặc mô tả hình; không tự suy ra lỗi từ bộ đếm dấu phẩy. [Đối chiếu U075](rewrites.md#u075).

## M100 — U080, [sections/methodology.tex:261](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:261), câu 2

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 18; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: true 3D.
Cụm thì cần sửa: không.

```latex
It simultaneously assigns true 3D spatial-and-channel attention weights, adaptively amplifying sparse LiDAR point clusters against flat terrain backgrounds.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U080](rewrites.md#u080).

## M101 — U080, [sections/methodology.tex:261](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:261), câu 3

Phân loại: LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 23; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: Empirical evaluation confirms.
Cụm thì cần sửa: Empirical evaluation confirms; achieves; outperforming stage-level.

```latex
Empirical evaluation confirms that embedding SimAM inside each \texttt{MobilePixorNeXt-S} block achieves $+0.88\%$ mAP over the no-attention configuration, outperforming stage-level adapters by up to $+2.10\%$ mAP (Table~\ref{tab:attention}).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U080](rewrites.md#u080).

## M102 — U081, [sections/methodology.tex:270](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:270), câu 1

Phân loại: vocabulary-context.
Từ hiển thị: 25; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: severe.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Standard Feature Pyramid Networks rely on transposed convolutions (deconvolutions) to upsample low-resolution feature maps, frequently introducing severe high-frequency checkerboard artifacts on continuous coordinate regression grids.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U081](rewrites.md#u081).

## M103 — U081, [sections/methodology.tex:270](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:270), câu 3

Phân loại: LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 21; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: To resolve this.
Cụm thì cần sửa: we design.

```latex
To resolve this, we design the \textbf{Top-Down Scale-Gated Feature Pyramid Network} (\texttt{SG-FPN Neck})---an efficient, unidirectional top-down fusion neck equipped with depthwise scale-gating.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U081](rewrites.md#u081).

## M104 — U083, [sections/methodology.tex:283](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:283), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 24; bộ đếm dấu phẩy/từ nối: 8.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
As shown in Figure~\ref{fig:scale_gated_fusion}, lateral features from Stages C4 ($128$\,ch), C3 ($96$\,ch), and C2 ($48$\,ch) are projected via $1 \times 1$ convolutions into matching lateral dimensions $\mathbf{L}_5 \in \mathbb{R}^{B \times 48 \times 50 \times 44}$, $\mathbf{L}_4 \in \mathbb{R}^{B \times 48 \times 100 \times 88}$, and $\mathbf{L}_3 \in \mathbb{R}^{B \times 32 \times 200 \times 176}$.
```

**Xử lý:** Giữ nguyên: cần đọc theo vai trò định nghĩa, danh sách hoặc mô tả hình; không tự suy ra lỗi từ bộ đếm dấu phẩy. [Đối chiếu U083](rewrites.md#u083).

## M105 — U087, [sections/methodology.tex:307](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:307), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 9; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: drastic morphological discrepancies.
Cụm thì cần sửa: không.

```latex
Autonomous driving involves drastic morphological discrepancies across object categories.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U087](rewrites.md#u087).

## M106 — U087, [sections/methodology.tex:307](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:307), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 20; bộ đếm dấu phẩy/từ nối: 11.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Cars exhibit rigid rectangular footprints ($l \approx 4.5$\,m, $w \approx 1.8$\,m), whereas Pedestrians ($l \approx 0.8$\,m, $w \approx 0.6$\,m) and Cyclists ($l \approx 1.8$\,m, $w \approx 0.7$\,m) are compact and highly deformable.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U087](rewrites.md#u087).

## M107 — U087, [sections/methodology.tex:307](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:307), câu 3

Phân loại: vocabulary-context.
Từ hiển thị: 18; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: severe.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Forcing a single shared prediction head to jointly regress all classes induces severe gradient competition and scale interference.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U087](rewrites.md#u087).

## M108 — U087, [sections/methodology.tex:307](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:307), câu 5

Phân loại: LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 8; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: We resolve this.
Cụm thì cần sửa: We resolve this.

```latex
We resolve this via \textbf{Shared Grouped Prediction Towers} (\texttt{GroupedHeader}).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U087](rewrites.md#u087).

## M109 — U088, [sections/methodology.tex:310](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:310), câu 1

Phân loại: tense-operation-or-finding.
Từ hiển thị: 16; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Classes are partitioned.

```latex
Classes are partitioned into two semantically compatible task groups: Group 1 for \texttt{car} ($K_1 = 1$), and Group 2 for \texttt{ped\_cyc} ($K_2 = 2$).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U088](rewrites.md#u088).

## M110 — U094, [sections/methodology.tex:325](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:325), câu 2

Phân loại: tense-operation-or-finding.
Từ hiển thị: 36; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Classification targets use; regression targets are assigned.

```latex
Classification targets use 2D Gaussian heatmaps centered at continuous object projections with a minimum radius of 4 cells, and regression targets are assigned within a radius of half the Gaussian peak, denoted by valid indicator mask $\mathbf{M}_{\mathrm{reg}}$.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U094](rewrites.md#u094).

## M111 — U095, [sections/methodology.tex:331](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:331), câu 1

Phân loại: LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 27; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: bridge the disparity.
Cụm thì cần sửa: We propose.

```latex
We propose \textbf{Oriented Geometric Alignment (OGA)} combined with \textbf{IoU-Aware Quality Alignment (IQA)} to bridge the disparity between independent coordinate parameter regression and true oriented bounding box overlap.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U095](rewrites.md#u095).

## M112 — U096, [sections/methodology.tex:334](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:334), câu 2

Phân loại: vocabulary-context.
Từ hiển thị: 6; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: synergistic.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
OGA couples two synergistic geometric objectives:
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U096](rewrites.md#u096).

## M113 — U097, [sections/methodology.tex:340](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:340), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 19; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
For each axis $\mathbf{n}_k$, 1D intervals of intersection length $I_k$, union length $U_k$, and smallest enclosing hull $E_k$ are determined, yielding 1D GIoU~\cite{rezatofighi2019giou}:
```

**Xử lý:** Giữ nguyên: cần đọc theo vai trò định nghĩa, danh sách hoặc mô tả hình; không tự suy ra lỗi từ bộ đếm dấu phẩy. [Đối chiếu U097](rewrites.md#u097).

## M114 — U101, [sections/methodology.tex:368](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:368), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 15; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: forcing the model to learn reliable confidence calibration.
Cụm thì cần sửa: không.

```latex
Detaching the target prevents gradient instability while forcing the model to learn reliable confidence calibration.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U101](rewrites.md#u101).

## M115 — U102, [sections/methodology.tex:371](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:371), câu 2

Phân loại: tense-operation-or-finding.
Từ hiển thị: 14; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: we formulate.

```latex
To support resource-constrained edge platforms, we formulate \textbf{Q-OGA} as a head-free quality calibration alternative.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U102](rewrites.md#u102).

## M116 — U103, [sections/methodology.tex:381](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:381), câu 2

Phân loại: vocabulary-context, tense-operation-or-finding.
Từ hiển thị: 33; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: ultra-compact, eliminates.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: reduces forward inference latency.

```latex
By training the classification head to output IoU-calibrated confidence scores intrinsically, Q-OGA eliminates the $18.59\,\text{K}$ parameter footprint of the explicit IQA heads and reduces forward inference latency by $1.25\,\text{ms}$, serving as an ultra-compact supervision strategy.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U103](rewrites.md#u103).

## M117 — U104, [sections/methodology.tex:384](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:384), câu 1

Phân loại: vocabulary-context, LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 26; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: eliminates, principled.
Cụm cấu trúc đã đọc theo ngữ cảnh: principled, bounded optimization framework; eliminates fragile manual hyperparameter tuning.
Cụm thì cần sửa: we formulate.

```latex
To dynamically harmonize multi-task gradient scales across six heterogeneous loss objectives $\mathcal{T} = \{\mathrm{cls}, \mathrm{off}, \mathrm{size}, \mathrm{yaw}, \mathrm{geo}, \mathrm{qual}\}$ ($J=6$), we formulate \textbf{T-SBUW} as a principled, bounded optimization framework that eliminates fragile manual hyperparameter tuning.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U104](rewrites.md#u104).

## M118 — U110, [sections/methodology.tex:459](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:459), câu 1

Phân loại: vocabulary-context, comma-marker-proxy>3.
Từ hiển thị: 20; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: ultra-compact.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
The complete MobilePIXORNeXt detector contains exactly \textbf{753,569 trainable parameters} ($< 0.76$\,M), corresponding to an ultra-compact FP32 checkpoint size of $9.34$\,MB:
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U110](rewrites.md#u110).

## M119 — U112, [sections/methodology.tex:466](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:466), câu 1

Phân loại: comma-marker-proxy>3, tense-operation-or-finding.
Từ hiển thị: 23; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: enables full end-to-end inference.

```latex
This compact footprint enables full end-to-end inference at $16.41$\,FPS ($60.96$\,ms) and network forward throughput of $55.94$\,FPS ($17.88$\,ms) on commodity NVIDIA Tesla T4 GPUs.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U112](rewrites.md#u112).

## M120 — U113, [sections/related_work.tex:4](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:4), câu 2

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 24; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: foundational paradigm.
Cụm thì cần sửa: không.

```latex
PIXOR~\cite{yang2018pixor} established the foundational paradigm of proposal-free Bird's-Eye View (BEV) detection by discretizing point clouds into fixed vertical occupancy slices and calibrated reflectance maps.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U113](rewrites.md#u113).

## M121 — U113, [sections/related_work.tex:4](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:4), câu 4

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 24; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: eliminated.
Cụm cấu trúc đã đọc theo ngữ cảnh: In parallel.
Cụm thì cần sửa: không.

```latex
In parallel, PointPillars~\cite{lang2019pointpillars} eliminated rigid vertical slicing by organizing point clouds into vertical columns (pillars) and learning spatial feature representations via localized PointNet encoders.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U113](rewrites.md#u113).

## M122 — U113, [sections/related_work.tex:4](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:4), câu 5

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 20; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: systematically modernized.
Cụm thì cần sửa: không.

```latex
Modern extensions such as PillarNeXt~\cite{li2023pillarnext} and PillarNeSt~\cite{mao2023pillarnest} systematically modernized pillar-based receptive field designs and backbone scaling for large-scale outdoor perception.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U113](rewrites.md#u113).

## M123 — U114, [sections/related_work.tex:6](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:6), câu 1

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 18; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: fundamental.
Cụm cấu trúc đã đọc theo ngữ cảnh: fundamental operational dichotomy.
Cụm thì cần sửa: không.

```latex
A fundamental operational dichotomy divides existing BEV representations between expressive PointNet-based pillar feature encoders and high-speed fixed-grid rasterization.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U114](rewrites.md#u114).

## M124 — U114, [sections/related_work.tex:6](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:6), câu 2

Phân loại: vocabulary-context, LLM-contextual-pattern, comma-marker-proxy>3.
Từ hiển thị: 31; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: prohibitive.
Cụm cấu trúc đã đọc theo ngữ cảnh: On one hand.
Cụm thì cần sửa: không.

```latex
On one hand, PointNet-based pillar encoders~\cite{lang2019pointpillars,li2023pillarnext} achieve expressive geometric learning, yet the non-deterministic point gathering, dynamic memory allocation, and sparse scatter operations incur prohibitive latency and energy penalties on edge accelerators.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U114](rewrites.md#u114).

## M125 — U114, [sections/related_work.tex:6](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:6), câu 3

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 45; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: severe.
Cụm cấu trúc đã đọc theo ngữ cảnh: On the other hand.
Cụm thì cần sửa: không.

```latex
On the other hand, fixed-grid rasterization methods~\cite{yang2018pixor,nguyen2024indoor} bypass point gathering entirely but suffer from severe spatial quantization error on standard $0.1\,\text{m}$ grids; by treating cells as binary flags or raw counters, they discard continuous subcell offsets and high-order vertical distributions, causing boundary staircasing and localization jitter.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U114](rewrites.md#u114).

## M126 — U114, [sections/related_work.tex:6](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:6), câu 6

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 38; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
By deterministically integrating four stratified vertical density bins, four continuous elevation moments ($\mu_z, \sigma_z, z_{\min}, z_{\max}$), and two continuous subcell centroid displacement offsets ($\Delta x_c, \Delta y_c$) into a 14-channel layout, Strata-BEV14 preserves sub-voxel geometric fidelity while reducing Host-to-Device transfer payloads by $60\%$ without learned encoder latency.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U114](rewrites.md#u114).

## M127 — U116, [sections/related_work.tex:13](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:13), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 15; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: have gained prominence.
Cụm thì cần sửa: không.

```latex
To overcome the latency bottleneck of volumetric processing, lightweight 2D convolutional designs have gained prominence.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U116](rewrites.md#u116).

## M128 — U116, [sections/related_work.tex:13](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:13), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 29; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
The ConvNeXt paradigm~\cite{liu2022convnext} demonstrated that conventional convolutions modernized with $7 \times 7$ depthwise kernels, inverted bottleneck expansions ($e=2.5$), and LayerScale can achieve representational power competitive with Vision Transformers while preserving convolutional hardware efficiency.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U116](rewrites.md#u116).

## M129 — U116, [sections/related_work.tex:13](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:13), câu 5

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 28; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: To address this weakness.
Cụm thì cần sửa: không.

```latex
To address this weakness, Yang et al.~\cite{yang2021simam} proposed SimAM, deriving full 3D spatial-channel attention weights analytically via an energy function inspired by visual neuroscience, requiring zero trainable parameters.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U116](rewrites.md#u116).

## M130 — U117, [sections/related_work.tex:15](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:15), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 32; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: Collectively, a shared limitation.
Cụm thì cần sửa: không.

```latex
Collectively, a shared limitation of prior mobile LiDAR detectors is the reliance on standard inverted residual bottlenecks with small $3 \times 3$ kernels~\cite{nguyen2024indoor}, which struggle to capture multi-scale context across disparate vehicle and pedestrian dimensions.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U117](rewrites.md#u117).

## M131 — U118, [sections/related_work.tex:20](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:20), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 18; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: has progressively evolved.
Cụm thì cần sửa: không.

```latex
Bounding box regression in oriented object detection has progressively evolved from unconstrained coordinate offsets toward geometrically bounded objectives.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U118](rewrites.md#u118).

## M132 — U118, [sections/related_work.tex:20](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:20), câu 3

Phân loại: vocabulary-context.
Từ hiển thị: 20; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: severe.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
However, regressing spatial coordinates, dimensions, and angles independently ignores the geometric interdependence of bounding boxes and induces severe gradient instability.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U118](rewrites.md#u118).

## M133 — U119, [sections/related_work.tex:22](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:22), câu 1

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 19; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: essential.
Cụm cấu trúc đã đọc theo ngữ cảnh: has emerged as an essential mechanism.
Cụm thì cần sửa: không.

```latex
In parallel with geometric loss formulation, joint quality calibration has emerged as an essential mechanism for accurate detection ranking.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U119](rewrites.md#u119).

## M134 — U120, [sections/related_work.tex:24](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:24), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 14; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: three pervasive limitations.
Cụm thì cần sửa: không.

```latex
Despite these advancements, existing oriented bounding box detectors still struggle with three pervasive limitations.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U120](rewrites.md#u120).

## M135 — U120, [sections/related_work.tex:24](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:24), câu 2

Phân loại: vocabulary-context.
Từ hiển thị: 29; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: severe.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
First, independent angle and corner regression objectives remain susceptible to the $\pi$-periodicity boundary jump problem, where slight angular deviations around $\pm \pi/2$ or $\pm \pi$ trigger severe loss spikes and yaw gradient collapse.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U120](rewrites.md#u120).

## M136 — U120, [sections/related_work.tex:24](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:24), câu 4

Phân loại: vocabulary-context, comma-marker-proxy>3.
Từ hiển thị: 26; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: leading.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Third, unconstrained multi-task uncertainty weighting allows task variance parameters to diverge, leading to training instability when balancing disparate loss scales across classification, geometric overlap, and orientation.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U120](rewrites.md#u120).

## M137 — U120, [sections/related_work.tex:24](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:24), câu 6

Phân loại: vocabulary-context, LLM-contextual-pattern, words>50, comma-marker-proxy>3.
Từ hiển thị: 73; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: guarantees, eliminate.
Cụm cấu trúc đã đọc theo ngữ cảnh: guarantees stable convergence; simultaneously resolving.
Cụm thì cần sửa: không.

```latex
Specifically, OGA unifies multi-axis MGIoU projections with scale-normalized $\pi$-symmetric corner supervision to eliminate boundary discontinuities; Q-OGA couples continuous IoU quality directly into classification supervision to achieve quality calibration with zero head overhead; IQA provides explicit multi-branch quality alignment scored via $S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}}\hat{q}}$ to suppress miscalibrated false positives; and Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW) guarantees stable convergence by constraining multi-task weights within a normalized, bounded simplex, simultaneously resolving orientation discontinuities, quality miscalibration, and loss scale divergence.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U120](rewrites.md#u120).

## M138 — U121, [sections/results.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:5), câu 2

Phân loại: LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 27; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: In accordance with standard benchmark conventions.
Cụm thì cần sửa: all reported scores reflect.

```latex
In accordance with standard benchmark conventions, all reported scores reflect deterministic single-seed evaluation ($\text{seed} = 42$); variance and standard deviation bounds ($\pm\,\text{std}$) across random seeds are reserved for extended evaluation splits.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U121](rewrites.md#u121).

## M139 — U121, [sections/results.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:5), câu 3

Phân loại: tense-operation-or-finding.
Từ hiển thị: 19; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: We benchmark.

```latex
We benchmark MobilePIXORNeXt against established LiDAR object detectors spanning compact BEV models, pillar-based networks, and 3D sparse convolutional architectures.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U121](rewrites.md#u121).

## M140 — U122, [sections/results.tex:12](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:12), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 29; bộ đếm dấu phẩy/từ nối: 6.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
\textbf{(a)} Detection accuracy ($\text{mAP}_{\mathrm{M}}$, \%) versus parameter footprint (M) on a logarithmic scale, highlighting that MobilePIXORNeXt (red star) achieves $71.46\%$ mAP with only $0.754\,\text{M}$ parameters, matching SECOND ($72.09\%$, $5.33\,\text{M}$) and outperforming PointPillars ($70.13\%$, $4.84\,\text{M}$) with $>84\%$ fewer parameters.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U122](rewrites.md#u122).

## M141 — U122, [sections/results.tex:12](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:12), câu 3

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 35; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
\textbf{(b)} Detection accuracy ($\text{mAP}_{\mathrm{M}}$, \%) versus end-to-end pipeline latency (ms) at batch size 1 (Tesla T4, except the historical MobilePIXOR result on L4), demonstrating that MobilePIXORNeXt achieves the fastest complete perception pipeline ($60.96\,\text{ms}$, $16.41\,\text{FPS}$) among all evaluated real-time detectors.}
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U122](rewrites.md#u122).

## M142 — U123, [sections/results.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:19), câu 1

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 20; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: leading.
Cụm cấu trúc đã đọc theo ngữ cảnh: leading accuracy-to-parameter.
Cụm thì cần sửa: không.

```latex
As visualized in the Pareto frontier plots (Figure~\ref{fig:pareto_tradeoff}), MobilePIXORNeXt established a leading accuracy-to-parameter and accuracy-to-latency trade-off for real-time edge perception.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U123](rewrites.md#u123).

## M143 — U123, [sections/results.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:19), câu 4

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 20; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: merely.
Cụm thì cần sửa: không.

```latex
Compared to the widely deployed SECOND detector~\cite{yan2018second} ($5.326\,\text{M}$ parameters), MobilePIXORNeXt trailed by merely $0.63\%$ Moderate mAP ($71.46\%$ vs. $72.09\%$) despite operating with $7.1\times$ fewer parameters.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U123](rewrites.md#u123).

## M144 — U124, [sections/results.tex:24](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:24), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 31; bộ đếm dấu phẩy/từ nối: 7.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
\caption{Nine-cell Precision--Recall ($R_{40}$) benchmark curves evaluated on the KITTI validation split across Car (top row, $\text{IoU} \ge 0.70$), Pedestrian (middle row, $\text{IoU} \ge 0.50$), and Cyclist (bottom row, $\text{IoU} \ge 0.50$) under Easy (left), Moderate (center), and Hard (right) difficulty regimes.
```

**Xử lý:** Khối hình PR do tác giả dành để tự xử lý; giữ nguyên cả caption và protected tokens. [Đối chiếu U124](rewrites.md#u124).

## M145 — U124, [sections/results.tex:24](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:24), câu 4

Phân loại: vocabulary-context.
Từ hiển thị: 26; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: superior.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Note that MobilePIXORNeXt consistently maintains superior recall and precision envelopes on vulnerable road users, attaining $60.75\%$ Moderate AP on Pedestrian compared to $56.01\%$ for PointPillars and $56.56\%$ for SECOND.}
```

**Xử lý:** Khối hình PR do tác giả dành để tự xử lý; giữ nguyên cả caption và protected tokens. [Đối chiếu U124](rewrites.md#u124).

## M146 — U125, [sections/results.tex:30](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:30), câu 1

Phân loại: vocabulary-context.
Từ hiển thị: 26; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: severe.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
In autonomous driving perception, detecting small, non-rigid objects such as pedestrians presents the most challenging operational regime due to extreme point sparsity and severe aspect-ratio disparity.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U125](rewrites.md#u125).

## M147 — U125, [sections/results.tex:30](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:30), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 35; bộ đếm dấu phẩy/từ nối: 7.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
As evidenced in Table~\ref{tab:comparison} and the Precision--Recall curves in Figure~\ref{fig:pr_curves}, MobilePIXORNeXt achieved \textbf{60.75\% Moderate AP} on Pedestrians, outperforming PointPillars ($56.01\%$, a $+4.74\%$ gain) and SECOND ($56.56\%$, a $+4.19\%$ gain), while more than doubling the baseline MobilePIXOR ($28.16\%$, a $+32.59\%$ gain).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U125](rewrites.md#u125).

## M148 — U125, [sections/results.tex:30](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:30), câu 3

Phân loại: LLM-contextual-pattern, comma-marker-proxy>3.
Từ hiển thị: 50; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: We attribute this performance advantage primarily.
Cụm thì cần sửa: không.

```latex
We attribute this performance advantage primarily to two architectural mechanisms: (i) Strata-BEV14's continuous subcell displacement offsets $(\Delta\bar{x}, \Delta\bar{y})$, which mitigate the $0.1\,\text{m}$ grid quantization error that inherently impairs small object localization; and (ii) the decoupled GroupedHeader architecture, which isolates pedestrian regression from large vehicle footprints to alleviate gradient interference across morphologically disparate categories.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U125](rewrites.md#u125).

## M149 — U126, [sections/results.tex:33](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:33), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 25; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
For passenger vehicles (Car), MobilePIXORNeXt achieved $86.90\%$ Moderate AP, closely trailing PointPillars ($88.02\%$) and SECOND ($88.56\%$) by less than $1.7\%$, while outperforming the original PIXOR ($72.02\%$) by $+14.88\%$ and MobilePIXOR ($75.52\%$) by $+11.38\%$.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U126](rewrites.md#u126).

## M150 — U126, [sections/results.tex:33](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:33), câu 3

Phân loại: comma-marker-proxy>3, tense-operation-or-finding.
Từ hiển thị: 33; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: detectors (PointRCNN, Part-A; they require.

```latex
While two-stage 3D anchor-based detectors (PointRCNN, Part-A$^2$, PV-RCNN) achieve higher absolute accuracy on rigid vehicles, they require between $4.0\,\text{M}$ and $63.8\,\text{M}$ parameters and operate at higher latencies ($133\text{--}195\,\text{ms}$), exceeding the real-time constraints of embedded edge processors.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U126](rewrites.md#u126).

## M151 — U127, [sections/results.tex:40](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:40), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 18; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: a misleading depiction of deployability.
Cụm thì cần sửa: không.

```latex
In real-time autonomous systems, evaluating neural network forward execution time in isolation yields a misleading depiction of deployability.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U127](rewrites.md#u127).

## M152 — U127, [sections/results.tex:40](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:40), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 25; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
A complete perception system must account for point cloud loading, input feature tensorization, Host-to-Device (H2D) memory transfer, forward execution, coordinate decoding, and Non-Maximum Suppression (NMS).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U127](rewrites.md#u127).

## M153 — U128, [sections/results.tex:45](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:45), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 28; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
As detailed in Table~\ref{tab:efficiency}, MobilePIXORNeXt achieved an End-to-End (E2E) pipeline latency of \textbf{60.96\,ms} (\textbf{16.41\,FPS}), delivering the lowest full-system latency among all evaluated real-time detectors.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U128](rewrites.md#u128).

## M154 — U128, [sections/results.tex:45](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:45), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 16; bộ đếm dấu phẩy/từ nối: 7.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
It operated faster than PointPillars ($63.84\,\text{ms}$, $15.66\,\text{FPS}$), MobilePIXOR ($85.92\,\text{ms}$, $11.64\,\text{FPS}$), and SECOND ($93.70\,\text{ms}$, $10.67\,\text{FPS}$), and ran over $2.3\times$ faster than the original PIXOR ($144.00\,\text{ms}$, $6.94\,\text{FPS}$).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U128](rewrites.md#u128).

## M155 — U128, [sections/results.tex:45](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:45), câu 3

Phân loại: comma-marker-proxy>3, tense-operation-or-finding.
Từ hiển thị: 33; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Percentile analysis indicates.

```latex
Percentile analysis indicates consistent latency bounds in the evaluated environment: MobilePIXORNeXt exhibited a median P50 latency of \textbf{58.28\,ms}, P95 of $80.85\,\text{ms}$, and P99 of $95.91\,\text{ms}$, demonstrating low latency dispersion under desktop GPU benchmarking.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U128](rewrites.md#u128).

## M156 — U129, [sections/results.tex:48](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:48), câu 1

Phân loại: LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 10; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: underlying mechanics of our throughput advantage.
Cụm thì cần sửa: profiling reveals.

```latex
Stage-by-stage profiling reveals the underlying mechanics of our throughput advantage:
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U129](rewrites.md#u129).

## M157 — U130, [sections/results.tex:50](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:50), câu 1

Phân loại: tense-operation-or-finding.
Từ hiển thị: 15; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: together take.

```latex
\item \textbf{Input Preprocessing ($20.94\,\text{ms}$)}: Point-cloud file loading and Strata-BEV14 rasterization through the Numba CPU backend together take $20.94\,\text{ms}$.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U130](rewrites.md#u130).

## M158 — U130, [sections/results.tex:50](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:50), câu 3

Phân loại: tense-operation-or-finding.
Từ hiển thị: 20; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: stages take.

```latex
The corresponding recorded preprocessing stages take $27.75\,\text{ms}$ for PointPillars and $27.20\,\text{ms}$ for SECOND; these are complete input-preparation timings rather than encoder-only microbenchmarks.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U130](rewrites.md#u130).

## M159 — U131, [sections/results.tex:51](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:51), câu 1

Phân loại: comma-marker-proxy>3, tense-operation-or-finding.
Từ hiển thị: 40; bộ đếm dấu phẩy/từ nối: 6.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: reduces.

```latex
\item \textbf{Host-to-Device Memory Transfer ($6.91\,\text{ms}$)}: Unlike 35-channel binary occupancy tensors that require transferring $78.8\,\text{MB}$ per scan, Strata-BEV14's compact 14-channel layout requires transferring only $31.5\,\text{MB}$ (a $60.0\%$ payload reduction), which, in conjunction with pinned-memory buffer optimization, reduces H2D latency to $6.91\,\text{ms}$ compared with $40.63\,\text{ms}$ for the MobilePIXOR baseline.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U131](rewrites.md#u131).

## M160 — U132, [sections/results.tex:52](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:52), câu 1

Phân loại: tense-operation-or-finding.
Từ hiển thị: 10; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: inference executes.

```latex
\item \textbf{Model Forward Propagation ($17.88\,\text{ms}$)}: Network inference executes in $17.88\,\text{ms}$ ($55.94\,\text{FPS}$) on Tesla T4.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U132](rewrites.md#u132).

## M161 — U132, [sections/results.tex:52](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:52), câu 2

Phân loại: tense-operation-or-finding.
Từ hiển thị: 10; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: network reaches.

```latex
In optimized standalone inference, the network reaches \textbf{115.32\,FPS} ($8.67\,\text{ms}$).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U132](rewrites.md#u132).

## M162 — U133, [sections/results.tex:53](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:53), câu 1

Phân loại: tense-operation-or-finding.
Từ hiển thị: 21; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: NMS execute.

```latex
\item \textbf{Decoding and Rotated NMS ($14.94\,\text{ms}$)}: Candidate peak selection and class-wise rotated NMS execute in $14.94\,\text{ms}$, maintaining GPU residency until final bounding box export.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U133](rewrites.md#u133).

## M163 — U135, [sections/results.tex:70](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:70), câu 1

Phân loại: vocabulary-context, comma-marker-proxy>3.
Từ hiển thị: 35; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: substantial.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Within the near-range $0\text{--}30\,\text{m}$ evaluation zone, MobilePIXORNeXt delivered substantial accuracy gains over baseline MobilePIXOR: Pedestrian AP reached \textbf{65.23\%} (more than double MobilePIXOR's $30.45\%$, a $+34.78\%$ improvement), while Cyclist AP reached \textbf{76.32\%} (nearly triple MobilePIXOR's $25.55\%$, a $+50.77\%$ improvement).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U135](rewrites.md#u135).

## M164 — U135, [sections/results.tex:70](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:70), câu 2

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 23; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: This improvement was facilitated by.
Cụm thì cần sửa: không.

```latex
This improvement was facilitated by Strata-BEV14's stratified height bins and continuous subcell offsets, which preserve fine morphological boundaries when object returns are dense.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U135](rewrites.md#u135).

## M165 — U136, [sections/results.tex:73](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:73), câu 1

Phân loại: vocabulary-context.
Từ hiển thị: 18; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: robust.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
At medium ranges ($30\text{--}50\,\text{m}$), MobilePIXORNeXt maintained robust detection capability, achieving $11.49\%$ Pedestrian AP (vs. $3.75\%$) and $44.30\%$ Cyclist AP (vs. $12.61\%$, a $+31.69\%$ gain).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U136](rewrites.md#u136).

## M166 — U136, [sections/results.tex:73](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:73), câu 3

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 29; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: severe.
Cụm cấu trúc đã đọc theo ngữ cảnh: This resilience against severe point sparsity.
Cụm thì cần sửa: không.

```latex
This resilience against severe point sparsity is consistent with the regularizing effect of our physics-aware range attenuation augmentation (\texttt{transform\_object}) and the broader spatial context captured by large $7 \times 7$ depthwise receptive fields.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U136](rewrites.md#u136).

## M167 — U138, [sections/results.tex:86](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:86), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 36; bộ đếm dấu phẩy/từ nối: 6.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Replacing MobilePIXOR's residual blocks with our ConvNeXt-style backbone ($7 \times 7$ depthwise convolutions, inverted bottleneck expansion $e=2.5$, SiLU activations, and LayerScale) elevated Moderate mAP from $41.40\%$ to $47.60\%$ (\textbf{+6.20\% mAP}), while nearly doubling standalone forward throughput from $90.57$ to \textbf{169.97\,FPS}.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U138](rewrites.md#u138).

## M168 — U138, [sections/results.tex:86](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:86), câu 2

Phân loại: vocabulary-context.
Từ hiển thị: 15; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: significantly.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Large depthwise kernels significantly broaden spatial receptive fields on BEV grids without inflating computational complexity.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U138](rewrites.md#u138).

## M169 — U139, [sections/results.tex:91](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:91), câu 3

Phân loại: vocabulary-context.
Từ hiển thị: 12; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: eliminated.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Encoding continuous elevation moments ($z_{\max}, z_{\mathrm{mean}}, z_{\mathrm{span}}, z_{\mathrm{std}}$) and subcell centroid displacements eliminated horizontal discretization errors.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U139](rewrites.md#u139).

## M170 — U140, [sections/results.tex:95](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:95), câu 3

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 17; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: mitigating gradient interference.
Cụm thì cần sửa: không.

```latex
Decoupling isolated class-specific representations, mitigating gradient interference between morphologically distinct categories such as passenger vehicles and pedestrians.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U140](rewrites.md#u140).

## M171 — U141, [sections/results.tex:100](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:100), câu 2

Phân loại: tense-operation-or-finding.
Từ hiển thị: 13; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: evaluated sampler combines.

```latex
The evaluated sampler combines terrain alignment, static obstacle rejection, and range-dependent point subsampling.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U141](rewrites.md#u141).

## M172 — U143, [sections/results.tex:108](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:108), câu 2

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 31; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: prevented high-scoring but misaligned false positives.
Cụm thì cần sửa: không.

```latex
Coupling Multi-Axis Projection GIoU with scale-normalized $\pi$-symmetric corner distances aligned loss gradients directly with rotated bounding box overlap, while IQA confidence calibration prevented high-scoring but misaligned false positives from surviving NMS.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U143](rewrites.md#u143).

## M173 — U144, [sections/results.tex:113](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:113), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 30; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: culminated in our final configuration.
Cụm thì cần sửa: không.

```latex
Embedding parameter-free Block-SimAM inside every residual block across stages C2, C3, and C4 culminated in our final configuration, achieving \textbf{71.46\% Moderate mAP} and \textbf{73.42\% mAP$_9$} (\textbf{+0.88\% mAP}).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U144](rewrites.md#u144).

## M174 — U148, [sections/results.tex:123](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:123), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 26; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
As shown in Table~\ref{tab:loss}, introducing OGA alone into baseline training increased Moderate mAP from $67.38\%$ to $68.97\%$ (\textbf{+1.59\% mAP}), with consistent gains across Car ($+0.17\%$), Pedestrian ($+2.45\%$), and Cyclist ($+2.14\%$).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U148](rewrites.md#u148).

## M175 — U148, [sections/results.tex:123](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:123), câu 4

Phân loại: LLM-contextual-pattern, comma-marker-proxy>3.
Từ hiển thị: 40; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: These findings demonstrate.
Cụm thì cần sửa: không.

```latex
These findings demonstrate that OGA and IQA provide complementary benefits: OGA refines continuous corner and boundary alignment during backpropagation, whereas IQA calibrates classification scores with true geometric overlap during inference ($S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}}\hat{q}}$), preventing misaligned boxes from suppressing high-quality candidates during rotated NMS.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U148](rewrites.md#u148).

## M176 — U149, [sections/results.tex:125](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:125), câu 2

Phân loại: tense-operation-or-finding.
Từ hiển thị: 14; bộ đếm dấu phẩy/từ nối: 0.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Models incorporating.

```latex
Models incorporating the dedicated IQA prediction branch require $0.721\,\text{M}$ parameters and execute at $8.19\,\text{ms}$ forward latency ($122.16\,\text{FPS}$).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U149](rewrites.md#u149).

## M177 — U149, [sections/results.tex:125](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:125), câu 4

Phân loại: tense-operation-or-finding.
Từ hiển thị: 25; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: Q-OGA reaches.

```latex
Without any extra prediction branches or post-processing score multiplications, Q-OGA reaches $69.20\%$ Moderate mAP, outperforming standalone OGA by $+0.23\%$ mAP and surpassing the baseline Smooth-L1 by $+1.82\%$ mAP.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U149](rewrites.md#u149).

## M178 — U149, [sections/results.tex:125](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:125), câu 5

Phân loại: LLM-contextual-pattern, comma-marker-proxy>3.
Từ hiển thị: 33; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: These findings confirm; parsimonious edge-deployment configuration.
Cụm thì cần sửa: không.

```latex
These findings confirm that Q-OGA provides a parsimonious edge-deployment configuration when memory and latency budgets are strictly capped, whereas the explicit IQA configuration offers peak accuracy ($70.58\%$) when resources accommodate a dedicated quality branch.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U149](rewrites.md#u149).

## M179 — U151, [sections/results.tex:132](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:132), câu 2

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 16; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: Because stage transitions undergo spatial downsampling.
Cụm thì cần sửa: không.

```latex
Because stage transitions undergo spatial downsampling, inserting global attention at these junctures blurs fine spatial gradients.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U151](rewrites.md#u151).

## M180 — U152, [sections/results.tex:137](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:137), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 46; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
The top subplots depict total composite training loss (left) and validation loss (right) on logarithmic scales; the remaining six panels illustrate constituent task components: Classification Focal Loss, Oriented Geometry Loss (OGA), IoU Quality Loss (IQA), Subcell Centroid Offset Loss, Footprint Size Loss, and Doubled-Angle Yaw Loss.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U152](rewrites.md#u152).

## M181 — U152, [sections/results.tex:137](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:137), câu 3

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 34; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Legend curves decode the architectural progression from baseline A1 (Binary35) through Strata-BEV (A2--A3), Decoupled GroupedHeader (A4--A5), PC-GTA augmentation (A6), SG-FPN neck (A7--A8), OGA and IQA supervision (A9), to the final peak configuration A10 (Block-SimAM).
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U152](rewrites.md#u152).

## M182 — U152, [sections/results.tex:137](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:137), câu 4

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 31; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: guarantees.
Cụm cấu trúc đã đọc theo ngữ cảnh: guarantees stable gradient optimization; lowest asymptotic validation error bounds.
Cụm thì cần sửa: không.

```latex
Dynamic multi-task loss weighting via T-SBUW guarantees stable gradient optimization without loss divergence or oscillation, with configuration A10 consistently achieving the lowest asymptotic validation error bounds across all continuous regression objectives.}
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U152](rewrites.md#u152).

## M183 — U153, [sections/results.tex:142](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:142), câu 1

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 48; bộ đếm dấu phẩy/từ nối: 7.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
To analyze empirical training stability and convergence behavior across the ablation sequence, Figure~\ref{fig:loss_curves} illustrates the training and validation loss trajectories across 50 epochs for the overall composite objective alongside the six constituent multi-task components (Classification, Geometry, IoU Quality, Subcell Offset, Size, and Yaw), tracking configurations A1 through A10.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U153](rewrites.md#u153).

## M184 — U154, [sections/results.tex:143](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:143), câu 3

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 32; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: demonstrating that; provides strong regularization without underfitting.
Cụm thì cần sửa: không.

```latex
The training and validation loss trajectories remained closely coupled throughout the optimization horizon without an expanding generalization gap, demonstrating that the physics-consistent point cloud augmentation (PC-GTA, A6) provides strong regularization without underfitting.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U154](rewrites.md#u154).

## M185 — U155, [sections/results.tex:144](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:144), câu 2

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 39; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: corroborating the quantitative gains.
Cụm thì cần sửa: không.

```latex
As depicted across individual loss subplots, embedding Block-SimAM directly inside residual blocks (configuration A10) consistently maintained lower training and validation loss bounds across all regression tasks compared with intermediate ablation baselines (A1--A9), corroborating the quantitative gains observed in Table~\ref{tab:components}.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U155](rewrites.md#u155).

## M186 — U158, [sections/results.tex:164](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:164), câu 3

Phân loại: LLM-contextual-pattern, tense-operation-or-finding.
Từ hiển thị: 21; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: Extended training sharpens.
Cụm thì cần sửa: Extended training sharpens.

```latex
Extended training sharpens classification heatmaps around true positive centers, suppressing ambiguous low-confidence background peaks and accelerating candidate filtering and rotated NMS.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U158](rewrites.md#u158).

## M187 — U165, [sections/results.tex:190](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:190), câu 1

Phân loại: LLM-contextual-pattern.
Từ hiển thị: 23; bộ đếm dấu phẩy/từ nối: 1.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: does not strictly necessitate.
Cụm thì cần sửa: không.

```latex
The empirical results indicate that high-accuracy 3D LiDAR object detection does not strictly necessitate parameter-heavy point-cloud encoders or computationally demanding 3D sparse convolutions.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U165](rewrites.md#u165).

## M188 — U165, [sections/results.tex:190](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:190), câu 2

Phân loại: vocabulary-context.
Từ hiển thị: 33; bộ đếm dấu phẩy/từ nối: 3.
Từ cần xem ngữ cảnh: streamlined.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
By enriching the deterministic 2D BEV representation with multi-band vertical densities, elevation dispersion moments, and continuous subcell centroid displacements, a streamlined ConvNeXt-style 2D convolutional network extracts expressive geometric features with fewer than $0.76\,\text{M}$ parameters.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U165](rewrites.md#u165).

## M189 — U166, [sections/results.tex:193](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:193), câu 1

Phân loại: vocabulary-context, LLM-contextual-pattern.
Từ hiển thị: 31; bộ đếm dấu phẩy/từ nối: 2.
Từ cần xem ngữ cảnh: optimal.
Cụm cấu trúc đã đọc theo ngữ cảnh: the optimal perception trade-off.
Cụm thì cần sửa: không.

```latex
The current MobilePIXORNeXt architecture operates primarily on oriented 2D Bird's-Eye View footprints ($x, y, l, w, \psi$), which represents the optimal perception trade-off for autonomous vehicle motion planning and collision avoidance under strict edge-computing power envelopes.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U166](rewrites.md#u166).

## M190 — U174, [tables/efficiency.tex:3](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/efficiency.tex:3), câu 4

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 32; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
Model forward latency measures neural network execution alone; End-to-End (E2E) pipeline latency spans raw LiDAR file I/O, statistical rasterization, Host-to-Device (H2D) transfer, network inference, subcell coordinate decoding, and class-wise rotated NMS.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U174](rewrites.md#u174).

## M191 — U175, [tables/efficiency.tex:23](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/efficiency.tex:23), câu 4

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 27; bộ đếm dấu phẩy/từ nối: 5.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
End-to-end (E2E) pipeline latency spans raw LiDAR point cloud loading, deterministic statistical rasterization, Host-to-Device (H2D) memory transfer, forward propagation, subcell coordinate decoding, and class-wise rotated NMS filtering.
```

**Xử lý:** Bản viết lại và lý do chi tiết nằm trong rewrites.md. [Đối chiếu U175](rewrites.md#u175).

## M192 — U177, [tables/loss.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/loss.tex:19), câu 2

Phân loại: comma-marker-proxy>3.
Từ hiển thị: 17; bộ đếm dấu phẩy/từ nối: 4.
Từ cần xem ngữ cảnh: không.
Cụm cấu trúc đã đọc theo ngữ cảnh: không.
Cụm thì cần sửa: không.

```latex
All variants share identical MobilePixorNeXt backbone, Strata-BEV14 input, Grouped H32 heads, F32 SG-FPN neck, and PC-GTA augmentation.
```

**Xử lý:** Giữ nguyên: cần đọc theo vai trò định nghĩa, danh sách hoặc mô tả hình; không tự suy ra lỗi từ bộ đếm dấu phẩy. [Đối chiếu U177](rewrites.md#u177).
