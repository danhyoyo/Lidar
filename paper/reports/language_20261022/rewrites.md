# Bản sửa văn phong và ngữ pháp LaTeX

107 đơn vị văn xuôi/chú thích trong 13 file. Một đơn vị có thể chứa nhiều câu. Dòng nguồn được giữ ổn định.

Mỗi mục dưới đây ghi nguyên văn trước và sau. `Meaning-Check: PRESERVED` là đánh giá biên tập về nội dung kỹ thuật, cần tác giả đọc lại; bảo toàn token được kiểm tra riêng bằng máy. Các mục có `Risk-Flags: overstatement` đã hạ mức khẳng định nhân quả/phổ quát, không thêm bằng chứng. Xem [over-claim guard](/home/duyennh/.agents/skills/latex-paper-en/references/evidence/over-claim-guard.md) và [reporting-verb ladder](/home/duyennh/.agents/skills/latex-paper-en/references/writing/style-guide.md).

<a id="u001"></a>

## U001 — [sections/abstract.tex:1](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/abstract.tex:1)

**Trước**

```latex
Three-dimensional object detection from Light Detection and Ranging (LiDAR) point clouds is critical for autonomous navigation; however, conventional volumetric and pillar-based architectures impose severe computational challenges on edge accelerators. We propose MobilePIXORNeXt, an ultra-compact, proposal-free Bird's-Eye View (BEV) detector operating under 753,569 parameters. The framework replaces learned point encoders with Strata-BEV14, a parameter-free deterministic NumPy/Numba CPU rasterizer that extracts 14 statistical feature channels---capturing stratified height distributions, elevation moments, surface reflectance, and continuous subcell centroid displacement offsets---with a measured input preprocessing time of 20.94 milliseconds, including point-cloud loading and rasterization. Spatial features are extracted by a single-branch ConvNeXt-style backbone with embedded parameter-free 3D spatial-energy attention (Block-SimAM), fused through a Scale-Gated Feature Pyramid Network, and predicted by decoupled grouped heads. Training is supervised by Physics-Consistent Ground-Truth Augmentation and Oriented Geometric Alignment, dynamically balanced via 12-parameter Temperature-Softmax Bounded Uncertainty Weighting. On the KITTI validation set using BEV AP $R_{40}$, MobilePIXORNeXt achieves 71.46\% Moderate mean Average Precision (mAP) and 73.42\% overall 9-cell mAP across Car, Pedestrian, and Cyclist categories. It outperforms PointPillars (70.13\% mAP) with an 84.4\% parameter reduction and matches SECOND (72.09\% mAP) with 85.8\% fewer parameters, while reaching 60.75\% Moderate AP on pedestrians. The complete pipeline executes in 60.96 milliseconds (16.41 frames per second) on an NVIDIA Tesla T4 GPU, delivering the fastest end-to-end throughput among evaluated real-time detectors. Our findings demonstrate that deterministic statistical encoding and oriented geometric supervision enable high-accuracy LiDAR perception on resource-constrained embedded systems.
```

**Sau — đã áp dụng**

```latex
Three-dimensional object detection from Light Detection and Ranging (LiDAR) point clouds supports autonomous navigation, but volumetric and pillar-based architectures can exceed the computational budgets of edge accelerators. We developed MobilePIXORNeXt, a compact, proposal-free Bird's-Eye View (BEV) detector with 753,569 parameters. The framework replaced learned point encoders with Strata-BEV14, a parameter-free deterministic NumPy/Numba CPU rasterizer. It extracted 14 statistical feature channels encoding stratified height distributions, elevation moments, surface reflectance, and continuous subcell centroid displacement offsets. Measured input preprocessing took 20.94 milliseconds, including point-cloud loading and rasterization. A single-branch ConvNeXt-style backbone with parameter-free 3D spatial-energy attention (Block-SimAM) extracted spatial features. A Scale-Gated Feature Pyramid Network fused these features for decoupled grouped prediction heads. Training used Physics-Consistent Ground-Truth Augmentation and Oriented Geometric Alignment, balanced by 12-parameter Temperature-Softmax Bounded Uncertainty Weighting. On the KITTI validation set using BEV AP $R_{40}$, MobilePIXORNeXt achieved 71.46\% Moderate mean Average Precision (mAP) and 73.42\% overall 9-cell mAP across Car, Pedestrian, and Cyclist categories. It outperformed PointPillars (70.13\% mAP) with an 84.4\% parameter reduction and matched SECOND (72.09\% mAP) with 85.8\% fewer parameters. Pedestrian Moderate AP reached 60.75\%. The complete pipeline ran in 60.96 milliseconds (16.41 frames per second) on an NVIDIA Tesla T4 GPU, the fastest end-to-end throughput among the evaluated real-time detectors. These results support deterministic statistical encoding and oriented geometric supervision for accurate LiDAR perception on resource-constrained embedded systems.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; tách hoặc rút gọn câu nhiều ý; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: 753,569; Strata-BEV14; 14; 20.94; 3D; 12-parameter; $R_{40}$; 71.46; 73.42; 9-cell; 70.13; 84.4; 72.09; 85.8; 60.75; 60.96; 16.41; T4
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u002"></a>

## U002 — [sections/conclusion.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:5)

**Trước**

```latex
In this work, we presented MobilePIXORNeXt, an ultra-compact, proposal-free 3D LiDAR object detector designed for real-time edge perception on resource-constrained platforms. By introducing the deterministic Strata-BEV14 rasterizer, the architecture eliminates computationally intensive PointNet pillar encoders and 3D sparse convolutions, preserving continuous vertical distribution moments, material surface reflectivity, and horizontal subcell centroid displacements with zero learned parameters. Multi-scale spatial features are processed via a single-branch ConvNeXt-style backbone hierarchy with embedded parameter-free Block-SimAM 3D energy attention, unified by a Scale-Gated Feature Pyramid Network into decoupled grouped prediction towers. Training is governed by Physics-Consistent Ground-Truth Augmentation (PC-GTA) alongside Oriented Geometric Alignment (OGA) and IoU-Aware Quality Alignment (IQA), dynamically balanced via 12-parameter Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW).
```

**Sau — đã áp dụng**

```latex
We presented MobilePIXORNeXt, a compact, proposal-free 3D LiDAR object detector for real-time edge perception on resource-constrained platforms. The deterministic Strata-BEV14 rasterizer replaced learned PointNet pillar encoders and 3D sparse convolutions. It preserved continuous vertical distribution moments, material surface reflectivity, and horizontal subcell centroid displacements without learned parameters. A single-branch ConvNeXt-style backbone with parameter-free Block-SimAM 3D energy attention processed multi-scale spatial features. A Scale-Gated Feature Pyramid Network fused these features for decoupled grouped prediction towers. Training used Physics-Consistent Ground-Truth Augmentation (PC-GTA), Oriented Geometric Alignment (OGA), and IoU-Aware Quality Alignment (IQA), balanced by 12-parameter Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW).
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; tách hoặc rút gọn câu nhiều ý
Protected: 3D; Strata-BEV14; 3D; 3D; 12-parameter
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u003"></a>

## U003 — [sections/conclusion.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:7)

**Trước**

```latex
Evaluations on the KITTI validation benchmark using Moderate BEV AP $R_{40}$ demonstrate that MobilePIXORNeXt establishes a leading Pareto trade-off between detection accuracy, model compactness, and operational latency. Containing exactly $753{,}569$ parameters ($9.34\,\text{MB}$ checkpoint), the model achieves $71.46\%$ Moderate mAP and $73.42\%$ $\text{mAP}_9$ across Car, Pedestrian, and Cyclist. These results surpass PointPillars by $1.33\%$ Moderate mAP with an $84.4\%$ parameter reduction, while matching the detection performance of SECOND ($72.09\%$ mAP) with an $85.8\%$ parameter reduction ($7.1\times$ fewer parameters). On vulnerable road users, the network achieves $60.75\%$ Moderate AP on pedestrians, outperforming PointPillars by $4.74\%$ and SECOND by $4.19\%$. In full-system benchmarking on an NVIDIA Tesla T4 GPU, MobilePIXORNeXt executes the complete input-to-detection pipeline in $60.96\,\text{ms}$ ($16.41\,\text{FPS}$), achieving the lowest operational latency among evaluated real-time detectors. Controlled ablation experiments confirm that PC-GTA yields a $+6.03\%$ mAP improvement by mitigating unphysical synthesis artifacts, while OGA and IQA deliver an additional $+3.20\%$ mAP gain through rotated geometric overlap alignment and confidence score calibration.
```

**Sau — đã áp dụng**

```latex
Evaluation on the KITTI validation benchmark using Moderate BEV AP $R_{40}$ showed a favorable trade-off between detection accuracy, model size, and latency. With $753{,}569$ parameters ($9.34\,\text{MB}$ checkpoint), MobilePIXORNeXt achieved $71.46\%$ Moderate mAP and $73.42\%$ $\text{mAP}_9$ across Car, Pedestrian, and Cyclist. It exceeded PointPillars by $1.33\%$ Moderate mAP with an $84.4\%$ parameter reduction and matched SECOND ($72.09\%$ mAP) with an $85.8\%$ parameter reduction ($7.1\times$ fewer parameters). Pedestrian Moderate AP reached $60.75\%$, exceeding PointPillars by $4.74\%$ and SECOND by $4.19\%$. On an NVIDIA Tesla T4 GPU, MobilePIXORNeXt's complete input-to-detection pipeline ran in $60.96\,\text{ms}$ ($16.41\,\text{FPS}$), the lowest latency among the evaluated real-time detectors. In controlled ablations, PC-GTA improved mAP by $+6.03\%$, consistent with its intended reduction of unphysical synthesis artifacts. OGA and IQA added $+3.20\%$ mAP through rotated geometric overlap alignment and confidence score calibration.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: $R_{40}$; $753{,}569$; $9.34\,\text{MB}$; $71.46\%$; $73.42\%$; $\text{mAP}_9$; $1.33\%$; $84.4\%$; $72.09\%$; $85.8\%$; $7.1\times$; $60.75\%$; $4.74\%$; $4.19\%$; T4; $60.96\,\text{ms}$; $16.41\,\text{FPS}$; $+6.03\%$; $+3.20\%$
Meaning-Check: PRESERVED
Risk-Flags: overstatement
```

<a id="u004"></a>

## U004 — [sections/conclusion.tex:9](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/conclusion.tex:9)

**Trước**

```latex
Despite these advantages, several practical limitations outline directions for future research. First, the primary detector estimates oriented 2D Bird's-Eye View bounding boxes. While vertical elevation and height regression are supported via an auxiliary branch, localization fidelity on steep grades, multi-level interchanges, and irregular non-planar terrains requires further investigation on dedicated off-road benchmarks. Second, the current model was benchmarked on 64-beam mechanical rotating LiDAR scans under fair-weather driving conditions; adapting the framework to ultra-sparse solid-state sensors (e.g., 16-beam or non-repetitive scanning patterns) and adverse precipitation remains an open challenge. Future work will investigate range-conditioned scale gating (RC-SGFPN) with continuous Fourier frequency embeddings to adapt to steep radial point density drop-offs, alongside hardware-aligned INT8 quantization for ultra-low-power micro-mobility microcontrollers.
```

**Sau — đã áp dụng**

```latex
Several limitations remain. The primary detector estimates oriented 2D Bird's-Eye View bounding boxes. An auxiliary branch supports vertical elevation and height regression, but localization on steep grades, multi-level interchanges, and irregular non-planar terrain requires evaluation on dedicated off-road benchmarks. The current evaluation used 64-beam mechanical rotating LiDAR scans under fair-weather driving conditions. Adaptation to ultra-sparse solid-state sensors (e.g., 16-beam or non-repetitive scanning patterns) and adverse precipitation remains an open challenge. Future work will investigate range-conditioned scale gating (RC-SGFPN) with continuous Fourier frequency embeddings to address radial point density drop-offs, and hardware-aligned INT8 quantization for low-power micro-mobility microcontrollers.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý; bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể
Protected: 2D; 64-beam; 16-beam; INT8
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u005"></a>

## U005 — [sections/declarations.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/declarations.tex:5)

**Trước**

```latex
The authors would like to acknowledge FPT University, Vietnam, for providing computational infrastructure, GPU workstation clusters, and laboratory facilities supporting this research.
```

**Sau — đã áp dụng**

```latex
The authors acknowledge FPT University, Vietnam, for providing the computational infrastructure, GPU workstation clusters, and laboratory facilities used in this research.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: Không có math/cite/ref/label/số trong đoạn; giữ nguyên thuật ngữ và lệnh LaTeX.
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u012"></a>

## U012 — [sections/declarations.tex:29](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/declarations.tex:29)

**Trước**

```latex
All authors contributed significantly to the study conception, architecture design, and empirical evaluation. 
```

**Sau — đã áp dụng**

```latex
All authors contributed to the study conception, architecture design, and empirical evaluation.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể
Protected: Không có math/cite/ref/label/số trong đoạn; giữ nguyên thuật ngữ và lệnh LaTeX.
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u028"></a>

## U028 — [sections/introduction.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:5)

**Trước**

```latex
Three-dimensional (3D) spatial perception from Light Detection and Ranging (LiDAR) point clouds provides essential sensory grounding for safe autonomous navigation, mobile robotics, and intelligent transportation systems. Unlike passive optical cameras, which remain vulnerable to ambient illumination fluctuations and projective scale ambiguities, LiDAR sensors actively emit pulsed laser beams to measure direct metric ranges, capturing high-precision geometric boundaries under arbitrary day-and-night conditions~\cite{geiger2012kitti}. In time-critical vehicular environments, perception systems must accurately detect, orient, and classify surrounding objects (from passenger vehicles to compact vulnerable road users such as pedestrians and cyclists) within strict latency envelopes, typically requiring sub-$100\,\text{ms}$ execution per scan~\cite{yang2018pixor}. However, deploying high-accuracy 3D object detection models on embedded vehicular platforms faces an acute operational tension between detection accuracy and hardware resource consumption. Commodity onboard edge accelerators operate under stringent thermal and electrical power budgets, where memory bandwidth, cache capacity, and arithmetic throughput are strictly constrained. Consequently, developing ultra-compact yet highly discriminative LiDAR object detection architectures that sustain real-time execution without forfeiting spatial localization fidelity remains an essential prerequisite for scalable autonomous driving.
```

**Sau — đã áp dụng**

```latex
Three-dimensional (3D) spatial perception from Light Detection and Ranging (LiDAR) point clouds supports autonomous navigation, mobile robotics, and intelligent transportation systems. Passive optical cameras are sensitive to ambient illumination and projective scale ambiguity. LiDAR sensors instead emit laser pulses to measure metric ranges and capture geometric boundaries during both day and night~\cite{geiger2012kitti}. Vehicle perception systems must detect, orient, and classify nearby objects, including passenger vehicles, pedestrians, and cyclists, within strict latency limits. Execution typically must remain below $100\,\text{ms}$ per scan~\cite{yang2018pixor}. Deploying accurate 3D object detectors on embedded vehicle platforms therefore requires balancing detection accuracy against hardware resource use. Onboard edge accelerators have limited memory bandwidth, cache capacity, and arithmetic throughput under thermal and power constraints. Compact LiDAR detectors must sustain real-time execution while preserving localization accuracy.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý
Protected: 3D; \cite{geiger2012kitti}; $100\,\text{ms}$; \cite{yang2018pixor}; 3D
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u029"></a>

## U029 — [sections/introduction.tex:7](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:7)

**Trước**

```latex
To extract 3D bounding boxes from unstructured point clouds, existing detection paradigms fall into three primary methodological families, each exhibiting inherent architectural trade-offs on resource-constrained platforms. First, volumetric voxel-based architectures, exemplified by SECOND~\cite{yan2018second}, and two-stage point-voxel frameworks, such as PointRCNN~\cite{shi2019pointrcnn}, Part-$A^2$~\cite{shi2020part}, and PV-RCNN~\cite{shi2020pvrcnn}, achieve competitive detection accuracy by deploying 3D sparse convolutions or localized point-cloud abstractions. Nevertheless, their multi-million parameter footprints ($4.0\text{--}63.8\,\text{M}$) and heavy memory access latencies ($100\text{--}195\,\text{ms}$) exceed the operational limits of embedded edge processors. Second, pillar-based detectors, pioneered by PointPillars~\cite{lang2019pointpillars} and modernized by PillarNeXt~\cite{li2023pillarnext} and PillarNeSt~\cite{mao2023pillarnest}, project points into vertical pillars processed via PointNet encoders to construct Bird's-Eye View (BEV) 2D feature representations. While faster than volumetric methods, non-deterministic point gathering, dynamic memory allocation, and sparse scattering introduce substantial runtime overhead and memory transfer latency. Third, proposal-free BEV rasterization methods, such as PIXOR~\cite{yang2018pixor} and MobilePIXOR~\cite{nguyen2024indoor}, eliminate learned encoders entirely by discretizing point clouds into fixed vertical occupancy grids. Although conceptually efficient, existing rasterization methods suffer from severe spatial quantization errors on standard $0.1\,\text{m}$ grids; by relying on crude binary occupancy slices, they discard continuous subcell offsets and high-order vertical dispersion statistics, inducing spatial staircasing and localization jitter on small objects. While recent studies such as PillarHist~\cite{zhou2025pillarhist} explore height histograms and TriBand-BEV~\cite{khoshkdahan2026triband} explores multi-band vertical slicing, they still omit continuous subcell displacements combined with continuous vertical spatial moments. In addition, naive cut-and-paste data augmentation creates physical artifacts, small-scale backbones experience receptive-field starvation across disparate object dimensions, and uncalibrated classification heatmaps cause misaligned false positives to suppress well-localized candidates during Non-Maximum Suppression (NMS)~\cite{li2020gfl}. The failure of existing compact detectors to simultaneously resolve fine subcell quantization errors, unphysical augmentation artifacts, multi-scale receptive field coverage, and geometric score misalignment constitutes the core bottleneck impeding sub-million-parameter LiDAR perception.
```

**Sau — đã áp dụng**

```latex
Existing detectors extract 3D bounding boxes from unstructured point clouds using three main approaches. Volumetric voxel architectures such as SECOND~\cite{yan2018second} and two-stage point-voxel frameworks such as PointRCNN~\cite{shi2019pointrcnn}, Part-$A^2$~\cite{shi2020part}, and PV-RCNN~\cite{shi2020pvrcnn} use 3D sparse convolutions or local point-cloud abstractions. These methods achieve competitive accuracy, but their parameter counts ($4.0\text{--}63.8\,\text{M}$) and memory access latencies ($100\text{--}195\,\text{ms}$) limit deployment on embedded edge processors. Pillar-based detectors include PointPillars~\cite{lang2019pointpillars}, PillarNeXt~\cite{li2023pillarnext}, and PillarNeSt~\cite{mao2023pillarnest}. They project points into vertical pillars and apply PointNet encoders to construct Bird's-Eye View (BEV) 2D feature representations. They are faster than volumetric methods, although point gathering, dynamic memory allocation, and sparse scattering add runtime and memory transfer overhead. Proposal-free BEV methods such as PIXOR~\cite{yang2018pixor} and MobilePIXOR~\cite{nguyen2024indoor} replace learned encoders with fixed vertical occupancy grids. Binary occupancy slices on standard $0.1\,\text{m}$ grids omit continuous subcell offsets and higher-order vertical dispersion statistics. This quantization can cause boundary staircasing and localization jitter on small objects. PillarHist~\cite{zhou2025pillarhist} explored height histograms, and TriBand-BEV~\cite{khoshkdahan2026triband} explored multi-band vertical slicing. Neither combines continuous subcell displacements with continuous vertical spatial moments. Other limitations include physical artifacts from naive cut-and-paste augmentation and insufficient receptive fields in small backbones. Uncalibrated classification heatmaps can also allow misaligned boxes to suppress well-localized candidates during Non-Maximum Suppression (NMS)~\cite{li2020gfl}. Compact detectors must address these representation, augmentation, context, and ranking limitations within a sub-million-parameter budget.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: 3D; \cite{yan2018second}; \cite{shi2019pointrcnn}; $A^2$; \cite{shi2020part}; \cite{shi2020pvrcnn}; 3D; $4.0\text{--}63.8\,\text{M}$; $100\text{--}195\,\text{ms}$; \cite{lang2019pointpillars}; \cite{li2023pillarnext}; \cite{mao2023pillarnest}; 2D; \cite{yang2018pixor}; \cite{nguyen2024indoor}; $0.1\,\text{m}$; \cite{zhou2025pillarhist}; \cite{khoshkdahan2026triband}; \cite{li2020gfl}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u030"></a>

## U030 — [sections/introduction.tex:9](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:9)

**Trước**

```latex
We address this fundamental challenge through a key scientific insight: sub-voxel geometric precision can be preserved deterministically at the representation level by coupling multi-band vertical densities and continuous elevation moments with horizontal subcell centroid displacements, bypassing learned PointNet encoders entirely without forfeiting fine spatial boundaries. Grounded in this principle, we investigate the central research question: \textit{Can an ultra-compact LiDAR detector operating on a parameter-free BEV tensor rival the localization accuracy of heavy volumetric and pillar architectures while achieving superior end-to-end throughput on commodity edge hardware?} To realize this objective, our framework couples this deterministic representation with a single-branch ConvNeXt-style backbone~\cite{liu2022convnext} utilizing large $7 \times 7$ depthwise kernels and parameter-free 3D spatial-energy attention (SimAM)~\cite{yang2021simam}, expanding physical receptive fields to envelop entire vehicles without multi-branch training overhead. Concurrently, physics-consistent data synthesis regularizes distant returns, while multi-axis geometric supervision and localization-quality ranking align backpropagation gradients directly with rotated bounding box overlap. Together, these complementary mechanisms demonstrate that disciplined geometric encoding and calibrated multi-task loss formulation bridge the performance divide between compact 2D representations and multi-million-parameter 3D networks.
```

**Sau — đã áp dụng**

```latex
We investigated whether a deterministic representation could preserve sub-voxel geometry without learned PointNet encoders. The representation combines multi-band vertical densities, continuous elevation moments, and horizontal subcell centroid displacements to retain fine spatial boundaries. Our research question was: \textit{Can an ultra-compact LiDAR detector operating on a parameter-free BEV tensor rival the localization accuracy of heavy volumetric and pillar architectures while achieving superior end-to-end throughput on commodity edge hardware?} We coupled this representation with a single-branch ConvNeXt-style backbone~\cite{liu2022convnext} using large $7 \times 7$ depthwise kernels and parameter-free 3D spatial-energy attention (SimAM)~\cite{yang2021simam}. This design expands receptive fields to cover vehicles without multi-branch training overhead. Physics-consistent synthesis regularizes distant returns, while multi-axis geometric supervision and localization-quality ranking align training and detection scores with rotated box overlap. Together, these mechanisms aim to narrow the accuracy gap between compact 2D representations and multi-million-parameter 3D networks.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: \textit{Can an ultra-compact LiDAR detector operating on a parameter-free BEV tensor rival the localization accuracy of heavy volumetric and pillar architectures while achieving superior end-to-end throughput on commodity edge hardware?}; \cite{liu2022convnext}; $7 \times 7$; 3D; \cite{yang2021simam}; 2D; 3D
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u031"></a>

## U031 — [sections/introduction.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:11)

**Trước**

```latex
Guided by this design philosophy, we propose \textbf{MobilePIXORNeXt}, an ultra-compact, proposal-free, multi-class 3D LiDAR object detector containing exactly $753{,}569$ trainable parameters ($<0.76\,\text{M}$). Figure~\ref{fig:r37_overview} illustrates the end-to-end data flow of our framework. An unordered LiDAR point cloud is deterministically transformed via the parameter-free \textbf{Strata-BEV14} rasterizer into a 14-channel statistical BEV tensor ($800 \times 704$ at $0.1\,\text{m}$ resolution) using NumPy or Numba on CPU, with point-cloud loading and rasterization taking $20.94\,\text{ms}$ in the measured pipeline, capturing height-band densities, vertical spatial moments ($z_{\max}, z_{\mathrm{mean}}, z_{\mathrm{span}}, z_{\mathrm{std}}$), surface reflectance statistics, and continuous subcell centroid displacement offsets $(\Delta\bar{x}, \Delta\bar{y})$ to eliminate quantization error. To regularize training without unphysical artifacts, we formulate \textbf{Physics-Consistent Ground-Truth Augmentation (PC-GTA)}, which combines local ground elevation fitting, static obstacle clearance, and range-adaptive point subsampling, with optional line-of-sight checks, raycast shadow masking, and intensity attenuation. Multi-scale feature extraction is performed by the \textbf{MobilePixorNeXt-S} backbone hierarchy across stages $C_2, C_3, C_4$, utilizing streamlined $7 \times 7$ depthwise inverted bottleneck blocks ($e=2.5$), SiLU activations, LayerScale stabilization, and embedded parameter-free \textbf{Block-SimAM} energy attention operating inside all nine residual blocks. Multi-scale semantic context and fine coordinates are balanced by the top-down \textbf{Scale-Gated Feature Pyramid Network (SG-FPN Neck, $F_{32}$)}, which employs bilinear upsampling, depthwise refinement, and learnable scale gates to eliminate deconvolution checkerboard artifacts. Predictions are decoupled into Car ($K_1=1$) and Pedestrian/Cyclist ($K_2=2$) categories via independent \textbf{GroupedHeader} towers sharing an initial $3 \times 3$ Conv-BN-SiLU block. During training, the network is supervised by \textbf{Oriented Geometric Alignment (OGA)}---coupling Multi-Axis Projection Generalized IoU (MGIoU)~\cite{le2026mgiou} with scale-normalized $\pi$-symmetric corner distance---and dynamically balanced via 12-parameter \textbf{Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW)}. At inference, predicted boxes are ranked using composite scores from \textbf{IoU-Aware Quality Alignment (IQA)} ($s_{\mathrm{det}} = \sqrt{p_{\mathrm{cls}}\hat{q}}$) before class-wise rotated Non-Maximum Suppression (NMS). By deterministically capturing continuous geometry at the input, expanding receptive fields via single-branch large-kernel convolution, and supervising rotated overlap directly, this coherent pipeline resolves each aforementioned bottleneck within a unified, edge-deployable framework.
```

**Sau — đã áp dụng**

```latex
We propose \textbf{MobilePIXORNeXt}, a compact, proposal-free, multi-class 3D LiDAR object detector with $753{,}569$ trainable parameters ($<0.76\,\text{M}$). Figure~\ref{fig:r37_overview} shows the end-to-end data flow. The parameter-free \textbf{Strata-BEV14} rasterizer transforms an unordered LiDAR point cloud into a 14-channel statistical BEV tensor ($800 \times 704$ at $0.1\,\text{m}$ resolution) using NumPy or Numba on CPU. Point-cloud loading and rasterization took $20.94\,\text{ms}$ in the measured pipeline. The tensor encodes height-band densities, vertical spatial moments ($z_{\max}, z_{\mathrm{mean}}, z_{\mathrm{span}}, z_{\mathrm{std}}$), surface reflectance statistics, and continuous subcell centroid displacement offsets $(\Delta\bar{x}, \Delta\bar{y})$ to reduce quantization error. \textbf{Physics-Consistent Ground-Truth Augmentation (PC-GTA)} combines local ground elevation fitting, static obstacle clearance, and range-adaptive point subsampling to reduce unphysical training artifacts. It also supports optional line-of-sight checks, raycast shadow masking, and intensity attenuation. The \textbf{MobilePixorNeXt-S} backbone extracts multi-scale features across stages $C_2, C_3, C_4$. Its $7 \times 7$ depthwise inverted bottleneck blocks ($e=2.5$) use SiLU activations, LayerScale stabilization, and parameter-free \textbf{Block-SimAM} energy attention in all nine residual blocks. The top-down \textbf{Scale-Gated Feature Pyramid Network (SG-FPN Neck, $F_{32}$)} balances semantic context and fine coordinates through bilinear upsampling, depthwise refinement, and learnable scale gates, avoiding deconvolution checkerboard artifacts. Car ($K_1=1$) and Pedestrian/Cyclist ($K_2=2$) predictions use independent \textbf{GroupedHeader} towers with an initial $3 \times 3$ Conv-BN-SiLU block. Training uses \textbf{Oriented Geometric Alignment (OGA)}, which combines Multi-Axis Projection Generalized IoU (MGIoU)~\cite{le2026mgiou} with scale-normalized $\pi$-symmetric corner distance. 12-parameter \textbf{Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW)} balances the task losses. At inference, \textbf{IoU-Aware Quality Alignment (IQA)} combines scores as $s_{\mathrm{det}} = \sqrt{p_{\mathrm{cls}}\hat{q}}$ before class-wise rotated Non-Maximum Suppression (NMS). The pipeline combines continuous input geometry, single-branch large-kernel convolution, and direct supervision of rotated overlap for edge deployment.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: \textbf{MobilePIXORNeXt}; 3D; $753{,}569$; $<0.76\,\text{M}$; \ref{fig:r37_overview}; \textbf{Strata-BEV14}; 14-channel; $800 \times 704$; $0.1\,\text{m}$; $20.94\,\text{ms}$; $z_{\max}, z_{\mathrm{mean}}, z_{\mathrm{span}}, z_{\mathrm{std}}$; $(\Delta\bar{x}, \Delta\bar{y})$; \textbf{Physics-Consistent Ground-Truth Augmentation (PC-GTA)}; \textbf{MobilePixorNeXt-S}; $C_2, C_3, C_4$; $7 \times 7$; $e=2.5$; \textbf{Block-SimAM}; \textbf{Scale-Gated Feature Pyramid Network (SG-FPN Neck, $F_{32}$)}; $K_1=1$; $K_2=2$; \textbf{GroupedHeader}; $3 \times 3$; \textbf{Oriented Geometric Alignment (OGA)}; \cite{le2026mgiou}; $\pi$; 12-parameter; \textbf{Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW)}; \textbf{IoU-Aware Quality Alignment (IQA)}; $s_{\mathrm{det}} = \sqrt{p_{\mathrm{cls}}\hat{q}}$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u032"></a>

## U032 — [sections/introduction.tex:15](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:15)

**Trước**

```latex
    \item \textbf{Ultra-compact proposal-free BEV detection.} We develop \textbf{MobilePIXORNeXt}, an ultra-compact ($753{,}569$ trainable parameters) LiDAR detector that eliminates learned PointNet encoders and 3D sparse convolutions. The architecture integrates the deterministic 14-channel \textbf{Strata-BEV14} representation, the streamlined \textbf{MobilePixorNeXt-S} backbone hierarchy with embedded parameter-free \textbf{Block-SimAM} energy attention, the \textbf{SG-FPN} neck ($F_{32}$), and decoupled \textbf{GroupedHeader} towers to balance multi-scale road categories.
```

**Sau — đã áp dụng**

```latex
    \item \textbf{Ultra-compact proposal-free BEV detection.} We develop \textbf{MobilePIXORNeXt}, a compact ($753{,}569$ trainable parameters) LiDAR detector without learned PointNet encoders or 3D sparse convolutions. It combines the deterministic 14-channel \textbf{Strata-BEV14} representation with the \textbf{MobilePixorNeXt-S} backbone and parameter-free \textbf{Block-SimAM} energy attention. The \textbf{SG-FPN} neck ($F_{32}$) fuses features for decoupled \textbf{GroupedHeader} towers to handle road categories at different scales.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; tách hoặc rút gọn câu nhiều ý
Protected: \textbf{Ultra-compact proposal-free BEV detection.}; \textbf{MobilePIXORNeXt}; $753{,}569$; 3D; 14-channel; \textbf{Strata-BEV14}; \textbf{MobilePixorNeXt-S}; \textbf{Block-SimAM}; \textbf{SG-FPN}; $F_{32}$; \textbf{GroupedHeader}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u033"></a>

## U033 — [sections/introduction.tex:17](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:17)

**Trước**

```latex
    \item \textbf{Physics-consistent data augmentation and oriented geometric supervision.} We formulate \textbf{PC-GTA}, which combines local terrain alignment, obstacle clearance, and range-dependent point subsampling, and supports optional visibility, shadow-masking, and intensity-adjustment heuristics to reduce synthesis artifacts. In conjunction, we develop the \textbf{OGA} loss strategy, combining Multi-Axis Projection GIoU with scale-normalized $\pi$-symmetric corner supervision, dynamically balanced across multi-task objectives via 12-parameter \textbf{T-SBUW} and coupled with \textbf{IQA} localization-quality confidence calibration.
```

**Sau — đã áp dụng**

```latex
    \item \textbf{Physics-consistent data augmentation and oriented geometric supervision.} We formulate \textbf{PC-GTA} to combine local terrain alignment, obstacle clearance, and range-dependent point subsampling. Optional visibility, shadow-masking, and intensity-adjustment heuristics address additional synthesis artifacts. We also develop \textbf{OGA}, combining Multi-Axis Projection GIoU with scale-normalized $\pi$-symmetric corner supervision. 12-parameter \textbf{T-SBUW} balances task objectives, and \textbf{IQA} calibrates confidence against localization quality.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý
Protected: \textbf{Physics-consistent data augmentation and oriented geometric supervision.}; \textbf{PC-GTA}; \textbf{OGA}; $\pi$; 12-parameter; \textbf{T-SBUW}; \textbf{IQA}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u034"></a>

## U034 — [sections/introduction.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:19)

**Trước**

```latex
    \item \textbf{Empirical validation and leading edge Pareto efficiency.} Evaluated using BEV AP $R_{40}$ on the KITTI validation set, MobilePIXORNeXt establishes a leading accuracy-to-parameter and accuracy-to-latency trade-off. It achieves \textbf{71.46\% Moderate mAP} and \textbf{73.42\% $\text{mAP}_9$} with only \textbf{0.754\,M parameters} ($9.34\,\text{MB}$ checkpoint), outperforming PointPillars ($70.13\%$) with an \textbf{84.4\% parameter reduction} and matching the accuracy of SECOND ($72.09\%$) while requiring \textbf{85.8\% fewer parameters} ($7.1\times$ parameter reduction). In complete end-to-end pipeline speed, it operates at \textbf{16.41\,FPS} (\textbf{60.96\,ms} on NVIDIA Tesla T4), running faster than all evaluated real-time detectors. On vulnerable road users, it attains \textbf{60.75\% Moderate AP} on Pedestrian (+4.74\% over PointPillars, +4.19\% over SECOND). Controlled ablations demonstrate that PC-GTA yields a decisive \textbf{+6.03\% Moderate mAP} gain ($60.22\%$ to $66.25\%$), OGA and IQA deliver an additional \textbf{+3.20\% mAP} gain, and Block-SimAM provides \textbf{+0.88\% mAP} at zero parameter overhead.
```

**Sau — đã áp dụng**

```latex
    \item \textbf{Empirical validation and leading edge Pareto efficiency.} On KITTI validation using BEV AP $R_{40}$, MobilePIXORNeXt achieved a favorable accuracy, parameter, and latency trade-off. It reached \textbf{71.46\% Moderate mAP} and \textbf{73.42\% $\text{mAP}_9$} with \textbf{0.754\,M parameters} ($9.34\,\text{MB}$ checkpoint). It outperformed PointPillars ($70.13\%$) with an \textbf{84.4\% parameter reduction} and matched SECOND ($72.09\%$) while requiring \textbf{85.8\% fewer parameters} ($7.1\times$ parameter reduction). The complete pipeline ran at \textbf{16.41\,FPS} (\textbf{60.96\,ms} on NVIDIA Tesla T4), faster than the other evaluated real-time detectors. Pedestrian AP reached \textbf{60.75\% Moderate AP} (+4.74\% over PointPillars, +4.19\% over SECOND). In controlled ablations, PC-GTA added \textbf{+6.03\% Moderate mAP} ($60.22\%$ to $66.25\%$), OGA and IQA added \textbf{+3.20\% mAP}, and Block-SimAM added \textbf{+0.88\% mAP} without additional parameters.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: \textbf{Empirical validation and leading edge Pareto efficiency.}; $R_{40}$; \textbf{71.46\% Moderate mAP}; \textbf{73.42\% $\text{mAP}_9$}; \textbf{0.754\,M parameters}; $9.34\,\text{MB}$; $70.13\%$; \textbf{84.4\% parameter reduction}; $72.09\%$; \textbf{85.8\% fewer parameters}; $7.1\times$; \textbf{16.41\,FPS}; \textbf{60.96\,ms}; T4; \textbf{60.75\% Moderate AP}; 4.74; 4.19; \textbf{+6.03\% Moderate mAP}; $60.22\%$; $66.25\%$; \textbf{+3.20\% mAP}; \textbf{+0.88\% mAP}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u035"></a>

## U035 — [sections/introduction.tex:22](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/introduction.tex:22)

**Trước**

```latex
The remainder of this paper is structured as follows. Section~\ref{sec:related_work} reviews related literature in LiDAR representation, lightweight backbones, and geometric loss formulations. Section~\ref{sec:method} presents the mathematical foundations of the proposed MobilePIXORNeXt framework. Section~\ref{sec:experiments} outlines the benchmark experimental setup and implementation details. Section~\ref{sec:results} presents quantitative comparisons, latency profiles, ablation studies, and qualitative analysis. Finally, Section~\ref{sec:conclusion} concludes the paper.
```

**Sau — đã áp dụng**

```latex
Section~\ref{sec:related_work} reviews LiDAR representations, lightweight backbones, and geometric losses. Section~\ref{sec:method} describes MobilePIXORNeXt. Section~\ref{sec:experiments} specifies the experimental setup and implementation. Section~\ref{sec:results} reports quantitative comparisons, latency profiles, ablations, and qualitative results. Section~\ref{sec:conclusion} concludes the paper.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \ref{sec:related_work}; \ref{sec:method}; \ref{sec:experiments}; \ref{sec:results}; \ref{sec:conclusion}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u113"></a>

## U113 — [sections/related_work.tex:4](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:4)

**Trước**

```latex
Dense horizontal projection provides an efficient pathway to apply high-throughput 2D convolutional networks to unstructured 3D LiDAR point clouds. PIXOR~\cite{yang2018pixor} established the foundational paradigm of proposal-free Bird's-Eye View (BEV) detection by discretizing point clouds into fixed vertical occupancy slices and calibrated reflectance maps. Building upon this formulation, Nguyen et al.~\cite{nguyen2024indoor} introduced MobilePIXOR, replacing heavyweight convolutional layers with inverted residual blocks and Gaussian heatmap supervision for indoor robotic perception. In parallel, PointPillars~\cite{lang2019pointpillars} eliminated rigid vertical slicing by organizing point clouds into vertical columns (pillars) and learning spatial feature representations via localized PointNet encoders. Modern extensions such as PillarNeXt~\cite{li2023pillarnext} and PillarNeSt~\cite{mao2023pillarnest} systematically modernized pillar-based receptive field designs and backbone scaling for large-scale outdoor perception.
```

**Sau — đã áp dụng**

```latex
Dense horizontal projection enables 2D convolutional networks to process unstructured 3D LiDAR point clouds. PIXOR~\cite{yang2018pixor} introduced proposal-free Bird's-Eye View (BEV) detection using fixed vertical occupancy slices and calibrated reflectance maps. Nguyen et al.~\cite{nguyen2024indoor} developed MobilePIXOR for indoor robotic perception, replacing large convolutional layers with inverted residual blocks and Gaussian heatmap supervision. PointPillars~\cite{lang2019pointpillars} instead organized points into vertical columns (pillars) and learned spatial features through local PointNet encoders. PillarNeXt~\cite{li2023pillarnext} and PillarNeSt~\cite{mao2023pillarnest} extended pillar-based receptive field design and backbone scaling for outdoor perception.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể
Protected: 2D; 3D; \cite{yang2018pixor}; \cite{nguyen2024indoor}; \cite{lang2019pointpillars}; \cite{li2023pillarnext}; \cite{mao2023pillarnest}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u114"></a>

## U114 — [sections/related_work.tex:6](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:6)

**Trước**

```latex
A fundamental operational dichotomy divides existing BEV representations between expressive PointNet-based pillar feature encoders and high-speed fixed-grid rasterization. On one hand, PointNet-based pillar encoders~\cite{lang2019pointpillars,li2023pillarnext} achieve expressive geometric learning, yet the non-deterministic point gathering, dynamic memory allocation, and sparse scatter operations incur prohibitive latency and energy penalties on edge accelerators. On the other hand, fixed-grid rasterization methods~\cite{yang2018pixor,nguyen2024indoor} bypass point gathering entirely but suffer from severe spatial quantization error on standard $0.1\,\text{m}$ grids; by treating cells as binary flags or raw counters, they discard continuous subcell offsets and high-order vertical distributions, causing boundary staircasing and localization jitter. While recent studies such as PillarHist~\cite{zhou2025pillarhist} investigate quantization-aware height histograms and TriBand-BEV~\cite{khoshkdahan2026triband} explores multi-band vertical slicing to assist pedestrian detection, a shared limitation across these paradigms is the absence of continuous subcell centroid displacements coupled with continuous vertical statistical moments inside a compact, hardware-friendly tensor. MobilePIXORNeXt overcomes this limitation through the Strata-BEV14 representation (Section~\ref{sec:strata_bev14}). By deterministically integrating four stratified vertical density bins, four continuous elevation moments ($\mu_z, \sigma_z, z_{\min}, z_{\max}$), and two continuous subcell centroid displacement offsets ($\Delta x_c, \Delta y_c$) into a 14-channel layout, Strata-BEV14 preserves sub-voxel geometric fidelity while reducing Host-to-Device transfer payloads by $60\%$ without learned encoder latency.
```

**Sau — đã áp dụng**

```latex
BEV representations commonly use either learned PointNet-based pillar encoders or fixed-grid rasterization. Pillar encoders~\cite{lang2019pointpillars,li2023pillarnext} learn geometric features but require point gathering, dynamic memory allocation, and sparse scattering, which add latency and energy costs on edge accelerators. Fixed-grid methods~\cite{yang2018pixor,nguyen2024indoor} avoid learned point encoding. However, binary flags or raw cell counts on $0.1\,\text{m}$ grids omit continuous subcell offsets and higher-order vertical distributions, causing boundary staircasing and localization jitter. PillarHist~\cite{zhou2025pillarhist} investigated quantization-aware height histograms, and TriBand-BEV~\cite{khoshkdahan2026triband} explored multi-band vertical slicing for pedestrian detection. These methods do not combine continuous subcell centroid displacements with continuous vertical statistical moments in a compact tensor. MobilePIXORNeXt uses Strata-BEV14 (Section~\ref{sec:strata_bev14}) to combine four stratified density bins, four continuous elevation moments ($\mu_z, \sigma_z, z_{\min}, z_{\max}$), and two subcell centroid displacement offsets ($\Delta x_c, \Delta y_c$). This 14-channel layout preserves sub-voxel geometry. Strata-BEV14 also reduces Host-to-Device transfer payloads by $60\%$ without a learned encoder.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: \cite{lang2019pointpillars,li2023pillarnext}; \cite{yang2018pixor,nguyen2024indoor}; $0.1\,\text{m}$; \cite{zhou2025pillarhist}; \cite{khoshkdahan2026triband}; Strata-BEV14; \ref{sec:strata_bev14}; $\mu_z, \sigma_z, z_{\min}, z_{\max}$; $\Delta x_c, \Delta y_c$; 14-channel; Strata-BEV14; $60\%$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u115"></a>

## U115 — [sections/related_work.tex:11](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:11)

**Trước**

```latex
Extracting discriminative spatial semantics from sparse BEV feature planes requires balancing expressive receptive fields with strict parameter and latency constraints. In 3D LiDAR perception, volumetric detectors such as SECOND~\cite{yan2018second} popularized sparse convolutions to process non-empty voxels, while two-stage point architectures such as PointRCNN~\cite{shi2019pointrcnn} and hybrid frameworks such as Part-$A^2$~\cite{shi2020part} and PV-RCNN~\cite{shi2020pvrcnn} introduced part-aware aggregation and voxel-set abstractions. Although these methods achieve high detection accuracy on benchmarks such as KITTI~\cite{geiger2012kitti}, their multi-million parameter footprints ($4.0\text{--}63.8\,\text{M}$) and heavy memory access latencies ($130\text{--}195\,\text{ms}$) prohibit real-time deployment on power-constrained edge accelerators.
```

**Sau — đã áp dụng**

```latex
Spatial feature extraction from sparse BEV planes must balance receptive field coverage against parameter and latency limits. For 3D LiDAR perception, SECOND~\cite{yan2018second} used sparse convolutions to process non-empty voxels. PointRCNN~\cite{shi2019pointrcnn}, Part-$A^2$~\cite{shi2020part}, and PV-RCNN~\cite{shi2020pvrcnn} developed two-stage point processing, part-aware aggregation, and voxel-set abstractions. These methods achieve high accuracy on KITTI~\cite{geiger2012kitti}, but their parameter counts ($4.0\text{--}63.8\,\text{M}$) and memory access latencies ($130\text{--}195\,\text{ms}$) limit real-time use on power-constrained edge accelerators.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý
Protected: 3D; \cite{yan2018second}; \cite{shi2019pointrcnn}; $A^2$; \cite{shi2020part}; \cite{shi2020pvrcnn}; \cite{geiger2012kitti}; $4.0\text{--}63.8\,\text{M}$; $130\text{--}195\,\text{ms}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u116"></a>

## U116 — [sections/related_work.tex:13](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:13)

**Trước**

```latex
To overcome the latency bottleneck of volumetric processing, lightweight 2D convolutional designs have gained prominence. The ConvNeXt paradigm~\cite{liu2022convnext} demonstrated that conventional convolutions modernized with $7 \times 7$ depthwise kernels, inverted bottleneck expansions ($e=2.5$), and LayerScale can achieve representational power competitive with Vision Transformers while preserving convolutional hardware efficiency. Concurrently, lightweight attention mechanisms have been developed to emphasize salient geometric features. While Efficient Channel Attention (ECA-Net)~\cite{wang2020eca} captures 1D cross-channel interactions with low complexity, it lacks spatial feature modulation. To address this weakness, Yang et al.~\cite{yang2021simam} proposed SimAM, deriving full 3D spatial-channel attention weights analytically via an energy function inspired by visual neuroscience, requiring zero trainable parameters.
```

**Sau — đã áp dụng**

```latex
Lightweight 2D convolutions offer an alternative to volumetric processing. ConvNeXt~\cite{liu2022convnext} combined $7 \times 7$ depthwise kernels, inverted bottleneck expansions ($e=2.5$), and LayerScale. It demonstrated accuracy competitive with Vision Transformers while retaining convolutional hardware efficiency. Lightweight attention mechanisms further modulate geometric features. Efficient Channel Attention (ECA-Net)~\cite{wang2020eca} modeled 1D cross-channel interactions with low complexity but did not modulate individual spatial locations. Yang et al.~\cite{yang2021simam} proposed SimAM, deriving 3D spatial-channel attention analytically from an energy function inspired by visual neuroscience, without trainable parameters.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: 2D; \cite{liu2022convnext}; $7 \times 7$; $e=2.5$; \cite{wang2020eca}; 1D; \cite{yang2021simam}; 3D
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u117"></a>

## U117 — [sections/related_work.tex:15](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:15)

**Trước**

```latex
Collectively, a shared limitation of prior mobile LiDAR detectors is the reliance on standard inverted residual bottlenecks with small $3 \times 3$ kernels~\cite{nguyen2024indoor}, which struggle to capture multi-scale context across disparate vehicle and pedestrian dimensions. Additionally, attempting to compensate by appending external attention modules or distillation schemes at coarse downsampling boundaries~\cite{tran2026rubber} introduces architectural overhead without resolving intra-block receptive field bottlenecks under sparse point distributions. In contrast, MobilePIXORNeXt incorporates the MobilePixorNeXt-S backbone with embedded parameter-free Block-SimAM (Sections~\ref{sec:backbone_architecture}--\ref{sec:next_s_block}) and a Top-Down Scale-Gated FPN (Section~\ref{sec:neck_architecture}), capturing expansive $7 \times 7$ depthwise receptive fields and modulating residual feature updates directly inside every residual block with zero added attention parameters.
```

**Sau — đã áp dụng**

```latex
Prior mobile LiDAR detectors commonly use inverted residual bottlenecks with small $3 \times 3$ kernels~\cite{nguyen2024indoor}, which limit context across vehicle and pedestrian scales. External attention modules or distillation at coarse downsampling boundaries~\cite{tran2026rubber} add overhead while leaving receptive field limits within blocks unresolved under sparse point distributions. MobilePIXORNeXt combines the MobilePixorNeXt-S backbone with parameter-free Block-SimAM (Sections~\ref{sec:backbone_architecture}--\ref{sec:next_s_block}) and a Top-Down Scale-Gated FPN (Section~\ref{sec:neck_architecture}). Large $7 \times 7$ depthwise kernels expand receptive fields, and attention modulates feature updates within each residual block without additional attention parameters.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý
Protected: $3 \times 3$; \cite{nguyen2024indoor}; \cite{tran2026rubber}; \ref{sec:backbone_architecture}; \ref{sec:next_s_block}; \ref{sec:neck_architecture}; $7 \times 7$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u118"></a>

## U118 — [sections/related_work.tex:20](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:20)

**Trước**

```latex
Bounding box regression in oriented object detection has progressively evolved from unconstrained coordinate offsets toward geometrically bounded objectives. Early detectors primarily relied on decoupled Smooth-L1 loss functions. However, regressing spatial coordinates, dimensions, and angles independently ignores the geometric interdependence of bounding boxes and induces severe gradient instability. To provide scale-invariant geometric feedback, Rezatofighi et al.~\cite{rezatofighi2019giou} introduced Generalized Intersection over Union (GIoU) to address gradient vanishing when candidate bounding boxes do not overlap. More recently, Le et al.~\cite{le2026mgiou} formulated Marginalized Generalized Intersection over Union (MGIoU), establishing a unified projection-based framework to optimize convex parametric shapes and arbitrary oriented bounding boxes without rotational singularities.
```

**Sau — đã áp dụng**

```latex
Oriented bounding box regression has developed from independent coordinate losses toward geometric objectives. Early detectors used decoupled Smooth-L1 losses. Independent regression of coordinates, dimensions, and angles ignores their geometric dependence and can destabilize gradients. Rezatofighi et al.~\cite{rezatofighi2019giou} introduced Generalized Intersection over Union (GIoU) to provide scale-invariant feedback when boxes do not overlap. Le et al.~\cite{le2026mgiou} later formulated Marginalized Generalized Intersection over Union (MGIoU), a projection-based objective for convex parametric shapes and arbitrarily oriented boxes without rotational singularities.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý
Protected: Smooth-L1; \cite{rezatofighi2019giou}; \cite{le2026mgiou}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u119"></a>

## U119 — [sections/related_work.tex:22](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:22)

**Trước**

```latex
In parallel with geometric loss formulation, joint quality calibration has emerged as an essential mechanism for accurate detection ranking. Li et al.~\cite{li2020gfl} introduced Generalized Focal Loss (GFL), demonstrating that uncalibrated classification scores frequently disagree with actual localization quality, causing low-quality boxes with high confidence to suppress well-localized candidates during Non-Maximum Suppression (NMS). Additionally, Kendall et al.~\cite{kendall2018uncertainty} introduced multi-task uncertainty weighting, utilizing homoscedastic task variance to balance heterogeneous loss objectives across classification and geometric regression tasks.
```

**Sau — đã áp dụng**

```latex
Localization quality calibration also affects detection ranking. Li et al.~\cite{li2020gfl} introduced Generalized Focal Loss (GFL). They showed that classification scores can disagree with localization quality, allowing high-confidence but poorly localized boxes to suppress accurate candidates during Non-Maximum Suppression (NMS). Kendall et al.~\cite{kendall2018uncertainty} introduced multi-task uncertainty weighting to balance classification and geometric regression losses through homoscedastic task variance.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý
Protected: \cite{li2020gfl}; \cite{kendall2018uncertainty}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u120"></a>

## U120 — [sections/related_work.tex:24](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/related_work.tex:24)

**Trước**

```latex
Despite these advancements, existing oriented bounding box detectors still struggle with three pervasive limitations. First, independent angle and corner regression objectives remain susceptible to the $\pi$-periodicity boundary jump problem, where slight angular deviations around $\pm \pi/2$ or $\pm \pi$ trigger severe loss spikes and yaw gradient collapse. Second, conventional quality calibration mechanisms rely on auxiliary decoupled prediction heads that inflate model parameters and exacerbate gradient contention across heterogeneous object classes. Third, unconstrained multi-task uncertainty weighting allows task variance parameters to diverge, leading to training instability when balancing disparate loss scales across classification, geometric overlap, and orientation. MobilePIXORNeXt addresses these multi-task supervision bottlenecks through a unified framework combining Oriented Geometric Alignment (OGA), its head-free alternative Q-OGA, and IoU-Aware Quality Alignment (IQA) (Section~\ref{sec:losses_and_oga}). Specifically, OGA unifies multi-axis MGIoU projections with scale-normalized $\pi$-symmetric corner supervision to eliminate boundary discontinuities; Q-OGA couples continuous IoU quality directly into classification supervision to achieve quality calibration with zero head overhead; IQA provides explicit multi-branch quality alignment scored via $S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}}\hat{q}}$ to suppress miscalibrated false positives; and Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW) guarantees stable convergence by constraining multi-task weights within a normalized, bounded simplex, simultaneously resolving orientation discontinuities, quality miscalibration, and loss scale divergence.
```

**Sau — đã áp dụng**

```latex
Three supervision limitations remain in oriented detection. Independent angle and corner losses are sensitive to the $\pi$-periodicity boundary: small angular deviations near $\pm \pi/2$ or $\pm \pi$ can cause loss spikes and unstable yaw gradients. Auxiliary quality heads add parameters and can increase gradient competition across classes. Unconstrained uncertainty weights can also diverge when classification, overlap, and orientation losses have different scales. MobilePIXORNeXt combines Oriented Geometric Alignment (OGA), its head-free alternative Q-OGA, and IoU-Aware Quality Alignment (IQA) (Section~\ref{sec:losses_and_oga}) to address these issues. OGA combines multi-axis MGIoU projections with scale-normalized $\pi$-symmetric corner supervision to avoid orientation boundary discontinuities. Q-OGA incorporates continuous IoU quality into classification supervision without an additional head. IQA provides explicit multi-branch quality prediction and uses $S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}}\hat{q}}$ to rank detections. Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW) constrains task weights to a normalized, bounded simplex to limit loss scale imbalance during optimization.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: $\pi$; $\pm \pi/2$; $\pm \pi$; \ref{sec:losses_and_oga}; $\pi$; $S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}}\hat{q}}$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u013"></a>

## U013 — [sections/experiments.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:5)

**Trước**

```latex
We evaluated our proposed detector on the benchmark standard KITTI 3D Object Detection dataset~\cite{geiger2012kitti}. The dataset comprises $7{,}481$ calibrated LiDAR point cloud scans equipped with synchronized rectified camera imagery and 3D bounding box ground-truth annotations across road scenes. We adopt the canonical 50/50 split introduced by Chen et al.~\cite{yang2018pixor}, which partitions the benchmark into $3{,}712$ training frames and $3{,}769$ validation frames. To prevent data leakage and ensure strict experimental isolation, all ground-truth sampling databases utilized by data augmentation pipelines are generated exclusively from the $3{,}712$ training frames.
```

**Sau — đã áp dụng**

```latex
We evaluated the detector on the KITTI 3D Object Detection dataset~\cite{geiger2012kitti}. The dataset contains $7{,}481$ calibrated LiDAR scans with synchronized rectified camera images and 3D bounding box annotations from road scenes. We used the 50/50 split introduced by Chen et al.~\cite{yang2018pixor}, with $3{,}712$ training frames and $3{,}769$ validation frames. Ground-truth sampling databases were generated exclusively from the $3{,}712$ training frames to prevent validation data leakage.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: 3D; \cite{geiger2012kitti}; $7{,}481$; 3D; 50; 50; \cite{yang2018pixor}; $3{,}712$; $3{,}769$; $3{,}712$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u014"></a>

## U014 — [sections/experiments.tex:8](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:8)

**Trước**

```latex
Following the geometric contract defined in Section~\ref{sec:problem_formulation}, the LiDAR coordinate frame is aligned with the vehicle body (forward $X$, leftward $Y$, upward $Z$). Perception is bounded within the Cartesian Region of Interest:
```

**Sau — đã áp dụng**

```latex
Following Section~\ref{sec:problem_formulation}, we aligned the LiDAR coordinate frame with the vehicle body (forward $X$, leftward $Y$, upward $Z$). We restricted perception to the Cartesian Region of Interest:
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: \ref{sec:problem_formulation}; $X$; $Y$; $Z$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u015"></a>

## U015 — [sections/experiments.tex:16](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:16)

**Trước**

```latex
An open-boundary inward margin of $\epsilon = 0.001\,\text{m}$ is enforced along all three axes. Discretizing the horizontal plane with a uniform grid resolution of $\Delta_x = \Delta_y = 0.1\,\text{m}$ yields an input tensor spatial dimension of $H_{\mathrm{in}} \times W_{\mathrm{in}} = 800 \times 704$ cells. At an output downsampling stride of $s_{\mathrm{out}} = 4$, the detector outputs feature maps of spatial size $H_{\mathrm{out}} \times W_{\mathrm{out}} = 200 \times 176$. Evaluation focuses on the three principal road categories: Car, Pedestrian, and Cyclist.
```

**Sau — đã áp dụng**

```latex
We applied an open-boundary inward margin of $\epsilon = 0.001\,\text{m}$ along all three axes. A horizontal grid resolution of $\Delta_x = \Delta_y = 0.1\,\text{m}$ produced an input tensor of $H_{\mathrm{in}} \times W_{\mathrm{in}} = 800 \times 704$ cells. An output stride of $s_{\mathrm{out}} = 4$ gave feature maps of size $H_{\mathrm{out}} \times W_{\mathrm{out}} = 200 \times 176$. We evaluated Car, Pedestrian, and Cyclist detections.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: $\epsilon = 0.001\,\text{m}$; $\Delta_x = \Delta_y = 0.1\,\text{m}$; $H_{\mathrm{in}} \times W_{\mathrm{in}} = 800 \times 704$; $s_{\mathrm{out}} = 4$; $H_{\mathrm{out}} \times W_{\mathrm{out}} = 200 \times 176$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u016"></a>

## U016 — [sections/experiments.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:19)

**Trước**

```latex
Detection performance is benchmarked on oriented 2D Bird's-Eye View (BEV) footprints using 40-point interpolated Average Precision (AP $R_{40}$) on the KITTI validation set, which calculates the area under the precision--recall curve over 40 recall positions $\mathcal{R}_{40} = \{1/40, 2/40, \dots, 1.0\}$:
```

**Sau — đã áp dụng**

```latex
We evaluated oriented 2D Bird's-Eye View (BEV) footprints on the KITTI validation set using 40-point interpolated Average Precision (AP $R_{40}$). This metric averages interpolated precision at 40 recall positions $\mathcal{R}_{40} = \{1/40, 2/40, \dots, 1.0\}$:
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; tách hoặc rút gọn câu nhiều ý
Protected: 2D; 40-point; $R_{40}$; 40; $\mathcal{R}_{40} = \{1/40, 2/40, \dots, 1.0\}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u017"></a>

## U017 — [sections/experiments.tex:23](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:23)

**Trước**

```latex
Following KITTI matching conventions, class-specific oriented bounding box Intersection-over-Union (IoU) overlap thresholds are set to $\tau_{\mathrm{IoU}} = 0.70$ for Car, and $\tau_{\mathrm{IoU}} = 0.50$ for Pedestrian and Cyclist. Detections are categorized into three difficulty regimes:
```

**Sau — đã áp dụng**

```latex
Following KITTI matching conventions, we set oriented bounding box Intersection-over-Union (IoU) thresholds to $\tau_{\mathrm{IoU}} = 0.70$ for Car and $\tau_{\mathrm{IoU}} = 0.50$ for Pedestrian and Cyclist. We grouped detections into three difficulty regimes:
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: $\tau_{\mathrm{IoU}} = 0.70$; $\tau_{\mathrm{IoU}} = 0.50$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u022"></a>

## U022 — [sections/experiments.tex:33](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:33)

**Trước**

```latex
We additionally report the unweighted mean across all nine class-difficulty combinations:
```

**Sau — đã áp dụng**

```latex
We also reported the unweighted mean across all nine class-difficulty combinations:
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: Không có math/cite/ref/label/số trong đoạn; giữ nguyên thuật ngữ và lệnh LaTeX.
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u023"></a>

## U023 — [sections/experiments.tex:37](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:37)

**Trước**

```latex
The MobilePIXORNeXt and historical MobilePIXOR results use the local loader-aligned BEV evaluator, while the reference detector results in Table~\ref{tab:comparison} use the reference KITTI evaluator. All report $R_{40}$ interpolation on the same validation split; the evaluator implementations are distinguished in the comparison table. Here, E, M, and H denote Easy, Moderate, and Hard regimes. Note that $\mathrm{mAP}_{9}$ represents the unweighted average across the 9-cell evaluation grid, rather than AP at an IoU threshold of 0.9.
```

**Sau — đã áp dụng**

```latex
MobilePIXORNeXt and historical MobilePIXOR results were computed with the local loader-aligned BEV evaluator. Reference detector results in Table~\ref{tab:comparison} used the reference KITTI evaluator. Both implementations used $R_{40}$ interpolation on the same validation split and are identified separately in the table. E, M, and H denote Easy, Moderate, and Hard. $\mathrm{mAP}_{9}$ is the unweighted average across the 9-cell evaluation grid, rather than AP at an IoU threshold of 0.9.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \ref{tab:comparison}; $R_{40}$; $\mathrm{mAP}_{9}$; 9-cell; 0.9.
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u024"></a>

## U024 — [sections/experiments.tex:43](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:43)

**Trước**

```latex
The proposed MobilePIXORNeXt architecture processes 14-channel Strata-BEV14 tensors. The network comprises a two-layer $3 \times 3$ convolutional stem ($32$ channels, stride 2), three hierarchical backbone stages with channel widths $(48, 96, 128)$ and block depths $(3, 4, 2)$ based on \texttt{MobilePixorNeXt-S} blocks with large $7 \times 7$ depthwise convolutions, inverted expansion ratio $e=2.5$, SiLU activations, and LayerScale parameters initialized to $10^{-5}$. Parameter-free Block-SimAM attention is embedded inside all nine residual blocks. The multiscale Scale-Gated Feature Pyramid Network (SG-FPN) fuses stages C4, C3, and C2 into an optimal $F_{32}$ (32-channel) representation at stride 4 ($200 \times 176$). Decoupled GroupedHeader towers process Car ($K_1=1$) and Pedestrian/Cyclist ($K_2=2$) independently, utilizing a shared initial $3 \times 3$ Conv-BN-SiLU transformation per group. The complete detector contains exactly $753{,}569$ trainable parameters, corresponding to an FP32 checkpoint size of $9.34\,\text{MB}$.
```

**Sau — đã áp dụng**

```latex
MobilePIXORNeXt processes 14-channel Strata-BEV14 tensors. We used a two-layer $3 \times 3$ convolutional stem ($32$ channels, stride 2) and three backbone stages with channel widths $(48, 96, 128)$ and block depths $(3, 4, 2)$. The stages used \texttt{MobilePixorNeXt-S} blocks with $7 \times 7$ depthwise convolutions, an inverted expansion ratio of $e=2.5$, SiLU activations, and LayerScale initialized to $10^{-5}$. Parameter-free Block-SimAM attention was embedded in all nine residual blocks. The Scale-Gated Feature Pyramid Network (SG-FPN) fused stages C4, C3, and C2 into a $F_{32}$ (32-channel) representation at stride 4 ($200 \times 176$). Decoupled GroupedHeader towers processed Car ($K_1=1$) and Pedestrian/Cyclist ($K_2=2$) independently, with an initial $3 \times 3$ Conv-BN-SiLU transformation per group. The detector contained $753{,}569$ trainable parameters, with an FP32 checkpoint size of $9.34\,\text{MB}$.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; tách hoặc rút gọn câu nhiều ý; bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể
Protected: 14-channel; Strata-BEV14; $3 \times 3$; $32$; 2; $(48, 96, 128)$; $(3, 4, 2)$; \texttt{MobilePixorNeXt-S}; $7 \times 7$; $e=2.5$; $10^{-5}$; C4; C3; C2; $F_{32}$; 32-channel; 4; $200 \times 176$; $K_1=1$; $K_2=2$; $3 \times 3$; $753{,}569$; FP32; $9.34\,\text{MB}$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u025"></a>

## U025 — [sections/experiments.tex:46](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:46)

**Trước**

```latex
The model was optimized using the AdamW optimizer with fused CUDA kernels. The base learning rate was set to $\eta_0 = 7 \times 10^{-4}$, decaying to a minimum floor of $\eta_{\min} = 1.0 \times 10^{-6}$. Weight decay was configured at $\lambda = 0.001$, applied strictly to multidimensional weight matrices ($\ge 2\mathrm{D}$ tensors); biases, Batch Normalization affine parameters, and task-uncertainty logits were exempted ($\lambda = 0.0$). Gradients were clamped to a maximum $L_2$ norm of $10.0$. The standard schedule spans 50 epochs and ends at epoch 50, comprising 4 linear warmup epochs followed by 46 epochs of cosine annealing down to $\eta_{\min}$. The final reported checkpoint comes from an extended 160-epoch run with 10 warmup epochs and was selected at epoch 130; the checkpoint-selection comparison retains these recorded epoch indices. The dynamic loss balancing across constituent multi-task objectives was governed by Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW, Section~\ref{sec:losses_and_oga}), whose empirical convergence stability is evaluated in Section~\ref{sec:ablations}.
```

**Sau — đã áp dụng**

```latex
We optimized the model with AdamW using fused CUDA kernels. The base learning rate was $\eta_0 = 7 \times 10^{-4}$, with a minimum of $\eta_{\min} = 1.0 \times 10^{-6}$. Weight decay of $\lambda = 0.001$ applied only to multidimensional weight matrices ($\ge 2\mathrm{D}$ tensors). Biases, Batch Normalization affine parameters, and task-uncertainty logits were exempt ($\lambda = 0.0$). Gradients were clipped to a maximum $L_2$ norm of $10.0$. The standard schedule covered 50 epochs, ending at epoch 50, after 4 linear warmup epochs and 46 cosine annealing epochs down to $\eta_{\min}$. The reported checkpoint came from an extended 160-epoch run with 10 warmup epochs and was selected at epoch 130. The checkpoint-selection comparison retains these recorded epoch indices. Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW, Section~\ref{sec:losses_and_oga}) balanced the task losses. Section~\ref{sec:ablations} evaluates its empirical convergence behavior.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: $\eta_0 = 7 \times 10^{-4}$; $\eta_{\min} = 1.0 \times 10^{-6}$; $\lambda = 0.001$; $\ge 2\mathrm{D}$; $\lambda = 0.0$; $L_2$; $10.0$; 50; 50,; 4; 46; $\eta_{\min}$; 160-epoch; 10; 130; \ref{sec:losses_and_oga}; \ref{sec:ablations}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u026"></a>

## U026 — [sections/experiments.tex:49](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:49)

**Trước**

```latex
Training used Physics-Consistent Ground-Truth Augmentation (\texttt{PC-GTA}; Section~\ref{sec:hybrid_augmentation}). In the final reported run, local ground fitting, static obstacle clearance, range-adaptive point subsampling, and visibility protection were enabled; LOS rejection, shadow masking, and radiometric calibration were available but disabled. Ground height was estimated within $R(d) = \operatorname{clip}(2.5 + 0.03\max(0, d - 15), 2.5, 4.0)\,\text{m}$. GT sampling was applied with probability $0.5$, followed by random world flipping along the $X$-axis with probability $0.5$, rotation $\theta \sim \mathcal{U}(-45^\circ,45^\circ)$, and isotropic scaling $s \sim \mathcal{U}(0.95,1.05)$. World translation was applied with probability $0.5$, using independent Gaussian noise with standard deviations $(0.2,0.2,0.1)\,\text{m}$ along $(X,Y,Z)$.
```

**Sau — đã áp dụng**

```latex
Training used Physics-Consistent Ground-Truth Augmentation (\texttt{PC-GTA}; Section~\ref{sec:hybrid_augmentation}). The final reported run enabled local ground fitting, static obstacle clearance, range-adaptive point subsampling, and visibility protection. LOS rejection, shadow masking, and radiometric calibration were available but disabled. Ground height was estimated within $R(d) = \operatorname{clip}(2.5 + 0.03\max(0, d - 15), 2.5, 4.0)\,\text{m}$. GT sampling was applied with probability $0.5$. Subsequent world transforms included flipping along the $X$-axis with probability $0.5$, rotation $\theta \sim \mathcal{U}(-45^\circ,45^\circ)$, and isotropic scaling $s \sim \mathcal{U}(0.95,1.05)$. World translation was applied with probability $0.5$, using independent Gaussian noise with standard deviations $(0.2,0.2,0.1)\,\text{m}$ along $(X,Y,Z)$.
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý
Protected: \texttt{PC-GTA}; \ref{sec:hybrid_augmentation}; $R(d) = \operatorname{clip}(2.5 + 0.03\max(0, d - 15), 2.5, 4.0)\,\text{m}$; $0.5$; $X$; $0.5$; $\theta \sim \mathcal{U}(-45^\circ,45^\circ)$; $s \sim \mathcal{U}(0.95,1.05)$; $0.5$; $(0.2,0.2,0.1)\,\text{m}$; $(X,Y,Z)$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u027"></a>

## U027 — [sections/experiments.tex:52](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/experiments.tex:52)

**Trước**

```latex
Training used a physical batch size of 16 on a single GPU, while validation used a batch size of 32. DataLoader processes ran with 6 persistent worker threads, pinned memory, and a prefetch factor of 2. Numerical forward and backward passes utilized Automatic Mixed Precision (AMP) in Bfloat16 (\texttt{bf16}), while loss evaluation, coordinate regression, and rotated NMS were conducted in full FP32 precision to maintain numerical stability. The final MobilePIXORNeXt timing and accuracy evaluation was performed on a server equipped with an NVIDIA Tesla T4 GPU (16\,GB GDDR6 VRAM, Turing architecture), host x86\_64 CPU, Linux OS, and PyTorch 2.11.0 with CUDA 13.0. Peak GPU memory allocated during this evaluation was $98.44\,\text{MB}$. The historical MobilePIXOR timing result was recorded on an NVIDIA L4 and is labeled separately in Table~\ref{tab:efficiency}. Following standard conventions on the competitive KITTI benchmark, all reported ablation and comparative metrics reflect controlled deterministic evaluation with a fixed random seed ($\text{seed} = 42$) under identical validation split conditions, where variance and standard deviation ($\pm\,\text{std}$) across multi-seed runs are demarcated for future multi-fold benchmarks. Table~\ref{tab:setup} summarizes the verified implementation specifications.
```

**Sau — đã áp dụng**

```latex
Training used a physical batch size of 16 on a single GPU; validation used a batch size of 32. DataLoader processes used 6 persistent worker threads, pinned memory, and a prefetch factor of 2. Forward and backward passes used Automatic Mixed Precision (AMP) in Bfloat16 (\texttt{bf16}). Loss evaluation, coordinate regression, and rotated NMS used full FP32 precision for numerical stability. Final MobilePIXORNeXt timing and accuracy evaluation used an NVIDIA Tesla T4 GPU (16\,GB GDDR6 VRAM, Turing architecture), an x86\_64 host CPU, Linux OS, and PyTorch 2.11.0 with CUDA 13.0. Peak allocated GPU memory was $98.44\,\text{MB}$. Historical MobilePIXOR timing was recorded on an NVIDIA L4 and is identified separately in Table~\ref{tab:efficiency}. All KITTI ablation and comparison metrics came from deterministic single-seed evaluation ($\text{seed} = 42$) on the same validation split. Variance and standard deviation ($\pm\,\text{std}$) across seeds were left for future multi-fold evaluation. Table~\ref{tab:setup} summarizes the implementation settings.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý
Protected: 16; 32.; 6; 2.; Bfloat16; \texttt{bf16}; FP32; T4; 16; GDDR6; x86; 2.11.0; 13.0.; $98.44\,\text{MB}$; L4; \ref{tab:efficiency}; $\text{seed} = 42$; $\pm\,\text{std}$; \ref{tab:setup}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u036"></a>

## U036 — [sections/methodology.tex:8](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:8)

**Trước**

```latex
    \captionof{figure}{End-to-end architecture and tensor flow of the MobilePIXORNeXt LiDAR object detector. Unordered 3D LiDAR point clouds are transformed via the parameter-free Strata-BEV14 rasterizer into a 14-channel statistical BEV tensor $\mathbf{X} \in \mathbb{R}^{B \times 14 \times 800 \times 704}$ without learned pillar encoders. The streamlined backbone extracts multi-scale features through ConvNeXt-style stages with embedded parameter-free Block-SimAM attention, fused by a top-down Scale-Gated Feature Pyramid Network (SG-FPN) into a 32-channel shared representation at stride 4 ($200 \times 176$). Decoupled GroupedHeader towers process Car and Pedestrian/Cyclist categories independently. Solid arrows trace the feedforward inference path, whereas dashed arrows indicate training-only supervision incorporating Oriented Geometric Alignment (OGA), IoU-Aware Quality Alignment (IQA), and dynamic loss balancing via Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW).}
```

**Sau — đã áp dụng**

```latex
    \captionof{figure}{End-to-end architecture and tensor flow of MobilePIXORNeXt. Unordered 3D LiDAR point clouds pass through the parameter-free Strata-BEV14 rasterizer into a 14-channel statistical BEV tensor $\mathbf{X} \in \mathbb{R}^{B \times 14 \times 800 \times 704}$ without learned pillar encoders. ConvNeXt-style stages with parameter-free Block-SimAM attention extract multi-scale features. A top-down Scale-Gated Feature Pyramid Network (SG-FPN) fuses them into a 32-channel shared representation at stride 4 ($200 \times 176$). Decoupled GroupedHeader towers process Car and Pedestrian/Cyclist categories independently. Solid arrows show the feedforward inference path. Dashed arrows show training-only Oriented Geometric Alignment (OGA), IoU-Aware Quality Alignment (IQA), and Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW).}
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; tách hoặc rút gọn câu nhiều ý
Protected: 3D; Strata-BEV14; 14-channel; $\mathbf{X} \in \mathbb{R}^{B \times 14 \times 800 \times 704}$; 32-channel; 4; $200 \times 176$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u037"></a>

## U037 — [sections/methodology.tex:12](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:12)

**Trước**

```latex
MobilePIXORNeXt is a compact, proposal-free, multi-class three-dimensional (3D) object detector designed for real-time Light Detection and Ranging (LiDAR) perception on resource-constrained embedded platforms. Operating strictly on an efficient Bird's-Eye View (BEV) representation, the framework eliminates computationally expensive learned point-cloud feature encoders and three-dimensional sparse convolutions. Figure~\ref{fig:r37_overview} illustrates the overall detection pipeline, detailing the end-to-end data flow from the raw point cloud to the final decoded oriented bounding boxes.
```

**Sau — đã áp dụng**

```latex
MobilePIXORNeXt is a compact, proposal-free, multi-class three-dimensional (3D) object detector for real-time Light Detection and Ranging (LiDAR) perception on resource-constrained embedded platforms. Its Bird's-Eye View (BEV) representation avoids learned point-cloud encoders and three-dimensional sparse convolutions. Figure~\ref{fig:r37_overview} shows the data flow from the raw point cloud to decoded oriented bounding boxes.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: 3D; \ref{fig:r37_overview}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u038"></a>

## U038 — [sections/methodology.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:19)

**Trước**

```latex
An unordered LiDAR point cloud is acquired via a rotating scanner and defined as a set $\mathcal{P} = \{\mathbf{p}_i\}_{i=1}^{N_{\mathrm{pts}}}$, where each point $\mathbf{p}_i = (x_i, y_i, z_i, \iota_i) \in \mathbb{R}^4$ encapsulates longitudinal distance $x_i$ (forward positive), lateral distance $y_i$ (leftward positive), vertical elevation $z_i$ (upward positive), and calibrated reflectance intensity $\iota_i \in [0, 1]$. Perception is constrained to a fixed Cartesian Region of Interest (ROI):
```

**Sau — đã áp dụng**

```latex
An unordered LiDAR point cloud acquired by a rotating scanner is represented as $\mathcal{P} = \{\mathbf{p}_i\}_{i=1}^{N_{\mathrm{pts}}}$. Each point $\mathbf{p}_i = (x_i, y_i, z_i, \iota_i) \in \mathbb{R}^4$ contains longitudinal distance $x_i$ (forward positive), lateral distance $y_i$ (leftward positive), vertical elevation $z_i$ (upward positive), and calibrated reflectance intensity $\iota_i \in [0, 1]$. Perception is restricted to a Cartesian Region of Interest (ROI):
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: $\mathcal{P} = \{\mathbf{p}_i\}_{i=1}^{N_{\mathrm{pts}}}$; $\mathbf{p}_i = (x_i, y_i, z_i, \iota_i) \in \mathbb{R}^4$; $x_i$; $y_i$; $z_i$; $\iota_i \in [0, 1]$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u039"></a>

## U039 — [sections/methodology.tex:27](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:27)

**Trước**

```latex
configured with longitudinal bounds $[0.0, 70.4]$\,m, lateral bounds $[-40.0, 40.0]$\,m, and vertical bounds $[-2.5, 1.0]$\,m. An open-boundary inward margin of $\epsilon = 0.001$\,m is enforced to eliminate boundary artifacts. Discretizing the horizontal plane with a uniform spatial resolution of $\Delta_x = \Delta_y = 0.1$\,m produces a discrete grid of size $H_{\mathrm{in}} \times W_{\mathrm{in}} = 800 \times 704$ cells. A point $\mathbf{p}_i$ is deterministically indexed into grid coordinate $(u_i, v_i)$ via:
```

**Sau — đã áp dụng**

```latex
We used longitudinal bounds $[0.0, 70.4]$\,m, lateral bounds $[-40.0, 40.0]$\,m, and vertical bounds $[-2.5, 1.0]$\,m. An open-boundary inward margin of $\epsilon = 0.001$\,m reduced boundary artifacts. A horizontal resolution of $\Delta_x = \Delta_y = 0.1$\,m gave a grid of $H_{\mathrm{in}} \times W_{\mathrm{in}} = 800 \times 704$ cells. A point $\mathbf{p}_i$ maps to grid coordinate $(u_i, v_i)$ as:
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể
Protected: $[0.0, 70.4]$; $[-40.0, 40.0]$; $[-2.5, 1.0]$; $\epsilon = 0.001$; $\Delta_x = \Delta_y = 0.1$; $H_{\mathrm{in}} \times W_{\mathrm{in}} = 800 \times 704$; $\mathbf{p}_i$; $(u_i, v_i)$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u041"></a>

## U041 — [sections/methodology.tex:39](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:39)

**Trước**

```latex
where $(x_k^*, y_k^*) \in \mathbb{R}^2$ denotes the metric continuous center coordinates, $(l_k^*, w_k^*) \in \mathbb{R}^2$ represents length and width footprints, $\psi_k^* \in [-\pi/2, \pi/2]$ is the orientation heading angle under a $\pi$-symmetric doubled-angle representation, and $s_k \in [0, 1]$ is the calibrated detection confidence score. Operating on this contract requires a structured, dense BEV representation that captures fine 3D geometric distributions without the latency of learned point encoders, which we formulate next.
```

**Sau — đã áp dụng**

```latex
where $(x_k^*, y_k^*) \in \mathbb{R}^2$ denotes continuous metric center coordinates and $(l_k^*, w_k^*) \in \mathbb{R}^2$ denotes length and width. The heading $\psi_k^* \in [-\pi/2, \pi/2]$ uses a $\pi$-symmetric doubled-angle representation, and $s_k \in [0, 1]$ is the calibrated confidence score. These targets require a dense BEV representation that retains fine 3D geometry without a learned point encoder.
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: $(x_k^*, y_k^*) \in \mathbb{R}^2$; $(l_k^*, w_k^*) \in \mathbb{R}^2$; $\psi_k^* \in [-\pi/2, \pi/2]$; $\pi$; $s_k \in [0, 1]$; 3D
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u042"></a>

## U042 — [sections/methodology.tex:50](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:50)

**Trước**

```latex
    \captionof{figure}{Visual breakdown of the fourteen deterministic feature channels in the Strata-BEV14 representation for a representative KITTI driving scene ($800 \times 704$ grid at $0.1\,\text{m}$ resolution). Channels 0--3 capture logarithmic point densities across four uniform vertical elevation strata ($[-2.5, 1.0]\,\text{m}$); Channels 4--7 encode vertical spatial moments (normalized maximum elevation $z_{\max}$, mean elevation $z_{\mathrm{mean}}$, elevation span $z_{\mathrm{span}}$, and sample standard deviation $z_{\mathrm{std}}$); Channels 8--9 capture calibrated surface reflectance statistics (maximum and mean intensity); Channel 10 records total cell logarithmic density; Channel 11 provides a binary occupancy indicator; and Channels 12--13 preserve continuous subcell horizontal centroid displacements $(\Delta\bar{x}, \Delta\bar{y}) \in [-0.5, 0.5]$ to mitigate spatial discretization errors. Brighter intensities signify higher point concentrations, elevated reflectance, or greater vertical dispersion.}
```

**Sau — đã áp dụng**

```latex
    \captionof{figure}{Fourteen deterministic feature channels in the Strata-BEV14 representation for a KITTI driving scene ($800 \times 704$ grid at $0.1\,\text{m}$ resolution). Channels 0--3 encode logarithmic point densities across four vertical strata ($[-2.5, 1.0]\,\text{m}$). Channels 4--7 encode normalized maximum elevation $z_{\max}$, mean elevation $z_{\mathrm{mean}}$, elevation span $z_{\mathrm{span}}$, and sample standard deviation $z_{\mathrm{std}}$. Channels 8--9 encode maximum and mean reflectance intensity. Channel 10 records total cell logarithmic density, and Channel 11 records binary occupancy. Channels 12--13 preserve continuous horizontal subcell centroid displacements $(\Delta\bar{x}, \Delta\bar{y}) \in [-0.5, 0.5]$ to reduce discretization error. Brighter values indicate higher point concentrations, reflectance, or vertical dispersion.}
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý
Protected: Strata-BEV14; $800 \times 704$; $0.1\,\text{m}$; 0--3; $[-2.5, 1.0]\,\text{m}$; 4--7; $z_{\max}$; $z_{\mathrm{mean}}$; $z_{\mathrm{span}}$; $z_{\mathrm{std}}$; 8--9; 10; 11; 12--13; $(\Delta\bar{x}, \Delta\bar{y}) \in [-0.5, 0.5]$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u043"></a>

## U043 — [sections/methodology.tex:56](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:56)

**Trước**

```latex
Voxel- and pillar-based detectors~\cite{lang2019pointpillars,li2023pillarnext} rely on PointNet-style column encoders, adding computation and padding overhead. By contrast, 35-channel binary slicing~\cite{yang2018pixor} omits continuous height and reflectance cues, while three-channel height bands~\cite{khoshkdahan2026triband} lack higher-order geometry. We therefore propose \textbf{Strata-BEV14} (Figure~\ref{fig:strata_bev14}), a deterministic, parameter-free rasterizer that balances detail and efficiency.
```

**Sau — đã áp dụng**

```latex
Voxel- and pillar-based detectors~\cite{lang2019pointpillars,li2023pillarnext} use PointNet-style column encoders, which add computation and padding overhead. 35-channel binary slicing~\cite{yang2018pixor} omits continuous height and reflectance cues, while three-channel height bands~\cite{khoshkdahan2026triband} omit higher-order geometry. We developed \textbf{Strata-BEV14} (Figure~\ref{fig:strata_bev14}), a deterministic, parameter-free rasterizer, to retain these features with low computational cost.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: \cite{lang2019pointpillars,li2023pillarnext}; 35-channel; \cite{yang2018pixor}; \cite{khoshkdahan2026triband}; \textbf{Strata-BEV14}; \ref{fig:strata_bev14}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u044"></a>

## U044 — [sections/methodology.tex:60](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:60)

**Trước**

```latex
For each occupied cell $(u, v)$ containing $N$ points $\mathcal{P}_{u, v} = \{\mathbf{p}_i\}_{i=1}^N$, vertical elevations are normalized linearly to unit interval $\zeta_i = (z_i - z_{\min}) / (z_{\max} - z_{\min}) \in [0, 1]$, and reflectance values $\iota_i$ are clamped to $[0, 1]$. Strata-BEV14 computes fourteen analytical features across four functional groups, summarized in Table~\ref{tab:strata_bev14}:
```

**Sau — đã áp dụng**

```latex
For each occupied cell $(u, v)$ with $N$ points $\mathcal{P}_{u, v} = \{\mathbf{p}_i\}_{i=1}^N$, we normalized vertical elevations to $\zeta_i = (z_i - z_{\min}) / (z_{\max} - z_{\min}) \in [0, 1]$ and clamped reflectance values $\iota_i$ to $[0, 1]$. Strata-BEV14 computes fourteen analytical features in four groups (Table~\ref{tab:strata_bev14}):
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: $(u, v)$; $N$; $\mathcal{P}_{u, v} = \{\mathbf{p}_i\}_{i=1}^N$; $\zeta_i = (z_i - z_{\min}) / (z_{\max} - z_{\min}) \in [0, 1]$; $\iota_i$; $[0, 1]$; Strata-BEV14; \ref{tab:strata_bev14}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u049"></a>

## U049 — [sections/methodology.tex:84](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:84)

**Trước**

```latex
4. \textit{Occupancy and Continuous Subcell Offsets ($F_{10}$--$F_{13}$)}: Channel $F_{10} = d_{\log}(N)$ records total point density, while $F_{11} = 1.0$ signifies non-empty occupancy. To prevent quantization error from coarse $0.1$\,m cell discretization, channels $F_{12}$ and $F_{13}$ preserve continuous subcell horizontal centroid displacements:
```

**Sau — đã áp dụng**

```latex
4. \textit{Occupancy and Continuous Subcell Offsets ($F_{10}$--$F_{13}$)}: Channel $F_{10} = d_{\log}(N)$ records total point density, and $F_{11} = 1.0$ indicates occupancy. For $0.1$\,m cells, channels $F_{12}$ and $F_{13}$ retain continuous horizontal subcell centroid displacements to reduce quantization error:
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: 4.; \textit{Occupancy and Continuous Subcell Offsets ($F_{10}$--$F_{13}$)}; $F_{10} = d_{\log}(N)$; $F_{11} = 1.0$; $0.1$; $F_{12}$; $F_{13}$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u052"></a>

## U052 — [sections/methodology.tex:122](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:122)

**Trước**

```latex
Strata-BEV14 is implemented with a NumPy reference rasterizer and a Numba-compiled CPU backend. Both compute the same fourteen deterministic features without learned parameters. The Numba implementation accumulates point statistics and normalizes occupied cells, with a direct CHW output path for model input. Point-cloud loading and rasterization together take $20.94$\,ms in the measured evaluation pipeline; this preprocessing measurement is not an encoder-only timing.
```

**Sau — đã áp dụng**

```latex
Strata-BEV14 has a NumPy reference rasterizer and a Numba-compiled CPU backend. Both compute the same fourteen deterministic features without learned parameters. The Numba backend accumulates point statistics, normalizes occupied cells, and can write CHW output directly for model input. Point-cloud loading and rasterization took $20.94$\,ms together in the measured pipeline. This measurement includes the complete preprocessing stage.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: Strata-BEV14; $20.94$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u053"></a>

## U053 — [sections/methodology.tex:128](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:128)

**Trước**

```latex
Cut-and-paste ground-truth (GT) augmentation balances rare LiDAR categories, but naive insertion ignores scene physics, producing floating or submerged objects, background collisions, missing raycast shadows, and unrealistic range-dependent point densities. Our \textbf{Physics-Consistent Ground-Truth Augmentation} (\texttt{PC-GTA}) addresses these artifacts and improves Moderate mAP by $+6.03\%$ over naive sampling in our ablations.
```

**Sau — đã áp dụng**

```latex
Cut-and-paste ground-truth (GT) augmentation balances rare LiDAR categories, but naive insertion ignores scene physics. It can produce floating or submerged objects, background collisions, missing raycast shadows, and unrealistic range-dependent point densities. We developed \textbf{Physics-Consistent Ground-Truth Augmentation} (\texttt{PC-GTA}) to address these artifacts. It improved Moderate mAP by $+6.03\%$ over naive sampling in the ablations.
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: \textbf{Physics-Consistent Ground-Truth Augmentation}; \texttt{PC-GTA}; $+6.03\%$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u060"></a>

## U060 — [sections/methodology.tex:146](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:146)

**Trước**

```latex
\item \textit{Optional Raycast LiDAR Shadow Masking}: In real LiDAR scans, solid foreground objects cast occlusive ray shadows that block returns from background objects behind them. When this mechanism is enabled, we compute geometric ray intersections for each inserted box $\mathbf{r}(t) = t \cdot \mathbf{p}$ for all background points $\mathbf{p}$. Points residing behind the exit face ($t_{\mathrm{leave}} \le 1.0$) with radial distance $(1 - t_{\mathrm{leave}})\|\mathbf{p}\| > 0.05$\,m are masked out:
```

**Sau — đã áp dụng**

```latex
\item \textit{Optional Raycast LiDAR Shadow Masking}: In LiDAR scans, foreground objects occlude returns from objects behind them. When enabled, this mechanism computes intersections for each inserted box using rays $\mathbf{r}(t) = t \cdot \mathbf{p}$ through background points $\mathbf{p}$. It masks points behind the exit face ($t_{\mathrm{leave}} \le 1.0$) whose radial distance satisfies $(1 - t_{\mathrm{leave}})\|\mathbf{p}\| > 0.05$\,m:
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \textit{Optional Raycast LiDAR Shadow Masking}; $\mathbf{r}(t) = t \cdot \mathbf{p}$; $\mathbf{p}$; $t_{\mathrm{leave}} \le 1.0$; $(1 - t_{\mathrm{leave}})\|\mathbf{p}\| > 0.05$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u063"></a>

## U063 — [sections/methodology.tex:186](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:186)

**Trước**

```latex
    \captionof{figure}{Structural topology of the MobilePIXORNeXt feedforward hierarchy, illustrating layer dimensions, channel widths, and receptive field expansion. A two-layer convolutional stem ingests 14-channel input at stride 2 ($400 \times 352$), followed by three hierarchical stages (C2: 48 channels, depth 3; C3: 96 channels, depth 4; C4: 128 channels, depth 2) composed of \texttt{MobilePixorNeXt-S} blocks with embedded Block-SimAM. The Scale-Gated FPN (SG-FPN) combines deep semantic context with fine spatial coordinates via lateral $1 \times 1$ projections, bilinear $2\times$ upsampling with depthwise $3 \times 3$ refinement, and learnable scale gates $\mathbf{G}_r$ into a shared 32-channel feature map ($F_{32}$) at stride 4 ($200 \times 176$). Independent grouped prediction towers decouple Car from Pedestrian and Cyclist to eliminate aspect-ratio conflicts while conserving model capacity (753,569 total parameters).}
```

**Sau — đã áp dụng**

```latex
    \captionof{figure}{MobilePIXORNeXt feedforward hierarchy with layer dimensions, channel widths, and receptive fields. A two-layer stem processes 14-channel input at stride 2 ($400 \times 352$). Three stages follow: C2 has 48 channels and depth 3; C3 has 96 channels and depth 4; C4 has 128 channels and depth 2. Each stage uses \texttt{MobilePixorNeXt-S} blocks with Block-SimAM. The Scale-Gated FPN (SG-FPN) uses lateral $1 \times 1$ projections, bilinear $2\times$ upsampling, depthwise $3 \times 3$ refinement, and learnable scale gates $\mathbf{G}_r$. It produces a shared 32-channel feature map ($F_{32}$) at stride 4 ($200 \times 176$). Independent grouped towers separate Car predictions from Pedestrian and Cyclist predictions to reduce aspect-ratio conflicts within the parameter budget (753,569 total parameters).}
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: 14-channel; 2; $400 \times 352$; C2; 48; 3; C3; 96; 4; C4; 128; 2; \texttt{MobilePixorNeXt-S}; $1 \times 1$; $2\times$; $3 \times 3$; $\mathbf{G}_r$; 32-channel; $F_{32}$; 4; $200 \times 176$; 753,569
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u064"></a>

## U064 — [sections/methodology.tex:192](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:192)

**Trước**

```latex
Target objects on the horizontal BEV plane exhibit wide geometric scale variations: spanning from compact pedestrians ($l \approx 0.6$\,m) and cyclists ($l \approx 1.8$\,m) to passenger vehicles ($l \approx 4.5$--$5.0$\,m). Conventional backbones with small $3 \times 3$ convolutions require excessive network depth to cover the requisite spatial context, ballooning latency and parameter count. In contrast, at our input grid resolution of $\Delta_x = \Delta_y = 0.1$\,m, a single $7 \times 7$ depthwise receptive field directly spans $0.7 \times 0.7$\,m of physical terrain at stride 1, and rapidly expands to over $5.6 \times 5.6$\,m by Stage 3 (stride 8)---sufficient to envelop an entire passenger vehicle in a single layer without dilation artifacts. To achieve this wide context under strict parameter parsimony, we design the \textbf{MobilePixorNeXt Backbone} hierarchy featuring large $7 \times 7$ depthwise convolutions, inverted bottleneck expansions, and LayerScale residual stabilization organized into repeated \texttt{MobilePixorNeXt-S} blocks.
```

**Sau — đã áp dụng**

```latex
BEV object footprints range from pedestrians ($l \approx 0.6$\,m) and cyclists ($l \approx 1.8$\,m) to passenger vehicles ($l \approx 4.5$--$5.0$\,m). Small $3 \times 3$ convolutions require deeper networks to cover this spatial context, increasing latency and parameter count. At an input resolution of $\Delta_x = \Delta_y = 0.1$\,m, a $7 \times 7$ depthwise receptive field spans $0.7 \times 0.7$\,m at stride 1. It expands to over $5.6 \times 5.6$\,m by Stage 3 (stride 8), enough to cover a passenger vehicle without dilation. We designed \textbf{MobilePixorNeXt Backbone} with large $7 \times 7$ depthwise convolutions, inverted bottleneck expansions, and LayerScale stabilization in repeated \texttt{MobilePixorNeXt-S} blocks to obtain this context within a small parameter budget.
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: $l \approx 0.6$; $l \approx 1.8$; $l \approx 4.5$; $5.0$; $3 \times 3$; $\Delta_x = \Delta_y = 0.1$; $7 \times 7$; $0.7 \times 0.7$; 1,; $5.6 \times 5.6$; 3; 8; \textbf{MobilePixorNeXt Backbone}; $7 \times 7$; \texttt{MobilePixorNeXt-S}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u065"></a>

## U065 — [sections/methodology.tex:197](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:197)

**Trước**

```latex
    \item \textit{Convolutional Stem}: Input $\mathbf{X} \in \mathbb{R}^{B \times 14 \times 800 \times 704}$ is ingested by two consecutive $3 \times 3$ convolutions (stride 2 followed by stride 1) with 32 output channels, Batch Normalization (BN), and Sigmoid Linear Unit (SiLU) activations, yielding stride-2 representation $\mathbf{F}^{(2)} \in \mathbb{R}^{B \times 32 \times 400 \times 352}$.
```

**Sau — đã áp dụng**

```latex
    \item \textit{Convolutional Stem}: Input $\mathbf{X} \in \mathbb{R}^{B \times 14 \times 800 \times 704}$ passes through two consecutive $3 \times 3$ convolutions, with stride 2 followed by stride 1 and 32 output channels. Batch Normalization (BN) and Sigmoid Linear Unit (SiLU) activations produce stride-2 representation $\mathbf{F}^{(2)} \in \mathbb{R}^{B \times 32 \times 400 \times 352}$.
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý
Protected: \textit{Convolutional Stem}; $\mathbf{X} \in \mathbb{R}^{B \times 14 \times 800 \times 704}$; $3 \times 3$; 2; 1; 32; stride-2; $\mathbf{F}^{(2)} \in \mathbb{R}^{B \times 32 \times 400 \times 352}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u070"></a>

## U070 — [sections/methodology.tex:207](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:207)

**Trước**

```latex
Unlike multi-branch reparameterization paradigms that require complex multi-path training topologies and post-hoc weight folding, MobilePIXORNeXt directly trains and deploys a clean single-branch large-kernel ($7 \times 7$) depthwise convolution. By pairing wide spatial receptive fields with inverted bottleneck expansion ($e=2.5$) and LayerScale stabilization, the backbone achieves rapid optimization convergence and fast throughput ($55.94$\,FPS on Tesla T4) without incurring auxiliary branch memory overhead during backpropagation. The complete backbone comprises only $604,864$ parameters.
```

**Sau — đã áp dụng**

```latex
MobilePIXORNeXt uses a single-branch large-kernel ($7 \times 7$) depthwise convolution during both training and deployment. It avoids the multi-path training topology and post-hoc weight folding of multi-branch reparameterization. Inverted bottleneck expansion ($e=2.5$) and LayerScale stabilization support optimization without auxiliary-branch memory during backpropagation. The backbone achieved $55.94$\,FPS on Tesla T4 and contained $604,864$ parameters.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: $7 \times 7$; $e=2.5$; $55.94$; T4; $604,864$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u071"></a>

## U071 — [sections/methodology.tex:217](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:217)

**Trước**

```latex
    \captionof{figure}{Micro-architecture of the \texttt{MobilePixorNeXt-S} residual block and the embedded parameter-free Block-SimAM energy attention module. \textbf{Top}: Complete residual transformation path from input $\mathbf{Z}$ to output $\mathbf{Z}'$, featuring a large-kernel $7 \times 7$ depthwise convolution, inverted bottleneck expansion with ratio $e = 2.5$, SiLU activation, pointwise projection, parameter-free 3D spatial-energy attention $\mathcal{S}(\mathbf{R})$, LayerScale scaling $\boldsymbol{\gamma}_{\mathrm{blk}}$ (initialized to $10^{-5}$), and an identity shortcut. \textbf{Bottom}: Internal operational mechanism of the SimAM operator, which calculates channel-wise spatial moments ($\mu_c, \nu_c$) in FP32 precision, evaluates visual saliency via inverse-energy importance $s_t = (t - \mu_c)^2 / (4(\nu_c + \lambda_{\mathrm{sim}})) + 0.5$, and produces a 3D modulation tensor $\mathbf{A} \in [0, 1]^{B \times C \times H \times W}$ via sigmoid scaling without introducing any learnable parameters.}
```

**Sau — đã áp dụng**

```latex
    \captionof{figure}{The \texttt{MobilePixorNeXt-S} residual block and parameter-free Block-SimAM attention. \textbf{Top}: Residual path from input $\mathbf{Z}$ to output $\mathbf{Z}'$. It applies a $7 \times 7$ depthwise convolution, inverted bottleneck expansion with ratio $e = 2.5$, and SiLU activation, followed by pointwise projection. Parameter-free 3D spatial-energy attention $\mathcal{S}(\mathbf{R})$ precedes LayerScale scaling $\boldsymbol{\gamma}_{\mathrm{blk}}$ (initialized to $10^{-5}$) and the identity shortcut. \textbf{Bottom}: SimAM computes channel-wise spatial moments ($\mu_c, \nu_c$) in FP32 precision and inverse-energy importance $s_t = (t - \mu_c)^2 / (4(\nu_c + \lambda_{\mathrm{sim}})) + 0.5$. Sigmoid scaling produces the 3D modulation tensor $\mathbf{A} \in [0, 1]^{B \times C \times H \times W}$ without learned parameters.}
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \texttt{MobilePixorNeXt-S}; \textbf{Top}; $\mathbf{Z}$; $\mathbf{Z}'$; $7 \times 7$; $e = 2.5$; 3D; $\mathcal{S}(\mathbf{R})$; $\boldsymbol{\gamma}_{\mathrm{blk}}$; $10^{-5}$; \textbf{Bottom}; $\mu_c, \nu_c$; FP32; $s_t = (t - \mu_c)^2 / (4(\nu_c + \lambda_{\mathrm{sim}})) + 0.5$; 3D; $\mathbf{A} \in [0, 1]^{B \times C \times H \times W}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u074"></a>

## U074 — [sections/methodology.tex:232](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:232)

**Trước**

```latex
Standard attention modules such as Efficient Channel Attention (ECA-Net)~\cite{wang2020eca} or Squeeze-and-Excitation compute 1D channel attention weights by pooling across spatial dimensions, thereby treating all spatial locations identically. Conversely, non-local spatial attention requires quadratic computational complexity $\mathcal{O}((HW)^2)$, rendering it prohibitive for real-time dense BEV grids ($200 \times 176$). To address this trade-off, we integrate \textbf{Block-SimAM}~\cite{yang2021simam}---a 3D spatial-energy attention module operating without adding a single learned parameter.
```

**Sau — đã áp dụng**

```latex
Efficient Channel Attention (ECA-Net)~\cite{wang2020eca} and Squeeze-and-Excitation compute 1D channel weights by spatial pooling, applying the same weight to every location in a channel. Non-local spatial attention instead has quadratic complexity $\mathcal{O}((HW)^2)$, which is costly on dense BEV grids ($200 \times 176$). We integrated \textbf{Block-SimAM}~\cite{yang2021simam}, a parameter-free 3D spatial-energy attention module, to provide spatial modulation without learned attention weights.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: \cite{wang2020eca}; 1D; $\mathcal{O}((HW)^2)$; $200 \times 176$; \textbf{Block-SimAM}; \cite{yang2021simam}; 3D
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u080"></a>

## U080 — [sections/methodology.tex:261](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:261)

**Trước**

```latex
Unlike stage-level adapters that add learned linear projections, Block-SimAM introduces exactly \textbf{zero parameters} ($0$\,M). It simultaneously assigns true 3D spatial-and-channel attention weights, adaptively amplifying sparse LiDAR point clusters against flat terrain backgrounds. Empirical evaluation confirms that embedding SimAM inside each \texttt{MobilePixorNeXt-S} block achieves $+0.88\%$ mAP over the no-attention configuration, outperforming stage-level adapters by up to $+2.10\%$ mAP (Table~\ref{tab:attention}).
```

**Sau — đã áp dụng**

```latex
Block-SimAM adds \textbf{zero parameters} ($0$\,M), whereas stage-level adapters add learned linear projections. Its 3D spatial-and-channel weights modulate sparse LiDAR clusters relative to flat terrain backgrounds. In the ablation, SimAM placement inside each \texttt{MobilePixorNeXt-S} block improved mAP by $+0.88\%$ over no attention and by up to $+2.10\%$ over stage-level adapters (Table~\ref{tab:attention}).
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: \textbf{zero parameters}; $0$; 3D; \texttt{MobilePixorNeXt-S}; $+0.88\%$; $+2.10\%$; \ref{tab:attention}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u081"></a>

## U081 — [sections/methodology.tex:270](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:270)

**Trước**

```latex
Standard Feature Pyramid Networks rely on transposed convolutions (deconvolutions) to upsample low-resolution feature maps, frequently introducing severe high-frequency checkerboard artifacts on continuous coordinate regression grids. While bidirectional multiscale networks (such as PANet or BiFPN) introduce cyclic bottom-up pathways that substantially inflate memory latency on edge accelerators, direct unweighted elementwise addition in conventional FPN causes semantic feature dilution. To resolve this, we design the \textbf{Top-Down Scale-Gated Feature Pyramid Network} (\texttt{SG-FPN Neck})---an efficient, unidirectional top-down fusion neck equipped with depthwise scale-gating.
```

**Sau — đã áp dụng**

```latex
Transposed convolutions used for feature upsampling can produce checkerboard artifacts in continuous coordinate regression maps. Bidirectional multiscale networks such as PANet and BiFPN add bottom-up pathways and memory latency. Unweighted addition in conventional FPN can also dilute semantic features. We designed \textbf{Top-Down Scale-Gated Feature Pyramid Network} (\texttt{SG-FPN Neck}) as a unidirectional top-down neck with depthwise scale gating to balance these features at lower cost.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: \textbf{Top-Down Scale-Gated Feature Pyramid Network}; \texttt{SG-FPN Neck}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u082"></a>

## U082 — [sections/methodology.tex:276](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:276)

**Trước**

```latex
    \captionof{figure}{Detailed schematic of the Scale-Gated Feature Pyramid Network (SG-FPN) fusion mechanism at stride $r \in \{8, 4\}$. Low-resolution top-down features are upsampled via bilinear interpolation followed by depthwise $3 \times 3$ convolution, Batch Normalization, and SiLU activation to yield refined semantic representation $\mathbf{U}_r$, eliminating deconvolution checkerboard artifacts. Lateral high-resolution features $\mathbf{L}_r$ are dynamically weighted via a zero-initialized depthwise convolution gate $\mathbf{G}_r = 2\sigma(\operatorname{DWConv}_{3\times3}(\mathbf{L}_r + \mathbf{U}_r))$ before element-wise summation $\mathbf{P}^{(r)} = \mathbf{U}_r + \mathbf{G}_r \odot \mathbf{L}_r$, adaptively preserving boundary details while suppressing background clutter.}
```

**Sau — đã áp dụng**

```latex
    \captionof{figure}{Scale-Gated Feature Pyramid Network (SG-FPN) fusion at stride $r \in \{8, 4\}$. Bilinear interpolation upsamples top-down features. Depthwise $3 \times 3$ convolution, Batch Normalization, and SiLU then produce semantic representation $\mathbf{U}_r$ without deconvolution checkerboard artifacts. Lateral high-resolution features $\mathbf{L}_r$ are weighted by a zero-initialized depthwise gate $\mathbf{G}_r = 2\sigma(\operatorname{DWConv}_{3\times3}(\mathbf{L}_r + \mathbf{U}_r))$ before element-wise summation $\mathbf{P}^{(r)} = \mathbf{U}_r + \mathbf{G}_r \odot \mathbf{L}_r$ to retain boundary detail and suppress background responses.}
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: $r \in \{8, 4\}$; $3 \times 3$; $\mathbf{U}_r$; $\mathbf{L}_r$; $\mathbf{G}_r = 2\sigma(\operatorname{DWConv}_{3\times3}(\mathbf{L}_r + \mathbf{U}_r))$; $\mathbf{P}^{(r)} = \mathbf{U}_r + \mathbf{G}_r \odot \mathbf{L}_r$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u084"></a>

## U084 — [sections/methodology.tex:285](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:285)

**Trước**

```latex
The feature aggregation proceeds strictly along a unidirectional top-down path. Spatial upsampling is performed using $2\times$ interpolation followed by depthwise $3 \times 3$ refinement convolution, BN, and SiLU to yield top-down features $\mathbf{U}_4$ and $\mathbf{U}_3$. At each fusion stride $r \in \{8, 4\}$, lateral and top-down representations are adaptively balanced via learnable depthwise scale gating:
```

**Sau — đã áp dụng**

```latex
Feature aggregation follows a unidirectional top-down path. $2\times$ interpolation upsamples features before depthwise $3 \times 3$ refinement, BN, and SiLU produce $\mathbf{U}_4$ and $\mathbf{U}_3$. At fusion stride $r \in \{8, 4\}$, a learnable depthwise gate balances lateral and top-down features:
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: $2\times$; $3 \times 3$; $\mathbf{U}_4$; $\mathbf{U}_3$; $r \in \{8, 4\}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u085"></a>

## U085 — [sections/methodology.tex:291](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:291)

**Trước**

```latex
Because depthwise gating weights and biases are initialized to zero, gating $\mathbf{G}_r$ initially evaluates to $1.0$, smoothly starting from standard residual addition. Finally, $\mathbf{P}^{(4)}$ is projected via a $3 \times 3$ convolution, BN, and SiLU to produce the shared detector representation $\mathbf{F}_{\mathrm{out}} \in \mathbb{R}^{B \times 32 \times 200 \times 176}$. The complete neck requires only $36,784$ parameters, delivering this calibrated multi-scale representation directly to the prediction towers.
```

**Sau — đã áp dụng**

```latex
Zero-initialized gating weights and biases make $\mathbf{G}_r$ initially equal to $1.0$, corresponding to standard residual addition. The feature $\mathbf{P}^{(4)}$ passes through a $3 \times 3$ convolution, BN, and SiLU to form the shared detector representation $\mathbf{F}_{\mathrm{out}} \in \mathbb{R}^{B \times 32 \times 200 \times 176}$. The neck contains $36,784$ parameters and passes the fused multi-scale features to the prediction towers.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể
Protected: $\mathbf{G}_r$; $1.0$; $\mathbf{P}^{(4)}$; $3 \times 3$; $\mathbf{F}_{\mathrm{out}} \in \mathbb{R}^{B \times 32 \times 200 \times 176}$; $36,784$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u086"></a>

## U086 — [sections/methodology.tex:301](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:301)

**Trước**

```latex
    \captionof{figure}{Geometric parameterization, orientation formulation, and quality-aware score calibration across prediction heads. \textbf{(a)} Continuous metric subcell coordinate offsets $(\hat{o}_x, \hat{o}_y)$ regressed from grid anchor center $P$ to the true object center $C$. \textbf{(b)} Log-transformed bounding box dimension regression targets $(t_w, t_l) = (\log w, \log l)$, decoded via exponential mapping. \textbf{(c)} Doubled-angle trigonometric yaw representation $(c_2, s_2) = (\cos 2\theta, \sin 2\theta)$ constrained on the unit circle, resolving $180^\circ$ heading ambiguity and enforcing $\pi$-rotational symmetry on BEV footprints. \textbf{(d)} Joint localization quality confidence calibration, where continuous overlap prediction $\hat{q} \in [0, 1]$ is combined with classification probability $p_{\mathrm{cls}}$ via $s_{\mathrm{det}} = (p_{\mathrm{cls}})^{1-\alpha} \hat{q}^\alpha$ ($\alpha = 0.5$) to prioritize well-aligned bounding boxes during candidate peak filtering and rotated NMS.}
```

**Sau — đã áp dụng**

```latex
    \captionof{figure}{Prediction-head geometry, orientation, and quality-aware scoring. \textbf{(a)} Metric subcell offsets $(\hat{o}_x, \hat{o}_y)$ from grid anchor center $P$ to object center $C$. \textbf{(b)} Log-dimension targets $(t_w, t_l) = (\log w, \log l)$ decoded by exponentiation. \textbf{(c)} Doubled-angle yaw representation $(c_2, s_2) = (\cos 2\theta, \sin 2\theta)$ on the unit circle, resolving $180^\circ$ heading ambiguity through $\pi$-rotational symmetry of BEV footprints. \textbf{(d)} Joint quality scoring combines overlap prediction $\hat{q} \in [0, 1]$ and classification probability $p_{\mathrm{cls}}$ as $s_{\mathrm{det}} = (p_{\mathrm{cls}})^{1-\alpha} \hat{q}^\alpha$ ($\alpha = 0.5$). This score prioritizes well-aligned boxes during peak filtering and rotated NMS.}
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \textbf{(a)}; $(\hat{o}_x, \hat{o}_y)$; $P$; $C$; \textbf{(b)}; $(t_w, t_l) = (\log w, \log l)$; \textbf{(c)}; $(c_2, s_2) = (\cos 2\theta, \sin 2\theta)$; $180^\circ$; $\pi$; \textbf{(d)}; $\hat{q} \in [0, 1]$; $p_{\mathrm{cls}}$; $s_{\mathrm{det}} = (p_{\mathrm{cls}})^{1-\alpha} \hat{q}^\alpha$; $\alpha = 0.5$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u087"></a>

## U087 — [sections/methodology.tex:307](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:307)

**Trước**

```latex
Autonomous driving involves drastic morphological discrepancies across object categories. Cars exhibit rigid rectangular footprints ($l \approx 4.5$\,m, $w \approx 1.8$\,m), whereas Pedestrians ($l \approx 0.8$\,m, $w \approx 0.6$\,m) and Cyclists ($l \approx 1.8$\,m, $w \approx 0.7$\,m) are compact and highly deformable. Forcing a single shared prediction head to jointly regress all classes induces severe gradient competition and scale interference. Conversely, dedicating entirely separate heads to each class inflates parameter size. We resolve this via \textbf{Shared Grouped Prediction Towers} (\texttt{GroupedHeader}).
```

**Sau — đã áp dụng**

```latex
Road categories have different BEV footprints. Cars are rectangular ($l \approx 4.5$\,m, $w \approx 1.8$\,m), while Pedestrians ($l \approx 0.8$\,m, $w \approx 0.6$\,m) and Cyclists ($l \approx 1.8$\,m, $w \approx 0.7$\,m) are smaller and more deformable. A shared prediction head can introduce gradient competition across these scales; separate heads for every class add parameters. We designed \textbf{Shared Grouped Prediction Towers} (\texttt{GroupedHeader}) to balance class separation and model size.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: $l \approx 4.5$; $w \approx 1.8$; $l \approx 0.8$; $w \approx 0.6$; $l \approx 1.8$; $w \approx 0.7$; \textbf{Shared Grouped Prediction Towers}; \texttt{GroupedHeader}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u088"></a>

## U088 — [sections/methodology.tex:310](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:310)

**Trước**

```latex
Classes are partitioned into two semantically compatible task groups: Group 1 for \texttt{car} ($K_1 = 1$), and Group 2 for \texttt{ped\_cyc} ($K_2 = 2$). For each group $g \in \{1, 2\}$, feature map $\mathbf{F}_{\mathrm{out}}$ passes through a group-shared initial $3 \times 3$ Conv-BN-SiLU block ($32 \rightarrow 32$ channels), before bifurcating into five task-specific prediction branches:
```

**Sau — đã áp dụng**

```latex
We partitioned classes into Group 1 for \texttt{car} ($K_1 = 1$) and Group 2 for \texttt{ped\_cyc} ($K_2 = 2$). For each group $g \in \{1, 2\}$, feature map $\mathbf{F}_{\mathrm{out}}$ passes through an initial $3 \times 3$ Conv-BN-SiLU block with $32 \rightarrow 32$ channels. Five task-specific branches follow:
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; tách hoặc rút gọn câu nhiều ý
Protected: 1; \texttt{car}; $K_1 = 1$; 2; \texttt{ped\_cyc}; $K_2 = 2$; $g \in \{1, 2\}$; $\mathbf{F}_{\mathrm{out}}$; $3 \times 3$; $32 \rightarrow 32$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u094"></a>

## U094 — [sections/methodology.tex:325](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:325)

**Trước**

```latex
The doubled-angle formulation $\mathbf{a}^*$ strictly enforces $\pi$-rotational symmetry, resolving the $180^\circ$ heading ambiguity inherent to symmetric BEV bounding boxes. Classification targets use 2D Gaussian heatmaps centered at continuous object projections with a minimum radius of 4 cells, and regression targets are assigned within a radius of half the Gaussian peak, denoted by valid indicator mask $\mathbf{M}_{\mathrm{reg}}$.
```

**Sau — đã áp dụng**

```latex
The doubled-angle formulation $\mathbf{a}^*$ enforces $\pi$-rotational symmetry and resolves $180^\circ$ heading ambiguity in symmetric BEV boxes. We used 2D Gaussian classification heatmaps centered at continuous object projections, with a minimum radius of 4 cells. Regression targets were assigned within half the Gaussian peak radius, using validity mask $\mathbf{M}_{\mathrm{reg}}$.
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: $\mathbf{a}^*$; $\pi$; $180^\circ$; 2D; 4; $\mathbf{M}_{\mathrm{reg}}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u095"></a>

## U095 — [sections/methodology.tex:331](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:331)

**Trước**

```latex
We propose \textbf{Oriented Geometric Alignment (OGA)} combined with \textbf{IoU-Aware Quality Alignment (IQA)} to bridge the disparity between independent coordinate parameter regression and true oriented bounding box overlap. In standard anchor-free detectors, box regression optimizes disjoint Smooth-L1 losses independently over spatial center $(x, y)$, footprint $(w, l)$, and heading angle $\psi$. However, independent parameter errors do not correlate monotonically with the actual Intersection-over-Union (IoU) overlap of rotated bounding boxes. Additionally, standard classification confidence scores fail to reflect spatial localization precision, degrading candidate ranking during rotated Non-Maximum Suppression (NMS).
```

**Sau — đã áp dụng**

```latex
We developed \textbf{Oriented Geometric Alignment (OGA)} with \textbf{IoU-Aware Quality Alignment (IQA)} to connect box-parameter regression with oriented overlap. Standard anchor-free regression applies separate Smooth-L1 losses to center $(x, y)$, footprint $(w, l)$, and heading $\psi$. These errors do not vary monotonically with rotated Intersection-over-Union (IoU). Classification confidence can also disagree with localization accuracy, degrading candidate ranking during rotated Non-Maximum Suppression (NMS).
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: \textbf{Oriented Geometric Alignment (OGA)}; \textbf{IoU-Aware Quality Alignment (IQA)}; Smooth-L1; $(x, y)$; $(w, l)$; $\psi$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u096"></a>

## U096 — [sections/methodology.tex:334](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:334)

**Trước**

```latex
For every cell $(u, v)$ where $\mathbf{M}_{\mathrm{reg}} = 1$, predicted and ground-truth boxes are decoded into four 2D corners $\{\widehat{\mathbf{c}}_m\}_{m=1}^4$ and $\{\mathbf{c}_m^*\}_{m=1}^4$. OGA couples two synergistic geometric objectives:
```

**Sau — đã áp dụng**

```latex
For each cell $(u, v)$ with $\mathbf{M}_{\mathrm{reg}} = 1$, we decoded predicted and ground-truth boxes into four 2D corners $\{\widehat{\mathbf{c}}_m\}_{m=1}^4$ and $\{\mathbf{c}_m^*\}_{m=1}^4$. OGA combines two geometric objectives:
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: $(u, v)$; $\mathbf{M}_{\mathrm{reg}} = 1$; 2D; $\{\widehat{\mathbf{c}}_m\}_{m=1}^4$; $\{\mathbf{c}_m^*\}_{m=1}^4$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u101"></a>

## U101 — [sections/methodology.tex:368](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:368)

**Trước**

```latex
Detaching the target prevents gradient instability while forcing the model to learn reliable confidence calibration.
```

**Sau — đã áp dụng**

```latex
Detaching the target limits gradient instability while retaining supervision for confidence calibration.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: Không có math/cite/ref/label/số trong đoạn; giữ nguyên thuật ngữ và lệnh LaTeX.
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u102"></a>

## U102 — [sections/methodology.tex:371](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:371)

**Trước**

```latex
Dedicating an explicit IQA branch provides peak ranking accuracy at the cost of $18.59\,\text{K}$ parameters and $1.25\,\text{ms}$ forward latency. To support resource-constrained edge platforms, we formulate \textbf{Q-OGA} as a head-free quality calibration alternative. Q-OGA directly modulates positive ground-truth classification targets ($y=1.0$) with continuous detached bounding box IoU quality $q^*$ without introducing an auxiliary prediction head:
```

**Sau — đã áp dụng**

```latex
The explicit IQA branch gave the highest ranking accuracy, adding $18.59\,\text{K}$ parameters and $1.25\,\text{ms}$ forward latency. We developed \textbf{Q-OGA} as a head-free alternative for resource-constrained platforms. Q-OGA modulates positive classification targets ($y=1.0$) with detached continuous bounding box IoU quality $q^*$, without an auxiliary head:
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: $18.59\,\text{K}$; $1.25\,\text{ms}$; \textbf{Q-OGA}; $y=1.0$; $q^*$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u103"></a>

## U103 — [sections/methodology.tex:381](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:381)

**Trước**

```latex
where $\gamma = 2.0$. By training the classification head to output IoU-calibrated confidence scores intrinsically, Q-OGA eliminates the $18.59\,\text{K}$ parameter footprint of the explicit IQA heads and reduces forward inference latency by $1.25\,\text{ms}$, serving as an ultra-compact supervision strategy.
```

**Sau — đã áp dụng**

```latex
where $\gamma = 2.0$. Training the classification head to predict IoU-calibrated scores removes the explicit IQA heads' $18.59\,\text{K}$ parameters. In the evaluation, Q-OGA reduced forward latency by $1.25\,\text{ms}$, providing a compact supervision option.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: $\gamma = 2.0$; $18.59\,\text{K}$; $1.25\,\text{ms}$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u104"></a>

## U104 — [sections/methodology.tex:384](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:384)

**Trước**

```latex
To dynamically harmonize multi-task gradient scales across six heterogeneous loss objectives $\mathcal{T} = \{\mathrm{cls}, \mathrm{off}, \mathrm{size}, \mathrm{yaw}, \mathrm{geo}, \mathrm{qual}\}$ ($J=6$), we formulate \textbf{T-SBUW} as a principled, bounded optimization framework that eliminates fragile manual hyperparameter tuning. The formulation operates via two coupled mechanisms:
```

**Sau — đã áp dụng**

```latex
To balance six task losses $\mathcal{T} = \{\mathrm{cls}, \mathrm{off}, \mathrm{size}, \mathrm{yaw}, \mathrm{geo}, \mathrm{qual}\}$ ($J=6$), we developed \textbf{T-SBUW} without manually tuning their relative weights. It uses two coupled mechanisms:
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: $\mathcal{T} = \{\mathrm{cls}, \mathrm{off}, \mathrm{size}, \mathrm{yaw}, \mathrm{geo}, \mathrm{qual}\}$; $J=6$; \textbf{T-SBUW}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u106"></a>

## U106 — [sections/methodology.tex:441](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:441)

**Trước**

```latex
The complete inference procedure is formalized in Algorithm~\ref{alg:inference_pipeline}. For each candidate cell $(u, v)$ where predicted classification probability $p_{\mathrm{cls}} > \tau_{\mathrm{score}} = 0.05$ forms a $3 \times 3$ local spatial peak, metric continuous bounding boxes are decoded via:
```

**Sau — đã áp dụng**

```latex
Algorithm~\ref{alg:inference_pipeline} specifies inference. For each candidate cell $(u, v)$ whose classification probability $p_{\mathrm{cls}} > \tau_{\mathrm{score}} = 0.05$ is a $3 \times 3$ local spatial maximum, bounding boxes are decoded in metric coordinates as:
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \ref{alg:inference_pipeline}; $(u, v)$; $p_{\mathrm{cls}} > \tau_{\mathrm{score}} = 0.05$; $3 \times 3$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u110"></a>

## U110 — [sections/methodology.tex:459](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:459)

**Trước**

```latex
The complete MobilePIXORNeXt detector contains exactly \textbf{753,569 trainable parameters} ($< 0.76$\,M), corresponding to an ultra-compact FP32 checkpoint size of $9.34$\,MB:
```

**Sau — đã áp dụng**

```latex
MobilePIXORNeXt contains \textbf{753,569 trainable parameters} ($< 0.76$\,M), with an FP32 checkpoint size of $9.34$\,MB:
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể
Protected: \textbf{753,569 trainable parameters}; $< 0.76$; FP32; $9.34$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u112"></a>

## U112 — [sections/methodology.tex:466](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/methodology.tex:466)

**Trước**

```latex
This compact footprint enables full end-to-end inference at $16.41$\,FPS ($60.96$\,ms) and network forward throughput of $55.94$\,FPS ($17.88$\,ms) on commodity NVIDIA Tesla T4 GPUs.
```

**Sau — đã áp dụng**

```latex
The complete pipeline achieved $16.41$\,FPS ($60.96$\,ms), while standalone network throughput reached $55.94$\,FPS ($17.88$\,ms) on NVIDIA Tesla T4 GPUs.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể
Protected: $16.41$; $60.96$; $55.94$; $17.88$; T4
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u121"></a>

## U121 — [sections/results.tex:5](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:5)

**Trước**

```latex
Table~\ref{tab:comparison} presents a quantitative comparison on the standard KITTI validation benchmark using BEV AP $R_{40}$ at Moderate difficulty. In accordance with standard benchmark conventions, all reported scores reflect deterministic single-seed evaluation ($\text{seed} = 42$); variance and standard deviation bounds ($\pm\,\text{std}$) across random seeds are reserved for extended evaluation splits. We benchmark MobilePIXORNeXt against established LiDAR object detectors spanning compact BEV models, pillar-based networks, and 3D sparse convolutional architectures.
```

**Sau — đã áp dụng**

```latex
Table~\ref{tab:comparison} compares detectors on KITTI validation using Moderate BEV AP $R_{40}$. All scores came from deterministic single-seed evaluation ($\text{seed} = 42$). Variance and standard deviation ($\pm\,\text{std}$) across random seeds were left for extended evaluation. We compared MobilePIXORNeXt with compact BEV models, pillar-based networks, and 3D sparse convolutional detectors.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \ref{tab:comparison}; $R_{40}$; $\text{seed} = 42$; $\pm\,\text{std}$; 3D
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u122"></a>

## U122 — [sections/results.tex:12](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:12)

**Trước**

```latex
\caption{Pareto trade-off efficiency of MobilePIXORNeXt against representative LiDAR detectors on the KITTI validation benchmark using Moderate BEV AP $R_{40}$. \textbf{(a)} Detection accuracy ($\text{mAP}_{\mathrm{M}}$, \%) versus parameter footprint (M) on a logarithmic scale, highlighting that MobilePIXORNeXt (red star) achieves $71.46\%$ mAP with only $0.754\,\text{M}$ parameters, matching SECOND ($72.09\%$, $5.33\,\text{M}$) and outperforming PointPillars ($70.13\%$, $4.84\,\text{M}$) with $>84\%$ fewer parameters. \textbf{(b)} Detection accuracy ($\text{mAP}_{\mathrm{M}}$, \%) versus end-to-end pipeline latency (ms) at batch size 1 (Tesla T4, except the historical MobilePIXOR result on L4), demonstrating that MobilePIXORNeXt achieves the fastest complete perception pipeline ($60.96\,\text{ms}$, $16.41\,\text{FPS}$) among all evaluated real-time detectors.}
```

**Sau — đã áp dụng**

```latex
\caption{MobilePIXORNeXt accuracy, parameter count, and latency on KITTI validation using Moderate BEV AP $R_{40}$. \textbf{(a)} Accuracy ($\text{mAP}_{\mathrm{M}}$, \%) versus parameter count (M, logarithmic scale). MobilePIXORNeXt (red star) achieves $71.46\%$ mAP with $0.754\,\text{M}$ parameters, matching SECOND ($72.09\%$, $5.33\,\text{M}$) and exceeding PointPillars ($70.13\%$, $4.84\,\text{M}$) with $>84\%$ fewer parameters. \textbf{(b)} Accuracy ($\text{mAP}_{\mathrm{M}}$, \%) versus end-to-end latency (ms) at batch size 1 (Tesla T4, except historical MobilePIXOR on L4). MobilePIXORNeXt has the lowest complete pipeline latency ($60.96\,\text{ms}$, $16.41\,\text{FPS}$) among the evaluated real-time detectors.}
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: $R_{40}$; \textbf{(a)}; $\text{mAP}_{\mathrm{M}}$; $71.46\%$; $0.754\,\text{M}$; $72.09\%$; $5.33\,\text{M}$; $70.13\%$; $4.84\,\text{M}$; $>84\%$; \textbf{(b)}; $\text{mAP}_{\mathrm{M}}$; 1; T4; L4; $60.96\,\text{ms}$; $16.41\,\text{FPS}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u123"></a>

## U123 — [sections/results.tex:19](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:19)

**Trước**

```latex
As visualized in the Pareto frontier plots (Figure~\ref{fig:pareto_tradeoff}), MobilePIXORNeXt established a leading accuracy-to-parameter and accuracy-to-latency trade-off for real-time edge perception. Operating with only $0.754\,\text{M}$ trainable parameters ($9.34\,\text{MB}$ checkpoint size), MobilePIXORNeXt achieved $71.46\%$ Moderate mAP and $73.42\%$ $\text{mAP}_{9}$. Compared to PointPillars~\cite{lang2019pointpillars} ($4.835\,\text{M}$ parameters), MobilePIXORNeXt achieved higher Moderate mAP ($71.46\%$ vs. $70.13\%$, a $+1.33\%$ margin) and higher $\text{mAP}_{9}$ ($73.42\%$ vs. $72.25\%$) while requiring an $84.4\%$ reduction in parameter count. Compared to the widely deployed SECOND detector~\cite{yan2018second} ($5.326\,\text{M}$ parameters), MobilePIXORNeXt trailed by merely $0.63\%$ Moderate mAP ($71.46\%$ vs. $72.09\%$) despite operating with $7.1\times$ fewer parameters. Compared to its direct lineage baseline MobilePIXOR~\cite{nguyen2024indoor} ($41.40\%$ mAP), MobilePIXORNeXt achieved an absolute improvement of $+30.06\%$ Moderate mAP ($71.46\%$ vs. $41.40\%$) and $+31.15\%$ $\text{mAP}_{9}$ ($73.42\%$ vs. $42.27\%$).
```

**Sau — đã áp dụng**

```latex
Figure~\ref{fig:pareto_tradeoff} summarizes MobilePIXORNeXt's observed accuracy, parameter, and latency trade-offs. With $0.754\,\text{M}$ trainable parameters ($9.34\,\text{MB}$ checkpoint), MobilePIXORNeXt achieved $71.46\%$ Moderate mAP and $73.42\%$ $\text{mAP}_{9}$. Compared with PointPillars~\cite{lang2019pointpillars} ($4.835\,\text{M}$ parameters), MobilePIXORNeXt achieved higher Moderate mAP ($71.46\%$ vs. $70.13\%$, a $+1.33\%$ margin) and $\text{mAP}_{9}$ ($73.42\%$ vs. $72.25\%$) with an $84.4\%$ parameter reduction. MobilePIXORNeXt trailed SECOND~\cite{yan2018second} ($5.326\,\text{M}$ parameters) by $0.63\%$ Moderate mAP ($71.46\%$ vs. $72.09\%$) with $7.1\times$ fewer parameters. Relative to MobilePIXOR~\cite{nguyen2024indoor} ($41.40\%$ mAP), MobilePIXORNeXt improved Moderate mAP by $+30.06\%$ ($71.46\%$ vs. $41.40\%$) and by $+31.15\%$ in $\text{mAP}_{9}$ ($73.42\%$ vs. $42.27\%$).
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý
Protected: \ref{fig:pareto_tradeoff}; $0.754\,\text{M}$; $9.34\,\text{MB}$; $71.46\%$; $73.42\%$; $\text{mAP}_{9}$; \cite{lang2019pointpillars}; $4.835\,\text{M}$; $71.46\%$; $70.13\%$; $+1.33\%$; $\text{mAP}_{9}$; $73.42\%$; $72.25\%$; $84.4\%$; \cite{yan2018second}; $5.326\,\text{M}$; $0.63\%$; $71.46\%$; $72.09\%$; $7.1\times$; \cite{nguyen2024indoor}; $41.40\%$; $+30.06\%$; $71.46\%$; $41.40\%$; $+31.15\%$; $\text{mAP}_{9}$; $73.42\%$; $42.27\%$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u125"></a>

## U125 — [sections/results.tex:30](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:30)

**Trước**

```latex
In autonomous driving perception, detecting small, non-rigid objects such as pedestrians presents the most challenging operational regime due to extreme point sparsity and severe aspect-ratio disparity. As evidenced in Table~\ref{tab:comparison} and the Precision--Recall curves in Figure~\ref{fig:pr_curves}, MobilePIXORNeXt achieved \textbf{60.75\% Moderate AP} on Pedestrians, outperforming PointPillars ($56.01\%$, a $+4.74\%$ gain) and SECOND ($56.56\%$, a $+4.19\%$ gain), while more than doubling the baseline MobilePIXOR ($28.16\%$, a $+32.59\%$ gain). We attribute this performance advantage primarily to two architectural mechanisms: (i) Strata-BEV14's continuous subcell displacement offsets $(\Delta\bar{x}, \Delta\bar{y})$, which mitigate the $0.1\,\text{m}$ grid quantization error that inherently impairs small object localization; and (ii) the decoupled GroupedHeader architecture, which isolates pedestrian regression from large vehicle footprints to alleviate gradient interference across morphologically disparate categories.
```

**Sau — đã áp dụng**

```latex
Sparse returns and aspect-ratio differences make small, non-rigid objects such as pedestrians difficult to detect. Table~\ref{tab:comparison} and Figure~\ref{fig:pr_curves} report MobilePIXORNeXt's \textbf{60.75\% Moderate AP} on Pedestrians. This exceeded PointPillars ($56.01\%$) by $+4.74\%$ and SECOND ($56.56\%$) by $+4.19\%$, and more than doubled MobilePIXOR ($28.16\%$, a $+32.59\%$ gain). Two mechanisms may explain this advantage. The first is Strata-BEV14's continuous subcell displacement offsets $(\Delta\bar{x}, \Delta\bar{y})$, which reduce quantization error from the $0.1\,\text{m}$ grid for small objects. The second is GroupedHeader, which separates pedestrian regression from vehicle footprints to reduce gradient interference between categories.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: \ref{tab:comparison}; \ref{fig:pr_curves}; \textbf{60.75\% Moderate AP}; $56.01\%$; $+4.74\%$; $56.56\%$; $+4.19\%$; $28.16\%$; $+32.59\%$; Strata-BEV14; $(\Delta\bar{x}, \Delta\bar{y})$; $0.1\,\text{m}$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u126"></a>

## U126 — [sections/results.tex:33](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:33)

**Trước**

```latex
For passenger vehicles (Car), MobilePIXORNeXt achieved $86.90\%$ Moderate AP, closely trailing PointPillars ($88.02\%$) and SECOND ($88.56\%$) by less than $1.7\%$, while outperforming the original PIXOR ($72.02\%$) by $+14.88\%$ and MobilePIXOR ($75.52\%$) by $+11.38\%$. On Cyclists, MobilePIXORNeXt attained $66.73\%$ Moderate AP, surpassing PointPillars ($66.34\%$) and trailing SECOND ($71.16\%$). While two-stage 3D anchor-based detectors (PointRCNN, Part-A$^2$, PV-RCNN) achieve higher absolute accuracy on rigid vehicles, they require between $4.0\,\text{M}$ and $63.8\,\text{M}$ parameters and operate at higher latencies ($133\text{--}195\,\text{ms}$), exceeding the real-time constraints of embedded edge processors.
```

**Sau — đã áp dụng**

```latex
For Car, MobilePIXORNeXt achieved $86.90\%$ Moderate AP. It trailed PointPillars ($88.02\%$) and SECOND ($88.56\%$) by less than $1.7\%$, while exceeding PIXOR ($72.02\%$) by $+14.88\%$ and MobilePIXOR ($75.52\%$) by $+11.38\%$. For Cyclists, MobilePIXORNeXt reached $66.73\%$ Moderate AP, exceeding PointPillars ($66.34\%$) and trailing SECOND ($71.16\%$). Two-stage 3D anchor-based detectors (PointRCNN, Part-A$^2$, PV-RCNN) achieved higher vehicle accuracy. Their parameter counts ranged from $4.0\,\text{M}$ to $63.8\,\text{M}$, and their higher latencies ($133\text{--}195\,\text{ms}$) limited deployment under embedded real-time constraints.
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: $86.90\%$; $88.02\%$; $88.56\%$; $1.7\%$; $72.02\%$; $+14.88\%$; $75.52\%$; $+11.38\%$; $66.73\%$; $66.34\%$; $71.16\%$; 3D; $^2$; $4.0\,\text{M}$; $63.8\,\text{M}$; $133\text{--}195\,\text{ms}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u127"></a>

## U127 — [sections/results.tex:40](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:40)

**Trước**

```latex
In real-time autonomous systems, evaluating neural network forward execution time in isolation yields a misleading depiction of deployability. A complete perception system must account for point cloud loading, input feature tensorization, Host-to-Device (H2D) memory transfer, forward execution, coordinate decoding, and Non-Maximum Suppression (NMS). Table~\ref{tab:efficiency} reports sequential latency at batch size 1 over $3{,}759$ timed frames after excluding 10 warmup frames from the $3{,}769$-frame validation split. MobilePIXORNeXt and the reference detectors were evaluated on Tesla T4; the historical MobilePIXOR result was measured on L4.
```

**Sau — đã áp dụng**

```latex
Network forward time covers only part of a real-time perception pipeline. Full-system latency includes point cloud loading, tensorization, Host-to-Device (H2D) transfer, network execution, coordinate decoding, and Non-Maximum Suppression (NMS). Table~\ref{tab:efficiency} reports sequential latency at batch size 1 over $3{,}759$ timed frames, excluding 10 warmup frames from the $3{,}769$-frame validation split. MobilePIXORNeXt and reference detectors were evaluated on Tesla T4; historical MobilePIXOR timing was measured on L4.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý
Protected: H2D; \ref{tab:efficiency}; 1; $3{,}759$; 10; $3{,}769$; T4; L4
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u128"></a>

## U128 — [sections/results.tex:45](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:45)

**Trước**

```latex
As detailed in Table~\ref{tab:efficiency}, MobilePIXORNeXt achieved an End-to-End (E2E) pipeline latency of \textbf{60.96\,ms} (\textbf{16.41\,FPS}), delivering the lowest full-system latency among all evaluated real-time detectors. It operated faster than PointPillars ($63.84\,\text{ms}$, $15.66\,\text{FPS}$), MobilePIXOR ($85.92\,\text{ms}$, $11.64\,\text{FPS}$), and SECOND ($93.70\,\text{ms}$, $10.67\,\text{FPS}$), and ran over $2.3\times$ faster than the original PIXOR ($144.00\,\text{ms}$, $6.94\,\text{FPS}$). Percentile analysis indicates consistent latency bounds in the evaluated environment: MobilePIXORNeXt exhibited a median P50 latency of \textbf{58.28\,ms}, P95 of $80.85\,\text{ms}$, and P99 of $95.91\,\text{ms}$, demonstrating low latency dispersion under desktop GPU benchmarking.
```

**Sau — đã áp dụng**

```latex
Table~\ref{tab:efficiency} reports MobilePIXORNeXt's End-to-End (E2E) latency of \textbf{60.96\,ms} (\textbf{16.41\,FPS}), the lowest among the evaluated real-time detectors. MobilePIXORNeXt ran faster than PointPillars ($63.84\,\text{ms}$, $15.66\,\text{FPS}$), MobilePIXOR ($85.92\,\text{ms}$, $11.64\,\text{FPS}$), and SECOND ($93.70\,\text{ms}$, $10.67\,\text{FPS}$). It also ran over $2.3\times$ faster than PIXOR ($144.00\,\text{ms}$, $6.94\,\text{FPS}$). Percentile analysis showed low latency dispersion in the evaluated desktop GPU environment: median P50 latency was \textbf{58.28\,ms}, P95 was $80.85\,\text{ms}$, and P99 was $95.91\,\text{ms}$.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \ref{tab:efficiency}; E2E; \textbf{60.96\,ms}; \textbf{16.41\,FPS}; $63.84\,\text{ms}$; $15.66\,\text{FPS}$; $85.92\,\text{ms}$; $11.64\,\text{FPS}$; $93.70\,\text{ms}$; $10.67\,\text{FPS}$; $2.3\times$; $144.00\,\text{ms}$; $6.94\,\text{FPS}$; P50; \textbf{58.28\,ms}; P95; $80.85\,\text{ms}$; P99; $95.91\,\text{ms}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u129"></a>

## U129 — [sections/results.tex:48](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:48)

**Trước**

```latex
Stage-by-stage profiling reveals the underlying mechanics of our throughput advantage:
```

**Sau — đã áp dụng**

```latex
Stage-level profiling identified the following sources of the throughput advantage:
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: Không có math/cite/ref/label/số trong đoạn; giữ nguyên thuật ngữ và lệnh LaTeX.
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u130"></a>

## U130 — [sections/results.tex:50](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:50)

**Trước**

```latex
    \item \textbf{Input Preprocessing ($20.94\,\text{ms}$)}: Point-cloud file loading and Strata-BEV14 rasterization through the Numba CPU backend together take $20.94\,\text{ms}$. The implementation uses compact occupied-cell accumulators and precomputed logarithmic lookup tables. The corresponding recorded preprocessing stages take $27.75\,\text{ms}$ for PointPillars and $27.20\,\text{ms}$ for SECOND; these are complete input-preparation timings rather than encoder-only microbenchmarks.
```

**Sau — đã áp dụng**

```latex
    \item \textbf{Input Preprocessing ($20.94\,\text{ms}$)}: Point-cloud file loading and Strata-BEV14 rasterization with the Numba CPU backend took $20.94\,\text{ms}$. The implementation uses occupied-cell accumulators and precomputed logarithmic lookup tables. Recorded preprocessing took $27.75\,\text{ms}$ for PointPillars and $27.20\,\text{ms}$ for SECOND. These measurements cover complete input preparation.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \textbf{Input Preprocessing ($20.94\,\text{ms}$)}; Strata-BEV14; $20.94\,\text{ms}$; $27.75\,\text{ms}$; $27.20\,\text{ms}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u131"></a>

## U131 — [sections/results.tex:51](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:51)

**Trước**

```latex
    \item \textbf{Host-to-Device Memory Transfer ($6.91\,\text{ms}$)}: Unlike 35-channel binary occupancy tensors that require transferring $78.8\,\text{MB}$ per scan, Strata-BEV14's compact 14-channel layout requires transferring only $31.5\,\text{MB}$ (a $60.0\%$ payload reduction), which, in conjunction with pinned-memory buffer optimization, reduces H2D latency to $6.91\,\text{ms}$ compared with $40.63\,\text{ms}$ for the MobilePIXOR baseline.
```

**Sau — đã áp dụng**

```latex
    \item \textbf{Host-to-Device Memory Transfer ($6.91\,\text{ms}$)}: Binary occupancy tensors with a 35-channel layout require $78.8\,\text{MB}$ per scan. The Strata-BEV14 14-channel layout required $31.5\,\text{MB}$, reducing payload by $60.0\%$. With pinned-memory buffers, H2D latency fell to $6.91\,\text{ms}$, compared with $40.63\,\text{ms}$ for MobilePIXOR.
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: \textbf{Host-to-Device Memory Transfer ($6.91\,\text{ms}$)}; 35-channel; $78.8\,\text{MB}$; Strata-BEV14; 14-channel; $31.5\,\text{MB}$; $60.0\%$; H2D; $6.91\,\text{ms}$; $40.63\,\text{ms}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u132"></a>

## U132 — [sections/results.tex:52](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:52)

**Trước**

```latex
    \item \textbf{Model Forward Propagation ($17.88\,\text{ms}$)}: Network inference executes in $17.88\,\text{ms}$ ($55.94\,\text{FPS}$) on Tesla T4. In optimized standalone inference, the network reaches \textbf{115.32\,FPS} ($8.67\,\text{ms}$).
```

**Sau — đã áp dụng**

```latex
    \item \textbf{Model Forward Propagation ($17.88\,\text{ms}$)}: Network inference took $17.88\,\text{ms}$ ($55.94\,\text{FPS}$) on Tesla T4. Optimized standalone inference reached \textbf{115.32\,FPS} ($8.67\,\text{ms}$).
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: \textbf{Model Forward Propagation ($17.88\,\text{ms}$)}; $17.88\,\text{ms}$; $55.94\,\text{FPS}$; T4; \textbf{115.32\,FPS}; $8.67\,\text{ms}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u133"></a>

## U133 — [sections/results.tex:53](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:53)

**Trước**

```latex
    \item \textbf{Decoding and Rotated NMS ($14.94\,\text{ms}$)}: Candidate peak selection and class-wise rotated NMS execute in $14.94\,\text{ms}$, maintaining GPU residency until final bounding box export.
```

**Sau — đã áp dụng**

```latex
    \item \textbf{Decoding and Rotated NMS ($14.94\,\text{ms}$)}: Candidate peak selection and class-wise rotated NMS took $14.94\,\text{ms}$. Data remained on the GPU until final bounding box export.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: \textbf{Decoding and Rotated NMS ($14.94\,\text{ms}$)}; $14.94\,\text{ms}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u135"></a>

## U135 — [sections/results.tex:70](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:70)

**Trước**

```latex
Within the near-range $0\text{--}30\,\text{m}$ evaluation zone, MobilePIXORNeXt delivered substantial accuracy gains over baseline MobilePIXOR: Pedestrian AP reached \textbf{65.23\%} (more than double MobilePIXOR's $30.45\%$, a $+34.78\%$ improvement), while Cyclist AP reached \textbf{76.32\%} (nearly triple MobilePIXOR's $25.55\%$, a $+50.77\%$ improvement). This improvement was facilitated by Strata-BEV14's stratified height bins and continuous subcell offsets, which preserve fine morphological boundaries when object returns are dense.
```

**Sau — đã áp dụng**

```latex
Compared with MobilePIXOR in the near-range $0\text{--}30\,\text{m}$ zone, MobilePIXORNeXt Pedestrian AP reached \textbf{65.23\%}, more than double MobilePIXOR's $30.45\%$ (a $+34.78\%$ improvement). Cyclist AP reached \textbf{76.32\%}, nearly triple MobilePIXOR's $25.55\%$ (a $+50.77\%$ improvement). These gains are consistent with Strata-BEV14's stratified height bins and continuous subcell offsets preserving fine boundaries when object returns are dense.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: $0\text{--}30\,\text{m}$; \textbf{65.23\%}; $30.45\%$; $+34.78\%$; \textbf{76.32\%}; $25.55\%$; $+50.77\%$; Strata-BEV14
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u136"></a>

## U136 — [sections/results.tex:73](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:73)

**Trước**

```latex
At medium ranges ($30\text{--}50\,\text{m}$), MobilePIXORNeXt maintained robust detection capability, achieving $11.49\%$ Pedestrian AP (vs. $3.75\%$) and $44.30\%$ Cyclist AP (vs. $12.61\%$, a $+31.69\%$ gain). In the far-range zone ($50\text{--}70.4\,\text{m}$), Cyclist AP remained at $14.94\%$ (vs. $0.56\%$). This resilience against severe point sparsity is consistent with the regularizing effect of our physics-aware range attenuation augmentation (\texttt{transform\_object}) and the broader spatial context captured by large $7 \times 7$ depthwise receptive fields.
```

**Sau — đã áp dụng**

```latex
At medium ranges ($30\text{--}50\,\text{m}$), MobilePIXORNeXt achieved $11.49\%$ Pedestrian AP (vs. $3.75\%$) and $44.30\%$ Cyclist AP (vs. $12.61\%$, a $+31.69\%$ gain). Far-range ($50\text{--}70.4\,\text{m}$) Cyclist AP was $14.94\%$ (vs. $0.56\%$). These results are consistent with range-adaptive augmentation (\texttt{transform\_object}) and the wider context of $7 \times 7$ depthwise receptive fields helping under sparse returns.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: $30\text{--}50\,\text{m}$; $11.49\%$; $3.75\%$; $44.30\%$; $12.61\%$; $+31.69\%$; $50\text{--}70.4\,\text{m}$; $14.94\%$; $0.56\%$; \texttt{transform\_object}; $7 \times 7$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u138"></a>

## U138 — [sections/results.tex:86](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:86)

**Trước**

```latex
Replacing MobilePIXOR's residual blocks with our ConvNeXt-style backbone ($7 \times 7$ depthwise convolutions, inverted bottleneck expansion $e=2.5$, SiLU activations, and LayerScale) elevated Moderate mAP from $41.40\%$ to $47.60\%$ (\textbf{+6.20\% mAP}), while nearly doubling standalone forward throughput from $90.57$ to \textbf{169.97\,FPS}. Large depthwise kernels significantly broaden spatial receptive fields on BEV grids without inflating computational complexity.
```

**Sau — đã áp dụng**

```latex
We replaced MobilePIXOR's residual blocks with a ConvNeXt-style backbone using $7 \times 7$ depthwise convolutions, inverted expansion $e=2.5$, SiLU, and LayerScale. Moderate mAP rose from $41.40\%$ to $47.60\%$ (\textbf{+6.20\% mAP}), and standalone throughput nearly doubled from $90.57$ to \textbf{169.97\,FPS}. Large depthwise kernels broaden BEV receptive fields with limited computational cost.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; tách hoặc rút gọn câu nhiều ý; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: $7 \times 7$; $e=2.5$; $41.40\%$; $47.60\%$; \textbf{+6.20\% mAP}; $90.57$; \textbf{169.97\,FPS}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u139"></a>

## U139 — [sections/results.tex:91](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:91)

**Trước**

```latex
Replacing 35-channel binary slicing (A1, $47.60\%$) with 8-channel RichBEV (A2, $47.37\%$) maintained comparable accuracy with fewer channels. Upgrading to our 14-channel Strata-BEV14 (A3) increased Moderate mAP to $50.14\%$ (\textbf{+2.54\% mAP}). Encoding continuous elevation moments ($z_{\max}, z_{\mathrm{mean}}, z_{\mathrm{span}}, z_{\mathrm{std}}$) and subcell centroid displacements eliminated horizontal discretization errors.
```

**Sau — đã áp dụng**

```latex
Replacing 35-channel binary slicing (A1, $47.60\%$) with 8-channel RichBEV (A2, $47.37\%$) maintained comparable accuracy with fewer channels. The 14-channel Strata-BEV14 representation (A3) then raised Moderate mAP to $50.14\%$ (\textbf{+2.54\% mAP}). Continuous elevation moments ($z_{\max}, z_{\mathrm{mean}}, z_{\mathrm{span}}, z_{\mathrm{std}}$) and subcell centroid displacements retained geometry otherwise lost through horizontal discretization.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: 35-channel; A1; $47.60\%$; 8-channel; A2; $47.37\%$; 14-channel; Strata-BEV14; A3; $50.14\%$; \textbf{+2.54\% mAP}; $z_{\max}, z_{\mathrm{mean}}, z_{\mathrm{span}}, z_{\mathrm{std}}$
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u140"></a>

## U140 — [sections/results.tex:95](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:95)

**Trước**

```latex
Partitioning prediction heads into separate Car and Ped/Cyc groups (A4) yielded an immediate gain to $51.30\%$ mAP over A3. Expanding grouped head capacity from 16 to 32 channels (A5) produced a further gain to $60.22\%$ Moderate mAP (\textbf{+8.92\% mAP}) and $61.11\%$ $\text{mAP}_9$ compared with configuration A4. Decoupling isolated class-specific representations, mitigating gradient interference between morphologically distinct categories such as passenger vehicles and pedestrians.
```

**Sau — đã áp dụng**

```latex
Separating Car and Ped/Cyc prediction heads (A4) raised mAP to $51.30\%$ over A3. Increasing grouped head width from 16 to 32 channels (A5) raised Moderate mAP to $60.22\%$ (\textbf{+8.92\% mAP}) and $61.11\%$ $\text{mAP}_9$ relative to configuration A4. Class separation may reduce gradient interference between categories with different footprints, such as vehicles and pedestrians.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: A4; $51.30\%$; A3; 16; 32; A5; $60.22\%$; \textbf{+8.92\% mAP}; $61.11\%$; $\text{mAP}_9$; A4
Meaning-Check: PRESERVED
Risk-Flags: overstatement
```

<a id="u141"></a>

## U141 — [sections/results.tex:100](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:100)

**Trước**

```latex
Transitioning from standard ground-truth sampling to our physics-consistent augmentation (\texttt{PC-GTA}) yielded a gain of \textbf{+6.03\% mAP} over configuration A5, advancing Moderate mAP from $60.22\%$ to $66.25\%$ and $\text{mAP}_9$ from $61.11\%$ to $68.45\%$ (\textbf{+7.34\% mAP}). The evaluated sampler combines terrain alignment, static obstacle rejection, and range-dependent point subsampling. Optional LOS, shadow-masking, and radiometric mechanisms were disabled in the final run.
```

**Sau — đã áp dụng**

```latex
Physics-consistent augmentation (\texttt{PC-GTA}) improved Moderate mAP by \textbf{+6.03\% mAP} over A5, from $60.22\%$ to $66.25\%$. It also raised $\text{mAP}_9$ from $61.11\%$ to $68.45\%$ (\textbf{+7.34\% mAP}). The evaluated sampler combined terrain alignment, static obstacle rejection, and range-dependent point subsampling. Optional LOS, shadow-masking, and radiometric mechanisms were disabled in the final run.
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại
Protected: \texttt{PC-GTA}; \textbf{+6.03\% mAP}; A5; $60.22\%$; $66.25\%$; $\text{mAP}_9$; $61.11\%$; $68.45\%$; \textbf{+7.34\% mAP}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u142"></a>

## U142 — [sections/results.tex:104](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:104)

**Trước**

```latex
Widening the neck fusion layer from 24 to 32 channels (A7) increased Moderate mAP to $68.12\%$ (\textbf{+1.87\% mAP}). Incorporating the Top-Down Scale-Gated FPN (A8) replaced transposed convolutions with bilinear upsampling and learnable depthwise scale gating, eliminating deconvolution checkerboard artifacts on continuous regression maps.
```

**Sau — đã áp dụng**

```latex
Increasing neck width from 24 to 32 channels (A7) raised Moderate mAP to $68.12\%$ (\textbf{+1.87\% mAP}). The Top-Down Scale-Gated FPN (A8) replaced transposed convolutions with bilinear upsampling and learnable depthwise gating to avoid deconvolution checkerboard artifacts in continuous regression maps.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: 24; 32; A7; $68.12\%$; \textbf{+1.87\% mAP}; A8
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u143"></a>

## U143 — [sections/results.tex:108](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:108)

**Trước**

```latex
Supervising the network with Oriented Geometric Alignment (OGA) and IoU-Aware Quality Alignment (IQA) drove Moderate mAP from $67.38\%$ to $70.58\%$ (\textbf{+3.20\% mAP}). Coupling Multi-Axis Projection GIoU with scale-normalized $\pi$-symmetric corner distances aligned loss gradients directly with rotated bounding box overlap, while IQA confidence calibration prevented high-scoring but misaligned false positives from surviving NMS.
```

**Sau — đã áp dụng**

```latex
Oriented Geometric Alignment (OGA) and IoU-Aware Quality Alignment (IQA) raised Moderate mAP from $67.38\%$ to $70.58\%$ (\textbf{+3.20\% mAP}). Multi-Axis Projection GIoU and scale-normalized $\pi$-symmetric corner distances aligned geometric supervision with rotated box overlap. IQA calibrated confidence against overlap to reduce the retention of high-scoring, misaligned boxes during NMS.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: $67.38\%$; $70.58\%$; \textbf{+3.20\% mAP}; $\pi$
Meaning-Check: PRESERVED
Risk-Flags: overstatement
```

<a id="u144"></a>

## U144 — [sections/results.tex:113](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:113)

**Trước**

```latex
Embedding parameter-free Block-SimAM inside every residual block across stages C2, C3, and C4 culminated in our final configuration, achieving \textbf{71.46\% Moderate mAP} and \textbf{73.42\% mAP$_9$} (\textbf{+0.88\% mAP}).
```

**Sau — đã áp dụng**

```latex
Embedding parameter-free Block-SimAM in every residual block across stages C2, C3, and C4 produced the final configuration, with \textbf{71.46\% Moderate mAP} and \textbf{73.42\% mAP$_9$} (\textbf{+0.88\% mAP}).
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: C2; C3; C4; \textbf{71.46\% Moderate mAP}; \textbf{73.42\% mAP$_9$}; \textbf{+0.88\% mAP}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u148"></a>

## U148 — [sections/results.tex:123](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:123)

**Trước**

```latex
As shown in Table~\ref{tab:loss}, introducing OGA alone into baseline training increased Moderate mAP from $67.38\%$ to $68.97\%$ (\textbf{+1.59\% mAP}), with consistent gains across Car ($+0.17\%$), Pedestrian ($+2.45\%$), and Cyclist ($+2.14\%$). Enabling IQA alone on the baseline elevated mAP to $70.31\%$ (\textbf{+2.93\% mAP}). Combining both OGA and IQA achieved the highest accuracy of \textbf{70.58\% Moderate mAP}, producing peak performance on Car ($84.87\%$) and Pedestrian ($59.63\%$). These findings demonstrate that OGA and IQA provide complementary benefits: OGA refines continuous corner and boundary alignment during backpropagation, whereas IQA calibrates classification scores with true geometric overlap during inference ($S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}}\hat{q}}$), preventing misaligned boxes from suppressing high-quality candidates during rotated NMS.
```

**Sau — đã áp dụng**

```latex
Table~\ref{tab:loss} reports the OGA and IQA ablation. With OGA, Moderate mAP rose from $67.38\%$ to $68.97\%$ (\textbf{+1.59\% mAP}), with gains for Car ($+0.17\%$), Pedestrian ($+2.45\%$), and Cyclist ($+2.14\%$). IQA alone raised mAP to $70.31\%$ (\textbf{+2.93\% mAP}). Together, OGA and IQA achieved \textbf{70.58\% Moderate mAP}, with the highest Car ($84.87\%$) and Pedestrian ($59.63\%$) scores. The gains are consistent with complementary roles: OGA supervises corner and boundary alignment during training; IQA calibrates scores against overlap during inference ($S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}}\hat{q}}$) to reduce suppression of well-localized candidates by misaligned boxes during rotated NMS.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: \ref{tab:loss}; $67.38\%$; $68.97\%$; \textbf{+1.59\% mAP}; $+0.17\%$; $+2.45\%$; $+2.14\%$; $70.31\%$; \textbf{+2.93\% mAP}; \textbf{70.58\% Moderate mAP}; $84.87\%$; $59.63\%$; $S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}}\hat{q}}$
Meaning-Check: PRESERVED
Risk-Flags: overstatement
```

<a id="u149"></a>

## U149 — [sections/results.tex:125](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:125)

**Trước**

```latex
The complexity metrics in Table~\ref{tab:loss} demonstrate an architectural trade-off between explicit multi-branch quality prediction and head-free quality coupling. Models incorporating the dedicated IQA prediction branch require $0.721\,\text{M}$ parameters and execute at $8.19\,\text{ms}$ forward latency ($122.16\,\text{FPS}$). In contrast, the head-free Q-OGA formulation (Eq.~(\ref{eq:cqfl_target})--(\ref{eq:cqfl_loss})) omits the auxiliary IQA heads, preserving the minimal $0.702\,\text{M}$ parameter budget (saving $18.59\,\text{K}$ parameters) and achieving a faster forward speed of $6.94\,\text{ms}$ ($144.16\,\text{FPS}$). Without any extra prediction branches or post-processing score multiplications, Q-OGA reaches $69.20\%$ Moderate mAP, outperforming standalone OGA by $+0.23\%$ mAP and surpassing the baseline Smooth-L1 by $+1.82\%$ mAP. These findings confirm that Q-OGA provides a parsimonious edge-deployment configuration when memory and latency budgets are strictly capped, whereas the explicit IQA configuration offers peak accuracy ($70.58\%$) when resources accommodate a dedicated quality branch.
```

**Sau — đã áp dụng**

```latex
Table~\ref{tab:loss} compares explicit quality prediction with head-free quality coupling. IQA models contained $0.721\,\text{M}$ parameters and required $8.19\,\text{ms}$ per forward pass ($122.16\,\text{FPS}$). Q-OGA (Eq.~(\ref{eq:cqfl_target})--(\ref{eq:cqfl_loss})) omitted the IQA heads, using $0.702\,\text{M}$ parameters (saving $18.59\,\text{K}$) and running at $6.94\,\text{ms}$ ($144.16\,\text{FPS}$). Without extra prediction branches or post-processing score multiplication, Q-OGA achieved $69.20\%$ Moderate mAP, exceeding OGA alone by $+0.23\%$ and baseline Smooth-L1 by $+1.82\%$. Q-OGA therefore offers a smaller, faster configuration for constrained deployment, while explicit IQA achieved the highest accuracy ($70.58\%$) with a dedicated quality branch.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; tách hoặc rút gọn câu nhiều ý
Protected: \ref{tab:loss}; $0.721\,\text{M}$; $8.19\,\text{ms}$; $122.16\,\text{FPS}$; \ref{eq:cqfl_target}; \ref{eq:cqfl_loss}; $0.702\,\text{M}$; $18.59\,\text{K}$; $6.94\,\text{ms}$; $144.16\,\text{FPS}$; $69.20\%$; $+0.23\%$; Smooth-L1; $+1.82\%$; $70.58\%$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u151"></a>

## U151 — [sections/results.tex:132](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:132)

**Trước**

```latex
Appending stage-level SimAM adapters at downsampling boundaries ($C_2, C_3, C_4$) degraded performance to $69.44\%$, $69.80\%$, and $69.36\%$ mAP, falling below the no-attention configuration ($70.58\%$). Because stage transitions undergo spatial downsampling, inserting global attention at these junctures blurs fine spatial gradients. In contrast, embedding Block-SimAM directly inside every residual block across all three stages achieved \textbf{71.46\% Moderate mAP} (\textbf{+0.88\% gain} over the no-attention configuration) and \textbf{73.42\% mAP$_9$}. Block-level placement modulates the intermediate residual feature update before LayerScale addition ($\mathbf{Z}' = \mathbf{Z} + \boldsymbol{\gamma}_{\mathrm{blk}} \odot \mathcal{S}(\mathbf{R})$), amplifying sparse LiDAR returns without disturbing inter-stage feature resolution.
```

**Sau — đã áp dụng**

```latex
Stage-level SimAM at downsampling boundaries ($C_2, C_3, C_4$) reduced mAP to $69.44\%$, $69.80\%$, and $69.36\%$, below the no-attention score ($70.58\%$). Attention at these transitions may blur fine spatial gradients after downsampling. Block-SimAM inside every residual block instead achieved \textbf{71.46\% Moderate mAP} (\textbf{+0.88\% gain} over no attention) and \textbf{73.42\% mAP$_9$}. It modulates the residual update before LayerScale addition ($\mathbf{Z}' = \mathbf{Z} + \boldsymbol{\gamma}_{\mathrm{blk}} \odot \mathcal{S}(\mathbf{R})$) without changing inter-stage feature resolution, allowing modulation of sparse returns within each block.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: $C_2, C_3, C_4$; $69.44\%$; $69.80\%$; $69.36\%$; $70.58\%$; \textbf{71.46\% Moderate mAP}; \textbf{+0.88\% gain}; \textbf{73.42\% mAP$_9$}; $\mathbf{Z}' = \mathbf{Z} + \boldsymbol{\gamma}_{\mathrm{blk}} \odot \mathcal{S}(\mathbf{R})$
Meaning-Check: PRESERVED
Risk-Flags: overstatement
```

<a id="u152"></a>

## U152 — [sections/results.tex:137](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:137)

**Trước**

```latex
    \caption{Multi-task training and validation loss trajectories across 50 epochs on the KITTI benchmark evaluated across ablation configurations A1 through A10 (Table~\ref{tab:components}). The top subplots depict total composite training loss (left) and validation loss (right) on logarithmic scales; the remaining six panels illustrate constituent task components: Classification Focal Loss, Oriented Geometry Loss (OGA), IoU Quality Loss (IQA), Subcell Centroid Offset Loss, Footprint Size Loss, and Doubled-Angle Yaw Loss. Legend curves decode the architectural progression from baseline A1 (Binary35) through Strata-BEV (A2--A3), Decoupled GroupedHeader (A4--A5), PC-GTA augmentation (A6), SG-FPN neck (A7--A8), OGA and IQA supervision (A9), to the final peak configuration A10 (Block-SimAM). Dynamic multi-task loss weighting via T-SBUW guarantees stable gradient optimization without loss divergence or oscillation, with configuration A10 consistently achieving the lowest asymptotic validation error bounds across all continuous regression objectives.}
```

**Sau — đã áp dụng**

```latex
    \caption{Training and validation loss trajectories over 50 epochs on KITTI for configurations A1 through A10 (Table~\ref{tab:components}). The top panels show total training loss (left) and validation loss (right) on logarithmic scales. Six task panels show Classification Focal Loss, Oriented Geometry Loss (OGA), IoU Quality Loss (IQA), Subcell Centroid Offset Loss, Footprint Size Loss, and Doubled-Angle Yaw Loss. The legend traces baseline A1 (Binary35), Strata-BEV (A2--A3), Decoupled GroupedHeader (A4--A5), PC-GTA (A6), SG-FPN (A7--A8), OGA and IQA (A9), and final configuration A10 (Block-SimAM). The plotted losses weighted by T-SBUW show no divergence or oscillation. Configuration A10 has the lowest late-training validation losses across the continuous regression tasks.}
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: 50; A1; A10; \ref{tab:components}; A1; Binary35; A2--A3; A4--A5; A6; A7--A8; A9; A10; A10
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u153"></a>

## U153 — [sections/results.tex:142](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:142)

**Trước**

```latex
To analyze empirical training stability and convergence behavior across the ablation sequence, Figure~\ref{fig:loss_curves} illustrates the training and validation loss trajectories across 50 epochs for the overall composite objective alongside the six constituent multi-task components (Classification, Geometry, IoU Quality, Subcell Offset, Size, and Yaw), tracking configurations A1 through A10.
```

**Sau — đã áp dụng**

```latex
Figure~\ref{fig:loss_curves} plots training and validation losses over 50 epochs for configurations A1 through A10. It includes the total objective and six tasks: Classification, Geometry, IoU Quality, Subcell Offset, Size, and Yaw. We used these trajectories to assess training stability and convergence.
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \ref{fig:loss_curves}; 50; A1; A10
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u154"></a>

## U154 — [sections/results.tex:143](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:143)

**Trước**

```latex
Across all 50 epochs, the aggregate loss and constituent task objectives converged smoothly without gradient oscillation or destabilizing loss spikes. Initial total loss dropped rapidly from $10^4$ down to approximately $1.0$ within the first 15 epochs, followed by steady cosine refinement toward convergence. The training and validation loss trajectories remained closely coupled throughout the optimization horizon without an expanding generalization gap, demonstrating that the physics-consistent point cloud augmentation (PC-GTA, A6) provides strong regularization without underfitting.
```

**Sau — đã áp dụng**

```latex
Across 50 epochs, total and task losses converged without visible oscillation or destabilizing spikes. Total loss fell from $10^4$ to approximately $1.0$ within the first 15 epochs, then decreased during cosine annealing. Training and validation curves remained close without a widening gap. This behavior is consistent with regularization from physics-consistent point cloud augmentation (PC-GTA, A6) without apparent underfitting.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: 50; $10^4$; $1.0$; 15; A6
Meaning-Check: PRESERVED
Risk-Flags: overstatement
```

<a id="u155"></a>

## U155 — [sections/results.tex:144](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:144)

**Trước**

```latex
The dynamic weighting governed by Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW, Eq.~(\ref{eq:tsbuw_weights})--(\ref{eq:balanced_loss})) maintained balanced loss scales across tasks, avoiding observable loss spikes between classification focal loss and coordinate regression objectives during training. As depicted across individual loss subplots, embedding Block-SimAM directly inside residual blocks (configuration A10) consistently maintained lower training and validation loss bounds across all regression tasks compared with intermediate ablation baselines (A1--A9), corroborating the quantitative gains observed in Table~\ref{tab:components}.
```

**Sau — đã áp dụng**

```latex
Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW, Eq.~(\ref{eq:tsbuw_weights})--(\ref{eq:balanced_loss})) balanced task loss scales without visible spikes between classification and coordinate regression. Block-SimAM (configuration A10) had lower training and validation losses across regression tasks than intermediate configurations (A1--A9), consistent with the accuracy gains in Table~\ref{tab:components}.
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: \ref{eq:tsbuw_weights}; \ref{eq:balanced_loss}; A10; A1--A9; \ref{tab:components}
Meaning-Check: PRESERVED
Risk-Flags: overstatement
```

<a id="u158"></a>

## U158 — [sections/results.tex:164](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:164)

**Trước**

```latex
Both criteria yielded nearly identical aggregate accuracy ($71.46\%$ vs. $71.48\%$ Moderate mAP). However, the Best AP checkpoint exhibited a $20.27\,\text{ms}$ lower end-to-end pipeline latency ($60.96\,\text{ms}$, $16.41\,\text{FPS}$ vs. $81.23\,\text{ms}$, $12.31\,\text{FPS}$). Extended training sharpens classification heatmaps around true positive centers, suppressing ambiguous low-confidence background peaks and accelerating candidate filtering and rotated NMS.
```

**Sau — đã áp dụng**

```latex
The two selection criteria yielded similar accuracy ($71.46\%$ vs. $71.48\%$ Moderate mAP). The Best AP checkpoint had $20.27\,\text{ms}$ lower end-to-end latency ($60.96\,\text{ms}$, $16.41\,\text{FPS}$ vs. $81.23\,\text{ms}$, $12.31\,\text{FPS}$). A possible explanation is that extended training sharpened classification heatmaps around positive centers, reducing ambiguous background peaks and the work required for candidate filtering and rotated NMS.
```

```text
[LLM]
Changed: đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: $71.46\%$; $71.48\%$; $20.27\,\text{ms}$; $60.96\,\text{ms}$; $16.41\,\text{FPS}$; $81.23\,\text{ms}$; $12.31\,\text{FPS}$
Meaning-Check: PRESERVED
Risk-Flags: overstatement
```

<a id="u165"></a>

## U165 — [sections/results.tex:190](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:190)

**Trước**

```latex
The empirical results indicate that high-accuracy 3D LiDAR object detection does not strictly necessitate parameter-heavy point-cloud encoders or computationally demanding 3D sparse convolutions. By enriching the deterministic 2D BEV representation with multi-band vertical densities, elevation dispersion moments, and continuous subcell centroid displacements, a streamlined ConvNeXt-style 2D convolutional network extracts expressive geometric features with fewer than $0.76\,\text{M}$ parameters. In addition, enforcing physical realism in data augmentation (\texttt{PC-GTA}) and resolving multi-task gradient competition through decoupled grouped heads provided larger empirical performance gains than model depth scaling alone.
```

**Sau — đã áp dụng**

```latex
The results showed that accurate 3D LiDAR detection was possible without large point-cloud encoders or 3D sparse convolutions. The deterministic 2D BEV representation retained multi-band densities, elevation dispersion, and continuous subcell centroid displacements. A ConvNeXt-style 2D network extracted geometric features with fewer than $0.76\,\text{M}$ parameters. Physics-consistent augmentation (\texttt{PC-GTA}) and decoupled grouped heads produced larger observed accuracy gains than model depth scaling alone.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; đổi phần thuật lại thao tác/kết quả sang quá khứ; giữ định nghĩa/caption ở hiện tại; tách hoặc rút gọn câu nhiều ý
Protected: 3D; 3D; 2D; 2D; $0.76\,\text{M}$; \texttt{PC-GTA}
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution
```

<a id="u166"></a>

## U166 — [sections/results.tex:193](/home/duyennh/AI_projects/research_lidar/Lidar/paper/sections/results.tex:193)

**Trước**

```latex
The current MobilePIXORNeXt architecture operates primarily on oriented 2D Bird's-Eye View footprints ($x, y, l, w, \psi$), which represents the optimal perception trade-off for autonomous vehicle motion planning and collision avoidance under strict edge-computing power envelopes. Although 3D bounding box estimation ($z, h$) is supported via our optional vertical prediction head, full 3D height estimation on heavily sloped terrains remains an active area of investigation. Future research will explore range-conditioned frequency modulations (RC-SGFPN) and hardware-specific TensorRT INT8 quantization for ultra-low-power micro-mobility microcontrollers.
```

**Sau — đã áp dụng**

```latex
MobilePIXORNeXt primarily predicts oriented 2D Bird's-Eye View footprints ($x, y, l, w, \psi$) for motion planning and collision avoidance under edge-computing constraints. An optional vertical head supports 3D bounding box estimation ($z, h$), but full 3D height estimation on heavily sloped terrain requires further investigation. Future work will explore range-conditioned frequency modulation (RC-SGFPN) and hardware-specific TensorRT INT8 quantization for low-power micro-mobility microcontrollers.
```

```text
[LLM]
Changed: bỏ từ khuếch đại hoặc thay bằng thông tin cụ thể; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; giới hạn kết luận vào bằng chứng quan sát; ghi cơ chế suy đoán dưới dạng giải thích có điều kiện
Protected: 2D; $x, y, l, w, \psi$; 3D; $z, h$; 3D; INT8
Meaning-Check: PRESERVED
Risk-Flags: lexical-substitution, overstatement
```

<a id="u168"></a>

## U168 — [tables/attention.tex:18](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/attention.tex:18)

**Trước**

```latex
{\footnotesize \textit{Note.} All configurations share the OGA loss, IQA quality head, Strata-BEV14 input, and PC-GTA data augmentation. Standalone forward latency is measured in FP32 on an NVIDIA Tesla T4 GPU on isolated batches. In the complete perception pipeline (Table~\ref{tab:efficiency}), model forward propagation operates at $55.94\,\text{FPS}$ ($17.88\,\text{ms}$) with an overall E2E pipeline speed of $16.41\,\text{FPS}$ ($60.96\,\text{ms}$). Stage-level variants append a single attention adapter at the stage boundary, whereas the proposed Block-SimAM embeds 3D parameter-free energy attention inside each of the nine \texttt{MobilePixorNeXt-S} residual blocks before LayerScale scaling.}
```

**Sau — đã áp dụng**

```latex
{\footnotesize \textit{Note.} All configurations share OGA, the IQA quality head, Strata-BEV14 input, and PC-GTA augmentation. Standalone forward latency is measured in FP32 on an NVIDIA Tesla T4 GPU using isolated batches. In the complete pipeline (Table~\ref{tab:efficiency}), forward propagation takes $55.94\,\text{FPS}$ ($17.88\,\text{ms}$), and overall E2E speed is $16.41\,\text{FPS}$ ($60.96\,\text{ms}$). Stage-level variants add one adapter at a stage boundary. Block-SimAM applies 3D parameter-free energy attention in each of the nine \texttt{MobilePixorNeXt-S} residual blocks before LayerScale scaling.}
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \textit{Note.}; Strata-BEV14; FP32; T4; \ref{tab:efficiency}; $55.94\,\text{FPS}$; $17.88\,\text{ms}$; E2E; $16.41\,\text{FPS}$; $60.96\,\text{ms}$; 3D; \texttt{MobilePixorNeXt-S}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u171"></a>

## U171 — [tables/components.tex:3](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/components.tex:3)

**Trước**

```latex
\caption{Systematic architectural ablation progression across the proposed design pillars on the KITTI validation benchmark using Moderate BEV AP $R_{40}$. Panel (a) outlines incremental component additions from baseline MobilePIXOR (A0) to the peak MobilePIXORNeXt configuration (A10); Panel (b) reports resulting detection accuracies ($\text{mAP}_{\mathrm{M}}$, $\text{mAP}_9$) and standalone model forward inference throughput (ms, FPS) measured on an NVIDIA Tesla T4 GPU. Best values are in \textbf{bold}, and second-best are \underline{underlined}.}
```

**Sau — đã áp dụng**

```latex
\caption{Architectural ablations on KITTI validation using Moderate BEV AP $R_{40}$. Panel (a) lists additions from baseline MobilePIXOR (A0) to the final MobilePIXORNeXt configuration (A10). Panel (b) reports accuracy ($\text{mAP}_{\mathrm{M}}$, $\text{mAP}_9$) and standalone forward latency and throughput (ms, FPS) on an NVIDIA Tesla T4 GPU. Best values are in \textbf{bold}, and second-best are \underline{underlined}.}
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý
Protected: $R_{40}$; A0; A10; $\text{mAP}_{\mathrm{M}}$; $\text{mAP}_9$; T4; \textbf{bold}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u172"></a>

## U172 — [tables/components.tex:45](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/components.tex:45)

**Trước**

```latex
{\footnotesize \textit{Note.} Step identifiers A0--A10 correspond to the experimental progression from baseline R01 to peak configuration. Standalone model forward latency is measured in FP32 on an NVIDIA Tesla T4 GPU on isolated batches excluding I/O and NMS. In the complete perception pipeline (Table~\ref{tab:efficiency}), model forward execution operates at $55.94\,\text{FPS}$ ($17.88\,\text{ms}$) due to runtime evaluation harness synchronization, with a full system E2E latency of $60.96\,\text{ms}$ ($16.41\,\text{FPS}$).}
```

**Sau — đã áp dụng**

```latex
{\footnotesize \textit{Note.} Step identifiers A0--A10 trace additions from baseline R01 to the final configuration. Standalone forward latency is measured in FP32 on an NVIDIA Tesla T4 GPU, excluding I/O and NMS. In the complete pipeline (Table~\ref{tab:efficiency}), forward execution takes $55.94\,\text{FPS}$ ($17.88\,\text{ms}$) with evaluation-harness synchronization. Full-system E2E latency is $60.96\,\text{ms}$ ($16.41\,\text{FPS}$).}
```

```text
[LLM]
Changed: bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp; tách hoặc rút gọn câu nhiều ý
Protected: \textit{Note.}; A0--A10; R01; FP32; T4; \ref{tab:efficiency}; $55.94\,\text{FPS}$; $17.88\,\text{ms}$; E2E; $60.96\,\text{ms}$; $16.41\,\text{FPS}$
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u174"></a>

## U174 — [tables/efficiency.tex:3](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/efficiency.tex:3)

**Trước**

```latex
\caption{Inference latency and throughput breakdown across pipeline stages on the KITTI validation benchmark (3,769 frames, with 3,759 timed after 10 warmup frames). Measurements use Tesla T4 except for the historical MobilePIXOR result on NVIDIA L4. All measurements reflect physical batch size 1 in FP32 precision. Model forward latency measures neural network execution alone; End-to-End (E2E) pipeline latency spans raw LiDAR file I/O, statistical rasterization, Host-to-Device (H2D) transfer, network inference, subcell coordinate decoding, and class-wise rotated NMS. Best values among compact models are in \textbf{bold}, and second-best are \underline{underlined}.}
```

**Sau — đã áp dụng**

```latex
\caption{Pipeline latency and throughput on KITTI validation (3,769 frames, including 3,759 timed frames after 10 warmup frames). Measurements use Tesla T4, except historical MobilePIXOR on NVIDIA L4. Batch size is 1, with FP32 precision. Forward latency covers network execution alone. End-to-End (E2E) latency includes raw LiDAR file I/O, statistical rasterization, Host-to-Device (H2D) transfer, network execution, coordinate decoding, and class-wise rotated NMS. Best values among compact models are in \textbf{bold}, and second-best are \underline{underlined}.}
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: 3,769; 3,759; 10; T4; L4; 1; FP32; E2E; H2D; \textbf{bold}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u175"></a>

## U175 — [tables/efficiency.tex:23](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/efficiency.tex:23)

**Trước**

```latex
{\footnotesize \textit{Note.} Model latency measures neural network execution time within the evaluation pipeline on Tesla T4. Under standalone batch inference without pipeline synchronization, model forward throughput reaches $115.32\,\text{FPS}$ ($8.67\,\text{ms}$, Table~\ref{tab:components}). End-to-end (E2E) pipeline latency spans raw LiDAR point cloud loading, deterministic statistical rasterization, Host-to-Device (H2D) memory transfer, forward propagation, subcell coordinate decoding, and class-wise rotated NMS filtering. Percentiles (P50, P95, P99) are computed over 3,759 timed frames after excluding 10 warmup frames. The historical MobilePIXOR row was measured on L4; its latency is retained with its original hardware provenance.}
```

**Sau — đã áp dụng**

```latex
{\footnotesize \textit{Note.} Model latency measures network execution in the evaluation pipeline on Tesla T4. Standalone throughput without pipeline synchronization reaches $115.32\,\text{FPS}$ ($8.67\,\text{ms}$, Table~\ref{tab:components}). End-to-end (E2E) latency includes LiDAR loading, statistical rasterization, Host-to-Device (H2D) transfer, network execution, coordinate decoding, and class-wise rotated NMS. Percentiles (P50, P95, P99) cover 3,759 timed frames after 10 warmup frames. Historical MobilePIXOR latency was measured on L4 and retains that hardware provenance.}
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: \textit{Note.}; T4; $115.32\,\text{FPS}$; $8.67\,\text{ms}$; \ref{tab:components}; E2E; H2D; P50; P95; P99; 3,759; 10; L4
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u176"></a>

## U176 — [tables/loss.tex:3](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/loss.tex:3)

**Trước**

```latex
\caption{Factorial evaluation of geometric supervision strategies (Baseline Smooth-L1 vs. OGA) and localization quality prediction (with and without the explicit IQA Head) alongside the head-free Q-OGA formulation on the KITTI validation benchmark at Moderate difficulty ($R_{40}$). Candidate detections in models with IQA are ranked during NMS via composite score $S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}} \hat{q}}$. Standalone forward latency is measured in FP32 on an NVIDIA Tesla T4 GPU. Best values are in \textbf{bold}, and second-best are \underline{underlined}.}
```

**Sau — đã áp dụng**

```latex
\caption{Geometric supervision (Baseline Smooth-L1 vs. OGA), localization quality prediction (with and without an IQA Head), and head-free Q-OGA on KITTI validation at Moderate difficulty ($R_{40}$). IQA models rank NMS candidates using $S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}} \hat{q}}$. Standalone forward latency is measured in FP32 on an NVIDIA Tesla T4 GPU. Best values are in \textbf{bold}, and second-best are \underline{underlined}.}
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: Smooth-L1; $R_{40}$; $S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}} \hat{q}}$; FP32; T4; \textbf{bold}
Meaning-Check: PRESERVED
Risk-Flags: none
```

<a id="u178"></a>

## U178 — [tables/setup.tex:3](/home/duyennh/AI_projects/research_lidar/Lidar/paper/tables/setup.tex:3)

**Trước**

```latex
\caption{Unified experimental specifications, architectural hyperparameters, and training configuration of the proposed MobilePIXORNeXt detector. All training runs use a deterministic single-seed setup ($\text{seed} = 42$) under PyTorch 2.11.0 with CUDA 13.0, and timing benchmarks are conducted on a dedicated NVIDIA Tesla T4 GPU (16\,GB GDDR6 VRAM) under batch size 1 in FP32 precision.}
```

**Sau — đã áp dụng**

```latex
\caption{Experimental settings, architecture hyperparameters, and training configuration for MobilePIXORNeXt. Training uses a deterministic single-seed setup ($\text{seed} = 42$) with PyTorch 2.11.0 and CUDA 13.0. Timing uses a dedicated NVIDIA Tesla T4 GPU (16\,GB GDDR6 VRAM), batch size 1, and FP32 precision.}
```

```text
[LLM]
Changed: tách hoặc rút gọn câu nhiều ý; bỏ câu dẫn/kết luận rập khuôn; dùng vị ngữ trực tiếp
Protected: $\text{seed} = 42$; 2.11.0; 13.0,; T4; 16; GDDR6; 1; FP32
Meaning-Check: PRESERVED
Risk-Flags: none
```
