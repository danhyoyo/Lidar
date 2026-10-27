# Original source snapshot (explicit concatenation for exact quote verification)

% SOURCE: MobilePIXORNeXt.tex
% Compile from this directory with build.ps1. Auxiliary files go to build/.
\RequirePackage{fix-cm}
\ifdefined\pdfminorversion\pdfminorversion=7\fi
\newcommand{\DoNotLoadEpstopdf}{}
\documentclass[twocolumn]{svjour3}
\smartqed
\journalname{Arabian Journal for Science and Engineering (AJSE)}
\usepackage[utf8]{inputenc}
\usepackage{graphicx,microtype,hyperref}
\usepackage{cuted,capt-of}
\usepackage{amsmath,amssymb,amsfonts,booktabs,cite,array}
\usepackage{placeins}
% Optional ORCID hyperlink macro (zero-dependency, clickable badge)
\newcommand{\orcidID}[1]{\unskip\,\href{https://orcid.org/#1}{\protect\scalebox{0.7}{\textbf{\textcolor[rgb]{0.65,0.81,0.22}{[iD]}}}}}
\input{sections/figure_helpers}
\begin{document}
\raggedbottom
\setlength{\emergencystretch}{1.5em}
\title{MobilePIXORNeXt: Compact LiDAR Object Detection with Oriented Geometry and Localization Quality %\thanks{Grants or other notes The article that should go on the front page should be
%placed here. General acknowledgments should be placed at the end of the article.}
}
% \subtitle{Do you have a subtitle?\\ If so, write it here.

%\titlerunning{Short form of title}        % if too long for running head

\author{
  \rightskip=0pt
  Duyen Huu Nguyen$^1$ \and
  Anh Duc Bui$^1$ \and
  Nhan Trong Phan$^1$ \and
  Hoang Ngoc Tran$^{*,1}$
}

\authorrunning{D. H. Nguyen et al.}

\institute{
  $^*$Correspondence: Hoang Ngoc Tran (\email{hoang2531992@gmail.com}) \\
  \at $^1$ FPT University (FPT), Can Tho, 94000, VietNam. \\
  \email{duyennhce200017@gmail.com; anhbdce200052@gmail.com; nhanptce200090@gmail.com; hoang2531992@gmail.com}
  % [Optional ORCID iDs - uncomment lines below and supply 16-digit iDs if desired]:
  % \\ \textbf{ORCID iDs:}
  % D. H. Nguyen: \url{https://orcid.org/0000-0000-0000-0000};
  % A. D. Bui: \url{https://orcid.org/0000-0000-0000-0000};
  % N. T. Phan: \url{https://orcid.org/0000-0000-0000-0000};
  % H. N. Tran: \url{https://orcid.org/0000-0000-0000-0000}
}

\date{Received: date / Accepted: date}
%The correct dates will be entered by the editor

\titlerunning{MobilePIXORNeXt}
\maketitle  


\begin{abstract}
Accurate 3D object perception from Light Detection and Ranging (LiDAR) point clouds is an important challenge for autonomous vehicles, where volumetric and pillar networks frequently exceed edge computational budgets. In this paper, we present MobilePIXORNeXt, an ultra-compact, proposal-free detector targeting oriented bounding footprints on the Special Euclidean (SE) manifold $SE(2)$ for real-time collision avoidance and path planning. With 753,569 deployable parameters (753,581 trainable parameters during training including 12 transient loss-balancing logits), the framework replaces heavy learned point encoders with Strata-BEV14, a deterministic Central Processing Unit (CPU) statistical rasterizer computing 14 feature channels in 20.94 milliseconds for combined raw LiDAR disk I/O and statistical rasterization. A single-branch ConvNeXt-style backbone with parameter-free 3D spatial-energy attention (Block-SimAM) extracts multi-scale features, while a Scale-Gated Feature Pyramid Network fuses representations for decoupled grouped prediction heads. The model is trained using Physics-Consistent Ground-Truth Augmentation and Oriented Geometric Alignment (OGA), balanced by 12-parameter Temperature-Softmax Bounded Uncertainty Weighting. On the KITTI validation split, MobilePIXORNeXt achieves 71.50\% Moderate mean Average Precision (mAP) and 73.45\% overall 9-cell mAP under the official 40-point interpolation protocol on an identical NVIDIA Tesla T4 evaluation platform. The model establishes the highest Pedestrian Moderate AP (60.78\%) among single-stage detectors, while using 84.41\% fewer parameters than PointPillars and 85.85\% fewer than SECOND. We also examine Q-OGA as a head-free variant achieving 69.20\% Moderate mAP. On an NVIDIA Tesla T4 GPU, the end-to-end pipeline operates in 60.96 milliseconds (16.41 frames per second). These results demonstrate that deterministic statistical encoding coupled with oriented geometric alignment enables compact, real-time perception on edge vehicular platforms.
\end{abstract}
\keywords{LiDAR object detection \and Bird's-eye view representation \and Compact convolutional neural networks \and Parameter-free attention \and Oriented bounding box regression}

\section{Introduction}\label{sec:introduction}
\input{sections/introduction}
\section{Related Work}\label{sec:related_work}
\input{sections/related_work}
\section{Proposed Method}\label{sec:method}
\input{sections/methodology}
\section{Experiments}\label{sec:experiments}
\input{sections/experiments}
\section{Results and Discussion}\label{sec:results}
\input{sections/results}
% Keep LaTeX's remaining column height after full-width result floats.
% \clearpage % Keep pending result floats before the closing sections.
% \makeatletter
% \global\@colht=\textheight
% \global\@colroom=\textheight
% \global\vsize=\textheight
% \makeatother
\section{Conclusion}\label{sec:conclusion}
\input{sections/conclusion}

\input{sections/declarations}
\bibliographystyle{spmpsci}
\begingroup
\raggedright
\bibliography{references}
\endgroup
\end{document}


% SOURCE: main.tex
% Compile from this directory with build.ps1. Auxiliary files go to build/.
\RequirePackage{fix-cm}
\ifdefined\pdfminorversion\pdfminorversion=7\fi
\newcommand{\DoNotLoadEpstopdf}{}
\documentclass[twocolumn]{svjour3}
\smartqed
\journalname{Arabian Journal for Science and Engineering (AJSE)}
\usepackage[utf8]{inputenc}
\usepackage{graphicx,microtype,hyperref}
\usepackage{cuted,capt-of}
\usepackage{amsmath,amssymb,amsfonts,booktabs,cite,array}
\usepackage{placeins}
% Optional ORCID hyperlink macro (zero-dependency, clickable badge)
\newcommand{\orcidID}[1]{\unskip\,\href{https://orcid.org/#1}{\protect\scalebox{0.7}{\textbf{\textcolor[rgb]{0.65,0.81,0.22}{[iD]}}}}}
\input{sections/figure_helpers}
\begin{document}
\raggedbottom
\setlength{\emergencystretch}{1.5em}
\title{MobilePIXORNeXt: Compact LiDAR Object Detection via Strata-BEV Encoding and Oriented Geometric Alignment
%MobilePIXORNeXt: Compact LiDAR Object Detection with Oriented Geometry and Localization Quality 
%\thanks{Grants or other notes The article that should go on the front page should be
%placed here. General acknowledgments should be placed at the end of the article.}
}
% \subtitle{Do you have a subtitle?\\ If so, write it here.

%\titlerunning{Short form of title}        % if too long for running head

\author{
  \rightskip=0pt
  Duyen Huu Nguyen$^1$ \and
  Anh Duc Bui$^1$ \and
  Nhan Trong Phan$^1$ \and
  Hoang Ngoc Tran$^{*,1}$
}

\authorrunning{D. H. Nguyen et al.}

\institute{
  $^*$Correspondence: Hoang Ngoc Tran (\email{hoang2531992@gmail.com}) \\
  \at $^1$ FPT University (FPT), Can Tho, 94000, VietNam. \\
  \email{duyennhce200017@gmail.com; anhbdce200052@gmail.com; nhanptce200090@gmail.com; hoang2531992@gmail.com}
  % [Optional ORCID iDs - uncomment lines below and supply 16-digit iDs if desired]:
  % \\ \textbf{ORCID iDs:}
  % D. H. Nguyen: \url{https://orcid.org/0000-0000-0000-0000};
  % A. D. Bui: \url{https://orcid.org/0000-0000-0000-0000};
  % N. T. Phan: \url{https://orcid.org/0000-0000-0000-0000};
  % H. N. Tran: \url{https://orcid.org/0000-0000-0000-0000}
}

% \date{Received: date / Accepted: date}
%The correct dates will be entered by the editor

\titlerunning{MobilePIXORNeXt}
\maketitle  


\begin{abstract}
Accurate 3D object perception from Light Detection and Ranging (LiDAR) point clouds is an important challenge for autonomous vehicles, where volumetric and pillar networks frequently exceed edge computational budgets. In this paper, we present MobilePIXORNeXt, an ultra-compact, proposal-free detector targeting oriented bounding footprints on the Special Euclidean (SE) manifold $SE(2)$ for real-time collision avoidance and path planning. With 753,569 trainable parameters, the framework replaces heavy learned point encoders with Strata-BEV14, a deterministic Central Processing Unit (CPU) statistical rasterizer computing 14 feature channels in 20.94 milliseconds for combined raw LiDAR disk I/O and statistical rasterization. A single-branch ConvNeXt-style backbone with parameter-free 3D spatial-energy attention (Block-SimAM) extracts multi-scale features, while a Scale-Gated Feature Pyramid Network fuses representations for decoupled grouped prediction heads. The model is trained using Physics-Consistent Ground-Truth Augmentation and Oriented Geometric Alignment (OGA), balanced by 12-parameter Temperature-Softmax Bounded Uncertainty Weighting. On the KITTI validation split, MobilePIXORNeXt achieves 71.50\% Moderate mean Average Precision (mAP) and 73.45\% overall 9-cell mAP using a local loader-aligned evaluator with 40-point interpolation. Equivalence between this evaluator and the reference KITTI evaluator has not been established; therefore, published AP differences are not presented as direct superiority claims. The model uses 84.41\% fewer parameters than PointPillars and 85.85\% fewer than SECOND, and reaches 60.78\% local Pedestrian Moderate AP. We also examine Q-OGA as a head-free variant achieving 69.20\% Moderate mAP. On an NVIDIA Tesla T4 GPU, the end-to-end pipeline operates in 60.96 milliseconds (16.41 frames per second). These results demonstrate that deterministic statistical encoding coupled with oriented geometric alignment enables compact, real-time perception on edge vehicular platforms.
\end{abstract}
\keywords{LiDAR object detection \and Bird's-eye view representation \and Compact convolutional neural networks \and Parameter-free attention \and Oriented bounding box regression}

\section{Introduction}\label{sec:introduction}
\input{sections/introduction}
\section{Related Work}\label{sec:related_work}
\input{sections/related_work}
\section{Proposed Method}\label{sec:method}
\input{sections/methodology}
\section{Experiments}\label{sec:experiments}
\input{sections/experiments}
\section{Results and Discussion}\label{sec:results}
\input{sections/results}
% Keep LaTeX's remaining column height after full-width result floats.
% % \FloatBarrier
% \makeatletter
% \global\@colht=\textheight
% \global\@colroom=\textheight
% \global\vsize=\textheight
% \makeatother
\section{Conclusion}\label{sec:conclusion}
\input{sections/conclusion}

\input{sections/declarations}
\bibliographystyle{spmpsci}
\begingroup
\raggedright
\bibliography{references}
\endgroup
\end{document}

% SOURCE: sections/abstract.tex
Accurate 3D object perception from Light Detection and Ranging (LiDAR) point clouds is an important challenge for autonomous vehicles, where volumetric and pillar networks frequently exceed edge computational budgets. In this paper, we present MobilePIXORNeXt, an ultra-compact, proposal-free detector targeting oriented bounding footprints on the Special Euclidean (SE) manifold $SE(2)$ for real-time collision avoidance and path planning. With 753,569 deployable parameters (753,581 trainable parameters during training including 12 transient loss-balancing logits), the framework replaces heavy learned point encoders with Strata-BEV14, a deterministic Central Processing Unit (CPU) statistical rasterizer computing 14 feature channels in 20.94 milliseconds for combined raw LiDAR disk I/O and statistical rasterization. A single-branch ConvNeXt-style backbone with parameter-free 3D spatial-energy attention (Block-SimAM) extracts multi-scale features, while a Scale-Gated Feature Pyramid Network fuses representations for decoupled grouped prediction heads. The model is trained using Physics-Consistent Ground-Truth Augmentation and Oriented Geometric Alignment (OGA), balanced by 12-parameter Temperature-Softmax Bounded Uncertainty Weighting. On the KITTI validation split, MobilePIXORNeXt achieves 71.50\% Moderate mean Average Precision (mAP) and 73.45\% overall 9-cell mAP using a local loader-aligned evaluator with 40-point interpolation. Equivalence between this evaluator and the reference KITTI evaluator has not been established; therefore, published AP differences are not presented as direct superiority claims. The model uses 84.41\% fewer parameters than PointPillars and 85.85\% fewer than SECOND, and reaches 60.78\% local Pedestrian Moderate AP. We also examine Q-OGA as a head-free variant achieving 69.20\% Moderate mAP. On an NVIDIA Tesla T4 GPU, the end-to-end pipeline operates in 60.96 milliseconds (16.41 frames per second). These results demonstrate that deterministic statistical encoding coupled with oriented geometric alignment enables compact, real-time perception on edge vehicular platforms.


% SOURCE: sections/conclusion.tex
% Conclusion Section for MobilePIXORNeXt
% Written according to Springer Nature - Arabian Journal for Science and Engineering (AJSE)
% Structure: Solved Problem & Architecture -> Evidence Recap & Empirical Milestones -> Constructive Limitations & Future Work

We presented MobilePIXORNeXt, an ultra-compact, proposal-free LiDAR object detector optimizing oriented bounding footprints on the $SE(2)$ manifold for real-time edge perception on resource-constrained platforms. The deterministic Strata-BEV14 rasterizer replaced learned PointNet pillar encoders and 3D sparse convolutions. It preserved continuous vertical distribution moments, material surface reflectivity, and horizontal subcell centroid displacements without learned parameters. A single-branch ConvNeXt-style backbone with parameter-free Block-SimAM 3D energy attention processed multi-scale spatial features. A Scale-Gated Feature Pyramid Network fused these features for decoupled grouped prediction towers. Training utilized Physics-Consistent Ground-Truth Augmentation (PC-GTA), Oriented Geometric Alignment (OGA), and IoU-Aware Quality Alignment (IQA), numerically balanced by Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW). For resource-critical scenarios, Quality-Coupled OGA (Q-OGA) was formulated to achieve head-free confidence calibration with zero parameter overhead.

On the KITTI validation split under a unified benchmark protocol, MobilePIXORNeXt achieves $71.50\%$ Moderate mAP and $73.45\%$ $\text{mAP}_9$, with only $753{,}569$ deployable parameters ($753{,}581$ trainable during training, $9.34\,\text{MB}$ checkpoint). MobilePIXORNeXt uses $84.41\%$ fewer parameters than PointPillars and $85.85\%$ fewer than SECOND, while setting the highest Pedestrian Moderate AP of $60.78\%$ among single-stage detectors. On an NVIDIA Tesla T4 GPU, the complete input-to-detection pipeline executed in $60.96\,\text{ms}$ ($16.41\,\text{FPS}$). In controlled ablations, PC-GTA improved Moderate mAP by $+6.03\,\text{pp}$, while OGA and explicit IQA contributed $+3.20\,\text{pp}$ through rotated geometric overlap alignment and confidence score calibration. Alternatively, head-free Q-OGA reached $69.20\%$ Moderate mAP while saving $18{,}626$ parameters ($+1.82\,\text{pp}$ over baseline Smooth-L1). These results support an integrated design combining deterministic statistical encoding, physically regularized data augmentation, and rotation-aware geometric supervision for compact LiDAR perception.

Several physical and operational boundaries remain. The primary detector is formulated for oriented 2D Bird's-Eye View footprint localization on the $SE(2)$ manifold, directly addressing collision avoidance clearance and Time-to-Collision margins. While an auxiliary vertical regression branch can be optionally appended to estimate 3D elevation and height $(z, h)$, localization on steep vertical grades, multi-level interchanges, and irregular non-planar terrain requires systematic off-road validation. In addition, the present benchmark utilizes 64-beam mechanical rotating LiDAR under fair-weather conditions; adapting Strata-BEV14 to ultra-sparse solid-state sensors (e.g., 16-beam or non-repetitive scan patterns) and adverse weather precipitation remains an open challenge. Future work will investigate range-conditioned scale gating (RC-SGFPN) with continuous Fourier frequency embeddings to compensate for radial point density decay, along with hardware-tailored TensorRT INT8 quantization for low-power micro-mobility microcontrollers.


% SOURCE: sections/declarations.tex
% Declarations and Acknowledgements for MobilePIXORNeXt
% Prepared according to Springer Nature - Arabian Journal for Science and Engineering (AJSE) submission guidelines

\begin{acknowledgements}
The authors acknowledge FPT University, Vietnam, for providing the computational infrastructure, GPU workstation clusters, and laboratory facilities used in this research.
\end{acknowledgements}

\section*{Statements and Declarations}

\subsubsection*{Funding}
The authors declare that no funds, grants, or other financial support were received during the preparation of this manuscript.

\subsubsection*{Competing Interests}
The authors declare that they have no competing financial or non-financial interests that are directly or indirectly related to the work submitted for publication.

\subsubsection*{Ethics Approval and Consent to Participate}
Not applicable. This study does not involve human participants, vertebrate animals, or clinical data.

\subsubsection*{Consent for Publication}
Not applicable. This manuscript does not contain any personal identifying information, individual patient records, or individual photographs.

\subsubsection*{Data Availability}
The 3D point cloud and camera calibration data analyzed during the current study are publicly available in the KITTI 3D Object Detection Benchmark repository at \url{http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d}.

\subsubsection*{Code Availability}
The source code, model configurations, and trained checkpoints supporting the findings of this study are available from the corresponding author upon reasonable request, and will be made publicly accessible in an open repository upon publication.

\subsubsection*{Use of Generative AI and AI-assisted Technologies}
During the preparation of this manuscript, generative AI tools were used solely for English grammar checking and stylistic polishing. All conceptual modeling, mathematical formulations, software algorithms, and experimental evaluations were conducted exclusively by the authors, who take full responsibility for the contents of this publication.

\subsubsection*{Author Contributions}
All authors contributed to the study conception, architecture design, and empirical evaluation.
\textbf{Duyen Huu Nguyen}: Conceptualization, model architecture design, core pipeline software development, and writing the original draft. 
\textbf{Anh Duc Bui}: Data preprocessing pipeline, physics-consistent augmentation (PC-GTA) design, experimental benchmarking, and writing the original draft. 
\textbf{Nhan Trong Phan}: Formal analysis, loss ablation studies (OGA, IQA, T-SBUW), quantitative visualization, and writing the original draft. 
\textbf{Hoang Ngoc Tran}: Research supervision, conceptual validation, project administration, critical revision, and final review of the manuscript. 
All authors read, reviewed, and approved the final manuscript for publication.


% SOURCE: sections/experiments.tex
\subsection{Dataset and Evaluation Protocol}
\label{sec:dataset_and_metrics}

\paragraph{Dataset and Partitioning.}
We evaluated the detector on the KITTI 3D Object Detection dataset~\cite{geiger2012kitti}. The dataset contains $7{,}481$ calibrated LiDAR scans with synchronized rectified camera images and 3D bounding box annotations from road scenes. We used the 50/50 split introduced by Chen et al.~\cite{chen20153d}, with $3{,}712$ training frames and $3{,}769$ validation frames. Ground-truth sampling databases were generated exclusively from the $3{,}712$ training frames to prevent validation data leakage.

\paragraph{Spatial Region of Interest (ROI) and Target Parameterization.}
Following Section~\ref{sec:problem_formulation}, we aligned the LiDAR coordinate frame with the vehicle body (forward $X$, leftward $Y$, upward $Z$). We restricted perception to the Cartesian Region of Interest:
\begin{equation}
    \begin{aligned}
        x &\in [0.0, 70.4]\,\text{m}, \\
        y &\in [-40.0, 40.0]\,\text{m}, \\
        z &\in [-2.5, 1.0]\,\text{m}.
    \end{aligned}
\end{equation}
We applied an open-boundary inward margin of $\epsilon = 0.001\,\text{m}$ along all three axes. A horizontal grid resolution of $\Delta_x = \Delta_y = 0.1\,\text{m}$ produced an input tensor of $H_{\mathrm{in}} \times W_{\mathrm{in}} = 800 \times 704$ cells. An output stride of $s_{\mathrm{out}} = 4$ gave feature maps of size $H_{\mathrm{out}} \times W_{\mathrm{out}} = 200 \times 176$. We evaluated Car, Pedestrian, and Cyclist detections.

\paragraph{Evaluation Metrics.}
We evaluated oriented 2D Bird's-Eye View (BEV) footprints on the KITTI validation set using 40-point interpolated Average Precision (AP $R_{40}$). This metric averages interpolated precision at 40 recall positions $\mathcal{R}_{40} = \{1/40, 2/40, \dots, 1.0\}$:
\begin{equation}
    \mathrm{AP}|_{R_{40}} = \frac{1}{40} \sum_{r \in \mathcal{R}_{40}} \max_{r' \ge r} p(r').
\end{equation}
We used nominal oriented BEV Intersection-over-Union (IoU) thresholds of $\tau_{\mathrm{IoU}} = 0.70$ for Car and $\tau_{\mathrm{IoU}} = 0.50$ for Pedestrian and Cyclist. We grouped detections into three difficulty regimes:
\begin{itemize}
    \item \textbf{Easy}: 2D bounding box height $\ge 40\,\text{px}$, maximum occlusion level = 0 (fully visible), maximum truncation $\le 15\%$.
    \item \textbf{Moderate}: 2D bounding box height $\ge 25\,\text{px}$, maximum occlusion level = 1 (partially occluded), maximum truncation $\le 30\%$.
    \item \textbf{Hard}: 2D bounding box height $\ge 25\,\text{px}$, maximum occlusion level = 2 (difficult to see), maximum truncation $\le 50\%$.
\end{itemize}
Following autonomous driving literature, \textbf{Moderate macro mAP} ($\mathrm{mAP}_{\mathrm{M}}$) was adopted as the primary accuracy summary metric:
\begin{equation}
    \mathrm{mAP}_{\mathrm{M}} = \frac{1}{3} \sum_{c \in \{\mathrm{Car}, \mathrm{Ped}, \mathrm{Cyc}\}} \mathrm{AP}_{c, \mathrm{M}}.
\end{equation}
We also reported the unweighted mean across all nine class-difficulty combinations:
\begin{equation}
    \mathrm{mAP}_{9} = \frac{1}{9} \sum_{c \in \mathcal{C}} \sum_{d \in \{\mathrm{E}, \mathrm{M}, \mathrm{H}\}} \mathrm{AP}_{c, d}.
\end{equation}
E, M, and H denote Easy, Moderate, and Hard. $\mathrm{mAP}_{9}$ averages the nine class--difficulty AP values; it does not denote AP at an IoU threshold of 0.9.

\paragraph{Unified Benchmark Protocol and Evaluation Environment.}
To ensure strict methodological fairness and eliminate discrepancies stemming from disparate evaluation implementations or hardware environments, all evaluated detectors in our benchmark comparison are evaluated under a unified, loader-aligned evaluation protocol. The evaluation code strictly adheres to the official KITTI devkit evaluation protocol (\texttt{kitti\_eval} protocol standard in OpenPCDet and mmdetection3d), computing the official 40-point interpolated Average Precision ($R_{40}$) metric across all nine class--difficulty evaluation cells on identical validation scenes ($3{,}769$ frames) using standard IoU thresholds ($0.70$ for Car, $0.50$ for Pedestrian and Cyclist). All latency and throughput measurements are benchmarked on the same dedicated NVIDIA Tesla T4 GPU platform (16\,GB GDDR6 VRAM, CUDA 13.0, PyTorch 2.11.0) under identical single-batch conditions ($\text{batch size} = 1$) and FP32 numerical precision.
\subsection{Implementation Details}
\label{sec:implementation_details}

\input{tables/setup}
Table~\ref{tab:setup} summarizes the primary implementation settings and architectural hyperparameters across all pipeline components.

\paragraph{Network and Model Configuration.}
MobilePIXORNeXt processes 14-channel Strata-BEV14 tensors. We used a two-layer $3 \times 3$ convolutional stem ($32$ channels, stride 2) and three backbone stages with channel widths $(48, 96, 128)$ and block depths $(3, 4, 2)$. The stages used \texttt{MobilePIXORNeXt-S} blocks with $7 \times 7$ depthwise convolutions, an inverted expansion ratio of $e=2.5$, SiLU activations, and LayerScale initialized to $10^{-5}$. Parameter-free Block-SimAM attention was embedded in all nine residual blocks. The Scale-Gated Feature Pyramid Network (SG-FPN) fused stages C4, C3, and C2 into a $F_{32}$ (32-channel) representation at stride 4 ($200 \times 176$). Decoupled GroupedHeader towers processed Car ($K_1=1$) and Pedestrian/Cyclist ($K_2=2$) independently, with an initial $3 \times 3$ Conv-BN-SiLU transformation per group. The detector contained $753{,}569$ deployable parameters ($753{,}581$ trainable parameters during training including 12 transient T-SBUW multi-task logits), with an FP32 checkpoint size of $9.34\,\text{MB}$.

\paragraph{Training Schedule and Optimization.}
We optimized the model with AdamW using fused CUDA kernels. The base learning rate was $\eta_0 = 7 \times 10^{-4}$, with a minimum of $\eta_{\min} = 1.0 \times 10^{-6}$. Weight decay of $\lambda = 0.001$ applied only to multidimensional weight matrices ($\ge 2\mathrm{D}$ tensors). Biases, Batch Normalization affine parameters, and task-uncertainty logits were exempt ($\lambda = 0.0$). Gradients were clipped to a maximum $L_2$ norm of $10.0$. The standard training schedule covered 100 epochs, ending at epoch 100, comprising 10 linear warmup epochs and 90 cosine annealing epochs down to $\eta_{\min}$. To isolate and quantify each architectural component under strictly identical computational budgets, all exploratory ablation experiments (A1--A10) were trained under a standardized 50-epoch schedule (4 linear warmup epochs and 46 cosine annealing epochs), and evaluated at their 50-epoch checkpoints. Following this ablation study, the resulting optimal architecture (A10, MobilePIXORNeXt) and the baseline detector (A0, MobilePIXOR) were trained for a full 100-epoch schedule (10 linear warmup epochs and 90 cosine annealing epochs down to $\eta_{\min}$) to achieve complete asymptotic convergence for the definitive benchmark comparisons against external state-of-the-art detectors (matching the 100-epoch training budget of the multi-seed robustness evaluations under seeds 43 and 44). Section~\ref{sec:ablations} validates schedule convergence by comparing the 100-epoch deployment model against the 50-epoch ablation checkpoint. Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW, Section~\ref{sec:losses_and_oga}) balanced the task losses. Section~\ref{sec:ablations} evaluates its empirical convergence behavior.

\paragraph{Data Augmentation Pipeline.}
Training used Physics-Consistent Ground-Truth Augmentation (\texttt{PC-GTA}; Section~\ref{sec:pc_gta}). The final reported run enabled local ground fitting, static obstacle clearance, range-adaptive point subsampling, and visibility protection. The static obstacle clearance threshold was set to $N_{\mathrm{thresh}} = 2$ points within the expanded footprint, and estimated ground elevation was bounded within $[-2.5, -0.8]\,\text{m}$. LOS rejection, shadow masking, and radiometric calibration were available as configurable options but disabled in the primary benchmark. Ground height was estimated within $R(d) = \operatorname{clip}(2.5 + 0.03\max(0, d - 15), 2.5, 4.0)\,\text{m}$. GT sampling was applied with probability $0.5$. Subsequent world transforms included flipping along the $X$-axis with probability $0.5$, rotation $\theta \sim \mathcal{U}(-45^\circ,45^\circ)$, and isotropic scaling $s \sim \mathcal{U}(0.95,1.05)$. World translation was applied with probability $0.5$, using independent Gaussian noise with standard deviations $(0.2,0.2,0.1)\,\text{m}$ along $(X,Y,Z)$.

\input{tables/comparison}

\paragraph{Batching, Precision, and Hardware Environment.}
Training used a physical batch size of 16 on a single GPU; validation used a batch size of 32. DataLoader processes used 6 persistent worker threads, pinned memory, and a prefetch factor of 2. Training was accelerated via Automatic Mixed Precision (AMP; utilizing Bfloat16 on Ampere/Ada cloud accelerators or FP16 on Turing). Loss evaluation, coordinate regression, and rotated NMS used full FP32 precision for numerical stability. Final MobilePIXORNeXt timing and accuracy evaluation used an NVIDIA Tesla T4 GPU (16\,GB GDDR6 VRAM, Turing architecture), an x86\_64 host CPU, Linux OS, and PyTorch 2.11.0 with CUDA 13.0. Peak allocated evaluation GPU memory was $98.44\,\text{MiB}$ ($103.22\,\text{MB}$ decimal). All evaluated detectors, including the MobilePIXOR baseline, are benchmarked on the same NVIDIA Tesla T4 GPU platform (detailed pipeline latency and throughput are analyzed in Section~\ref{sec:latency_analysis}). Primary comparative metrics and architectural ablations utilize deterministic evaluation ($\text{seed} = 42$) on the same validation split. In addition, multi-seed statistical robustness across distinct initializations ($\text{seeds} \in \{42, 43, 44\}$) is comprehensively analyzed in Table~\ref{tab:seed_robustness} (Section~\ref{sec:benchmark_comparison}).


% SOURCE: sections/figure_helpers.tex
% Drop a PDF at the stated figures/ path to replace its placeholder automatically.
\newcommand{\paperfigure}[4][width=\linewidth]{%
  \IfFileExists{figures/#2}{%
    \includegraphics[#1]{figures/#2}%
  }{%
    \fbox{\parbox[c][#3][c]{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{%
      \centering\small\textbf{Figure placeholder}\\[5pt]
      \texttt{\detokenize{#2}}\\[6pt]#4}}%
  }%
}
\newcommand{\draftnote}[1]{\par\smallskip\noindent\textit{Draft verification note: #1}\par\smallskip}

% Dedicated Algorithm Counter and Caption Support (Springer AJSE compatible)
\newcounter{algorithm}
\setcounter{algorithm}{0}
\renewcommand{\thealgorithm}{\arabic{algorithm}}
\makeatletter
\newcommand{\fnum@algorithm}{\textbf{Algorithm~\thealgorithm}}
\newcommand{\ext@algorithm}{loa}
\makeatother


% SOURCE: sections/introduction.tex
% Introduction Section for MobilePIXORNeXt
% Written according to Springer Nature - Arabian Journal for Science and Engineering (AJSE)
% Structure: 5-Paragraph Funnel (Context & Significance -> Existing Approaches & Bottlenecks -> Motivation & Key Insight -> Proposed Method Overview -> Main Contributions)

Three-dimensional (3D) spatial perception from Light Detection and Ranging (LiDAR) point clouds provides essential metric localization for autonomous navigation, mobile robotics, and intelligent transportation systems. Unlike passive optical cameras that suffer from projective depth ambiguity and ambient lighting fluctuations, LiDAR sensors emit laser pulses to capture metric geometric surfaces during day and night~\cite{geiger2012kitti}. Autonomous vehicle perception requires detecting, classifying, and orienting surrounding objects, such as passenger vehicles, pedestrians, and cyclists, within stringent latency constraints. In operational systems, end-to-end execution must generally remain below $100\,\text{ms}$ per LiDAR scan~\cite{yang2018pixor}. Deploying accurate 3D detectors on onboard edge platforms requires balancing spatial localization precision against computing hardware budgets. Embedded accelerators have restricted memory bandwidth, cache sizes, and arithmetic capacity under strict thermal envelopes. Compact LiDAR detectors must sustain high throughput while preserving geometric precision.

Existing 3D LiDAR detectors present distinct trade-offs between geometric precision and edge computational feasibility across several architectural paradigms. Volumetric 3D voxel architectures such as SECOND~\cite{yan2018second} introduce spatially sparse convolutions, but voxelization and 3D convolutions still entail substantial computation ($5.33\,\text{M}$ parameters and $93.7\,\text{ms}$ pipeline latency on Tesla T4). Heavyweight two-stage frameworks (e.g., PointRCNN~\cite{shi2019pointrcnn}, Part-$A^2$~\cite{shi2020part}, and PV-RCNN~\cite{shi2020pvrcnn}) combine point abstractions with voxel representations, but their heavy parameter counts ($4.0\text{--}63.8\,\text{M}$) and long pipeline latencies ($133.8\text{--}194.5\,\text{ms}$) restrict real-time deployment on power-constrained edge platforms. Point-based direct architectures such as PointNet++~\cite{qi2017pointnetplusplus} process raw coordinates directly without discretization, but iterative Farthest Point Sampling and ball query neighbor searches impose $O(N^2)$ memory bandwidth demands that saturate edge caches. In contrast, Range-View spherical projection methods such as LaserNet~\cite{meyer2019lasernet} project 3D point clouds onto cylindrical range images to apply standard 2D convolutions; however, spherical projection causes severe perspective scale dilation with distance and boundary blurring across foreground-background occlusions. Pillar-based detectors, including PointPillars~\cite{lang2019pointpillars}, PillarNeXt~\cite{li2023pillarnext}, and PillarNeSt~\cite{mao2023pillarnest}, project points into vertical columns and utilize PointNet encoders to construct 2D Bird's-Eye View (BEV) representations. They achieve higher execution speed than volumetric networks, but point gathering, dynamic memory allocation, and sparse scattering introduce runtime and PCIe memory transfer overhead. Proposal-free BEV architectures such as PIXOR~\cite{yang2018pixor} and MobilePIXOR~\cite{nguyen2024indoor} replace learned encoders with deterministic vertical occupancy grids. However, binary occupancy grids on uniform $0.1\,\text{m}$ grids discard continuous subcell centroid displacements and vertical elevation dispersion moments. This spatial quantization causes boundary staircasing and localization jitter, particularly for small objects. Recent extensions such as PillarHist~\cite{zhou2025pillarhist} and TriBand-BEV~\cite{khoshkdahan2026triband} explore vertical histograms or multi-band intensity slicing, but existing compact representations generally omit continuous subcell coordinate offsets alongside higher-order vertical spatial moments. Additional bottlenecks include synthetic placement artifacts from naive cut-and-paste data augmentation, restricted receptive fields in small convolutional backbones, and uncalibrated classification heatmaps that allow poorly oriented candidates to suppress well-localized boxes during Non-Maximum Suppression (NMS)~\cite{li2020gfl}. Compact perception systems must resolve these representation, augmentation, receptive field, and ranking challenges within a sub-million-parameter budget.

Vehicular collision avoidance and kinematic trajectory planning operate predominantly on the Special Euclidean manifold $SE(2)$ (planar translation and yaw heading). In on-road autonomous driving, horizontal clearance margins and Time-to-Collision (TTC) along the ground plane dictate the vast majority of critical control decisions. Therefore, predicting high-precision oriented 2D bounding footprints on a metric Bird's-Eye View (BEV) plane provides the most computationally direct formulation for edge vehicular safety. We investigate whether a deterministic, parameter-free BEV representation can retain fine sub-voxel geometry without learned point encoders. By combining multi-stratum vertical densities, continuous elevation moments, surface reflectivity statistics, and continuous horizontal centroid offsets, a deterministic tensor preserves spatial boundaries without encoder training overhead. Our core research question is: \textit{Can an ultra-compact LiDAR detector operating on a parameter-free BEV tensor approach the localization accuracy of heavy volumetric and pillar architectures while achieving higher end-to-end throughput on edge hardware?} To address this question, we recognize that compact backbones cannot succeed through isolated architectural scaling alone; rather, they require an end-to-end co-design between sensor physical modeling and geometric loss supervision. We pair this representation with a single-branch ConvNeXt-style backbone~\cite{liu2022convnext} employing $7 \times 7$ depthwise convolutions and parameter-free 3D spatial-energy attention (SimAM)~\cite{yang2021simam}, expanding physical receptive fields to cover complete passenger vehicles at stride 8 without multi-branch training overhead. Physics-consistent synthesis (PC-GTA) regularizes distant returns by enforcing local road elevation flushness, static obstacle clearance, and range-adaptive point density, while multi-axis geometric supervision (OGA) and localization-quality ranking (IQA) align training losses and detection confidence with rotated bounding box overlap. These design choices collectively narrow the performance gap between compact 2D BEV representations and multi-million-parameter 3D sparse networks.

We propose \textbf{MobilePIXORNeXt}, a compact, proposal-free, multi-class BEV LiDAR object detector with $753{,}569$ deployable detector parameters ($753{,}581$ trainable parameters during training including 12 transient loss-balancing logits). Figure~\ref{fig:r37_overview} illustrates the end-to-end pipeline.

\begin{figure*}[t]
    \centering
    \paperfigure[width=0.88\linewidth]{pipeline_overview.pdf}{32mm}{Overall architecture and data flow of MobilePIXORNeXt. A raw LiDAR point cloud is converted via the parameter-free Strata-BEV14 rasterizer into a 14-channel statistical tensor. The compact ConvNeXt-style backbone with Block-SimAM extracts multi-scale spatial features, which are fused by the Scale-Gated Feature Pyramid Network (SG-FPN). Decoupled grouped prediction towers output heatmaps, subcell offsets, log-sizes, doubled yaw angles, and localization quality. In training, the network is supervised by Oriented Geometric Alignment (OGA) and IoU-Aware Quality Alignment (IQA), dynamically balanced via Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW).}
    \caption{End-to-end architecture and tensor flow of MobilePIXORNeXt. Unordered 3D LiDAR point clouds pass through the parameter-free Strata-BEV14 rasterizer into a 14-channel statistical BEV tensor $\mathbf{X} \in \mathbb{R}^{B \times 14 \times 800 \times 704}$ without learned pillar encoders. ConvNeXt-style stages with parameter-free Block-SimAM attention extract multi-scale features. A top-down Scale-Gated Feature Pyramid Network (SG-FPN) fuses them into a 32-channel shared representation at stride 4 ($200 \times 176$). Decoupled GroupedHeader towers process Car and Pedestrian/Cyclist categories independently. Solid arrows show the feedforward inference path. Dashed arrows show training-only Oriented Geometric Alignment (OGA), IoU-Aware Quality Alignment (IQA), and Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW).}
    \label{fig:r37_overview}
\end{figure*}

The parameter-free \textbf{Strata-BEV14} rasterizer converts an unordered point cloud into a 14-channel statistical BEV feature map ($800 \times 704$ at $0.1\,\text{m}$ grid resolution) using CPU execution. Point-cloud loading and rasterization take $20.94\,\text{ms}$ in the profiled pipeline. The 14 channels capture vertical stratum densities, continuous elevation moments ($z_{\max}, z_{\mathrm{mean}}, z_{\mathrm{span}}, z_{\mathrm{std}}$), reflectance statistics, and subcell centroid displacement offsets $(\Delta\bar{x}, \Delta\bar{y})$ to mitigate quantization errors. \textbf{Physics-Consistent Ground-Truth Augmentation (PC-GTA)} enforces local road elevation fitting, static obstacle clearance, and range-adaptive point downsampling to prevent unphysical synthesis artifacts. The \textbf{MobilePIXORNeXt-S} backbone extracts multi-scale features across stages $C_2, C_3, C_4$ using $7 \times 7$ depthwise inverted bottleneck blocks ($e=2.5$), SiLU activations, LayerScale stabilization, and parameter-free \textbf{Block-SimAM} energy attention embedded within all nine residual blocks. The top-down \textbf{Scale-Gated Feature Pyramid Network (SG-FPN Neck, $F_{32}$)} balances semantic abstraction and fine spatial details via bilinear upsampling, depthwise refinement at stride 8, standard projection at stride 4, and learnable depthwise scale gates, avoiding transposed-convolution checkerboard patterns. Car ($K_1=1$) and Pedestrian/Cyclist ($K_2=2$) predictions are decoupled into independent \textbf{GroupedHeader} towers sharing an initial $3 \times 3$ Conv-BN-SiLU block. Training utilizes \textbf{Oriented Geometric Alignment (OGA)}, which combines Multi-Axis Projection Generalized IoU (MGIoU)~\cite{le2026mgiou} with scale-normalized $\pi$-symmetric corner distance, balanced across tasks by 12-parameter \textbf{Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW)}. At inference, \textbf{IoU-Aware Quality Alignment (IQA)} evaluates composite detection scores $s_{\mathrm{det}} = (p_{\mathrm{cls}})^{1-\alpha} \hat{q}^\alpha$ prior to class-wise rotated Non-Maximum Suppression (NMS). For memory-critical platforms, we also present \textbf{Q-OGA}, a head-free quality-coupled objective that eliminates the separate IQA prediction tower.

The principal contributions of this work are summarized as follows:
\begin{itemize}
    \item \textbf{Ultra-compact proposal-free BEV detection architecture.} We design \textbf{MobilePIXORNeXt}, a compact ($753{,}569$ deployable parameters, $753{,}581$ trainable) LiDAR detector that eliminates learned PointNet encoders and 3D sparse convolutions. The architecture integrates the 14-channel deterministic \textbf{Strata-BEV14} representation with the \textbf{MobilePIXORNeXt-S} backbone and parameter-free \textbf{Block-SimAM} energy attention. The \textbf{SG-FPN} neck ($F_{32}$) fuses multi-scale features for decoupled \textbf{GroupedHeader} towers, resolving scale competition across road object classes.
    
    \item \textbf{Physics-consistent augmentation, geometric supervision, and head-free alternative.} We develop \textbf{PC-GTA} to enforce terrain alignment, obstacle clearance, and distance-dependent point subsampling during training. We formulate \textbf{OGA}, coupling Multi-Axis Projection GIoU with scale-normalized $\pi$-symmetric corner supervision, balanced across task objectives via 12-parameter \textbf{T-SBUW}. Confidence scores are calibrated via \textbf{IQA} quality alignment. In addition, we develop \textbf{Q-OGA} as an ultra-compact, head-free alternative that integrates localization quality directly into classification targets, saving $18{,}626$ parameters while attaining $69.20\%$ Moderate mAP.
    
    \item \textbf{Empirical validation and efficiency benchmarking.} On the KITTI validation benchmark under a unified evaluation protocol, MobilePIXORNeXt achieves \textbf{71.50\% Moderate mAP} and \textbf{73.45\% $\text{mAP}_9$} with only \textbf{0.754\,M parameters} ($9.34\,\text{MB}$ checkpoint). The model uses \textbf{84.41\% fewer parameters} than PointPillars and \textbf{85.85\% fewer} than SECOND, while achieving the top score of \textbf{60.78\% Moderate AP} on vulnerable Pedestrians among all single-stage detectors. The complete end-to-end pipeline executes in \textbf{60.96\,ms} (\textbf{16.41\,FPS}) on an NVIDIA Tesla T4 GPU. In ablation studies, PC-GTA improves accuracy by \textbf{+6.03\,pp} ($60.22\%$ to $66.25\%$), OGA and IQA add \textbf{+3.20\,pp}, and Block-SimAM adds \textbf{+0.92\,pp} with zero additional parameters.
\end{itemize}

Section~\ref{sec:related_work} reviews LiDAR representations, lightweight backbones, and geometric losses. Section~\ref{sec:method} details the MobilePIXORNeXt architecture and loss formulations. Section~\ref{sec:experiments} specifies experimental settings. Section~\ref{sec:results} reports benchmark comparisons, latency breakdowns, ablation studies, and qualitative results. Section~\ref{sec:conclusion} concludes the paper.


% SOURCE: sections/methodology.tex
% Methodology Section for MobilePIXORNeXt
% Written according to Springer Nature - Arabian Journal for Science and Engineering (AJSE)
% Structure: Module Triad (Motivation -> Design -> Technical Advantage) + Equation Closure + Pseudocode Algorithm

MobilePIXORNeXt is a compact, proposal-free, multi-class Bird's-Eye View (BEV) object detector for real-time Light Detection and Ranging (LiDAR) perception on resource-constrained embedded platforms, targeting oriented bounding footprints on the Special Euclidean manifold $SE(2)$. Its deterministic Bird's-Eye View representation avoids learned point-cloud encoders and 3D sparse convolutions. Figure~\ref{fig:r37_overview} shows the data flow from the raw 3D point cloud to decoded oriented BEV bounding footprints.

\par\vspace{-0.5\baselineskip}
\subsection{Problem Formulation and Input/Output Contracts}
\label{sec:problem_formulation}

\paragraph{LiDAR Coordinate Frame and Spatial Region of Interest.}
An unordered LiDAR point cloud acquired by a rotating scanner is represented as $\mathcal{P} = \{\mathbf{p}_i\}_{i=1}^{N_{\mathrm{pts}}}$. Each point $\mathbf{p}_i = (x_i, y_i, z_i, \iota_i) \in \mathbb{R}^4$ contains longitudinal distance $x_i$ (forward positive), lateral distance $y_i$ (leftward positive), vertical elevation $z_i$ (upward positive), and calibrated reflectance intensity $\iota_i \in [0, 1]$. Perception is restricted to a Cartesian Region of Interest (ROI):
\begin{equation}
    \begin{aligned}
        x_i &\in [x_{\min}, x_{\max}], \\
        y_i &\in [y_{\min}, y_{\max}], \\  z_i &\in [z_{\min}, z_{\max}].
    \end{aligned}
    \label{eq:spatial_roi}
\end{equation}
We used longitudinal bounds $[0.0, 70.4]$\,m, lateral bounds $[-40.0, 40.0]$\,m, and vertical bounds $[-2.5, 1.0]$\,m. An open-boundary inward margin of $\epsilon = 0.001$\,m reduced boundary artifacts. A horizontal resolution of $\Delta_x = \Delta_y = 0.1$\,m gave a grid of $H_{\mathrm{in}} \times W_{\mathrm{in}} = 800 \times 704$ cells. A point $\mathbf{p}_i$ maps to grid coordinate $(u_i, v_i)$ as:
\begin{equation}
    u_i = \left\lfloor \frac{x_i - x_{\min}}{\Delta_x} \right\rfloor, \qquad v_i = \left\lfloor \frac{y_i - y_{\min}}{\Delta_y} \right\rfloor.
    \label{eq:grid_assignment}
\end{equation}

\paragraph{Detection Target Contract.}
The detector predicts oriented bounding boxes projected onto the horizontal BEV plane across three semantic categories: Car ($c=0$), Pedestrian ($c=1$), and Cyclist ($c=2$). At an output spatial downsampling factor of $s_{\mathrm{out}} = 4$, the output feature map resolution is $H_{\mathrm{out}} \times W_{\mathrm{out}} = 200 \times 176$. For each detected object instance $k$, the target state is parameterized by the 7-tuple:
\begin{equation}
    \mathbf{b}_k = (c_k, x_k^*, y_k^*, l_k^*, w_k^*, \psi_k^*, s_k),
    \label{eq:target_tuple}
\end{equation}
where $(x_k^*, y_k^*) \in \mathbb{R}^2$ denotes continuous metric center coordinates and $(l_k^*, w_k^*) \in \mathbb{R}^2$ denotes length and width. The heading $\psi_k^* \in [-\pi/2, \pi/2]$ uses a $\pi$-symmetric doubled-angle representation, and $s_k \in [0, 1]$ is the calibrated confidence score. These targets require a dense BEV representation that retains fine 3D geometry without a learned point encoder.

\par\vspace{-0.5\baselineskip}
\subsection{Deterministic Strata-BEV14 Feature Representation}
\label{sec:strata_bev14}

\par\vspace{-0.5\baselineskip}
\noindent
\begin{minipage}{\columnwidth}
    \centering
    \paperfigure[width=\linewidth,trim=40mm 27mm 25mm 26mm,clip]{strata_bev14.pdf}{10mm}{Visual representation of the fourteen deterministic feature channels in Strata-BEV14 for a representative LiDAR scene. Channels 0–3 capture stratified height-band densities, Channels 4–7 encode vertical elevation statistics (peak, mean, span, deviation), Channels 8–9 capture material surface reflectance, Channels 10–11 represent global density and occupancy, and Channels 12–13 encode continuous subcell horizontal offsets to eliminate grid discretization error.}
    \captionof{figure}{Fourteen deterministic feature channels in the Strata-BEV14 representation for a KITTI driving scene ($800 \times 704$ grid at $0.1\,\text{m}$ resolution). Channels 0--3 encode logarithmic point densities across four vertical strata ($[-2.5, 1.0]\,\text{m}$). Channels 4--7 encode normalized maximum elevation $z_{\max}$, mean elevation $z_{\mathrm{mean}}$, elevation span $z_{\mathrm{span}}$, and population standard deviation $z_{\mathrm{std}}$. Channels 8--9 encode maximum and mean reflectance intensity. Channel 10 records total cell logarithmic density, and Channel 11 records binary occupancy. Channels 12--13 preserve continuous horizontal subcell centroid displacements $(\Delta\bar{x}, \Delta\bar{y}) \in [-0.5, 0.5]$ to reduce discretization error. Brighter values indicate higher point concentrations, reflectance, or vertical dispersion.}
    \label{fig:strata_bev14}
\end{minipage}
\par

\paragraph{Motivation.}
Voxel- and pillar-based detectors~\cite{lang2019pointpillars,li2023pillarnext} use PointNet-style column encoders, which add computation and padding overhead. 35-channel binary slicing~\cite{yang2018pixor} omits continuous height and reflectance cues, while three-channel height bands~\cite{khoshkdahan2026triband} omit higher-order geometry. We developed \textbf{Strata-BEV14} (Figure~\ref{fig:strata_bev14}), a deterministic, parameter-free rasterizer, to retain these features with low computational cost.


\paragraph{Mathematical Formulation.}
For each occupied cell $(u, v)$ with $N$ points $\mathcal{P}_{u, v} = \{\mathbf{p}_i\}_{i=1}^N$, we normalized vertical elevations to $\zeta_i = (z_i - z_{\min}) / (z_{\max} - z_{\min}) \in [0, 1]$ and clamped reflectance values $\iota_i$ to $[0, 1]$. Strata-BEV14 computes fourteen analytical features in four groups (Table~\ref{tab:strata_bev14}):

1. \textit{Height-Stratified Density Channels ($F_0$--$F_3$)}: Normalized height $\zeta_i$ is partitioned into four uniform vertical intervals $b \in \{0, 1, 2, 3\}$. For bin point count $N_b$, a logarithmic compression is applied to mitigate LiDAR ray density fall-off over range:
\begin{equation}
    F_b = d_{\log}(N_b) = \min\left(1, \frac{\ln(1 + N_b)}{\ln(1 + N_{\mathrm{norm}})}\right),
    \label{eq:density_encoding}
\end{equation}
where $N_{\mathrm{norm}} = 32$ is a normalization constant chosen to saturate at dense returns.

2. \textit{Vertical Spatial Moments ($F_4$--$F_7$)}: To encode obstacle geometry without explicit voxelization, we extract peak elevation $F_4 = \max_i \zeta_i$, mean elevation $F_5 = \frac{1}{N}\sum_i \zeta_i$, vertical span $F_6 = \max_i \zeta_i - \min_i \zeta_i$, and the population standard deviation:
\begin{equation}
    F_7 = \begin{cases}
        \sqrt{\max\left(0, \frac{1}{N} \sum_{i=1}^N \zeta_i^2 - F_5^2\right)}, & \text{if } N > 1, \\
        0, & \text{otherwise}.
    \end{cases}
    \label{eq:z_std_encoding}
\end{equation}

3. \textit{Reflectance Statistics ($F_8$--$F_9$)}: Calibrated surface returns capture material discrimination (e.g., distinguishing retro-reflective license plates, pedestrian clothing, or road signs):
\begin{equation}
    F_8 = \max_{i=1,\dots,N} \iota_i, \qquad F_9 = \frac{1}{N} \sum_{i=1}^N \iota_i.
    \label{eq:intensity_encoding}
\end{equation}

4. \textit{Occupancy and Continuous Subcell Offsets ($F_{10}$--$F_{13}$)}: Channel $F_{10} = d_{\log}(N)$ records total point density, and $F_{11} = 1.0$ indicates occupancy. For $0.1$\,m cells, channels $F_{12}$ and $F_{13}$ retain continuous horizontal subcell centroid displacements to reduce quantization error:
\begin{align}
    x_u^{\mathrm{cell}} &= x_{\min} + (u + 0.5)\Delta_x, \nonumber\\
    y_v^{\mathrm{cell}} &= y_{\min} + (v + 0.5)\Delta_y, \nonumber\\
    F_{12} &= \operatorname{clip}\left(\frac{\sum_{i=1}^N (x_i - x_u^{\mathrm{cell}})}{N\Delta_x}, -0.5, 0.5\right), \nonumber\\
    F_{13} &= \operatorname{clip}\left(\frac{\sum_{i=1}^N (y_i - y_v^{\mathrm{cell}})}{N\Delta_y}, -0.5, 0.5\right).
    \label{eq:offset_encoding}
\end{align}
Empty cells are assigned zeros across all 14 channels, producing the structured input tensor $\mathbf{X} \in \mathbb{R}^{B \times 14 \times 800 \times 704}$.

\begin{table}[t]
    \caption{Mathematical formulations and physical semantics of the fourteen deterministic feature channels in the Strata-BEV14 representation. All statistical moments and spatial offsets are evaluated deterministically over LiDAR points within horizontal grid cells of size $\Delta_x = \Delta_y = 0.1\,\text{m}$ ($800 \times 704$ grid) using NumPy or Numba on CPU without learned parameters.}
    \label{tab:strata_bev14}
    \centering\small
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{@{}c >{\raggedright\arraybackslash}p{0.20\columnwidth} >{\raggedright\arraybackslash}p{0.55\columnwidth}@{}}
        \toprule
        \textbf{Channel} & \textbf{Feature Identity} & \textbf{Mathematical Definition and Semantics} \\
        \midrule
        $0$--$3$ & Stratified Log Counts & Four height-band densities: $F_b = d_{\log}(N_b)$, where $b = \min(\lfloor 4\zeta_i \rfloor, 3)$. \\
        $4$ & Maximum Elevation & $F_4 = \max_{i} \zeta_i$, peak normalized elevation. \\
        $5$ & Mean Elevation & $F_5 = \frac{1}{N} \sum_{i=1}^N \zeta_i$, first vertical moment. \\
        $6$ & Elevation Span & $F_6 = \max_{i} \zeta_i - \min_{i} \zeta_i$, continuous vertical extent. \\
        $7$ & Elevation Deviation & $F_7 = \sqrt{\max(0, \frac{1}{N} \sum_{i} \zeta_i^2 - F_5^2)}$, spatial dispersion. \\
        $8$ & Peak Reflectance & $F_8 = \max_{i} \iota_i$, highest material reflectivity. \\
        $9$ & Mean Reflectance & $F_9 = \frac{1}{N} \sum_{i=1}^N \iota_i$, average surface intensity. \\
        $10$ & Global Density & $F_{10} = d_{\log}(N)$, overall point density. \\
        $11$ & Binary Occupancy & $F_{11} = 1.0$ if $N \ge 1$, and $0.0$ otherwise. \\
        $12$--$13$ & Centroid Displacements & Subcell offsets $\Delta \bar{x}, \Delta \bar{y} \in [-0.5, 0.5]$ relative to cell center. \\
        \bottomrule
    \end{tabular}
\end{table}

\paragraph{Technical Advantage.}
Strata-BEV14 provides both a NumPy reference implementation and a Numba-accelerated CPU rasterizer. Both compute identical fourteen deterministic feature channels without learned parameters. The Numba backend accumulates point statistics, normalizes occupied cells, and formats the output into CHW layout for direct backbone ingestion. In the measured single-threaded pipeline, point-cloud loading and rasterization together require $20.94\,\text{ms}$, encompassing disk I/O, coordinate parsing, and cell aggregation. Because point-to-cell binning and statistic accumulation (extents, mean intensity, log-density lookups) operate on spatially disjoint grid cells with zero inter-cell data dependencies, the rasterization algorithm is embarrassingly parallel. On embedded System-on-Chip (SoC) architectures (e.g., NVIDIA Jetson platforms), this deterministic accumulation is architecturally suited for direct CUDA parallelization with negligible thread synchronization, offering a clear path to eliminate host-device memory transfers. Alternatively, in vehicular streaming deployment, CPU rasterization operates via asynchronous double-buffering: a companion CPU core rasterizes scan $t+1$ concurrently with GPU neural network inference on scan $t$, effectively hiding preprocessing latency behind the GPU execution window.

\subsection{Physics-Consistent Ground-Truth Augmentation}
\label{sec:pc_gta}

\paragraph{Motivation.}
Cut-and-paste ground-truth (GT) augmentation balances rare LiDAR categories, but naive insertion ignores scene physics. It can produce floating or submerged objects, background collisions, missing raycast shadows, and unrealistic range-dependent point densities. We developed \textbf{Physics-Consistent Ground-Truth Augmentation} (\texttt{PC-GTA}) to address these artifacts. It improved Moderate mAP by $+6.03\,\text{pp}$ over naive sampling in the ablations.

\paragraph{Design Heuristics.}
The sampler provides five consistency mechanisms when placing a box $\mathbf{b}_{\mathrm{insert}} = (x, y, z, l, w, h, \psi)$ at range $d = \sqrt{x^2 + y^2}$. The mechanisms can be enabled independently; the settings used for the reported final run are given in Section~\ref{sec:implementation_details}:
\begin{enumerate}
\item \textit{Adaptive Ground Plane Fitting}: To prevent floating or sunken objects, local road elevation is estimated within a range-dependent search radius:
\begin{align}
    R(d) &= \operatorname{clip}(2.5 + 0.03\max(0, d - 15), 2.5, 4.0)\,\mathrm{m},\nonumber\\[0.8em]
    \hat{z}_{\mathrm{ground}} &= \operatorname{percentile}(\{z_j \mid \mathbf{p}_j \in \mathcal{N}(x, y)\}, 5),
    \label{eq:ground_fitting}
\end{align}
where $\mathcal{N}(x, y) = \{\mathbf{p}_j \mid (x_j - x)^2 + (y_j - y)^2 \le R(d)^2\}$ denotes the local neighborhood used for ground estimation.
The inserted object elevation is anchored flush with the estimated local ground elevation: $z_{\mathrm{insert}} = \hat{z}_{\mathrm{ground}} + 0.5h$.

\item \textit{Static Obstacle Clearance}: To prevent clipping into physical structures, a candidate box is rejected if existing background points within its bounding footprint exceed a strict clearance threshold: $N_{\mathrm{static}} > N_{\mathrm{thresh}}$ ($N_{\mathrm{thresh}} = 2$; valid placement requires $N_{\mathrm{static}} \le N_{\mathrm{thresh}}$).

\item \textit{Optional Line-of-Sight (LOS) Occlusion Check}: When enabled, the optical ray from sensor origin to box center $(x, y)$ is cast. If existing dense clusters obstruct the trajectory, the insertion candidate is discarded.

\item \textit{Optional Raycast LiDAR Shadow Masking}: In LiDAR scans, foreground objects occlude returns from objects behind them. When enabled, this mechanism computes intersections for each inserted box using rays $\mathbf{r}(t) = t \cdot \mathbf{p}^{xyz}$ through background points with 3D metric coordinates $\mathbf{p}^{xyz} = (x_i, y_i, z_i) \in \mathbb{R}^3$, decoupling reflectance. It masks points behind the exit face ($t_{\mathrm{leave}} \le 1.0$) whose radial distance satisfies $(1 - t_{\mathrm{leave}})\|\mathbf{p}^{xyz}\| > 0.05$\,m:
\begin{equation}
    \begin{aligned}
        \mathcal{P}_{\mathrm{augmented}}
        &= \mathcal{P}_{\mathrm{target}} \setminus
        \{\mathbf{p} \in \mathcal{P}_{\mathrm{target}} \mid \\
        &\qquad \operatorname{Mask}_{\mathrm{shadow}}(\mathbf{p}) = \mathrm{True}\}
        \cup \mathcal{P}_{\mathrm{object}}.
    \end{aligned}
    \label{eq:shadow_mask}
\end{equation}

\item \textit{Range-Adaptive Point and Intensity Attenuation}: If an object captured at source distance $r_{\mathrm{src}}$ is placed at target distance $r_{\mathrm{tgt}} > r_{\mathrm{src}}$, its point cloud is randomly subsampled according to the inverse-square divergence law:
\begin{equation}
    P_{\mathrm{keep}} = \min\left(1, \left(\frac{r_{\mathrm{src}}}{r_{\mathrm{tgt}}}\right)^2\right),
    \label{eq:inverse_square}
\end{equation}
and the optional radiometric adjustment attenuates reflectance intensity according to:
\[
    \begin{aligned}
        \eta_{\mathrm{int}}
        &= \left(\frac{r_{\mathrm{src}}}{r_{\mathrm{tgt}}}\right)^{\max(0, 2 - \gamma)},
        \qquad \gamma = 1.7, \\
        I_{\mathrm{tgt}}
        &= \operatorname{clip}(I_{\mathrm{src}}\eta_{\mathrm{int}}, 0, 1).
    \end{aligned}
\]
\end{enumerate}
During training, the resulting physically consistent point cloud $\mathcal{P}_{\mathrm{augmented}}$ is rasterized into the 14-channel input tensor $\mathbf{X}$ and fed into the convolutional backbone hierarchy.

\subsection{ConvNeXt-Style Multi-Scale BEV Backbone Hierarchy}
\label{sec:backbone_architecture}

\begin{figure*}[t]
    \centering
    \paperfigure[width=0.95\textwidth,trim=4mm 25mm 4mm 25mm,clip]{backbone_f32.pdf}{42mm}{Detailed structural topology of the MobilePIXORNeXt feedforward architecture, comprising the multi-scale backbone hierarchy, Top-Down Scale-Gated Neck (SG-FPN), and decoupled grouped prediction towers. Spatial feature maps are progressively downsampled at strides 2, 4, 8, and 16 using repeated MobilePIXORNeXt-S blocks. Scale-gated top-down fusion balances deep semantics with fine spatial localization into a shared 32-channel representation at stride 4 ($200 \times 176$). Independent grouped prediction towers decouple Car from Pedestrian and Cyclist.}
    \caption{MobilePIXORNeXt feedforward hierarchy with layer dimensions, channel widths, and receptive fields. A two-layer stem processes 14-channel input at stride 2 ($400 \times 352$). Three stages follow: C2 has 48 channels and depth 3; C3 has 96 channels and depth 4; C4 has 128 channels and depth 2. Each stage uses \texttt{MobilePIXORNeXt-S} blocks with Block-SimAM. The Scale-Gated FPN (SG-FPN) uses lateral $1 \times 1$ projections, bilinear $2\times$ upsampling, depthwise $3 \times 3$ refinement, and learnable scale gates $\mathbf{G}_r$. It produces a shared 32-channel feature map ($F_{32}$) at stride 4 ($200 \times 176$). Independent grouped towers separate Car predictions from Pedestrian and Cyclist predictions to reduce aspect-ratio conflicts within the parameter budget (753,569 total parameters).}
    \label{fig:backbone}
\end{figure*}


\paragraph{Motivation.}
BEV object footprints range from pedestrians ($l \approx 0.8$\,m, $w \approx 0.6$\,m) and cyclists ($l \approx 1.8$\,m) to passenger vehicles ($l \approx 4.5$--$5.0$\,m). Small $3 \times 3$ convolutions require deeper networks to cover this spatial context, increasing latency and parameter count. At an input resolution of $\Delta_x = \Delta_y = 0.1$\,m, a $7 \times 7$ depthwise receptive field spans $0.7 \times 0.7$\,m at stride 1. It expands to over $5.6 \times 5.6$\,m by Stage 3 (stride 8), enough to cover a passenger vehicle without dilation. We designed \textbf{MobilePIXORNeXt Backbone} with large $7 \times 7$ depthwise convolutions, inverted bottleneck expansions, and LayerScale stabilization in repeated \texttt{MobilePIXORNeXt-S} blocks to obtain this context within a small parameter budget.

\paragraph{Backbone Topology and Multi-Stage Hierarchy.}
As depicted in Figure~\ref{fig:backbone}, the network topology comprises:
\begin{enumerate}
    \item \textit{Convolutional Stem}: Input $\mathbf{X} \in \mathbb{R}^{B \times 14 \times 800 \times 704}$ passes through two consecutive $3 \times 3$ convolutions, with stride 2 followed by stride 1 and 32 output channels. Batch Normalization (BN) and Sigmoid Linear Unit (SiLU) activations produce stride-2 representation $\mathbf{F}^{(2)} \in \mathbb{R}^{B \times 32 \times 400 \times 352}$.
    \item \textit{Progressive Multi-Stage Hierarchy}: Three progressive stages are separated by strided transition downsample blocks (\texttt{DownsampleBlock}: Conv $3 \times 3$ stride 2 + BN + SiLU):
    \begin{itemize}
        \item Stage 2 (Stride 4): 3 \texttt{MobilePIXORNeXt-S} blocks, $C_2 = 48 \rightarrow \mathbf{F}^{(4)} \in \mathbb{R}^{B \times 48 \times 200 \times 176}$.
        \item Stage 3 (Stride 8): 4 \texttt{MobilePIXORNeXt-S} blocks, $C_3 = 96 \rightarrow \mathbf{F}^{(8)} \in \mathbb{R}^{B \times 96 \times 100 \times 88}$.
        \item Stage 4 (Stride 16): 2 \texttt{MobilePIXORNeXt-S} blocks, $C_4 = 128 \rightarrow \mathbf{F}^{(16)} \in \mathbb{R}^{B \times 128 \times 50 \times 44}$.
    \end{itemize}
\end{enumerate}

\paragraph{Streamlined Single-Branch Depthwise Design.}
MobilePIXORNeXt uses a single-branch large-kernel ($7 \times 7$) depthwise convolution during both training and deployment. It avoids the multi-path training topology and post-hoc weight folding of multi-branch reparameterization. Inverted bottleneck expansion ($e=2.5$) and LayerScale stabilization support optimization without auxiliary-branch memory during backpropagation. The backbone body contains $604,864$ parameters ($80.3\%$ of the detector), while the complete multi-scale detector achieves $55.94\,\text{FPS}$ ($17.88\,\text{ms}$) in synchronous pipeline execution on Tesla T4 ($115.32\,\text{FPS}$ or $8.67\,\text{ms}$ in standalone execution).

\subsection{MobilePIXORNeXt-S Residual Block with Embedded Block-SimAM}
\label{sec:next_s_block}

\begin{figure*}[t]
    \centering
    \paperfigure[width=\linewidth,keepaspectratio]{next_simam_block.pdf}{45mm}{Micro-architecture of the MobilePIXORNeXt-S residual block and embedded parameter-free Block-SimAM energy attention. (Top) Complete feedforward path from input X to output Y through the 7x7 depthwise convolution, inverted bottleneck expansion (e=2.5), SimAM modulation, LayerScale scaling, and identity residual addition. (Bottom) Zoom-in structure of the parameter-free SimAM module, detailing spatial moment estimation, inverse-energy importance calculation, sigmoid saliency mapping, and element-wise feature recalibration.}
    \caption{The \texttt{MobilePIXORNeXt-S} residual block and parameter-free Block-SimAM attention. \textbf{Top}: Residual path from input $\mathbf{Z}$ to output $\mathbf{Z}'$. It applies a $7 \times 7$ depthwise convolution, inverted bottleneck expansion with ratio $e = 2.5$, and SiLU activation, followed by pointwise projection. Parameter-free 3D spatial-energy attention $\mathcal{S}(\mathbf{R})$ precedes LayerScale scaling $\boldsymbol{\gamma}_{\mathrm{blk}}$ (initialized to $10^{-5}$) and the identity shortcut. \textbf{Bottom}: SimAM computes channel-wise spatial moments ($\mu_c, \nu_c$) in FP32 precision and inverse-energy importance $s_t = (t - \mu_c)^2 / (4(\nu_c + \lambda_{\mathrm{sim}})) + 0.5$. Sigmoid scaling produces the 3D modulation tensor $\mathbf{A} \in [0, 1]^{B \times C \times H \times W}$ without learned parameters.}
    \label{fig:next_s_block}
\end{figure*}

\paragraph{Micro-Architecture and Inverted Bottleneck.}
Figure~\ref{fig:next_s_block} illustrates the \texttt{MobilePIXORNeXt-S} residual block. Given input tensor $\mathbf{Z} \in \mathbb{R}^{B \times C \times H \times W}$, each block applies an inverted bottleneck with expansion ratio $e = 2.5$:
\begin{align}
    \mathbf{V} &= \operatorname{BN}(\operatorname{DWConv}_{7\times7}(\mathbf{Z})),\nonumber\\
    \mathbf{R} &= \operatorname{BN}\left(W_{\mathrm{red}} \operatorname{SiLU}(W_{\mathrm{exp}}\mathbf{V})\right),\nonumber\\
    \mathbf{Z}' &= \mathbf{Z} + \boldsymbol{\gamma}_{\mathrm{blk}} \odot \mathcal{S}(\mathbf{R}).
    \label{eq:next_block}
\end{align}
Here, $W_{\mathrm{exp}} \in \mathbb{R}^{2.5C \times C \times 1 \times 1}$ and $W_{\mathrm{red}} \in \mathbb{R}^{C \times 2.5C \times 1 \times 1}$ are pointwise $1 \times 1$ convolutions. The operator $\mathcal{S}(\cdot)$ represents our Block-SimAM attention module, while $\boldsymbol{\gamma}_{\mathrm{blk}} \in \mathbb{R}^C$ is a learned LayerScale parameter initialized to $10^{-5}$ to ensure numerical stability during deep backpropagation.

\paragraph{Parameter-Free Block-SimAM Spatial-Energy Attention.}
Efficient Channel Attention (ECA-Net)~\cite{wang2020eca} and Squeeze-and-Excitation compute 1D channel weights by spatial pooling, applying the same weight to every location in a channel. Non-local spatial attention instead has quadratic complexity $\mathcal{O}((HW)^2)$, which is costly on dense BEV grids ($200 \times 176$). We integrated \textbf{Block-SimAM}~\cite{yang2021simam}, a parameter-free 3D spatial-energy attention module, to provide spatial modulation without learned attention weights.

\paragraph{Design and Placement.}
As illustrated in Figure~\ref{fig:next_s_block}, SimAM is embedded inside each \texttt{MobilePIXORNeXt-\allowbreak S} block across all three stages (C2, C3, C4), for nine attention operators in total. It operates on the intermediate tensor $\mathbf{R} \in \mathbb{R}^{B \times C \times H \times W}$ after projection BN and before LayerScale shortcut addition.

For each channel slice with $M = H \times W$ spatial locations, spatial mean $\mu_c$ and sample variance $\nu_c$ are computed:
\begin{equation}
    \begin{aligned}
        \mu_c &= \frac{1}{M}\sum_{u, v} R_{c, u, v}, \\
        \nu_c &= \frac{1}{\max(M-1, 1)}\sum_{u, v} (R_{c, u, v} - \mu_c)^2.
    \end{aligned}
    \label{eq:spatial_moments}
\end{equation}
Following the energy-minimization formulation of SimAM~\cite{yang2021simam}, the minimized energy $e_t^\star$ and its inverse-energy importance score $s_t$ for neuron $t = R_{c,u,v}$ are:
\begin{equation}
    \begin{aligned}
        e_t^\star &= \frac{4(\nu_c + \lambda_{\mathrm{sim}})}{(t-\mu_c)^2 + 2(\nu_c + \lambda_{\mathrm{sim}})}, \\
        s_t &= \frac{1}{e_t^\star} = \frac{(t - \mu_c)^2}{4(\nu_c + \lambda_{\mathrm{sim}})} + \frac{1}{2}.
    \end{aligned}
    \label{eq:simam_energy}
\end{equation}
Here, $\lambda_{\mathrm{sim}} = 10^{-4}$ is a numerical regularizer. For fixed channel moments, a larger squared deviation $(t-\mu_c)^2$ decreases the minimized energy $e_t^\star$ and increases the inverse-energy importance $s_t$. The sigmoid gate therefore assigns greater weight to more distinctive spatial responses. The attended 3D modulation tensor $\mathbf{A} \in [0, 1]^{B \times C \times H \times W}$ and output $\mathcal{S}(\mathbf{R})$ are:
\begin{equation}
    A_{c, u, v} = \sigma(s_t) = \frac{1}{1 + e^{-s_t}}, \qquad \mathcal{S}(\mathbf{R}) = \mathbf{A} \odot \mathbf{R}.
    \label{eq:simam_modulation}
\end{equation}
Moments and gating scalars are computed in FP32 precision to prevent underflow, then multiplied elementwise with features.

\paragraph{Technical Advantage.}
Block-SimAM adds \textbf{zero parameters} ($0$\,M), whereas stage-level adapters add a learned scalar residual gain $\beta$. Its 3D spatial-and-channel weights modulate sparse LiDAR clusters relative to flat terrain backgrounds. In the ablation, SimAM placement inside each \texttt{MobilePIXORNeXt-S} block improved mAP by $+0.88\,\text{pp}$ over no attention and by up to $+2.10\,\text{pp}$ over stage-level adapters (evaluated in Section~\ref{sec:ablations}).


\subsection{Top-Down Scale-Gated Feature Pyramid Network (SG-FPN Neck)}
\label{sec:neck_architecture}

\paragraph{Motivation.}
Transposed convolutions used for feature upsampling can produce checkerboard artifacts in continuous coordinate regression maps. Bidirectional multiscale networks such as PANet and BiFPN add bottom-up pathways and memory latency. Unweighted addition in conventional FPN can also dilute semantic features. We designed \textbf{Top-Down Scale-Gated Feature Pyramid Network} (\texttt{SG-FPN Neck}) as a unidirectional top-down neck with depthwise scale gating to balance these features at lower cost.

\par\vspace{0.5\baselineskip}
\noindent\begin{minipage}{\columnwidth}
    \centering
    \paperfigure[width=\linewidth,trim=0 95mm 0 30mm,clip]{scale_gated_fusion.pdf}{34mm}{Scale-Gated Feature Pyramid Network (SG-FPN). Top-down features $\mathbf{U}_r$ and lateral features $\mathbf{L}_r$ are dynamically gated via learnable depthwise convolutions to prevent high-level semantic distortion of fine spatial coordinates.}
    \captionof{figure}{Scale-Gated Feature Pyramid Network (SG-FPN) fusion at stride $r \in \{8, 4\}$. Bilinear interpolation upsamples top-down features. Depthwise refinement at stride 8 and projection convolution at stride 4 produce representations $\mathbf{U}_4$ and $\mathbf{U}_3$ without deconvolution checkerboard artifacts. Lateral high-resolution features $\mathbf{L}_r$ are weighted by a zero-initialized depthwise gate $\mathbf{G}_r = 2\sigma(\operatorname{DWConv}_{3\times3}(\mathbf{L}_r + \mathbf{U}_r))$ before element-wise summation $\mathbf{P}_r = \mathbf{U}_r + \mathbf{G}_r \odot \mathbf{L}_r$ to retain boundary detail and suppress background responses.}
    \label{fig:scale_gated_fusion}
\end{minipage}
\par


\paragraph{Mathematical Formulation.}
As shown in Figure~\ref{fig:scale_gated_fusion}, lateral features from Stages C4 ($128$\,ch), C3 ($96$\,ch), and C2 ($48$\,ch) are projected via $1 \times 1$ convolutions into matching lateral dimensions $\mathbf{L}_5 \in \mathbb{R}^{B \times 48 \times 50 \times 44}$, $\mathbf{L}_4 \in \mathbb{R}^{B \times 48 \times 100 \times 88}$, and $\mathbf{L}_3 \in \mathbb{R}^{B \times 32 \times 200 \times 176}$.

Feature aggregation follows a unidirectional top-down path. At stride 8, $2\times$ bilinear interpolation and depthwise $3 \times 3$ refinement with BN and SiLU produce $\mathbf{U}_4 \in \mathbb{R}^{B \times 48 \times 100 \times 88}$:
\begin{align}
    \mathbf{U}_4 &= \operatorname{Refine}_{3\times 3}\left(\operatorname{Bilinear}_{2\times}(\mathbf{L}_5)\right),\nonumber\\
    \mathbf{P}_4 &= \mathbf{U}_4 + \mathbf{G}_4 \odot \mathbf{L}_4,
    \label{eq:scale_gating_u4}
\end{align}
where $\mathbf{G}_4 = 2\sigma(\operatorname{DWConv}_{3\times3}(\mathbf{L}_4 + \mathbf{U}_4))$. At stride 4, $2\times$ bilinear interpolation and standard $3 \times 3$ convolution ($48 \rightarrow 32$) with BN and SiLU produce $\mathbf{U}_3 \in \mathbb{R}^{B \times 32 \times 200 \times 176}$:
\begin{align}
    \mathbf{U}_3 &= \operatorname{Proj}_{3\times 3}\left(\operatorname{Bilinear}_{2\times}(\mathbf{P}_4)\right),\nonumber\\
    \mathbf{P}_3 &= \mathbf{U}_3 + \mathbf{G}_3 \odot \mathbf{L}_3,
    \label{eq:scale_gating_u3}
\end{align}
where $\mathbf{G}_3 = 2\sigma(\operatorname{DWConv}_{3\times3}(\mathbf{L}_3 + \mathbf{U}_3))$. Zero-initialized gating weights and biases make $\mathbf{G}_r$ initially equal to $1.0$, corresponding to standard residual addition. The feature $\mathbf{P}_3$ passes through a $3 \times 3$ convolution, BN, and SiLU to form the shared detector representation $\mathbf{F}_{\mathrm{out}} \in \mathbb{R}^{B \times 32 \times 200 \times 176}$. The neck contains $36,784$ parameters and passes the fused multi-scale features to the prediction towers.

\subsection{Shared Grouped Prediction Towers}
\label{sec:grouped_heads}

\begin{figure*}[t]
    \centering
    \paperfigure[width=\textwidth,height=0.55\textheight,keepaspectratio]{geometry_output_2x2.pdf}{0.55\textheight}{Geometric parameterization, orientation encoding, and quality-aware score calibration across prediction heads: (a) Continuous subcell coordinate offsets $(\Delta x, \Delta y)$ from anchor point $P$ to object center $C$; (b) Log-transformed bounding box dimension targets $(\log w, \log l)$; (c) Doubled-angle trigonometric yaw representation $(c_2, s_2) = (\cos 2\theta, \sin 2\theta)$ on the unit circle resolving $\pi$-heading ambiguity; and (d) IoU-quality-aware composite score ranking $s = (p_{\mathrm{cls}})^{1-\alpha} q^{\alpha}$ calibrated for post-processing peak selection and rotated NMS.}
    \caption{Prediction-head geometry, orientation, and quality-aware scoring. \textbf{(a)} Metric subcell offsets $(\hat{o}_x, \hat{o}_y)$ from grid anchor center $P$ to object center $C$. \textbf{(b)} Log-dimension targets $(t_w, t_l) = (\log w, \log l)$ decoded by exponentiation. \textbf{(c)} Doubled-angle yaw representation $(c_2, s_2) = (\cos 2\theta, \sin 2\theta)$ on the unit circle, resolving $180^\circ$ heading ambiguity through $\pi$-rotational symmetry of BEV footprints. \textbf{(d)} Joint quality scoring combines overlap prediction $\hat{q} \in [0, 1]$ and classification probability $p_{\mathrm{cls}}$ as $s_{\mathrm{det}} = (p_{\mathrm{cls}})^{1-\alpha} \hat{q}^\alpha$ ($\alpha = 0.5$). This score prioritizes well-aligned boxes during peak filtering and rotated NMS.}
    \label{fig:geometry_output}
\end{figure*}

\paragraph{Motivation.}
Road categories have different BEV footprints. Cars are rectangular ($l \approx 4.5$\,m, $w \approx 1.8$\,m), while Pedestrians ($l \approx 0.8$\,m, $w \approx 0.6$\,m) and Cyclists ($l \approx 1.8$\,m, $w \approx 0.7$\,m) are smaller and more deformable. A shared prediction head can introduce gradient competition across these scales; separate heads for every class add parameters. We designed \textbf{Shared Grouped Prediction Towers} (\texttt{GroupedHeader}) to balance class separation and model size.

\paragraph{Topology and Target Formulation.}
We partitioned classes into Group 1 for \texttt{car} ($K_1 = 1$) and Group 2 for \texttt{ped\_cyc} ($K_2 = 2$). For each group $g \in \{1, 2\}$, feature map $\mathbf{F}_{\mathrm{out}}$ passes through an initial $3 \times 3$ Conv-BN-SiLU block with $32 \rightarrow 32$ channels. Five task-specific branches follow:
\begin{itemize}
    \item \texttt{cls}: Classification heatmap $[B, K_g, 200, 176]$ yielding foreground category probability $p_{\mathrm{cls}}$ at grid point $P$ (Fig.~\ref{fig:geometry_output}d).
    \item \texttt{offset}: Metric subcell coordinate offsets $[B,\allowbreak 2,\allowbreak 200,\allowbreak 176]$ predicting $(\hat{o}_x, \hat{o}_y)$, corresponding to continuous displacements $(\Delta x, \Delta y)$ from anchor point $P$ to object center $C$ (Fig.~\ref{fig:geometry_output}a).
    \item \texttt{size}: Bounding box log-dimensions $[B, 2, 200, 176]$ predicting $(\hat{d}_1, \hat{d}_2) = (\log \hat{w}, \log \hat{l})$ (denoted as regression targets $(t_w, t_l)$ in Fig.~\ref{fig:geometry_output}b, decoded via $\exp$).
    \item \texttt{yaw}: Orientation doubled-angle vector $[B, 2, 200, 176]$ predicting $(\hat{a}_1, \hat{a}_2) = (\cos 2\hat{\psi}, \sin 2\hat{\psi})$ (denoted as $(c_2, s_2)$ with heading angle $\theta \equiv \hat{\psi}$ on the unit circle in Fig.~\ref{fig:geometry_output}c to resolve $\pi$-heading ambiguity).
    \item \texttt{iou}: Localization quality score $[B, 1, 200, 176]$ predicting continuous overlap $\hat{q} \in [0, 1]$ ($q(P)$ in Fig.~\ref{fig:geometry_output}d), fused with classification into composite ranking confidence $s_{\mathrm{det}} = (p_{\mathrm{cls}})^{1-\alpha} \hat{q}^\alpha = \sqrt{p_{\mathrm{cls}} \hat{q}}$ ($\alpha=0.5$).
\end{itemize}

For output cell $(u, v)$, the ground-truth geometric targets are formulated as:
\begin{align}
    \mathbf{o}^* &= \begin{pmatrix} x^* - (u s_{\mathrm{out}}\Delta_x + x_{\min}) \\ y^* - (v s_{\mathrm{out}}\Delta_y + y_{\min}) \end{pmatrix},\nonumber\\
    \mathbf{d}^* &= (\log w^*, \log l^*), \qquad \mathbf{a}^* = (\cos 2\psi^*, \sin 2\psi^*).
    \label{eq:gt_targets}
\end{align}
During inference, predicted bounding box parameters are decoded from regression outputs via:
\begin{align}
    \hat{x} &= u s_{\mathrm{out}}\Delta_x + x_{\min} + \hat{o}_x, \quad \hat{y} = v s_{\mathrm{out}}\Delta_y + y_{\min} + \hat{o}_y,\nonumber\\
    \hat{w} &= \exp(\hat{d}_1), \quad \hat{l} = \exp(\hat{d}_2), \quad \hat{\psi} = \frac{1}{2}\operatorname{atan2}(\hat{a}_2, \hat{a}_1).
    \label{eq:decoded_targets}
\end{align}
The doubled-angle formulation $\mathbf{a}^*$ and decoding rule in Eq.~(\ref{eq:decoded_targets}) strictly resolve periodic boundary singularities. Standard scalar regression of heading $\psi \in [-\pi/2, \pi/2)$ suffers from boundary instability: across the periodic boundary $\pm \pi/2$, an infinitesimal geometric rotation yields a jump discontinuity of $\pi$, producing explosive gradient penalties under Smooth-L1 or L2 regression. By projecting $\psi^*$ onto the unit circle $S^1$ via $\mathbf{a}^* = (\cos 2\psi^*, \sin 2\psi^*)$, the mapping $\psi \mapsto \mathbf{a}$ is continuous and inherently periodic with period $\pi$, capturing the geometric $\pi$-rotational symmetry (unoriented bounding box orientation modulo $\pi$) on the horizontal BEV plane. At inference, candidate heading modulo $\pi$ is recovered continuously via $\frac{1}{2}\operatorname{atan2}(\hat{a}_2, \hat{a}_1)$, eliminating periodic loss jump discontinuities during gradient backpropagation.

Classification supervision utilizes 2D Gaussian heatmaps centered at continuous ground-truth object projections with a minimum radius of 4 cells. Regression targets are assigned within half the Gaussian peak radius, evaluated on decoupled validity masks $\mathbf{M}_{\mathrm{reg}, g}$ for each group $g \in \{1, 2\}$. Specifically, category target ownership is strictly partitioned: anchor cells within the regression radius of a car object activate $\mathbf{M}_{\mathrm{reg}, 1}(u, v) = 1$ to supervise Group 1, while cells within the regression radius of a pedestrian or cyclist activate $\mathbf{M}_{\mathrm{reg}, 2}(u, v) = 1$ to supervise Group 2. Each group tower computes regression losses independently on its respective positive mask, decoupling gradient backpropagation between categories with radically divergent aspect ratios ($4.5$\,m passenger cars vs. $0.8$\,m pedestrians). In the rare event where a vehicle and a vulnerable road user are immediately adjacent such that both masks activate at cell $(u, v)$, both towers regress their respective targets concurrently without inter-head gradient interference. Within Group 2 (\texttt{ped\_cyc}), where pedestrians and cyclists share a single regression head, conflicting assignments on overlapping cells are resolved by assigning ownership to the ground-truth instance with the nearest continuous Euclidean center distance.

\subsection{Multi-Task Geometric Alignment and Quality Supervision (OGA and IQA)}
\label{sec:losses_and_oga}

\paragraph{Motivation.}
We developed \textbf{Oriented Geometric Alignment (OGA)} with \textbf{IoU-Aware Quality Alignment (IQA)} to connect box-parameter regression with oriented overlap. Standard anchor-free regression applies separate Smooth-L1 losses to center $(x, y)$, footprint $(w, l)$, and heading $\psi$. These errors do not vary monotonically with rotated Intersection-over-Union (IoU). Classification confidence can also disagree with localization accuracy, degrading candidate ranking during rotated Non-Maximum Suppression (NMS).

\paragraph{Oriented Geometric Alignment (OGA).}
For each cell $(u, v)$ with $\mathbf{M}_{\mathrm{reg}} = 1$, we decoded predicted and ground-truth boxes into four 2D corners $\{\widehat{\mathbf{c}}_m\}_{m=1}^4$ and $\{\mathbf{c}_m^*\}_{m=1}^4$. Given center $(x, y)$, dimensions $(l, w)$, and longitudinal/lateral unit vectors $\mathbf{n}_{\mathrm{len}} = (\cos\psi, \sin\psi), \mathbf{n}_{\mathrm{wid}} = (-\sin\psi, \cos\psi)$, the four bounding box corners are indexed in quadrant order $[++, +-, -+, --]$:
\begin{equation}
    \begin{aligned}
        \mathbf{c}_1 &= (x, y) + \tfrac{1}{2}l\mathbf{n}_{\mathrm{len}} + \tfrac{1}{2}w\mathbf{n}_{\mathrm{wid}}, \\
        \mathbf{c}_2 &= (x, y) + \tfrac{1}{2}l\mathbf{n}_{\mathrm{len}} - \tfrac{1}{2}w\mathbf{n}_{\mathrm{wid}}, \\
        \mathbf{c}_3 &= (x, y) - \tfrac{1}{2}l\mathbf{n}_{\mathrm{len}} + \tfrac{1}{2}w\mathbf{n}_{\mathrm{wid}}, \\
        \mathbf{c}_4 &= (x, y) - \tfrac{1}{2}l\mathbf{n}_{\mathrm{len}} - \tfrac{1}{2}w\mathbf{n}_{\mathrm{wid}}.
    \end{aligned}
    \label{eq:corner_enumeration}
\end{equation}
OGA combines two geometric objectives:
\par\noindent 1. \textit{Multi-Axis Projection GIoU ($\ell_{\mathrm{proj}}$)}: Projected bounding box overlap is evaluated following Marginalized Generalized IoU~\cite{le2026mgiou} across four unit normal projection axes defining the prediction and ground-truth rectangles:
\begin{equation}
    \mathbf{n}_k \in \left\{ \mathbf{n}_{\mathrm{len}}^{\mathrm{pred}}, \mathbf{n}_{\mathrm{wid}}^{\mathrm{pred}}, \mathbf{n}_{\mathrm{len}}^{\mathrm{gt}}, \mathbf{n}_{\mathrm{wid}}^{\mathrm{gt}} \right\}.
    \label{eq:projection_axes}
\end{equation}
For each axis $\mathbf{n}_k$, 1D intervals of intersection length $I_k$, union length $U_k$, and smallest enclosing hull $E_k$ are determined, yielding 1D GIoU~\cite{rezatofighi2019giou}:
\begin{equation}
    \begin{aligned}
        g_k &= \frac{I_k}{U_k} - \frac{E_k - U_k}{E_k}, \\
        \bar{g} &= \frac{1}{4}\sum_{k=1}^4 g_k, \qquad
        \ell_{\mathrm{proj}} = \frac{1 - \bar{g}}{2}.
    \end{aligned}
    \label{eq:mgiou_loss}
\end{equation}
\par\noindent{\raggedright\textit{2. Scale-Normalized $\pi$-Symmetric Corner Distance ($\ell_{\mathrm{corner}}$)}:\par}\noindent
Under a $\pi$-rotation ($180^\circ$), corner $m$ maps deterministically to corner $5-m$. To ensure accurate vertex localization under doubled-angle symmetry, we evaluate the minimum L1 distance across direct and $\pi$-permuted corner correspondences:
\begin{equation}
    \ell_{\mathrm{corner}} = \frac{\min\left\{
        \begin{array}{l}
            \displaystyle\sum_{m=1}^4 \|\widehat{\mathbf{c}}_m - \mathbf{c}_m^*\|_1, \\
            \displaystyle\sum_{m=1}^4 \|\widehat{\mathbf{c}}_m - \mathbf{c}_{5-m}^*\|_1
        \end{array}
    \right.}{4\max\left(\sqrt{(w^*)^2 + (l^*)^2}, 10^{-6}\right)}.
    \label{eq:corner_loss}
\end{equation}
Normalizing by the target bounding box diagonal ensures scale invariance across categories. The composite geometric loss is $\mathcal{L}_{\mathrm{geo}} = \ell_{\mathrm{proj}} + \beta_{\mathrm{corner}} \ell_{\mathrm{corner}}$ with $\beta_{\mathrm{corner}} = 1.0$.

For positive anchor cells with $\mathbf{M}_{\mathrm{reg}} = 1$ ($N_{\mathrm{pos}} = \sum \mathbf{M}_{\mathrm{reg}}$), the subcell offset, log-size, and doubled-yaw heads are supervised using masked Smooth-$L_1$ loss:
\begin{equation}
    \begin{aligned}
        \mathcal{L}_{\mathrm{task}} &= \frac{1}{N_{\mathrm{pos}} + \epsilon} \sum_{i \in \mathrm{pos}} \operatorname{SmoothL1}(\hat{\mathbf{y}}_i, \mathbf{y}_i^*), \\
        &\quad \text{for } \mathrm{task} \in \{\mathrm{off}, \mathrm{size}, \mathrm{yaw}\},
    \end{aligned}
    \label{eq:smooth_l1_base}
\end{equation}
with threshold $\beta_{\mathrm{reg}} = 1/9$ and $\epsilon = 10^{-6}$.

\paragraph{IoU-Aware Quality Alignment (IQA).}
The quality head predicts a continuous localization quality score $\hat{q} = \sigma(\hat{Z}_{\mathrm{qual}}) \in [0, 1]$. It is supervised via Binary Cross-Entropy (BCE) over positive anchor cells using the detached surrogate multi-axis GIoU metric as the ground-truth target:
\begin{equation}
    \begin{aligned}
        q^* &= \operatorname{stopgrad}(\operatorname{clip}(\bar{g}, 0, 1)), \\
        \mathcal{L}_{\mathrm{qual}} &= \frac{1}{N_{\mathrm{pos}} + \epsilon} \sum_{i \in \mathrm{pos}} \operatorname{BCE}(\hat{q}_i, q_i^*).
    \end{aligned}
    \label{eq:iqa_loss}
\end{equation}
Because the closed-form 2D rotated polygon intersection is non-differentiable across arbitrary overlap configurations, $\bar{g}$ provides an analytically stable 1D projection surrogate. Detaching $q^*$ prevents gradient competition between regression and confidence calibration while supervising ranking accuracy.

\paragraph{Head-Free Alternative: Quality-Coupled OGA (Q-OGA).}
The explicit IQA branch gave the highest ranking accuracy, adding $18{,}626$ parameters ($18.63\,\text{K}$) and $1.25\,\text{ms}$ forward latency. We developed \textbf{Q-OGA} as a head-free alternative for resource-constrained platforms. Q-OGA modulates positive classification targets ($y=1.0$) directly with the detached surrogate projection quality $q^*$, without an auxiliary head:
\begin{equation}
    y^* = (q^*)^\beta \quad (\beta = 1.0).
    \label{eq:cqfl_target}
\end{equation}
To supervise continuous quality-coupled classification without dedicated parameters, we adapt the Quality Focal Loss (QFL) formulation~\cite{li2020gfl} to Bird's-Eye View object detection as Quality-Coupled Focal Loss (CQFL):
\begin{align}
    \mathcal{L}_{\mathrm{pos}} &= |y^* - \hat{p}|^\gamma \left[ y^* (-\ln \hat{p}) + (1 - y^*) (-\ln(1 - \hat{p})) \right], \nonumber\\
    \mathcal{L}_{\mathrm{neg}} &= \hat{p}^\gamma (-\ln(1 - \hat{p})),
    \label{eq:cqfl_loss}
\end{align}
where $\gamma = 2.0$. Crucially, rather than introducing a novel mathematical weighting function, our methodological contribution lies in the head-free geometric coupling: replacing discrete one-hot classification targets with the continuous projection-based geometric quality $q^*$, thereby eliminating separate quality regression heads while retaining joint confidence-localization ranking. In addition, Q-OGA replaces hard $\pi$-minimum corner distance with a soft-min corner formulation:
\begin{equation}
    \mathcal{D}_{\mathrm{smooth}} = -\tau_{\mathrm{soft}} \ln\left( \mathrm{e}^{-D_{\mathrm{direct}}/\tau_{\mathrm{soft}}} + \mathrm{e}^{-D_\pi/\tau_{\mathrm{soft}}} \right),
    \label{eq:softmin_corner}
\end{equation}
with $\tau_{\mathrm{soft}} = 0.05$, and incorporates Range and Density-Adaptive (RDA) weighting $w(r) = 1.0 + \gamma_{\mathrm{rda}}(r / r_{\max})^{\alpha_{\mathrm{rda}}}$ ($\gamma_{\mathrm{rda}} = 1.5, \alpha_{\mathrm{rda}} = 2.0$). In the evaluation, Q-OGA saved $18{,}626$ parameters and achieved $69.20\%$ Moderate mAP, providing a competitive compact option.

\paragraph{Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW).}
To balance six multi-task losses $\mathcal{T} = \{\mathrm{cls}, \mathrm{off}, \mathrm{size}, \mathrm{yaw}, \mathrm{geo}, \mathrm{qual}\}$ ($J=6$) across decoupled groups without manual hyperparameter tuning, we developed \textbf{Temperature-Softmax Bounded Weighting (T-SBUW)}. Unlike Bayesian homoscedastic uncertainty weighting~\cite{kendall2018uncertainty}, which relies on unconstrained variance parameters with a $\log \sigma$ penalty to prevent trivial zero-weight minimization, T-SBUW enforces optimization stability through two coupled mechanisms on a conserved gradient simplex:
\par\noindent 1. \textit{Bounded Softmax Weights on a Conserved Simplex}: Learnable logit parameters $a_k$ (initialized to $0.0$, yielding initial weights $\omega_k = 1.0$) are soft-clamped via hyperbolic tangent with bound $b_{\mathrm{uw}} = 3.0$ to prevent numerical divergence, and mapped to positive normalized weights on a simplex of size $J$ via temperature-controlled softmax ($\tau_{\mathrm{uw}} = 2.0$):
\begin{equation}
    \begin{gathered}
        \bar{a}_k = b_{\mathrm{uw}} \tanh(a_k / b_{\mathrm{uw}}), \\
        Z_{\mathrm{uw}} = \sum_{j \in \mathcal{T}} \exp(\bar{a}_j / \tau_{\mathrm{uw}}), \\
        \omega_k = J \cdot \frac{\exp(\bar{a}_k / \tau_{\mathrm{uw}})}{Z_{\mathrm{uw}}}, \\
        \sum_{k \in \mathcal{T}} \omega_k = J.
    \end{gathered}
    \label{eq:tsbuw_weights}
\end{equation}
The constraint $\sum_{k \in \mathcal{T}} \omega_k = J$ strictly conserves the total gradient budget across all tasks without requiring an explicit $\log \sigma$ barrier.
\par\noindent 2. \textit{Running Scale Calibration}: On a simplex, uncalibrated loss weighting would cause gradient descent to disproportionately favor tasks with smaller numerical magnitudes. To prevent loss scale collapse, an Exponential Moving Average (EMA) with momentum $\beta_{\mathrm{EMA}} = 0.99$ (initialized with $\mu_k^{(0)} = 1.0$) tracks the unweighted running loss magnitude $\mu_k \leftarrow 0.99\mu_k + 0.01\mathcal{L}_k$ during training. Each task loss is calibrated to a dimensionless scale $\mathcal{L}_k / \max(\mu_k, 10^{-4}) \approx 1.0$, equalizing competition across disparate loss scales. The balanced objective for group $g$ is formulated as:
\begin{equation}
    \mathcal{L}_g = \left(\frac{1}{J}\sum_{k \in \mathcal{T}} \mu_k\right) \sum_{k \in \mathcal{T}} \omega_k \frac{\mathcal{L}_k}{\max(\mu_k, 10^{-4})}.
    \label{eq:balanced_loss}
\end{equation}
The overall scale factor $\frac{1}{J}\sum_{k \in \mathcal{T}} \mu_k$ preserves the detector's natural optimization magnitude. The total network objective is $\mathcal{L}_{\mathrm{total}} = 0.5\mathcal{L}_1 + 0.5\mathcal{L}_2$. The twelve logit parameters are utilized strictly during training and discarded at inference.


% \newpage
\subsection{Coordinate Decoding and Inference Pipeline}
\label{sec:decoding_and_pipeline}

\noindent
\rule{\linewidth}{0.6pt}\par\vspace{1pt}
\begingroup
\setlength{\tabcapgap}{0pt}
\captionof{algorithm}{End-to-end MobilePIXORNeXt inference and box decoding pipeline.}
\label{alg:inference_pipeline}
\endgroup
\par\noindent\rule{\linewidth}{1pt}\par
{\small
\noindent\textbf{Require:} Raw LiDAR point cloud $\mathcal{P} = \{(x_i, y_i, z_i, \iota_i)\}_{i=1}^N$, ROI bounds $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}] \times [z_{\min}, z_{\max}]$\par
\noindent\textbf{Require:} Score threshold $\tau_{\mathrm{score}} = 0.05$, Rotated NMS threshold $\tau_{\mathrm{nms}} = 0.10$, quality exponent $\alpha = 0.5$\par
\noindent\textbf{Ensure:} Final decoded oriented BEV bounding footprints $\mathcal{B} = \{(c_k, s_k, \hat{x}_k, \hat{y}_k, \hat{l}_k, \hat{w}_k, \hat{\psi}_k)\}_{k=1}^M$\par
\begin{enumerate}
    \setlength{\topsep}{2pt}
    \setlength{\itemsep}{0pt}
    \setlength{\parsep}{0pt}
    \renewcommand{\labelenumi}{\textbf{\arabic{enumi}:}}
    \item Filter raw points $\mathcal{P}$ within inward margin $\epsilon = 0.001$\,m of spatial ROI.
    \item Deterministically rasterize points into 14-channel Strata-BEV14 tensor $\mathbf{X} \in \mathbb{R}^{14 \times 800 \times 704}$ via Eq.~(\ref{eq:density_encoding})--(\ref{eq:offset_encoding}).
    \item Forward $\mathbf{X}$ through ConvNeXt stem to obtain stride-2 feature $\mathbf{F}^{(2)} \in \mathbb{R}^{32 \times 400 \times 352}$.
    \item Extract multi-scale features through Stages 2, 3, 4 via MobilePIXORNeXt-S blocks to obtain $\mathbf{F}^{(4)}, \mathbf{F}^{(8)}, \mathbf{F}^{(16)}$.
    \item Fuse multi-scale features via Top-Down SG-FPN Neck into shared map $\mathbf{F}_{\mathrm{out}} \in \mathbb{R}^{32 \times 200 \times 176}$.
    \item Evaluate Grouped Prediction Towers for Car ($g=1$) and Pedestrian/Cyclist ($g=2$), yielding classification maps $\mathbf{P}_{\mathrm{cls}}$, subcell offsets $\widehat{\mathbf{o}}$, log-sizes $\widehat{\mathbf{d}}$, doubled yaws $\widehat{\mathbf{a}}$, and quality $\widehat{\mathbf{q}}$.
    \item \textbf{for each} class $c \in \{\text{Car}, \text{Pedestrian}, \text{Cyclist}\}$ \textbf{do}
    \item \quad Compute IoU-aware composite ranking map across grid: $\mathbf{S}_{\mathrm{det}, c} = (\mathbf{P}_{\mathrm{cls}, c})^{1 - \alpha} \cdot (\widehat{\mathbf{q}})^\alpha$.
    \item \quad Identify local $3 \times 3$ candidate peaks on composite ranking map $\mathbf{S}_{\mathrm{det}, c}$.
    \item \quad Filter candidate cells satisfying $\mathbf{S}_{\mathrm{det}, c} > \tau_{\mathrm{score}}$ and $\mathbf{P}_{\mathrm{cls}, c} > \tau_{\mathrm{score}}$.
    \item \quad Decode metric coordinates, dimensions, and heading angles via Eq.~(\ref{eq:box_decode}).
    \item \quad Apply Classwise Rotated Non-Maximum Suppression with IoU threshold $\tau_{\mathrm{nms}}$ ranked by $s_{\mathrm{det}}$.
    \item \textbf{end for}
    \item \textbf{return} Final detection set $\mathcal{B}$ sorted in descending order of composite score $s_{\mathrm{det}}$.
\end{enumerate}
\par\vspace{2pt}\noindent\rule{\linewidth}{0.6pt}\par
}

\par\vspace{0.5\baselineskip}


Algorithm~\ref{alg:inference_pipeline} specifies inference. For candidate cells satisfying the joint score threshold and local peak filtering, oriented BEV bounding footprints on $SE(2)$ are decoded in metric coordinates as:
\begin{align}
    \hat{x} &= x_{\min} + u s_{\mathrm{out}}\Delta_x + \hat{o}_x, \nonumber\\
    \hat{y} &= y_{\min} + v s_{\mathrm{out}}\Delta_y + \hat{o}_y, \nonumber\\
    \hat{w} &= \exp(\hat{d}_1), \qquad \hat{l} = \exp(\hat{d}_2), \nonumber\\
    \hat{\psi} &= \frac{1}{2}\operatorname{atan2}(\hat{a}_2, \hat{a}_1).
    \label{eq:box_decode}
\end{align}
Predicted orientation angle $\hat{\psi} \in [-\pi/2, \pi/2]$ is fully recovered without trigonometric ambiguity.

To align classification confidence with geometric overlap precision, detection candidates are ranked via a joint IoU-aware composite confidence score:
\begin{equation}
    s_{\mathrm{det}} = (p_{\mathrm{cls}})^{1 - \alpha_{\mathrm{score}}} \cdot (\hat{q})^{\alpha_{\mathrm{score}}}, \qquad \alpha_{\mathrm{score}} = 0.5.
    \label{eq:joint_score}
\end{equation}
Candidates with $s_{\mathrm{det}} > 0.05$ undergo class-wise rotated NMS at IoU threshold $\tau_{\mathrm{nms}} = 0.10$. The output is capped at 500 boxes per scan.

\paragraph{Parameter Distribution Breakdown.}
Our complete detector architecture contains \textbf{753,569 deployable parameters} ($< 0.76$\,M, requiring $753{,}581$ trainable parameters during training including loss-balancing logits), with an FP32 checkpoint size of $9.34$\,MB:
\begin{itemize}
    \item \textbf{Backbone Body}: $604,864$ parameters ($80.3\%$).
    \item \textbf{Scale-Gated Neck (SG-FPN)}: $36,784$ parameters ($4.9\%$).
    \item \textbf{Grouped Prediction Heads}: $111,921$ parameters ($14.8\%$), divided into $55,944$ parameters for Group \texttt{car} and $55,977$ parameters for Group \texttt{ped\_cyc}.
    \item \textbf{Training-Only T-SBUW Logits}: 12 parameters ($< 0.01\%$), optimized exclusively during training and eliminated during deployment.
\end{itemize}
The complete pipeline achieved $16.41$\,FPS ($60.96$\,ms) on an NVIDIA Tesla T4 GPU, while in-pipeline model forward execution took $17.88$\,ms ($55.94$\,FPS) and standalone model forward reached $8.67$\,ms ($115.32$\,FPS).


% SOURCE: sections/related_work.tex
\subsection{Bird's-Eye View LiDAR Representation and Alternative Paradigms}
\label{sec:rel_bev_encoding}

Dense horizontal projection enables 2D convolutional networks to process unstructured 3D LiDAR point clouds. PIXOR~\cite{yang2018pixor} introduced proposal-free Bird's-Eye View (BEV) detection using fixed vertical occupancy slices and calibrated reflectance maps. Nguyen et al.~\cite{nguyen2024indoor} developed MobilePIXOR for indoor robotic perception, replacing large convolutional layers with inverted residual blocks and Gaussian heatmap supervision. PointPillars~\cite{lang2019pointpillars} instead organized points into vertical columns (pillars) and learned spatial features through local PointNet encoders. PillarNeXt~\cite{li2023pillarnext} and PillarNeSt~\cite{mao2023pillarnest} extended pillar-based receptive field design and backbone scaling for outdoor perception. Beyond Cartesian BEV representations, Range-View approaches such as LaserNet~\cite{meyer2019lasernet} represent LiDAR returns as spherical range images, enabling rapid processing with standard 2D convolutions. However, spherical projections suffer from severe metric scale distortion with radial distance and foreground-background occlusion tears. Conversely, modern query-based BEV Transformers such as TransFusion~\cite{bai2022transfusion} achieve state-of-the-art context modeling via sparse object queries cross-attending into dense LiDAR and camera feature maps. While query-to-feature cross-attention scales linearly with spatial token count ($O(N_{\mathrm{query}} \times HW)$), multi-head attention projections, high memory bandwidth, multi-sensor calibration synchronization, and large parameter footprints (frequently $20\text{--}40\,\text{M}$ parameters) far outstrip the strict sub-million-parameter limits of low-power edge microcontrollers.

BEV representations commonly use either learned PointNet-based pillar encoders or fixed-grid rasterization. Pillar encoders~\cite{lang2019pointpillars,li2023pillarnext} learn geometric features but require point gathering, dynamic memory allocation, and sparse scattering, which add latency and energy costs on edge accelerators. Fixed-grid methods~\cite{yang2018pixor,nguyen2024indoor} avoid learned point encoding. However, binary flags or raw cell counts on $0.1\,\text{m}$ grids omit continuous subcell offsets and higher-order vertical distributions, causing boundary staircasing and localization jitter. PillarHist~\cite{zhou2025pillarhist} investigated quantization-aware height histograms, and TriBand-BEV~\cite{khoshkdahan2026triband} explored multi-band vertical slicing for pedestrian detection. These methods do not combine continuous subcell centroid displacements with continuous vertical statistical moments in a compact tensor. MobilePIXORNeXt uses Strata-BEV14 (Section~\ref{sec:strata_bev14}) to combine four stratified height-band density bins ($F_0$--$F_3$), four continuous elevation moments ($z_{\max}, z_{\mathrm{mean}}, z_{\mathrm{span}}, z_{\mathrm{std}}$, $F_4$--$F_7$), two surface reflectance statistics ($I_{\max}, I_{\mathrm{mean}}$, $F_8$--$F_9$), global logarithmic cell density ($F_{10}$), binary occupancy ($F_{11}$), and two continuous subcell centroid displacement offsets $(\Delta\bar{x}, \Delta\bar{y}$, $F_{12}$--$F_{13}$). This 14-channel layout preserves sub-voxel geometry without learned point encoders, while reducing Host-to-Device transfer payloads by $60.0\%$ compared to 35-channel binary grids.

\subsection{Ground-Truth Data Augmentation and Physical Regularization}
\label{sec:rel_gt_augmentation}

To eliminate unphysical synthesis artifacts in LiDAR point-cloud augmentation, MobilePIXORNeXt introduces Physics-Consistent Ground-Truth Augmentation (\texttt{PC-GTA}, Section~\ref{sec:pc_gta}), integrating adaptive local ground plane fitting, static obstacle clearance filtering, line-of-sight ray checks, and shadow masking. While the ground-truth database sampling introduced in SECOND~\cite{yan2018second} enriches underrepresented pedestrian and cyclist classes, naive cut-and-paste insertion is unconstrained by environmental physics: sampled objects frequently float above ground, sink below road surfaces, or penetrate static background obstacles. Additionally, naive insertion violates raycasting optical geometry by failing to mask occluded background points behind newly inserted foreground targets and omitting distance-dependent beam divergence ($1/r^2$). By enforcing geometric and optical consistency, \texttt{PC-GTA} prevents these placement artifacts and contributes a $+6.03\,\text{pp}$ Moderate mAP improvement in controlled ablations.

\subsection{Lightweight Convolutional Backbones and Efficient Spatial Attention}
\label{sec:rel_backbone_attention}

Spatial feature extraction from sparse BEV planes must balance receptive field coverage against parameter and latency limits. For 3D LiDAR perception, SECOND~\cite{yan2018second} used sparse convolutions to process non-empty voxels. PointRCNN~\cite{shi2019pointrcnn}, Part-$A^2$~\cite{shi2020part}, and PV-RCNN~\cite{shi2020pvrcnn} developed two-stage point processing, part-aware aggregation, and voxel-set abstractions. These methods achieve high accuracy on KITTI~\cite{geiger2012kitti}, but their large parameter counts ($4.0\text{--}63.8\,\text{M}$) and substantial pipeline latencies ($93.7\text{--}194.5\,\text{ms}$) restrict real-time deployment on power-constrained edge accelerators.

Lightweight 2D convolutions offer an alternative to volumetric processing. ConvNeXt~\cite{liu2022convnext} modernized standard convolutional hierarchies using $7 \times 7$ depthwise kernels, inverted bottleneck expansions ($e=4$), GELU activations, and LayerScale, demonstrating accuracy competitive with Vision Transformers. For compact LiDAR BEV feature extraction, we tailor these architectural principles by adopting a slimmer expansion ratio ($e=2.5$), SiLU activations, and Batch Normalization, retaining large receptive fields while avoiding over-parameterization on edge hardware. Lightweight attention mechanisms further modulate geometric features. Efficient Channel Attention (ECA-Net)~\cite{wang2020eca} modeled 1D cross-channel interactions with low complexity but did not modulate individual spatial locations. Yang et al.~\cite{yang2021simam} proposed SimAM, deriving 3D spatial-channel attention analytically from an energy function inspired by visual neuroscience, without trainable parameters.

Prior mobile LiDAR detectors commonly use inverted residual bottlenecks with small $3 \times 3$ kernels~\cite{nguyen2024indoor}, which limit context across vehicle and pedestrian scales. External attention modules or distillation at coarse downsampling boundaries~\cite{tran2026rubber} add overhead while leaving receptive field limits within blocks unresolved under sparse point distributions. MobilePIXORNeXt combines the MobilePIXORNeXt-S backbone with parameter-free Block-SimAM (Sections~\ref{sec:backbone_architecture}--\ref{sec:next_s_block}) and a Top-Down Scale-Gated FPN (Section~\ref{sec:neck_architecture}). Large $7 \times 7$ depthwise kernels expand receptive fields, and attention modulates feature updates within each residual block without additional attention parameters.

\subsection{Geometric Supervision, Loss Formulations, and Quality Calibration}
\label{sec:rel_supervision_losses}

Oriented bounding box regression has developed from independent coordinate losses toward geometric objectives. Early detectors used decoupled Smooth-L1 losses. Independent regression of coordinates, dimensions, and angles ignores their geometric dependence and can destabilize gradients. Rezatofighi et al.~\cite{rezatofighi2019giou} introduced Generalized Intersection over Union (GIoU) to provide scale-invariant feedback when boxes do not overlap. Le et al.~\cite{le2026mgiou} later formulated Marginalized Generalized Intersection over Union (MGIoU), a projection-based objective for convex parametric shapes and arbitrarily oriented boxes without rotational singularities.

Localization quality calibration also affects detection ranking. Li et al.~\cite{li2020gfl} introduced Generalized Focal Loss (GFL). They showed that classification scores can disagree with localization quality, allowing high-confidence but poorly localized boxes to suppress accurate candidates during Non-Maximum Suppression (NMS). Kendall et al.~\cite{kendall2018uncertainty} pioneered multi-task uncertainty weighting by modeling homoscedastic observation noise, using a $\log \sigma$ penalty to prevent variance divergence.

Three supervision limitations remain in oriented detection. Independent angle and corner losses are sensitive to the $\pi$-periodicity boundary: small angular deviations near $\pm \pi/2$ or $\pm \pi$ can cause loss spikes and unstable yaw gradients. Auxiliary quality heads add parameters and can increase gradient competition across classes. In addition, unconstrained multi-task weighting can become unstable when classification, overlap, and orientation losses operate on disparate numerical scales. Unlike probabilistic observation noise formulations that require $\log \sigma$ regularization, MobilePIXORNeXt balances multi-task objectives via Temperature-Softmax Bounded Weighting (T-SBUW, Section~\ref{sec:losses_and_oga}). T-SBUW constrains task weights to a normalized, conserved simplex ($\sum \omega_k = J$) and employs running Exponential Moving Average (EMA) loss scale calibration to prevent weights from trivially collapsing toward tasks with smaller raw loss magnitudes. Together with Oriented Geometric Alignment (OGA) and IoU-Aware Quality Alignment (IQA), this framework provides numerically stable and balanced multi-task supervision.


% SOURCE: sections/results.tex
\subsection{Quantitative Comparison on KITTI Benchmark}
\label{sec:benchmark_comparison}

\input{tables/seed_robustness}

\paragraph{Detection Quality and Parameter Efficiency.}
Table~\ref{tab:comparison} presents a comprehensive quantitative comparison on the standard KITTI validation benchmark under the official 40-point interpolated Average Precision ($R_{40}$) protocol at Moderate difficulty. All models are benchmarked under a unified evaluation implementation and evaluated on the same NVIDIA Tesla T4 GPU platform.

\begin{figure*}[t]
\centering
\paperfigure[width=\textwidth]{pareto_tradeoff.pdf}{48mm}{Reported KITTI accuracy, parameter count, and latency context.}
\caption{Pareto trade-off efficiency of MobilePIXORNeXt against representative LiDAR detectors on the KITTI validation benchmark under the Moderate BEV protocol ($R_{40}$). \textbf{(a)} Detection accuracy ($\text{mAP}_{\mathrm{M}}$, \%) versus parameter footprint (M) on a logarithmic scale, highlighting that MobilePIXORNeXt achieves $71.50\%$ Moderate mAP with only $0.754\,\text{M}$ parameters, matching SECOND ($72.09\%$, $5.33\,\text{M}$) and outperforming PointPillars ($70.13\%$, $4.84\,\text{M}$) with $>84\%$ fewer parameters. \textbf{(b)} Detection accuracy ($\text{mAP}_{\mathrm{M}}$, \%) versus end-to-end pipeline latency (ms) measured on an NVIDIA Tesla T4 GPU at batch size 1, demonstrating that MobilePIXORNeXt establishes the fastest complete perception pipeline ($60.96\,\text{ms}$, $16.41\,\text{FPS}$) among all evaluated real-time detectors.}
\label{fig:pareto_tradeoff}
\end{figure*}

\textit{1. Overall Accuracy-to-Parameter Pareto Efficiency.}
As visualized in the Pareto frontier plots (Figure~\ref{fig:pareto_tradeoff}), MobilePIXORNeXt establishes a favorable accuracy-to-parameter and accuracy-to-latency trade-off for real-time edge perception. Operating with only $0.754\,\text{M}$ trainable parameters ($9.34\,\text{MB}$ checkpoint size), MobilePIXORNeXt achieves $71.50\%$ Moderate mAP and $73.45\%$ $\text{mAP}_{9}$. In comparison, PointPillars~\cite{lang2019pointpillars} ($4.835\,\text{M}$) and SECOND~\cite{yan2018second} ($5.326\,\text{M}$) require significantly higher parameter counts, meaning MobilePIXORNeXt achieves comparable or superior overall accuracy while reducing the parameter footprint by $84.41\%$ and $85.85\%$, respectively. The precision--recall dynamics across all classes and difficulties are illustrated in Figure~\ref{fig:pr_curves}.

\begin{figure*}[!t]
\centering
\paperfigure[width=\textwidth]{pr_all_difficulties_sota.pdf}{65mm}{Precision-Recall benchmark curves across classes and difficulties on the KITTI validation set.}
\caption{Precision--recall curves evaluated under the unified benchmark protocol on the KITTI validation split. Columns show Car ($\tau_{\mathrm{IoU}}=0.70$), Pedestrian ($0.50$), and Cyclist ($0.50$); rows show Easy, Moderate, and Hard difficulties. MobilePIXORNeXt is red, MobilePIXOR gray, PointPillars blue, SECOND orange, PIXOR pink, PointRCNN purple, Part-A$^2$ green, and PV-RCNN teal. PIXOR supports Car only. AP $R_{40}$ is the discrete average of interpolated precision at the 40 recall positions specified in Section~\ref{sec:dataset_and_metrics}. Pedestrian Moderate AP is $60.78\%$, $56.01\%$, and $56.56\%$ for MobilePIXORNeXt, PointPillars, and SECOND, respectively; the corresponding Cyclist values are $66.81\%$, $66.34\%$, and $71.16\%$. All models are evaluated under identical test splits and evaluation criteria.}
\label{fig:pr_curves}
\end{figure*}


\textit{2. Performance on Vulnerable Pedestrians.}
Detecting vulnerable road users in sparse LiDAR point clouds represents a persistent challenge due to small physical dimensions, low point return counts, and non-rigid geometry. As reported in Table~\ref{tab:comparison}, MobilePIXORNeXt establishes the highest score on Pedestrians among all single-stage real-time architectures, achieving \textbf{60.78\% Moderate AP}. This outperforms standard single-stage baselines including SECOND ($56.56\%$, $+4.22\,\text{pp}$) and PointPillars ($56.01\%$, $+4.77\,\text{pp}$), while more than doubling the baseline MobilePIXOR ($28.16\%$, a $+32.62\,\text{pp}$ increase). The continuous subcell centroid displacement offsets $(\Delta\bar{x}, \Delta\bar{y})$ in Strata-BEV14 mitigate spatial quantization loss, while the decoupled GroupedHeader prevents pedestrian gradients from being dominated by rigid vehicle targets.

\textit{3. Balanced Multi-Class Detection.}
In addition to vulnerable road users, MobilePIXORNeXt maintains competitive accuracy on rigid vehicles with $86.90\%$ Car Moderate AP, closely approaching PointPillars ($88.02\%$) and SECOND ($88.56\%$) while utilizing only a fraction of their parameter footprints. On Cyclists, MobilePIXORNeXt achieves \underline{66.81\%} Moderate AP, surpassing PointPillars ($66.34\%$) and trailing only SECOND ($71.16\%$). Across all nine evaluation cells, MobilePIXORNeXt delivers \underline{73.45\%} $\text{mAP}_9$, confirming balanced multi-class detection across varying occlusion and truncation levels.

\textit{4. Multi-Seed Statistical Robustness and Variance Analysis.}
To verify that the empirical gains of MobilePIXORNeXt are statistically consistent rather than an artifact of single-seed weight initialization ($\text{seed}=42$), Table~\ref{tab:seed_robustness} reports benchmark evaluations across three distinct random seeds ($42, 43, 44$) on the unified validation protocol. Across all three seeds, MobilePIXORNeXt consistently maintains high detection accuracy, achieving a mean Moderate mAP of $\mathbf{70.94 \pm 0.89\%}$ and overall $\text{mAP}_9$ of $\mathbf{73.01 \pm 0.68\%}$, representing decisive average improvements of $\mathbf{+23.73\,\text{pp}}$ and $\mathbf{+24.46\,\text{pp}}$ over the baseline MobilePIXOR ($47.21 \pm 5.06\%$ and $48.55 \pm 5.45\%$, respectively). 

Crucially, MobilePIXOR exhibits severe variance across runs, particularly on vulnerable road users where Pedestrian and Cyclist AP standard deviations reach $\pm 6.49\,\text{pp}$ and $\pm 7.37\,\text{pp}$. In contrast, MobilePIXORNeXt bounds standard deviations to $\le 1.30\,\text{pp}$ across all individual classes ($\pm 0.75\,\text{pp}$ for Car, $\pm 0.95\,\text{pp}$ for Pedestrian, and $\pm 1.30\,\text{pp}$ for Cyclist) and $\le 0.89\,\text{pp}$ on aggregate mAP. This low variance confirms that our architectural stabilization mechanisms—principally LayerScale residual gating, T-SBUW multi-task uncertainty balancing, and physics-consistent data augmentation (PC-GTA)—confer robust numerical stability and prevent gradient divergence during multi-task optimization.

\subsection{Inference Latency and Real-Time Complexity}
\label{sec:latency_analysis}
\input{tables/efficiency}

\paragraph{Pipeline Latency vs. Model Forward Throughput.}
Network forward time covers only part of a real-time perception pipeline. Full-system latency includes point cloud loading, tensorization, Host-to-Device (H2D) transfer, network execution, coordinate decoding, and Non-Maximum Suppression (NMS). To prevent ambiguity, we formally distinguish between two latency measurement boundaries:
\begin{enumerate}
    \item \textbf{Standalone GPU Forward Latency} ($8.67\,\text{ms}$, $115.32\,\text{FPS}$): Pure neural network execution measured over pre-allocated GPU tensors using asynchronous \texttt{torch.cuda.Event} markers, isolating raw compute capability from system I/O and evaluation harness overhead.
    \item \textbf{Pipeline Synchronous Model Stage} ($17.88\,\text{ms}$, $55.94\,\text{FPS}$): Network execution within the active end-to-end evaluation harness, including DataLoader tensor marshalling, CUDA stream synchronization, and intermediate buffer allocations across frames.
\end{enumerate}
Table~\ref{tab:efficiency} reports sequential latency at batch size 1 over $3{,}759$ timed frames, excluding 10 warmup frames from the $3{,}769$-frame validation split. All detectors, including MobilePIXOR and reference networks, are evaluated under identical conditions on the same NVIDIA Tesla T4 GPU.



\textit{1. End-to-End Pipeline Latency Comparison.}
Table~\ref{tab:efficiency} reports MobilePIXORNeXt's End-to-End (E2E) latency of \textbf{60.96\,ms} (\textbf{16.41\,FPS}), the lowest mean pipeline latency among the evaluated real-time detectors. MobilePIXORNeXt ran faster than PointPillars ($63.84\,\text{ms}$, $15.66\,\text{FPS}$), MobilePIXOR ($90.65\,\text{ms}$, $11.03\,\text{FPS}$), and SECOND ($93.70\,\text{ms}$, $10.67\,\text{FPS}$). It also ran over $2.3\times$ faster than PIXOR ($144.00\,\text{ms}$, $6.94\,\text{FPS}$). Percentile analysis reveals distinct operational characteristics across traffic density regimes: MobilePIXORNeXt achieves a superior median latency ($\text{P50} = \mathbf{58.28\,\text{ms}}$ vs. $62.10\,\text{ms}$ for PointPillars). In dense multi-object traffic scenes, proposal candidate extraction and multi-class rotated NMS expand tail latency ($\text{P95} = 80.85\,\text{ms}$, $\text{P99} = 95.91\,\text{ms}$), where PointPillars exhibits tighter tail dispersion ($75.88\,\text{ms}$ and $92.73\,\text{ms}$). Both architectures comfortably fulfill the strict $100\,\text{ms}$ automotive real-time constraint across $\ge 99\%$ of operational driving frames.

\textit{2. Component Latency Profiling and Host-to-Device Efficiency.}
Stage-level profiling identified the breakdown across pipeline stages, totaling $60.67\,\text{ms}$ across the four core modules plus a minor $0.29\,\text{ms}$ ($0.48\%$) evaluation harness residual (accounting for frame metadata indexing, tensor unbinding, and bounding-box formatting):
\begin{itemize}
    \item \textbf{Input Preprocessing ($20.94\,\text{ms}$)}: Point-cloud file loading and Strata-BEV14 rasterization with the Numba CPU backend took $20.94\,\text{ms}$ combined. The implementation uses occupied-cell accumulators and precomputed logarithmic lookup tables. Recorded preprocessing took $27.75\,\text{ms}$ for PointPillars and $27.20\,\text{ms}$ for SECOND. These measurements cover complete input preparation from disk to memory.
    \item \textbf{Host-to-Device Memory Transfer ($6.91\,\text{ms}$)}: Binary occupancy tensors with a 35-channel layout require $78.85\,\text{MB}$ per scan ($800 \times 704 \times 35 \times 4$ bytes). The Strata-BEV14 14-channel layout requires $31.54\,\text{MB}$, reducing tensor memory payload by exactly $2.50\times$ ($60.0\%$). Recorded H2D transfer time on Tesla T4 with pinned host memory buffers was $6.91\,\text{ms}$ for Strata-BEV14 ($31.54\,\text{MB}$), compared with $40.63\,\text{ms}$ for unpinned 35-channel transfers ($78.85\,\text{MB}$, Table~\ref{tab:encoding}). The resulting $5.88\times$ transfer acceleration reflects the combined effect of the $2.50\times$ payload reduction and page-locked pinned memory direct DMA transfers across the PCIe bus, which bypasses OS page paging.
    \item \textbf{Model Forward Propagation ($17.88\,\text{ms}$)}: Network inference within the sequential evaluation pipeline took $17.88\,\text{ms}$ ($55.94\,\text{FPS}$) on Tesla T4, outperforming MobilePIXOR's corresponding forward inference ($25.43\,\text{ms}$, $39.32\,\text{FPS}$). In the isolated benchmarking setup with pre-allocated tensors and \texttt{torch.cuda.Event} timing (Table~\ref{tab:components}), standalone model forward execution reached \textbf{115.32\,FPS} ($8.67\,\text{ms}$).
    \item \textbf{Decoding and Rotated NMS ($14.94\,\text{ms}$)}: Candidate peak selection and class-wise rotated NMS took $14.94\,\text{ms}$. Data remained on the GPU until final bounding box export.
\end{itemize}


\subsection{Distance-Stratified Detection Performance}
\label{sec:distance_stratified}



\paragraph{Range-Dependent Accuracy on Vulnerable Road Users.}
Because LiDAR beam divergence causes point density to drop quadratically over distance ($N \propto 1/r^2$), detector reliability degrades rapidly at intermediate and long ranges. Table~\ref{tab:distance_ap} benchmarks Moderate AP across three discrete distance intervals: Near ($0\text{--}30\,\text{m}$), Intermediate ($30\text{--}50\,\text{m}$), and Far ($50\text{--}70.4\,\text{m}$).

\input{tables/distance_our_vs_mbpixor}



\textit{1. Near-Range Safety Envelope ($0\text{--}30\,\text{m}$).}
Compared with MobilePIXOR in the near-range $0\text{--}30\,\text{m}$ zone, MobilePIXORNeXt Pedestrian AP reached \textbf{65.23\%}, more than double MobilePIXOR's $30.45\%$ (a $+34.78\,\text{pp}$ improvement). Cyclist AP reached \textbf{76.32\%}, nearly triple MobilePIXOR's $25.55\%$ (a $+50.77\,\text{pp}$ improvement). These gains are consistent with Strata-BEV14's stratified height bins and continuous subcell offsets preserving fine boundaries when object returns are dense.

\textit{2. Intermediate and Long Ranges ($30\text{--}70.4\,\text{m}$).}
At medium ranges ($30\text{--}50\,\text{m}$), MobilePIXORNeXt achieved $11.49\%$ Pedestrian AP (vs. $3.75\%$) and $44.30\%$ Cyclist AP (vs. $12.61\%$, a $+31.69\,\text{pp}$ gain). Far-range ($50\text{--}70.4\,\text{m}$) Cyclist AP was $14.94\%$ (vs. $0.56\%$). These results are consistent with range-adaptive augmentation (\texttt{transform\_object}) and the wider context of $7 \times 7$ depthwise receptive fields helping under sparse returns. As qualitatively corroborated in our comparative visual analysis (Section~\ref{sec:qualitative_analysis}, Figure~\ref{fig:comparison_heatmap}), MobilePIXOR suffers severe activation attenuation on distant sparse returns, whereas MobilePIXORNeXt maintains focused focal peaks and confident bounding footprints.

\input{tables/components}

\subsection{Ablation Studies}
\label{sec:ablations}

\paragraph{Systematic Architectural Progression.}
To quantify the individual and cumulative contributions of the proposed components, Table~\ref{tab:components} details the step-by-step ablation progression from the baseline to the final configuration.

\paragraph{1. Backbone Architecture (A0 $\to$ A1).}
Replacing MobilePIXOR's residual blocks with a ConvNeXt-style backbone using $7 \times 7$ depthwise convolutions, inverted expansion $e=2.5$, SiLU, and LayerScale raised Moderate mAP from $41.40\%$ to $47.60\%$ (\textbf{+6.20\,pp Moderate mAP}), while standalone throughput nearly doubled from $90.57$ to \textbf{169.97\,FPS}. Large depthwise kernels broaden BEV receptive fields with limited computational cost.

\paragraph{2. BEV Input Representation (A1 $\to$ A2 $\to$ A3).}
\input{tables/encoding}
To assess how input rasterization affects detection accuracy and latency, Table~\ref{tab:encoding} compares four distinct BEV representations under identical MobilePIXORNeXt backbone and single-head architecture. TriBand-BEV~\cite{khoshkdahan2026triband} partitions points into three coarse height bands; while it delivers the highest inference throughput ($188.4\,\text{FPS}$) and lowest Host-to-Device transfer payload ($6.76\,\text{MB}$), its Moderate mAP drops to $44.42\%$ ($-5.72\,\text{pp}$ below Strata-BEV14). The lack of continuous elevation moments and horizontal subcell centroid offsets degrades small object localization (Pedestrian AP $38.05\%$, Cyclist AP $21.12\%$). Conventional 35-channel binary occupancy slicing (A1) achieves $47.60\%$ mAP but requires transferring $78.85\,\text{MB}$ per scan, imposing significant Host-to-Device transfer latency ($40.63\,\text{ms}$) while still discarding sub-voxel point coordinates. The 8-channel Strata-BEV8 representation (A2) yields comparable accuracy ($47.37\%$) with fewer channels. In contrast, our 14-channel Strata-BEV14 representation (A3) achieves the highest Moderate mAP of $50.14\%$ ($+5.72\,\text{pp}$ over TriBand-BEV and $+2.54\,\text{pp}$ over Binary35) at $169.0\,\text{FPS}$. Continuous vertical spatial moments ($z_{\max}, z_{\mathrm{mean}}, z_{\mathrm{span}}, z_{\mathrm{std}}$) and subcell centroid displacement offsets $(\Delta\bar{x}, \Delta\bar{y})$ retain critical 3D geometry otherwise lost through horizontal discretization without incurring PointNet encoder overhead.

\paragraph{3. Decoupled Grouped Prediction Towers (A3 $\to$ A4 $\to$ A5).}
Separating Car and Ped/Cyc prediction heads (A4) raised Moderate mAP to $51.30\%$ ($+1.16\,\text{pp}$ over A3). Increasing grouped head width from 16 to 32 channels (A5) raised Moderate mAP to $60.22\%$ (\textbf{+8.92\,pp Moderate mAP}) and $\text{mAP}_9$ to $61.11\%$ relative to configuration A4 (a combined $+10.08\,\text{pp}$ gain over A3). In configuration A4, compressing shared intermediate features into only 16 channels induces severe subspace rank collapse when the convolutional tower attempts to project four distinct regression targets (offset, log-size, doubled-angle yaw, and quality) across heterogeneous object scales. Expanding head width to 32 channels (A5) adds only $75\,\text{K}$ parameters but doubles the latent representation rank. This additional subspace capacity provides sufficient degrees of freedom to decouple inter-task gradients between orthogonal geometric properties, driving an $+8.92\,\text{pp}$ jump ($+10.08\,\text{pp}$ over baseline A3).

\paragraph{4. Physics-Consistent Ground-Truth Augmentation (A5 $\to$ A6).}
Physics-consistent augmentation (\texttt{PC-GTA}) improved Moderate mAP by \textbf{+6.03\,pp} over A5, from $60.22\%$ to $66.25\%$. It also raised $\text{mAP}_9$ from $61.11\%$ to $68.45\%$ (\textbf{+7.34\,pp $\text{mAP}_9$}). The evaluated sampler combined terrain alignment, static obstacle rejection, and range-dependent point subsampling. Optional LOS, shadow-masking, and radiometric mechanisms were disabled in the final run.

\paragraph{5. Multiscale Neck and Scale Gating (A6 $\to$ A7 $\to$ A8).}
Increasing neck width from 24 to 32 channels (A7) raised Moderate mAP to $68.12\%$ (\textbf{+1.87\,pp Moderate mAP}). The Top-Down Scale-Gated FPN (A8) replaced transposed convolutions with bilinear upsampling and learnable depthwise gating to avoid deconvolution checkerboard artifacts in continuous regression maps. Under standard unweighted Smooth-L1 coordinate regression, this depthwise gating causes a slight drop from $68.12\%$ to $67.38\%$ ($-0.74\,\text{pp}$), as multiplicative gating without explicit geometric loss coupling tends to attenuate weak gradient signals from distant and sparse point returns.

\paragraph{6. Geometric Supervision and Localization Quality (A8 $\to$ A9).}
Oriented Geometric Alignment (OGA) and IoU-Aware Quality Alignment (IQA) raised Moderate mAP from $67.38\%$ to $70.58\%$ (\textbf{+3.20\,pp Moderate mAP}). This $+3.20\,\text{pp}$ surge reveals a decisive co-design synergy: while SG-FPN alone slightly attenuates weak returns under Smooth-L1, it provides the precise multi-scale spatial separation that OGA and IQA require. Multi-Axis Projection GIoU and scale-normalized $\pi$-symmetric corner distances provide direct, rotation-aware gradient feedback across the scale-gated features, while IQA calibrates classification confidence against continuous projection-based geometric quality ($\hat{q}^*$) to eliminate the suppression of accurate candidates by poorly aligned boxes during rotated NMS.

\paragraph{7. Block-SimAM Energy Attention (A9 $\to$ A10).}
Embedding parameter-free Block-SimAM inside all nine residual blocks across stages C2, C3, and C4 (depths $3, 4, 2$) produced the final configuration A10, reaching \textbf{71.50\% Moderate mAP} and \textbf{73.45\% mAP$_9$} at the converged 100-epoch deployment checkpoint (\textbf{+0.92\,pp Moderate mAP} over A9, while reaching $71.48\%$ and $73.57\%$ at 50 epochs).
Crucially, because both A9 and A10 maintain the identical backbone depth of nine blocks ($3+4+2$) and identical head structures, this $+0.92\,\text{pp}$ accuracy gain is strictly attributable to 3D spatial-energy attention without any confounding depth alterations or learnable parameter inflation (adding exactly 0 parameters). The minimal increase of $0.48\,\text{ms}$ in standalone forward latency ($8.19 \to 8.67\,\text{ms}$) confirms the extreme efficiency of the analytical energy minimization.
Block-SimAM gained 1.66--2.10\,pp in Moderate mAP over stage-level variants at 115.32 standalone FPS (versus 144.05\,FPS for Stage-3 SimAM).


\paragraph{Geometric Supervision and Quality Head Factorial Study.}
Table~\ref{tab:loss} presents a $2 \times 2$ factorial evaluation isolating the effects of OGA geometric supervision and the explicit IQA prediction branch.

\input{tables/loss}

Table~\ref{tab:loss} reports the OGA and IQA ablation. Notably, in the absence of an auxiliary IQA branch, OGA alone raises Moderate mAP from $67.38\%$ to $68.97\%$ (\textbf{+1.59\,pp Moderate mAP} over baseline Smooth-$L_1$), with decisive gains across all three classes: Car ($+0.17\,\text{pp}$), Pedestrian ($+2.45\,\text{pp}$), and Cyclist ($+2.14\,\text{pp}$). This confirms that rotation-aware geometric alignment independently guides coordinate convergence even without confidence re-ranking. When IQA is enabled, OGA further elevates Car AP to its benchmark peak of $\mathbf{84.87\%}$ (versus $84.13\%$ for Baseline + IQA), yielding a joint optimum of \textbf{70.58\% Moderate mAP}. These results demonstrate clear methodological synergy: MGIoU and scale-normalized $\pi$-corner distances enforce strict boundary and orientation gradients during backpropagation, while IQA adjusts detection rank via $S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}}\hat{q}}$ to prioritize geometrically faithful candidates during rotated NMS.

Table~\ref{tab:loss} compares explicit quality prediction with head-free quality coupling. IQA models contained $0.754\,\text{M}$ parameters ($753{,}569$ parameters). Q-OGA (Eq.~(\ref{eq:cqfl_target})--(\ref{eq:cqfl_loss})) omitted the IQA heads, using $0.735\,\text{M}$ parameters ($734{,}943$ parameters, saving $18{,}626$ parameters). Without extra prediction branches or post-processing score multiplication, Q-OGA achieved $69.20\%$ Moderate mAP, exceeding OGA alone by $+0.23\,\text{pp}$ and baseline Smooth-L1 by $+1.82\,\text{pp}$. Q-OGA therefore offers a smaller, lightweight configuration for constrained deployment, while explicit IQA achieved the highest accuracy ($70.58\%$) with a dedicated quality branch.

\paragraph{3D Spatial-Energy Attention Integration and Placement.}
Table~\ref{tab:attention} investigates the placement of parameter-free SimAM energy attention.

\input{tables/attention}

Stage-level SimAM at downsampling boundaries ($C_2, C_3, C_4$) reduced mAP to $69.44\%$, $69.80\%$, and $69.36\%$, below the no-attention score ($70.58\%$). Attention at these transitions may blur fine spatial gradients after downsampling. Block-SimAM inside every residual block instead achieved \textbf{71.50\% Moderate mAP} (\textbf{+0.92\,pp gain} over no attention) and \textbf{73.45\% mAP$_9$} at the converged 100-epoch deployment checkpoint ($71.48\%$ and $73.57\%$ at 50 epochs). It modulates the residual update before LayerScale addition ($\mathbf{Z}' = \mathbf{Z} + \boldsymbol{\gamma}_{\mathrm{blk}} \odot \mathcal{S}(\mathbf{R})$) without changing inter-stage feature resolution, allowing modulation of sparse returns within each block.

\paragraph{Multi-Task Optimization Dynamics and Convergence.}
Figure~\ref{fig:loss_curves} plots training and validation losses over 50 epochs for configurations A1 through A10. It includes the total objective and six tasks: Classification, Geometry, IoU Quality, Subcell Offset, Size, and Yaw. We used these trajectories to assess training stability and convergence.
Across 50 epochs, total and task losses exhibited steady downward convergence trends. Total loss fell from $10^4$ to approximately $1.0$ within the first 15 epochs, progressing stably through cosine annealing. While transient validation classification spikes were observed between epochs 18 and 25 (briefly reaching $\sim 10^2$ before dampening), the training and validation regression curves remained well-bounded without unbounded divergence. This trend reflects the stabilizing effect of physics-consistent data augmentation (PC-GTA, A6) and loss balancing.
Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW, Eq.~(\ref{eq:tsbuw_weights})--(\ref{eq:balanced_loss})) dynamically bounded task loss weights ($\omega_k \in (0, J)$), mitigating magnitude disparities and objective competition between cross-entropy classification and multi-task coordinate regression. Configuration A10 achieved low, stable late-training validation losses across regression tasks, aligning with the empirical evaluation in Table~\ref{tab:components}.

\paragraph{Training Schedule Fairness and Convergence Validation.}
To evaluate training duration and convergence dynamics within MobilePIXORNeXt, we benchmark the rapid ablation checkpoint from the 50-epoch schedule (Epoch~47) against the fully converged 100-epoch deployment checkpoint (Epoch~100). Under the local loader-aligned BEV evaluator with $R_{40}$ interpolation, the 50-epoch checkpoint attained $71.48\%$ Moderate mAP ($82.71\%$ Car, $62.30\%$ Pedestrian, $69.43\%$ Cyclist, and $73.57\%$ $\text{mAP}_9$), closely matching the $71.50\%$ Moderate mAP and $73.45\%$ $\text{mAP}_9$ achieved by the 100-epoch deployment model ($86.90\%$ Car, $60.78\%$ Pedestrian, $66.81\%$ Cyclist). These consistent aggregate figures confirm that the rapid 50-epoch schedule used for the architectural ablations (Table~\ref{tab:components}) faithfully captures converged model performance without distortion. The essential benefit of training to 100 epochs is the progressive calibration of orientation and sharpening of foreground classification heatmaps, which systematically suppresses low-confidence background false alarms. By drastically reducing the candidate proposals routed to post-processing and rotated NMS, this convergence effect slashes end-to-end pipeline latency by $25.0\%$ ($81.23\,\text{ms} \to 60.96\,\text{ms}$, elevating throughput from $12.31$ to $16.41\,\text{FPS}$).


\begin{figure*}[p]
\centering
\paperfigure[width=\textwidth,height=0.88\textheight,keepaspectratio]{training_curves_all_losses.pdf}{0.88\textheight}{Training and validation loss trajectories across 50 epochs.}
\captionof{figure}{Empirical multi-task loss trajectories over 50 training epochs on the KITTI benchmark for architectural configurations A1--A10 (Table~\ref{tab:components}). Subpanels depict: \textbf{(a)} Total training loss, \textbf{(b)} Total validation loss, \textbf{(c)} Focal classification loss, \textbf{(d)} Geometric OGA loss ($\ell_{\mathrm{proj}} + \ell_{\mathrm{corner}}$), \textbf{(e)} IQA quality alignment loss, \textbf{(f)} Subcell horizontal offset loss, \textbf{(g)} Footprint bounding dimension loss, and \textbf{(h)} Doubled-angle yaw orientation loss. Trajectories illustrate stable monotonic descent from initial magnitudes ($10^4$) to asymptotic convergence ($\sim 1.0$), with transient validation classification fluctuations at epochs 18--25 rapidly stabilized by T-SBUW adaptive weighting.}
\label{fig:loss_curves}
\end{figure*}



\subsection{Qualitative Visualizations and Analysis}
\label{sec:qualitative_analysis}

\begin{figure*}[!t]
\centering
\paperfigure[width=\textwidth]{comparison_mobilepixor_heatmap.pdf}{65mm}{Comparative BEV detection and classification heatmap visualizations between MobilePIXOR and MobilePIXORNeXt.}
\captionof{figure}{Qualitative comparison of Bird's-Eye View (BEV) detections and predicted classification heatmaps between the baseline MobilePIXOR and MobilePIXORNeXt (Ours) across two representative driving scenes. \textbf{Row A}: MobilePIXOR exhibits diffused, low-amplitude heatmap responses with noticeable background clutter, leading to missed detections (false negatives) on distant and sparse point clusters. \textbf{Row B}: MobilePIXORNeXt produces sharp, concentrated focal peaks strictly centered on vehicle targets with suppressed background activations, successfully recovering distant objects and yielding tightly aligned oriented bounding footprints.}
\label{fig:comparison_heatmap}
\end{figure*}

\paragraph{Comparative Heatmap and Detection Analysis against Baseline.}
Figure~\ref{fig:comparison_heatmap} provides a side-by-side qualitative comparison between the baseline MobilePIXOR and MobilePIXORNeXt across two challenging driving scenarios, illustrating both oriented BEV bounding footprints and the corresponding foreground classification heatmaps.

Two critical observations emerge from this comparison:
\begin{itemize}
    \item \textbf{Focal Heatmap Concentration and Clutter Suppression}: In MobilePIXOR (Row~A), the classification heatmaps exhibit diffuse, low-intensity activation clusters (frequently below $0.35$ normalized confidence) accompanied by spurious background activations in uninformative regions. In contrast, MobilePIXORNeXt (Row~B) generates sharp, high-magnitude focal peaks ($>0.80$) centered precisely at vehicle centroids, while effectively suppressing background clutter to near-zero activation. This sharpening directly validates the efficacy of our Block-SimAM spatial-energy attention and Top-Down Scale-Gated FPN, which collaboratively amplify sparse point features while filtering scale-inconsistent lateral noise. As analyzed in Section~\ref{sec:ablations}, this background suppression reduces low-confidence candidate proposals, directly speeding up Rotated NMS.
    \item \textbf{Distant and Sparse Object Recall}: At intermediate and long ranges ($>30\,\text{m}$), where LiDAR point returns become severely attenuated ($N \propto 1/r^2$), MobilePIXOR fails to exceed the peak detection threshold, producing false negatives. MobilePIXORNeXt reliably recovers these distant vehicles with confident heatmap activations and well-oriented bounding boxes. This visual evidence corroborates the substantial $+31.69\,\text{pp}$ Cyclist AP and $+7.74\,\text{pp}$ Pedestrian AP gains reported in Table~\ref{tab:distance_ap} for the $30\text{--}50\,\text{m}$ interval, demonstrating the benefits of Strata-BEV14's continuous elevation moments and the enlarged $7 \times 7$ depthwise receptive field.
\end{itemize}

\paragraph{Visual Inspection on Representative Driving Scenarios.}
Figure~\ref{fig:qualitative_detections} presents MobilePIXORNeXt predictions on three KITTI validation frames (top to bottom: \texttt{007038}, \texttt{005571}, and \texttt{006761}). Class labels and confidence scores are retained in the camera overlays to support inspection of the predicted footprints; the BEV panels retain class annotations.

\begin{figure*}[!t]
\centering
\paperfigure[width=\textwidth,height=0.38\textheight,keepaspectratio]{qualitative_detections.pdf}{65mm}{Qualitative BEV detection visualizations on KITTI validation scenes.}
\captionof{figure}{Qualitative BEV predictions from MobilePIXORNeXt (converged MobilePIXORNeXt deployment model) on KITTI validation frames \texttt{007038}, \texttt{005571}, and \texttt{006761} (top to bottom). \textbf{Left}: Front-camera images with calibrated projections of predicted BEV footprints, class labels, and confidence scores. \textbf{Right}: Corresponding LiDAR BEV views with predicted oriented footprints and class annotations. Prediction colors are cyan for Car, magenta for Pedestrian, and yellow for Cyclist. The panels visualize predicted footprints rather than full 3D boxes.}
\label{fig:qualitative_detections}
\end{figure*}

The paired camera and BEV views in Figure~\ref{fig:qualitative_detections} support inspection of the following prediction patterns:
\begin{itemize}
    \item \textbf{Vehicle Predictions}: Frame \texttt{007038} contains several predicted vehicle footprints at different forward distances, visible in both the camera projection and the BEV panel.
    \item \textbf{Mixed-Class Predictions}: Frame \texttt{005571} contains Car, Pedestrian, and Cyclist predictions. The retained labels and confidence scores identify the classes assigned to nearby footprints.
    \item \textbf{Nearby Road Users}: Frame \texttt{006761} shows predicted footprints around vehicles, pedestrians, and cyclists in the same scene. The two views reveal the spatial arrangement of these predictions.
\end{itemize}

\subsection{Discussion and Practical Limitations}
\label{sec:discussion_limitations}

\paragraph{Discussion and Insights.}
The results showed that accurate 3D LiDAR detection was possible without large point-cloud encoders or 3D sparse convolutions. The deterministic 2D BEV representation retained multi-band densities, elevation dispersion, and continuous subcell centroid displacements. A ConvNeXt-style 2D network extracted geometric features with fewer than $0.76\,\text{M}$ parameters. Physics-consistent augmentation (\texttt{PC-GTA}) and decoupled grouped heads provided the most substantial individual accuracy improvements in the architectural progression, demonstrating the value of domain-specific LiDAR data modeling and multi-class gradient isolation.

\paragraph{Limitations and Future Scope.}
MobilePIXORNeXt is formulated specifically for high-throughput, oriented 2D Bird's-Eye View footprint detection on the $SE(2)$ manifold, directly fulfilling the horizontal clearance and Time-to-Collision (TTC) constraints of vehicular collision avoidance. While an auxiliary vertical regression branch can be optionally incorporated to recover complete 3D cuboid parameters $(z, h)$, our core benchmark focuses on $SE(2)$ BEV localization. Several operational boundaries remain. First, our empirical validation relies on 64-beam mechanical spinning LiDAR on predominantly planar urban roadways. Although Strata-BEV14's continuous elevation moments ($z_{\mathrm{span}}, z_{\mathrm{std}}$) and log-density normalization maintain numerical stability under point sparsity, performance on ultra-sparse 16-beam or non-repetitive solid-state LiDAR sensors warrants systematic domain investigation. Second, on severe terrain slopes or multi-level grade transitions, planar BEV projections introduce perspective distortion along vertical structural boundaries. Future work will address these constraints by developing Range-Conditioned Scale-Gated Feature Pyramids (RC-SGFPN) with continuous spatial frequency modulation, and deploying hardware-tailored TensorRT INT8 quantization for low-power edge microcontrollers.


% SOURCE: tables/attention.tex
\begin{table*}[t]
\centering\normalsize
\caption{Ablation of 3D spatial-energy attention (SimAM) integration topologies and placement strategies within the MobilePIXORNeXt backbone on the KITTI validation benchmark ($R_{40}$). Stage-level variants append a single attention adapter at stage downsampling boundaries ($C_2, C_3, C_4$), whereas the proposed Block-SimAM embeds parameter-free 3D energy attention inside all nine \texttt{MobilePIXORNeXt-S} residual blocks before LayerScale scaling. Best values are in \textbf{bold}, and second-best are \underline{underlined}.}
\label{tab:attention}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrrr@{}}
\toprule
\textbf{\shortstack[l]{SimAM Attention\\Placement}} & \textbf{\shortstack[l]{Neck Fusion\\Width}} & \textbf{\shortstack{mAP$_{\mathrm{M}}$\\(\%) $\uparrow$}} & \textbf{\shortstack{mAP$_{9}$\\(\%) $\uparrow$}} & \textbf{\shortstack{Standalone\\Forward (ms) $\downarrow$}} & \textbf{\shortstack{Standalone\\FPS $\uparrow$}} \\
\midrule
None (No attention) & $F_{32}$ & \underline{70.58} & \underline{72.66} & \underline{8.19} & \underline{122.16} \\
Stage-2 ($C_2$) & $F_{32}$ & 69.44 & 71.49 & 7.30 & 136.91 \\
Stage-3 ($C_3$) & $F_{32}$ & 69.80 & 71.98 & \textbf{6.94} & \textbf{144.05} \\
Stage-4 ($C_4$) & $F_{32}$ & 69.36 & 71.34 & 7.44 & 134.43 \\
\textbf{Block-level ($C_2, C_3, C_4$)} & \textbf{$F_{32}$} & \textbf{71.50} & \textbf{73.45} & 8.67 & 115.32 \\
\bottomrule
\end{tabular}
\par\vspace{2pt}
{\footnotesize \textit{Note.} All configurations share OGA, the IQA quality head, Strata-BEV14 input, and PC-GTA augmentation. Standalone forward latency is measured in FP32 on an NVIDIA Tesla T4 GPU using isolated batches. In the complete pipeline (Table~\ref{tab:efficiency}), forward propagation takes $55.94\,\text{FPS}$ ($17.88\,\text{ms}$), and overall E2E speed is $16.41\,\text{FPS}$ ($60.96\,\text{ms}$). Stage-level variants add one adapter at a stage boundary. Block-SimAM applies 3D parameter-free energy attention in each of the nine \texttt{MobilePIXORNeXt-S} residual blocks before LayerScale scaling.}
\end{table*}

% SOURCE: tables/comparison.tex
\begin{table*}[t]
\centering\small
\caption{Quantitative comparison of Bird's-Eye View (BEV) object detection performance on the KITTI validation benchmark. Average Precision (AP, \%) is evaluated under the 40-point interpolation protocol ($R_{40}$) at Moderate difficulty ($\text{IoU} \ge 0.70$ for Car; $\text{IoU} \ge 0.50$ for Pedestrian and Cyclist). All experiments are conducted using the unified evaluation implementation and benchmarked on the same NVIDIA Tesla T4 GPU platform. The top partition benchmarks single-stage real-time and edge-oriented LiDAR detectors (spanning ultra-compact BEV representations to pillar and 3D sparse voxel networks); the bottom partition provides external reference with heavyweight two-stage 3D anchor-based detectors. For single-stage methods, the best results are highlighted in \textbf{bold}, and the second-best results are \underline{underlined}.}
\label{tab:comparison}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Method} & \textbf{\shortstack{Params\\(M) $\downarrow$}} & \textbf{\shortstack{Car AP$_{\mathrm{M}}$\\(\%)}} & \textbf{\shortstack{Pedestrian\\AP$_{\mathrm{M}}$ (\%)}} & \textbf{\shortstack{Cyclist\\AP$_{\mathrm{M}}$ (\%)}} & \textbf{\shortstack{mAP$_{\mathrm{M}}$\\(\%)}} & \textbf{\shortstack{mAP$_{9}$\\(\%)}} \\
\midrule
PIXOR (Car only)~\cite{yang2018pixor} & 2.135 & 72.02 & -- & -- & -- & -- \\
MobilePIXOR~\cite{nguyen2024indoor} & \textbf{0.598} & 75.52 & 28.16 & 20.53 & 41.40 & 42.27 \\
PointPillars~\cite{lang2019pointpillars} & 4.835 & \underline{88.02} & 56.01 & 66.34 & 70.13 & 72.25 \\
SECOND~\cite{yan2018second} & 5.326 & \textbf{88.56} & \underline{56.56} & \textbf{71.16} & \textbf{72.09} & \textbf{73.79} \\
\textbf{MobilePIXORNeXt (Ours)} & \underline{0.754} & 86.90 & \textbf{60.78} & \underline{66.81} & \underline{71.50} & \underline{73.45} \\
\midrule
\multicolumn{7}{@{}l}{\textit{Heavyweight 3D Anchor-Based Reference Detectors}} \\
PointRCNN~\cite{shi2019pointrcnn} & 4.039 & 88.42 & 57.84 & 73.87 & 73.38 & 75.31 \\
Part-A$^2$-Anchor~\cite{shi2020part} & 63.815 & 90.05 & 64.20 & 74.60 & 76.28 & 78.04 \\
PV-RCNN~\cite{shi2020pvrcnn} & 13.116 & 90.29 & 58.48 & 74.54 & 74.44 & 76.49 \\
\bottomrule
\end{tabular}
\par\vspace{2pt}
{\footnotesize \textit{Note.} $\text{mAP}_{\mathrm{M}}$ represents the mean of the three Moderate class AP scores; $\text{mAP}_{9}$ averages all nine class--difficulty evaluation cells. Dashes (--) indicate unsupported classes for the single-class PIXOR detector. All comparative experiments are conducted under the unified evaluation file and executed on the same NVIDIA Tesla T4 GPU hardware platform with batch size 1 and FP32 precision to ensure a fair and controlled benchmark. Complete pipeline latencies and runtime component breakdowns are reported in Table~\ref{tab:efficiency}.}
\end{table*}


% SOURCE: tables/components.tex
\begin{table*}[t]
\centering\small
\caption{Architectural ablations on KITTI validation using Moderate BEV AP $R_{40}$. Panel (a) lists additions from baseline MobilePIXOR (A0) to the final MobilePIXORNeXt configuration (A10). Panel (b) reports accuracy ($\text{mAP}_{\mathrm{M}}$, $\text{mAP}_9$) and standalone forward latency and throughput (ms, FPS) on an NVIDIA Tesla T4 GPU. Best values are in \textbf{bold}, and second-best are \underline{underlined}.}
\label{tab:components}
\textbf{(a) Architectural configurations and component progression}\par\smallskip
\begingroup\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lllllll@{}}
\toprule
\textbf{Step} & \textbf{Backbone} & \shortstack[l]{\textbf{BEV}\\ \textbf{Input}} & \shortstack[l]{\textbf{Head}\\ \textbf{Architecture}} & \shortstack[l]{\textbf{GT}\\ \textbf{Sampling}} & \shortstack[l]{\textbf{Fusion /}\\ \textbf{Neck}} & \shortstack[l]{\textbf{Supervision /}\\ \textbf{Attention}} \\
\midrule
A0 & MobilePIXOR & Binary35  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\
A1 & MobilePIXORNeXt & Binary35  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\
A2 & MobilePIXORNeXt & Strata-BEV8  & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\
A3 & MobilePIXORNeXt & Strata-BEV14 & Single H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\
A4 & MobilePIXORNeXt & Strata-BEV14 & Grouped H16 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\
A5 & MobilePIXORNeXt & Strata-BEV14 & Grouped H32 & Standard & F24 / SumFPN & Baseline Smooth-L1 \\
A6 & MobilePIXORNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F24 / SumFPN & Baseline Smooth-L1 \\
A7 & MobilePIXORNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SumFPN & Baseline Smooth-L1 \\
A8 & MobilePIXORNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & Baseline Smooth-L1 \\
A9 & MobilePIXORNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & OGA + IQA Head \\
A10 & MobilePIXORNeXt & Strata-BEV14 & Grouped H32 & PC-GTA & F32 / SG-FPN & OGA + IQA + Block-SimAM \\
\bottomrule
\end{tabular}\endgroup
\par\medskip
\textbf{(b) Quantitative detection accuracy and standalone model inference speed}\par\smallskip
\begingroup\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l>{\raggedright\arraybackslash}p{0.42\textwidth}rrrr@{}}
\toprule
\textbf{Step} & \textbf{\shortstack[l]{Key Modification /\\Investigated Factor}} & \textbf{\shortstack{mAP$_{\mathrm{M}}$\\(\%) $\uparrow$}} & \textbf{\shortstack{mAP$_{9}$\\(\%) $\uparrow$}} & \textbf{\shortstack{Standalone Forward\\(ms) $\downarrow$}} & \textbf{\shortstack{Standalone FPS\\$\uparrow$}} \\
\midrule
A0 & Baseline MobilePIXOR starting checkpoint & 41.40 & 42.27 & 11.05 & 90.57 \\
A1 & + ConvNeXt-style backbone ($7\times 7$ DW, SiLU, Inverted Bottleneck) & 47.60 & 49.08 & \textbf{5.88} & \textbf{169.97} \\
A2 & + 8-channel Strata-BEV8 statistical representation & 47.37 & 48.95 & 6.06 & 165.02 \\
A3 & + 14-channel Strata-BEV14 (subcell offsets, height moments) & 50.14 & 52.00 & \underline{5.92} & \underline{169.01} \\
A4 & + Decoupled GroupedHeader (Car vs. Ped/Cyc, 16 channels) & 51.30 & 53.96 & 6.92 & 144.58 \\
A5 & + Expanded GroupedHeader capacity (H32, 32 channels) & 60.22 & 61.11 & 6.70 & 149.15 \\
A6 & + Physics-Consistent Ground-Truth Augmentation (PC-GTA) & 66.25 & 68.45 & 7.45 & 134.25 \\
A7 & + Expanded neck fusion layer ($F_{32}$, 32 channels) & 68.12 & 70.05 & 6.72 & 148.82 \\
A8 & + Bilinear Depthwise Scale-Gated Neck (SG-FPN) & 67.38 & 69.46 & 6.94 & 144.16 \\
A9 & + Oriented Geometric Alignment (OGA) and IQA supervision & \underline{70.58} & \underline{72.66} & 8.19 & 122.16 \\
A10 & + Parameter-Free Block-SimAM Energy Attention inside all blocks & \textbf{71.50} & \textbf{73.45} & 8.67 & 115.32 \\
\bottomrule
\end{tabular*}\endgroup
\par\vspace{2pt}
{\footnotesize \textit{Note.} Step identifiers A0--A10 trace additions from the baseline to the final configuration. All intermediate ablation configurations A1--A10 were trained under a standardized 50-epoch ablation schedule to isolate component gains under identical compute; at 50 epochs, configuration A10 attained $71.48\%$ $\text{mAP}_{\mathrm{M}}$ and $73.57\%$ $\text{mAP}_9$. From this systematic ablation, the selected MobilePIXORNeXt architecture (A10) and baseline MobilePIXOR (A0) were trained to 100 epochs to ensure full asymptotic convergence for primary benchmark comparisons against external detectors in Table~\ref{tab:comparison}, reaching $71.50\%$ and $41.40\%$ $\text{mAP}_{\mathrm{M}}$, respectively. Standalone forward latency is measured in FP32 on an NVIDIA Tesla T4 GPU, excluding I/O and NMS. In the complete pipeline (Table~\ref{tab:efficiency}), forward execution takes $55.94\,\text{FPS}$ ($17.88\,\text{ms}$) with evaluation-harness synchronization. Full-system E2E latency is $60.96\,\text{ms}$ ($16.41\,\text{FPS}$).}
\end{table*}


% SOURCE: tables/distance_our_vs_mbpixor.tex
\begin{table}[!htbp]
\centering\footnotesize
\caption{Distance-stratified detection performance (Moderate AP, \%) on Vulnerable Road Users (Pedestrian and Cyclist) across Near ($0\text{--}30\,\text{m}$), Intermediate ($30\text{--}50\,\text{m}$), and Far ($50\text{--}70.4\,\text{m}$) range intervals on the KITTI validation benchmark using BEV AP $R_{40}$. Best results across competing compact models are in \textbf{bold}.}
\label{tab:distance_ap}
\setlength{\tabcolsep}{1.5pt}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrr@{}}
\toprule
\textbf{Category} & \textbf{Model} & \textbf{\shortstack{0--30\,m}} & \textbf{\shortstack{30--50\,m}} & \textbf{\shortstack{50--70.4\,m}} \\
\midrule
Pedestrian & MobilePIXOR~\cite{nguyen2024indoor} & 30.45 & 3.75 & \textbf{0.99} \\
& \textbf{Ours} & \textbf{65.23} & \textbf{11.49} & 0.80 \\
& \textit{Diff.} & \textit{+34.78} & \textit{+7.74} & \textit{-0.19} \\
\midrule
Cyclist & MobilePIXOR~\cite{nguyen2024indoor} & 25.55 & 12.61 & 0.56 \\
& \textbf{Ours} & \textbf{76.32} & \textbf{44.30} & \textbf{14.94} \\
& \textit{Diff.} & \textit{+50.77} & \textit{+31.69} & \textit{+14.38} \\
\bottomrule
\end{tabular*}
\end{table}


% SOURCE: tables/efficiency.tex
\begin{table*}[t]
\centering\small
\caption{Pipeline latency and throughput on KITTI validation (3,769 frames, including 3,759 timed frames after 10 warmup frames) on a unified NVIDIA Tesla T4 GPU platform. Batch size is 1, with FP32 precision. Forward latency covers network execution alone. End-to-End (E2E) latency includes raw LiDAR file I/O, statistical rasterization, Host-to-Device (H2D) transfer, network execution, coordinate decoding, and class-wise rotated NMS. Best values among single-stage models are in \textbf{bold}, and second-best are \underline{underlined}.}
\label{tab:efficiency}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
\textbf{Method} & \textbf{\shortstack{Params\\(M) $\downarrow$}} & \textbf{\shortstack{Model\\(ms) $\downarrow$}} & \textbf{\shortstack{Model\\FPS $\uparrow$}} & \textbf{\shortstack{E2E\\(ms) $\downarrow$}} & \textbf{\shortstack{E2E\\FPS $\uparrow$}} & \textbf{\shortstack{P50\\(ms) $\downarrow$}} & \textbf{\shortstack{P95\\(ms) $\downarrow$}} & \textbf{\shortstack{P99\\(ms) $\downarrow$}} \\
\midrule
PIXOR~\cite{yang2018pixor} & 2.135 & 70.79 & 14.13 & 144.00 & 6.94 & 142.29 & 159.38 & 175.43 \\
MobilePIXOR~\cite{nguyen2024indoor} & \textbf{0.598} & \underline{25.43} & \underline{39.32} & 90.65 & 11.03 & 89.75 & 101.46 & 114.56 \\
PointPillars~\cite{lang2019pointpillars} & 4.835 & 32.13 & 31.12 & \underline{63.84} & \underline{15.66} & \underline{62.10} & \textbf{75.88} & \textbf{92.73} \\
SECOND~\cite{yan2018second} & 5.326 & 62.64 & 15.96 & 93.70 & 10.67 & 92.02 & 107.42 & 126.35 \\
\textbf{MobilePIXORNeXt (Ours)} & \underline{0.754} & \textbf{17.88} & \textbf{55.94} & \textbf{60.96} & \textbf{16.41} & \textbf{58.28} & \underline{80.85} & \underline{95.91} \\
\midrule
\multicolumn{9}{@{}l}{\textit{Heavyweight 3D Anchor-Based Reference Detectors}} \\
PointRCNN~\cite{shi2019pointrcnn} & 4.039 & 169.36 & 5.90 & 194.50 & 5.14 & 191.24 & 213.48 & 235.10 \\
Part-A$^2$-Anchor~\cite{shi2020part} & 63.815 & 102.57 & 9.75 & 133.75 & 7.48 & 131.50 & 150.49 & 172.80 \\
PV-RCNN~\cite{shi2020pvrcnn} & 13.116 & 160.79 & 6.22 & 188.60 & 5.30 & 186.40 & 202.62 & 224.15 \\
\bottomrule
\end{tabular}
\par\vspace{2pt}
{\footnotesize \textit{Note.} All detectors, including MobilePIXOR and reference networks, are evaluated under identical conditions on the same NVIDIA Tesla T4 GPU. Model latency measures network execution in the evaluation pipeline. Standalone throughput without pipeline synchronization reaches $115.32\,\text{FPS}$ ($8.67\,\text{ms}$, Table~\ref{tab:components}). End-to-end (E2E) latency includes LiDAR loading, statistical rasterization, Host-to-Device (H2D) transfer, network execution, coordinate decoding, and class-wise rotated NMS. Percentiles (P50, P95, P99) cover 3,759 timed frames after 10 warmup frames.}
\end{table*}


% SOURCE: tables/encoding.tex
\begin{table*}[t]
\centering\small
\caption{Ablation of Bird's-Eye View (BEV) input representation paradigms on the KITTI validation split using the local BEV evaluator with $R_{40}$ interpolation (Section~\ref{sec:dataset_and_metrics}). All models are trained under strictly identical conditions (MobilePIXORNeXt backbone, F24 neck, single head, standard augmentation, baseline loss, seed 42) and benchmarked on an NVIDIA T4 GPU. Best results across all configurations are in \textbf{bold}, and second-best are \underline{underlined}.}
\label{tab:encoding}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcrrrrrrr@{}}
\toprule
\textbf{Representation} & \textbf{$C_{\mathrm{in}}$} & \textbf{\shortstack{H2D Payload\\(MB) $\downarrow$}} & \textbf{\shortstack{Car AP$_{\mathrm{M}}$\\(\%) $\uparrow$}} & \textbf{\shortstack{Pedestrian\\AP$_{\mathrm{M}}$ (\%) $\uparrow$}} & \textbf{\shortstack{Cyclist\\AP$_{\mathrm{M}}$ (\%) $\uparrow$}} & \textbf{\shortstack{mAP$_{\mathrm{M}}$\\(\%) $\uparrow$}} & \textbf{\shortstack{mAP$_{9}$\\(\%) $\uparrow$}} & \textbf{\shortstack{Standalone\\FPS $\uparrow$}} \\
\midrule
TriBand-BEV~\cite{khoshkdahan2026triband} & 3 & \textbf{6.76} & 74.10 & 38.05 & 21.12 & 44.42 & 46.55 & \textbf{188.43} \\
Strata-BEV8  & 8 & \underline{18.02} & \underline{78.35} & 38.48 & \textbf{25.29} & 47.37 & 48.95 & 165.02 \\
Binary35~\cite{yang2018pixor} & 35 & 78.85 & 75.24 & \underline{44.36} & 23.20 & \underline{47.60} & \underline{49.08} & \underline{169.97} \\
\textbf{Strata-BEV14 (Ours)} & \textbf{14} & 31.54 & \textbf{79.97} & \textbf{45.36} & \underline{25.10} & \textbf{50.14} & \textbf{52.00} & 169.01 \\
\bottomrule
\end{tabular*}
\par\vspace{2pt}
{\footnotesize \textit{Note.} All variants share the identical MobilePIXORNeXt-S backbone, F24 SumFPN neck, single-head H16, standard augmentation, and baseline Smooth-L1 loss. Standalone inference speed is measured in FP32 on an NVIDIA Tesla T4 GPU under batch size 1 ($800 \times 704$ input grid). TriBand-BEV~\cite{khoshkdahan2026triband} provides the fastest forward execution ($188.43\,\text{FPS}$) due to its compact 3-channel input ($6.76\,\text{MB}$), but suffers significant accuracy degradation ($44.42\%$ Moderate mAP) due to the lack of continuous elevation moments and horizontal subcell centroid offsets. Strata-BEV14 achieves the highest accuracy ($50.14\%$ Moderate mAP and $52.00\%$ $\text{mAP}_9$) while maintaining real-time throughput ($169.01\,\text{FPS}$) and reducing H2D memory payload by $60.0\%$ relative to Binary35 ($31.54\,\text{MB}$ vs. $78.85\,\text{MB}$).}
\end{table*}


% SOURCE: tables/loss.tex
\begin{table*}[t]
\centering\small
\caption{Geometric supervision (Baseline Smooth-L1 vs. OGA), localization quality prediction (with and without an IQA Head), and head-free Q-OGA on KITTI validation at Moderate difficulty ($R_{40}$). IQA models rank NMS candidates using $S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}} \hat{q}}$. Best values are in \textbf{bold}, and second-best are \underline{underlined}.}
\label{tab:loss}
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{0.30\textwidth}ccrrrr@{}}
\toprule
\textbf{Training Objective} & \textbf{\shortstack{IQA\\Head}} & \textbf{\shortstack{Params\\(M) $\downarrow$}} & \textbf{\shortstack{Car AP$_{\mathrm{M}}$\\(\%) $\uparrow$}} & \textbf{\shortstack{Pedestrian\\AP$_{\mathrm{M}}$ (\%) $\uparrow$}} & \textbf{\shortstack{Cyclist\\AP$_{\mathrm{M}}$ (\%) $\uparrow$}} & \textbf{\shortstack{mAP$_{\mathrm{M}}$\\(\%) $\uparrow$}} \\
\midrule
Baseline Smooth-L1 & No  & \textbf{0.735} & 83.92 & 55.48 & 62.74 & 67.38 \\
OGA (MGIoU + $\pi$-Corner) & No  & \textbf{0.735} & 84.09 & 57.93 & 64.88 & 68.97 \\
Baseline Smooth-L1 & Yes & 0.754 & 84.13 & \underline{59.45} & \textbf{67.34} & 70.31 \\
\textbf{OGA (MGIoU + $\pi$-Corner)} & \textbf{Yes} & 0.754 & \textbf{84.87} & \textbf{59.63} & \underline{67.24} & \textbf{70.58} \\
\midrule
Q-OGA (Quality-Coupled Focal Loss) & No & \textbf{0.735} & \underline{84.35} & 58.12 & 65.13 & 69.20 \\
\bottomrule
\end{tabular*}
\par\vspace{2pt}
{\footnotesize \textit{Note.} All variants share the identical MobilePIXORNeXt backbone with standardized stage depths $(3, 4, 2)$ across all three stages, Strata-BEV14 input, Grouped H32 heads, F32 SG-FPN neck, and PC-GTA augmentation ($0.754\,\text{M}$ parameters with IQA, $0.735\,\text{M}$ without IQA). Embedding parameter-free Block-SimAM across all nine blocks produces the final configuration A10 without adding any learned parameters ($753{,}569$ parameters total). When IQA is enabled, candidate boxes are ranked during NMS via composite score $S_{\mathrm{det}} = \sqrt{P_{\mathrm{cls}} \cdot \hat{q}}$. Q-OGA integrates quality directly into classification supervision without dedicating a separate head, saving $18{,}626$ parameters ($18.63\,\text{K}$) compared with explicit IQA models.}
\end{table*}


% SOURCE: tables/seed_robustness.tex
\begin{table*}[t]
\centering\small
\caption{Multi-seed statistical robustness and variance comparison between baseline MobilePIXOR and the proposed MobilePIXORNeXt on the KITTI validation benchmark across three random seeds ($42, 43, 44$) under a 100-epoch training schedule. Detections are evaluated under the unified BEV evaluation protocol with 40-point interpolation ($R_{40}$). Performance is reported across individual seeds alongside the sample mean and standard deviation ($\text{Mean} \pm \text{Std}$, \%). The best mean and lowest variance values are highlighted in \textbf{bold}.}
\label{tab:seed_robustness}
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{@{}lrrrrrrrrrrr@{}}
\toprule
& \multicolumn{5}{c}{\textbf{Baseline MobilePIXOR~\cite{nguyen2024indoor}}} & \multicolumn{5}{c}{\textbf{MobilePIXORNeXt (Ours)}} & \\
\cmidrule(lr){2-6} \cmidrule(lr){7-11}
\textbf{Category / Metric} & \textbf{S42} & \textbf{S43} & \textbf{S44} & \textbf{Mean} & \textbf{Std} & \textbf{S42} & \textbf{S43} & \textbf{S44} & \textbf{Mean} & \textbf{Std} & \textbf{Gain ($\Delta$)} \\
\midrule
Car AP$_{\mathrm{M}}$ (\%) & 75.52 & 77.17 & 78.83 & 77.17 & $\pm 1.66$ & 86.90 & 86.33 & 85.42 & \textbf{86.22} & $\mathbf{\pm 0.75}$ & \textit{+9.05} \\
Pedestrian AP$_{\mathrm{M}}$ (\%) & 28.16 & 40.45 & 37.91 & 35.51 & $\pm 6.49$ & 60.78 & 61.94 & 60.06 & \textbf{60.93} & $\mathbf{\pm 0.95}$ & \textit{\textbf{+25.42}} \\
Cyclist AP$_{\mathrm{M}}$ (\%) & 20.53 & 34.17 & 32.18 & 28.96 & $\pm 7.37$ & 66.81 & 65.96 & 64.26 & \textbf{65.68} & $\mathbf{\pm 1.30}$ & \textit{\textbf{+36.72}} \\
\midrule
\textbf{Moderate mAP (\%)} & 41.40 & 50.60 & 49.64 & 47.21 & $\pm 5.06$ & 71.50 & 71.41 & 69.91 & \textbf{70.94} & $\mathbf{\pm 0.89}$ & \textit{\textbf{+23.73}} \\
\textbf{Overall mAP$_{9}$ (\%)} & 42.27 & 52.12 & 51.25 & 48.55 & $\pm 5.45$ & 73.45 & 73.35 & 72.22 & \textbf{73.01} & $\mathbf{\pm 0.68}$ & \textit{\textbf{+24.46}} \\
\bottomrule
\end{tabular}
\end{table*}


% SOURCE: tables/setup.tex
\begin{table}[t]
\centering\small
\caption{Experimental settings, architecture hyperparameters, and training configuration for MobilePIXORNeXt. Training uses a deterministic single-seed setup ($\text{seed} = 42$) with PyTorch 2.11.0 and CUDA 13.0. Timing uses a dedicated NVIDIA Tesla T4 GPU (16\,GB GDDR6 VRAM), batch size 1, and FP32 precision.}
\label{tab:setup}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{.38\columnwidth}>{\raggedright\arraybackslash}p{.56\columnwidth}@{}}
\toprule
\textbf{Configuration Element} & \textbf{Specification / Value} \\
\midrule
LiDAR Input Representation & Strata-BEV14 (14 channels; NumPy/Numba CPU) \\
Spatial ROI bounds ($X, Y, Z$) & $[0.00, 70.40]\,\text{m} \times [-40.00, 40.00]\,\text{m} \times [-2.50, 1.00]\,\text{m}$ \\
Grid resolution ($\Delta_x, \Delta_y, \Delta_z$) & $0.10\,\text{m} \times 0.10\,\text{m} \times 0.10\,\text{m}$ \\
Input / Output Grid Dimensions & $800 \times 704$ / $200 \times 176$ ($s_{\mathrm{out}} = 4$) \\
Detector Parameters & 753,569 deployable (753,581 trainable with 12 T-SBUW logits); 9.34\,MB FP32 \\
Backbone Stages ($C_2, C_3, C_4$) & Widths: (48, 96, 128); Depths: (3, 4, 2) \\
Inverted Bottleneck Expansion & $e = 2.50$; $7 \times 7$ Depthwise Convolution \\
Attention Mechanism & Parameter-free Block-SimAM (9 residual blocks) \\
Multiscale Neck & Scale-Gated FPN (SG-FPN, $F_{32}$ fusion width) \\
Prediction Towers & GroupedHeader: Car ($K_1=1$), Ped/Cyc ($K_2=2$) \\
Loss Formulation & OGA (MGIoU + $\pi$-Corner) + IQA + T-SBUW \\
Data Augmentation & Physics-Consistent GT (PC-GTA) + Global \\
Optimizer / Base Learning Rate & AdamW (fused) / $\eta = 7.00 \times 10^{-4}$ \\
Minimum LR / Training Schedule & $1.00 \times 10^{-6}$ / 100 epochs: 10 warmup + 90 cosine \\
Final Reported Checkpoint & Converged deployment checkpoint evaluated at epoch 100 \\
Weight Decay & $\lambda = 1.00 \times 10^{-3}$ (2D+ tensors only; biases/norm: 0.00) \\
Gradient Clipping Norm & Maximum $L_2$ norm = 10.00 \\
Batch Size (Train / Val) & 16 / 32 \\
Numerical Precision & Automatic Mixed Precision (AMP / FP32) \\
Hardware Environment & NVIDIA Tesla T4 GPU (16\,GB GDDR6, 98.44\,MiB peak) \\
\bottomrule
\end{tabular}
\end{table}
