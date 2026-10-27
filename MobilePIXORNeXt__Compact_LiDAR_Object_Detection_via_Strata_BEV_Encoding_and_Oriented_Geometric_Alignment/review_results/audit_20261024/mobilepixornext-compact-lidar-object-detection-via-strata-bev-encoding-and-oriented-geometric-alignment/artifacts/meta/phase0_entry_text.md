fix-cm
=7
svjour3
Arabian Journal for Science and Engineering (AJSE)
inputenc
graphicx,microtype,hyperref
cuted,capt-of
amsmath,amssymb,amsfonts,booktabs,cite,array
placeins
[1]\,https://orcid.org/#10.7[rgb]0.65,0.81,0.22[iD]
sections/figure_helpers
document
1.5em
MobilePIXORNeXt: Compact LiDAR Object Detection via Strata-BEV Encoding and Oriented Geometric Alignment
 =0pt
 Duyen Huu Nguyen 
 Anh Duc Bui 
 Nhan Trong Phan 
 Hoang Ngoc Tran
D. H. Nguyen et al.
 Correspondence: Hoang Ngoc Tran (hoang2531992@gmail.com) \\
 FPT University (FPT), Can Tho, 94000, VietNam. \\
 duyennhce200017@gmail.com; anhbdce200052@gmail.com; nhanptce200090@gmail.com; hoang2531992@gmail.com
 
 
 
 
 
 
MobilePIXORNeXt
 
abstract
Accurate 3D object perception from Light Detection and Ranging (LiDAR) point clouds is an important challenge for autonomous vehicles, where volumetric and pillar networks frequently exceed edge computational budgets. In this paper, we present MobilePIXORNeXt, an ultra-compact, proposal-free detector targeting oriented bounding footprints on the Special Euclidean (SE) manifold for real-time collision avoidance and path planning. With 753,569 trainable parameters, the framework replaces heavy learned point encoders with Strata-BEV14, a deterministic Central Processing Unit (CPU) statistical rasterizer computing 14 feature channels in 20.94 milliseconds for combined raw LiDAR disk I/O and statistical rasterization. A single-branch ConvNeXt-style backbone with parameter-free 3D spatial-energy attention (Block-SimAM) extracts multi-scale features, while a Scale-Gated Feature Pyramid Network fuses representations for decoupled grouped prediction heads. The model is trained using Physics-Consistent Ground-Truth Augmentation and Oriented Geometric Alignment (OGA), balanced by 12-parameter Temperature-Softmax Bounded Uncertainty Weighting. On the KITTI validation split, MobilePIXORNeXt achieves 71.50\% Moderate mean Average Precision (mAP) and 73.45\% overall 9-cell mAP using a local loader-aligned evaluator with 40-point interpolation. Equivalence between this evaluator and the reference KITTI evaluator has not been established; therefore, published AP differences are not presented as direct superiority claims. The model uses 84.41\% fewer parameters than PointPillars and 85.85\% fewer than SECOND, and reaches 60.78\% local Pedestrian Moderate AP. We also examine Q-OGA as a head-free variant achieving 69.20\% Moderate mAP. On an NVIDIA Tesla T4 GPU, the end-to-end pipeline operates in 60.96 milliseconds (16.41 frames per second). These results demonstrate that deterministic statistical encoding coupled with oriented geometric alignment enables compact, real-time perception on edge vehicular platforms.
abstract
LiDAR object detection Bird's-eye view representation Compact convolutional neural networks Parameter-free attention Oriented bounding box regression
## Introduction
sec:introduction
sections/introduction
## Related Work
sec:related_work
sections/related_work
## Proposed Method
sec:method
sections/methodology
## Experiments
sec:experiments
sections/experiments
## Results and Discussion
sec:results
sections/results
## Conclusion
sec:conclusion
sections/conclusion
sections/declarations
spmpsci
references
document