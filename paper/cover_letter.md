Date: October 22, 2026

To: The Editor-in-Chief, Arabian Journal for Science and Engineering (AJSE), Springer Nature

Dear Editor-in-Chief,

We submit our original research manuscript entitled "MobilePIXORNeXt: Compact LiDAR Object Detection with Oriented Geometry and Localization Quality" for consideration in the Arabian Journal for Science and Engineering.

Three-dimensional LiDAR perception is fundamental for autonomous navigation, but mainstream detectors rely on heavy point encoders or 3D convolutions (4.0–63.8M parameters), exceeding edge budgets. On-road vehicular safety decisions operate predominantly on the Special Euclidean manifold $SE(2)$. These results demonstrate that deterministic statistical encoding coupled with oriented geometric alignment enables accurate, real-time perception on edge vehicular platforms.

The manuscript presents four evidence-backed contributions:

- Strata-BEV14 rasterizes 14 statistical BEV channels (vertical densities, elevation moments, subcell centroid offsets) in 20.94 ms on CPU without learned encoders.

- MobilePIXORNeXt-S integrates $7 \times 7$ depthwise convolutions with parameter-free Block-SimAM 3D energy attention and a Scale-Gated Feature Pyramid Network with decoupled grouped heads.

- Physics-Consistent Ground-Truth Augmentation (PC-GTA) prevents unphysical synthesis (+6.03 pp mAP), while Oriented Geometric Alignment (OGA) and IoU-Aware Quality Alignment (IQA) supervise rotated bounding box overlaps (+3.20 pp mAP).

- With 753,569 parameters (<0.76M), MobilePIXORNeXt achieves 71.46% Moderate mAP on KITTI, exceeding PointPillars by 1.33 pp with an 84.41% parameter reduction, running in 60.96 ms (16.41 FPS) on Tesla T4 GPU.

This work aligns with the journal's scope in intelligent transportation systems, embedded systems, and applied robotics, offering practical utility for real-time vehicular computing.

Statements and Declarations:

- Originality & Exclusivity: The manuscript represents original research and has not been published elsewhere; it is not under concurrent consideration by any other journal or conference.

- Authorship Approval: All authors have read and approved the final manuscript for submission.

- Competing Interests: The authors declare that they have no competing financial or non-financial interests.

- Data and Code Availability: The benchmark data are available from the public KITTI repository, and the source code will be made available upon publication.

- AI Assistance Disclosure: Generative AI assistance was employed solely for English copy-editing and grammar verification; all conceptual ideas, mathematical formulations, software development, and experimental analyses were performed entirely by the authors.

Sincerely,

Hoang Ngoc Tran, Ph.D. (Corresponding Author, hoang2531992@gmail.com)  
FPT University, Can Tho, Vietnam (on behalf of Duyen Huu Nguyen, Anh Duc Bui, and Nhan Trong Phan)  
