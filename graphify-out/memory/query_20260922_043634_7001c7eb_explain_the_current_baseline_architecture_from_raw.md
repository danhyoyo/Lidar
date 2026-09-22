---
type: "query"
date: "2026-09-22T04:36:34.301310+00:00"
question: "Explain the current baseline architecture from raw LiDAR through preprocessing, augmentation, encoding, backbone, FPN, heads, loss, training, postprocessing, and determine whether it is clean of claims made in the senior thesis."
contributor: "graphify"
outcome: "useful"
source_nodes: ["Dataset", "MobilePixorBackBone", "LossFunction", "filter_pred()", "voxelize()", "CustomModel"]
---

# Q: Explain the current baseline architecture from raw LiDAR through preprocessing, augmentation, encoding, backbone, FPN, heads, loss, training, postprocessing, and determine whether it is clean of claims made in the senior thesis.

## Answer

Expanded from original query via graph vocabulary: architecture, baseline, mobilepixor, backbone, voxel, augmentation, gaussian, heatmap, fusion, head, loss, postprocess. The pure runnable control is configs/kitti/kitti_mobilepixor_baseline.json, not B0 or notebook A0. It uses Legacy35 binary occupancy, plain MobilePIXOR with inverted residual stages and sum FPN, four BEV heads, modified focal plus masked L1, no augmentation, no CoordAtt, and no UWAG. It is clean of the thesis-specific CoordAtt, augmentation, and UWAG claims, but it remains an IRBGHR-PIXOR-derived published baseline that must cite Nguyen et al. 2024 plus PIXOR and MobileNetV2. Notebook A0 and MobileBEV A1-A4 are not thesis-clean controls.

## Outcome

- Signal: useful

## Source Nodes

- Dataset
- MobilePixorBackBone
- LossFunction
- filter_pred()
- voxelize()
- CustomModel