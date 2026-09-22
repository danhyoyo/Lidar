# Graph Report - Lidar  (2026-09-22)

## Corpus Check
- Corpus is ~46,358 words - fits in a single context window. You may not need a graph.

## Summary
- 455 nodes · 866 edges · 24 communities (20 shown, 4 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 50 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Evaluation and Deployment
- Losses and Detection Heads
- Coordinate Attention Backbone
- Notebook and Reporting Tests
- Postprocessing and Model Tests
- Torch Utility Wrappers
- Model Assembly and Backbones
- KITTI Data Preparation
- Baseline MobilePIXOR Backbone
- Legacy Architecture Diagram
- Training Loop
- Dataset and Target Encoding
- Runtime Regression Tests
- Model Comparison Reports
- Gaussian Target Generation
- BEV Input Encoding
- Geometry Transforms
- Augmentation Orchestration
- Data Utility Tests
- MobileBEV Ablation Design
- Scaling Augmentation
- Pipeline Documentation
- Thesis-Derived Recipe
- KITTI Dataset Splits

## God Nodes (most connected - your core abstractions)
1. `Dataset` - 27 edges
2. `run_evaluation()` - 21 edges
3. `write_json()` - 17 edges
4. `main()` - 16 edges
5. `read_json()` - 11 edges
6. `build_model()` - 11 edges
7. `sha256()` - 10 edges
8. `main()` - 10 edges
9. `CustomModel` - 9 edges
10. `Sequential` - 9 edges

## Surprising Connections (you probably didn't know these)
- `check_targets()` --uses--> `Dataset`  [INFERRED]
  tests/test_mobile_bev.py → detector/core/datasets/dataset.py
- `DataUtilityTests` --uses--> `Dataset`  [INFERRED]
  tests/test_regressions.py → detector/core/datasets/dataset.py
- `run_evaluation()` --uses--> `Dataset`  [INFERRED]
  tools/kitti_training_pipeline/evaluate_kitti_bev.py → detector/core/datasets/dataset.py
- `main()` --uses--> `Dataset`  [INFERRED]
  tools/kitti_training_pipeline/train.py → detector/core/datasets/dataset.py
- `main()` --uses--> `LossFunction`  [INFERRED]
  tools/kitti_training_pipeline/train.py → detector/core/losses/loss_fn.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **MobileBEV Controlled Ablation Protocol** — docs_mobile_bev_lightweight_spec_ablation_matrix, docs_mobile_bev_lightweight_spec_richbev_8, docs_mobile_bev_lightweight_spec_sg_fpn [EXTRACTED 1.00]
- **Multi-Scale Backbone Hierarchy** — mermaid_diagram_c2, mermaid_diagram_c3, mermaid_diagram_c4, mermaid_diagram_c5 [EXTRACTED 1.00]
- **Supervised Training Loop** — mermaid_diagram_target_encoder, mermaid_diagram_dense_detection_heads, mermaid_diagram_loss, mermaid_diagram_adam_backpropagation [EXTRACTED 1.00]
- **Detection Inference Pipeline** — mermaid_diagram_dense_detection_heads, mermaid_diagram_decode_threshold_nms, mermaid_diagram_bev_3d_boxes_uncertainty [EXTRACTED 1.00]

## Communities (24 total, 4 thin omitted)

### Community 0 - "Evaluation and Deployment"
Cohesion: 0.07
Nodes (58): argparse, dataclasses, hashlib, os, pathlib, platform, subprocess, build_parser() (+50 more)

### Community 1 - "Losses and Detection Heads"
Cohesion: 0.08
Nodes (22): core_losses_focal_loss, core_losses_l1_loss, one_hot(), device, Tensor, r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor. Args:…, focal_loss(), FocalLoss (+14 more)

### Community 2 - "Coordinate Attention Backbone"
Cohesion: 0.09
Nodes (19): Conv2d, Conv2dNormActivation, conv3x3(), conv3x3_dw(), ConvNormActivation, CoordAtt, InvertedResidual, MobilePixorBackBone (+11 more)

### Community 3 - "Notebook and Reporting Tests"
Cohesion: 0.09
Nodes (24): core_datasets_dataset, core_datasets_utils_1_gaussian, core_datasets_utils_1_preprocess, csv, json, re, sys, tempfile (+16 more)

### Community 4 - "Postprocessing and Model Tests"
Cohesion: 0.10
Nodes (24): compute_iou(), convert_format(), _empty_detections(), filter_pred(), non_max_suppression(), :param array: an array of shape [# bboxs, 4, 2] :return: a…, Calculates IoU of the given box with the array of the given boxes. box: a…, Performs non-maximum suppression and returns indices of kept boxes. boxes: [N,… (+16 more)

### Community 5 - "Torch Utility Wrappers"
Cohesion: 0.09
Nodes (17): collections, conv3x3(), 3x3 convolution with padding, SCConv, change_default_args(), layer_wrapper(), __init__(), Empty (+9 more)

### Community 6 - "Model Assembly and Backbones"
Cohesion: 0.11
Nodes (14): core_models_backbones_mobilepixor, core_models_backbones_mobilepixor_coordinate_attention, core_models_backbones_pixor, core_models_backbones_rpn, core_models_heads_cnn, Bottleneck, conv3x3(), PixorBackBone (+6 more)

### Community 7 - "KITTI Data Preparation"
Cohesion: 0.20
Nodes (18): Counter, shutil, inside_roi(), load_ground_truth(), build_parser(), convert_labels(), generated_config(), main() (+10 more)

### Community 8 - "Baseline MobilePIXOR Backbone"
Cohesion: 0.16
Nodes (10): Conv2dNormActivation, conv3x3(), ConvNormActivation, InvertedResidual, MobilePixorBackBone, Module, Tensor, Configurable block used for Convolution2d-Normalization-Activation blocks.… (+2 more)

### Community 9 - "Legacy Architecture Diagram"
Cohesion: 0.13
Nodes (19): Adam and Backpropagation (Scheduler, BF16, Accumulation), BEV/3D Boxes and Uncertainty, BEV Encoder (Legacy35 or RichBEV-8/11), C2 (24 Channels, Stride 2), C3 (32 Channels, Stride 4), C4 (64 Channels, Stride 8), C4→C3 Fusion (Sum-FPN or SG-FPN Gate C3), C5 (96 Channels, Stride 16) (+11 more)

### Community 10 - "Training Loop"
Cohesion: 0.20
Nodes (17): datetime, no_grad, random, autocast_context(), build_parser(), checkpoint_payload(), loader_kwargs(), main() (+9 more)

### Community 11 - "Dataset and Target Encoding"
Cohesion: 0.18
Nodes (3): Dataset, :param i: the ith velodyne scan in the train/val set : return boxes of shape N:8, :param boxes: numpy array of shape N:8 :return: label map: <--- This is the…

### Community 12 - "Runtime Regression Tests"
Cohesion: 0.13
Nodes (5): RuntimeAndLossTests, device_timed(), device, PyTorchRunner, TensorRTRunner

### Community 13 - "Model Comparison Reports"
Cohesion: 0.22
Nodes (15): gc, time, flatten(), fmt(), main(), markdown_cell(), markdown_table(), nested() (+7 more)

### Community 14 - "Gaussian Target Generation"
Cohesion: 0.18
Nodes (10): get_points_in_a_rotated_box(), :param label: numpy array of shape [..., 2] of coordinates in label map space…, trasform_label2metric(), draw_heatmap_gaussian(), gaussian_2d(), gaussian_radius(), Get gaussian masked heatmap. Args: heatmap (torch.Tensor): Heatmap to be…, Get radius of gaussian. Args: det_size (tuple[torch.Tensor]): Size of the… (+2 more)

### Community 15 - "BEV Input Encoding"
Cohesion: 0.18
Nodes (9): encode_bev(), get_points_in_a_rotated_box(), _grid_shape(), :param label: numpy array of shape [..., 2] of coordinates in label map space…, :param label: numpy array of shape [..., 2] of coordinates in metric space…, Encode KITTI ``(x, y, z, intensity)`` points as legacy or RichBEV., transform_metric2label(), trasform_label2metric() (+1 more)

### Community 16 - "Geometry Transforms"
Cohesion: 0.24
Nodes (8): box_transform(), center_to_corner_box3d(), corner_to_center_box3d(), inverse_rigid_trans(), point_transform(), Random_Rotation, Inverse a rigid body transform matrix (3x4 as [R|t]) [R'|-R't; 0|1], :param labels: # (N', 7) h, w, l, x, y, z, r :return:

### Community 17 - "Augmentation Orchestration"
Cohesion: 0.18
Nodes (5): Compose, OneOf, Random_Translation, :param labels: # (N', 7) h, w, l, x, y, z, r :return:, object

### Community 19 - "MobileBEV Ablation Design"
Cohesion: 0.50
Nodes (5): A0-A4 Ablation Matrix, MobileBEV-Lite, RichBEV-8, SG-FPN, Minimum Validation Loss Checkpoint Rule

### Community 21 - "Pipeline Documentation"
Cohesion: 0.67
Nodes (3): Local Rotated BEV AP R40 Evaluation, MobilePIXOR Detector, KITTI Training Pipeline Guide

## Knowledge Gaps
- **8 isolated node(s):** `MobilePIXOR Detector`, `UWAG and Coordinate Attention Training Recipe`, `Geometric Augmentation`, `Local Rotated BEV AP R40 Evaluation`, `KITTI Training Split Manifest (5984 Frames)` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 138 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Dataset` connect `Dataset and Target Encoding` to `Evaluation and Deployment`, `Postprocessing and Model Tests`, `Training Loop`, `Gaussian Target Generation`, `Geometry Transforms`, `Augmentation Orchestration`, `Data Utility Tests`, `Scaling Augmentation`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `run_evaluation()` connect `Evaluation and Deployment` to `Postprocessing and Model Tests`, `KITTI Data Preparation`, `Dataset and Target Encoding`, `Runtime Regression Tests`, `Model Comparison Reports`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `CustomModel` connect `Model Assembly and Backbones` to `Baseline MobilePIXOR Backbone`, `Evaluation and Deployment`, `Coordinate Attention Backbone`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `Dataset` (e.g. with `OneOf` and `Random_Rotation`) actually correct?**
  _`Dataset` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `run_evaluation()` (e.g. with `.test_invalid_evaluation_options_fail_before_io()` and `Dataset`) actually correct?**
  _`run_evaluation()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `main()` (e.g. with `Dataset` and `LossFunction`) actually correct?**
  _`main()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `MobilePIXOR Detector`, `UWAG and Coordinate Attention Training Recipe`, `Geometric Augmentation` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._