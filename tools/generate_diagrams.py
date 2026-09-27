#!/usr/bin/env python3
"""Script to generate all architecture diagrams for the LiDAR detection codebase."""

import os
import subprocess
from pathlib import Path

DIAGRAMS_DIR = Path("/home/duyennh/AI_projects/research_lidar/Lidar/diagrams")
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

DIAGRAMS = {}

# ==============================================================================
# 1. CODEBASE OVERVIEW
# ==============================================================================
DIAGRAMS["01_codebase_overview"] = """digraph CodebaseOverview {
    rankdir=TB;
    bgcolor="#0d1117";
    compound=true;
    newrank=true;
    nodesep=0.4;
    ranksep=0.5;

    node [shape=box, style="filled,rounded", fontname="DejaVu Sans,Arial", fontsize=11, fontcolor="#ffffff", margin="0.2,0.12"];
    edge [color="#58a6ff", penwidth=1.6, arrowsize=0.8, fontname="DejaVu Sans,Arial", fontsize=9, fontcolor="#8b949e"];

    subgraph cluster_dataset {
        label="1. Data Ingestion & Preprocessing (KittiDataset)";
        fontcolor="#58a6ff"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        raw_data [label="KITTI LiDAR Data\\n• Velodyne Point Cloud (.bin): (N, 4) [x, y, z, r]\\n• Calibration (calib) & 3D Bounding Boxes (label_2)", fillcolor="#1f242c", color="#388bfd"];
        filter_pts [label="Spatial Cropping & Filtering\\nx ∈ [0, 70.4]m | y ∈ [-40, 40]m | z ∈ [-2.5, 1.0]m", fillcolor="#21262d", color="#58a6ff"];
        geom_aug [label="Geometric Augmentation (p=0.5)\\n• Yaw Rotation: [-20°, +20°]\\n• Scaling: [0.95, 1.05]\\n• Gaussian Translation: σ=0.4m", fillcolor="#21262d", color="#58a6ff"];
        bev_enc [label="BEV Voxelization / Encoding\\n• RichBEV-8 (8 channels) OR Legacy35 (35 channels)\\n• Grid Resolution: 0.1m/pixel -> Shape: (C_in, 800, 704)", fillcolor="#1f242c", color="#388bfd", penwidth=2];
        tgt_gen [label="Target Generation (Stride 4: 200 x 176)\\n• Cls Heatmap (Gaussian radius per class)\\n• Continuous Regression: Offset (dx, dy), Size (log_w, log_l), Yaw (cos2θ, sin2θ)\\n• Active Object Mask (reg_mask)", fillcolor="#21262d", color="#f0883e"];

        raw_data -> filter_pts -> geom_aug -> bev_enc;
        filter_pts -> tgt_gen;
    }

    subgraph cluster_model {
        label="2. Neural Network Architecture (CustomModel)";
        fontcolor="#7ee787"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        backbone [label="Modular Backbone (Registry)\\nBEVNeXt (Flagship) / MobilePIXOR / MobilePIXOR-CoordAtt / Pixor\\nInput: (B, C_in, 800, 704) -> Downsampling & Scale-Gated Neck\\nOutput: Feature Map (B, 16, 200, 176) [Stride 4]", fillcolor="#1f242c", color="#7ee787", penwidth=2];

        subgraph cluster_header {
            label="Detection Header (4 Parallel Heads)";
            fontcolor="#f778ba"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";

            h_cls [label="Classification Head\\nConv 3x3 -> BN/SiLU -> Conv 1x1\\nOutput: (B, C_cls, 200, 176)", fillcolor="#21262d", color="#f778ba"];
            h_off [label="Offset Head\\nConv 3x3 -> BN/SiLU -> Conv 1x1\\nOutput: (B, 2, 200, 176) [dx, dy]", fillcolor="#21262d", color="#f778ba"];
            h_siz [label="Size Head\\nConv 3x3 -> BN/SiLU -> Conv 1x1\\nOutput: (B, 2, 200, 176) [log_w, log_l]", fillcolor="#21262d", color="#f778ba"];
            h_yaw [label="Yaw Head\\nConv 3x3 -> BN/SiLU -> Conv 1x1\\nOutput: (B, 2, 200, 176) [cos2θ, sin2θ]", fillcolor="#21262d", color="#f778ba"];
        }

        backbone -> h_cls;
        backbone -> h_off;
        backbone -> h_siz;
        backbone -> h_yaw;
    }

    subgraph cluster_train {
        label="3. Training & Loss Optimization Pipeline";
        fontcolor="#d2a8ff"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        loss_facade [label="LossFunction Façade (Registry)\\nStrategies: baseline | uwag | oga (Flagship)", fillcolor="#1f242c", color="#d2a8ff", penwidth=2];
        loss_tasks [label="Composite Objectives:\\n• Modified Focal Loss (cls)\\n• Smooth L1 / L1 (offset, size, yaw with reg_mask)\\n• Oriented Geometry Loss (MGIoU + π-Symmetric Corner Distance)\\n• Adaptive Balancing: T-SBUW / Homoscedastic Weighting", fillcolor="#21262d", color="#d2a8ff"];
        optimizer [label="Optimization Engine\\n• Optimizer: Adam / AdamW with Cosine Warmup\\n• Precision: Mixed Precision (BF16 / FP32)\\n• Gradient Accumulation & Minimum-Loss Checkpointing", fillcolor="#21262d", color="#d2a8ff"];

        loss_facade -> loss_tasks -> optimizer;
    }

    subgraph cluster_infer {
        label="4. Inference, Postprocessing & Evaluation";
        fontcolor="#ffa657"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        decode [label="Metric Box Decoding (postprocess.py)\\n• Sigmoid + Local Peak / Top-k Selection\\n• Metric Position: (grid_x + dx) * 0.4m\\n• Metric Dimensions: exp(log_w), exp(log_l)\\n• Yaw Angle: 0.5 * atan2(sin2θ, cos2θ)", fillcolor="#21262d", color="#ffa657"];
        nms [label="Rotated BEV NMS\\nPruning overlapping rotated bounding boxes\\nIoU threshold filtering", fillcolor="#21262d", color="#ffa657"];
        eval_export [label="Deployment & Benchmark\\n• KITTI BEV AP R40 (Car, Pedestrian, Cyclist)\\n• ONNX Graph Export & TensorRT Engine Builder", fillcolor="#1f242c", color="#ffa657", penwidth=2];

        decode -> nms -> eval_export;
    }

    // Connect Pipeline
    bev_enc -> backbone [penwidth=2.5, color="#58a6ff", label="Input Tensor"];
    
    // Connect to Training Loss
    h_cls -> loss_facade;
    h_off -> loss_facade;
    h_siz -> loss_facade;
    h_yaw -> loss_facade;
    tgt_gen -> loss_facade [label="Targets & Masks", color="#f0883e"];

    // Connect to Inference
    h_cls -> decode;
    h_off -> decode;
    h_siz -> decode;
    h_yaw -> decode;
}
"""

# ==============================================================================
# 2. BEV ENCODING DETAIL
# ==============================================================================
DIAGRAMS["02_bev_encoding_detail"] = """digraph BevEncodingDetail {
    rankdir=TB;
    bgcolor="#0d1117";
    compound=true;
    nodesep=0.5;
    ranksep=0.6;

    node [shape=box, style="filled,rounded", fontname="DejaVu Sans,Arial", fontsize=11, fontcolor="#ffffff", margin="0.2,0.12"];
    edge [color="#58a6ff", penwidth=1.6, arrowsize=0.8, fontname="DejaVu Sans,Arial", fontsize=9, fontcolor="#8b949e"];

    subgraph cluster_input {
        label="Input LiDAR Point Cloud";
        fontcolor="#58a6ff"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        pts_in [label="Raw LiDAR Point Cloud P = {(x_i, y_i, z_i, r_i)}\\nCoordinate frame: Velodyne LiDAR\\nPoint count N ~ 100,000 points / frame", fillcolor="#1f242c", color="#388bfd", penwidth=2];
        grid_spec [label="BEV Grid Geometry Parameters:\\nx_range: [0.0, 70.4] m  (resolution Δx = 0.1 m -> W = 704)\\ny_range: [-40.0, 40.0] m (resolution Δy = 0.1 m -> H = 800)\\nz_range: [-2.5, 1.0] m  (height span = 3.5 m)", fillcolor="#21262d", color="#58a6ff"];
        pts_in -> grid_spec;
    }

    subgraph cluster_legacy {
        label="Option A: Legacy Binary Slices (voxelize)";
        fontcolor="#f0883e"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        leg_voxel [label="3D Voxel Quantization\\nx_idx = ⌊(x - x_min) / Δx⌋ ∈ [0, 703]\\ny_idx = ⌊(y - y_min) / Δy⌋ ∈ [0, 799]\\nz_idx = ⌊(z - z_min) / Δz⌋ ∈ [0, 34] (Δz = 0.1m)", fillcolor="#21262d", color="#f0883e"];
        leg_binary [label="Binary Occupancy Binarization\\nVoxels[x_idx, y_idx, z_idx] = 1.0\\n(Only records presence, discards density & reflectance)", fillcolor="#21262d", color="#f0883e"];
        leg_swap [label="Axis Swap to Channel-First (PyTorch)\\nSwap axes: (W, H, Z) -> (Z, H, W)\\nTensor Shape: (35, 800, 704) float32\\nSize: ~78.8 MB per frame", fillcolor="#1f242c", color="#f0883e", penwidth=2];
        leg_eval [label="Trade-offs:\\n❌ 35 channels: Heavy VRAM & bandwidth overhead\\n❌ Loses point density and reflectance (intensity)\\n✔ Simple legacy baseline", fillcolor="#21262d", color="#8b949e"];

        leg_voxel -> leg_binary -> leg_swap -> leg_eval;
    }

    subgraph cluster_rich8 {
        label="Option B: Modern RichBEV-8 Encoding (encode_bev)";
        fontcolor="#7ee787"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        rich_norm [label="Coordinate & Attribute Normalization\\n• flat = y_idx * W + x_idx  (1D cell address)\\n• z_norm = clip((z - z_min)/(z_max - z_min), 0.0, 1.0)\\n• intensity = clip(r * intensity_scale, 0.0, 1.0)\\n• count = bincount(flat) (total points per vertical pillar)", fillcolor="#21262d", color="#7ee787"];

        subgraph cluster_channels {
            label="8 Dense Statistical BEV Channels";
            fontcolor="#58a6ff"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";

            ch_bands [label="Channels 0 - 2: 3 Height Band Occupancies\\n• Band = min(⌊z_norm * 3⌋, 2)\\n• Ch 0: Low Band [0, 1/3)\\n• Ch 1: Mid Band [1/3, 2/3)\\n• Ch 2: High Band [2/3, 1.0]", fillcolor="#21262d", color="#58a6ff"];
            ch_height [label="Channels 3 - 4: Metric Height Distribution\\n• Ch 3 (Max Z): np.maximum.at(output[3], flat, z_norm)\\n• Ch 4 (Mean Z): bincount(flat, weights=z_norm) / max(count, 1)", fillcolor="#21262d", color="#58a6ff"];
            ch_intensity [label="Channels 5 - 6: Reflectance Distribution\\n• Ch 5 (Max I): np.maximum.at(output[5], flat, intensity)\\n• Ch 6 (Mean I): bincount(flat, weights=intensity) / max(count, 1)", fillcolor="#21262d", color="#58a6ff"];
            ch_density [label="Channel 7: Log-Normalized Pillar Density\\n• Ch 7 (Density): min(1.0, ln(1 + count) / ln(1 + density_norm))\\n(Default density_norm = 32.0)", fillcolor="#21262d", color="#58a6ff"];
        }

        rich_out [label="Packed RichBEV-8 Tensor\\nReshape & Transpose: (8, 800, 704) float32\\nSize: ~18.0 MB per frame (77.1% Reduction!)", fillcolor="#1f242c", color="#7ee787", penwidth=2];
        rich_eval [label="Trade-offs:\\n✔ 77.1% memory & PCIe transfer savings!\\n✔ Preserves multi-slice vertical structure (3 bands)\\n✔ Encodes metric height, surface reflectance & density", fillcolor="#21262d", color="#3fb950"];

        rich_norm -> ch_bands;
        rich_norm -> ch_height;
        rich_norm -> ch_intensity;
        rich_norm -> ch_density;

        ch_bands -> rich_out;
        ch_height -> rich_out;
        ch_intensity -> rich_out;
        ch_density -> rich_out;
        rich_out -> rich_eval;
    }

    grid_spec -> leg_voxel [label="Legacy Flow", color="#f0883e"];
    grid_spec -> rich_norm [label="RichBEV Flow", color="#7ee787", penwidth=2];
}
"""

# ==============================================================================
# 3. BACKBONE ARCHITECTURE
# ==============================================================================
DIAGRAMS["03_backbone_architecture"] = """digraph BackboneArchitecture {
    rankdir=TB;
    bgcolor="#0d1117";
    compound=true;
    nodesep=0.45;
    ranksep=0.5;

    node [shape=box, style="filled,rounded", fontname="DejaVu Sans,Arial", fontsize=11, fontcolor="#ffffff", margin="0.2,0.1"];
    edge [color="#58a6ff", penwidth=1.6, arrowsize=0.8, fontname="DejaVu Sans,Arial", fontsize=9, fontcolor="#8b949e"];

    subgraph cluster_bevnext {
        label="BEVNeXt Backbone (< 2.0M Params, Stride 4 Output)";
        fontcolor="#58a6ff"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        in_bev [label="BEV Input Tensor\\nShape: (B, C_in, 800, 704)\\nC_in = 8 (RichBEV) or 35 (Legacy)", fillcolor="#1f242c", color="#388bfd", penwidth=2];

        subgraph cluster_stem {
            label="Stage 1: Fast Spatial Stem (Stride 2)";
            fontcolor="#58a6ff"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";
            stem1 [label="Conv 3x3 (s=2, p=1) + BN + SiLU\\nC_in -> 32 | (B, 32, 400, 352)", fillcolor="#21262d", color="#58a6ff"];
            stem2 [label="Conv 3x3 (s=1, p=1) + BN + SiLU\\n32 -> 32 | (B, 32, 400, 352)", fillcolor="#21262d", color="#58a6ff"];
            stem1 -> stem2;
        }

        subgraph cluster_stage2 {
            label="Stage 2: Low-Level Metric Geometry C2 (Stride 4)";
            fontcolor="#f0883e"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";
            d2 [label="DownsampleBlock (Conv 3x3, s=2) + BN + SiLU\\n32 -> 48 | (B, 48, 200, 176)", fillcolor="#21262d", color="#f0883e"];
            b2 [label="2x BEVNeXtBlock (48 ch, exp=2.5)\\n7x7 DW-Conv (0.7m Receptive Field) + LayerScale\\nOutput C2: (B, 48, 200, 176)", fillcolor="#1f242c", color="#f0883e"];
            d2 -> b2;
        }

        subgraph cluster_stage3 {
            label="Stage 3: Core Semantic & Geometry C3/C4 (Stride 8)";
            fontcolor="#7ee787"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";
            d3 [label="DownsampleBlock (Conv 3x3, s=2) + BN + SiLU\\n48 -> 96 | (B, 96, 100, 88)", fillcolor="#21262d", color="#7ee787"];
            b3 [label="4x BEVNeXtBlock (96 ch, exp=2.5)\\nLarge metric receptive field spatial aggregation\\nFeature Map C3: (B, 96, 100, 88)", fillcolor="#1f242c", color="#7ee787"];
            attn [label="LiteMLARefinement (Linear Multi-Scale Attention)\\n• Multi-scale QKV (5x5 DW + Native)\\n• FP32 Linear Kernel V(K^T Q)/(1^T K Q + eps)\\n• LayerScale (init=0.01) -> Refined C4: (B, 96, 100, 88)", fillcolor="#238636", color="#3fb950", penwidth=2];
            d3 -> b3 -> attn;
        }

        subgraph cluster_stage4 {
            label="Stage 4: High-Level Context C5 (Stride 16)";
            fontcolor="#d2a8ff"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";
            d4 [label="DownsampleBlock (Conv 3x3, s=2) + BN + SiLU\\n96 -> 128 | (B, 128, 50, 44)", fillcolor="#21262d", color="#d2a8ff"];
            b4 [label="2x BEVNeXtBlock (128 ch, exp=2.5)\\nPure Convolution (No Attention Interference)\\nOutput C5: (B, 128, 50, 44)", fillcolor="#1f242c", color="#d2a8ff"];
            d4 -> b4;
        }

        subgraph cluster_fpn {
            label="Bilinear Scale-Gated FPN Neck (Top-Down Fusion)";
            fontcolor="#79c0ff"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";

            lat5 [label="Lateral C5: 1x1 Conv\\n128 -> 48 | (B, 48, 50, 44)", fillcolor="#21262d", color="#79c0ff"];
            lat4 [label="Lateral C4: 1x1 Conv\\n96 -> 48 | (B, 48, 100, 88)", fillcolor="#21262d", color="#79c0ff"];
            lat3 [label="Lateral C3: 1x1 Conv\\n48 -> 24 | (B, 24, 200, 176)", fillcolor="#21262d", color="#79c0ff"];

            u4 [label="Bilinear Upsample 2x + Refine 3x3 DW\\nShape: (B, 48, 100, 88) [U4]", fillcolor="#21262d", color="#58a6ff"];
            fuse4 [label="Scale-Gated Fusion (P4)\\nGate4 = 2 * sigmoid(Conv3x3(L4 + U4))\\nP4 = U4 + Gate4 * L4\\nShape: (B, 48, 100, 88)", fillcolor="#1f242c", color="#58a6ff", penwidth=2];

            u3 [label="Bilinear Upsample 2x + Proj 3x3 (48 -> 24)\\nShape: (B, 24, 200, 176) [U3]", fillcolor="#21262d", color="#58a6ff"];
            fuse3 [label="Scale-Gated Fusion (P3)\\nGate3 = 2 * sigmoid(Conv3x3(L3 + U3))\\nP3 = U3 + Gate3 * L3\\nShape: (B, 24, 200, 176)", fillcolor="#1f242c", color="#58a6ff", penwidth=2];

            out_proj [label="Final Output Projection\\nConv 3x3 + BN + SiLU: 24 -> 16 channels\\nOutput Feature: (B, 16, 200, 176)", fillcolor="#1f242c", color="#388bfd", penwidth=2];

            lat5 -> u4 -> fuse4;
            lat4 -> fuse4;
            fuse4 -> u3 -> fuse3;
            lat3 -> fuse3;
            fuse3 -> out_proj;
        }

        in_bev -> stem1;
        stem2 -> d2;
        b2 -> d3;
        attn -> d4;

        b4 -> lat5;
        attn -> lat4;
        b2 -> lat3;
    }

    subgraph cluster_mobilepixor {
        label="Alternative Backbone: MobilePIXOR with Macro-CoordAtt";
        fontcolor="#ffa657"; fontsize=12; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        mp_stem [label="Stem: 2x Conv 3x3 (stride 1) + BN + ReLU (32 ch)", fillcolor="#21262d", color="#ffa657"];
        mp_blocks [label="Bottom-Up InvertedResidual Blocks:\\n• Block 2: 1 block, t=1, 24 ch, stride 2 (H/2, W/2)\\n• Block 3: 3 blocks, t=6, 32 ch, stride 2 (H/4, W/4)\\n• Block 4: 4 blocks, t=6, 64 ch, stride 2 (H/8, W/8)\\n• Block 5: 3 blocks, t=6, 96 ch, stride 2 (H/16, W/16)", fillcolor="#21262d", color="#ffa657"];
        mp_c5_attn [label="Macro CoordAtt at Apex (c5)\\nCoordAtt(96, 96, reduction=32)\\nRefines global spatial coordinate attention", fillcolor="#1f242c", color="#f778ba", penwidth=2];
        mp_fpn [label="Deconv FPN (ConvTranspose2d)\\n• lat_c5 (96->64) + deconv1 -> p5 (32 ch)\\n• lat_c4 (64->32) + deconv2 -> p4 (16 ch)\\n• Optional Scale-Gating on top-down path", fillcolor="#21262d", color="#ffa657"];

        mp_stem -> mp_blocks -> mp_c5_attn -> mp_fpn;
    }
}
"""

# ==============================================================================
# 4. LOSS SYSTEM
# ==============================================================================
DIAGRAMS["04_loss_system"] = """digraph LossSystem {
    rankdir=TB;
    bgcolor="#0d1117";
    compound=true;
    nodesep=0.5;
    ranksep=0.6;

    node [shape=box, style="filled,rounded", fontname="DejaVu Sans,Arial", fontsize=11, fontcolor="#ffffff", margin="0.2,0.12"];
    edge [color="#58a6ff", penwidth=1.6, arrowsize=0.8, fontname="DejaVu Sans,Arial", fontsize=9, fontcolor="#8b949e"];

    subgraph cluster_inputs {
        label="Loss Inputs: Predictions & Ground Truth Targets";
        fontcolor="#58a6ff"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        preds [label="Header Predictions (B, C, 200, 176):\\n• cls: Class logits (B, C_cls, 200, 176)\\n• offset: Sub-pixel grid shifts (dx, dy)\\n• size: Log dimensions (log_w, log_l)\\n• yaw: Doubled-angle vector (cos2θ, sin2θ)", fillcolor="#1f242c", color="#388bfd"];
        targets [label="Ground Truth Targets (B, C, 200, 176):\\n• cls: Adaptive Gaussian heatmaps\\n• offset, size, yaw: Metric box targets\\n• reg_mask: Boolean foreground mask", fillcolor="#1f242c", color="#f0883e"];
    }

    subgraph cluster_facade {
        label="LossFunction Façade & Strategy Registry";
        fontcolor="#d2a8ff"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        loss_router [label="LossFunction (Modular Strategy Selector)\\nConfig: 'baseline' | 'uwag' | 'oga'", fillcolor="#1f242c", color="#d2a8ff", penwidth=2];
    }

    subgraph cluster_oga {
        label="SOTA Strategy: OGA (Oriented Geometric Alignment & Adaptive Loss)";
        fontcolor="#7ee787"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        subgraph cluster_sublosses {
            label="5 Supervised Task Objectives";
            fontcolor="#58a6ff"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";

            l_cls [label="1. Modified Focal Loss (cls)\\nHeatmap penalty with Gaussian focal decay\\nα = 2, β = 4", fillcolor="#21262d", color="#58a6ff"];
            l_off [label="2. Smooth L1 Loss (offset)\\nMasked center alignment: reg_mask * SmoothL1(dx, dy)", fillcolor="#21262d", color="#58a6ff"];
            l_siz [label="3. Smooth L1 Loss (size)\\nMasked log-dimension alignment: reg_mask * SmoothL1(log_w, log_l)", fillcolor="#21262d", color="#58a6ff"];
            l_yaw [label="4. Smooth L1 Loss (yaw)\\nMasked doubled-angle alignment: reg_mask * SmoothL1(cos2θ, sin2θ)", fillcolor="#21262d", color="#58a6ff"];

            subgraph cluster_geo {
                label="5. OrientedGeometryLoss (Composite L_geo = L_MGIoU + β * L_corner)";
                fontcolor="#f778ba"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#161b22";

                geo_corner [label="Scale-Normalized π-Symmetric Corner Distance\\n• Computes 4 BEV corners from (offset, size, yaw)\\n• Evaluates min(||C - C_gt||, ||C - C_gt^π||) (180° invariance)\\n• Normalizes by target diagonal: √(W_gt² + L_gt²)", fillcolor="#21262d", color="#f778ba"];
                geo_mgiou [label="Multi-Axis Projection GIoU (MGIoU)\\n• Projects 4 corners onto 4 normal axes (2 pred + 2 target)\\n• Computes 1D GIoU along each axis: (I / U) - (Hull - U)/Hull\\n• Specialized 2D rotated BEV overlap loss (AAAI 2026)", fillcolor="#21262d", color="#f778ba"];
                geo_corner -> geo_mgiou [style=invis];
            }
        }

        subgraph cluster_tsbuw {
            label="T-SBUW: Temperature-Softmax Bounded Uncertainty Weighting";
            fontcolor="#ffa657"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";

            tsbuw_scale [label="Smooth Tanh Soft-Bounding\\ns_bounded = B * tanh(s_i / B)  (Bound B = 3.0)", fillcolor="#21262d", color="#ffa657"];
            tsbuw_weight [label="Conserved Gradient Weights (Softmax with Temp τ=2.0)\\nw_i = M * exp(s_bounded_i / τ) / ∑_j exp(s_bounded_j / τ)\\nGuarantee: ∑ w_i == M (Total gradient budget conserved!)", fillcolor="#21262d", color="#ffa657"];
            tsbuw_ema [label="Running Scale Normalization (EMA momentum 0.99)\\nL_calibrated_i = L_i / EMA(L_i)\\nPrevents simplex collapse across disparate loss scales", fillcolor="#21262d", color="#ffa657"];
            tsbuw_total [label="Conserved Composite Total Loss\\nL_total = (1/M * ∑ EMA(L_i)) * ∑ (w_i * L_calibrated_i)", fillcolor="#1f242c", color="#ffa657", penwidth=2];

            tsbuw_scale -> tsbuw_weight -> tsbuw_ema -> tsbuw_total;
        }

        l_cls -> tsbuw_total;
        l_off -> tsbuw_total;
        l_siz -> tsbuw_total;
        l_yaw -> tsbuw_total;
        geo_corner -> tsbuw_total;
        geo_mgiou -> tsbuw_total;
    }

    subgraph cluster_others {
        label="Alternative Strategies";
        fontcolor="#8b949e"; fontsize=12; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        strat_baseline [label="Baseline Strategy:\\nL_cls(Focal) + L1(off) + L1(size) + L1(yaw)\\nSimple unweighted sum", fillcolor="#21262d", color="#8b949e"];
        strat_uwag [label="UWAG Strategy:\\nHomoscedastic log-scales: ∑ (e^(-s_i) * L_i + s_i)\\n+ Yaw-aware Footprint IoU: IoU_bev * 0.5 * (1 + cos2Δθ)", fillcolor="#21262d", color="#8b949e"];
    }

    preds -> loss_router;
    targets -> loss_router;
    loss_router -> l_cls [label="OGA Selected", color="#7ee787", penwidth=2];
    loss_router -> strat_baseline [color="#8b949e", style="dashed"];
    loss_router -> strat_uwag [color="#8b949e", style="dashed"];
}
"""

# ==============================================================================
# 5. BEVNEXT BLOCK
# ==============================================================================
DIAGRAMS["05_block_bevnext"] = """digraph BevNextBlock {
    rankdir=TB;
    bgcolor="#0d1117";
    nodesep=0.4;
    ranksep=0.45;

    node [shape=box, style="filled,rounded", fontname="DejaVu Sans,Arial", fontsize=11, fontcolor="#ffffff", margin="0.2,0.12"];
    edge [color="#58a6ff", penwidth=1.6, arrowsize=0.8, fontname="DejaVu Sans,Arial", fontsize=9, fontcolor="#8b949e"];

    subgraph cluster_block {
        label="BEVNeXtBlock Micro-Architecture (ConvNeXt / UIB Style)";
        fontcolor="#58a6ff"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        in_x [label="Input Feature Map X\\nShape: (B, C, H, W)", fillcolor="#1f242c", color="#388bfd", penwidth=2];

        dwconv [label="1. Large-Kernel Depthwise Convolution\\n• Conv2d(C, C, kernel_size=7, stride=1, padding=3, groups=C, bias=False)\\n• Physical BEV Receptive Field: 0.7m x 0.7m per block\\n• Captures spatial context without channel mixing", fillcolor="#21262d", color="#58a6ff"];
        norm1 [label="2. Normalization\\nBatchNorm2d(C)", fillcolor="#21262d", color="#58a6ff"];

        pw_exp [label="3. Inverted Bottleneck Pointwise Expansion\\n• Conv2d(C, hidden_dim, kernel_size=1, bias=False)\\n• Expansion ratio e = 2.5 -> hidden_dim = round(C * 2.5)\\n• Stage 2 (48 -> 120) | Stage 3 (96 -> 240) | Stage 4 (128 -> 320)", fillcolor="#21262d", color="#f0883e"];
        act [label="4. Non-Linear Activation\\n• SiLU(x) = x * sigmoid(x)\\n• Smooth, continuously differentiable; prevents dying ReLU on sparse LiDAR", fillcolor="#21262d", color="#f0883e"];

        pw_proj [label="5. Pointwise Linear Projection\\n• Conv2d(hidden_dim, C, kernel_size=1, bias=False)\\n• Projects expanded representation back to original channel dimension C", fillcolor="#21262d", color="#7ee787"];
        norm2 [label="6. Normalization\\nBatchNorm2d(C)", fillcolor="#21262d", color="#7ee787"];

        layer_scale [label="7. LayerScale Transformation\\n• Parameter γ ∈ ℝ^(1 x C x 1 x 1), initialized to 1e-5\\n• Scaled feature: γ * F(X)\\n• Enables deep network training stability without gradient explosion", fillcolor="#21262d", color="#d2a8ff"];

        add [label="8. Residual Addition\\nOutput Y = X + γ * F(X)\\nShape: (B, C, H, W)", fillcolor="#1f242c", color="#3fb950", penwidth=2];

        in_x -> dwconv -> norm1 -> pw_exp -> act -> pw_proj -> norm2 -> layer_scale -> add;
        in_x -> add [label="Identity Shortcut Connection", color="#3fb950", penwidth=2.0];
    }
}
"""

# ==============================================================================
# 6. LITEMLA REFINEMENT
# ==============================================================================
DIAGRAMS["06_block_litemla"] = """digraph LiteMLARefinement {
    rankdir=TB;
    bgcolor="#0d1117";
    nodesep=0.45;
    ranksep=0.45;

    node [shape=box, style="filled,rounded", fontname="DejaVu Sans,Arial", fontsize=11, fontcolor="#ffffff", margin="0.2,0.12"];
    edge [color="#58a6ff", penwidth=1.6, arrowsize=0.8, fontname="DejaVu Sans,Arial", fontsize=9, fontcolor="#8b949e"];

    subgraph cluster_litemla {
        label="LiteMLARefinement (Linear Multi-Scale Attention Hook at Stage 3)";
        fontcolor="#7ee787"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        x_in [label="Input Feature Map X\\nShape: (B, 96, 100, 88)", fillcolor="#1f242c", color="#388bfd", penwidth=2];

        qkv_conv [label="1. QKV Linear Projection\\n• Conv2d(96, 3 * 96, kernel_size=1, bias=False)\\n• Generates Query, Key, Value representations (Shape: B, 288, 100, 88)", fillcolor="#21262d", color="#58a6ff"];

        subgraph cluster_branches {
            label="2. Multi-Scale Context Aggregation";
            fontcolor="#58a6ff"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";

            br_native [label="Branch 0: Native QKV\\nIdentity (Shape: B, 288, H, W)", fillcolor="#21262d", color="#58a6ff"];
            br_scale5 [label="Branch 1: 5x5 DW Aggregation\\n• Conv2d 5x5 (padding=2, groups=288)\\n• Conv2d 1x1 (groups=3 * heads = 18)\\nCaptures local multi-scale neighborhood", fillcolor="#21262d", color="#58a6ff"];
        }

        pack_cat [label="Channel Concatenation of Scales\\nPacked QKV: [QKV_native, QKV_scale5]\\nShape: (B, 576, H, W)", fillcolor="#21262d", color="#f0883e"];

        subgraph cluster_linear_kernel {
            label="3. O(N) Linear Attention Core (FP32 Precision Protection)";
            fontcolor="#f778ba"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";

            reshape [label="Reshape & Split:\\n• heads = 6, head_dim = 16 (6 x 16 = 96)\\n• N = H * W = 8800 tokens\\n• Q, K, V ∈ ℝ^(B x heads x head_dim x N)\\n• Positive Mapping: Q = ReLU(Q), K = ReLU(K)", fillcolor="#21262d", color="#f778ba"];

            linear_math [label="Linear Attention Kernel (No N x N Attention Matrix!):\\n• Numerator = (V @ K^T) @ Q   [Shape: B, heads, head_dim, N]\\n• Denominator = (1^T @ K)^T @ Q + ε [Shape: B, heads, 1, N]\\n• Attended = Numerator / Denominator\\nComplexity: O(N * d²) instead of O(N² * d) -> 100x faster in BEV!", fillcolor="#1f242c", color="#f778ba", penwidth=2];

            reshape -> linear_math;
        }

        out_proj [label="4. Output Projection & Normalization\\n• Conv2d(96 * (1 + scales), 96, kernel_size=1, bias=False)\\n• BatchNorm2d(96)", fillcolor="#21262d", color="#7ee787"];
        lscale [label="5. LayerScale Adaptation\\nγ * Attended (init γ = 0.01 for identity start)", fillcolor="#21262d", color="#d2a8ff"];
        out_add [label="6. Residual Connection\\nOutput Y = X + γ * Attended(X)\\nShape: (B, 96, 100, 88)", fillcolor="#1f242c", color="#3fb950", penwidth=2];

        x_in -> qkv_conv;
        qkv_conv -> br_native;
        qkv_conv -> br_scale5;
        br_native -> pack_cat;
        br_scale5 -> pack_cat;
        pack_cat -> reshape;
        linear_math -> out_proj -> lscale -> out_add;
        x_in -> out_add [label="Identity Shortcut", color="#3fb950", penwidth=2.0];
    }
}
"""

# ==============================================================================
# 7. SCALE-GATED FPN BLOCK
# ==============================================================================
DIAGRAMS["07_block_scale_gated_fpn"] = """digraph ScaleGatedFpnBlock {
    rankdir=TB;
    bgcolor="#0d1117";
    nodesep=0.45;
    ranksep=0.45;

    node [shape=box, style="filled,rounded", fontname="DejaVu Sans,Arial", fontsize=11, fontcolor="#ffffff", margin="0.2,0.12"];
    edge [color="#58a6ff", penwidth=1.6, arrowsize=0.8, fontname="DejaVu Sans,Arial", fontsize=9, fontcolor="#8b949e"];

    subgraph cluster_sgfpn {
        label="Scale-Gated Bilinear FPN Fusion Unit (Level k)";
        fontcolor="#58a6ff"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        in_top [label="Higher-Level Top-Down Feature P_(k+1)\\nResolution: (H/2, W/2) | Channels: C_top", fillcolor="#1f242c", color="#388bfd"];
        in_lat [label="Lateral Backbone Feature L_k\\nResolution: (H, W) | Channels: C_lat", fillcolor="#1f242c", color="#f0883e"];

        subgraph cluster_upsample {
            label="Anti-Artifact Bilinear Upsampling";
            fontcolor="#58a6ff"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";

            bilinear [label="Bilinear Interpolation 2x\\nSmooth spatial expansion (Eliminates ConvTranspose deconvolution checkerboards!)", fillcolor="#21262d", color="#58a6ff"];
            refine [label="Refinement Stage:\\n• Level 4: Depthwise Conv 3x3 + BN + SiLU\\n• Level 3: Conv 3x3 Projection (48 -> 24) + BN + SiLU\\nOutput: Upsampled Feature U_k (Shape: C_lat, H, W)", fillcolor="#21262d", color="#58a6ff"];

            bilinear -> refine;
        }

        subgraph cluster_gate {
            label="Dynamic Scale-Gating Mechanism";
            fontcolor="#7ee787"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";

            sum_feat [label="Combined Feature Addition\\nS_k = L_k + U_k", fillcolor="#21262d", color="#7ee787"];
            dw_gate [label="Zero-Initialized Depthwise Conv 3x3\\n• Conv2d(C_lat, C_lat, kernel_size=3, padding=1, groups=C_lat, bias=True)\\n• Initialized: weights = 0, bias = 0", fillcolor="#21262d", color="#7ee787"];
            sig_scale [label="Bounded Gating Function\\nGate_k = 2 * sigmoid(DWConv(S_k))\\nAt initialization: Gate_k = 2 * 0.5 = 1.0 (Identical to standard sum!)", fillcolor="#1f242c", color="#7ee787", penwidth=2];

            sum_feat -> dw_gate -> sig_scale;
        }

        modulate [label="Gated Lateral Modulation\\nModulated = Gate_k ⊙ L_k\\nDynamically suppresses redundant or noisy lateral activations", fillcolor="#21262d", color="#d2a8ff"];
        fuse [label="Final Fused Feature Map\\nP_k = U_k + Gate_k ⊙ L_k\\nShape: (B, C_lat, H, W)", fillcolor="#1f242c", color="#3fb950", penwidth=2];

        in_top -> bilinear;
        refine -> sum_feat;
        in_lat -> sum_feat;

        sig_scale -> modulate;
        in_lat -> modulate;

        refine -> fuse;
        modulate -> fuse;
    }
}
"""

# ==============================================================================
# 8. COORDINATE ATTENTION BLOCK
# ==============================================================================
DIAGRAMS["08_block_coordatt"] = """digraph CoordinateAttentionBlock {
    rankdir=TB;
    bgcolor="#0d1117";
    nodesep=0.45;
    ranksep=0.45;

    node [shape=box, style="filled,rounded", fontname="DejaVu Sans,Arial", fontsize=11, fontcolor="#ffffff", margin="0.2,0.12"];
    edge [color="#58a6ff", penwidth=1.6, arrowsize=0.8, fontname="DejaVu Sans,Arial", fontsize=9, fontcolor="#8b949e"];

    subgraph cluster_coordatt {
        label="Coordinate Attention Block (Hou et al., CVPR 2021) at Apex c5";
        fontcolor="#f778ba"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        x_in [label="Input Feature Map X\\nShape: (B, C, H, W) [C = 96 at c5 apex]", fillcolor="#1f242c", color="#388bfd", penwidth=2];

        subgraph cluster_pooling {
            label="1. Directional Coordinate Information Embedding";
            fontcolor="#58a6ff"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";

            pool_h [label="Y-Axis 1D Global Pooling\\nAdaptiveAvgPool2d((None, 1))\\nEncodes vertical position -> (B, C, H, 1)", fillcolor="#21262d", color="#58a6ff"];
            pool_w [label="X-Axis 1D Global Pooling\\nAdaptiveAvgPool2d((1, None)) + Permute\\nEncodes horizontal position -> (B, C, W, 1)", fillcolor="#21262d", color="#58a6ff"];
        }

        concat [label="2. Spatial Concatenation along Dimension 2\\nY = cat([X_h, X_w], dim=2)\\nShape: (B, C, H + W, 1)", fillcolor="#21262d", color="#f0883e"];

        shared_conv [label="3. Shared Bottleneck Transformation\\n• Conv2d(C, mip, kernel_size=1, stride=1)\\n• mip = max(8, C // reduction) = max(8, 96//32) = 8\\n• BatchNorm2d(mip) + ReLU", fillcolor="#21262d", color="#f0883e"];

        subgraph cluster_split {
            label="4. Coordinate Attention Weight Generation";
            fontcolor="#7ee787"; fontsize=11; color="#484f58"; style="rounded"; bgcolor="#0d1117";

            split_h [label="Vertical Attention Branch (a_h)\\n• Split Y -> (B, mip, H, 1)\\n• Conv2d(mip, C, 1x1) + Sigmoid\\n• Output a_h ∈ ℝ^(B, C, H, 1)", fillcolor="#21262d", color="#7ee787"];
            split_w [label="Horizontal Attention Branch (a_w)\\n• Split Y -> (B, mip, W, 1) + Permute -> (B, mip, 1, W)\\n• Conv2d(mip, C, 1x1) + Sigmoid\\n• Output a_w ∈ ℝ^(B, C, 1, W)", fillcolor="#21262d", color="#7ee787"];
        }

        multiply [label="5. Spatial Attention Gating\\nOutput Y = X ⊙ a_h ⊙ a_w\\nBroadcast Multiplication: (B, C, H, W) * (B, C, H, 1) * (B, C, 1, W)\\nLocates precise Pedestrian/Cyclist coordinates without losing spatial alignment!", fillcolor="#1f242c", color="#3fb950", penwidth=2];

        x_in -> pool_h;
        x_in -> pool_w;
        pool_h -> concat;
        pool_w -> concat;
        concat -> shared_conv;
        shared_conv -> split_h;
        shared_conv -> split_w;
        split_h -> multiply;
        split_w -> multiply;
        x_in -> multiply [label="Feature Multiplication", color="#3fb950", penwidth=2.0];
    }
}
"""

# ==============================================================================
# 9. INVERTED RESIDUAL BLOCK
# ==============================================================================
DIAGRAMS["09_block_inverted_residual"] = """digraph InvertedResidualBlock {
    rankdir=TB;
    bgcolor="#0d1117";
    nodesep=0.45;
    ranksep=0.45;

    node [shape=box, style="filled,rounded", fontname="DejaVu Sans,Arial", fontsize=11, fontcolor="#ffffff", margin="0.2,0.12"];
    edge [color="#58a6ff", penwidth=1.6, arrowsize=0.8, fontname="DejaVu Sans,Arial", fontsize=9, fontcolor="#8b949e"];

    subgraph cluster_inv_res {
        label="InvertedResidual Block (MobileNetV2 Style in MobilePIXOR)";
        fontcolor="#ffa657"; fontsize=13; color="#30363d"; style="rounded,dashed"; bgcolor="#161b22";

        x_in [label="Input Feature Map X\\nShape: (B, C_in, H, W)", fillcolor="#1f242c", color="#388bfd", penwidth=2];

        pw_exp [label="1. Pointwise 1x1 Expansion (if expand_ratio t != 1)\\n• Conv2d(C_in, hidden_dim, kernel_size=1, bias=False)\\n• hidden_dim = round(C_in * t)  (typically t=6)\\n• BatchNorm2d(hidden_dim) + ReLU", fillcolor="#21262d", color="#f0883e"];

        dw_conv [label="2. Depthwise 3x3 Convolution\\n• Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=s, padding=1, groups=hidden_dim, bias=False)\\n• Spatial filtering (stride s=1 or s=2)\\n• BatchNorm2d(hidden_dim) + ReLU", fillcolor="#21262d", color="#58a6ff"];

        opt_attn [label="3. Optional Attention Hook\\nCoordAtt(hidden_dim, hidden_dim) (if use_attn=True)\\nDefault in main model: False (CoordAtt placed at macro-level c5)", fillcolor="#21262d", color="#8b949e", style="rounded,dashed"];

        pw_linear [label="4. Pointwise 1x1 Linear Bottleneck Projection\\n• Conv2d(hidden_dim, C_out, kernel_size=1, bias=False)\\n• BatchNorm2d(C_out)\\n• Linear (NO non-linear activation to preserve manifold)", fillcolor="#21262d", color="#7ee787"];

        res_add [label="5. Residual Addition (Only if stride==1 and C_in == C_out)\\nOutput Y = X + Conv(X)\\nShape: (B, C_out, H', W')", fillcolor="#1f242c", color="#3fb950", penwidth=2];

        x_in -> pw_exp -> dw_conv -> opt_attn -> pw_linear -> res_add;
        x_in -> res_add [label="Shortcut (if s=1 and C_in==C_out)", color="#3fb950", penwidth=2.0];
    }
}
"""

def main():
    print(f"Generating diagrams into {DIAGRAMS_DIR} ...")
    for name, dot_content in DIAGRAMS.items():
        dot_path = DIAGRAMS_DIR / f"{name}.dot"
        png_path = DIAGRAMS_DIR / f"{name}.png"
        
        with open(dot_path, "w") as f:
            f.write(dot_content.strip() + "\\n")
            
        print(f"-> Rendering {name}.png ...")
        cmd = ["dot", "-Tpng", "-Gdpi=220", str(dot_path), "-o", str(png_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error rendering {name}: {res.stderr}")
            raise RuntimeError(res.stderr)
        else:
            size_kb = os.path.getsize(png_path) / 1024
            print(f"   [OK] {name}.png ({size_kb:.1f} KB)")

    print("\nAll 9 diagrams successfully generated!")

if __name__ == "__main__":
    main()
