# MobilePixorNeXt Architectural Evolution Roadmap (v2)

## 1. Executive Summary

This roadmap defines the next generation of the **MobilePixorNeXt** 3D LiDAR Object Detection framework. Building upon the solid foundation of `MobilePixorNeXt + OGA Loss` (Baseline $M_0$), this project introduces four synergistic architectural improvements designed to push detection accuracy (KITTI 3D / BEV mAP) while strictly preserving real-time edge deployability (<2.0M parameters, >45 FPS on embedded GPUs):

```
+----------------------------------------------------------------------------------------------------+
|                                    MobilePixorNeXt v2 Roadmap                                     |
+----------------------------------------------------------------------------------------------------+
                                                  │
            ┌─────────────────────────────────────┴─────────────────────────────────────┐
            ▼                                                                           ▼
   [Backbone Improvements]                                                     [Neck & Head Improvements]
  ┌──────────────────────────────────────────┐                               ┌──────────────────────────────────────────┐
  │ Pillar 1: Structural Reparameterization │                               │ Pillar 3: Bi-directional SG-FPN          │
  │ (Rep-MobilePixorNeXt Block)              │                               │ (Bi-SGFPN Neck)                          │
  │ • Train: Multi-branch (7x7 + 3x3 + Id)   │                               │ • Bidirectional feature propagation      │
  │ • Infer: Fused single 7x7 conv           │                               │ • Learned scale-gating on both paths     │
  │ • 0 extra inference latency              │                               │ • Multi-scale metric resolution          │
  └──────────────────────────────────────────┘                               └──────────────────────────────────────────┘
            │                                                                           │
            ▼                                                                           ▼
  ┌──────────────────────────────────────────┐                               ┌──────────────────────────────────────────┐
  │ Pillar 2: Robust Multi-Scale LiteMLA     │                               │ Pillar 4: IoU-Aware Quality Header       │
  │ (Linear Attention Refinement)            │                               │ (IQA-Header & Joint NMS)                 │
  │ • Multi-scale kernels (3x3, 5x5)         │                               │ • 5th branch predicting 3D IoU           │
  │ • QK-LayerNorm numerical stability       │                               │ • Score alignment: S = P_cls^(1-a) * IoU^a│
  │ • BF16 mixed-precision safety            │                               │ • False-positive suppression in NMS      │
  └──────────────────────────────────────────┘                               └──────────────────────────────────────────┘
```

---

## 2. Document Index & Navigation

| Document | Component | Key Innovation | Inference Impact |
| :--- | :--- | :--- | :--- |
| [00_ABLATION_STUDY_PROTOCOL.md](./00_ABLATION_STUDY_PROTOCOL.md) | **Evaluation Suite** | End-to-end ablation protocol anchoring MobilePixorNeXt + OGA loss ($M_0$) | Controlled benchmarking protocol |
| [01_STRUCTURAL_REPARAMETERIZATION.md](./01_STRUCTURAL_REPARAMETERIZATION.md) | **Backbone (Stages 2-4)** | Multi-branch training collapsed into single $7\times 7$ DW-Conv | **Zero** latency/memory overhead at inference |
| [02_ROBUST_MULTISCALE_LITEMLA.md](./02_ROBUST_MULTISCALE_LITEMLA.md) | **Attention (C3)** | Multi-scale regional kernels $(3\times 3, 5\times 5)$ + QK-Norm | $+0.05$ ms latency, robust under BF16 |
| [03_BIDIRECTIONAL_SGFPN.md](./03_BIDIRECTIONAL_SGFPN.md) | **Neck (FPN)** | Top-down & bottom-up scale-gated pyramid fusion | $+0.4$ ms latency, $+85\text{k}$ parameters |
| [04_IOU_AWARE_QUALITY_HEADER.md](./04_IOU_AWARE_QUALITY_HEADER.md) | **Detection Header** | 3D IoU prediction branch + Joint NMS Scoring | $+0.08$ ms latency, fixes score-box misalignment |

---

## 3. Design Principles & Hard Constraints

1. **Strict Latency Budget**: Inference on an NVIDIA RTX 3060 Laptop GPU (batch size 1, FP16/BF16) must remain below **15 ms** (>65 FPS), and on Jetson AGX Orin below **25 ms** (>40 FPS).
2. **Compact Model Footprint**: Total model parameters after structural reparameterization must not exceed **2.0M** (target: $\approx 1.35\text{M}$ to $1.65\text{M}$).
3. **No External Heavy Dependencies**: All implementations use pure PyTorch 2.x and NumPy primitives without requiring exotic CUDA extensions (e.g. `spconv` or heavy mmcv).
4. **Numerical Precision Robustness**: Full compatibility with BF16, FP16, and TF32 mixed-precision training. Attention operations must include bounded normalization to prevent overflow/underflow.
5. **Ablation Traceability**: Every individual improvement can be toggled on/off independently via configuration flags (`configs/kitti/...json`) for unambiguous experimental validation.

---

## 4. Planned Progressive Milestones

```
   M0: Baseline MobilePixorNeXt + OGA Loss (Current Master Anchor)
    │
    ▼
   M1: M0 + Structural Reparameterization (Rep-MobilePixorNeXt)
    │
    ▼
   M2: M1 + Robust Multi-Scale LiteMLA Attention
    │
    ▼
   M3: M2 + Bi-directional Scale-Gated Neck (Bi-SGFPN)
    │
    ▼
   M4: M3 + IoU-Aware Quality Detection Header (IQA-Header)  ==> [MobilePixorNeXt v2 SOTA]
```
