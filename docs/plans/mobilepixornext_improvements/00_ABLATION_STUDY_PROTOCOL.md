# Ablation Study Protocol: MobilePixorNeXt v2

## 1. Overview & Objectives

The goal of this ablation study protocol is to systematically measure, validate, and isolate the performance contributions of each architectural upgrade proposed in the **MobilePixorNeXt v2** roadmap. Every experiment is strictly benchmarked against the verified baseline: **MobilePixorNeXt + OGA Loss (Anchor $M_0$)**.

This protocol defines:
1. The **exact specifications of the baseline ($M_0$)**.
2. Standardized **evaluation metrics** (KITTI 3D / BEV mAP, latency, FLOPs, memory).
3. The **progressive cumulative ablation matrix** ($M_0 \to M_1 \to M_2 \to M_3 \to M_4$).
4. The **component isolation and sensitivity experiments** (leave-one-out, kernel scales, gating weights, IoU quality factors).
5. Step-by-step **execution instructions and reporting templates**.

---

## 2. Baseline Definition (Anchor $M_0$)

All experiments MUST be strictly compared against the official baseline defined below:

| Dimension | Specification | Configuration Location / Parameter |
| :--- | :--- | :--- |
| **Input Representation** | Rich8 BEV Tensor | 8 channels: `[max_height, mean_height, count_norm, intensity_max, intensity_mean, density, occupancy, range]` |
| **Input Resolution** | $8 \times 800 \times 704$ | Grid range: $x \in [0, 70.4\text{m}]$, $y \in [-40.0, 40.0\text{m}]$, $z \in [-3.0, 1.0\text{m}]$ at $0.1\text{m}$ resolution |
| **Stem** | Fast Spatial Stem (Stride 2) | Conv $3\times 3$ ($s=2, p=1$, $8 \to 32$) + BN + SiLU $\to$ Conv $3\times 3$ ($s=1$, $32 \to 32$) |
| **Backbone Stages** | 3-stage Modern Inverted Bottlenecks | Stage 2: $32 \to 48$ (2x $7\times 7$ DW blocks)<br/>Stage 3: $48 \to 96$ (4x $7\times 7$ DW blocks + C3 LiteMLA)<br/>Stage 4: $96 \to 128$ (2x $7\times 7$ DW blocks, pure conv) |
| **Attention** | Single-scale LiteMLA @ C3 | Kernel scale: $5\times 5$ DW, Head dim: 16, FP32 linear accumulation, LayerScale init: $0.01$ |
| **Neck** | Bilinear Scale-Gated FPN (SG-FPN) | Top-down pyramid with zero-initialized adaptive scale gates: $2 \cdot \sigma(\text{Conv}_{3\times 3}(\cdot))$. Out: 16 channels @ stride 4 ($200 \times 176$) |
| **Detection Header** | 4-Task Decoupled Heads | Conv $3\times 3$ + BN + SiLU $\to$ $1\times 1$ Conv:<br/>1. Classification: 3 ch (Car, Ped, Cyc)<br/>2. Offset: 2 ch ($dx, dy$)<br/>3. Size: 2 ch ($\log l, \log w$)<br/>4. Yaw: 2 ch ($\cos \theta, \sin \theta$) |
| **Loss Function** | Oriented Geometric Adaptive Loss (OGA) | • **Regression**: MGIoU (Metric-Guided Rotated 3D IoU)<br/>• **Classification**: Dynamic Gaussian Focal Loss with distance-adaptive $\beta$<br/>• **Heading**: Smooth $L_1$ + Continuous angular consistency<br/>• **Multi-Task Balance**: `TemperatureSoftmaxUncertainty` ($T=2.0$) |
| **Optimizer & Schedule** | AdamW + Cosine Warmup | • Optimizer: AdamW ($lr=1\times 10^{-3}$, $\beta=(0.9, 0.999)$, weight decay $=1\times 10^{-4}$)<br/>• Warmup: 5 epochs linear warmup from $1\times 10^{-6}$<br/>• Annealing: Cosine Annealing to 100 epochs ($lr_{min}=1\times 10^{-6}$)<br/>• Batch size: 16<br/>• Precision: BF16 mixed-precision (`torch.cuda.amp.autocast`) + TF32 enabled |
| **Config File** | `configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json` | Fully synchronized in repository |

---

## 3. Evaluation Protocol & Metrics

### 3.1. Benchmark Dataset & Split
- **Dataset**: KITTI 3D Object Detection Benchmark.
- **Split**: Official standard training split:
  - **Train**: 3,712 frames.
  - **Validation**: 3,769 frames.
- **Classes**: Car, Pedestrian, Cyclist.

### 3.2. Primary Detection Accuracy Metrics (R40 Benchmark)
Evaluate using the official KITTI 40-recall position metric (R40) for both **3D Bounding Box Detection ($AP_{3D}$)** and **Bird's Eye View Detection ($AP_{BEV}$)**:
- **Car**: IoU threshold $\ge 0.70$ (Easy, Moderate, Hard).
- **Pedestrian**: IoU threshold $\ge 0.50$ (Easy, Moderate, Hard).
- **Cyclist**: IoU threshold $\ge 0.50$ (Easy, Moderate, Hard).
- **Primary Metric for Ranking**: **Car $AP_{3D}$ (Moderate)** and **Overall Mean $AP_{3D}$**.

### 3.3. Edge Hardware & Efficiency Metrics
All efficiency metrics must be benchmarked on dedicated edge-relevant hardware:
- **Target Platforms**:
  1. NVIDIA RTX 3060 Mobile / Laptop (Host baseline).
  2. NVIDIA Jetson AGX Orin / Xavier (Target embedded deploy).
- **Metrics**:
  - **Total Parameters (M)**: Training parameters vs. Inference deployment parameters.
  - **Computational Complexity (GFLOPs / GMACs)**: Evaluated at standard input $(1, 8, 800, 704)$.
  - **End-to-End Latency (ms)**: Batch size = 1, measured as mean over 500 forward runs following 100 warmup iterations with CUDA synchronizations.
  - **Throughput (FPS)**: $1000 / \text{Latency (ms)}$.
  - **Peak VRAM (MB)**: Measured via `torch.cuda.max_memory_allocated()`.

---

## 4. Progressive Cumulative Ablation Matrix

The roadmap progresses through 4 cumulative models. Each milestone integrates one validated improvement on top of the preceding version:

```
               [M0: Baseline MobilePixorNeXt + OGA Loss]
                                 │
                   + Structural Reparameterization
                                 ▼
                     [M1: Rep-MobilePixorNeXt]
                                 │
                   + Robust Multi-Scale LiteMLA
                                 ▼
                [M2: Rep-MobilePixorNeXt-MS-LiteMLA]
                                 │
                   + Bi-directional Scale-Gated Neck
                                 ▼
                         [M3: Bi-SGFPN]
                                 │
                   + IoU-Aware Quality Detection Header
                                 ▼
                    [M4: MobilePixorNeXt v2 SOTA]
```

### Milestone Specifications Table

| Milestone | Code | Components Added / Modified | Rationale | Expected Primary Benefit |
| :--- | :--- | :--- | :--- | :--- |
| **$M_0$** | `baseline` | MobilePixorNeXt + OGA Loss | Controlled Anchor | Establishes reproducible baseline performance |
| **$M_1$** | `+rep` | $M_0$ + Rep-MobilePixorNeXt Block ($7\times 7 + 3\times 3 + \text{Id} \to 7\times 7$) | Multi-branch gradient flow in training; zero inference overhead | $+0.8$ to $+1.5\%$ $AP_{3D}$, **0 ms** extra inference latency |
| **$M_2$** | `+ms_attn` | $M_1$ + Multi-Scale LiteMLA $(3, 5)$ + QK-Norm | Multi-scale context for small (pedestrian) and long (truck) objects | $+1.0$ to $+1.8\%$ $AP_{3D}$ on Pedestrian/Cyclist, zero BF16 overflow |
| **$M_3$** | `+bi_sgfpn` | $M_2$ + Bi-directional Scale-Gated Neck (Bi-SGFPN) | Bidirectional feature propagation preserves fine metric corner geometry | $+1.2$ to $+2.0\%$ $AP_{3D}$ on Moderate/Hard distant vehicles |
| **$M_4$** | `+iqa_head` | $M_3$ + IoU-Aware Quality Header + Joint NMS Scoring | Calibrates classification confidence with physical bounding box overlap | $+1.5$ to $+2.5\%$ $AP_{3D}$, eliminates false-positive high-conf outliers |

---

## 5. Component Isolation & Sensitivity Experiments

To prevent confounding effects, each individual mechanism is subjected to isolation and sensitivity tests:

### 5.1. Experiment Set A: Structural Reparameterization Equivalence & Branches
- **A1 (Exact Equivalence Check)**: Verify that the PyTorch forward output of the multi-branch block in `eval()` mode matches the fused `deploy()` block within machine precision:
  $$\max |Y_{\text{multi-branch}} - Y_{\text{fused}}| < 10^{-5}$$
- **A2 (Branch Contribution)**:
  - A2.1: $7\times 7$ DW only (Baseline)
  - A2.2: $7\times 7$ DW + Identity
  - A2.3: $7\times 7$ DW + $3\times 3$ DW
  - A2.4: Full Rep: $7\times 7$ DW + $3\times 3$ DW + Identity

### 5.2. Experiment Set B: LiteMLA Attention Kernel Scales & Normalization
- **B1 (Kernel Scales)**:
  - B1.1: Single scale $k=5$ (Baseline)
  - B1.2: Multi-scale $k \in \{3, 5\}$
  - B1.3: Multi-scale $k \in \{3, 5, 7\}$
  - B1.4: Pure convolution (No attention at C3)
- **B2 (Numerical Stability)**:
  - B2.1: Raw linear attention (Unnormalized)
  - B2.2: QK-LayerNorm
  - B2.3: QK-RMSNorm (Lightest compute)

### 5.3. Experiment Set C: Neck Fusion Architectures
- **C1 (Routing Comparison)**:
  - C1.1: Baseline Top-Down SG-FPN
  - C1.2: Standard PANet (Naive element-wise sum top-down + bottom-up)
  - C1.3: Bi-SGFPN (Scale-Gated bidirectional top-down + bottom-up)
- **C2 (Gate Initialization)**:
  - C2.1: Zero-init (Gate multiplier initially $= 1.0$)
  - C2.2: Random normal init
  - C2.3: Fixed gate $= 1.0$ (Static residual)

### 5.4. Experiment Set D: IoU-Aware Quality Branch & NMS Calibration
- **D1 (Supervision Target & Loss)**:
  - D1.1: Rotated 3D IoU target with Binary Cross-Entropy Loss
  - D1.2: Rotated 3D IoU target with Smooth $L_1$ Loss
  - D1.3: Centerness target (FCOS style)
- **D2 (Quality Weighting Parameter $\alpha$)**:
  Evaluate inference scoring formula:
  $$S_{\text{final}} = P_{\text{cls}}^{1 - \alpha} \cdot S_{\text{iou}}^{\alpha}$$
  Sweep $\alpha \in \{0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0\}$.
  *(Note: $\alpha = 0.0$ represents standard classification-only NMS; $\alpha = 1.0$ represents IoU-only ranking).*

### 5.5. Experiment Set E: Physics-Consistent 3D Data Augmentation (PCU-Aug)
- **E1 (Transform Pipeline Comparison)**:
  - E1.1: Baseline Augmentations (`OneOf([Rot, Scale, Trans])`)
  - E1.2: + 3D Horizontal Flip (`Flip-Y` along lateral axis)
  - E1.3: + Standard SECOND GT-Sampling (without shadow masking or radiometric calibration)
  - E1.4: + Full PCU-Aug (Ray-consistent shadow masking + $(R_1/R_2)^2$ subsampling + range intensity calibration + uncertainty curriculum)
- **E2 (Ray Shadow Masking Impact)**:
  - Evaluate false-positive rate and background clutter suppression when shadow frustum masking is toggled on vs off.

---

## 6. Standardized Results Reporting Template

All ablation results must be logged in the following standardized Markdown format to allow direct comparative verification:

### Main Milestone Ablation Table

| Model ID | Method / Variant | Params (M) Train/Deploy | GFLOPs | Latency (ms) RTX 3060 | FPS | Car $AP_{3D}$ (Easy / Mod / Hard) | Ped $AP_{3D}$ (Easy / Mod / Hard) | Cyc $AP_{3D}$ (Easy / Mod / Hard) | Mean $AP_{3D}$ Mod |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M_0$** | Baseline MobilePixorNeXt + OGA | 1.34 / 1.34 | 14.2 | 10.8 ms | 92.6 | - / - / - | - / - / - | - / - / - | - |
| **$M_1$** | + Rep-MobilePixorNeXt Block | 1.82 / 1.34 | 14.2 | 10.8 ms | 92.6 | - / - / - | - / - / - | - / - / - | - |
| **$M_2$** | + Robust MS-LiteMLA $(3, 5)$ | 1.84 / 1.36 | 14.5 | 10.9 ms | 91.7 | - / - / - | - / - / - | - / - / - | - |
| **$M_3$** | + Bi-SGFPN Neck | 1.93 / 1.45 | 16.1 | 11.4 ms | 87.7 | - / - / - | - / - / - | - / - / - | - |
| **$M_4$** | + IQA-Header ($\alpha=0.5$) | 1.95 / 1.47 | 16.3 | 11.5 ms | 86.9 | - / - / - | - / - / - | - / - / - | - |
| **$M_5$** | + PCU-Augmentation (SOTA v2) | 1.95 / 1.47 | 16.3 | 11.5 ms | 86.9 | - / - / - | - / - / - | - / - / - | - |

---

## 7. Execution Commands & Verification Guide

### 7.1. Running the Baseline ($M_0$)
```bash
# Run training in background with BF16 and 100 epochs
python train.py --config configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json --work-dir runs/ablation_m0_baseline

# Evaluate checkpoint on KITTI Validation Set (R40)
python evaluate.py --config configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json --checkpoint runs/ablation_m0_baseline/best_checkpoint.pth --eval-r40
```

### 7.2. Benchmark Speed & Profile Resources
```bash
# Profile FPS, latency, and memory allocation
python tools/benchmark_model.py \
    --config configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json \
    --device cuda:0 \
    --warmup 100 \
    --iters 500 \
    --export-deploy
```

### 7.3. Reparameterization Unit Verification
```bash
# Verify mathematical equivalence before running any full training
python -m pytest tests/test_rep_block.py -k test_reparameterization_numerical_equivalence -v
```
