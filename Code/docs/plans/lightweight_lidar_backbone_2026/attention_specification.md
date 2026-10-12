# Focal context and lightweight attention specification

Updated: 2026-10-07. Status: tasks 10–21 implement feature-option resolution, focal/ECA/SimAM backbone hooks, optional SG-FPN fusion32/detail path and actual full-resolution parameter audits. CPU numerical checks and the current full regression suite pass. Grouped heads, dedicated GPU validation and accuracy evidence remain pending. See [execution log](execution_progress.md).

This document extends [specification.md](specification.md). Implementation order and verification commands are in [implementation_plan.md](implementation_plan.md). The main candidate uses **focal context at stride 8**. **ECA at stride 4 is optional**, with SimAM as a separate comparison option. Implementing these options does not require training every variant.

## 1. Scope and configuration policy

| Component | Main training candidate | Additional supported design |
| --- | --- | --- |
| Encoding | hist14 v1 | Historical encodings remain available |
| Stem / stages | Legacy stem, widths 32/48/96/128, depths 3/4/2 | Historical depth 2/4/2 |
| Stride-8 context | Feature-based focal context | Context none for the reference configuration; historical LiteMLA in matching legacy configs |
| Stride-4 attention | none | ECA first optional choice; SimAM comparison |
| Neck | Existing SG-FPN, internal width 24, output 32 | Internal width 32 and early-detail path as optional follow-ups |
| Heads / quality | Car and Pedestrian/Cyclist, supported OGA/IQA contracts | Existing strategy capability matrix |
| Deployment | GPU training/evaluation | Jetson/TensorRT/INT8 remain deferred |

Do not enable focal context, ECA, SimAM, detail skip and wider fusion simultaneously by default. The main recipe selects focal context only. A separately named focal+ECA preset exists for an explicit next comparison; focal+SimAM and other combinations are clearly labelled optional. New presets preserve sampling/loss/schedule unless their names expressly identify another objective.

Configuration fields under `model`:

`detector/core/backbone_config.py` is the planned pure resolver shared by backbone construction and central detection validation. It owns these defaults/validation rules; the notebook must not maintain a second interpretation.

```json
{
  "c4_attention": "none",
  "c4_attention_scales": [],
  "c4_attention_qk_norm": "none",
  "c4_context": "focal",
  "c4_context_version": 1,
  "c4_context_bottleneck": 64,
  "c4_context_dilations": [1, 2, 3],
  "c4_context_layer_scale_init": 0.001,
  "local_attention": "none",
  "neck_fusion_channels": 24,
  "detail_path": false
}
```

The ECA variant sets `local_attention=eca`, `local_attention_eca_kernel_size=3` and `local_attention_layer_scale_init=0.001`. The SimAM variant sets `local_attention=simam`, `local_attention_simam_lambda=0.0001` and the same initial residual scale. Missing new fields in historical configs imply context none, local attention none, fusion width 24 and no detail path, preserving historical state keys and behavior.

Validation rules:

- `c4_context` is `none|focal`; `local_attention` is `none|eca|simam`. ECA and SimAM are mutually exclusive at this hook.
- Focal and LiteMLA are mutually exclusive at the stride-8 context hook. Reject both enabled; do not silently prefer one.
- Context v1 requires finite nonnegative LayerScale initialization, positive integer bottleneck width and distinct ascending positive integer dilations. The first preset fixes D=64 and dilations [1,2,3]; other settings are explicit variants, not silent defaults.
- ECA kernel is an odd positive integer, initially 3; kernel 5 is an explicit comparison. SimAM lambda is finite and strictly positive. Booleans are invalid where numerical integers/scalars are required.
- Configuring a module-specific field while selecting a different module is rejected or stripped only during an explicitly logged preset-resolution step. It must not disappear silently in a saved resolved config.
- Compatibility exception for existing LiteMLA fields: historical no-attention configs can retain their saved inactive scales/norm settings in resolved identity (the existing `e0_conv` preset retains scales `[5]`). Candidate topologies require inactive LiteMLA scales `[]` and norm `none`. New focal/ECA/SimAM-specific fields always require their corresponding selected module.
- Optional fusion width is 24 or 32; detail path is a boolean. These options initially apply only to standard SG-FPN on MobilePixorNeXt; reject incompatible RC neck/backbone selections until implemented.
- New candidate grids must be divisible by 16 so existing downsampling/top-down fusion shapes agree. Preserve historical geometry behavior in legacy configurations.
- Config/hash/checkpoint identity includes module kind/version, placement, width/dilations/kernel/lambda, residual initialization, fusion width and detail-path settings.

## 2. Placement and tensor flow

Use explicit names for feature levels, since current source variables use different c-index names:

```text
stem_feature = stem(input)                              # [B,32,H/2,W/2]
local_feature = stage2(down2(stem_feature))              # [B,48,H/4,W/4]
local_feature = local_attention(local_feature)          # optional ECA OR SimAM
context_feature = stage3(down3(local_feature))          # [B,96,H/8,W/8]
context_feature = selected_context(context_feature)    # focal OR legacy LiteMLA OR identity
deep_feature = stage4(down4(context_feature))           # [B,128,H/16,W/16]
neck_output = sgfpn(local_feature, context_feature, deep_feature,
                   optional_stem_detail=stem_feature)  # [B,32,H/4,W/4]
```

Refined stride-4 features feed **both** the next downsampling stage and the stride-4 neck lateral. Focal-refined stride-8 features likewise feed both the deep stage and their neck lateral. Neither mechanism runs on classification/regression/IQA logits.

New production files are planned as `detector/core/models/backbones/focal_context.py` and `detector/core/models/backbones/light_attention.py`. `MobilePixorNeXtBackbone` owns the adapters; all heads share their output. No attention is duplicated per head. Suggested state prefixes are `c3_light_attention.*` and `c4_context.*`; disabled modules are `nn.Identity` without new persistent state.

## 3. Feature-based focal context v1

### 3.1 Exact operations

Let X have shape `[B,C,H,W]`, C=96, D=64, L=3. All convolutions are bias-free. BN uses the repository's existing default eps/momentum and affine/running-stat conventions; record these in module version metadata.

1. `Xn = BN_pre(X)`.
2. `Q,S,G = split(PW_in(Xn), [D,D,L+1])`, where PW_in is 1×1, C→2D+L+1 (96→132). Q is the query, S the context seed and G the per-position gate logits.
3. Set `H0=S`. For i=1..L, compute `Hi=SiLU(DW3_i(Hi-1))`, using stride 1, dilation di and padding di. The three levels are **sequential**, with d=[1,2,3]; they are not parallel independent convolutions.
4. `Hg=mean_spatial(HL)` in FP32, shape `[B,D,1,1]`. Broadcast Hg for fusion. No BN operates on this pooled 1×1 tensor.
5. `A=softmax(G.float(),dim=channel)`, then cast weights to the feature dtype. At each position, the four nonnegative weights sum to 1. There is no sigmoid gate on each level and no occupancy mask.
6. `F=sum_i(A_i*Hi)+A_global*Hg`.
7. `M=PW_context(F)`, a D→D 1×1 projection.
8. `U=Q*M`, elementwise modulation; accumulate this product in FP32 under mixed precision before the output projection's supported autocast boundary.
9. `R=BN_out(PW_out(U))`, with D→C 1×1 projection.
10. `Y=X+gamma*R`, where gamma is a learned `[1,C,1,1]` LayerScale parameter initialized to 1e-3 in the preset.

Shape is preserved. Use out-of-place operations where aliases could corrupt residual inputs. Pooling/gate reductions use FP32; tensors and parameters remain on the input device. Returning the input dtype must not introduce hidden CPU transfers. Verify supported FP32/FP16/BF16 behavior rather than claiming all magnitudes are overflow-proof.

Implemented precision boundary: form modulation in FP32, then match the output projection's weight dtype before its autocast boundary. With ordinary FP32 weights under autocast, the projection receives FP32 modulation; an explicitly FP16/BF16 model requires a matching low-precision projection input outside autocast. Residual accumulation is FP32 for FP16/BF16 inputs and returns the input dtype. CPU tests cover these modes; CUDA remains a later gate.

Standard BN still needs sufficient training samples/spatial elements. A `[1,C,1,1]` corner case is checked in eval mode or rejected by the existing training-BN constraint; do not invent a separate silent BN fallback.

### 3.2 Initialization and parameter count

Use standard existing convolution initialization and BN weight=1/bias=0. Nonzero small gamma allows branch gradients from the first step. Setting gamma to zero is supported for an exact identity fixture, but branch-weight gradients will initially be zero in that fixture. Do not simultaneously zero gamma and output projection weights in the training preset.

For C=96, D=64 and L=3:

```text
PW_in:           96*132       = 12,672
Three DW3:       3*64*9       =  1,728
PW_context:      64*64        =  4,096
PW_out:          64*96        =  6,144
Two affine BN:   2*(2*96)     =    384
LayerScale:      96           =     96
Total                         = 25,120
```

BN running buffers are excluded from learned parameter counts. No extra gate projection, bias or conditioning adapter is hidden in this figure. These are structural projections, to be recounted from the implemented model.

This dense BEV adaptation is motivated by [SFMNet](https://arxiv.org/html/2503.12093v1) and receptive-field design in [PillarNeXt](https://arxiv.org/abs/2305.04925). It is not their exact architecture and does not obtain sparse execution savings automatically. Its AP/latency are unknown until measured.

## 4. ECA and SimAM residual adapters

### 4.1 Shared adapter contract

For stride-4 X with C=48, let `T(X)` be the selected attention core:

```text
Y = X + beta * (T(X) - X)
```

Beta is one learned scalar, initialized to 1e-3. Compute the residual combination in a numerically appropriate dtype and return the input dtype. The core transforms are not appended repeatedly inside every inverted block. When disabled, use identity with zero additional parameters. Beta is included in optimizer, checkpoint and parameter reports.

At beta=0, output exactly equals X for finite core outputs. With the recommended beta=1e-3 and a core gate of 0.5, initial scaling is 0.9995 rather than abruptly halving the entire trunk. Beta is learned without an undocumented clamp; parameter count includes it.

### 4.2 ECA core

1. `z=mean_spatial(X)` in FP32, `[B,C,1,1]`.
2. Reshape to `[B,1,C]`; apply Conv1d 1→1, kernel k, padding `(k-1)/2`, bias=false.
3. `a=sigmoid(conv(z))`, reshape to `[B,C,1,1]`.
4. `T(X)=X*a`, broadcast along H/W.

Initialize the small Conv1d weights to zero for this adapter version, making initial gates 0.5. The kernel can learn from the first step when beta is nonzero; document this explicit initialization rather than assuming the author's default initialization. For k=3, core has 3 weights, wrapper has 1: **4 total learned parameters**. For k=5, total is 6. Global pooling is not point-support-aware and does not retain spatial location.

The implemented core also computes the small channel correlation and sigmoid in FP32 with autocast disabled, casting registered kernel weights differentiably on their existing device. Gates are cast to the input dtype for multiplication. This adds no learned parameters; meta-device shape audits require no autocast backend. The scalar residual wrapper and stride-4 hook are implemented.

The [ECA paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html) and [author implementation](https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py) establish the core mechanism; they do not establish KITTI gains for this placement or wrapper.

### 4.3 SimAM core

In FP32, independently per batch/channel:

```text
mu = mean_spatial(X)
d = (X - mu)^2
n = max(H*W - 1, 1)
v = max(sum_spatial(d)/n, 0)
e = d / (4*(v + lambda)) + 0.5
a = sigmoid(e)
T(X) = X*a
```

Lambda defaults to 1e-4. The n guard defines behavior for H*W=1; a constant map produces gate sigmoid(0.5), including that corner case. The core has zero learned parameters, while this residual wrapper has **one** learned scalar. Do not report the whole adapter as parameter-free.

The implemented `SimAMCore` returns FP32 gates and casts them to input dtype for the core product. Both residual adapters accumulate FP16/BF16 residual combinations in FP32 and return the input dtype. The factory uses the shared resolver, rejects inactive module overrides and returns a parameter/state-free Identity for `none`. Beta is registered once and remains unclamped after initialization.

Statistics and intermediate elementwise tensors touch the entire feature map. Parameter-free is not compute-free; report measured memory/timing if comparing efficiency. A sparse BEV feature's statistics may be dominated by background/outliers; this is a hypothesis to investigate, not a proven failure of SimAM. The term three-dimensional attention in [SimAM](https://proceedings.mlr.press/v139/yang21o.html) refers to feature attention across C/H/W, not an explicit 3D LiDAR box representation. The [author core](https://github.com/ZjjConan/SimAM/blob/master/networks/attentions/simam_module.py) is the reference for the formula, with the single-element guard/residual wrapper documented as adaptations.

## 5. Optional detail path and wider final neck fusion

These follow-ups are implemented only if selected after the primary focal candidate; no additional long runs are mandatory for design completion.

### 5.1 Fusion width

`neck_fusion_channels=F`, F=24 default or 32 optional. The standard SG-FPN changes exactly:

- Stride-4 lateral: 1×1, 48→F, bias-free.
- Top-down projection: 3×3, 48→F, bias-free, BN(F), SiLU.
- Stride-4 gate: existing depthwise 3×3, F channels, bias=true, existing zero initialization/formula.
- Final projection: 3×3, F→32, bias-free, BN(32), SiLU.

Stride-8/deep laterals remain 48 channels. At F=32, the neck adds **6,240** parameters relative to F=24 for the current SG-FPN, with scale gates enabled. Recount if other neck/gate options change. This removes the internal 24-channel final-fusion width, but no AP improvement is assumed.

The no-context reference changes from 635,408 to 641,648 parameters with output32 and scale gates, retaining `[1,32,200,176]` on the full KITTI grid. F=24 preserves legacy initialization/state keys/predictions and RC-neck defaults; F=32 rejects RC necks and misaligned grids. Constructor/registry wiring is implemented; the complete focal/grouped fusion32 runtime recipe is created in task 48.

### 5.2 Early detail branch

When `detail_path=true`, take stem output `[B,32,H/2,W/2]` through DW3 stride2/pad1 (bias-free), BN32, SiLU, PW1 32→F (bias-free), BN(F), SiLU. Fuse it **after standard stride-4 SG-FPN fusion and before the final output projection**:

```text
final_fusion = standard_fusion + detail_gamma * detail_feature
```

Detail gamma is learned `[1,F,1,1]`, initialized to 1e-3. The branch adds 1,192 parameters at F=24 or 1,472 at F=32. Keep output stride 4; this does not increase prediction-grid resolution. Validate shape alignment on asymmetric divisible-by-16 grids; never silently resize a misaligned branch.

Implemented branch initialization follows the historical modules so same-seed shared weights remain identical. Disabled detail adds no persistent state; gamma-zero eval matches the disabled output exactly. Shape mismatches raise an error. The profiler includes both branch and direct gamma parameters in the neck budget.

Stage-level dense/one-shot aggregation and support/height-conditioned focal gates remain research backlog, rather than unspecified switches in this release. Coordinate Attention and CBAM are not included in the accepted lightweight-attention set.

## 6. Verified backbone budgets and projected grouped detector totals

The backbone-plus-neck counts below are now verified from actual integrated modules, including full-resolution meta shapes. The grouped-head/vertical totals remain projections: the current profiler uses a single BEV head with IQA (93,130 parameters) and reports the six OGA criterion parameters separately. Two grouped BEV heads with IQA project to 186,161. A later vertical branch projects to 18,626 per group, 37,252 total.

| Variant | Context | Local attention | Fusion / detail | Backbone + neck, verified | BEV detector + IQA, projected |
| --- | --- | --- | --- | ---: | ---: |
| Reference | none | none | 24 / off | 635,408 | 821,569 |
| **Main** | focal | none | 24 / off | **660,528** | **846,689** |
| Optional ECA | focal | ECA k3 | 24 / off | 660,532 | 846,693 |
| Optional SimAM | focal | SimAM | 24 / off | 660,529 | 846,690 |
| Optional wider fusion | focal | none | 32 / off | 666,768 | 852,929 |
| Optional detail | focal | none | 24 / on | 661,720 | 847,881 |

Do not select combined optional variants implicitly. For an explicitly selected focal+ECA+fusion32+detail variant, the arithmetic estimate is 668,244 backbone including neck. This is a budget illustration, not the default candidate. The main 3D detector with two vertical branches projects to 883,941; recalculate if branch topology changes. Train-only criterion parameters and BN buffers remain separate.

## 7. Compatibility, diagnostics and release checks

- Backbone output/tensor ordering is unchanged; grouped targets, box ownership, OGA/IQA detach boundaries and NMS policy retain their main-spec contracts.
- Context/local-attention parameters join the model optimizer exactly once. Warm-start loads only compatible tensors and never invents missing context weights; strict resume requires identical adapter metadata and restores BN/LayerScale/beta state.
- Expose opt-in diagnostics for focal level weights, gamma/beta and finite activation statistics. Observability must not add extra losses or unconditional GPU synchronization to every training step.
- Context and ECA are learned from features. No mask suppresses features/predictions at empty raw-point cells. SimAM has no learned spatial gate despite its adapter's learned scalar.
- Unit fixtures cover constant/zero maps, impulse/sparse maps, random maps, asymmetric shapes, kernel/dilation/lambda validation, beta/gamma-zero identity and nonzero-scale gradients.
- Gate weights normalize across the four levels; varying one batch item cannot affect another through pooling/SimAM statistics in eval mode. Training BN has its normal cross-sample behavior and is not expected to satisfy that eval isolation property.
- Test FP32, then supported AMP on GPU, deterministic eval/state reload and full-resolution shape/parameter reports. Check all supported Gaussian losses with none/ECA/SimAM; representative binary baseline/OGA/UWAG paths; exact Q-OGA restrictions stay in force.
- Regression/quality tests must show box gradients reach shared adapters and IQA targets remain detached. Empty groups remain finite and classification negatives still contribute.
- Matched training/evaluation evidence is required for accuracy claims. SOTA requires the correct 3D protocol and external comparison; small attention modules alone do not establish it.

Batch16 materializes complete BEV runtime presets from the manifest, including focal/local-none, separate ECA/SimAM and optional fusion32/detail recipes. Hooks, grouped objective/checkpoint integration and recipe counts have synthetic CPU evidence; full-resolution meta-device shape audits cover every grouped recipe. New candidate names show context/local/fusion/detail and include architecture/encoding/objective identity; historical default names retain their behavior. Main local attention remains none. Dedicated GPU/AMP/Inductor, true 3D and trained accuracy gates remain pending. No accuracy or latency improvement has been established.
