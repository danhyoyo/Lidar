# Attribution for adapted C4 attention implementations

Reviewed 2026-09-26. The following notices concern the indicated mechanisms in
`detector/core/models/backbones/c4_attention.py`, not a relicensing of this entire
repository. These are modified local implementations, not upstream checkpoints.

## LSKRefinement

Based on LSKNet's LSKblock/Attention spatial-selection mechanism.

- Creators: Yuxuan Li, Qibin Hou, Zhaohui Zheng, Ming-Ming Cheng, Jian Yang,
  and Xiang Li (ICCV 2023).
- Upstream copyright notice: Copyright (c) 2022 MCG-NKU.
- Source: https://github.com/zcablii/LSKNet/blob/main/mmrotate/models/backbones/lsknet.py
- License: Creative Commons Attribution-NonCommercial 4.0 International,
  https://creativecommons.org/licenses/by-nc/4.0/ . Retained upstream text:
  `LSKNet-LICENSE.txt`, including its warranty disclaimer.
- Modifications: dependency-free PyTorch adapter, pre-normalization, one residual
  LayerScale, different local names, and omission of full backbone/MLP/drop-path.
  The noncommercial restriction applies to the adapted LSK material. Attribution
  and a research citation do not remove that restriction.

## LiteMLARefinement

Based on the LiteMLA mechanism from MIT Han Lab's EfficientViT.

- Creators: Han Cai, Junyan Li, Muyan Hu, Chuang Gan, and Song Han.
- Upstream copyright notice: Copyright [2023] [Han Cai].
- Source: https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/models/nn/ops.py
- License: Apache License 2.0; full upstream text and notice are retained in
  `EfficientViT-LICENSE.txt`. Distributed on an AS IS basis, without warranties
  or conditions, as described in that license.
- Modifications: standalone residual adapter, always-linear computation, explicit
  FP32 accumulation under FP16/BF16 autocast, eps=1e-6, configurable small heads,
  per-channel LayerScale, and no separate EfficientViT local MBConv.

## DeformableAttentionRefinement

Based on the grouped learned-offset deformable self-attention mechanism from DAT.

- Creators: Zhuofan Xia, Xuran Pan, Shiji Song, Li Erran Li, and Gao Huang
  (CVPR 2022).
- Source: https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py
- License: Apache License 2.0. The full Apache 2.0 text already retained as
  `EfficientViT-LICENSE.txt` also covers this permissively licensed adaptation.
- Modifications: dependency-free NCHW residual adapter for dynamic BEV shapes,
  four channel/offset groups, stride-4 sampled keys and values, depthwise local
  positional encoding, explicit FP32 sampling/softmax under mixed precision,
  no complete DAT backbone, MLP, stochastic depth, or learned relative-position
  table.

## BiLevelRoutingAttentionRefinement

Based on the NCHW Bi-Level Routing Attention mechanism from BiFormer.

- Creators: Lei Zhu, Xinjiang Wang, Zhanghan Ke, Wayne Zhang, and Rynson Lau
  (CVPR 2023).
- Source: https://github.com/rayleizhu/BiFormer/blob/public_release/ops/bra_nchw.py
- License: MIT; full upstream text retained in `BiFormer-LICENSE.txt`.
- Modifications: dependency-free PyTorch routed gather, explicit padding for
  arbitrary BEV sizes, four attention heads, 8x8 regions with top-4 routing,
  FP32 routing/softmax under mixed precision, per-channel LayerScale, and no
  complete BiFormer backbone, MLP, or stochastic depth.

Academic papers should cite the original mechanisms and describe these
adaptations. Integration into MobilePIXOR alone is not evidence of novelty.
