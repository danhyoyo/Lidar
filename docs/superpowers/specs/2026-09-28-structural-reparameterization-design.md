# Structural Reparameterization for MobilePixorNeXt (Pillar 1 Design Spec)

**Date**: 2026-09-28  
**Topic**: Pillar 1 - Structural Reparameterization (`Rep-MobilePixorNeXt`)  
**Status**: Approved Specification  
**Baseline Anchor**: MobilePixorNeXt + OGA Loss ($M_0$)  
**Target Milestone**: $M_1$ (`+rep`)  

---

## 1. Overview & Motivation

In 3D LiDAR Object Detection from Bird's Eye View (BEV), large depthwise convolutions ($7\times 7$, covering $0.7\text{m} \times 0.7\text{m}$ in physical metric space) are essential for capturing extended objects like cars ($4.5\text{m} \times 1.8\text{m}$) and trucks without feature fragmentation.

However, training a deep neural network with large kernel depthwise convolutions from scratch poses an optimization challenge:
1. **At Training Time**: $7\times 7$ kernels have diffuse gradient dynamics in early epochs, where peripheral weights disperse gradient signals away from the central reflection point. A multi-branch topology ($7\times 7 + 3\times 3 + \text{Identity}$) provides rich gradient flow and inductive bias towards object centers.
2. **At Inference Time**: Multi-branch architectures incur substantial memory access overhead (MACs), kernel launch latency, and GPU cache fragmentation.

**Solution**: Decouple training and inference via **Structural Reparameterization**:
- **Training Topology**: Multi-branch depthwise block ($7\times 7\text{ DW+BN} + 3\times 3\text{ DW+BN} + \text{Identity BN}$).
- **Inference Topology**: A single equivalent $7\times 7$ depthwise convolution with bias ($W_{\text{deploy}}, b_{\text{deploy}}$) obtained via exact linear algebraic fusion.
- **Inference Cost**: **0 ms latency overhead, 0 extra memory allocation.**

---

## 2. Mathematical Derivation of Kernel Fusion

Let input feature map be $X \in \mathbb{R}^{B \times C \times H \times W}$. In depthwise convolution, each channel $c \in \{1, \dots, C\}$ is convolved independently.

### 2.1. Fusing Depthwise Conv with BatchNorm
For a depthwise convolution with weights $W \in \mathbb{R}^{C \times 1 \times K \times K}$ and a following BatchNorm with running mean $\mu$, running variance $\sigma^2$, learnable scale $\gamma$, shift $\beta$, and stabilization constant $\epsilon$:

$$\text{BN}(\text{Conv}(X; W))_{c} = \gamma_c \cdot \frac{W_c * X_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c$$

By linearity, this is identical to a single convolution with fused weight $W'$ and bias $b'$:

$$W'_{c, 0, :, :} = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}} W_{c, 0, :, :}$$
$$b'_c = \beta_c - \frac{\gamma_c \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

### 2.2. Transforming All Branches to a $7\times 7$ Footprint

1. **Branch 1 ($7\times 7$ DW + BN)**:
   - Kernel $W_7 \in \mathbb{R}^{C \times 1 \times 7 \times 7}$.
   - Fuses into $W'_7 \in \mathbb{R}^{C \times 1 \times 7 \times 7}$ and $b'_7 \in \mathbb{R}^C$.

2. **Branch 2 ($3\times 3$ DW + BN)**:
   - Kernel $W_3 \in \mathbb{R}^{C \times 1 \times 3 \times 3}$.
   - Fuses into $W'_3 \in \mathbb{R}^{C \times 1 \times 3 \times 3}$ and $b'_3 \in \mathbb{R}^C$.
   - Symmetrically padded with zeros by 2 pixels on all four sides to form $\widetilde{W}'_3 \in \mathbb{R}^{C \times 1 \times 7 \times 7}$:
     $$\widetilde{W}'_{3, c, 0, i, j} = \begin{cases} W'_{3, c, 0, i-2, j-2} & \text{if } 2 \le i, j \le 4 \\ 0 & \text{otherwise} \end{cases}$$

3. **Branch 3 (Identity + BN)**:
   - The identity mapping on each channel $X_c$ is mathematically identical to convolution with a Kronecker delta kernel $W_{\text{id}} \in \mathbb{R}^{C \times 1 \times 7 \times 7}$:
     $$W_{\text{id}, c, 0, i, j} = \begin{cases} 1.0 & \text{if } i = 3 \text{ and } j = 3 \\ 0.0 & \text{otherwise} \end{cases}$$
   - Fused with BatchNorm parameters $(\gamma_{\text{id}}, \beta_{\text{id}}, \mu_{\text{id}}, \sigma^2_{\text{id}}, \epsilon_{\text{id}})$:
     $$W'_{\text{id}, c, 0, :, :} = \frac{\gamma_{\text{id}, c}}{\sqrt{\sigma^2_{\text{id}, c} + \epsilon_{\text{id}}}} W_{\text{id}, c, 0, :, :}$$
     $$b'_{\text{id}, c} = \beta_{\text{id}, c} - \frac{\gamma_{\text{id}, c} \mu_{\text{id}, c}}{\sqrt{\sigma^2_{\text{id}, c} + \epsilon_{\text{id}}}}$$

### 2.3. Exact Linear Summation
Because convolution is linear over the kernel tensor:

$$W_{\text{deploy}} = W'_7 + \widetilde{W}'_3 + W'_{\text{id}} \in \mathbb{R}^{C \times 1 \times 7 \times 7}$$
$$b_{\text{deploy}} = b'_7 + b'_3 + b'_{\text{id}} \in \mathbb{R}^C$$

In `deploy` mode, all branches are replaced by:
$$\text{Conv2d}(C, C, \text{kernel\_size}=7, \text{stride}=1, \text{padding}=3, \text{groups}=C, \text{bias}=\text{True})$$

---

## 3. Architecture & File Structure

### 3.1. New File: `detector/core/models/backbones/rep_blocks.py`
Contains:
- `trans_conv_bn_to_kernel_bias(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[Tensor, Tensor]`
- `trans_identity_bn_to_kernel_bias(bn: nn.BatchNorm2d, channels: int, kernel_size: int = 7) -> Tuple[Tensor, Tensor]`
- `class RepConv7x7(nn.Module)`:
  - `__init__(self, channels: int, deploy: bool = False)`
  - `forward(self, x: Tensor) -> Tensor`
  - `get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]`
  - `switch_to_deploy(self)`: Computes $W_{\text{deploy}}, b_{\text{deploy}}$, initializes `rbr_reparam`, removes training branches (`rbr_conv7`, `rbr_conv3`, `rbr_identity`), sets `self.deploy = True`.

### 3.2. Modifications to Existing Files

1. **`detector/core/models/backbones/mobilepixornext_blocks.py`**:
   - Update `MobilePixorNeXtBlock.__init__` with `use_reparam: bool = False`.
   - When `use_reparam=True`, replace `self.dwconv` + `self.norm1` with `self.dw_block = RepConv7x7(channels)`.
   - Forward pass routes through `self.dw_block` when `use_reparam=True`.
   - Add `switch_to_deploy(self)` to delegate to `self.dw_block.switch_to_deploy()`.

2. **`detector/core/models/backbones/mobilepixornext.py`**:
   - Update `MobilePixorNeXtBackbone.__init__` with `use_reparam: bool = False`.
   - Propagate `use_reparam=self.use_reparam` to `stage2`, `stage3`, and `stage4`.
   - Add `switch_to_deploy(self)` iterating through all submodules.

3. **`detector/core/models/backbones/registry.py`**:
   - In `_build_mobilepixornext`: pass `use_reparam=cfg.get("use_reparam", False)`.

4. **`detector/core/models/model.py`**:
   - In `CustomModel`: add `switch_to_deploy(self)` calling `self.backbone.switch_to_deploy()`.

5. **`configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json`**:
   - Add `"use_reparam": false` to anchor config for full backward compatibility.
   - Create experiment config `configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga_reparam.json` with `"use_reparam": true`.

---

## 4. Verification & Testing Strategy

### Test File: `tests/test_rep_block.py`
Must test:
1. **Numerical Equivalence (`test_repconv7x7_numerical_equivalence`)**:
   - Run forward pass on `RepConv7x7(channels=48, deploy=False).eval()`.
   - Call `switch_to_deploy()`.
   - Re-run forward pass on identical input.
   - Verify `torch.max(torch.abs(y_train - y_deploy)).item() < 1e-5`.
2. **Block-Level Equivalence (`test_rep_block_equivalence`)**:
   - Verify `MobilePixorNeXtBlock(channels=48, use_reparam=True)` produces identical output before and after `switch_to_deploy()`.
3. **Full Model Equivalence (`test_full_model_reparam_equivalence`)**:
   - Build `CustomModel` with `use_reparam=True`.
   - Compare `pred_train` vs `pred_deploy` on input tensor `(2, 8, 800, 704)` for `cls`, `offset`, `size`, `yaw`.
4. **Gradient Flow Test (`test_repconv7x7_gradient_flow`)**:
   - Verify all 3 branches in `train()` mode receive non-zero gradients.
5. **Backward Compatibility & Parameter Count (`test_backward_compat_and_params`)**:
   - When `use_reparam=False`, architecture and parameters must match baseline $M_0$ exactly.
   - Post `switch_to_deploy()`, total parameter count must match baseline $M_0$.

---

## 5. Non-Functional Constraints
- Pure PyTorch 2.x implementation (no external C++/CUDA custom extensions).
- Full compatibility with BF16, FP16, and TF32 mixed precision.
- Zero latency overhead and zero memory overhead during inference.
