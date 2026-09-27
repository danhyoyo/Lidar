# Pillar 1: Structural Reparameterization (Rep-MobilePixorNeXt)

## 1. Motivation & Problem Statement

In 3D LiDAR Object Detection in Bird's Eye View (BEV), bounding boxes span significant physical metric dimensions (e.g. cars are $4.5\text{m} \times 1.8\text{m} \approx 45 \times 18$ cells at $0.1\text{m}$ resolution; trucks/buses exceed $100$ cells). Consequently, large depthwise kernels ($7\times 7$, covering $0.7\text{m} \times 0.7\text{m}$ per step) are essential to prevent feature fragmentation.

However, standard architectures face a fundamental **training-inference dilemma**:
1. **At Training Time**: Pure $7\times 7$ convolutions suffer from diffuse gradient dynamics during early training epochs. The central anchor cell does not receive strong differential gradient focus compared to a tight $3\times 3$ kernel, leading to slower convergence and suboptimal edge localization.
2. **At Inference Time**: Multi-branch designs (like Inception or DBB) provide rich gradient paths during training, but introduce severe GPU memory fragmentation, non-coalesced memory access, and kernel launch overhead that destroys edge throughput (FPS).

**Solution**: **Structural Reparameterization**. We decouple the training-time architecture from the inference-time architecture:
- **Training Topology**: Multi-branch depthwise block ($7\times 7$ DW + $3\times 3$ DW + Identity, each with its own BatchNorm).
- **Inference Topology**: A single equivalent $7\times 7$ depthwise convolution with fused bias. **Zero additional latency, zero extra memory at deployment.**

```
       [Training Topology]                                 [Inference Topology]
      Input Feature Map X                                  Input Feature Map X
        ┌──────┼──────┐                                             │
        │      │      │                                             │
      7x7 DW 3x3 DW Identity                                   Single 7x7 DW
      + BN   + BN   + BN                                    (Fused W_total, b_total)
        │      │      │        === Reparameterization ==>           │
        └──────┼──────┘              [deploy()]                     │
               ▼                                                    ▼
             Add (+)                                            Output Y
               │
               ▼
            Output Y
```

---

## 2. Mathematical Derivation of Exact Kernel Fusion

Let the input feature map be $X \in \mathbb{R}^{B \times C \times H \times W}$. In a depthwise convolution, each channel $c \in \{1, \dots, C\}$ is convolved independently.

### 2.1. Fusing Convolution with BatchNorm

A convolution followed by BatchNorm computes:
$$Y = \text{BN}(\text{Conv}(X; W)) = \gamma \cdot \frac{W * X - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

By linearity, this can be rewritten as a single convolution with fused weight $W'$ and bias $b'$:
$$W'_{c, :, :, :} = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon_c}} W_{c, :, :, :}$$
$$b'_c = \beta_c - \frac{\gamma_c \mu_c}{\sqrt{\sigma_c^2 + \epsilon_c}}$$

### 2.2. Transforming Branch Kernels to Equivalent $7\times 7$ Footprint

1. **Branch 1 ($7\times 7$ DW + BN)**:
   - Kernel $W_7 \in \mathbb{R}^{C \times 1 \times 7 \times 7}$.
   - Fuses directly into $W'_7 \in \mathbb{R}^{C \times 1 \times 7 \times 7}$ and $b'_7 \in \mathbb{R}^C$.

2. **Branch 2 ($3\times 3$ DW + BN)**:
   - Kernel $W_3 \in \mathbb{R}^{C \times 1 \times 3 \times 3}$.
   - Fuses into $W'_3 \in \mathbb{R}^{C \times 1 \times 3 \times 3}$ and $b'_3 \in \mathbb{R}^C$.
   - Pad $W'_3$ with zero-padding of width 2 on all 4 spatial borders to match the $7\times 7$ shape:
     $$\widetilde{W}'_3 = \text{pad}(W'_3, (2, 2, 2, 2)) \in \mathbb{R}^{C \times 1 \times 7 \times 7}$$

3. **Branch 3 (Identity + BN)**:
   - The identity mapping $X$ is mathematically equivalent to convolving $X$ with a Dirac delta (Kronecker) kernel $W_{\text{id}} \in \mathbb{R}^{C \times 1 \times 7 \times 7}$:
     $$W_{\text{id}, c, 0, i, j} = \begin{cases} 1.0 & \text{if } i = 3 \text{ and } j = 3 \\ 0.0 & \text{otherwise} \end{cases}$$
   - Fused with its BatchNorm parameters $(\gamma_{\text{id}}, \beta_{\text{id}}, \mu_{\text{id}}, \sigma^2_{\text{id}}, \epsilon_{\text{id}})$:
     $$W'_{\text{id}, c, :, :, :} = \frac{\gamma_{\text{id}, c}}{\sqrt{\sigma^2_{\text{id}, c} + \epsilon_{\text{id}, c}}} W_{\text{id}, c, :, :, :}$$
     $$b'_{\text{id}, c} = \beta_{\text{id}, c} - \frac{\gamma_{\text{id}, c} \mu_{\text{id}, c}}{\sqrt{\sigma^2_{\text{id}, c} + \epsilon_{\text{id}, c}}}$$

### 2.3. Exact Linear Summation

Because convolution is a linear operator:
$$\text{Conv}(X; W_7) + \text{Conv}(X; \widetilde{W}_3) + \text{Conv}(X; W_{\text{id}}) = \text{Conv}(X; W_{\text{deploy}}) + b_{\text{deploy}}$$

Where:
$$\mathbf{W}_{\text{deploy}} = W'_7 + \widetilde{W}'_3 + W'_{\text{id}}$$
$$\mathbf{b}_{\text{deploy}} = b'_7 + b'_3 + b'_{\text{id}}$$

The deployed layer is simply:
```python
self.fused_conv = nn.Conv2d(
    in_channels=channels,
    out_channels=channels,
    kernel_size=7,
    padding=3,
    groups=channels,
    bias=True
)
self.fused_conv.weight.data.copy_(W_deploy)
self.fused_conv.bias.data.copy_(b_deploy)
```

---

## 3. Component Architecture & Implementation Blueprint

### 3.1. File Location
Create: `detector/core/models/backbones/rep_blocks.py`

### 3.2. Class `RepConv7x7` Interface

```python
class RepConv7x7(nn.Module):
    """
    Structural Reparameterization 7x7 Depthwise Convolution.
    
    Training:
        Branch 1: 7x7 Depthwise Conv + BatchNorm
        Branch 2: 3x3 Depthwise Conv + BatchNorm
        Branch 3: Identity + BatchNorm
        Output = Branch1(X) + Branch2(X) + Branch3(X)
        
    Inference (after switch_to_deploy()):
        Single 7x7 Depthwise Conv (bias=True)
    """
    def __init__(self, channels: int, deploy: bool = False):
        super().__init__()
        self.channels = channels
        self.deploy = deploy
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(
                channels, channels, kernel_size=7, stride=1, padding=3, groups=channels, bias=True
            )
        else:
            self.rbr_conv7 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, groups=channels, bias=False),
                nn.BatchNorm2d(channels)
            )
            self.rbr_conv3 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels)
            )
            self.rbr_identity = nn.BatchNorm2d(channels)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.rbr_reparam(x)
        return self.rbr_conv7(x) + self.rbr_conv3(x) + self.rbr_identity(x)
        
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Implementation of Section 2 mathematical fusion
        ...
        
    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.channels, self.channels, kernel_size=7, stride=1, padding=3, groups=self.channels, bias=True
        )
        self.rbr_reparam.weight.data.copy_(kernel)
        self.rbr_reparam.bias.data.copy_(bias)
        self.__delattr__('rbr_conv7')
        self.__delattr__('rbr_conv3')
        self.__delattr__('rbr_identity')
        self.deploy = True
```

### 3.3. Class `RepMobilePixorNeXtBlock`

Replaces the depthwise 7x7 convolution in the standard `MobilePixorNeXtBlock` with `RepConv7x7`:

```
Input X ──┬──> [ RepConv7x7 ] ──> [ BN ] ──> [ 1x1 PW Expand (2.5x) ] ──> [ SiLU ]
          │                                                                      │
          │                                                                      ▼
          │                                                         [ 1x1 PW Project ]
          │                                                                      │
          │                                                                      ▼
          │                                                                [ LayerScale ]
          │                                                                      │
          └───────────────────────────── ( + ) <─────────────────────────────────┘
                                          │
                                          ▼
                                       Output
```

---

## 4. Verification Protocol & Unit Tests

### Test File: `tests/test_rep_block.py`

1. **Numerical Equivalence Test**:
   - Initialize `RepConv7x7(channels=48, deploy=False)`.
   - Set to `eval()` mode.
   - Run forward pass on random tensor: $Y_{\text{train}} = \text{block}(X)$.
   - Call `block.switch_to_deploy()`.
   - Run forward pass on identical tensor: $Y_{\text{deploy}} = \text{block}(X)$.
   - Assert: `torch.max(torch.abs(Y_train - Y_deploy)).item() < 1e-5`.

2. **Full Model Equivalence Test**:
   - Instantiate `CustomModel` with `use_reparam=True`.
   - In `eval()` mode, verify full network outputs match before and after `model.switch_to_deploy()`.

3. **Inference Latency Benchmark**:
   - Compare throughput before and after `switch_to_deploy()`. Expectation: **$\approx 20-30\%$ faster backbone forward time** post-fusion due to eliminating multi-branch memory stalls.
