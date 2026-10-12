# Pillar 2: Robust Multi-Scale LiteMLA Attention

## 1. Motivation & Problem Statement

In **MobilePixorNeXt**, LiteMLA (Lightweight Multi-Head Linear Attention) was integrated at Stage 3 ($C_3$, spatial resolution $100 \times 88 = 8,800$ tokens) to provide global receptive fields with linear computational complexity $\mathcal{O}(N)$ instead of quadratic $\mathcal{O}(N^2)$.

However, two significant challenges have been observed during empirical training and low-precision deployment:

1. **Fixed Single-Scale Aggregation Bottleneck**:
   Current LiteMLA applies a single $5\times 5$ depthwise convolution to generate regional context before computing $Q, K, V$.
   - **For Small Objects (Pedestrians/Cyclists)**: At $0.1\text{m}$ BEV resolution, a pedestrian occupies only $\approx 6 \times 8$ cells. A $5\times 5$ kernel over-smoothes fine boundary features against background noise.
   - **For Large Objects (Trucks/Buses)**: A vehicle extending $12-15\text{m}$ ($120-150$ cells) requires broader regional guidance to associate disconnected point clusters.

2. **Numerical Instability in Mixed Precision (BF16 / FP16)**:
   In linear attention:
   $$O = \frac{Q (K^T V)}{\sum_j Q_i \cdot \sum_k K^T_{j, k}}$$
   When accumulated across $N = 8,800$ tokens, the unconstrained dot-products $K^T V$ can grow large. Under BF16 (which has only 8 bits of mantissa precision) or FP16 (limited dynamic range), unnormalized $Q$ and $K$ values can trigger gradient explosions, NaN losses, or loss of small gradient signals.

**Solution**: **Robust Multi-Scale LiteMLA (RMS-LiteMLA)**:
- **Multi-Scale Regional Aggregation**: Parallel depthwise branches with kernel scales $(3\times 3, 5\times 5)$ covering multi-scale objects.
- **$QK$-RMSNorm**: Bounded normalization on query and key vectors before dot-product accumulation, guaranteeing mathematical stability under BF16/FP16.
- **FP32 Linear Accumulation Kernel**: Ensures high precision for the $K^T V$ outer product while running the rest of the network at BF16.

```
                         Input Feature Map X (C x H x W)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
             Conv 3x3 DW (s=1)                     Conv 5x5 DW (s=1)
             (Fine pedestrian)                     (Standard vehicle)
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                               Fused Context Map
                                       │
                      ┌────────────────┼────────────────┐
                      ▼                ▼                ▼
                    Query Q          Key K           Value V
                      │                │                │
                  [ Q-Norm ]       [ K-Norm ]           │
                      │                │                │
                      └───────┬────────┘                │
                              ▼                         │
                      Linear Attention <────────────────┘
                     O = Q * (K^T * V)
                              │
                              ▼
                     1x1 Conv + LayerScale
                              │
                              ▼
                       Residual Output
```

---

## 2. Mathematical Formulation

### 2.1. Multi-Scale Context Aggregation

Given input $X \in \mathbb{R}^{B \times C \times H \times W}$ (where $C=96$ at Stage 3):
$$X_3 = \text{DWConv}_{3\times 3}(X)$$
$$X_5 = \text{DWConv}_{5\times 5}(X)$$
$$X_{\text{fused}} = \text{PWConv}_{1\times 1}(\text{Concat}(X_3, X_5))$$

This dual-branch structure simultaneously retains fine-grained localized edge boundaries ($3\times 3$) and captures broader regional vehicle geometry ($5\times 5$) with minimal FLOPs.

### 2.2. $QK$-RMSNorm for Bounded Linear Attention

For each attention head $h \in \{1, \dots, H\}$ with dimension $d = C / H$:
$$Q_h \in \mathbb{R}^{N \times d}, \quad K_h \in \mathbb{R}^{N \times d}, \quad V_h \in \mathbb{R}^{N \times d}$$

To prevent divergence, apply Root Mean Square Normalization (RMSNorm) along the channel dimension $d$:
$$Q'_h = \frac{Q_h}{\sqrt{\frac{1}{d} \sum_{i=1}^d Q_{h, i}^2 + \epsilon}} \odot \gamma_Q$$
$$K'_h = \frac{K_h}{\sqrt{\frac{1}{d} \sum_{i=1}^d K_{h, i}^2 + \epsilon}} \odot \gamma_K$$

Where $\gamma_Q, \gamma_K \in \mathbb{R}^d$ are learnable affine parameters initialized to $1.0$, and $\epsilon = 10^{-6}$.

### 2.3. Safe FP32 Linear Accumulation

Even when the network runs in BF16 autocast, the attention reduction is executed in float32:
```python
with torch.cuda.amp.autocast(enabled=False):
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()
    
    # 1. Compute global key-value summary: (B, H, d_k, d_v)
    kv = torch.einsum('b h n k, b h n v -> b h k v', k_f32, v_f32)
    
    # 2. Compute normalizer: sum(K, dim=N)
    k_sum = k_f32.sum(dim=2)  # (B, H, d_k)
    normalizer = torch.einsum('b h n k, b h k -> b h n', q_f32, k_sum).unsqueeze(-1) + 1e-6
    
    # 3. Compute output: (B, H, N, d_v)
    out = torch.einsum('b h n k, b h k v -> b h n v', q_f32, kv) / normalizer
```

---

## 3. Implementation Blueprint

### 3.1. File Location
Create: `detector/core/models/backbones/ms_litemla.py`

### 3.2. Class `RobustMultiScaleLiteMLA`

```python
class RobustMultiScaleLiteMLA(nn.Module):
    def __init__(
        self,
        channels: int = 96,
        head_dim: int = 16,
        scales: Tuple[int, ...] = (3, 5),
        layer_scale_init: float = 0.01,
        use_qk_norm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.channels = channels
        self.head_dim = head_dim
        self.num_heads = channels // head_dim
        self.scales = scales
        self.use_qk_norm = use_qk_norm
        
        # Multi-scale regional depthwise branches
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2, groups=channels, bias=False)
            for k in scales
        ])
        self.scale_fuse = nn.Conv2d(channels * len(scales), channels, kernel_size=1, bias=False)
        self.scale_bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        
        # QKV Projections
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        
        # QK Normalization
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
            
        # Out projection & LayerScale
        self.proj = nn.Linear(channels, channels, bias=False)
        self.proj_bn = nn.BatchNorm2d(channels)
        self.layer_scale = nn.Parameter(layer_scale_init * torch.ones(channels, 1, 1))
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
```

---

## 4. Verification & Testing Protocol

### Test File: `tests/test_ms_litemla.py`

1. **Shape & Forward Invariance**:
   Verify for arbitrary input batch sizes and dimensions $(B, 96, 100, 88)$ that output tensor shape matches $(B, 96, 100, 88)$.
2. **BF16 Numerical Stability Stress Test**:
   - Feed inputs containing extreme values ($\pm 10^3$ and $\pm 10^{-4}$).
   - Execute forward and backward under `torch.cuda.amp.autocast(dtype=torch.bfloat16)`.
   - Assert: Zero `NaN` or `Inf` in outputs or parameter gradients.
3. **Latency Profiling**:
   - Compare single-scale $(5)$ vs multi-scale $(3, 5)$.
   - Added latency must be $\le 0.08\text{ ms}$ on RTX 3060.
