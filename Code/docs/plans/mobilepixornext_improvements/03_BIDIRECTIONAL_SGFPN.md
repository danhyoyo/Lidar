# Pillar 3: Bi-directional Scale-Gated Neck (Bi-SGFPN)

## 1. Motivation & Problem Statement

In the baseline **MobilePixorNeXt**, the feature pyramid neck operates in a strictly **unidirectional top-down** hierarchy ($C_5 \to C_4 \to C_3$). While this injects high-level semantic context from deep layers into higher-resolution feature maps, it suffers from a fundamental limitation in 3D point cloud detection:

1. **Loss of High-Frequency Metric Boundaries**:
   The highest spatial resolution features reside at $C_3$ (stride 4, $200 \times 176$, where each pixel represents a precise $0.4\text{m} \times 0.4\text{m}$ physical area). These low-level layers contain sharp vehicle corners, boundary reflections, and obstacle profiles. In a top-down neck, these fine geometric cues **never flow back up** to refine medium and large scale representations ($C_4$ and $C_5$).
2. **Failure of Naive Bottom-Up Fusion (PANet)**:
   Adding a naive bottom-up path via direct addition or concatenation often harms performance on sparse LiDAR data. High-resolution LiDAR representations contain substantial ground-plane clutter and sensor noise. Direct addition pollutes high-level semantic channels with noisy point clusters.
3. **Overhead of Complex Pyramids (BiFPN)**:
   Standard BiFPN (EfficientDet) employs repetitive multi-stage bidirectional blocks that introduce excessive FLOPs and latency unacceptable for edge deployment.

**Solution**: **Bi-directional Scale-Gated FPN (Bi-SGFPN)**:
A single-iteration, lightweight bidirectional pyramid that couples top-down semantic distribution with bottom-up geometric enrichment. Both pathways are modulated by **learnable adaptive scale gates**, enabling the network to selectively filter spatial noise while propagating crucial corner geometry.

```
                  Top-Down Pathway                      Bottom-Up Pathway
  
      [ C5: 128ch ] ──> [ L5: 48ch ] ─────────────────────────> [ P5_BU: 48ch ]
                             │                                         ▲
                         Interpolate                               Downsample
                             │                                     + BU-Gate5
                             ▼                                         │
      [ C4:  96ch ] ──> [ L4: 48ch ] ──> [ P4_TD: 48ch ] ─────> [ P4_BU: 48ch ]
                             │                 ▲                       ▲
                         Interpolate       TD-Gate4                Downsample
                             │                 │                   + BU-Gate4
                             ▼                 │                       │
      [ C3:  48ch ] ──> [ L3: 24ch ] ──────────┴──────────────> [ P3_final: 24ch ]
                                                                       │
                                                                   [ Conv 3x3 ]
                                                                       │
                                                                       ▼
                                                             Output to Detection Header
                                                            (16 channels @ stride 4)
```

---

## 2. Mathematical Formulation & Gating Mechanism

### 2.1. Lateral Feature Projection
The backbone multi-scale feature maps $\{C_3, C_4, C_5\}$ are mapped to standard neck channel dimensions ($24, 48, 48$) via $1\times 1$ convolutions:
$$L_3 = \text{Conv}_{1\times 1}(C_3), \quad L_4 = \text{Conv}_{1\times 1}(C_4), \quad L_5 = \text{Conv}_{1\times 1}(C_5)$$

### 2.2. Top-Down Pathway with Scale-Gating
1. **At Level 4 ($100 \times 88$)**:
   Upsample $L_5$ via bilinear interpolation followed by depthwise refinement:
   $$U_4 = \text{DWConv}_{3\times 3}(\text{Interp}_{2\times}(L_5))$$
   The top-down scale gate $G_4^{\text{TD}} \in [0, 2]$ evaluates feature compatibility:
   $$G_4^{\text{TD}} = 2 \cdot \sigma(\text{Conv}_{3\times 3}(L_4 + U_4))$$
   Fused intermediate feature:
   $$P_4^{\text{TD}} = U_4 + G_4^{\text{TD}} \odot L_4$$

2. **At Level 3 ($200 \times 176$)**:
   $$U_3 = \text{Interp}_{2\times}(P_4^{\text{TD}})$$
   $$G_3^{\text{TD}} = 2 \cdot \sigma(\text{Conv}_{3\times 3}(L_3 + U_3))$$
   $$P_3 = U_3 + G_3^{\text{TD}} \odot L_3$$

### 2.3. Bottom-Up Pathway with Geometric Reinforcement
Now, fine metric geometry from $P_3$ is propagated upwards to reinforce $P_4$ and $P_5$:

1. **Reinforcing Level 4**:
   Downsample $P_3$ using a strided $3\times 3$ convolution:
   $$D_4 = \text{Conv}_{3\times 3, s=2}(P_3)$$
   The bottom-up scale gate dynamically assesses whether the low-level geometry represents a valid object or background clutter:
   $$G_4^{\text{BU}} = 2 \cdot \sigma(\text{Conv}_{3\times 3}(P_4^{\text{TD}} + D_4))$$
   With direct cross-scale skip connection from lateral $L_4$:
   $$P_4^{\text{BU}} = P_4^{\text{TD}} + G_4^{\text{BU}} \odot D_4 + 0.5 \cdot L_4$$

2. **Reinforcing Level 5**:
   $$D_5 = \text{Conv}_{3\times 3, s=2}(P_4^{\text{BU}})$$
   $$G_5^{\text{BU}} = 2 \cdot \sigma(\text{Conv}_{3\times 3}(L_5 + D_5))$$
   $$P_5^{\text{BU}} = L_5 + G_5^{\text{BU}} \odot D_5$$

### 2.4. Zero-Initialization Stability Guarantee
All gating convolution weights and biases are initialized to **zero**:
$$\text{Gate} = 2 \cdot \sigma(0) = 2 \cdot 0.5 = 1.0$$
This guarantees that at epoch 0, the model behaves as a standard balanced residual network without chaotic amplification or suppression.

---

## 3. Implementation Blueprint

### 3.1. File Location
Create: `detector/core/models/backbones/bi_sgfpn.py`

### 3.2. Class `BiScaleGatedFPN`

```python
class BiScaleGatedFPN(nn.Module):
    def __init__(
        self,
        in_channels: List[int] = [48, 96, 128],  # C3, C4, C5
        out_channels: int = 16,
        lateral_channels: List[int] = [24, 48, 48]
    ):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels
        l3_ch, l4_ch, l5_ch = lateral_channels
        
        # 1. Lateral Projections
        self.lat_c5 = nn.Conv2d(c5_in, l5_ch, 1, bias=False)
        self.lat_c4 = nn.Conv2d(c4_in, l4_ch, 1, bias=False)
        self.lat_c3 = nn.Conv2d(c3_in, l3_ch, 1, bias=False)
        
        # 2. Top-Down Path
        self.u4_refine = nn.Conv2d(l5_ch, l4_ch, 3, padding=1, groups=min(l4_ch, l5_ch), bias=False)
        self.gate_td4 = self._make_gate_conv(l4_ch)
        
        self.u3_proj = nn.Conv2d(l4_ch, l3_ch, 1, bias=False)
        self.gate_td3 = self._make_gate_conv(l3_ch)
        
        # 3. Bottom-Up Path
        self.d4_conv = nn.Conv2d(l3_ch, l4_ch, 3, stride=2, padding=1, bias=False)
        self.gate_bu4 = self._make_gate_conv(l4_ch)
        
        self.d5_conv = nn.Conv2d(l4_ch, l5_ch, 3, stride=2, padding=1, bias=False)
        self.gate_bu5 = self._make_gate_conv(l5_ch)
        
        # 4. Final Aggregation to Detection Header (Stride 4)
        self.out_conv = nn.Sequential(
            nn.Conv2d(l3_ch, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
    def _make_gate_conv(self, channels: int) -> nn.Conv2d:
        conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv
        
    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> torch.Tensor:
        # Implementation of Section 2 equations
        ...
```

---

## 4. Verification & Testing Protocol

### Test File: `tests/test_bi_sgfpn.py`

1. **Shape Consistency & Stride Verification**:
   Input tensors:
   - $C_3: (B, 48, 200, 176)$
   - $C_4: (B, 96, 100, 88)$
   - $C_5: (B, 128, 50, 44)$
   Output tensor must strictly be $(B, 16, 200, 176)$.
2. **Bidirectional Gradient Flow Check**:
   Compute loss $\mathcal{L} = \text{mean}(Y)$. Backpropagate and verify that `.grad` is non-zero and non-NaN on all lateral weights, top-down gates, and bottom-up gates simultaneously.
3. **Resource & Latency Constraint**:
   - Parameter addition: $\le 90\text{k}$ parameters.
   - Latency increase over unidirectional SG-FPN: $\le 0.4\text{ ms}$ on RTX 3060.
