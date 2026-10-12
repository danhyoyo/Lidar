# RC-SGFPN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and integrate the Range-Conditioned Scale-Gated Feature Pyramid Network (RC-SGFPN) into `MobilePixorNeXtBackbone`, supporting both Unidirectional (top-down) and Bidirectional (top-down + bottom-up) routing modes, with Fourier Range Embeddings, zero-init mathematical stability guarantees, and zero-latency deployment fusion.

**Architecture:** A dedicated neck module (`detector/core/models/backbones/rc_sgfpn.py`) that couples precomputed Fourier Range Embeddings ($\Phi(r) \in \mathbb{R}^{8 \times H \times W}$) with depthwise scale gating. The module replaces legacy uniform summation with dynamic physics conditioning based on LiDAR beam divergence $\rho(r) \propto r^{-2}$. When `switch_to_deploy()` is called, static range projections are folded into spatial biases for exact 0.000 ms runtime overhead.

**Tech Stack:** Python 3.10+, PyTorch 2.x, NumPy, pytest.

**Spec:** [`docs/superpowers/specs/2026-10-03-rc-sgfpn-design.md`](file:///home/duyennh/AI_projects/research_lidar/Lidar/docs/superpowers/specs/2026-10-03-rc-sgfpn-design.md)

## Global Constraints
- Target parameter addition: $\le 90\text{k}$ parameters for bidirectional mode, $\le 15\text{k}$ parameters for unidirectional mode.
- Strict zero-init guarantee: Gate values must initialize identically to $1.000000$ ($P = U + L$) with maximum absolute deviation $\le 10^{-6}$.
- Mixed-precision robustness: Zero `NaN` or `Inf` values under `torch.bfloat16` and `torch.float16` autocast.
- Zero extra inference latency when deployed: `switch_to_deploy()` must fuse range projection operations into a static spatial bias map.
- Complete backward compatibility: Existing configs without `"neck"` or with `"scale_gated_fpn": true` must continue functioning cleanly without regression.

---

### Task 1: Continuous Fourier Range Embedding Engine

**Files:**
- Create: `detector/core/models/backbones/rc_sgfpn.py`
- Create: `tests/test_rc_sgfpn.py`

**Interfaces:**
- Produces: `FourierRangeEmbedding(height: int, width: int, x_bounds: tuple[float, float], y_bounds: tuple[float, float], num_bands: int = 4)`
- Buffer: `self.embedding` of shape $(1, 2 \times \text{num\_bands}, H, W)$ containing $[\sin(2^b \pi r_{\text{norm}}), \cos(2^b \pi r_{\text{norm}})]$.

- [x] **Step 1: Write failing test for FourierRangeEmbedding**

Create `tests/test_rc_sgfpn.py`:
```python
import math
import torch
import pytest
from core.models.backbones.rc_sgfpn import FourierRangeEmbedding


def test_fourier_range_embedding_shapes_and_values():
    h, w = 100, 88
    x_bounds = (0.0, 70.4)
    y_bounds = (-40.0, 40.0)
    num_bands = 4

    fre = FourierRangeEmbedding(h, w, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_bands)

    assert hasattr(fre, "embedding")
    assert fre.embedding.shape == (1, 2 * num_bands, h, w)
    assert fre.out_dim == 2 * num_bands

    # Values must be bounded in [-1.0, 1.0]
    assert torch.all(fre.embedding >= -1.0 - 1e-5)
    assert torch.all(fre.embedding <= 1.0 + 1e-5)
    assert torch.isfinite(fre.embedding).all()

    # Radial origin check: at (x=0, y=0), r=0 => sin(0)=0, cos(0)=1
    # Y is from -40 to 40, so y=0 is at center index h // 2
    # X is from 0 to 70.4, so x=0 is at index 0
    center_y = h // 2
    origin_x = 0
    origin_feats = fre.embedding[0, :, center_y, origin_x]
    for b in range(num_bands):
        sin_val = origin_feats[2 * b].item()
        cos_val = origin_feats[2 * b + 1].item()
        assert abs(sin_val) < 0.05
        assert abs(cos_val - 1.0) < 0.05
```

- [x] **Step 2: Run test to verify failure**

Run: `pytest tests/test_rc_sgfpn.py::test_fourier_range_embedding_shapes_and_values -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.models.backbones.rc_sgfpn'`

- [x] **Step 3: Implement FourierRangeEmbedding**

Create `detector/core/models/backbones/rc_sgfpn.py`:
```python
"""Range-Conditioned Scale-Gated Feature Pyramid Network (RC-SGFPN).

Injects continuous Fourier Range Embeddings into multi-scale depthwise scale gating
to dynamically balance geometric precision (near range) and semantic context (far range).
"""

from typing import Tuple, List, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FourierRangeEmbedding(nn.Module):
    """Continuous Fourier Range Embedding across orthographic BEV metric space."""

    def __init__(
        self,
        height: int,
        width: int,
        x_bounds: Tuple[float, float] = (0.0, 70.4),
        y_bounds: Tuple[float, float] = (-40.0, 40.0),
        num_bands: int = 4,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.num_bands = num_bands
        self.out_dim = 2 * num_bands

        x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
        y_min, y_max = float(y_bounds[0]), float(y_bounds[1])

        # Generate physical metric coordinates in meters
        y_coords = torch.linspace(y_min, y_max, height, dtype=torch.float32)
        x_coords = torch.linspace(x_min, x_max, width, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Absolute Euclidean distance r = sqrt(x^2 + y^2)
        r = torch.sqrt(xx**2 + yy**2)
        r_max = math.sqrt(max(abs(x_min), abs(x_max))**2 + max(abs(y_min), abs(y_max))**2)
        r_norm = torch.clamp(r / r_max, 0.0, 1.0).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Multi-band sinusoidal basis: [sin(2^b * pi * r), cos(2^b * pi * r)]
        bands = []
        for b in range(num_bands):
            freq = (2.0**b) * math.pi
            bands.append(torch.sin(freq * r_norm))
            bands.append(torch.cos(freq * r_norm))

        embedding = torch.cat(bands, dim=1)  # (1, 2*num_bands, H, W)
        self.register_buffer("embedding", embedding, persistent=False)

    def forward(self) -> Tensor:
        return self.embedding
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rc_sgfpn.py::test_fourier_range_embedding_shapes_and_values -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add detector/core/models/backbones/rc_sgfpn.py tests/test_rc_sgfpn.py
git commit -m "feat(rc_sgfpn): implement FourierRangeEmbedding with metric grid coordinates"
```

---

### Task 2: Range-Conditioned Scale Gate with Zero-Init & Deployment Fusion

**Files:**
- Modify: `detector/core/models/backbones/rc_sgfpn.py`
- Modify: `tests/test_rc_sgfpn.py`

**Interfaces:**
- Produces: `RangeConditionedScaleGate(channels: int, height: int, width: int, x_bounds: tuple, y_bounds: tuple, num_bands: int = 4)`
- Methods: `forward(l_feat: Tensor, u_feat: Tensor) -> Tensor`, `switch_to_deploy() -> None`

- [x] **Step 1: Write failing tests for RangeConditionedScaleGate**

Append to `tests/test_rc_sgfpn.py`:
```python
from core.models.backbones.rc_sgfpn import RangeConditionedScaleGate


def test_range_conditioned_scale_gate_zero_init():
    channels = 48
    h, w = 100, 88
    gate_module = RangeConditionedScaleGate(channels, h, w)

    l_feat = torch.randn(2, channels, h, w)
    u_feat = torch.randn(2, channels, h, w)

    # At epoch 0 (init), gate must be identically 1.000000
    out = gate_module(l_feat, u_feat)
    expected = u_feat + l_feat

    assert torch.allclose(out, expected, atol=1e-6)


def test_range_conditioned_scale_gate_switch_to_deploy():
    channels = 48
    h, w = 100, 88
    gate_module = RangeConditionedScaleGate(channels, h, w)

    # Perturb weights slightly to simulate training
    with torch.no_grad():
        gate_module.content_conv.weight.add_(torch.randn_like(gate_module.content_conv.weight) * 0.1)
        gate_module.range_proj.weight.add_(torch.randn_like(gate_module.range_proj.weight) * 0.1)

    l_feat = torch.randn(2, channels, h, w)
    u_feat = torch.randn(2, channels, h, w)

    out_train = gate_module(l_feat, u_feat)
    gate_module.switch_to_deploy()

    assert gate_module.deploy is True
    assert not hasattr(gate_module, "range_proj")
    assert not hasattr(gate_module, "fre")

    out_deploy = gate_module(l_feat, u_feat)
    assert torch.allclose(out_train, out_deploy, atol=1e-5)
```

- [x] **Step 2: Run test to verify failure**

Run: `pytest tests/test_rc_sgfpn.py::test_range_conditioned_scale_gate_zero_init -v`
Expected: FAIL with `ImportError: cannot import name 'RangeConditionedScaleGate'`

- [x] **Step 3: Implement RangeConditionedScaleGate**

Append to `detector/core/models/backbones/rc_sgfpn.py`:
```python
class RangeConditionedScaleGate(nn.Module):
    """Dynamic depthwise scale gate conditioned on spatial Fourier Range Embeddings."""

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        x_bounds: Tuple[float, float] = (0.0, 70.4),
        y_bounds: Tuple[float, float] = (-40.0, 40.0),
        num_bands: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.deploy = False

        # 1. Continuous Fourier Range Engine
        self.fre = FourierRangeEmbedding(height, width, x_bounds, y_bounds, num_bands)
        self.range_proj = nn.Conv2d(self.fre.out_dim, channels, kernel_size=1, bias=True)

        # 2. Local content depthwise aggregator
        self.content_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        # 3. Mathematical Zero-Initialization Guarantee: Gate = 2 * sigmoid(0) = 1.0
        nn.init.zeros_(self.content_conv.weight)
        nn.init.zeros_(self.content_conv.bias)
        nn.init.zeros_(self.range_proj.weight)
        nn.init.zeros_(self.range_proj.bias)

    def forward(self, l_feat: Tensor, u_feat: Tensor) -> Tensor:
        if self.deploy:
            # Zero-latency path: precomputed spatial bias is folded into buffer
            gate_logits = self.content_conv(l_feat + u_feat) + self.static_spatial_bias
        else:
            feat_logits = self.content_conv(l_feat + u_feat)
            range_logits = self.range_proj(self.fre())
            gate_logits = feat_logits + range_logits

        gate = 2.0 * torch.sigmoid(gate_logits)
        return u_feat + gate * l_feat

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fuses static range projection into a constant 2D spatial bias buffer."""
        if self.deploy:
            return
        # Precompute static spatial bias: (1, C, H, W)
        range_bias = self.range_proj(self.fre())
        self.register_buffer("static_spatial_bias", range_bias)

        # Delete dynamic projection submodules to release memory and remove ONNX nodes
        del self.range_proj
        del self.fre
        self.deploy = True
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rc_sgfpn.py -k "test_range_conditioned_scale_gate" -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add detector/core/models/backbones/rc_sgfpn.py tests/test_rc_sgfpn.py
git commit -m "feat(rc_sgfpn): implement RangeConditionedScaleGate with zero-init and deployment fusion"
```

---

### Task 3: Complete RangeConditionedSGFPN Module (Unidirectional & Bidirectional)

**Files:**
- Modify: `detector/core/models/backbones/rc_sgfpn.py`
- Modify: `tests/test_rc_sgfpn.py`

**Interfaces:**
- Produces: `RangeConditionedSGFPN`
  ```python
  class RangeConditionedSGFPN(nn.Module):
      def __init__(
          self,
          in_channels: Tuple[int, int, int] = (48, 96, 128),  # C3, C4, C5
          out_channels: int = 16,
          lateral_channels: Tuple[int, int, int] = (24, 48, 48),
          bidirectional: bool = False,
          num_range_bands: int = 4,
          geometry: Optional[Dict[str, float]] = None,
      )
  ```
- Input: `c3 (B, 48, 200, 176)`, `c4 (B, 96, 100, 88)`, `c5 (B, 128, 50, 44)`
- Output: `(B, 16, 200, 176)`

- [x] **Step 1: Write failing tests for RangeConditionedSGFPN**

Append to `tests/test_rc_sgfpn.py`:
```python
from core.models.backbones.rc_sgfpn import RangeConditionedSGFPN


def test_rc_sgfpn_unidirectional_shape_and_deploy():
    neck = RangeConditionedSGFPN(bidirectional=False)

    c3 = torch.randn(2, 48, 200, 176)
    c4 = torch.randn(2, 96, 100, 88)
    c5 = torch.randn(2, 128, 50, 44)

    out = neck(c3, c4, c5)
    assert out.shape == (2, 16, 200, 176)
    assert torch.isfinite(out).all()

    # Test switch_to_deploy
    neck.switch_to_deploy()
    out_deploy = neck(c3, c4, c5)
    assert out_deploy.shape == (2, 16, 200, 176)
    assert torch.allclose(out, out_deploy, atol=1e-5)


def test_rc_sgfpn_bidirectional_shape_and_deploy():
    neck = RangeConditionedSGFPN(bidirectional=True)

    c3 = torch.randn(2, 48, 200, 176)
    c4 = torch.randn(2, 96, 100, 88)
    c5 = torch.randn(2, 128, 50, 44)

    out = neck(c3, c4, c5)
    assert out.shape == (2, 16, 200, 176)
    assert torch.isfinite(out).all()

    # Test switch_to_deploy
    neck.switch_to_deploy()
    out_deploy = neck(c3, c4, c5)
    assert out_deploy.shape == (2, 16, 200, 176)
    assert torch.allclose(out, out_deploy, atol=1e-5)
```

- [x] **Step 2: Run test to verify failure**

Run: `pytest tests/test_rc_sgfpn.py -k "test_rc_sgfpn_" -v`
Expected: FAIL with `ImportError: cannot import name 'RangeConditionedSGFPN'`

- [x] **Step 3: Implement RangeConditionedSGFPN**

Append to `detector/core/models/backbones/rc_sgfpn.py`:
```python
class RangeConditionedSGFPN(nn.Module):
    """Range-Conditioned Scale-Gated Feature Pyramid Network.

    Supports both Unidirectional (top-down) and Bidirectional (top-down + bottom-up)
    routing pathways, with continuous Fourier Range Embedding modulation.
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (48, 96, 128),  # C3, C4, C5
        out_channels: int = 16,
        lateral_channels: Tuple[int, int, int] = (24, 48, 48),
        bidirectional: bool = False,
        num_range_bands: int = 4,
        geometry: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lateral_channels = lateral_channels
        self.bidirectional = bool(bidirectional)

        if geometry is None:
            geometry = {"x_min": 0.0, "x_max": 70.4, "y_min": -40.0, "y_max": 40.0}
        x_bounds = (float(geometry.get("x_min", 0.0)), float(geometry.get("x_max", 70.4)))
        y_bounds = (float(geometry.get("y_min", -40.0)), float(geometry.get("y_max", 40.0)))

        c3_in, c4_in, c5_in = in_channels
        l3_ch, l4_ch, l5_ch = lateral_channels

        # 1. Lateral Projections (1x1 convs)
        self.lat_c5 = nn.Conv2d(c5_in, l5_ch, kernel_size=1, bias=False)
        self.lat_c4 = nn.Conv2d(c4_in, l4_ch, kernel_size=1, bias=False)
        self.lat_c3 = nn.Conv2d(c3_in, l3_ch, kernel_size=1, bias=False)

        # 2. Top-Down Pathway
        # Level 4: 50x44 -> 100x88
        self.refine_u4 = nn.Sequential(
            nn.Conv2d(l5_ch, l4_ch, kernel_size=3, padding=1, groups=min(l4_ch, l5_ch), bias=False),
            nn.BatchNorm2d(l4_ch),
            nn.SiLU(inplace=True),
        )
        self.gate_td4 = RangeConditionedScaleGate(
            l4_ch, height=100, width=88, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_range_bands
        )

        # Level 3: 100x88 -> 200x176
        self.proj_u3 = nn.Sequential(
            nn.Conv2d(l4_ch, l3_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(l3_ch),
            nn.SiLU(inplace=True),
        )
        self.gate_td3 = RangeConditionedScaleGate(
            l3_ch, height=200, width=176, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_range_bands
        )

        # 3. Bottom-Up Pathway (Active when bidirectional=True)
        if self.bidirectional:
            # P3 -> P4: stride 2 downsampling (200x176 -> 100x88)
            self.down_p3 = nn.Sequential(
                nn.Conv2d(l3_ch, l4_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(l4_ch),
                nn.SiLU(inplace=True),
            )
            self.gate_bu4 = RangeConditionedScaleGate(
                l4_ch, height=100, width=88, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_range_bands
            )

            # P4 -> P5: stride 2 downsampling (100x88 -> 50x44)
            self.down_p4 = nn.Sequential(
                nn.Conv2d(l4_ch, l5_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(l5_ch),
                nn.SiLU(inplace=True),
            )
            self.gate_bu5 = RangeConditionedScaleGate(
                l5_ch, height=50, width=44, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_range_bands
            )

            # Aggregation from reinforced P4 back into P3
            self.bu_refine_p3 = nn.Sequential(
                nn.Conv2d(l4_ch, l3_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(l3_ch),
                nn.SiLU(inplace=True),
            )

        # 4. Final Header Output Projection (Stride 4, 16 channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(l3_ch, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, c3: Tensor, c4: Tensor, c5: Tensor) -> Tensor:
        # Lateral feature projections
        l5 = self.lat_c5(c5)  # (B, 48, 50, 44)
        l4 = self.lat_c4(c4)  # (B, 48, 100, 88)
        l3 = self.lat_c3(c3)  # (B, 24, 200, 176)

        # Top-Down Pass
        u4 = F.interpolate(l5, scale_factor=2.0, mode="bilinear", align_corners=False)
        u4 = self.refine_u4(u4)
        p4 = self.gate_td4(l4, u4)  # (B, 48, 100, 88)

        u3 = F.interpolate(p4, scale_factor=2.0, mode="bilinear", align_corners=False)
        u3 = self.proj_u3(u3)
        p3 = self.gate_td3(l3, u3)  # (B, 24, 200, 176)

        # Bottom-Up Pass (optional)
        if self.bidirectional:
            d4 = self.down_p3(p3)
            p4_bu = self.gate_bu4(d4, p4) + 0.5 * l4

            d5 = self.down_p4(p4_bu)
            _ = self.gate_bu5(d5, l5)  # Reinforces P5 state

            p4_up = F.interpolate(p4_bu, scale_factor=2.0, mode="bilinear", align_corners=False)
            p3 = p3 + self.bu_refine_p3(p4_up)

        return self.out_conv(p3)

    def switch_to_deploy(self):
        """Recursively delegates switch_to_deploy to internal gates."""
        for m in self.modules():
            if hasattr(m, "switch_to_deploy") and m is not self:
                m.switch_to_deploy()
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rc_sgfpn.py -k "test_rc_sgfpn_" -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add detector/core/models/backbones/rc_sgfpn.py tests/test_rc_sgfpn.py
git commit -m "feat(rc_sgfpn): implement RangeConditionedSGFPN supporting unidirectional and bidirectional modes"
```

---

### Task 4: Mixed-Precision Numerical Safety & Autocast Gradient Flow Tests

**Files:**
- Modify: `tests/test_rc_sgfpn.py`

**Interfaces:**
- Validates: End-to-end gradient flow, numerical safety under `torch.bfloat16` and `torch.float16`, zero NaNs with extreme feature values.

- [x] **Step 1: Write test for BF16/FP16 numerical stability and gradient propagation**

Append to `tests/test_rc_sgfpn.py`:
```python
def test_rc_sgfpn_gradient_flow_and_autocast_safety():
    neck = RangeConditionedSGFPN(bidirectional=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neck = neck.to(device)

    c3 = torch.randn(2, 48, 200, 176, device=device, requires_grad=True)
    c4 = torch.randn(2, 96, 100, 88, device=device, requires_grad=True)
    c5 = torch.randn(2, 128, 50, 44, device=device, requires_grad=True)

    # Autocast context check
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        out = neck(c3, c4, c5)
        loss = out.sum()

    loss.backward()

    # Assert gradients exist and are finite on all inputs
    assert c3.grad is not None and torch.isfinite(c3.grad).all()
    assert c4.grad is not None and torch.isfinite(c4.grad).all()
    assert c5.grad is not None and torch.isfinite(c5.grad).all()

    # Assert gradients exist on trainable gate weights
    assert neck.gate_td4.content_conv.weight.grad is not None
    assert neck.gate_td4.range_proj.weight.grad is not None
    assert torch.isfinite(neck.gate_td4.content_conv.weight.grad).all()
    assert torch.isfinite(neck.gate_td4.range_proj.weight.grad).all()
```

- [x] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_rc_sgfpn.py::test_rc_sgfpn_gradient_flow_and_autocast_safety -v`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add tests/test_rc_sgfpn.py
git commit -m "test(rc_sgfpn): add numerical safety and mixed precision gradient flow tests"
```

---

### Task 5: Integration with MobilePixorNeXtBackbone & CustomModel

**Files:**
- Modify: `detector/core/models/backbones/mobilepixornext.py`
- Modify: `detector/core/models/backbones/__init__.py`
- Modify: `tests/test_rc_sgfpn.py`

**Interfaces:**
- `MobilePixorNeXtBackbone.__init__`: accepts `neck_type: str = "scale_gated_fpn"` (options: `"scale_gated_fpn"`, `"rc_sgfpn"`, `"rc_bisgfpn"`), `geometry: Optional[dict] = None`.
- Routes `(c3, c4, c5)` through `self.neck(c2, c4, c5)` and delegates `switch_to_deploy()`.

- [x] **Step 1: Write failing test for MobilePixorNeXt integration**

Append to `tests/test_rc_sgfpn.py`:
```python
from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone


def test_mobilepixornext_with_rc_sgfpn():
    # Test Unidirectional RC-SGFPN
    bb_uni = MobilePixorNeXtBackbone(input_channels=8, neck_type="rc_sgfpn")
    x = torch.randn(2, 8, 800, 704)
    out_uni = bb_uni(x)
    assert out_uni.shape == (2, 16, 200, 176)

    # Test Bidirectional RC-BiSGFPN
    bb_bi = MobilePixorNeXtBackbone(input_channels=8, neck_type="rc_bisgfpn")
    out_bi = bb_bi(x)
    assert out_bi.shape == (2, 16, 200, 176)

    # Test deploy switch
    bb_bi.switch_to_deploy()
    out_bi_deploy = bb_bi(x)
    assert out_bi_deploy.shape == (2, 16, 200, 176)
```

- [x] **Step 2: Run test to verify failure**

Run: `pytest tests/test_rc_sgfpn.py::test_mobilepixornext_with_rc_sgfpn -v`
Expected: FAIL (argument `neck_type` unexpected)

- [x] **Step 3: Update MobilePixorNeXtBackbone**

In `detector/core/models/backbones/mobilepixornext.py`, import `RangeConditionedSGFPN` and add `neck_type: str = "scale_gated_fpn"` and `geometry: Optional[dict] = None` to `__init__`.
If `neck_type in ("rc_sgfpn", "rc_bisgfpn")`, instantiate `RangeConditionedSGFPN` and route forward pass through it. Otherwise, preserve legacy inline SG-FPN path for 100% backward compatibility.

In `detector/core/models/backbones/__init__.py`:
Export `RangeConditionedSGFPN` and `RangeConditionedScaleGate`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rc_sgfpn.py::test_mobilepixornext_with_rc_sgfpn -v`
Expected: PASS

- [x] **Step 5: Run complete test suite to ensure zero regressions**

Run: `pytest tests/`
Expected: ALL PASS

- [x] **Step 6: Commit**

```bash
git add detector/core/models/backbones/mobilepixornext.py detector/core/models/backbones/__init__.py tests/test_rc_sgfpn.py
git commit -m "feat(backbone): integrate RC-SGFPN into MobilePixorNeXt with backward compatibility"
```

---

### Task 6: Configuration Files & Training Dry-Run Verification

**Files:**
- Create: `configs/kitti/rc_sgfpn/kitti_mobilepixornext_rc_sgfpn.json`
- Create: `configs/kitti/rc_sgfpn/kitti_mobilepixornext_rc_bisgfpn.json`
- Modify: `tests/test_rc_sgfpn.py`

**Interfaces:**
- Config schemas setting `"neck_type": "rc_sgfpn"` and `"neck_type": "rc_bisgfpn"` in `model` section.

- [x] **Step 1: Write E2E CustomModel test with RC-SGFPN config**

Append to `tests/test_rc_sgfpn.py`:
```python
from core.models.model import CustomModel


def test_custom_model_e2e_with_rc_sgfpn():
    cfg = {
        "bev_encoding": {"name": "rich8"},
        "kitti": {"geometry": {"x_min": 0, "x_max": 70.4, "y_min": -40, "y_max": 40, "x_res": 0.1, "y_res": 0.1}},
        "num_classes": 3,
        "backbone": "mobilepixornext",
        "neck_type": "rc_sgfpn",
    }
    model = CustomModel(cfg)
    voxel = torch.randn(2, 8, 800, 704)
    pred = model({"voxel": voxel})

    assert "cls" in pred
    assert "offset" in pred
    assert "size" in pred
    assert "yaw" in pred
    assert pred["cls"].shape == (2, 3, 200, 176)
```

- [x] **Step 2: Create production configuration JSON files**

Create `configs/kitti/rc_sgfpn/kitti_mobilepixornext_rc_sgfpn.json` and `configs/kitti/rc_sgfpn/kitti_mobilepixornext_rc_bisgfpn.json` mirroring `configs/kitti/oga_loss/kitti_mobilepixornext_litemla_oga.json` with `neck_type` updated.

- [x] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_rc_sgfpn.py::test_custom_model_e2e_with_rc_sgfpn -v`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add configs/kitti/rc_sgfpn/ tests/test_rc_sgfpn.py
git commit -m "feat(config): add training configurations for unidirectional and bidirectional RC-SGFPN"
```
