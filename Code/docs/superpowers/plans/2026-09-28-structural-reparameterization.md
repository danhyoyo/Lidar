# Structural Reparameterization for MobilePixorNeXt (Pillar 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Structural Reparameterization (`Rep-MobilePixorNeXt` / Pillar 1) for the MobilePixorNeXt LiDAR backbone, decoupling multi-branch training ($7\times 7\text{ DW} + 3\times 3\text{ DW} + \text{Identity}$) from single-kernel inference ($7\times 7\text{ DW}$) with zero latency overhead.

**Architecture:**
- Create a dedicated algebraic fusion module `RepConv7x7` in `detector/core/models/backbones/rep_blocks.py` that computes equivalent kernel and bias ($W_{\text{deploy}}, b_{\text{deploy}}$) fusing Conv and BatchNorm parameters.
- Equip `MobilePixorNeXtBlock` with an optional `use_reparam: bool = False` flag to route depthwise spatial aggregation through `RepConv7x7`.
- Propagate `use_reparam` through `MobilePixorNeXtBackbone`, `registry.py`, and `CustomModel`.
- Provide a global `model.switch_to_deploy()` method that fuses multi-branch weights in place, removes training branches, and reduces deployment model size to baseline $M_0$.

**Tech Stack:** Python 3.14, PyTorch 2.11, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-structural-reparameterization-design.md`

## Global Constraints
- Baseline anchor is strictly **MobilePixorNeXt + OGA Loss** ($M_0$).
- Numerical equivalence: $\max |Y_{\text{multi}} - Y_{\text{deploy}}| < 1\times 10^{-5}$ under `eval()` mode.
- Inference impact: **Exact 0.00 ms** additional inference latency and **0 extra parameters** after `switch_to_deploy()`.
- Backward compatibility: All existing models, checkpoints, and configurations must remain 100% operational when `use_reparam: false` or omitted.
- Execution environment: Run pytest using `PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint`.

---

## Task Breakdown

### Task 1: Linear Algebraic Fusion Primitives & Core `RepConv7x7` Module

**Files:**
- Create: `detector/core/models/backbones/rep_blocks.py`
- Create: `tests/test_rep_block.py`

**Interfaces:**
- Produces: `trans_conv_bn_to_kernel_bias(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]`
- Produces: `trans_identity_bn_to_kernel_bias(bn: nn.BatchNorm2d, channels: int, kernel_size: int = 7) -> Tuple[torch.Tensor, torch.Tensor]`
- Produces: `class RepConv7x7(nn.Module)` with:
  - `__init__(self, channels: int, deploy: bool = False)`
  - `forward(self, x: torch.Tensor) -> torch.Tensor`
  - `get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]`
  - `switch_to_deploy(self)`

- [x] **Step 1: Write failing unit test for `RepConv7x7` numerical equivalence and gradient flow**

Create `tests/test_rep_block.py`:
```python
from typing import Tuple
import pytest
import torch
import torch.nn as nn

from core.models.backbones.rep_blocks import (
    RepConv7x7,
    trans_conv_bn_to_kernel_bias,
    trans_identity_bn_to_kernel_bias,
)


def test_trans_conv_bn_fusion_math():
    channels = 8
    conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
    bn = nn.BatchNorm2d(channels)
    conv.eval()
    bn.eval()

    x = torch.randn(2, channels, 16, 16)
    expected = bn(conv(x))

    kernel, bias = trans_conv_bn_to_kernel_bias(conv, bn)
    fused_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
    fused_conv.weight.data.copy_(kernel)
    fused_conv.bias.data.copy_(bias)
    actual = fused_conv(x)

    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)


def test_trans_identity_bn_fusion_math():
    channels = 8
    bn = nn.BatchNorm2d(channels)
    bn.eval()

    x = torch.randn(2, channels, 16, 16)
    expected = bn(x)

    kernel, bias = trans_identity_bn_to_kernel_bias(bn, channels=channels, kernel_size=7)
    fused_conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=True)
    fused_conv.weight.data.copy_(kernel)
    fused_conv.bias.data.copy_(bias)
    actual = fused_conv(x)

    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)


def test_repconv7x7_numerical_equivalence():
    torch.manual_seed(42)
    channels = 48
    rep = RepConv7x7(channels=channels, deploy=False)
    rep.eval()

    x = torch.randn(2, channels, 50, 44)
    with torch.no_grad():
        out_multi = rep(x)

    assert not rep.deploy
    assert hasattr(rep, "rbr_conv7")
    assert hasattr(rep, "rbr_conv3")
    assert hasattr(rep, "rbr_identity")

    rep.switch_to_deploy()

    assert rep.deploy
    assert hasattr(rep, "rbr_reparam")
    assert not hasattr(rep, "rbr_conv7")
    assert not hasattr(rep, "rbr_conv3")
    assert not hasattr(rep, "rbr_identity")

    with torch.no_grad():
        out_fused = rep(x)

    diff = torch.max(torch.abs(out_multi - out_fused)).item()
    assert diff < 1e-5, f"Max difference {diff} exceeds tolerance 1e-5"


def test_repconv7x7_gradient_flow():
    channels = 16
    rep = RepConv7x7(channels=channels, deploy=False)
    rep.train()

    x = torch.randn(2, channels, 20, 20, requires_grad=True)
    out = rep(x)
    loss = out.sum()
    loss.backward()

    assert rep.rbr_conv7[0].weight.grad is not None
    assert rep.rbr_conv7[1].weight.grad is not None
    assert rep.rbr_conv3[0].weight.grad is not None
    assert rep.rbr_conv3[1].weight.grad is not None
    assert rep.rbr_identity.weight.grad is not None
    assert x.grad is not None
```

- [x] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_rep_block.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'core.models.backbones.rep_blocks'`

- [x] **Step 3: Write implementation in `detector/core/models/backbones/rep_blocks.py`**

Create `detector/core/models/backbones/rep_blocks.py`:
```python
"""Structural Reparameterization Building Blocks for MobilePixorNeXt.

Decouples multi-branch depthwise training (7x7 DW + 3x3 DW + Identity)
into a single equivalent 7x7 depthwise convolution for zero-latency inference.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def trans_conv_bn_to_kernel_bias(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
) -> Tuple[Tensor, Tensor]:
    """Fuse a Conv2d and BatchNorm2d pair into equivalent (kernel, bias)."""
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)
    t = (gamma / std).reshape(-1, 1, 1, 1)

    kernel = conv.weight * t
    if conv.bias is not None:
        bias = beta + (conv.bias - mean) * (gamma / std)
    else:
        bias = beta - mean * (gamma / std)
    return kernel, bias


def trans_identity_bn_to_kernel_bias(
    bn: nn.BatchNorm2d,
    channels: int,
    kernel_size: int = 7,
) -> Tuple[Tensor, Tensor]:
    """Fuse an Identity BatchNorm2d into equivalent depthwise (kernel, bias)."""
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)

    # Depthwise identity kernel (groups=channels) has shape (channels, 1, K, K)
    kernel = torch.zeros((channels, 1, kernel_size, kernel_size), device=bn.weight.device, dtype=bn.weight.dtype)
    center = kernel_size // 2
    for c in range(channels):
        kernel[c, 0, center, center] = 1.0

    t = (gamma / std).reshape(-1, 1, 1, 1)
    kernel = kernel * t
    bias = beta - mean * (gamma / std)
    return kernel, bias


class RepConv7x7(nn.Module):
    """Structural Reparameterization 7x7 Depthwise Convolution.

    Training Topology:
        Branch 1: 7x7 Depthwise Conv + BatchNorm
        Branch 2: 3x3 Depthwise Conv + BatchNorm
        Branch 3: Identity + BatchNorm
        Output = Branch1(X) + Branch2(X) + Branch3(X)

    Inference Topology (after switch_to_deploy()):
        Single 7x7 Depthwise Conv with bias (groups=channels).
    """

    def __init__(self, channels: int, deploy: bool = False):
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")

        self.channels = channels
        self.deploy = deploy

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                channels,
                channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=channels,
                bias=True,
            )
        else:
            self.rbr_conv7 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
            )
            self.rbr_conv3 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
            )
            self.rbr_identity = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            return self.rbr_reparam(x)
        return self.rbr_conv7(x) + self.rbr_conv3(x) + self.rbr_identity(x)

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        """Compute the equivalent single 7x7 DW kernel and bias from the 3 branches."""
        assert not self.deploy, "Already deployed; cannot get equivalent kernel."

        kernel7, bias7 = trans_conv_bn_to_kernel_bias(self.rbr_conv7[0], self.rbr_conv7[1])
        kernel3, bias3 = trans_conv_bn_to_kernel_bias(self.rbr_conv3[0], self.rbr_conv3[1])
        kernel_id, bias_id = trans_identity_bn_to_kernel_bias(self.rbr_identity, self.channels, kernel_size=7)

        # Pad 3x3 kernel by 2 on all borders: (left, right, top, bottom)
        kernel3_padded = F.pad(kernel3, (2, 2, 2, 2))

        kernel_fused = kernel7 + kernel3_padded + kernel_id
        bias_fused = bias7 + bias3 + bias_id
        return kernel_fused, bias_fused

    def switch_to_deploy(self):
        """Fuse all branches into a single 7x7 Conv2d and delete training branches."""
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=self.channels,
            bias=True,
        )
        self.rbr_reparam.weight.data.copy_(kernel)
        self.rbr_reparam.bias.data.copy_(bias)

        self.__delattr__("rbr_conv7")
        self.__delattr__("rbr_conv3")
        self.__delattr__("rbr_identity")
        self.deploy = True
```

- [x] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_rep_block.py -v
```
Expected: PASS (4 tests passed).

- [x] **Step 5: Git commit task 1**

```bash
git add detector/core/models/backbones/rep_blocks.py tests/test_rep_block.py
git commit -m "feat(backbone): implement RepConv7x7 structural reparameterization module"
```

---

### Task 2: Integrate `RepConv7x7` into `MobilePixorNeXtBlock` with Backward Compatibility

**Files:**
- Modify: `detector/core/models/backbones/mobilepixornext_blocks.py:22-80`
- Modify: `tests/test_rep_block.py`

**Interfaces:**
- Consumes: `RepConv7x7` from `core.models.backbones.rep_blocks`
- Produces: `MobilePixorNeXtBlock(channels: int, expansion: float = 2.5, layer_scale_init: float = 1e-5, use_reparam: bool = False)`
- Produces: `MobilePixorNeXtBlock.switch_to_deploy()`

- [x] **Step 1: Write failing unit test for `MobilePixorNeXtBlock` with reparameterization**

Append to `tests/test_rep_block.py`:
```python
from core.models.backbones.mobilepixornext_blocks import MobilePixorNeXtBlock


def test_mobilepixornext_block_backward_compat():
    torch.manual_seed(42)
    block_default = MobilePixorNeXtBlock(48)
    assert not block_default.use_reparam
    assert hasattr(block_default, "dwconv")
    assert hasattr(block_default, "norm1")

    x = torch.randn(2, 48, 20, 20)
    out = block_default(x)
    assert out.shape == (2, 48, 20, 20)


def test_mobilepixornext_block_reparam_equivalence():
    torch.manual_seed(42)
    block = MobilePixorNeXtBlock(48, use_reparam=True)
    block.eval()

    x = torch.randn(2, 48, 20, 20)
    with torch.no_grad():
        out_multi = block(x)

    assert block.use_reparam
    assert hasattr(block, "dw_block")
    assert not block.dw_block.deploy

    block.switch_to_deploy()
    assert block.dw_block.deploy

    with torch.no_grad():
        out_fused = block(x)

    diff = torch.max(torch.abs(out_multi - out_fused)).item()
    assert diff < 1e-5, f"Block diff {diff} exceeds tolerance 1e-5"
```

- [x] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_rep_block.py::test_mobilepixornext_block_reparam_equivalence -v
```
Expected: FAIL with `TypeError: MobilePixorNeXtBlock.__init__() got an unexpected keyword argument 'use_reparam'`

- [x] **Step 3: Modify `detector/core/models/backbones/mobilepixornext_blocks.py`**

In `detector/core/models/backbones/mobilepixornext_blocks.py`:
Import `RepConv7x7`:
```python
from core.models.backbones.rep_blocks import RepConv7x7
```

Update `MobilePixorNeXtBlock.__init__` and `forward`:
```python
class MobilePixorNeXtBlock(nn.Module):
    """Modern BEV building block with 7x7 Depthwise Conv and Inverted Expansion.

    Structure:
        x -> DW-Conv 7x7 (or RepConv7x7) -> BN -> 1x1 PW (C -> e*C) -> SiLU
          -> 1x1 PW (e*C -> C) -> BN -> LayerScale -> (+) -> Output
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 2.5,
        layer_scale_init: float = 1e-5,
        use_reparam: bool = False,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")

        hidden_dim = int(round(channels * expansion))
        self.channels = channels
        self.use_reparam = bool(use_reparam)

        # 1. Spatial aggregation
        if self.use_reparam:
            self.dw_block = RepConv7x7(channels)
        else:
            self.dwconv = nn.Conv2d(
                channels,
                channels,
                kernel_size=7,
                padding=3,
                groups=channels,
                bias=False,
            )
            self.norm1 = nn.BatchNorm2d(channels)

        # 2. Pointwise inverted bottleneck expansion
        self.pw_expand = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=False)

        # 3. Pointwise projection back to input channels
        self.pw_project = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

        # 4. LayerScale for stable training dynamics
        self.layer_scale = _layer_scale(channels, layer_scale_init)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        if self.use_reparam:
            feat = self.dw_block(x)
        else:
            feat = self.norm1(self.dwconv(x))
        feat = self.pw_expand(feat)
        feat = self.act(feat)
        feat = self.pw_project(feat)
        feat = self.norm2(feat)
        return residual + self.layer_scale.to(feat.dtype) * feat

    def switch_to_deploy(self):
        """Delegate reparameterization to internal RepConv7x7."""
        if self.use_reparam and hasattr(self.dw_block, "switch_to_deploy"):
            self.dw_block.switch_to_deploy()
```

- [x] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_rep_block.py -v
```
Expected: PASS (6 tests passed).

- [x] **Step 5: Git commit task 2**

```bash
git add detector/core/models/backbones/mobilepixornext_blocks.py tests/test_rep_block.py
git commit -m "feat(backbone): integrate RepConv7x7 into MobilePixorNeXtBlock"
```

---

### Task 3: Integrate Reparameterization into `MobilePixorNeXtBackbone`, Registry, and `CustomModel`

**Files:**
- Modify: `detector/core/models/backbones/mobilepixornext.py:23-110`
- Modify: `detector/core/models/backbones/registry.py:62-72`
- Modify: `detector/core/models/model.py:7-36`
- Modify: `tests/test_rep_block.py`

**Interfaces:**
- Produces: `MobilePixorNeXtBackbone(..., use_reparam: bool = False)` with `switch_to_deploy()`
- Produces: `CustomModel.switch_to_deploy()`

- [x] **Step 1: Write failing unit test for full model reparameterization flow**

Append to `tests/test_rep_block.py`:
```python
from core.models.model import CustomModel
from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone


def test_mobilepixornext_backbone_reparam_equivalence():
    torch.manual_seed(42)
    backbone = MobilePixorNeXtBackbone(input_channels=8, backbone_out_dim=16, use_reparam=True)
    backbone.eval()

    x = torch.randn(1, 8, 400, 352)
    with torch.no_grad():
        out_multi = backbone(x)

    backbone.switch_to_deploy()

    with torch.no_grad():
        out_fused = backbone(x)

    diff = torch.max(torch.abs(out_multi - out_fused)).item()
    assert diff < 1e-5, f"Backbone diff {diff} exceeds tolerance 1e-5"


def test_custom_model_reparam_flow():
    cfg = {
        "backbone": "mobilepixornext",
        "backbone_out_dim": 16,
        "cls_encoding": "gaussian",
        "use_reparam": True,
    }
    model = CustomModel(cfg, num_classes=3, input_channels=8)
    model.eval()

    x = torch.randn(1, 8, 400, 352)
    with torch.no_grad():
        pred_multi = model(x)

    model.switch_to_deploy()

    with torch.no_grad():
        pred_fused = model(x)

    for head_name in ("cls", "offset", "size", "yaw"):
        diff = torch.max(torch.abs(pred_multi[head_name] - pred_fused[head_name])).item()
        assert diff < 1e-5, f"Head {head_name} diff {diff} exceeds tolerance 1e-5"
```

- [x] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_rep_block.py::test_mobilepixornext_backbone_reparam_equivalence -v
```
Expected: FAIL with `TypeError: MobilePixorNeXtBackbone.__init__() got an unexpected keyword argument 'use_reparam'`

- [x] **Step 3: Modify `mobilepixornext.py`, `registry.py`, and `model.py`**

In `detector/core/models/backbones/mobilepixornext.py`:
Add `use_reparam: bool = False` to `MobilePixorNeXtBackbone.__init__`:
```python
    def __init__(
        self,
        input_channels: int = 8,
        backbone_out_dim: int = 16,
        c4_attention: str = "litemla",
        scale_gated_fpn: bool = True,
        expansion: float = 2.5,
        use_reparam: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.backbone_out_dim = backbone_out_dim
        self.scale_gated_fpn = scale_gated_fpn
        self.use_reparam = bool(use_reparam)
```
Pass `use_reparam=self.use_reparam` to stages 2, 3, 4:
```python
        # Stage 2
        self.down2 = DownsampleBlock(32, 48, stride=2)
        self.stage2 = nn.Sequential(
            MobilePixorNeXtBlock(48, expansion=expansion, use_reparam=self.use_reparam),
            MobilePixorNeXtBlock(48, expansion=expansion, use_reparam=self.use_reparam),
        )

        # Stage 3
        self.down3 = DownsampleBlock(48, 96, stride=2)
        self.stage3 = nn.Sequential(
            MobilePixorNeXtBlock(96, expansion=expansion, use_reparam=self.use_reparam),
            MobilePixorNeXtBlock(96, expansion=expansion, use_reparam=self.use_reparam),
            MobilePixorNeXtBlock(96, expansion=expansion, use_reparam=self.use_reparam),
            MobilePixorNeXtBlock(96, expansion=expansion, use_reparam=self.use_reparam),
        )

        # Stage 4
        self.down4 = DownsampleBlock(96, 128, stride=2)
        self.stage4 = nn.Sequential(
            MobilePixorNeXtBlock(128, expansion=expansion, use_reparam=self.use_reparam),
            MobilePixorNeXtBlock(128, expansion=expansion, use_reparam=self.use_reparam),
        )
```
Add `switch_to_deploy` method to `MobilePixorNeXtBackbone`:
```python
    def switch_to_deploy(self):
        """Recursively trigger switch_to_deploy across all submodules."""
        for m in self.modules():
            if hasattr(m, "switch_to_deploy") and m is not self:
                m.switch_to_deploy()
```

In `detector/core/models/backbones/registry.py`:
Update `_build_mobilepixornext`:
```python
@register_backbone("mobilepixornext")
def _build_mobilepixornext(cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    return MobilePixorNeXtBackbone(
        input_channels=input_channels,
        backbone_out_dim=cfg.get("backbone_out_dim", 16),
        c4_attention=cfg.get("c4_attention", "litemla"),
        scale_gated_fpn=cfg.get("scale_gated_fpn", True),
        expansion=cfg.get("expansion", 2.5),
        use_reparam=cfg.get("use_reparam", False),
    )
```

In `detector/core/models/model.py`:
Add `switch_to_deploy` method to `CustomModel`:
```python
    def switch_to_deploy(self):
        """Deploy-time weight fusion for the backbone."""
        if hasattr(self.backbone, "switch_to_deploy"):
            self.backbone.switch_to_deploy()
```

- [x] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_rep_block.py -v
```
Expected: PASS (8 tests passed).

- [x] **Step 5: Git commit task 3**

```bash
git add detector/core/models/backbones/mobilepixornext.py detector/core/models/backbones/registry.py detector/core/models/model.py tests/test_rep_block.py
git commit -m "feat(model): wire structural reparameterization through backbone, registry, and CustomModel"
```

---

### Task 4: Add Experiment Config, Parameter Count Validation & Full Suite Regression Test

**Files:**
- Modify: `configs/kitti/oga_loss/kitti_mobilepixornext_litemla_oga.json`
- Create: `configs/kitti/reparameterization/kitti_mobilepixornext_litemla_oga_reparam.json`
- Modify: `tests/test_rep_block.py`

**Interfaces:**
- Consumes: `CustomModel` with config
- Produces: `configs/kitti/reparameterization/kitti_mobilepixornext_litemla_oga_reparam.json`

- [x] **Step 1: Write unit test validating parameter reduction and config parity**

Append to `tests/test_rep_block.py`:
```python
import json
from pathlib import Path


def test_config_reparam_loading_and_parameter_reduction():
    config_path = Path("configs/kitti/reparameterization/kitti_mobilepixornext_litemla_oga_reparam.json")
    assert config_path.exists(), "Reparam config must exist"

    with open(config_path, "r") as f:
        cfg_reparam = json.load(f)

    assert cfg_reparam.get("use_reparam") is True

    # Baseline M0 model
    cfg_base = dict(cfg_reparam)
    cfg_base["use_reparam"] = False
    model_m0 = CustomModel(cfg_base, num_classes=3, input_channels=8)
    params_m0 = sum(p.numel() for p in model_m0.parameters())

    # M1 model in training mode
    model_m1 = CustomModel(cfg_reparam, num_classes=3, input_channels=8)
    params_m1_train = sum(p.numel() for p in model_m1.parameters())

    # M1 training parameters must be greater than M0 due to 3x3 and identity branches
    assert params_m1_train > params_m0, f"Expected {params_m1_train} > {params_m0}"

    # After switch_to_deploy(), M1 parameter count must equal M0 exactly
    model_m1.switch_to_deploy()
    params_m1_deploy = sum(p.numel() for p in model_m1.parameters())

    assert params_m1_deploy == params_m0, (
        f"Post-deploy parameter count {params_m1_deploy} must match baseline M0 {params_m0}"
    )
```

- [x] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/test_rep_block.py::test_config_reparam_loading_and_parameter_reduction -v
```
Expected: FAIL with `AssertionError: Reparam config must exist`

- [x] **Step 3: Update anchor config and create reparam config**

In `configs/kitti/oga_loss/kitti_mobilepixornext_litemla_oga.json`:
Add `"use_reparam": false` to explicitly state anchor baseline configuration.

Create `configs/kitti/reparameterization/kitti_mobilepixornext_litemla_oga_reparam.json` matching `kitti_mobilepixornext_litemla_oga.json` with `"use_reparam": true`.

- [x] **Step 4: Run full test suite across the repository**

Run:
```bash
PYTHONPATH=. /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint -v
```
Expected: All 81+ tests PASS (0 failures, 0 errors).

- [x] **Step 5: Git commit task 4**

```bash
git add configs/kitti/mobilepixornext_oga/ tests/test_rep_block.py
git commit -m "feat(config): establish MobilePixorNeXt M1 Reparameterization configuration and verification"
```

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-28-structural-reparameterization.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
