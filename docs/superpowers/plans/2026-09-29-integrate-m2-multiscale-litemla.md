# Robust Multi-Scale LiteMLA Integration & Cumulative Verification Plan (Pillar 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the validated Robust Multi-Scale LiteMLA Attention (`RMS-LiteMLA` / Pillar 2) from `origin/feature/m2-robust-multiscale-litemla` into `feature/mobilepixornext-improvements`, harmonize backbone signatures with Pillar 1 Reparameterization and Pillar 4 IQA Header, reorganize config structures, establish the cumulative $M_1 + M_2 + M_4$ configuration, and verify complete test suite pass rate and smoke training stability.

**Architecture:**
- Upgrade `LiteMLARefinement` in `detector/core/models/backbones/mobilepixornext_blocks.py` with multi-scale aggregation (`scales=(3, 5)`), $QK$-RMSNorm over `head_dim`, and safe FP32 linear attention accumulation.
- Wire `c4_attention_scales` and `c4_attention_qk_norm` through `MobilePixorNeXtBackbone` and `detector/core/models/backbones/registry.py`, seamlessly harmonizing with `use_reparam` and `deploy` flags.
- Establish standardized config location `configs/kitti/multiscale_attention/` for isolated ablation configs, and `configs/kitti/cumulative/` for the integrated $M_1 + M_2 + M_4$ model.
- Port and adapt `tests/test_ms_litemla.py` (10 unit tests) with updated paths, plus add an end-to-end integration test verifying $M_1 + M_2 + M_4$ model creation, inference deploy switch, and OGA loss calculation.

**Tech Stack:** Python 3.14, PyTorch 2.11, NumPy, pytest.

**Spec:** `docs/plans/mobilepixornext_improvements/02_ROBUST_MULTISCALE_LITEMLA.md`

## Global Constraints
- Baseline anchor is strictly **MobilePixorNeXt + OGA Loss** ($M_0$).
- 100% Backward Compatibility: When `c4_attention_scales` is omitted, default to `(5,)`; when `c4_attention_qk_norm` is omitted, default to `"none"`.
- Complexity bound: Linear attention must maintain strict $\mathcal{O}(N \cdot d^2)$ computational complexity without ever materializing an $N \times N$ attention matrix ($N=8,800$).
- Parameter budget: Full model parameters post-integration must remain strictly below **2.0M** (target $\approx 1.36\text{M}$ deploy).
- Execution environment: `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint`.

---

## File Structure

| File | Responsibility |
| :--- | :--- |
| `detector/core/models/backbones/mobilepixornext_blocks.py` | Add `qk_norm` support (`rmsnorm`, `layernorm`, `none`), `_normalize_qk` over `head_dim`, and configurable `scales` to `LiteMLARefinement`. |
| `detector/core/models/backbones/mobilepixornext.py` | Add `c4_attention_scales` and `c4_attention_qk_norm` to `MobilePixorNeXtBackbone.__init__` while preserving `use_reparam` and `deploy`. |
| `detector/core/models/backbones/registry.py` | Forward `c4_attention_scales` and `c4_attention_qk_norm` from config dictionary in `_build_mobilepixornext`. |
| `configs/kitti/multiscale_attention/kitti_mobilepixornext_ms_litemla_oga.json` | Isolated M2 configuration with `(3, 5)` kernel scales and `rmsnorm`. |
| `configs/kitti/multiscale_attention/kitti_mobilepixornext_ms_litemla_no_norm_oga.json` | Isolated M2 ablation configuration with `(3, 5)` kernel scales and `none` norm. |
| `configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_oga.json` | Full cumulative configuration integrating Pillar 1 (`reparam`), Pillar 2 (`ms_litemla`), and Pillar 4 (`iqa_header`). |
| `tests/test_ms_litemla.py` | Comprehensive unit tests for M2 (shape, gradient, norm modes, head_dim RMS, BF16 extreme values, TorchScript) and cumulative integration. |

---

## Task Breakdown

### Task 1: Integrate Robust Multi-Scale LiteMLA into `mobilepixornext_blocks.py`

**Files:**
- Modify: `detector/core/models/backbones/mobilepixornext_blocks.py`
- Test: `tests/test_ms_litemla.py`

**Interfaces:**
- `LiteMLARefinement(channels: int = 96, head_dim: int = 16, scales: tuple = (5,), eps: float = 1e-6, layer_scale_init: float = 0.01, qk_norm: str = "none")`
- `LiteMLARefinement._normalize_qk(query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]`
- Maintains compatibility defaults: `scales=(5,)`, `qk_norm="none"`.

- [x] **Step 1: Write unit tests in `tests/test_ms_litemla.py` for module behavior**

Create `tests/test_ms_litemla.py` incorporating the unit tests from M2 (testing `scales=(3, 5)`, `qk_norm="rmsnorm"`, `_normalize_qk` on `head_dim`, TorchScript scripting, and BF16 stability):

```python
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))

from core.models.backbones.mobilepixornext_blocks import LiteMLARefinement


def test_multiscale_litemla_shape_and_gradient():
    module = LiteMLARefinement(
        channels=96,
        head_dim=16,
        scales=(3, 5),
        qk_norm="rmsnorm",
    )
    x = torch.randn(2, 96, 20, 16, requires_grad=True)

    output = module(x)
    output.mean().backward()

    assert output.shape == x.shape
    assert len(module.aggregations) == 2
    assert x.grad is not None
    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    [
        ("none", torch.nn.Identity),
        ("rmsnorm", torch.nn.RMSNorm),
        ("layernorm", torch.nn.LayerNorm),
    ],
)
def test_qk_norm_modes(mode, expected_type):
    module = LiteMLARefinement(
        channels=32,
        head_dim=8,
        scales=(3, 5),
        qk_norm=mode,
    )
    assert isinstance(module.query_norm, expected_type)
    assert isinstance(module.key_norm, expected_type)
    assert module.qk_norm_name == mode


def test_rmsnorm_uses_head_dimension_not_token_dimension():
    module = LiteMLARefinement(
        channels=32,
        head_dim=8,
        scales=(3,),
        qk_norm="rmsnorm",
        eps=1e-6,
    )
    query = torch.randn(2, 4, 8, 37)
    key = torch.randn(2, 4, 8, 37)

    normalized_query, normalized_key = module._normalize_qk(query, key)
    query_rms = normalized_query.square().mean(dim=2).sqrt()
    key_rms = normalized_key.square().mean(dim=2).sqrt()

    assert normalized_query.shape == query.shape
    assert normalized_key.shape == key.shape
    assert torch.allclose(query_rms, torch.ones_like(query_rms), atol=2e-4, rtol=2e-4)
    assert torch.allclose(key_rms, torch.ones_like(key_rms), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    "scales",
    [(), (2,), (3, 3), (3, 4), (3, True), "3,5"],
)
def test_invalid_scales_are_rejected(scales):
    with pytest.raises(ValueError, match="distinct odd integers"):
        LiteMLARefinement(channels=32, head_dim=8, scales=scales)


def test_invalid_qk_norm_is_rejected():
    with pytest.raises(ValueError, match="qk_norm"):
        LiteMLARefinement(
            channels=32,
            head_dim=8,
            scales=(3, 5),
            qk_norm="batchnorm",
        )


def test_legacy_defaults_keep_state_dict_compatible():
    reference = LiteMLARefinement(channels=32, head_dim=8)
    restored = LiteMLARefinement(channels=32, head_dim=8)

    restored.load_state_dict(reference.state_dict(), strict=True)

    assert reference.scales == (5,)
    assert reference.qk_norm_name == "none"
    assert len(reference.aggregations) == 1
    assert not any("query_norm" in key or "key_norm" in key for key in reference.state_dict())


def test_multiscale_litemla_torchscript_matches_eager():
    torch.manual_seed(42)
    module = LiteMLARefinement(
        channels=32,
        head_dim=8,
        scales=(3, 5),
        qk_norm="rmsnorm",
    ).eval()
    x = torch.randn(1, 32, 12, 10)

    scripted = torch.jit.script(module)
    with torch.no_grad():
        expected = module(x)
        actual = scripted(x)

    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 support is required",
)
def test_multiscale_litemla_bf16_extreme_values_are_finite():
    device = torch.device("cuda")
    module = LiteMLARefinement(
        channels=32,
        head_dim=8,
        scales=(3, 5),
        qk_norm="rmsnorm",
    ).to(device)
    values = torch.randn(2, 32, 20, 16, device=device)
    values[:, :, ::2, ::2] *= 1e3
    values[:, :, 1::2, 1::2] *= 1e-4
    x = values.requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = module(x)
        loss = output.square().mean()
    loss.backward()

    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
```

- [x] **Step 2: Run tests to verify failure on current codebase**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_ms_litemla.py -k "test_multiscale_litemla_shape_and_gradient" -v
```
Expected: FAIL (`TypeError: LiteMLARefinement.__init__() got an unexpected keyword argument 'qk_norm'`)

- [x] **Step 3: Implement QK normalization and multi-scale aggregation in `LiteMLARefinement`**

Update `LiteMLARefinement` in `detector/core/models/backbones/mobilepixornext_blocks.py`:
- Add `qk_norm: str = "none"` to `__init__`.
- Validate `qk_norm in {"none", "rmsnorm", "layernorm"}`.
- Store `self.scales = tuple(scales)` and `self.qk_norm_name = qk_norm`.
- Instantiate `self.query_norm` and `self.key_norm` (`nn.RMSNorm(head_dim, eps=self.eps)` if `"rmsnorm"`, `nn.LayerNorm(head_dim, eps=self.eps)` if `"layernorm"`, else `nn.Identity()`).
- Add helper `_normalize_qk`:
  ```python
  def _normalize_qk(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
      query = self.query_norm(query.transpose(-1, -2)).transpose(-1, -2)
      key = self.key_norm(key.transpose(-1, -2)).transpose(-1, -2)
      return query, key
  ```
- In `_linear_attention_core`, call `_normalize_qk` before `.relu()`.

- [x] **Step 4: Run tests to verify they pass**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_ms_litemla.py -v
```
Expected: PASS (all tests pass)

- [x] **Step 5: Commit changes**

```bash
git add detector/core/models/backbones/mobilepixornext_blocks.py tests/test_ms_litemla.py
git commit -m "feat(attention): implement robust multi-scale LiteMLA with QK-RMSNorm"
```

---

### Task 2: Wire M2 Options into `MobilePixorNeXtBackbone` and `registry.py`

**Files:**
- Modify: `detector/core/models/backbones/mobilepixornext.py`
- Modify: `detector/core/models/backbones/registry.py`
- Test: `tests/test_ms_litemla.py`
- Test: `tests/test_mobilepixornext_backbone.py`
- Test: `tests/test_model_registry.py`

**Interfaces:**
- `MobilePixorNeXtBackbone.__init__(input_channels=8, backbone_out_dim=16, c4_attention="litemla", c4_attention_scales=(5,), c4_attention_qk_norm="none", scale_gated_fpn=True, expansion=2.5, use_reparam=False, deploy=False)`
- `_build_mobilepixornext(cfg, input_channels)` forwards `c4_attention_scales` and `c4_attention_qk_norm`.

- [x] **Step 1: Add backbone wiring tests in `tests/test_ms_litemla.py`**

Append to `tests/test_ms_litemla.py`:
```python
from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone
from core.models.backbones.registry import build_backbone


def test_backbone_wires_multiscale_and_reparam_together():
    backbone = MobilePixorNeXtBackbone(
        input_channels=8,
        c4_attention="litemla",
        c4_attention_scales=(3, 5),
        c4_attention_qk_norm="rmsnorm",
        use_reparam=True,
    )
    assert backbone.c4_attention.scales == (3, 5)
    assert backbone.c4_attention.qk_norm_name == "rmsnorm"
    assert backbone.use_reparam is True
    assert hasattr(backbone.stage2[0], "dw_block")

    # Verify forward pass
    x = torch.randn(2, 8, 800, 704)
    out = backbone(x)
    assert out.shape == (2, 16, 200, 176)

    # Verify switch_to_deploy recursion
    backbone.eval()
    backbone.switch_to_deploy()
    out_deploy = backbone(x)
    assert out_deploy.shape == (2, 16, 200, 176)


def test_registry_builds_mobilepixornext_with_m2_options():
    cfg = {
        "c4_attention_scales": [3, 5],
        "c4_attention_qk_norm": "rmsnorm",
        "use_reparam": True,
    }
    model = build_backbone("mobilepixornext", cfg, input_channels=8)
    assert model.c4_attention.scales == (3, 5)
    assert model.c4_attention.qk_norm_name == "rmsnorm"
    assert model.use_reparam is True
```

- [x] **Step 2: Run test to verify failure**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_ms_litemla.py -k "test_backbone_wires_multiscale_and_reparam_together" -v
```
Expected: FAIL (`TypeError: MobilePixorNeXtBackbone.__init__() got unexpected keyword arguments`)

- [x] **Step 3: Modify `MobilePixorNeXtBackbone` and `registry.py`**

In `detector/core/models/backbones/mobilepixornext.py`:
- Update `__init__` signature to include `c4_attention_scales: tuple = (5,)` and `c4_attention_qk_norm: str = "none"`.
- Pass `scales=tuple(c4_attention_scales)` and `qk_norm=c4_attention_qk_norm` to `LiteMLARefinement`.
In `detector/core/models/backbones/registry.py`:
- In `_build_mobilepixornext`, pass:
  ```python
  c4_attention_scales=cfg.get("c4_attention_scales", (5,)),
  c4_attention_qk_norm=cfg.get("c4_attention_qk_norm", "none"),
  ```

- [x] **Step 4: Run tests to verify they pass**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_ms_litemla.py tests/test_mobilepixornext_backbone.py tests/test_model_registry.py -v
```
Expected: PASS

- [x] **Step 5: Commit changes**

```bash
git add detector/core/models/backbones/mobilepixornext.py detector/core/models/backbones/registry.py tests/test_ms_litemla.py
git commit -m "feat(backbone): wire multi-scale LiteMLA options through backbone and registry"
```

---

### Task 3: Establish Modular Configurations & Update Paths

**Files:**
- Create: `configs/kitti/multiscale_attention/kitti_mobilepixornext_ms_litemla_oga.json`
- Create: `configs/kitti/multiscale_attention/kitti_mobilepixornext_ms_litemla_no_norm_oga.json`
- Create: `configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_oga.json`
- Modify: `tests/test_ms_litemla.py`

- [x] **Step 1: Create isolated M2 configs in `configs/kitti/multiscale_attention/`**

Create `configs/kitti/multiscale_attention/kitti_mobilepixornext_ms_litemla_oga.json`:
- Base on `configs/kitti/oga_loss/kitti_mobilepixornext_litemla_oga.json`.
- Add `"c4_attention_scales": [3, 5]`.
- Add `"c4_attention_qk_norm": "rmsnorm"`.

Create `configs/kitti/multiscale_attention/kitti_mobilepixornext_ms_litemla_no_norm_oga.json`:
- Same as above, but with `"c4_attention_qk_norm": "none"`.

- [x] **Step 2: Create cumulative $M_1 + M_2 + M_4$ config in `configs/kitti/cumulative/`**

Create `configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_oga.json`:
- Base on `configs/kitti/iou_aware_header/kitti_mobilepixornext_litemla_oga_reparam_iqa.json` (which already has `"use_reparam": true`, `"header_use_iou": true`, `"nms_alpha": 0.5`).
- Add `"c4_attention_scales": [3, 5]`.
- Add `"c4_attention_qk_norm": "rmsnorm"`.
- Set note: `"mobilepixornext_m1_m2_m4_cumulative_oga"`.

- [x] **Step 3: Update config integration tests in `tests/test_ms_litemla.py`**

Add tests to `tests/test_ms_litemla.py`:
```python
def test_m2_config_wires_multiscale_rmsnorm():
    config_path = (
        ROOT
        / "configs/kitti/multiscale_attention/kitti_mobilepixornext_ms_litemla_oga.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    backbone = build_backbone("mobilepixornext", config["model"], input_channels=8)

    assert backbone.c4_attention.scales == (3, 5)
    assert backbone.c4_attention.qk_norm_name == "rmsnorm"
    assert len(backbone.c4_attention.aggregations) == 2
    assert sum(parameter.numel() for parameter in backbone.parameters()) < 2_000_000


def test_scale_only_ablation_config_disables_qk_norm():
    config_path = (
        ROOT
        / "configs/kitti/multiscale_attention/"
        "kitti_mobilepixornext_ms_litemla_no_norm_oga.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    backbone = build_backbone("mobilepixornext", config["model"], input_channels=8)

    assert backbone.c4_attention.scales == (3, 5)
    assert backbone.c4_attention.qk_norm_name == "none"
    assert isinstance(backbone.c4_attention.query_norm, torch.nn.Identity)


def test_cumulative_m1_m2_m4_config_instantiates_full_model():
    from core.models.model import CustomModel
    from core.losses.strategies.oga import OgaLossStrategy

    config_path = (
        ROOT
        / "configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_oga.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = CustomModel(config["model"], input_channels=8)

    # 1. Verify M1 (use_reparam)
    assert model.backbone.use_reparam is True

    # 2. Verify M2 (scales=(3, 5), rmsnorm)
    assert model.backbone.c4_attention.scales == (3, 5)
    assert model.backbone.c4_attention.qk_norm_name == "rmsnorm"

    # 3. Verify M4 (Header with 5 branches including IoU)
    assert model.header.use_iou is True
    assert hasattr(model.header, "iou")

    # 4. Verify Parameter budget
    train_params = sum(p.numel() for p in model.parameters())
    assert train_params < 2_000_000

    # 5. Forward pass
    x = torch.randn(2, 8, 800, 704)
    preds = model(x)
    assert "cls" in preds and "offset" in preds and "size" in preds and "yaw" in preds and "iou" in preds
    assert preds["iou"].shape == (2, 1, 200, 176)

    # 6. Verify switch_to_deploy
    model.eval()
    model.switch_to_deploy()
    deploy_params = sum(p.numel() for p in model.parameters())
    assert deploy_params < train_params

    # 7. Verify Loss Strategy compatibility
    loss_strategy = OgaLossStrategy(config["loss"])
    assert loss_strategy.use_iou is True
    assert loss_strategy.uncertainty.num_tasks == 6
```

- [x] **Step 4: Run test to verify config validity**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_ms_litemla.py -v
```
Expected: PASS (all tests pass)

- [x] **Step 5: Commit changes**

```bash
git add configs/kitti/multiscale_attention/ configs/kitti/cumulative/ tests/test_ms_litemla.py
git commit -m "feat(config): establish M2 multiscale attention and cumulative M1+M2+M4 configs"
```

---

### Task 4: Full Regression Test Suite & Smoke Verification

**Files:**
- Modify: `docs/plans/mobilepixornext_improvements/00_ABLATION_STUDY_PROTOCOL.md` (sync M2 status)
- Modify: `README.md` (document M2 and cumulative configs)

- [x] **Step 1: Run the full test suite**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint -v
```
Expected: PASS with 124+ passed tests and 0 failures.

- [x] **Step 2: Run a 1-epoch / limited-batches training smoke test with BF16**

Execute a fast training smoke test:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -c "
import torch
from core.models.model import CustomModel
from core.losses.strategies.oga import OgaLossStrategy
import json

config = json.loads(open('configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_oga.json').read())
model = CustomModel(config['model'], input_channels=8).cuda()
loss_fn = OgaLossStrategy(config['loss']).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randn(2, 8, 800, 704, device='cuda')
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    preds = model(x)
    targets = {
        'cls': torch.zeros(2, 3, 200, 176, device='cuda'),
        'reg': torch.zeros(2, 6, 200, 176, device='cuda'),
        'reg_mask': torch.zeros(2, 1, 200, 176, dtype=torch.bool, device='cuda'),
    }
    targets['reg_mask'][0, 0, 100, 88] = True
    targets['reg'][0, :, 100, 88] = torch.tensor([0.1, 0.2, 1.5, 1.8, 4.5, 0.5], device='cuda')
    targets['cls'][0, 0, 100, 88] = 1.0
    loss_dict = loss_fn(preds, targets)
    loss = loss_dict['total_loss']

assert torch.isfinite(loss).all()
loss.backward()
optimizer.step()
print('Smoke training step successful! Loss:', loss.item())
"
```
Expected: Prints `Smoke training step successful! Loss: ...` with zero NaN/Inf.

- [x] **Step 3: Update documentation and ablation study roadmap**

Update `README.md` and `docs/plans/mobilepixornext_improvements/00_ABLATION_STUDY_PROTOCOL.md` to reference the modular `configs/kitti/multiscale_attention/` and `configs/kitti/cumulative/` files.

- [x] **Step 4: Commit final documentation update**

```bash
git add README.md docs/plans/mobilepixornext_improvements/00_ABLATION_STUDY_PROTOCOL.md
git commit -m "docs: document M2 integration and cumulative M1+M2+M4 ablation config"
```

---

## Acceptance Criteria
1. `LiteMLARefinement` supports both `scales=(3, 5)` and bounded $QK$-RMSNorm along `head_dim`.
2. Old checkpoints and configurations without `c4_attention_scales` / `c4_attention_qk_norm` remain 100% backward compatible.
3. Reparameterization ($M_1$), Multi-Scale LiteMLA ($M_2$), and IoU Header ($M_4$) operate harmoniously in a single cumulative model (`kitti_mobilepixornext_m1_m2_m4_oga.json`).
4. Total deployment parameter count remains below **2.0M** ($\approx 1.36\text{M}$ deploy).
5. All 124+ unit tests pass without failure, and the BF16 autocast smoke training step executes without NaN/Inf.
