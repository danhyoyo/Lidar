# MobilePIXOR Modular Ablation Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable systematic, component-level ablation studies on the original `mobilepixor` backbone (pure MobileNetV2 inverted residuals, strictly without CoordAtt) across Loss (Baseline vs OGA), FPN (SumFPN vs SG-FPN), Header (Baseline vs Modern vs IQA), BEV representation (35ch vs Rich8), and Augmentation (No Aug vs PCU-Aug).

**Architecture:** Parameterize `MobilePixorBackBone` to accept `input_channels` and `scale_gated_fpn`, wire the backbone registry builder to propagate JSON config parameters, and provide a standardized progressive ablation config matrix (`00_baseline` through `05_full_pcu`) with automated test coverage.

**Tech Stack:** PyTorch 2.x, NumPy, Python 3.14 (conda `AI_env`), pytest.

**Spec:** Documented in conversation transcript and [mobilepixor_ablation_analysis.md](file:///home/duyennh/.gemini/antigravity-cli/brain/d96d5546-ab83-4ab7-9a82-b9b276ba80f9/mobilepixor_ablation_analysis.md).

## Global Constraints

- Do not alter `mobilepixor_coordatt` or `mobilepixornext` backbones.
- `mobilepixor` must remain pure MobileNetV2 (no CoordAtt or self-attention at any stage).
- Zero-initialized scale gates in `scale_gated_fpn` must guarantee exact mathematical equivalence to SumFPN at initialization ($2 \cdot \sigma(0) = 1.0$).
- All config files must adhere to the standardized schema and pass `tests/test_run_name_and_logging.py`.
- Tests must be executed using:
  `PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros <test_file>`

---

## File Structure

- **Modified:**
  - `detector/core/models/backbones/mobilepixor.py`: Parameterize `MobilePixorBackBone` with `input_channels` and `scale_gated_fpn` gate convolutions and forward path.
  - `detector/core/models/backbones/registry.py`: Update `_build_mobilepixor` to pass `input_channels` and `cfg.get("scale_gated_fpn", False)`.
- **Created:**
  - `tests/test_mobilepixor_backbone.py`: Unit tests for `MobilePixorBackBone` shapes, input channels, and FPN gating.
  - `tests/test_mobilepixor_integration.py`: End-to-end model and loss integration test with `CustomModel`.
  - `configs/kitti/mobilepixor_ablation/00_mobilepixor_baseline.json`: Pure MobilePIXOR baseline (35ch, SumFPN, Baseline loss, Plain head, no aug).
  - `configs/kitti/mobilepixor_ablation/01_mobilepixor_oga.json`: + OGA loss (MGIoU + corner distance + T-SBUW).
  - `configs/kitti/mobilepixor_ablation/02_mobilepixor_oga_sgfpn.json`: + Scale-Gated FPN (`scale_gated_fpn: true`).
  - `configs/kitti/mobilepixor_ablation/03_mobilepixor_oga_sgfpn_iqa.json`: + IQA Header & Joint NMS (`header_use_iou: true`, `nms_alpha: 0.5`).
  - `configs/kitti/mobilepixor_ablation/04_mobilepixor_oga_sgfpn_iqa_rich8.json`: + Rich8 BEV representation (8ch input).
  - `configs/kitti/mobilepixor_ablation/05_mobilepixor_full_pcu.json`: + Physics-Consistent Augmentation (`use_pcu_aug: true`).
  - `tests/test_mobilepixor_ablation_configs.py`: Verification test suite loading and running dummy forward passes on all 6 ablation configs.

---

### Task 1: Parameterize MobilePixorBackBone with `input_channels` and `scale_gated_fpn`

**Files:**
- Create: `tests/test_mobilepixor_backbone.py`
- Modify: `detector/core/models/backbones/mobilepixor.py:169-229`
- Modify: `detector/core/models/backbones/registry.py:49-52`

**Interfaces:**
- Consumes: `input_channels: int` and `scale_gated_fpn: bool` from registry builder.
- Produces: `MobilePixorBackBone(input_channels=input_channels, scale_gated_fpn=scale_gated_fpn)` outputting Tensor of shape `(B, 16, H/4, W/4)`.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_mobilepixor_backbone.py`:
```python
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.models.backbones.mobilepixor import MobilePixorBackBone
from core.models.backbones.registry import build_backbone


class TestMobilePixorBackbone(unittest.TestCase):
    def test_default_backward_compatible_initialization(self):
        bb = MobilePixorBackBone()
        self.assertEqual(bb.conv1.in_channels, 35)
        self.assertFalse(hasattr(bb, "scale_gated_fpn") and bb.scale_gated_fpn)
        x = torch.randn(2, 35, 128, 128)
        out = bb(x)
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_configurable_input_channels_rich8(self):
        bb = MobilePixorBackBone(input_channels=8)
        self.assertEqual(bb.conv1.in_channels, 8)
        x = torch.randn(2, 8, 128, 128)
        out = bb(x)
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_scale_gated_fpn_initialization_and_forward(self):
        bb = MobilePixorBackBone(input_channels=8, scale_gated_fpn=True)
        self.assertTrue(bb.scale_gated_fpn)
        self.assertTrue(hasattr(bb, "gate_c4"))
        self.assertTrue(hasattr(bb, "gate_c3"))
        x = torch.randn(2, 8, 128, 128)
        out = bb(x)
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_scale_gated_fpn_zero_init_equivalence(self):
        """Zero-initialized gates (2 * sigmoid(0) = 1.0) must produce identical output to SumFPN."""
        torch.manual_seed(42)
        bb_sum = MobilePixorBackBone(input_channels=8, scale_gated_fpn=False)
        bb_sg = MobilePixorBackBone(input_channels=8, scale_gated_fpn=True)

        # Copy identical backbone feature weights
        bb_sg.load_state_dict(
            {k: v for k, v in bb_sum.state_dict().items()}, strict=False
        )

        x = torch.randn(2, 8, 64, 64)
        bb_sum.eval()
        bb_sg.eval()
        with torch.no_grad():
            out_sum = bb_sum(x)
            out_sg = bb_sg(x)

        diff = (out_sum - out_sg).abs().max().item()
        self.assertLess(diff, 1e-5, f"SG-FPN at zero-init should match SumFPN, got max diff {diff}")

    def test_registry_builder_passes_arguments(self):
        cfg = {"scale_gated_fpn": True}
        bb = build_backbone("mobilepixor", cfg, input_channels=8)
        self.assertIsInstance(bb, MobilePixorBackBone)
        self.assertEqual(bb.conv1.in_channels, 8)
        self.assertTrue(bb.scale_gated_fpn)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_mobilepixor_backbone.py
```
Expected: FAIL (`TypeError: MobilePixorBackBone.__init__() got an unexpected keyword argument 'input_channels'`).

- [ ] **Step 3: Modify `detector/core/models/backbones/mobilepixor.py` and `registry.py`**

In `detector/core/models/backbones/mobilepixor.py`, update `MobilePixorBackBone`:
```python
class MobilePixorBackBone(nn.Module):

    def __init__(
        self,
        block=InvertedResidual,
        use_bn: bool = True,
        input_channels: int = 35,
        scale_gated_fpn: bool = False,
    ):
        super(MobilePixorBackBone, self).__init__()

        self.use_bn = use_bn
        self.scale_gated_fpn = bool(scale_gated_fpn)

        # Block 1
        self.conv1 = conv3x3(input_channels, 32)
        self.conv2 = conv3x3(32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Block 2-5
        self.input_channel = 32
        self.block2 = self._make_layer(block, 1, 24, 1, 2)
        self.block3 = self._make_layer(block, 6, 32, 3, 2)
        self.block4 = self._make_layer(block, 6, 64, 4, 2)
        self.block5 = self._make_layer(block, 6, 96, 3, 2)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        p = 1
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, p))

        if self.scale_gated_fpn:
            self.gate_c4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=True)
            self.gate_c3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=True)
            nn.init.zeros_(self.gate_c4.weight)
            nn.init.zeros_(self.gate_c4.bias)
            nn.init.zeros_(self.gate_c3.weight)
            nn.init.zeros_(self.gate_c3.bias)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu(x)

        # bottom up layers
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        u4 = self.deconv1(l5)
        if self.scale_gated_fpn:
            p5 = u4 + 2 * torch.sigmoid(self.gate_c4(l4 + u4)) * l4
        else:
            p5 = l4 + u4

        l3 = self.latlayer3(c3)
        u3 = self.deconv2(p5)
        if self.scale_gated_fpn:
            p4 = u3 + 2 * torch.sigmoid(self.gate_c3(l3 + u3)) * l3
        else:
            p4 = l3 + u3

        return p4
```

In `detector/core/models/backbones/registry.py`:
```python
@register_backbone("mobilepixor")
def _build_mobilepixor(cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    return MobilePixorBackBone(
        input_channels=input_channels,
        scale_gated_fpn=cfg.get("scale_gated_fpn", False),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_mobilepixor_backbone.py
```
Expected: PASS (5 passed).

- [ ] **Step 5: Run existing model registry tests to ensure no regressions**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_model_registry.py
```
Expected: PASS (8 passed).

- [ ] **Step 6: Commit**

```bash
git add detector/core/models/backbones/mobilepixor.py detector/core/models/backbones/registry.py tests/test_mobilepixor_backbone.py
git commit -m "feat(mobilepixor): add configurable input channels and scale-gated FPN"
```

---

### Task 2: End-to-End Model Integration & Loss Compatibility Tests for MobilePIXOR

**Files:**
- Create: `tests/test_mobilepixor_integration.py`

**Interfaces:**
- Consumes: `CustomModel`, `LossFunction` from detector core.
- Produces: Passing test verifying combinations of `mobilepixor` with baseline/OGA losses and plain/IQA headers.

- [ ] **Step 1: Write integration tests**

Create `tests/test_mobilepixor_integration.py`:
```python
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.models.model import CustomModel
from core.losses.loss_fn import LossFunction
from postprocess import filter_pred


class TestMobilePixorIntegration(unittest.TestCase):
    def test_mobilepixor_with_baseline_loss(self):
        cfg_model = {
            "backbone": "mobilepixor",
            "backbone_out_dim": 16,
            "cls_encoding": "gaussian",
            "header_use_bn": False,
            "header_act": "none",
        }
        model = CustomModel(cfg_model, num_classes=3, input_channels=35)
        criterion = LossFunction("gaussian", {"name": "baseline"})

        x = torch.randn(2, 35, 128, 128)
        pred = model(x)
        self.assertIn("cls", pred)
        self.assertNotIn("iou", pred)

        target = {
            "cls": torch.zeros(2, 3, 32, 32),
            "offset": torch.zeros(2, 2, 32, 32),
            "size": torch.zeros(2, 2, 32, 32),
            "yaw": torch.zeros(2, 2, 32, 32),
            "reg_mask": torch.ones(2, 32, 32),
        }
        loss_dict = criterion(pred, target)
        self.assertIn("loss", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))

    def test_mobilepixor_with_oga_loss_and_iqa_header(self):
        cfg_model = {
            "backbone": "mobilepixor",
            "backbone_out_dim": 16,
            "cls_encoding": "gaussian",
            "header_use_bn": True,
            "header_act": "silu",
            "header_use_iou": True,
            "scale_gated_fpn": True,
        }
        model = CustomModel(cfg_model, num_classes=3, input_channels=8)
        loss_cfg = {
            "name": "oga",
            "temperature": 2.0,
            "clamp_bound": 3.0,
            "corner_beta": 1.0,
            "use_iou": True,
            "iou_target_type": "mgiou",
            "iou_loss_weight": 1.0,
        }
        criterion = LossFunction("gaussian", loss_cfg)

        x = torch.randn(2, 8, 128, 128)
        pred = model(x)
        self.assertIn("iou", pred)
        self.assertEqual(pred["iou"].shape, (2, 1, 32, 32))

        target = {
            "cls": torch.zeros(2, 3, 32, 32),
            "offset": torch.zeros(2, 2, 32, 32),
            "size": torch.zeros(2, 2, 32, 32),
            "yaw": torch.zeros(2, 2, 32, 32),
            "reg_mask": torch.ones(2, 32, 32),
        }
        loss_dict = criterion(pred, target)
        self.assertIn("loss", loss_dict)
        self.assertIn("iou", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))

    def test_mobilepixor_iqa_postprocess_nms(self):
        pred_single = {
            "cls": torch.randn(1, 3, 200, 176),
            "offset": torch.randn(1, 2, 200, 176),
            "size": torch.randn(1, 2, 200, 176),
            "yaw": torch.randn(1, 2, 200, 176),
            "iou": torch.randn(1, 1, 200, 176),
        }
        config = {
            "geometry": {
                "x_min": 0, "x_max": 70.4, "x_res": 0.1,
                "y_min": -40, "y_max": 40, "y_res": 0.1,
            },
            "nms_alpha": 0.5,
        }
        boxes = filter_pred(pred_single, config, out_size_factor=4, thres=0.1, nms_thres=0.5)
        self.assertIsInstance(boxes, type(torch.empty(0).numpy()))
        self.assertEqual(boxes.ndim, 2)
        if len(boxes) > 0:
            self.assertEqual(boxes.shape[1], 7)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run integration tests to verify they pass**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_mobilepixor_integration.py
```
Expected: PASS (3 passed).

- [ ] **Step 3: Commit**

```bash
git add tests/test_mobilepixor_integration.py
git commit -m "test(mobilepixor): add end-to-end integration tests for loss and IQA header"
```

---

### Task 3: Create Ablation Experiment Configurations for MobilePIXOR

**Files:**
- Create directory: `configs/kitti/mobilepixor_ablation/`
- Create: `configs/kitti/mobilepixor_ablation/00_mobilepixor_baseline.json`
- Create: `configs/kitti/mobilepixor_ablation/01_mobilepixor_oga.json`
- Create: `configs/kitti/mobilepixor_ablation/02_mobilepixor_oga_sgfpn.json`
- Create: `configs/kitti/mobilepixor_ablation/03_mobilepixor_oga_sgfpn_iqa.json`
- Create: `configs/kitti/mobilepixor_ablation/04_mobilepixor_oga_sgfpn_iqa_rich8.json`
- Create: `configs/kitti/mobilepixor_ablation/05_mobilepixor_full_pcu.json`
- Create: `tests/test_mobilepixor_ablation_configs.py`

**Interfaces:**
- Produces: 6 self-contained JSON configs isolating each proposed improvement sequentially.

- [ ] **Step 1: Write test verifying all 6 ablation configs exist and construct valid models**

Create `tests/test_mobilepixor_ablation_configs.py`:
```python
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "detector"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "kitti_training_pipeline"))

import torch
from common import build_model, generate_run_name, input_shape
from core.losses.loss_fn import LossFunction


class TestMobilePixorAblationConfigs(unittest.TestCase):
    CONFIG_DIR = REPO_ROOT / "configs" / "kitti" / "mobilepixor_ablation"
    EXPECTED_CONFIGS = [
        "00_mobilepixor_baseline.json",
        "01_mobilepixor_oga.json",
        "02_mobilepixor_oga_sgfpn.json",
        "03_mobilepixor_oga_sgfpn_iqa.json",
        "04_mobilepixor_oga_sgfpn_iqa_rich8.json",
        "05_mobilepixor_full_pcu.json",
    ]

    def test_all_expected_configs_exist(self):
        for name in self.EXPECTED_CONFIGS:
            path = self.CONFIG_DIR / name
            self.assertTrue(path.is_file(), f"Missing config: {path}")

    def test_configs_instantiate_model_and_criterion(self):
        for name in self.EXPECTED_CONFIGS:
            path = self.CONFIG_DIR / name
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            model = build_model(cfg)
            criterion = LossFunction(cfg["model"]["cls_encoding"], cfg.get("loss"))
            shape = input_shape(cfg)
            dummy_in = torch.randn(shape)
            out = model(dummy_in)
            self.assertIn("cls", out)

            # Verify run name format is valid
            run_name = generate_run_name(cfg, seed=42)
            self.assertTrue(run_name.startswith("mobilepixor-"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure (configs not yet created)**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_mobilepixor_ablation_configs.py
```
Expected: FAIL (`AssertionError: Missing config`).

- [ ] **Step 3: Create the 6 JSON configuration files**

**Config 00: `configs/kitti/mobilepixor_ablation/00_mobilepixor_baseline.json`**
```json
{
  "date": "reproducible",
  "ver": 1,
  "note": "00_mobilepixor_baseline_legacy35_sumfpn_l1",
  "seed": 42,
  "device": "cuda",
  "multi_gpu": false,
  "data": {
    "location": "data/kitti/processed",
    "num_classes": 3,
    "out_size_factor": 4,
    "gaussian_overlap": 0.1,
    "min_radius": 4,
    "bev_encoding": {
      "name": "binary_slices"
    },
    "kitti": {
      "location": "data/kitti/processed",
      "objects": {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2
      },
      "geometry": {
        "x_min": 0.0,
        "x_max": 70.4,
        "x_res": 0.1,
        "y_min": -40.0,
        "y_max": 40.0,
        "y_res": 0.1,
        "z_min": -2.5,
        "z_max": 1.0,
        "z_res": 0.1
      }
    }
  },
  "model": {
    "backbone": "mobilepixor",
    "backbone_out_dim": 16,
    "cls_encoding": "gaussian",
    "scale_gated_fpn": false,
    "header_use_bn": false,
    "header_act": "none",
    "header_use_iou": false
  },
  "loss": {
    "name": "baseline"
  },
  "augmentation": {
    "p": 0.0
  },
  "train": {
    "data": "splits/kitti/train.txt",
    "epochs": 100,
    "physical_batch_size": 16,
    "accumulation_steps": 1,
    "precision": "bf16",
    "optimizer": "adamw",
    "learning_rate": 0.002,
    "min_lr": 1e-06,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "save_every": 5
  },
  "val": {
    "data": "splits/kitti/val.txt",
    "physical_batch_size": 2,
    "val_every": 1
  }
}
```

**Config 01: `configs/kitti/mobilepixor_ablation/01_mobilepixor_oga.json`**
(Same as 00, but replace `"loss"` with OGA loss):
```json
{
  "date": "reproducible",
  "ver": 1,
  "note": "01_mobilepixor_oga_loss",
  "seed": 42,
  "device": "cuda",
  "multi_gpu": false,
  "data": {
    "location": "data/kitti/processed",
    "num_classes": 3,
    "out_size_factor": 4,
    "gaussian_overlap": 0.1,
    "min_radius": 4,
    "bev_encoding": {
      "name": "binary_slices"
    },
    "kitti": {
      "location": "data/kitti/processed",
      "objects": {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2
      },
      "geometry": {
        "x_min": 0.0,
        "x_max": 70.4,
        "x_res": 0.1,
        "y_min": -40.0,
        "y_max": 40.0,
        "y_res": 0.1,
        "z_min": -2.5,
        "z_max": 1.0,
        "z_res": 0.1
      }
    }
  },
  "model": {
    "backbone": "mobilepixor",
    "backbone_out_dim": 16,
    "cls_encoding": "gaussian",
    "scale_gated_fpn": false,
    "header_use_bn": false,
    "header_act": "none",
    "header_use_iou": false
  },
  "loss": {
    "name": "oga",
    "temperature": 2.0,
    "clamp_bound": 3.0,
    "corner_beta": 1.0,
    "ema_momentum": 0.99,
    "epsilon": 1e-06,
    "max_abs_log_size": 10.0
  },
  "augmentation": {
    "p": 0.0
  },
  "train": {
    "data": "splits/kitti/train.txt",
    "epochs": 100,
    "physical_batch_size": 16,
    "accumulation_steps": 1,
    "precision": "bf16",
    "optimizer": "adamw",
    "learning_rate": 0.002,
    "min_lr": 1e-06,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "save_every": 5
  },
  "val": {
    "data": "splits/kitti/val.txt",
    "physical_batch_size": 2,
    "val_every": 1
  }
}
```

**Config 02: `configs/kitti/mobilepixor_ablation/02_mobilepixor_oga_sgfpn.json`**
(Same as 01, but `"scale_gated_fpn": true`):
```json
{
  "date": "reproducible",
  "ver": 1,
  "note": "02_mobilepixor_oga_sgfpn",
  "seed": 42,
  "device": "cuda",
  "multi_gpu": false,
  "data": {
    "location": "data/kitti/processed",
    "num_classes": 3,
    "out_size_factor": 4,
    "gaussian_overlap": 0.1,
    "min_radius": 4,
    "bev_encoding": {
      "name": "binary_slices"
    },
    "kitti": {
      "location": "data/kitti/processed",
      "objects": {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2
      },
      "geometry": {
        "x_min": 0.0,
        "x_max": 70.4,
        "x_res": 0.1,
        "y_min": -40.0,
        "y_max": 40.0,
        "y_res": 0.1,
        "z_min": -2.5,
        "z_max": 1.0,
        "z_res": 0.1
      }
    }
  },
  "model": {
    "backbone": "mobilepixor",
    "backbone_out_dim": 16,
    "cls_encoding": "gaussian",
    "scale_gated_fpn": true,
    "header_use_bn": false,
    "header_act": "none",
    "header_use_iou": false
  },
  "loss": {
    "name": "oga",
    "temperature": 2.0,
    "clamp_bound": 3.0,
    "corner_beta": 1.0,
    "ema_momentum": 0.99,
    "epsilon": 1e-06,
    "max_abs_log_size": 10.0
  },
  "augmentation": {
    "p": 0.0
  },
  "train": {
    "data": "splits/kitti/train.txt",
    "epochs": 100,
    "physical_batch_size": 16,
    "accumulation_steps": 1,
    "precision": "bf16",
    "optimizer": "adamw",
    "learning_rate": 0.002,
    "min_lr": 1e-06,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "save_every": 5
  },
  "val": {
    "data": "splits/kitti/val.txt",
    "physical_batch_size": 2,
    "val_every": 1
  }
}
```

**Config 03: `configs/kitti/mobilepixor_ablation/03_mobilepixor_oga_sgfpn_iqa.json`**
(Same as 02, but `"header_use_bn": true`, `"header_act": "silu"`, `"header_use_iou": true`, `"loss.use_iou": true`, `"nms_alpha": 0.5`):
```json
{
  "date": "reproducible",
  "ver": 1,
  "note": "03_mobilepixor_oga_sgfpn_iqa",
  "seed": 42,
  "device": "cuda",
  "multi_gpu": false,
  "nms_alpha": 0.5,
  "data": {
    "location": "data/kitti/processed",
    "num_classes": 3,
    "out_size_factor": 4,
    "gaussian_overlap": 0.1,
    "min_radius": 4,
    "nms_alpha": 0.5,
    "bev_encoding": {
      "name": "binary_slices"
    },
    "kitti": {
      "location": "data/kitti/processed",
      "objects": {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2
      },
      "geometry": {
        "x_min": 0.0,
        "x_max": 70.4,
        "x_res": 0.1,
        "y_min": -40.0,
        "y_max": 40.0,
        "y_res": 0.1,
        "z_min": -2.5,
        "z_max": 1.0,
        "z_res": 0.1
      }
    }
  },
  "model": {
    "backbone": "mobilepixor",
    "backbone_out_dim": 16,
    "cls_encoding": "gaussian",
    "scale_gated_fpn": true,
    "header_use_bn": true,
    "header_act": "silu",
    "header_use_iou": true
  },
  "loss": {
    "name": "oga",
    "temperature": 2.0,
    "clamp_bound": 3.0,
    "corner_beta": 1.0,
    "ema_momentum": 0.99,
    "epsilon": 1e-06,
    "max_abs_log_size": 10.0,
    "use_iou": true,
    "iou_target_type": "mgiou",
    "iou_loss_weight": 1.0
  },
  "augmentation": {
    "p": 0.0
  },
  "train": {
    "data": "splits/kitti/train.txt",
    "epochs": 100,
    "physical_batch_size": 16,
    "accumulation_steps": 1,
    "precision": "bf16",
    "optimizer": "adamw",
    "learning_rate": 0.002,
    "min_lr": 1e-06,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "save_every": 5
  },
  "val": {
    "data": "splits/kitti/val.txt",
    "physical_batch_size": 2,
    "val_every": 1
  }
}
```

**Config 04: `configs/kitti/mobilepixor_ablation/04_mobilepixor_oga_sgfpn_iqa_rich8.json`**
(Same as 03, but `"bev_encoding": {"name": "rich8", "density_norm": 32, "intensity_scale": 1}`):
```json
{
  "date": "reproducible",
  "ver": 1,
  "note": "04_mobilepixor_oga_sgfpn_iqa_rich8",
  "seed": 42,
  "device": "cuda",
  "multi_gpu": false,
  "nms_alpha": 0.5,
  "data": {
    "location": "data/kitti/processed",
    "num_classes": 3,
    "out_size_factor": 4,
    "gaussian_overlap": 0.1,
    "min_radius": 4,
    "nms_alpha": 0.5,
    "bev_encoding": {
      "name": "rich8",
      "density_norm": 32,
      "intensity_scale": 1
    },
    "kitti": {
      "location": "data/kitti/processed",
      "objects": {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2
      },
      "geometry": {
        "x_min": 0.0,
        "x_max": 70.4,
        "x_res": 0.1,
        "y_min": -40.0,
        "y_max": 40.0,
        "y_res": 0.1,
        "z_min": -2.5,
        "z_max": 1.0,
        "z_res": 0.1
      }
    }
  },
  "model": {
    "backbone": "mobilepixor",
    "backbone_out_dim": 16,
    "cls_encoding": "gaussian",
    "scale_gated_fpn": true,
    "header_use_bn": true,
    "header_act": "silu",
    "header_use_iou": true
  },
  "loss": {
    "name": "oga",
    "temperature": 2.0,
    "clamp_bound": 3.0,
    "corner_beta": 1.0,
    "ema_momentum": 0.99,
    "epsilon": 1e-06,
    "max_abs_log_size": 10.0,
    "use_iou": true,
    "iou_target_type": "mgiou",
    "iou_loss_weight": 1.0
  },
  "augmentation": {
    "p": 0.0
  },
  "train": {
    "data": "splits/kitti/train.txt",
    "epochs": 100,
    "physical_batch_size": 16,
    "accumulation_steps": 1,
    "precision": "bf16",
    "optimizer": "adamw",
    "learning_rate": 0.002,
    "min_lr": 1e-06,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "save_every": 5
  },
  "val": {
    "data": "splits/kitti/val.txt",
    "physical_batch_size": 2,
    "val_every": 1
  }
}
```

**Config 05: `configs/kitti/mobilepixor_ablation/05_mobilepixor_full_pcu.json`**
(Same as 04, but with PCU Augmentation enabled):
```json
{
  "date": "reproducible",
  "ver": 1,
  "note": "05_mobilepixor_full_pcu_rich8_sgfpn_iqa_oga",
  "seed": 42,
  "device": "cuda",
  "multi_gpu": false,
  "nms_alpha": 0.5,
  "data": {
    "location": "data/kitti/processed",
    "num_classes": 3,
    "out_size_factor": 4,
    "gaussian_overlap": 0.1,
    "min_radius": 4,
    "nms_alpha": 0.5,
    "bev_encoding": {
      "name": "rich8",
      "density_norm": 32,
      "intensity_scale": 1
    },
    "kitti": {
      "location": "data/kitti/processed",
      "objects": {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2
      },
      "geometry": {
        "x_min": 0.0,
        "x_max": 70.4,
        "x_res": 0.1,
        "y_min": -40.0,
        "y_max": 40.0,
        "y_res": 0.1,
        "z_min": -2.5,
        "z_max": 1.0,
        "z_res": 0.1
      }
    }
  },
  "model": {
    "backbone": "mobilepixor",
    "backbone_out_dim": 16,
    "cls_encoding": "gaussian",
    "scale_gated_fpn": true,
    "header_use_bn": true,
    "header_act": "silu",
    "header_use_iou": true
  },
  "loss": {
    "name": "oga",
    "temperature": 2.0,
    "clamp_bound": 3.0,
    "corner_beta": 1.0,
    "ema_momentum": 0.99,
    "epsilon": 1e-06,
    "max_abs_log_size": 10.0,
    "use_iou": true,
    "iou_target_type": "mgiou",
    "iou_loss_weight": 1.0
  },
  "augmentation": {
    "use_pcu_aug": true,
    "pcu_aug": {
      "enable_gt_sampling": true,
      "gt_database_path": "data/kitti/kitti_gt_database.pkl",
      "p": 0.5,
      "sample_counts": {
        "Car": 1,
        "Pedestrian": 3,
        "Cyclist": 3
      },
      "enable_physics": true,
      "enable_ground_validation": true,
      "enable_static_collision": true,
      "enable_line_of_sight": true,
      "enable_shadow_masking": true,
      "enable_density_subsample": true,
      "enable_radiometric_calibration": true,
      "min_visible_points": 5,
      "min_visible_ratio": 0.5
    },
    "flip_y": {
      "use": true,
      "p": 0.5
    },
    "rotation": {
      "use": true,
      "limit_angle": 20,
      "p": 1
    },
    "scaling": {
      "use": true,
      "range": [0.95, 1.05],
      "p": 1
    },
    "translation": {
      "use": true,
      "scale": 0.4,
      "p": 1
    }
  },
  "train": {
    "data": "splits/kitti/train.txt",
    "epochs": 100,
    "physical_batch_size": 16,
    "accumulation_steps": 1,
    "precision": "bf16",
    "optimizer": "adamw",
    "learning_rate": 0.002,
    "min_lr": 1e-06,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "save_every": 5
  },
  "val": {
    "data": "splits/kitti/val.txt",
    "physical_batch_size": 32,
    "val_every": 1
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_mobilepixor_ablation_configs.py
```
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add configs/kitti/mobilepixor_ablation/ tests/test_mobilepixor_ablation_configs.py
git commit -m "feat(configs): add 6-stage progressive ablation configs for MobilePIXOR"
```

---

### Task 4: Run Name Generation & Pipeline Parity Verification

**Files:**
- Test: `tests/test_run_name_and_logging.py`

**Interfaces:**
- Consumes: All JSON files under `configs/`.
- Verifies: All new configs generate valid standardized run names.

- [ ] **Step 1: Run the run name tests across all configs**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_run_name_and_logging.py
```
Expected: PASS (7 passed).

- [ ] **Step 2: Run the full test suite to guarantee zero regressions**

Run:
```bash
PYTHONPATH=detector /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/
```
Expected: All tests pass.

- [ ] **Step 3: Document ablation execution guide**

Add execution examples in the plan summary for how to launch each of the 6 ablation runs via `train.py`.
