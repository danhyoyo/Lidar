import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
import torch.nn as nn
from core.models.backbones.registry import (
    register_backbone,
    build_backbone,
    get_available_backbones,
)
from core.models.model import CustomModel
from core.models.backbones.bevnext import BEVNeXtBackbone
from core.models.backbones.mobilepixor import MobilePixorBackBone


class TestBackboneRegistry(unittest.TestCase):
    def test_default_backbones_registered(self):
        available = get_available_backbones()
        for name in ("bevnext", "mobilepixor", "mobilepixor_coordatt", "pixor", "rpn"):
            self.assertIn(name, available)

    def test_unsupported_backbone_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            build_backbone("nonexistent_backbone", {})
        self.assertIn("Unsupported backbone", str(ctx.exception))
        self.assertIn("Available backbones", str(ctx.exception))

    def test_build_registered_backbones(self):
        bb_pixor = build_backbone("mobilepixor", {}, input_channels=35)
        self.assertIsInstance(bb_pixor, MobilePixorBackBone)

        bb_bevnext = build_backbone(
            "bevnext",
            {"backbone_out_dim": 16, "c4_attention": "litemla", "scale_gated_fpn": True},
            input_channels=8,
        )
        self.assertIsInstance(bb_bevnext, BEVNeXtBackbone)

    def test_custom_backbone_registration_and_model_integration(self):
        @register_backbone("custom_dummy_test")
        def _build_dummy(cfg, input_channels=35):
            return nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=1),
                nn.Identity(),
            )

        self.assertIn("custom_dummy_test", get_available_backbones())

        # Test instantiation via CustomModel without touching model.py
        cfg = {
            "backbone": "custom_dummy_test",
            "backbone_out_dim": 16,
            "cls_encoding": "gaussian",
        }
        model = CustomModel(cfg, num_classes=3, input_channels=8)
        x = torch.randn(1, 8, 32, 32)
        out = model(x)
        self.assertIn("cls", out)
        self.assertEqual(out["cls"].shape, (1, 3, 32, 32))

    def test_duplicate_registration_raises_key_error(self):
        with self.assertRaises(KeyError):
            @register_backbone("bevnext")
            def _duplicate(cfg, in_c):
                return nn.Identity()


if __name__ == "__main__":
    unittest.main()
