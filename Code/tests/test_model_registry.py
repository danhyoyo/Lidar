import copy
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
import torch.nn as nn
from core.models.backbones.registry import (
    register_backbone,
    build_backbone,
    get_available_backbones,
)
from core.models.model import CustomModel
from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone
from core.models.backbones.mobilepixor import MobilePixorBackBone
from tools.kitti_training_pipeline.common import validate_backbone


class TestBackboneRegistry(unittest.TestCase):
    def test_default_backbones_registered(self):
        available = get_available_backbones()
        for name in ("mobilepixornext", "mobilepixor", "mobilepixor_coordatt", "pixor", "rpn"):
            self.assertIn(name, available)

    def test_unsupported_backbone_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            build_backbone("nonexistent_backbone", {})
        self.assertIn("Unsupported backbone", str(ctx.exception))
        self.assertIn("Available backbones", str(ctx.exception))

    def test_build_registered_backbones(self):
        bb_pixor = build_backbone("mobilepixor", {}, input_channels=35)
        self.assertIsInstance(bb_pixor, MobilePixorBackBone)

        bb_mobilepixornext = build_backbone(
            "mobilepixornext",
            {"backbone_out_dim": 16, "c4_attention": "none", "scale_gated_fpn": True},
            input_channels=8,
        )
        self.assertIsInstance(bb_mobilepixornext, MobilePixorNeXtBackbone)

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
        validate_backbone({"model": cfg})
        x = torch.randn(1, 8, 32, 32)
        out = model(x)
        self.assertIn("cls", out)
        self.assertEqual(out["cls"].shape, (1, 3, 32, 32))

    def test_duplicate_registration_raises_key_error(self):
        with self.assertRaises(KeyError):
            @register_backbone("mobilepixornext")
            def _duplicate(cfg, in_c):
                return nn.Identity()

    def test_allow_override_registration(self):
        @register_backbone("temp_override_test")
        def _builder1(cfg, in_c):
            return nn.Identity()

        # Không bật override -> lỗi KeyError
        with self.assertRaises(KeyError):
            @register_backbone("temp_override_test")
            def _builder2(cfg, in_c):
                return nn.Identity()

        # Bật allow_override=True -> thành công
        @register_backbone("temp_override_test", allow_override=True)
        def _builder3(cfg, in_c):
            return nn.ReLU()

        bb = build_backbone("temp_override_test", {})
        self.assertIsInstance(bb, nn.ReLU)

    def test_custom_model_case_insensitive_mobilepixornext_header(self):
        cfg_upper = {
            "backbone": "MobilePixorNeXt",
            "backbone_out_dim": 16,
            "cls_encoding": "gaussian",
        }
        model = CustomModel(cfg_upper, num_classes=3, input_channels=8)
        self.assertTrue(model.header.cls.bn1.__class__.__name__ == "BatchNorm2d")
        self.assertEqual(model.header.cls.act1.__class__.__name__, "SiLU")

    def test_custom_model_default_fallback_values(self):
        cfg_minimal = {"backbone": "mobilepixor"}
        model = CustomModel(cfg_minimal)
        self.assertIsNotNone(model)
        self.assertEqual(model.num_classes, 4)


def test_hist14_direct_model_uses_schema_width_and_legacy_head():
    cfg = {"backbone": "mobilepixornext", "backbone_out_dim": 16,
           "bev_encoding": {"name": "hist14"}, "c4_attention": "none"}
    model = CustomModel(cfg, num_classes=3, input_channels=35).eval()
    assert model.backbone.stem[0].in_channels == 14
    with torch.no_grad():
        result = model(torch.zeros(1, 14, 32, 48))
    assert result["cls"].shape == (1, 3, 8, 12)
    assert set(result) == {"cls", "offset", "size", "yaw"}


def test_pipeline_injects_authoritative_encoding_without_mutating_config():
    from tools.kitti_training_pipeline.common import build_model, input_shape
    geometry = {"x_min": 0, "x_max": 16, "x_res": 0.5,
                "y_min": -6, "y_max": 6, "y_res": 0.25,
                "z_min": -2.5, "z_max": 1, "z_res": 0.1}
    cfg = {"model": {"backbone": "mobilepixornext", "c4_attention": "none",
                     "bev_encoding": {"name": "rich8"}},
           "data": {"num_classes": 3, "bev_encoding": {"name": "hist14"},
                    "kitti": {"geometry": geometry}}}
    original = copy.deepcopy(cfg)
    model = build_model(cfg)
    assert model.backbone.stem[0].in_channels == input_shape(cfg)[1] == 14
    assert cfg == original


def test_binary_direct_constructor_preserves_explicit_geometry_free_width():
    model = CustomModel({"backbone": "mobilepixornext"}, num_classes=3, input_channels=8)
    assert model.backbone.stem[0].in_channels == 8


def test_registry_threads_nondefault_stage_depths_into_real_backbone():
    model = CustomModel({"backbone": "mobilepixornext", "stage_depths": [3, 4, 2],
                         "c4_attention": "none"}, num_classes=3, input_channels=14)
    assert [len(model.backbone.stage2), len(model.backbone.stage3), len(model.backbone.stage4)] == [3, 4, 2]


if __name__ == "__main__":
    unittest.main()
