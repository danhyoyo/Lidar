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
