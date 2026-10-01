import json
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))
sys.path.insert(0, str(REPO_ROOT / "detector" / "core" / "datasets"))

import torch
from core.losses.loss_fn import LossFunction


class TestQOgaIntegration(unittest.TestCase):
    def test_json_config_loading_and_backward(self):
        cfg_path = REPO_ROOT / "configs" / "kitti" / "oga_loss" / "kitti_mobilepixornext_q_oga.json"
        self.assertTrue(cfg_path.exists(), f"Missing config: {cfg_path}")
        with open(cfg_path) as f:
            cfg = json.load(f)

        self.assertEqual(cfg["loss"]["name"], "q_oga")
        criterion = LossFunction("gaussian", cfg["loss"])
        B, H, W = 2, 16, 16
        pred = {
            "cls": torch.randn(B, 3, H, W, requires_grad=True),
            "offset": torch.randn(B, 2, H, W, requires_grad=True),
            "size": torch.randn(B, 2, H, W, requires_grad=True),
            "yaw": torch.randn(B, 2, H, W, requires_grad=True),
        }
        target = {
            "cls": torch.sigmoid(torch.randn(B, 3, H, W)),
            "offset": torch.zeros(B, 2, H, W),
            "size": torch.zeros(B, 2, H, W),
            "yaw": torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W),
            "reg_mask": torch.ones(B, H, W),
        }
        loss_dict = criterion(pred, target)
        loss = loss_dict["loss"]
        loss.backward()
        self.assertTrue(pred["offset"].grad is not None)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
