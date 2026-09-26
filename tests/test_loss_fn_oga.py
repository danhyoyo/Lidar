import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))
sys.path.insert(0, str(REPO_ROOT / "detector" / "core" / "datasets"))

import torch
from core.losses.loss_fn import LossFunction


class TestLossFunctionOGA(unittest.TestCase):
    def setUp(self):
        self.B, self.H, self.W = 2, 16, 16
        self.pred = {
            "cls": torch.randn(self.B, 3, self.H, self.W, requires_grad=True),
            "offset": torch.randn(self.B, 2, self.H, self.W, requires_grad=True),
            "size": torch.randn(self.B, 2, self.H, self.W, requires_grad=True),
            "yaw": torch.randn(self.B, 2, self.H, self.W, requires_grad=True),
        }
        self.target = {
            "cls": torch.sigmoid(torch.randn(self.B, 3, self.H, self.W)),
            "offset": torch.zeros(self.B, 2, self.H, self.W),
            "size": torch.zeros(self.B, 2, self.H, self.W),
            "yaw": torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(self.B, 2, self.H, self.W),
            "reg_mask": torch.ones(self.B, self.H, self.W),
        }

    def test_oga_loss_initialization_and_forward(self):
        config = {
            "name": "oga",
            "temperature": 2.0,
            "clamp_bound": 3.0,
            "corner_beta": 1.0,
            "geometric_weight": 0.5,
        }
        criterion = LossFunction("gaussian", config)
        loss_dict = criterion(self.pred, self.target)

        for key in ("loss", "cls", "offset", "size", "yaw", "geo", "corner_dist", "proj_giou"):
            self.assertIn(key, loss_dict)
            val = loss_dict[key]
            self.assertTrue(torch.isfinite(val).all() if isinstance(val, torch.Tensor) else math.isfinite(val))

        for task in ("cls", "offset", "size", "yaw", "geo"):
            self.assertIn(f"weight_{task}", loss_dict)

        # Verify backward pass computes valid gradients for all prediction heads and criterion parameters
        loss_dict["loss"].backward()
        for head in ("cls", "offset", "size", "yaw"):
            self.assertTrue(self.pred[head].grad is not None)
            self.assertTrue(torch.isfinite(self.pred[head].grad).all())

        # Check criterion parameters
        params = list(criterion.parameters())
        self.assertGreater(len(params), 0)
        self.assertTrue(params[0].grad is not None)
        self.assertTrue(torch.isfinite(params[0].grad).all())

    def test_oga_loss_empty_mask(self):
        config = {
            "name": "oga",
            "temperature": 2.0,
            "clamp_bound": 3.0,
            "corner_beta": 1.0,
        }
        criterion = LossFunction("gaussian", config)
        empty_target = {
            "cls": self.target["cls"],
            "offset": self.target["offset"],
            "size": self.target["size"],
            "yaw": self.target["yaw"],
            "reg_mask": torch.zeros(self.B, self.H, self.W),
        }
        pred = {k: v.clone().detach().requires_grad_(True) for k, v in self.pred.items()}
        loss_dict = criterion(pred, empty_target)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))
        loss_dict["loss"].backward()
        for head in ("cls", "offset", "size", "yaw"):
            self.assertTrue(pred[head].grad is not None)
            self.assertTrue(torch.isfinite(pred[head].grad).all())

    def test_oga_loss_autocast_bf16(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for BF16 test")
        device = torch.device("cuda")
        pred = {k: v.to(device).detach().requires_grad_(True) for k, v in self.pred.items()}
        target = {k: v.to(device) for k, v in self.target.items()}
        config = {
            "name": "oga",
            "temperature": 2.0,
            "clamp_bound": 3.0,
            "corner_beta": 1.0,
        }
        criterion = LossFunction("gaussian", config).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_dict = criterion(pred, target)
            loss = loss_dict["loss"]
            self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for head in ("cls", "offset", "size", "yaw"):
            self.assertTrue(pred[head].grad is not None)
            self.assertTrue(torch.isfinite(pred[head].grad).all())


if __name__ == "__main__":
    unittest.main()
