import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))
sys.path.insert(0, str(REPO_ROOT / "detector" / "core" / "datasets"))

import torch
from core.losses.strategies import build_loss_strategy, get_available_loss_strategies
from core.losses.loss_fn import LossFunction


class TestLossStrategyGwQal(unittest.TestCase):
    def setUp(self):
        self.B, self.H, self.W = 2, 8, 8
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

    def test_gw_qal_registered(self):
        self.assertIn("gw_qal", get_available_loss_strategies())

    def test_gw_qal_forward_backward(self):
        criterion = LossFunction("gaussian", {
            "name": "gw_qal",
            "temperature": 2.0,
            "clamp_bound": 3.0,
            "tau_gwa": 2.0,
            "tau_sim": 2.0,
            "beta_q": 1.0,
            "rda_gamma": 1.5,
        })
        loss_dict = criterion(self.pred, self.target)
        for key in ("loss", "cls", "offset", "size", "yaw", "geo", "w2_mean_dist", "rda_mean_weight"):
            self.assertIn(key, loss_dict)
            self.assertTrue(torch.isfinite(loss_dict[key]))

        loss_dict["loss"].backward()
        for head in ("cls", "offset", "size", "yaw"):
            self.assertTrue(self.pred[head].grad is not None)
            self.assertTrue(torch.isfinite(self.pred[head].grad).all())

    def test_gw_qal_empty_mask(self):
        criterion = LossFunction("gaussian", {"name": "gw_qal"})
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


if __name__ == "__main__":
    unittest.main()
