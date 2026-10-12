import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))
sys.path.insert(0, str(REPO_ROOT / "detector" / "core" / "datasets"))

import torch
from core.losses.strategies import build_loss_strategy


class TestMetricRDA(unittest.TestCase):

    def test_q_oga_differentiates_near_vs_far_rda(self):
        """Q-OGA must also compute RDA weights from true metric coordinates."""
        strategy = build_loss_strategy("q_oga", "gaussian", {
            "name": "q_oga",
            "rda_gamma": 1.5,
            "rda_alpha": 2.0,
            "x_min": 0.0,
            "x_max": 70.4,
            "y_min": -40.0,
            "y_max": 40.0,
        })

        B, H, W = 1, 200, 176
        pred_far = {
            "cls": torch.zeros(B, 3, H, W, requires_grad=True),
            "offset": torch.zeros(B, 2, H, W, requires_grad=True),
            "size": torch.zeros(B, 2, H, W, requires_grad=True),
            "yaw": torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W).contiguous().requires_grad_(True),
        }
        target_far = {
            "cls": torch.zeros(B, 3, H, W),
            "offset": torch.zeros(B, 2, H, W),
            "size": torch.zeros(B, 2, H, W),
            "yaw": torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W),
            "reg_mask": torch.zeros(B, H, W),
        }
        target_far["reg_mask"][0, 100, 150] = 1.0
        target_far["cls"][0, 0, 100, 150] = 1.0

        out_far = strategy(pred_far, target_far)
        w_far = float(out_far["rda_mean_weight"].item())
        self.assertAlmostEqual(w_far, 2.09, delta=0.15)


if __name__ == "__main__":
    unittest.main()
