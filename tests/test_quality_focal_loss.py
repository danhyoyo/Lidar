import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.quality_focal_loss import quality_focal_loss


class TestQualityFocalLoss(unittest.TestCase):
    def test_quality_focal_loss_forward_and_backward(self):
        B, C, H, W = 2, 3, 4, 4
        logits = torch.randn(B, C, H, W, requires_grad=True)
        target = torch.zeros(B, C, H, W)
        target[:, 0, 1, 1] = 1.0  # Positive anchor
        quality = torch.ones(B, H, W) * 0.8

        loss = quality_focal_loss(logits, target, quality, beta=1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)

        loss.backward()
        self.assertTrue(logits.grad is not None)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_quality_focal_target_calibration(self):
        # When prediction is low (logits = -3.0), higher quality target produces higher loss
        B, C, H, W = 1, 1, 1, 1
        logits_low = torch.tensor([[[[-3.0]]]], requires_grad=False)
        target = torch.ones(B, C, H, W)
        q_high = torch.tensor([[[1.0]]])
        q_low = torch.tensor([[[0.1]]])

        loss_high = quality_focal_loss(logits_low, target, q_high, beta=1.0)
        loss_low = quality_focal_loss(logits_low, target, q_low, beta=1.0)
        self.assertGreater(float(loss_high.detach()), float(loss_low.detach()))

    def test_empty_positives_returns_background_focal_loss(self):
        B, C, H, W = 2, 2, 4, 4
        logits = torch.randn(B, C, H, W, requires_grad=True)
        target = torch.zeros(B, C, H, W)
        quality = torch.zeros(B, H, W)
        loss = quality_focal_loss(logits, target, quality)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(logits.grad is not None)


if __name__ == "__main__":
    unittest.main()
