import math
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))
sys.path.insert(0, str(REPO_ROOT / "detector" / "core" / "datasets"))

import torch
from core.losses.quality_focal_loss import quality_focal_loss
from core.losses.focal_loss import modified_focal_loss


class TestQualityFocalLossCQFL(unittest.TestCase):
    def test_exact_equivalence_to_centernet_when_quality_is_one(self):
        """When quality Q = 1.0 everywhere, CQFL must match CenterNet modified_focal_loss."""
        B, C, H, W = 2, 3, 16, 16
        logits = torch.randn(B, C, H, W, dtype=torch.float32, requires_grad=True)

        # Create realistic CenterNet Gaussian target
        target = torch.zeros(B, C, H, W, dtype=torch.float32)
        target[:, 0, 4, 4] = 1.0  # Center peak
        target[:, 0, 4, 5] = 0.8  # Gaussian neighbor
        target[:, 0, 5, 4] = 0.8
        target[:, 0, 5, 5] = 0.6
        target[:, 1, 10, 10] = 1.0  # Another center peak
        target[:, 1, 10, 11] = 0.7

        quality = torch.ones(B, H, W, dtype=torch.float32)

        loss_cqfl = quality_focal_loss(logits, target, quality, beta=1.0, gamma=2.0)
        loss_centernet = modified_focal_loss(logits, target)

        self.assertTrue(torch.isfinite(loss_cqfl))
        self.assertTrue(torch.isfinite(loss_centernet))
        self.assertAlmostEqual(float(loss_cqfl.item()), float(loss_centernet.item()), places=5)

    def test_suppression_of_gaussian_neighbor_plateau(self):
        """At neighbor locations with 0 < Y < 1.0, loss gradient must push p -> 0."""
        B, C, H, W = 1, 1, 3, 3
        # Neighbor pixel at (0, 0, 0, 1) has target 0.85
        target = torch.zeros(B, C, H, W, dtype=torch.float32)
        target[0, 0, 0, 0] = 1.0   # Center
        target[0, 0, 0, 1] = 0.85  # Gaussian neighbor

        # Model currently predicts non-zero probability at the neighbor
        logits = torch.zeros(B, C, H, W, dtype=torch.float32, requires_grad=True)
        quality = torch.ones(B, H, W, dtype=torch.float32)

        loss = quality_focal_loss(logits, target, quality)
        loss.backward()

        # For neighbor pixel (0, 1), target is 0 in CenterNet, so higher prediction should be penalized:
        # positive gradient w.r.t logit means increasing logit increases loss (pushes p -> 0)
        neighbor_grad = logits.grad[0, 0, 0, 1].item()
        self.assertGreater(neighbor_grad, 0.0, "Gradient at Gaussian neighbor must push prediction toward zero!")

    def test_quality_modulation_at_positive_peak(self):
        """At center peak (Y = 1.0), quality score Q scales the optimal prediction."""
        B, C, H, W = 1, 1, 1, 1
        target = torch.ones(B, C, H, W, dtype=torch.float32)
        q_val = 0.6
        quality = torch.tensor([[[q_val]]], dtype=torch.float32)

        # Logit corresponding to p = q_val
        logit_optimal = torch.logit(torch.tensor([[[[q_val]]]]), eps=1e-5).requires_grad_(True)
        loss = quality_focal_loss(logit_optimal, target, quality, beta=1.0, gamma=2.0)
        loss.backward()

        # At the exact optimum p = q^beta, focal BCE loss gradient should be approximately zero
        self.assertAlmostEqual(float(logit_optimal.grad.item()), 0.0, places=4)

    def test_empty_mask_stability(self):
        """When no objects are present (target = 0 everywhere), loss is finite and valid."""
        B, C, H, W = 2, 3, 8, 8
        logits = torch.randn(B, C, H, W, dtype=torch.float32, requires_grad=True)
        target = torch.zeros(B, C, H, W, dtype=torch.float32)
        quality = torch.zeros(B, H, W, dtype=torch.float32)

        loss = quality_focal_loss(logits, target, quality)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)

        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
