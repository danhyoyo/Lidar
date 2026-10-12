import math
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.smooth_corner_loss import (
    smooth_pi_symmetric_corner_distance,
    compute_range_weights,
)


class TestSmoothCornerLoss(unittest.TestCase):
    def test_identical_corners_yield_near_zero(self):
        corners = torch.tensor([
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
        ], dtype=torch.float32)
        target_log_size = torch.tensor([[math.log(2.0), math.log(2.0)]], dtype=torch.float32)
        loss = smooth_pi_symmetric_corner_distance(corners, corners, target_log_size, tau=0.05)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 1e-3)

    def test_smooth_transition_around_pi_half(self):
        corners_a = torch.tensor([
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
        ], dtype=torch.float32, requires_grad=True)
        corners_b = torch.tensor([
            [[1.001, 1.0], [1.0, -1.001], [-1.0, 1.001], [-1.001, -1.0]]
        ], dtype=torch.float32, requires_grad=True)
        target_log_size = torch.tensor([[math.log(2.0), math.log(2.0)]], dtype=torch.float32)

        loss_a = smooth_pi_symmetric_corner_distance(corners_a, corners_a, target_log_size, tau=0.05)
        loss_b = smooth_pi_symmetric_corner_distance(corners_b, corners_a, target_log_size, tau=0.05)
        loss_b.backward()
        self.assertTrue(corners_b.grad is not None)
        self.assertTrue(torch.isfinite(corners_b.grad).all())

    def test_compute_range_weights_scaling(self):
        offsets = torch.tensor([[0.0, 0.0], [35.2, 0.0], [70.4, 0.0]], dtype=torch.float32)
        weights = compute_range_weights(offsets, r_max=70.4, gamma=1.5, alpha=2.0)
        self.assertAlmostEqual(float(weights[0]), 1.0, places=4)
        self.assertAlmostEqual(float(weights[1]), 1.0 + 1.5 * 0.25, places=4)
        self.assertAlmostEqual(float(weights[2]), 2.5, places=4)

    def test_empty_corners_return_zero(self):
        empty_corners = torch.empty((0, 4, 2), dtype=torch.float32)
        empty_size = torch.empty((0, 2), dtype=torch.float32)
        loss = smooth_pi_symmetric_corner_distance(empty_corners, empty_corners, empty_size)
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
