import math
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.gaussian_wasserstein import (
    box_covariance_2d,
    closed_form_2d_gaussian_wasserstein,
    bounded_gwa_loss,
)


class TestGaussianWasserstein(unittest.TestCase):
    def test_identical_distributions_have_zero_distance(self):
        offset = torch.tensor([[10.0, 5.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32)
        yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        cov, _ = box_covariance_2d(log_size, yaw)
        w2_sq = closed_form_2d_gaussian_wasserstein(offset, cov, offset, cov)
        self.assertTrue(torch.isfinite(w2_sq).all())
        self.assertAlmostEqual(float(w2_sq.item()), 0.0, places=5)

    def test_translation_only_matches_euclidean_distance(self):
        offset_p = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        offset_t = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(2.0), math.log(2.0)]], dtype=torch.float32)
        yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        cov_p, _ = box_covariance_2d(log_size, yaw)
        cov_t, _ = box_covariance_2d(log_size, yaw)
        w2_sq = closed_form_2d_gaussian_wasserstein(offset_p, cov_p, offset_t, cov_t)
        # Expected: 3^2 + 4^2 = 25.0
        self.assertAlmostEqual(float(w2_sq.item()), 25.0, places=4)

    def test_rotation_invariance_pi(self):
        offset = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(1.0), math.log(3.0)]], dtype=torch.float32)
        theta = 0.42
        yaw_a = torch.tensor([[math.cos(2.0 * theta), math.sin(2.0 * theta)]], dtype=torch.float32)
        yaw_b = torch.tensor([[math.cos(2.0 * (theta + math.pi)), math.sin(2.0 * (theta + math.pi))]], dtype=torch.float32)

        cov_a, _ = box_covariance_2d(log_size, yaw_a)
        cov_b, _ = box_covariance_2d(log_size, yaw_b)
        self.assertTrue(torch.allclose(cov_a, cov_b, atol=1e-5))

    def test_finite_gradients_under_extreme_aspect_ratios(self):
        offset = torch.tensor([[0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        log_size_p = torch.tensor([[math.log(0.1), math.log(10.0)]], dtype=torch.float32, requires_grad=True)
        yaw_p = torch.tensor([[0.5, 0.866]], dtype=torch.float32, requires_grad=True)

        target_offset = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_size = torch.tensor([[math.log(0.1), math.log(10.0)]], dtype=torch.float32)
        target_yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        cov_p, _ = box_covariance_2d(log_size_p, yaw_p)
        cov_t, _ = box_covariance_2d(target_size, target_yaw)
        w2_sq = closed_form_2d_gaussian_wasserstein(offset, cov_p, target_offset, cov_t)
        loss = bounded_gwa_loss(w2_sq, target_size)

        loss.backward()
        self.assertTrue(torch.isfinite(offset.grad).all())
        self.assertTrue(torch.isfinite(log_size_p.grad).all())
        self.assertTrue(torch.isfinite(yaw_p.grad).all())

    def test_empty_wasserstein_returns_zero(self):
        offset = torch.empty((0, 2), dtype=torch.float32)
        cov = torch.empty((0, 2, 2), dtype=torch.float32)
        w2_sq = closed_form_2d_gaussian_wasserstein(offset, cov, offset, cov)
        self.assertEqual(w2_sq.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
