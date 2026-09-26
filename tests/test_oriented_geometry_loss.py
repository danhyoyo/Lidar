import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.oriented_geometry_loss import (
    box_corners,
    pi_symmetric_corner_distance,
)


class TestOrientedGeometryCornerLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_identical_boxes_give_zero_distance(self):
        offset = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(1.6), math.log(4.0)]], dtype=torch.float32)
        yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        corners, axes, _, _ = box_corners(offset, log_size, yaw)
        loss = pi_symmetric_corner_distance(corners, corners, log_size)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_pi_rotation_invariance(self):
        offset = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32)
        theta1 = math.pi / 6
        theta2 = theta1 + math.pi
        yaw1 = torch.tensor([[math.cos(2 * theta1), math.sin(2 * theta1)]], dtype=torch.float32)
        yaw2 = torch.tensor([[math.cos(2 * theta2), math.sin(2 * theta2)]], dtype=torch.float32)

        corners1, _, _, _ = box_corners(offset, log_size, yaw1)
        corners2, _, _, _ = box_corners(offset, log_size, yaw2)

        tgt_corners, _, _, _ = box_corners(offset + 1.0, log_size, yaw1)
        loss1 = pi_symmetric_corner_distance(corners1, tgt_corners, log_size)
        loss2 = pi_symmetric_corner_distance(corners2, tgt_corners, log_size)
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=5)

    def test_non_overlapping_box_provides_finite_nonzero_gradient(self):
        pred_offset = torch.tensor([[0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        pred_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32, requires_grad=True)
        pred_yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32, requires_grad=True)

        tgt_offset = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
        tgt_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32)
        tgt_yaw = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        pred_c, _, _, _ = box_corners(pred_offset, pred_size, pred_yaw)
        tgt_c, _, _, _ = box_corners(tgt_offset, tgt_size, tgt_yaw)

        loss = pi_symmetric_corner_distance(pred_c, tgt_c, tgt_size)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(pred_offset.grad).all())
        self.assertFalse((pred_offset.grad == 0).all())


if __name__ == "__main__":
    unittest.main()
