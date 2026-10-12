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

    def test_multiaxis_projection_giou_identical(self):
        offset = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        log_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32)
        yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        corners, axes, _, _ = box_corners(offset, log_size, yaw)

        from core.losses.oriented_geometry_loss import multiaxis_projection_giou
        loss = multiaxis_projection_giou(corners, axes, corners, axes)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_oriented_geometry_loss_empty_mask(self):
        from core.losses.oriented_geometry_loss import OrientedGeometryLoss
        module = OrientedGeometryLoss(beta=1.0)
        B, H, W = 2, 8, 8
        pred_offset = torch.zeros((B, 2, H, W), requires_grad=True)
        pred_size = torch.zeros((B, 2, H, W), requires_grad=True)
        pred_yaw = torch.zeros((B, 2, H, W), requires_grad=True)
        tgt_offset = torch.zeros((B, 2, H, W))
        tgt_size = torch.zeros((B, 2, H, W))
        tgt_yaw = torch.zeros((B, 2, H, W))
        reg_mask = torch.zeros((B, H, W))

        loss, metrics = module(pred_offset, pred_size, pred_yaw, tgt_offset, tgt_size, tgt_yaw, reg_mask)
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(pred_offset.grad is not None)
        self.assertEqual(pred_offset.grad.sum().item(), 0.0)

    def test_oriented_geometry_loss_positive_mask(self):
        from core.losses.oriented_geometry_loss import OrientedGeometryLoss
        module = OrientedGeometryLoss(beta=1.0)
        B, H, W = 2, 4, 4
        pred_offset = torch.randn((B, 2, H, W), requires_grad=True)
        pred_size = torch.randn((B, 2, H, W), requires_grad=True)
        pred_yaw = torch.randn((B, 2, H, W), requires_grad=True)
        tgt_offset = torch.zeros((B, 2, H, W))
        tgt_size = torch.zeros((B, 2, H, W))
        tgt_yaw = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W)
        reg_mask = torch.ones((B, H, W))

        loss, metrics = module(pred_offset, pred_size, pred_yaw, tgt_offset, tgt_size, tgt_yaw, reg_mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        # Check telemetry metrics are on-device tensors
        for key in ("proj_giou", "corner_dist", "clamp_count", "fallback_count"):
            self.assertIn(key, metrics)
            self.assertTrue(torch.is_tensor(metrics[key]))
            self.assertTrue(torch.isfinite(metrics[key]).all())

        loss.backward()
        for head in (pred_offset, pred_size, pred_yaw):
            self.assertTrue(head.grad is not None)
            self.assertTrue(torch.isfinite(head.grad).all())
            self.assertFalse((head.grad == 0).all())

    def test_multiaxis_projection_giou_disjoint_boxes_finite_gradients(self):
        from core.losses.oriented_geometry_loss import multiaxis_projection_giou

        pred_offset = torch.tensor([[0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        pred_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32, requires_grad=True)
        pred_yaw = torch.tensor([[1.0, 0.0]], dtype=torch.float32, requires_grad=True)

        tgt_offset = torch.tensor([[20.0, 20.0]], dtype=torch.float32)
        tgt_size = torch.tensor([[math.log(2.0), math.log(4.0)]], dtype=torch.float32)
        tgt_yaw = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        pred_c, pred_ax, _, _ = box_corners(pred_offset, pred_size, pred_yaw)
        tgt_c, tgt_ax, _, _ = box_corners(tgt_offset, tgt_size, tgt_yaw)

        loss = multiaxis_projection_giou(pred_c, pred_ax, tgt_c, tgt_ax)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 1.0)

        loss.backward()
        self.assertTrue(pred_offset.grad is not None and torch.isfinite(pred_offset.grad).all())
        self.assertFalse((pred_offset.grad == 0).all())


if __name__ == "__main__":
    unittest.main()
