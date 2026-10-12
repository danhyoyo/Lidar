"""Exact quality must match polygon IoU, rather than a projection proxy."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detector"))
from core.losses import iou_targets


def encoded_boxes(boxes):
    """Rows are x, y, width, length, theta; maps are [1, 2, 1, N]."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    centers = boxes[:, :2].T.reshape(1, 2, 1, -1)
    sizes = boxes[:, 2:4].log().T.reshape(1, 2, 1, -1)
    yaw = torch.stack((torch.cos(2 * boxes[:, 4]), torch.sin(2 * boxes[:, 4])))
    return centers, sizes, yaw.reshape(1, 2, 1, -1)


def reference_polygon(box):
    x, y, width, length, angle = map(float, box)
    c, s = math.cos(angle), math.sin(angle)
    return Polygon([
        (x + dx * c - dy * s, y + dx * s + dy * c)
        for dx, dy in ((-length / 2, width / 2), (-length / 2, -width / 2),
                       (length / 2, -width / 2), (length / 2, width / 2))
    ])


class TestRotatedIoUTargets(unittest.TestCase):
    def quality(self, pred_boxes, target_boxes, mask=None):
        self.assertTrue(hasattr(iou_targets, "compute_rotated_iou_targets"),
                        "Exact rotated BEV IoU target generator is missing")
        pred, target = encoded_boxes(pred_boxes), encoded_boxes(target_boxes)
        if mask is None:
            mask = torch.ones(1, 1, len(pred_boxes), dtype=torch.bool)
        return iou_targets.compute_rotated_iou_targets(*pred, *target, mask)

    def test_identity_disjoint_touching_containment_and_crossing(self):
        pred = [[0, 0, 1, 1, 0], [1.5, 0, 1, 1, 0], [1, 0, 1, 1, 0],
                [0, 0, 1, 1, 0], [0, 0, .5, 4, math.pi / 2]]
        target = [[0, 0, 1, 1, 0]] * 3 + [[0, 0, 2, 2, 0], [0, 0, .5, 4, 0]]
        actual = self.quality(pred, target).flatten()
        torch.testing.assert_close(actual, torch.tensor([1., 0., 0., .25, 1 / 15]),
                                   atol=2e-6, rtol=2e-6)

    def test_rotated_square_is_not_gaussian_identity(self):
        q = self.quality([[0, 0, 1, 1, math.pi / 4]], [[0, 0, 1, 1, 0]])
        self.assertAlmostEqual(q.item(), 1 / math.sqrt(2), places=6)

    def test_pi_and_width_length_equivalence(self):
        box = [1, -2, .6, 4, .37]
        pred = [box, [*box[:4], box[4] + math.pi],
                [*box[:2], box[3], box[2], box[4] + math.pi / 2]]
        torch.testing.assert_close(self.quality(pred, [box] * 3), torch.ones(1, 1, 3),
                                   atol=2e-6, rtol=2e-6)

    def test_scale_and_common_translation(self):
        pred = np.array([[.3, .1, .5, 2, .3]], dtype=np.float32)
        target = np.array([[0, 0, .6, 2.2, -.1]], dtype=np.float32)
        expected = self.quality(pred, target).item()
        for scale in (1e-4, .1, 1000):
            p, t = pred.copy(), target.copy()
            p[:, :4] *= scale
            t[:, :4] *= scale
            self.assertAlmostEqual(self.quality(p, t).item(), expected, places=5)
        p, t = pred.copy(), target.copy()
        p[:, :2] += [50, -20]
        t[:, :2] += [50, -20]
        self.assertAlmostEqual(self.quality(p, t).item(), expected, places=5)

    def test_randomized_against_evaluator_polygon_convention(self):
        rng = np.random.default_rng(42)
        n = 512
        pred = np.column_stack((rng.uniform(-2, 2, (n, 2)),
                                np.exp(rng.uniform(-2, 2, (n, 2))),
                                rng.uniform(-math.pi, math.pi, n)))
        target = np.column_stack((rng.uniform(-2, 2, (n, 2)),
                                  np.exp(rng.uniform(-2, 2, (n, 2))),
                                  rng.uniform(-math.pi, math.pi, n)))
        expected = []
        for p, t in zip(pred, target):
            a, b = reference_polygon(p), reference_polygon(t)
            intersection = a.intersection(b).area
            expected.append(intersection / (a.area + b.area - intersection))
        np.testing.assert_allclose(self.quality(pred, target).flatten().numpy(),
                                   expected, atol=3e-6, rtol=3e-5)

    def test_parallel_near_parallel_and_extreme_aspect(self):
        for angle in (0., 1e-7, 1e-5, .2):
            pred, target = [0, 0, .01, 10, angle], [.001, 0, .01, 10, 0]
            a, b = reference_polygon(pred), reference_polygon(target)
            overlap = a.intersection(b).area
            expected = overlap / (a.area + b.area - overlap)
            self.assertAlmostEqual(self.quality([pred], [target]).item(), expected, places=5)

    def test_sparse_and_empty_masks(self):
        boxes = [[0, 0, 1, 1, 0]] * 3
        mask = torch.tensor([[[True, False, True]]])
        torch.testing.assert_close(self.quality(boxes, boxes, mask),
                                   torch.tensor([[[1., 0., 1.]]]))
        torch.testing.assert_close(self.quality(boxes, boxes, mask & False),
                                   torch.zeros(1, 1, 3))

    def test_detached_fp32_under_bf16_and_zero_yaw(self):
        self.assertTrue(hasattr(iou_targets, "compute_rotated_iou_targets"))
        pred = tuple(x.bfloat16().detach().requires_grad_() for x in
                     encoded_boxes([[0, 0, 1, 1, math.pi / 4]]))
        target = encoded_boxes([[0, 0, 1, 1, 0]])
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            q = iou_targets.compute_rotated_iou_targets(*pred, *target, mask)
        self.assertEqual(q.dtype, torch.float32)
        self.assertFalse(q.requires_grad)
        self.assertAlmostEqual(q.item(), 1 / math.sqrt(2), places=6)
        zero_yaw = (*target[:2], torch.zeros_like(target[2]))
        q = iou_targets.compute_rotated_iou_targets(*zero_yaw, *target, mask)
        self.assertAlmostEqual(q.item(), 1., places=6)


if __name__ == "__main__":
    unittest.main()
