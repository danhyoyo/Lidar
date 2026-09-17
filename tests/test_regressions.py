import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "tools" / "kitti_training_pipeline"),
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
]

from core.datasets.dataset import Dataset
from core.datasets.utils_1.gaussian import gaussian_radius
from core.datasets.utils_1.preprocess import voxel_to_points, voxelize
from core.losses.focal_loss import focal_loss
from core.models.backbones.mobilepixor import conv3x3_dw
from evaluate_kitti_bev import device_timed, run_evaluation
from postprocess import filter_pred
from prepare_kitti import make_split


GEOMETRY = {
    "x_min": 0.0,
    "x_max": 3.0,
    "x_res": 1.0,
    "y_min": 0.0,
    "y_max": 3.0,
    "y_res": 1.0,
    "z_min": 0.0,
    "z_max": 4.0,
    "z_res": 1.0,
}


def predictions(classes=2, height=3, width=3):
    return {
        "cls": torch.full((1, classes, height, width), -20.0),
        "offset": torch.zeros((1, 2, height, width)),
        "size": torch.full((1, 2, height, width), math.log(4.0)),
        "yaw": torch.cat((
            torch.ones((1, 1, height, width)),
            torch.zeros((1, 1, height, width)),
        ), dim=1),
    }


class PostprocessTests(unittest.TestCase):
    def test_nms_is_class_aware(self):
        pred = predictions()
        pred["cls"][0, 0, 0, 0] = 10.0
        pred["cls"][0, 1, 0, 2] = 9.0

        boxes = filter_pred(
            pred, {"geometry": GEOMETRY}, out_size_factor=1,
            thres=0.8, nms_thres=0.1,
        )

        self.assertEqual(boxes.shape, (2, 7))
        self.assertEqual(set(boxes[:, 0]), {0.0, 1.0})

    def test_single_class_and_empty_shapes(self):
        pred = predictions(classes=1, height=1, width=1)
        pred["cls"][0, 0, 0, 0] = 10.0
        geometry = dict(GEOMETRY)
        geometry.update(x_max=1.0, y_max=1.0)
        boxes = filter_pred(
            pred, {"geometry": geometry}, 1, thres=0.8, nms_thres=None
        )
        self.assertEqual(boxes.shape, (1, 7))
        empty = filter_pred(
            predictions(classes=1, height=1, width=1),
            {"geometry": geometry}, 1, thres=0.8, nms_thres=None,
        )
        self.assertEqual(empty.shape, (0, 7))
        self.assertEqual(empty.dtype, np.float32)


class DataUtilityTests(unittest.TestCase):
    def test_voxel_centres_round_trip_with_axis_order(self):
        point = np.array([[1.2, 2.2, 3.2, 0.5]], dtype=np.float32)
        voxel = voxelize(point, GEOMETRY)
        restored = voxel_to_points(voxel, GEOMETRY)
        np.testing.assert_allclose(restored, [[1.5, 2.5, 3.5]])

    def test_gaussian_radius_accepts_numbers_and_tensors(self):
        numeric = gaussian_radius((4.0, 8.0), 0.1)
        tensor = gaussian_radius((torch.tensor(4.0), torch.tensor(8.0)), 0.1)
        self.assertGreater(numeric, 0)
        self.assertAlmostEqual(numeric, tensor)

    def test_label_parser_handles_whitespace_and_empty_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "label").mkdir()
            (root / "label" / "000001.txt").write_text(
                "Car   1.5  1.6 3.7 10 0 -1 0\n", encoding="utf-8"
            )
            dataset = Dataset.__new__(Dataset)
            dataset.data_list = ["000001"]
            dataset.data_type_list = ["kitti"]
            dataset.config = {
                "kitti": {
                    "location": str(root),
                    "objects": {"Car": 0},
                    "geometry": GEOMETRY,
                }
            }
            boxes = dataset.get_boxes(0)
            self.assertEqual(boxes.shape, (1, 8))
            self.assertEqual(dataset.filter_boxes(boxes, "kitti").shape, (0, 8))


class RuntimeAndLossTests(unittest.TestCase):
    def test_cpu_timer(self):
        value, elapsed = device_timed(lambda: 42, torch.device("cpu"))
        self.assertEqual(value, 42)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_depthwise_helper_allows_different_output_channels(self):
        layer = conv3x3_dw(4, 6)
        self.assertEqual(layer(torch.zeros(1, 4, 8, 8)).shape, (1, 6, 8, 8))

    def test_fractional_focal_class_weights_are_applied(self):
        logits = torch.zeros((1, 2, 1, 1))
        target = torch.zeros((1, 1, 1), dtype=torch.int64)
        baseline = focal_loss(logits, target)
        weighted = focal_loss(logits, target, alphas=(1.5, 1.0))
        self.assertGreater(weighted, baseline)

    def test_invalid_evaluation_options_fail_before_io(self):
        with self.assertRaisesRegex(ValueError, "Unsupported backend"):
            run_evaluation(
                name="bad", backend="other", model_path=Path("missing"),
                config_path=Path("missing"), detector_root=Path("missing"),
                kitti_root=Path("missing"), split_path=Path("missing"),
            )

    def test_random_split_rejects_empty_partition(self):
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            make_split(["a", "b"], 2, 42, None, None)


if __name__ == "__main__":
    unittest.main()
