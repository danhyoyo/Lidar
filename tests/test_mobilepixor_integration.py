import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "detector"))
sys.path.insert(0, str(REPO_ROOT / "detector" / "core" / "datasets"))

import torch
from core.models.model import CustomModel
from core.losses.loss_fn import LossFunction
from postprocess import filter_pred


class TestMobilePixorIntegration(unittest.TestCase):
    def test_mobilepixor_with_baseline_loss(self):
        cfg_model = {
            "backbone": "mobilepixor",
            "backbone_out_dim": 16,
            "cls_encoding": "gaussian",
            "header_use_bn": False,
            "header_act": "none",
        }
        model = CustomModel(cfg_model, num_classes=3, input_channels=35)
        criterion = LossFunction("gaussian", {"name": "baseline"})

        x = torch.randn(2, 35, 128, 128)
        pred = model(x)
        self.assertIn("cls", pred)
        self.assertNotIn("iou", pred)

        target = {
            "cls": torch.zeros(2, 3, 32, 32),
            "offset": torch.zeros(2, 2, 32, 32),
            "size": torch.zeros(2, 2, 32, 32),
            "yaw": torch.zeros(2, 2, 32, 32),
            "reg_mask": torch.ones(2, 32, 32),
        }
        loss_dict = criterion(pred, target)
        self.assertIn("loss", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))

    def test_mobilepixor_with_oga_loss_and_iqa_header(self):
        cfg_model = {
            "backbone": "mobilepixor",
            "backbone_out_dim": 16,
            "cls_encoding": "gaussian",
            "header_use_bn": True,
            "header_act": "silu",
            "header_use_iou": True,
            "scale_gated_fpn": True,
        }
        model = CustomModel(cfg_model, num_classes=3, input_channels=8)
        loss_cfg = {
            "name": "oga",
            "temperature": 2.0,
            "clamp_bound": 3.0,
            "corner_beta": 1.0,
            "use_iou": True,
            "iou_target_type": "mgiou",
            "iou_loss_weight": 1.0,
        }
        criterion = LossFunction("gaussian", loss_cfg)

        x = torch.randn(2, 8, 128, 128)
        pred = model(x)
        self.assertIn("iou", pred)
        self.assertEqual(pred["iou"].shape, (2, 1, 32, 32))

        target = {
            "cls": torch.zeros(2, 3, 32, 32),
            "offset": torch.zeros(2, 2, 32, 32),
            "size": torch.zeros(2, 2, 32, 32),
            "yaw": torch.zeros(2, 2, 32, 32),
            "reg_mask": torch.ones(2, 32, 32),
        }
        loss_dict = criterion(pred, target)
        self.assertIn("loss", loss_dict)
        self.assertIn("iou", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))

    def test_mobilepixor_iqa_postprocess_nms(self):
        pred_single = {
            "cls": torch.randn(1, 3, 200, 176),
            "offset": torch.randn(1, 2, 200, 176),
            "size": torch.randn(1, 2, 200, 176),
            "yaw": torch.randn(1, 2, 200, 176),
            "iou": torch.randn(1, 1, 200, 176),
        }
        config = {
            "geometry": {
                "x_min": 0, "x_max": 70.4, "x_res": 0.1,
                "y_min": -40, "y_max": 40, "y_res": 0.1,
            },
            "nms_alpha": 0.5,
        }
        boxes = filter_pred(pred_single, config, out_size_factor=4, thres=0.1, nms_thres=0.5)
        self.assertIsInstance(boxes, type(torch.empty(0).numpy()))
        self.assertEqual(boxes.ndim, 2)
        if len(boxes) > 0:
            self.assertEqual(boxes.shape[1], 7)


if __name__ == "__main__":
    unittest.main()
