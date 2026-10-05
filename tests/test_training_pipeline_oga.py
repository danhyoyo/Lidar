import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))
sys.path.insert(0, str(REPO_ROOT / "detector" / "core" / "datasets"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "kitti_training_pipeline"))

import torch
from common import build_model, create_experiment_config
from core.losses.loss_fn import LossFunction


class TestTrainingPipelineOGA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPO_ROOT / "configs/config.json", "r", encoding="utf-8") as f:
            cls.base_config = json.load(f)

    def test_pipeline_forward_backward_with_oga(self):
        config = create_experiment_config(self.base_config, {
            "loss": {"name": "oga"}
        })

        device = torch.device("cpu")
        model = build_model(config).to(device)
        criterion = LossFunction(config["model"]["cls_encoding"], config["loss"]).to(device)

        # Mock a voxel batch [B, 8, 64, 64]
        B = 2
        voxel = torch.randn(B, 8, 64, 64, device=device)
        outputs = model(voxel)

        H_out, W_out = outputs["cls"].shape[2], outputs["cls"].shape[3]
        batch = {
            "cls": torch.zeros(B, 3, H_out, W_out, device=device),
            "offset": torch.zeros(B, 2, H_out, W_out, device=device),
            "size": torch.zeros(B, 2, H_out, W_out, device=device),
            "yaw": torch.tensor([1.0, 0.0], device=device).view(1, 2, 1, 1).expand(B, 2, H_out, W_out),
            "reg_mask": torch.ones(B, H_out, W_out, device=device),
        }

        # Forward pass
        loss_dict = criterion(outputs, batch)
        loss = loss_dict["loss"]
        self.assertTrue(torch.isfinite(loss))

        # Backward pass
        loss.backward()

        # Check gradients
        has_model_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        has_crit_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in criterion.parameters())
        self.assertTrue(has_model_grad)
        self.assertTrue(has_crit_grad)

    def test_pipeline_forward_backward_with_oga_iqa_bf16(self):
        config = create_experiment_config(self.base_config, {
            "model": {
                "use_reparam": True,
                "header_use_iou": True,
            },
            "loss": {
                "name": "oga",
                "use_iou": True,
            }
        })

        device = torch.device("cpu")
        model = build_model(config).to(device)
        criterion = LossFunction(config["model"]["cls_encoding"], config["loss"]).to(device)

        B = 2
        voxel = torch.randn(B, 8, 64, 64, device=device)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            outputs = model(voxel)
            H_out, W_out = outputs["cls"].shape[2], outputs["cls"].shape[3]
            batch = {
                "cls": torch.zeros(B, 3, H_out, W_out, device=device),
                "offset": torch.zeros(B, 2, H_out, W_out, device=device),
                "size": torch.zeros(B, 2, H_out, W_out, device=device),
                "yaw": torch.tensor([1.0, 0.0], device=device).view(1, 2, 1, 1).expand(B, 2, H_out, W_out),
                "reg_mask": torch.ones(B, H_out, W_out, device=device),
            }
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["loss"]

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("iou", loss_dict)
        loss.backward()
        has_model_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        has_crit_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in criterion.parameters())
        self.assertTrue(has_model_grad)
        self.assertTrue(has_crit_grad)


if __name__ == "__main__":
    unittest.main()
