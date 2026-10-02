import pytest
import torch
from detector.core.models.model import CustomModel

def test_custom_model_with_rich_mamba():
    cfg = {
        "backbone": "mobilepixornext",
        "backbone_out_dim": 16,
        "cls_encoding": "gaussian",
        "bev_encoding": {
            "name": "rich_mamba",
            "d_model": 16,
            "d_state": 16,
            "max_points_per_pillar": 20,
            "max_pillars": 500,
            "out_channels": 8,
        },
        "geometry": {
            "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
            "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
            "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
        }
    }
    model = CustomModel(cfg, num_classes=4, input_channels=8)
    assert hasattr(model, "encoder")
    assert model.encoder is not None

    # Synthetic point cloud (N, 4)
    pts = torch.rand(150, 4)
    pts[:, 0] = pts[:, 0] * 60.0 + 5.0
    pts[:, 1] = pts[:, 1] * 70.0 - 35.0
    pts[:, 2] = pts[:, 2] * 3.0 - 2.0
    pts[:, 3] = torch.rand(150)

    outputs = model(pts)
    assert "cls" in outputs
    assert "offset" in outputs
    assert "size" in outputs
    assert "yaw" in outputs
    assert outputs["cls"].shape == (1, 4, 200, 176)
