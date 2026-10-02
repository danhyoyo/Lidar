import json
import pytest
import torch
from pathlib import Path
from detector.core.models.model import CustomModel

def test_richmamba_config_instantiation():
    config_path = Path("configs/kitti/rich_mamba/kitti_mobilepixornext_richmamba.json")
    assert config_path.exists()

    with open(config_path) as f:
        cfg = json.load(f)

    model = CustomModel(cfg["model"])
    assert hasattr(model, "encoder")
    assert model.encoder is not None
    assert type(model.backbone).__name__ == "MobilePixorNeXtBackbone"

    # Forward dummy points (50, 4)
    pts = torch.rand(50, 4)
    pts[:, 0] = pts[:, 0] * 50.0 + 5.0
    pts[:, 1] = pts[:, 1] * 60.0 - 30.0
    pts[:, 2] = pts[:, 2] * 2.5 - 2.0
    pts[:, 3] = torch.rand(50)

    out = model(pts)
    assert out["cls"].shape == (1, 4, 200, 176)
    assert out["offset"].shape == (1, 2, 200, 176)
    assert out["size"].shape == (1, 2, 200, 176)
    assert out["yaw"].shape == (1, 2, 200, 176)
