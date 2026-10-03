import json
import sys
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (str(REPO_ROOT), str(REPO_ROOT / "detector")):
    if path not in sys.path:
        sys.path.insert(0, path)

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


@pytest.mark.parametrize("channels", [16, 32])
def test_richmamba_dynamic_out_channels(channels):
    from tools.kitti_training_pipeline.common import input_shape, generate_run_name
    config_path = Path("configs/kitti/rich_mamba/kitti_mobilepixornext_richmamba.json")
    with open(config_path) as f:
        cfg = json.load(f)

    cfg["data"]["bev_encoding"]["out_channels"] = channels
    cfg["model"]["bev_encoding"]["out_channels"] = channels

    # Check input_shape
    shape = input_shape(cfg)
    assert shape == (1, channels, 800, 704)

    # Check run_name reflects channel count
    run_name = generate_run_name(cfg)
    assert f"rich_mamba_{channels}ch" in run_name

    # Check model instantiation
    model = CustomModel(cfg["model"])
    assert model.encoder.out_channels == channels
    assert model.backbone.input_channels == channels
    assert model.backbone.stem[0].in_channels == channels

    # Forward dummy points through custom model
    pts = torch.rand(40, 4)
    pts[:, 0] = pts[:, 0] * 50.0 + 5.0
    pts[:, 1] = pts[:, 1] * 60.0 - 30.0
    pts[:, 2] = pts[:, 2] * 2.5 - 2.0
    pts[:, 3] = torch.rand(40)

    out = model(pts)
    assert out["cls"].shape == (1, 4, 200, 176)
    assert torch.isfinite(out["cls"]).all()

