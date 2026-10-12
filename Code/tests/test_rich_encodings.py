import json
from pathlib import Path
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
    str(ROOT / "tools" / "kitti_training_pipeline"),
]

from common import build_model, generate_run_name, input_shape, create_experiment_config


def test_run_name_generation_rich_encodings():
    with open(ROOT / "configs/config.json") as f:
        base = json.load(f)

    for name in ["rich8"]:
        cfg = create_experiment_config(base, {"data": {"bev_encoding": {"name": name}}})
        run_name = generate_run_name(cfg, seed=42)
        assert name in run_name


def test_model_build_and_forward_rich_encodings():
    with open(ROOT / "configs/config.json") as f:
        base = json.load(f)

    for name, expected_ch in [
        ("rich8", 8),
    ]:
        cfg = create_experiment_config(base, {"data": {"bev_encoding": {"name": name}}})
        shape = input_shape(cfg)
        assert shape[1] == expected_ch

        model = build_model(cfg)
        assert model.backbone.stem[0].in_channels == expected_ch
        dummy_x = torch.zeros((2, expected_ch, shape[2], shape[3]))
        out = model(dummy_x)
        assert "cls" in out
