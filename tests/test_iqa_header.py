import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
]

from core.models.heads.cnn import Header
from core.models.model import CustomModel


def test_header_backward_compatibility():
    """Verify Header without use_iou preserves exactly 4 heads."""
    header = Header(num_classes=3, in_channels=16, use_bn=True, act="silu", use_iou=False)
    x = torch.randn(2, 16, 50, 44)
    pred = header(x)
    assert set(pred.keys()) == {"cls", "offset", "size", "yaw"}
    assert "iou" not in pred


def test_header_with_iou_branch():
    """Verify Header with use_iou outputs 1-channel 'iou' logit."""
    header = Header(num_classes=3, in_channels=16, use_bn=True, act="silu", use_iou=True)
    x = torch.randn(2, 16, 50, 44)
    pred = header(x)
    assert set(pred.keys()) == {"cls", "offset", "size", "yaw", "iou"}
    assert pred["iou"].shape == (2, 1, 50, 44)


def test_custom_model_threads_iou_flag():
    """Verify CustomModel initializes Header with use_iou according to config."""
    cfg = {
        "backbone": "mobilepixornext",
        "backbone_out_dim": 16,
        "header_use_bn": True,
        "header_act": "silu",
        "header_use_iou": True,
    }
    model = CustomModel(cfg, num_classes=3, input_channels=35)
    x = torch.randn(2, 35, 800, 704)
    pred = model(x)
    assert "iou" in pred
    assert pred["iou"].shape == (2, 1, 200, 176)
