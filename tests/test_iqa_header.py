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


from core.losses.iou_targets import compute_iou_targets


def test_compute_iou_targets_identical_boxes():
    """Verify target IoU is 1.0 when prediction matches target exactly."""
    B, H, W = 2, 10, 10
    offset = torch.zeros(B, 2, H, W)
    size = torch.zeros(B, 2, H, W)  # exp(0) = 1.0m width, 1.0m length
    yaw = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W)
    reg_mask = torch.ones(B, H, W, dtype=torch.bool)

    pred = {"offset": offset.clone(), "size": size.clone(), "yaw": yaw.clone()}
    target = {"offset": offset.clone(), "size": size.clone(), "yaw": yaw.clone(), "reg_mask": reg_mask}

    for method in ["mgiou", "yaw_footprint"]:
        target_iou = compute_iou_targets(pred, target, method=method)
        assert target_iou.shape == (B, H, W)
        assert torch.allclose(target_iou, torch.ones_like(target_iou), atol=1e-4)


def test_compute_iou_targets_orthogonal_yaw():
    """Verify target IoU is heavily penalized (near 0) when yaw is rotated 90 degrees."""
    B, H, W = 1, 4, 4
    offset = torch.zeros(B, 2, H, W)
    # Use rectangular box (w=1.0, l=4.0, exp(0)=1, exp(1.386)=4.0) so 90-deg rotation creates distinct shape
    size = torch.tensor([0.0, 1.386294]).view(1, 2, 1, 1).expand(B, 2, H, W)
    pred_yaw = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W)  # cos(2*0) = 1
    tgt_yaw = torch.tensor([-1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W)  # cos(2*pi/2) = -1 (90 deg)
    reg_mask = torch.ones(B, H, W, dtype=torch.bool)

    pred = {"offset": offset, "size": size, "yaw": pred_yaw}
    target = {"offset": offset, "size": size, "yaw": tgt_yaw, "reg_mask": reg_mask}

    for method in ["mgiou", "yaw_footprint"]:
        target_iou = compute_iou_targets(pred, target, method=method)
        assert (target_iou < 0.35).all(), f"Method {method} did not penalize 90 deg yaw: max={target_iou.max()}"


def test_compute_iou_targets_distant_boxes():
    """Verify target IoU is 0.0 for completely disjoint boxes."""
    B, H, W = 1, 2, 2
    pred_offset = torch.zeros(B, 2, H, W)
    tgt_offset = torch.full((B, 2, H, W), 100.0)  # 100m away
    size = torch.zeros(B, 2, H, W)
    yaw = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(B, 2, H, W)
    reg_mask = torch.ones(B, H, W, dtype=torch.bool)

    pred = {"offset": pred_offset, "size": size, "yaw": yaw}
    target = {"offset": tgt_offset, "size": size, "yaw": yaw, "reg_mask": reg_mask}

    for method in ["mgiou", "yaw_footprint"]:
        target_iou = compute_iou_targets(pred, target, method=method)
        assert torch.all(target_iou == 0.0)
