import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
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


from core.losses.strategies.oga import OgaLossStrategy


def test_oga_loss_strategy_with_iou():
    """Verify OgaLossStrategy computes iou loss and uncertainty weight when use_iou=True."""
    config = {
        "use_iou": True,
        "iou_target_type": "mgiou",
        "iou_loss_weight": 1.0,
    }
    strategy = OgaLossStrategy(cls_encoding="gaussian", config=config)
    assert "iou" in strategy.TASKS

    B, C, H, W = 2, 3, 20, 20
    pred = {
        "cls": torch.randn(B, C, H, W, requires_grad=True),
        "offset": torch.randn(B, 2, H, W, requires_grad=True),
        "size": torch.randn(B, 2, H, W, requires_grad=True),
        "yaw": torch.randn(B, 2, H, W, requires_grad=True),
        "iou": torch.randn(B, 1, H, W, requires_grad=True),
    }
    target = {
        "cls": torch.rand(B, C, H, W),
        "offset": torch.randn(B, 2, H, W),
        "size": torch.randn(B, 2, H, W),
        "yaw": torch.randn(B, 2, H, W),
        "reg_mask": torch.randint(0, 2, (B, H, W), dtype=torch.bool),
    }

    out = strategy(pred, target)
    assert "loss" in out
    assert "iou" in out
    assert "weight_iou" in out
    assert out["loss"].requires_grad

    # Test gradient flow to pred["iou"]
    out["loss"].backward()
    assert pred["iou"].grad is not None
    assert torch.isfinite(pred["iou"].grad).all()


def test_oga_loss_gradient_isolation_on_regression():
    """Verify that pred['offset'], pred['size'], pred['yaw'] do NOT receive gradient from iou_target."""
    config = {"use_iou": True, "iou_target_type": "mgiou"}
    strategy = OgaLossStrategy(cls_encoding="gaussian", config=config)

    B, C, H, W = 1, 1, 10, 10
    pred = {
        "cls": torch.zeros(B, C, H, W),
        "offset": torch.zeros(B, 2, H, W, requires_grad=True),
        "size": torch.zeros(B, 2, H, W, requires_grad=True),
        "yaw": torch.zeros(B, 2, H, W, requires_grad=True),
        "iou": torch.zeros(B, 1, H, W, requires_grad=True),
    }
    target = {
        "cls": torch.zeros(B, C, H, W),
        "offset": torch.zeros(B, 2, H, W),
        "size": torch.zeros(B, 2, H, W),
        "yaw": torch.zeros(B, 2, H, W),
        "reg_mask": torch.ones(B, H, W, dtype=torch.bool),
    }

    # Compute ONLY iou loss isolated
    target_iou = compute_iou_targets(pred, target, method="mgiou")
    loss_iou = torch.nn.functional.binary_cross_entropy_with_logits(
        pred["iou"].squeeze(1), target_iou
    )
    loss_iou.backward()

    # Regression heads must receive ZERO gradient
    assert pred["offset"].grad is None or (pred["offset"].grad == 0).all()
    assert pred["size"].grad is None or (pred["size"].grad == 0).all()
    assert pred["yaw"].grad is None or (pred["yaw"].grad == 0).all()
    # iou logit head MUST receive valid gradient
    assert pred["iou"].grad is not None and not (pred["iou"].grad == 0).all()
