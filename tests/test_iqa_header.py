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


from postprocess import filter_pred


def test_filter_pred_score_reordering_with_iou():
    """Verify that a candidate with high IoU and moderate cls score beats a candidate with high cls and low IoU when nms_alpha=0.5."""
    config = {
        "geometry": {
            "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
            "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        },
        "nms_alpha": 0.5,
    }
    out_size_factor = 4
    # Grid shape: H=200, W=176
    H, W = 200, 176
    cls_pred = torch.full((1, 1, H, W), -10.0)  # all low logits
    offset_pred = torch.zeros(1, 2, H, W)
    size_pred = torch.zeros(1, 2, H, W)
    yaw_pred = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(1, 2, H, W)
    iou_pred = torch.zeros(1, 1, H, W)

    # Candidate A at (50, 50): high cls logit (+4.0 -> p~0.98), low iou logit (-3.0 -> s~0.047)
    # S_final = sqrt(0.98 * 0.047) ~ 0.21
    cls_pred[0, 0, 50, 50] = 4.0
    iou_pred[0, 0, 50, 50] = -3.0

    # Candidate B at (60, 60): moderate cls logit (+1.5 -> p~0.81), high iou logit (+3.0 -> s~0.95)
    # S_final = sqrt(0.81 * 0.95) ~ 0.87
    cls_pred[0, 0, 60, 60] = 1.5
    iou_pred[0, 0, 60, 60] = 3.0

    pred_with_iou = {
        "cls": cls_pred, "offset": offset_pred, "size": size_pred, "yaw": yaw_pred, "iou": iou_pred
    }

    # At threshold 0.3: Candidate A (0.21) should be filtered out, Candidate B (0.87) should survive
    dets = filter_pred(pred_with_iou, config, out_size_factor=out_size_factor, thres=0.3, nms_thres=0.5)
    assert len(dets) == 1
    # Candidate B is at grid (60, 60): metric coordinates center_x = 60 * 0.4 + 0 = 24.0, center_y = 60 * 0.4 - 40 = -16.0
    # Output columns: [cls_id, score, center_x, center_y, l, w, yaw]
    assert abs(dets[0, 2] - 24.0) < 1.0
    assert abs(dets[0, 3] - (-16.0)) < 1.0


def test_filter_pred_alpha_zero_reverts_to_old_nms():
    """Verify that when nms_alpha=0.0, NMS uses pure cls_probs (Candidate A survives, Candidate B is ranked lower or both survive based on cls)."""
    config = {
        "geometry": {
            "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
            "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        },
        "nms_alpha": 0.0,  # 0.0 means standard/old NMS ignoring IoU head
    }
    H, W = 200, 176
    cls_pred = torch.full((1, 1, H, W), -10.0)
    offset_pred = torch.zeros(1, 2, H, W)
    size_pred = torch.zeros(1, 2, H, W)
    yaw_pred = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(1, 2, H, W)
    iou_pred = torch.zeros(1, 1, H, W)

    # Candidate A has high cls logit (+4.0 -> p~0.98), Candidate B has low cls logit (-1.0 -> p~0.27)
    cls_pred[0, 0, 50, 50] = 4.0
    iou_pred[0, 0, 50, 50] = -3.0

    cls_pred[0, 0, 60, 60] = -1.0
    iou_pred[0, 0, 60, 60] = 3.0

    pred_with_iou = {
        "cls": cls_pred, "offset": offset_pred, "size": size_pred, "yaw": yaw_pred, "iou": iou_pred
    }

    # At threshold 0.5: With alpha=0.0, Candidate A survives (p=0.98 > 0.5) even though its IoU logit is low
    dets = filter_pred(pred_with_iou, config, out_size_factor=4, thres=0.5, nms_thres=0.5)
    assert len(dets) == 1
    assert abs(dets[0, 2] - 20.0) < 1.0  # 50 * 0.4 = 20.0 (Candidate A)


def test_filter_pred_fallback_when_iou_absent():
    """Verify that filter_pred works identically to baseline when 'iou' head is absent."""
    config = {
        "geometry": {
            "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
            "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        },
        "nms_alpha": 0.5,
    }
    H, W = 200, 176
    cls_pred = torch.full((1, 1, H, W), -10.0)
    cls_pred[0, 0, 50, 50] = 3.0  # p~0.95
    pred_no_iou = {
        "cls": cls_pred,
        "offset": torch.zeros(1, 2, H, W),
        "size": torch.zeros(1, 2, H, W),
        "yaw": torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(1, 2, H, W),
    }
    dets = filter_pred(pred_no_iou, config, out_size_factor=4, thres=0.3, nms_thres=0.5)
    assert len(dets) == 1


import json


def test_full_iqa_pipeline_integration():
    """Verify that CustomModel and OgaLossStrategy initialize, forward, and backward cleanly using IQA config."""
    config_path = ROOT / "configs" / "kitti" / "mobilepixornext_oga" / "kitti_mobilepixornext_litemla_oga_reparam_iqa.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = CustomModel(cfg["model"], num_classes=cfg["data"]["num_classes"], input_channels=35)
    loss_strategy = OgaLossStrategy(cls_encoding=cfg["model"].get("cls_encoding", "gaussian"), config=cfg["loss"])

    B = 2
    x = torch.randn(B, 35, 800, 704)
    pred = model(x)
    assert "iou" in pred
    assert pred["iou"].shape == (B, 1, 200, 176)

    target = {
        "cls": torch.rand(B, 3, 200, 176),
        "offset": torch.randn(B, 2, 200, 176),
        "size": torch.randn(B, 2, 200, 176),
        "yaw": torch.randn(B, 2, 200, 176),
        "reg_mask": torch.randint(0, 2, (B, 200, 176), dtype=torch.bool),
    }

    out = loss_strategy(pred, target)
    assert "loss" in out
    assert "iou" in out
    assert "weight_iou" in out
    out["loss"].backward()
