import sys
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (str(REPO_ROOT), str(REPO_ROOT / "detector")):
    if path not in sys.path:
        sys.path.insert(0, path)

from detector.core.models.encoders.rich_mamba import RichMambaEncoder

def test_rich_mamba_encoder_forward_and_backward():
    geometry = {
        "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
        "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
    }
    cfg = {
        "d_model": 16,
        "d_state": 16,
        "max_points_per_pillar": 20,
        "max_pillars": 1000,
        "out_channels": 8,
        "use_dual_pooling": True,
    }
    encoder = RichMambaEncoder(cfg, geometry)

    # 200 random points in ROI
    N = 200
    pts = torch.rand(N, 4)
    pts[:, 0] = pts[:, 0] * 60.0 + 5.0
    pts[:, 1] = pts[:, 1] * 70.0 - 35.0
    pts[:, 2] = pts[:, 2] * 3.0 - 2.0
    pts[:, 3] = torch.rand(N)

    bev = encoder(pts)
    assert bev.shape == (1, 8, 800, 704)
    assert torch.isfinite(bev).all()

    loss = bev.sum()
    loss.backward()

    # Verify gradients flow into encoder weights
    assert encoder.in_proj[0].weight.grad is not None
    assert torch.isfinite(encoder.in_proj[0].weight.grad).all()

def test_rich_mamba_batch_list_inputs():
    geometry = {
        "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
        "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
    }
    encoder = RichMambaEncoder({"d_model": 16, "out_channels": 8}, geometry)
    pts1 = torch.rand(100, 4)
    pts1[:, 0] = pts1[:, 0] * 50.0 + 5.0
    pts1[:, 1] = pts1[:, 1] * 60.0 - 30.0
    pts1[:, 2] = pts1[:, 2] * 2.0 - 1.5

    pts2 = torch.rand(80, 4)
    pts2[:, 0] = pts2[:, 0] * 50.0 + 5.0
    pts2[:, 1] = pts2[:, 1] * 60.0 - 30.0
    pts2[:, 2] = pts2[:, 2] * 2.0 - 1.5

    bev_batch = encoder([pts1, pts2])
    assert bev_batch.shape == (2, 8, 800, 704)
    assert torch.isfinite(bev_batch).all()

def test_rich_mamba_empty_scene():
    geometry = {
        "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
        "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
    }
    encoder = RichMambaEncoder({}, geometry)
    empty_pts = torch.zeros((0, 4))
    bev = encoder(empty_pts)
    assert bev.shape == (1, 8, 800, 704)
    assert (bev == 0).all()

def test_rich_mamba_single_pillar_training_mode():
    geometry = {
        "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
        "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
    }
    encoder = RichMambaEncoder({"d_model": 16, "out_channels": 8}, geometry)
    encoder.train()  # Explicitly set to training mode

    # Exactly 1 point -> exactly 1 pillar
    single_pt = torch.tensor([[10.05, 5.05, -0.5, 0.5]], dtype=torch.float32)
    bev = encoder(single_pt)
    assert bev.shape == (1, 8, 800, 704)
    assert torch.isfinite(bev).all()
    # Gradient flow check
    loss = bev.sum()
    loss.backward()
    assert encoder.in_proj[0].weight.grad is not None
