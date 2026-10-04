import math
import torch
import pytest
from core.models.backbones.rc_sgfpn import FourierRangeEmbedding


def test_fourier_range_embedding_shapes_and_values():
    h, w = 100, 88
    x_bounds = (0.0, 70.4)
    y_bounds = (-40.0, 40.0)
    num_bands = 4

    fre = FourierRangeEmbedding(h, w, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_bands)

    assert hasattr(fre, "embedding")
    assert fre.embedding.shape == (1, 2 * num_bands, h, w)
    assert fre.out_dim == 2 * num_bands

    # Values must be bounded in [-1.0, 1.0]
    assert torch.all(fre.embedding >= -1.0 - 1e-5)
    assert torch.all(fre.embedding <= 1.0 + 1e-5)
    assert torch.isfinite(fre.embedding).all()

    # Exact origin check with odd dimensions: at (x=0, y=0), r=0 => sin(0)=0, cos(0)=1
    fre_odd = FourierRangeEmbedding(101, 89, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_bands)
    origin_y = 50
    origin_x = 0
    origin_feats = fre_odd.embedding[0, :, origin_y, origin_x]
    for b in range(num_bands):
        sin_val = origin_feats[2 * b].item()
        cos_val = origin_feats[2 * b + 1].item()
        assert abs(sin_val) < 1e-5
        assert abs(cos_val - 1.0) < 1e-5


from core.models.backbones.rc_sgfpn import RangeConditionedScaleGate


def test_range_conditioned_scale_gate_zero_init():
    channels = 48
    h, w = 100, 88
    gate_module = RangeConditionedScaleGate(channels, h, w)

    l_feat = torch.randn(2, channels, h, w)
    u_feat = torch.randn(2, channels, h, w)

    # At epoch 0 (init), gate must be identically 1.000000
    out = gate_module(l_feat, u_feat)
    expected = u_feat + l_feat

    assert torch.allclose(out, expected, atol=1e-6)


def test_range_conditioned_scale_gate_switch_to_deploy():
    channels = 48
    h, w = 100, 88
    gate_module = RangeConditionedScaleGate(channels, h, w)

    # Perturb weights slightly to simulate training
    with torch.no_grad():
        gate_module.content_conv.weight.add_(torch.randn_like(gate_module.content_conv.weight) * 0.1)
        gate_module.range_proj.weight.add_(torch.randn_like(gate_module.range_proj.weight) * 0.1)

    l_feat = torch.randn(2, channels, h, w)
    u_feat = torch.randn(2, channels, h, w)

    out_train = gate_module(l_feat, u_feat)
    gate_module.switch_to_deploy()

    assert gate_module.deploy is True
    assert not hasattr(gate_module, "range_proj")
    assert not hasattr(gate_module, "fre")

    out_deploy = gate_module(l_feat, u_feat)
    assert torch.allclose(out_train, out_deploy, atol=1e-5)



from core.models.backbones.rc_sgfpn import RangeConditionedSGFPN


def test_rc_sgfpn_unidirectional_shape_and_deploy():
    neck = RangeConditionedSGFPN(bidirectional=False)

    c3 = torch.randn(2, 48, 200, 176)
    c4 = torch.randn(2, 96, 100, 88)
    c5 = torch.randn(2, 128, 50, 44)

    out = neck(c3, c4, c5)
    assert out.shape == (2, 16, 200, 176)
    assert torch.isfinite(out).all()

    # Test switch_to_deploy
    neck.switch_to_deploy()
    out_deploy = neck(c3, c4, c5)
    assert out_deploy.shape == (2, 16, 200, 176)
    assert torch.allclose(out, out_deploy, atol=1e-5)


def test_rc_sgfpn_bidirectional_shape_and_deploy():
    neck = RangeConditionedSGFPN(bidirectional=True)

    c3 = torch.randn(2, 48, 200, 176)
    c4 = torch.randn(2, 96, 100, 88)
    c5 = torch.randn(2, 128, 50, 44)

    out = neck(c3, c4, c5)
    assert out.shape == (2, 16, 200, 176)
    assert torch.isfinite(out).all()

    # Test switch_to_deploy
    neck.switch_to_deploy()
    out_deploy = neck(c3, c4, c5)
    assert out_deploy.shape == (2, 16, 200, 176)
    assert torch.allclose(out, out_deploy, atol=1e-5)



def test_rc_sgfpn_gradient_flow_and_autocast_safety():
    neck = RangeConditionedSGFPN(bidirectional=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neck = neck.to(device)

    c3 = torch.randn(2, 48, 200, 176, device=device, requires_grad=True)
    c4 = torch.randn(2, 96, 100, 88, device=device, requires_grad=True)
    c5 = torch.randn(2, 128, 50, 44, device=device, requires_grad=True)

    # Autocast context check
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        out = neck(c3, c4, c5)
        loss = out.sum()

    loss.backward()

    # Assert gradients exist and are finite on all inputs
    assert c3.grad is not None and torch.isfinite(c3.grad).all()
    assert c4.grad is not None and torch.isfinite(c4.grad).all()
    assert c5.grad is not None and torch.isfinite(c5.grad).all()

    # Assert gradients exist on trainable gate weights across all pathways
    assert neck.gate_td4.content_conv.weight.grad is not None
    assert neck.gate_td4.range_proj.weight.grad is not None
    assert torch.isfinite(neck.gate_td4.content_conv.weight.grad).all()
    assert torch.isfinite(neck.gate_td4.range_proj.weight.grad).all()

    # Assert gradients exist on bottom-up reinforced pathways (no dead parameters)
    assert neck.gate_bu5.content_conv.weight.grad is not None
    assert neck.gate_bu5.range_proj.weight.grad is not None
    assert torch.isfinite(neck.gate_bu5.content_conv.weight.grad).all()
    assert neck.bu_refine_p4.weight.grad is not None
    assert torch.isfinite(neck.bu_refine_p4.weight.grad).all()
    assert neck.bu_refine_p3.weight.grad is not None
    assert torch.isfinite(neck.bu_refine_p3.weight.grad).all()



from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone


def test_mobilepixornext_with_rc_sgfpn():
    # Test Unidirectional RC-SGFPN
    bb_uni = MobilePixorNeXtBackbone(input_channels=8, neck_type="rc_sgfpn")
    x = torch.randn(2, 8, 800, 704)
    out_uni = bb_uni(x)
    assert out_uni.shape == (2, 16, 200, 176)

    # Test Bidirectional RC-BiSGFPN
    bb_bi = MobilePixorNeXtBackbone(input_channels=8, neck_type="rc_bisgfpn")
    out_bi = bb_bi(x)
    assert out_bi.shape == (2, 16, 200, 176)

    # Test deploy switch
    bb_bi.switch_to_deploy()
    out_bi_deploy = bb_bi(x)
    assert out_bi_deploy.shape == (2, 16, 200, 176)



from core.models.model import CustomModel


def test_custom_model_e2e_with_rc_sgfpn():
    cfg = {
        "bev_encoding": {"name": "rich8"},
        "kitti": {"geometry": {"x_min": 0, "x_max": 70.4, "y_min": -40, "y_max": 40, "x_res": 0.1, "y_res": 0.1}},
        "num_classes": 3,
        "backbone": "mobilepixornext",
        "neck_type": "rc_sgfpn",
    }
    model = CustomModel(cfg, num_classes=3, input_channels=8)
    voxel = torch.randn(2, 8, 800, 704)
    pred = model({"voxel": voxel})

    assert "cls" in pred
    assert "offset" in pred
    assert "size" in pred
    assert "yaw" in pred
    assert pred["cls"].shape == (2, 3, 200, 176)


def test_custom_model_e2e_with_rc_bisgfpn():
    cfg = {
        "bev_encoding": {"name": "rich8"},
        "kitti": {"geometry": {"x_min": 0, "x_max": 70.4, "y_min": -40, "y_max": 40, "x_res": 0.1, "y_res": 0.1}},
        "num_classes": 3,
        "backbone": "mobilepixornext",
        "neck_type": "rc_bisgfpn",
    }
    model = CustomModel(cfg, num_classes=3, input_channels=8)
    voxel = torch.randn(2, 8, 800, 704)
    pred = model({"voxel": voxel})

    assert "cls" in pred
    assert "offset" in pred
    assert "size" in pred
    assert "yaw" in pred
    assert pred["cls"].shape == (2, 3, 200, 176)


def test_mobilepixornext_rc_sgfpn_no_dead_parameters():
    bb = MobilePixorNeXtBackbone(input_channels=8, neck_type="rc_sgfpn")
    # Verify baseline FPN layers are not instantiated
    assert not hasattr(bb, "lat_c5")
    assert not hasattr(bb, "lat_c4")
    assert not hasattr(bb, "lat_c3")
    assert not hasattr(bb, "refine_u4")
    assert not hasattr(bb, "proj_u3")
    assert not hasattr(bb, "gate_c4")
    assert not hasattr(bb, "gate_c3")
    assert not hasattr(bb, "out_conv")

    # Verify parameter count is lean (around 675k, exactly 674,904)
    param_count = sum(p.numel() for p in bb.parameters())
    assert param_count == 674904


def test_rc_sgfpn_empty_background_zero_activation():
    """Verify FiLM modulation prevents hallucinating on empty background (zeros)."""
    channels = 48
    h, w = 100, 88
    gate = RangeConditionedScaleGate(channels, h, w)

    # Simulate trained non-zero range projector
    with torch.no_grad():
        gate.range_proj.weight.fill_(1.5)
        gate.range_proj.bias.fill_(0.5)

    l_empty = torch.zeros(1, channels, h, w)
    u_empty = torch.zeros(1, channels, h, w)

    out = gate(l_empty, u_empty)
    # When input features are zero, output must be strictly zero (no ghost activation)
    assert torch.all(out == 0.0)


def test_rc_bisgfpn_deploy_state_dict_roundtrip(tmp_path):
    """Verify that a deployed RC-BiSGFPN checkpoint loads cleanly with strict=True."""
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_dir = str(repo_root / "tools" / "kitti_training_pipeline")
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    from common import build_model
    from evaluate_kitti_bev import PyTorchRunner

    cfg = {
        "model": {
            "backbone": "mobilepixornext",
            "neck_type": "rc_bisgfpn",
            "backbone_out_dim": 16,
            "c4_attention": "litemla",
            "scale_gated_fpn": True,
            "use_reparam": False,
            "deploy": False,
        },
        "data": {
            "num_classes": 3,
            "out_size_factor": 4,
            "bev_encoding": {"name": "rich8"},
            "kitti": {
                "geometry": {
                    "x_min": 0, "x_max": 70.4, "y_min": -40, "y_max": 40,
                    "z_min": -2.5, "z_max": 1, "x_res": 0.1, "y_res": 0.1, "z_res": 0.1
                }
            },
        },
    }

    # 1. Instantiate training model and save checkpoint
    model_train = build_model(cfg)
    ckpt_path = tmp_path / "train_model.pt"
    deploy_path = tmp_path / "fused_deploy_model.pt"
    torch.save({"model_state_dict": model_train.state_dict()}, ckpt_path)

    # 2. Evaluate with deploy fusion and save deploy checkpoint
    runner = PyTorchRunner(ckpt_path, cfg, device="cpu", deploy=True, save_deploy=deploy_path)
    assert deploy_path.is_file()

    # 3. Reload deployed checkpoint into PyTorchRunner without error
    runner_fused = PyTorchRunner(deploy_path, cfg, device="cpu", deploy=True)
    assert runner_fused.is_deployed is True

    # 4. Check inference output match
    voxel = torch.randn(8, 800, 704)
    out1, _ = runner.infer(runner.transfer(voxel)[0])
    out2, _ = runner_fused.infer(runner_fused.transfer(voxel)[0])
    for k in ("cls", "offset", "size", "yaw"):
        assert torch.allclose(out1[k], out2[k], atol=1e-5)


def test_rc_bisgfpn_multi_step_training_gradient_unfreezing():
    """Verify that multi-step training activates strictly positive gradient norms across 100% of parameters.

    Exposes any 'fake-pass' checks where zero-valued gradients could mask dead branches.
    """
    neck = RangeConditionedSGFPN(bidirectional=True)
    opt = torch.optim.Adam(neck.parameters(), lr=1e-3)

    for step in range(3):
        c3 = torch.randn(2, 48, 200, 176)
        c4 = torch.randn(2, 96, 100, 88)
        c5 = torch.randn(2, 128, 50, 44)
        loss = neck(c3, c4, c5).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    # After warm-up steps, every single parameter in the bidirectional neck must have strictly positive grad norm
    c3 = torch.randn(2, 48, 200, 176)
    c4 = torch.randn(2, 96, 100, 88)
    c5 = torch.randn(2, 128, 50, 44)
    loss = neck(c3, c4, c5).sum()
    loss.backward()

    dead_parameters = []
    for name, p in neck.named_parameters():
        if p.grad is None or p.grad.norm().item() == 0.0:
            dead_parameters.append(name)

    assert len(dead_parameters) == 0, f"Found dead parameters with zero gradients: {dead_parameters}"


def test_rc_sgfpn_dtype_preservation_under_autocast():
    """Verify that RangeConditionedScaleGate preserves input dtype under FP16 and BF16 in deploy mode."""
    for dt in (torch.float16, torch.bfloat16):
        gate_deploy = RangeConditionedScaleGate(48, 100, 88, deploy=True).to(dt)
        l = torch.randn(1, 48, 100, 88, dtype=dt)
        u = torch.randn(1, 48, 100, 88, dtype=dt)
        out = gate_deploy(l, u)
        assert out.dtype == dt, f"Deploy gate modified dtype from {dt} to {out.dtype}"

        # Test switch_to_deploy flow with half precision
        gate_train = RangeConditionedScaleGate(48, 100, 88, deploy=False).to(dt)
        gate_train.switch_to_deploy()
        assert gate_train.static_spatial_scale.dtype == dt
        out_fused = gate_train(l, u)
        assert out_fused.dtype == dt

