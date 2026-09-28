import sys
from pathlib import Path
from typing import Tuple
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
]

from core.models.backbones.rep_blocks import (
    RepConv7x7,
    trans_conv_bn_to_kernel_bias,
    trans_identity_bn_to_kernel_bias,
)


def test_trans_conv_bn_fusion_math():
    channels = 8
    conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
    bn = nn.BatchNorm2d(channels)
    conv.eval()
    bn.eval()

    x = torch.randn(2, channels, 16, 16)
    expected = bn(conv(x))

    kernel, bias = trans_conv_bn_to_kernel_bias(conv, bn)
    fused_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
    fused_conv.weight.data.copy_(kernel)
    fused_conv.bias.data.copy_(bias)
    actual = fused_conv(x)

    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)


def test_trans_identity_bn_fusion_math():
    channels = 8
    bn = nn.BatchNorm2d(channels)
    bn.eval()

    x = torch.randn(2, channels, 16, 16)
    expected = bn(x)

    kernel, bias = trans_identity_bn_to_kernel_bias(bn, channels=channels, kernel_size=7)
    fused_conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=True)
    fused_conv.weight.data.copy_(kernel)
    fused_conv.bias.data.copy_(bias)
    actual = fused_conv(x)

    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)


def test_repconv7x7_numerical_equivalence():
    torch.manual_seed(42)
    channels = 48
    rep = RepConv7x7(channels=channels, deploy=False)
    rep.eval()

    x = torch.randn(2, channels, 50, 44)
    with torch.no_grad():
        out_multi = rep(x)

    assert not rep.deploy
    assert hasattr(rep, "rbr_conv7")
    assert hasattr(rep, "rbr_conv3")
    assert hasattr(rep, "rbr_identity")

    rep.switch_to_deploy()

    assert rep.deploy
    assert hasattr(rep, "rbr_reparam")
    assert not hasattr(rep, "rbr_conv7")
    assert not hasattr(rep, "rbr_conv3")
    assert not hasattr(rep, "rbr_identity")

    with torch.no_grad():
        out_fused = rep(x)

    diff = torch.max(torch.abs(out_multi - out_fused)).item()
    assert diff < 1e-5, f"Max difference {diff} exceeds tolerance 1e-5"


def test_repconv7x7_gradient_flow():
    channels = 16
    rep = RepConv7x7(channels=channels, deploy=False)
    rep.train()

    x = torch.randn(2, channels, 20, 20, requires_grad=True)
    out = rep(x)
    loss = out.sum()
    loss.backward()

    assert rep.rbr_conv7[0].weight.grad is not None
    assert rep.rbr_conv7[1].weight.grad is not None
    assert rep.rbr_conv3[0].weight.grad is not None
    assert rep.rbr_conv3[1].weight.grad is not None
    assert rep.rbr_identity.weight.grad is not None
    assert x.grad is not None


def test_mobilepixornext_block_backward_compat():
    from core.models.backbones.mobilepixornext_blocks import MobilePixorNeXtBlock

    torch.manual_seed(42)
    block_default = MobilePixorNeXtBlock(48)
    assert not getattr(block_default, "use_reparam", False)
    assert hasattr(block_default, "dwconv")
    assert hasattr(block_default, "norm1")

    x = torch.randn(2, 48, 20, 20)
    out = block_default(x)
    assert out.shape == (2, 48, 20, 20)


def test_mobilepixornext_block_reparam_equivalence():
    from core.models.backbones.mobilepixornext_blocks import MobilePixorNeXtBlock

    torch.manual_seed(42)
    block = MobilePixorNeXtBlock(48, use_reparam=True)
    block.eval()

    x = torch.randn(2, 48, 20, 20)
    with torch.no_grad():
        out_multi = block(x)

    assert block.use_reparam
    assert hasattr(block, "dw_block")
    assert not block.dw_block.deploy

    block.switch_to_deploy()
    assert block.dw_block.deploy

    with torch.no_grad():
        out_fused = block(x)

    diff = torch.max(torch.abs(out_multi - out_fused)).item()
    assert diff < 1e-5, f"Block diff {diff} exceeds tolerance 1e-5"


def test_mobilepixornext_backbone_reparam_equivalence():
    from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone

    torch.manual_seed(42)
    backbone = MobilePixorNeXtBackbone(input_channels=8, backbone_out_dim=16, use_reparam=True)
    backbone.eval()

    x = torch.randn(1, 8, 400, 352)
    with torch.no_grad():
        out_multi = backbone(x)

    backbone.switch_to_deploy()

    with torch.no_grad():
        out_fused = backbone(x)

    diff = torch.max(torch.abs(out_multi - out_fused)).item()
    assert diff < 1e-5, f"Backbone diff {diff} exceeds tolerance 1e-5"


def test_custom_model_reparam_flow():
    from core.models.model import CustomModel

    cfg = {
        "backbone": "mobilepixornext",
        "backbone_out_dim": 16,
        "cls_encoding": "gaussian",
        "use_reparam": True,
    }
    model = CustomModel(cfg, num_classes=3, input_channels=8)
    model.eval()

    x = torch.randn(1, 8, 400, 352)
    with torch.no_grad():
        pred_multi = model(x)

    model.switch_to_deploy()

    with torch.no_grad():
        pred_fused = model(x)

    for head_name in ("cls", "offset", "size", "yaw"):
        diff = torch.max(torch.abs(pred_multi[head_name] - pred_fused[head_name])).item()
        assert diff < 1e-5, f"Head {head_name} diff {diff} exceeds tolerance 1e-5"


def test_config_reparam_loading_and_parameter_reduction():
    import json
    from core.models.model import CustomModel

    config_path = ROOT / "configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga_reparam.json"
    assert config_path.exists(), "Reparam config must exist"

    with open(config_path, "r") as f:
        full_cfg = json.load(f)

    cfg_reparam = full_cfg.get("model", full_cfg)
    assert cfg_reparam.get("use_reparam") is True

    # Baseline M0 model
    cfg_base = dict(cfg_reparam)
    cfg_base["use_reparam"] = False
    model_m0 = CustomModel(cfg_base, num_classes=3, input_channels=8)
    params_m0 = sum(p.numel() for p in model_m0.parameters())

    # M1 model in training mode
    model_m1 = CustomModel(cfg_reparam, num_classes=3, input_channels=8)
    params_m1_train = sum(p.numel() for p in model_m1.parameters())

    # M1 training parameters must be greater than M0 due to 3x3 and identity branches
    assert params_m1_train > params_m0, f"Expected {params_m1_train} > {params_m0}"

    # After switch_to_deploy(), M1 eliminates the BatchNorm layers (gamma & beta fused to 1 bias)
    # saving exactly 1 parameter per channel across all 8 depthwise blocks
    model_m1.switch_to_deploy()
    params_m1_deploy = sum(p.numel() for p in model_m1.parameters())

    expected_saved = (2 * 48) + (4 * 96) + (2 * 128)  # 736 parameters
    assert params_m0 - params_m1_deploy == expected_saved, (
        f"Expected {expected_saved} parameters saved from fused BN, got {params_m0 - params_m1_deploy}"
    )
    assert params_m1_deploy < params_m0


def test_repconv7x7_dtype_preservation():
    channels = 32
    rep_bf16 = RepConv7x7(channels=channels, deploy=False).to(torch.bfloat16)
    rep_bf16.eval()

    x = torch.randn(2, channels, 16, 16, dtype=torch.bfloat16)
    rep_bf16.switch_to_deploy()

    assert rep_bf16.rbr_reparam.weight.dtype == torch.bfloat16
    assert rep_bf16.rbr_reparam.bias.dtype == torch.bfloat16

    out = rep_bf16(x)
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, channels, 16, 16)


def test_direct_deploy_instantiation():
    from core.models.model import CustomModel

    cfg = {
        "backbone": "mobilepixornext",
        "backbone_out_dim": 16,
        "cls_encoding": "gaussian",
        "use_reparam": True,
        "deploy": True,
    }
    model_direct = CustomModel(cfg, num_classes=3, input_channels=8)
    model_direct.eval()

    # Verify that submodules are directly in deploy mode
    for m in model_direct.modules():
        if isinstance(m, RepConv7x7):
            assert m.deploy
            assert hasattr(m, "rbr_reparam")
            assert not hasattr(m, "rbr_conv7")

    x = torch.randn(1, 8, 400, 352)
    pred = model_direct(x)
    assert "cls" in pred
    assert pred["cls"].shape[1] == 3


def test_repconv7x7_train_to_eval_auto_switch():
    channels = 16
    rep = RepConv7x7(channels=channels, deploy=False)
    rep.train()
    assert rep.training

    # Calling switch_to_deploy on a training model automatically forces eval mode
    rep.switch_to_deploy()
    assert not rep.training
    assert rep.deploy
    assert hasattr(rep, "rbr_reparam")


def test_repconv7x7_float16_preservation():
    channels = 16
    rep_fp16 = RepConv7x7(channels=channels, deploy=False).to(torch.float16)
    rep_fp16.eval()

    x = torch.randn(2, channels, 16, 16, dtype=torch.float16)
    rep_fp16.switch_to_deploy()

    assert rep_fp16.rbr_reparam.weight.dtype == torch.float16
    assert rep_fp16.rbr_reparam.bias.dtype == torch.float16

    out = rep_fp16(x)
    assert out.dtype == torch.float16


def test_state_dict_save_and_direct_load():
    from core.models.model import CustomModel

    cfg_train = {
        "backbone": "mobilepixornext",
        "backbone_out_dim": 16,
        "cls_encoding": "gaussian",
        "use_reparam": True,
    }
    model_train = CustomModel(cfg_train, num_classes=3, input_channels=8).eval()
    deploy_state_dict = model_train.export_deploy_state_dict()

    cfg_deploy = {
        "backbone": "mobilepixornext",
        "backbone_out_dim": 16,
        "cls_encoding": "gaussian",
        "use_reparam": True,
        "deploy": True,
    }
    model_deploy = CustomModel(cfg_deploy, num_classes=3, input_channels=8).eval()
    model_deploy.load_state_dict(deploy_state_dict)

    x = torch.randn(1, 8, 400, 352)
    with torch.no_grad():
        out1 = model_train(x)["cls"]
        out2 = model_deploy(x)["cls"]

    diff = torch.max(torch.abs(out1 - out2)).item()
    assert diff == 0.0, f"Expected exact 0 diff between export and loaded model, got {diff}"


def test_rep_model_torch_jit_trace():
    from core.models.model import CustomModel

    cfg = {
        "backbone": "mobilepixornext",
        "backbone_out_dim": 16,
        "cls_encoding": "gaussian",
        "use_reparam": True,
    }
    model = CustomModel(cfg, num_classes=3, input_channels=8).eval()
    model.switch_to_deploy()

    x = torch.randn(1, 8, 400, 352)

    # 1. Trace backbone directly (single Tensor output)
    traced_backbone = torch.jit.trace(model.backbone, x)
    out_bb = traced_backbone(x)
    assert out_bb.shape == (1, 16, 100, 88)

    # 2. Trace full model (dict output with strict=False)
    traced_model = torch.jit.trace(model, x, strict=False)
    out_m = traced_model(x)
    assert "cls" in out_m
    assert torch.allclose(model(x)["cls"], out_m["cls"])


def test_pytorch_runner_deploy_and_save_deploy(tmp_path):
    sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))
    from core.models.model import CustomModel
    from evaluate_kitti_bev import PyTorchRunner

    cfg = {
        "model": {
            "backbone": "mobilepixornext",
            "backbone_out_dim": 16,
            "cls_encoding": "gaussian",
            "use_reparam": True,
        },
        "data": {
            "num_classes": 3,
            "kitti": {
                "geometry": {
                    "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
                    "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
                    "z_min": -3.0, "z_max": 1.0, "z_res": 0.1,
                }
            },
            "bev_encoding": {"name": "rich8"},
        },
    }

    # 1. Create a training checkpoint
    train_model = CustomModel(cfg["model"], num_classes=3, input_channels=8)
    ckpt_path = tmp_path / "train_best.pt"
    torch.save(train_model.state_dict(), ckpt_path)

    # 2. PyTorchRunner with deploy=True and save_deploy
    deploy_ckpt_path = tmp_path / "saved_deploy.pt"
    runner = PyTorchRunner(
        path=ckpt_path,
        config=cfg,
        device="cpu",
        deploy=True,
        save_deploy=deploy_ckpt_path,
    )

    assert runner.is_deployed is True
    assert deploy_ckpt_path.is_file()
    assert runner.metadata()["deploy"] is True
    deploy_param_count = runner.metadata()["parameters"]
    assert deploy_param_count == 692361

    # 3. Load the saved deploy checkpoint directly in another runner
    runner_from_deploy = PyTorchRunner(
        path=deploy_ckpt_path,
        config=cfg,
        device="cpu",
    )
    assert runner_from_deploy.is_deployed is True
    assert runner_from_deploy.metadata()["parameters"] == 692361







