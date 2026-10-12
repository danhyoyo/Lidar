#!/usr/bin/env python3
"""Source-verified OGA/Q-OGA and Focal Context/SimAM paper diagrams."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

from draw_mobilepixornext_paper import ROOT, VENDOR, Scene, render, write_tikz


SOURCES = [
    "detector/core/losses/strategies/oga.py",
    "detector/core/losses/strategies/q_oga.py",
    "detector/core/losses/strategies/base.py",
    "detector/core/losses/grouped.py",
    "detector/core/losses/oriented_geometry_loss.py",
    "detector/core/losses/smooth_corner_loss.py",
    "detector/core/losses/focal_loss.py",
    "detector/core/losses/quality_focal_loss.py",
    "detector/core/losses/iou_targets.py",
    "detector/core/losses/uncertainty_weighting.py",
    "detector/core/losses/l1_loss.py",
    "detector/core/models/backbones/focal_context.py",
    "detector/core/models/backbones/light_attention.py",
    "detector/core/models/backbones/mobilepixornext.py",
]


def audit(config):
    """Check labels against actual operators and fresh production calculations."""
    import torch
    from core.losses.strategies.oga import OgaLossStrategy
    from core.losses.strategies.q_oga import QOgaLossStrategy
    from core.losses.focal_loss import modified_focal_loss
    from core.losses.quality_focal_loss import quality_focal_loss
    from core.models.backbones.focal_context import FocalContext
    from core.models.backbones.light_attention import SimAMCore, ResidualLightAttention
    from core.models.model import CustomModel
    from core.task_groups import resolve_task_groups

    model_cfg, data, loss_cfg = config["model"], config["data"], config["loss"]
    expected = {"backbone": "mobilepixornext", "c4_context": "focal",
                "c4_context_bottleneck": 64, "c4_context_dilations": [1, 2, 3],
                "c4_context_layer_scale_init": .001, "local_attention": "simam",
                "local_attention_simam_lambda": .0001, "local_attention_layer_scale_init": .001,
                "head_mode": "grouped", "cls_encoding": "gaussian", "header_use_iou": True,
                "backbone_out_dim": 32, "neck_fusion_channels": 32, "stage_depths": [3, 4, 2]}
    for key, value in expected.items():
        if model_cfg.get(key) != value:
            raise ValueError(f"This figure preset requires model.{key}={value!r}")
    if data.get("box_mode", "bev") != "bev" or loss_cfg["name"] != "oga" or not loss_cfg.get("use_iou"):
        raise ValueError("This figure preset requires Gaussian BEV OGA with IQA")
    geometry = data["kitti"]["geometry"]
    if (data.get("bev_encoding", {}).get("name") != "hist14" or data.get("out_size_factor") != 4
            or any(geometry.get(key) != value for key, value in {
                "x_min": 0., "x_max": 70.4, "x_res": .1,
                "y_min": -40., "y_max": 40., "y_res": .1}.items())):
        raise ValueError("Displayed spatial shapes/ranges require Hist14 800×704 BEV and stride 4")
    # These are the defaults actually used by the displayed equations/numbers.
    defaults = {"corner_beta": 1., "temperature": 2., "clamp_bound": 3.,
                "ema_momentum": .99, "iou_target_type": "mgiou", "iou_loss_weight": 1.}
    for key, value in defaults.items():
        if loss_cfg.get(key, value) != value:
            raise ValueError(f"This figure preset requires loss.{key}={value!r}")
    groups = resolve_task_groups(data["head_groups"], data["kitti"]["objects"])
    if [(g.name, g.num_classes) for g in groups] != [("car", 1), ("ped_cyc", 2)]:
        raise ValueError("This figure preset expects car / ped_cyc groups")
    group_weights = loss_cfg.get("group_weights", {g.name: 1. for g in groups})
    if group_weights["car"] != group_weights["ped_cyc"]:
        raise ValueError("Displayed group aggregation requires equal group weights")

    torch.set_num_threads(2)
    torch.manual_seed(config.get("seed", 42))
    detector = CustomModel(dict(model_cfg, geometry=data["kitti"]["geometry"],
                                bev_encoding=data["bev_encoding"]),
                           num_classes=data["num_classes"], task_groups=groups).eval()
    focal = detector.backbone.c4_context
    adapter = detector.backbone.c3_light_attention
    assert isinstance(focal, FocalContext)
    assert isinstance(adapter, ResidualLightAttention) and isinstance(adapter.core, SimAMCore)
    with torch.inference_mode():
        feature = torch.randn(1, 96, 100, 88)
        query, seed, logits = focal.project(feature)
        levels, pooled = focal.context_levels(seed)
        fused, gates = focal.context_from_seed(seed, logits)
        reconstructed = sum(gates[:, i:i+1] * level for i, level in enumerate(levels))
        reconstructed += gates[:, -1:] * pooled
        torch.testing.assert_close(fused, reconstructed)
        torch.testing.assert_close(gates.sum(1), torch.ones(1, 100, 88))
        product = query.float() * focal.context_projection(fused).float()
        expected_focal = feature + focal.gamma * focal.out_norm(focal.out_projection(product))
        torch.testing.assert_close(focal(feature), expected_focal)

        x = torch.randn(1, 48, 200, 176)
        d = (x - x.mean((2, 3), keepdim=True)).square()
        variance = d.sum((2, 3), keepdim=True) / (200 * 176 - 1)
        gate = (d / (4 * (variance + .0001)) + .5).sigmoid()
        torch.testing.assert_close(adapter.core.spatial_weights(x), gate)
        torch.testing.assert_close(adapter(x), x + adapter.beta * (x * gate - x))
        assert torch.isfinite(focal(feature)).all() and torch.isfinite(adapter(x)).all()

    target = {"cls": torch.zeros(1, 2, 4, 6), "offset": torch.zeros(1, 2, 4, 6),
              "size": torch.ones(1, 2, 4, 6), "yaw": torch.zeros(1, 2, 4, 6),
              "reg_mask": torch.zeros(1, 4, 6)}
    target["yaw"][:, 0] = 1.
    target["reg_mask"][0, 1, 2] = 1.
    target["reg_mask"][0, 2, 4] = 1.
    target["cls"][0, 0, 1, 2] = 1.
    target["cls"][0, 1, 2, 4] = 1.
    target["cls"][0, 0, 1, 3] = .6
    def prediction():
        return {"cls": torch.randn(1, 2, 4, 6, requires_grad=True),
                "offset": (.05 * torch.randn(1, 2, 4, 6)).requires_grad_(),
                "size": (target["size"] + .05 * torch.randn(1, 2, 4, 6)).requires_grad_(),
                "yaw": (target["yaw"] + .05 * torch.randn(1, 2, 4, 6)).requires_grad_()}
    p = prediction()
    p["iou"] = torch.randn(1, 1, 4, 6, requires_grad=True)
    oga = OgaLossStrategy("gaussian", loss_cfg).train()
    oga_result = oga(p, target)
    oga_result["loss"].backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in p.values())
    assert torch.isfinite(oga_result["loss"])
    torch.testing.assert_close(sum(oga_result[f"weight_{name}"] for name in oga.TASKS), torch.tensor(6.))
    q_report = {}
    for quality_target, warmup in (("mgiou", 0), ("rotated_iou", 0), ("rotated_iou", 4)):
        qcfg = {"name": "q_oga", "use_iou": False, "quality_target": quality_target,
                "quality_warmup_epochs": warmup}
        qoga = QOgaLossStrategy("gaussian", qcfg).train()
        qpred = prediction()
        if warmup:
            for epoch, expected_mix in ((0, 0.), (2, .5), (4, 1.)):
                qoga.set_epoch(epoch)
                q, metrics = qoga._classification_quality(qpred, target)
                assert not q.requires_grad
                assert float(metrics["quality_iou_mix"]) == expected_mix
                if epoch == 0:
                    assert torch.equal(q[target["reg_mask"].bool()], torch.ones(2))
        q, metrics = qoga._classification_quality(qpred, target)
        assert not q.requires_grad and bool(((q >= 0) & (q <= 1)).all())
        result = qoga(qpred, target)
        result["loss"].backward()
        assert torch.isfinite(result["loss"])
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in qpred.values())
        q_report[f"{quality_target}_warmup{warmup}"] = {
            "quality_range": [float(q.min()), float(q.max())],
            "loss": float(result["loss"].detach()),
            "task_names": list(qoga.TASKS), "quality_detached": not q.requires_grad,
            "learned_weight_parameters": sum(p.numel() for p in qoga.parameters()),
        }
    p = torch.randn(1, 2, 4, 6)
    torch.testing.assert_close(quality_focal_loss(p, target["cls"], torch.ones(1, 4, 6)),
                               modified_focal_loss(p, target["cls"]))
    count = lambda module: sum(p.numel() for p in module.parameters())
    assert count(focal) == 25120 and count(adapter.core) == 0 and count(adapter) == 1
    return {
        "focal": {"parameters": count(focal), "input_output_shape": [1, 96, 100, 88],
                  "projection_split": [64, 64, 4], "sequential_dilations": list(focal.dilations),
                  "global_pool_source": "last local level", "gamma_shape": list(focal.gamma.shape)},
        "simam": {"core_parameters": count(adapter.core), "adapter_parameters": count(adapter),
                  "input_output_shape": [1, 48, 200, 176], "lambda": adapter.core.lambda_value,
                  "beta_initial": float(adapter.beta.detach())},
        "oga": {"tasks": list(oga.TASKS), "learned_weight_parameters": count(oga),
                "loss": float(oga_result["loss"].detach()), "all_prediction_gradients_finite": True},
        "q_oga_alternatives": q_report,
        "checks": ["focal reconstruction and gates sum to 1", "SimAM formula and residual reconstruction",
                   "OGA and all Q-OGA alternatives forward/backward finite",
                   "detached quality in [0,1]; curriculum epochs 0/2/4",
                   "CQFL at quality=1 equals modified focal loss"],
    }


def title(s, width, height, label, subtitle):
    s.text(width/2, height-.6, label, 22, True)
    s.text(width/2, height-1.4, subtitle, 11, color="muted")


def tensor(s, x, y, w, h, label, fill="offset", size=11):
    s.tensor(x, y, w, h, fill, .3)
    s.text(x, y, label, size)


def loss_inputs(s, y, include_iou=False):
    tensor(s, 18, y, 24, 1.3,
           "Prediction + ground truth · heatmap, metric offset, log-size, doubled-yaw vector"
           + (" [+ IoU logit]" if include_iou else ""), size=12)


def loss_row(s, y, qoga=False):
    labels = ["Heatmap classification\nCQFL(p, Y, q)\nall grid cells" if qoga else
              "Heatmap classification\nModified focal loss\nall grid cells",
              "Offset regression\nRange-weighted Smooth-L1\npositive cells" if qoga else
              "Offset regression\nMasked Smooth-L1\npositive cells",
              "Log-size regression\nRange-weighted Smooth-L1\npositive cells" if qoga else
              "Log-size regression\nMasked Smooth-L1\npositive cells",
              "Doubled-yaw regression\nRange-weighted Smooth-L1\npositive cells" if qoga else
              "Doubled-yaw regression\nMasked Smooth-L1\npositive cells",
              "Oriented geometry\nLgeo = Lproj + βcorner D\nderived from regression"]
    xs = [4, 11, 18, 25, 32]
    for x, label, fill in zip(xs, labels, ["heat", "offset", "size", "yaw", "green"]):
        s.operation(x, y, 5.8, 2.3, label, fill, 10)
        s.arrow([(x, y-1.15), (x, y-1.7)])
    s.arrow([(4, y-1.7), (32, y-1.7)])
    return y-1.7


def geometry_detail(s, y, qoga=False):
    height = 7.2 if qoga else 5.7
    s.panel(.5, y, 24.5, height, "Geometry supervision · positive cells only", "blue")
    cy = y + height - 2.7
    s.operation(6.5, cy, 10.5, 2.7,
                "Decode predicted and GT corners\nsize = exp(clamp(log-size, −10, 10))\n"
                "angle = 0.5 atan2(sin2, cos2)\ncenter = metric offset", "white", 10)
    s.arrow([(11.75, cy), (13.1, cy)])
    s.operation(18.8, cy, 10.5, 2.7,
                "Four projection axes · pred + GT\nLproj = mean[(1 − meanaxis GIoU)/2]\n"
                "Corner distances: direct d0, reversed d180\nNormalize by GT diagonal", "green", 10)
    if qoga:
        s.text(12.75, y+1.9, "Dsoft = mean[−tau logsumexp(−d0/tau, −d180/tau) / diagGT]", 11)
        s.text(12.75, y+1.1, "tau=0.05 · βcorner=1 · no extra geometry head", 10, color="muted")
    else:
        s.text(12.75, y+.9, "Dpi = mean[min(d0, d180) / diagGT] · βcorner=1 · no extra geometry head", 10)


def group_total(s, y, height):
    s.panel(26, y, 9.5, height, "Grouped objective", "yellow")
    cy = y+height-2.2
    s.operation(28.5, cy, 4, 1.3, "Car\nLcar", "white", 11)
    s.operation(33, cy, 4, 1.3, "Ped / Cyc\nLvru", "white", 11)
    s.arrow([(28.5, cy-.65), (28.5, cy-1.3), (30.75, cy-1.3)])
    s.arrow([(33, cy-.65), (33, cy-1.3), (30.75, cy-1.3)])
    s.operation(30.75, y+1.15, 8, 1.1, "Ltotal = 0.5 Lcar + 0.5 Lvru", "peach", 10)
    s.arrow([(30.75, cy-1.3), (30.75, y+1.7)])


def oga_scene():
    s = Scene()
    title(s, 36, 24, "OGA · Oriented Geometric Alignment & Adaptive Loss",
          "Gaussian BEV · one independent criterion per task group · optional IoU quality supervision")
    s.panel(.5, 13, 35, 8.6, "Dual-stream supervision: parameter losses + joint geometry", "panel")
    loss_inputs(s, 19.7, include_iou=True)
    s.arrow([(18, 19.05), (18, 17.7), (4, 17.7)])
    s.arrow([(4, 17.7), (32, 17.7)])
    for x in (4, 11, 18, 25, 32):
        s.arrow([(x, 17.7), (x, 16.95)])
    bus_y = loss_row(s, 15.8)
    s.operation(15, 11.1, 21, 2.0,
                "T-SBUW · EMA calibration + learned temperature-softmax task weights\n"
                "Tasks: cls, offset, size, yaw, geo [+ iou] · M=5 or 6", "purple", 12)
    s.arrow([(18, bus_y), (18, 12.1)])
    s.panel(26, 8.6, 9.5, 4.2, "Optional IQA · enabled here", "yellow")
    s.operation(30.75, 10.5, 8, 2.2,
                "Quality target q = stop-gradient MGIoU\nBCEWithLogits(iou_logit, q)\n"
                "reg_mask only · iou_loss_weight=1", "white", 10)
    s.arrow([(30, 19.7), (35.75, 19.7), (35.75, 10.5), (34.75, 10.5)])
    s.arrow([(26.75, 10.5), (25.9, 10.5), (25.9, 11.1), (25.5, 11.1)])
    s.text(13, 9.1, "Repeat independently per group: separate learned task logits and EMA buffers", 10, color="muted")
    geometry_detail(s, 2.8)
    group_total(s, 2.8, 5.7)
    s.text(18, 1.55, "Masked Smooth-L1 divides by selected scalar coordinates + 1e−4; geometry averages positive boxes.", 11)
    s.text(18, .7, "Empty targets: regression, geometry and IQA terms return differentiable zero; background focal supervision remains.", 10, color="muted")
    return s, (36, 24)


def qoga_scene():
    s = Scene()
    title(s, 36, 27, "Q-OGA · Quality-Aligned & Range-Adaptive OGA",
          "Gaussian BEV alternative · classification quality from decoded boxes · five adaptive tasks, without a separate IQA head")
    s.panel(.5, 13, 35, 11.6, "Quality-coupled heatmap + range-weighted regression + smooth geometry", "panel")
    loss_inputs(s, 22.9)
    s.operation(11.2, 20, 20.3, 3.4,
                "Detached quality map q · no gradient into regression\n"
                "Default mgiou: clamp(mean 4-axis GIoU, 0, 1)\n"
                "Alternative rotated_iou: exact aligned polygon IoU\n"
                "Optional rotated_iou curriculum: a=clip(epoch/Ewarm, 0, 1)\nqpeak=(1−a)+a IoU; default Ewarm=0", "green", 11)
    s.operation(28.8, 20, 11.1, 3.4,
                "Range-adaptive weights from GT\n"
                "world = cell-center grid + GT offset\n"
                "r = Euclidean distance to origin\n"
                "wr = 1 + 1.5 clip(r/70.4, 0, 1.5)^2", "blue", 11)
    s.arrow([(11.2, 22.25), (11.2, 21.7)])
    s.arrow([(28.8, 22.25), (28.8, 21.7)])
    s.arrow([(4, 18.3), (4, 16.65)])
    s.arrow([(28.8, 18.3), (28.8, 17.4), (11, 17.4)])
    for x in (11, 18, 25):
        s.arrow([(x, 17.4), (x, 16.65)])
    s.arrow([(30, 22.9), (35.75, 22.9), (35.75, 17.4), (32, 17.4), (32, 16.65)])
    bus_y = loss_row(s, 15.5, qoga=True)
    s.operation(18, 11.4, 24, 2,
                "T-SBUW · EMA-calibrated five-task objective\n"
                "Lcls + Loffset + Lsize + Lyaw + Lgeo with learned task weights", "purple", 12)
    s.arrow([(18, bus_y), (18, 12.4)])
    s.text(18, 9.9, "CQFL peak target = q^βq; Gaussian neighbors keep negative targets. βq=1, focal power=2.", 11)
    geometry_detail(s, 2.4, qoga=True)
    group_total(s, 2.4, 7.2)
    s.text(18, 1.35, "Coordinate loss = meanpositive[wr × sumcoordinate Smooth-L1] · range weighting applies to offset, size and yaw.", 11)
    s.text(18, .55, "Default mgiou quality is a projection proxy; rotated_iou is area IoU. Curriculum epoch starts at zero.", 10, color="muted")
    return s, (36, 27)


def focal_context_scene():
    s = Scene()
    title(s, 34, 25, "Focal Context · Feature-based Context Modulation",
          "C4 / stride 8 · C=96, bottleneck D=64, dilations=(1,2,3) · 25,120 parameters")
    s.panel(.5, 1.6, 33, 21.2, "FocalContext v1 · sequential local levels + gated global context", "panel")
    tensor(s, 4.5, 20.5, 5, 1.5, "Input X\n96 × 100 × 88")
    s.operation(12, 20.5, 7, 1.7, "BatchNorm\nConv1×1: 96→132, bias=False", "peach", 11)
    s.operation(18.3, 20.5, 4.4, 1.7, "Split Q / S / G\n64 / 64 / 4", "purple", 11)
    s.arrow([(7, 20.5), (8.5, 20.5)])
    s.arrow([(15.5, 20.5), (16.1, 20.5)])
    s.arrow([(18.3, 19.65), (18.3, 19.2), (4.5, 19.2)])
    s.arrow([(4.5, 19.2), (29.5, 19.2)])
    for x in (4.5, 13.7, 29.5):
        s.arrow([(x, 19.2), (x, 18.4)])
    tensor(s, 4.5, 17.8, 5, 1.2, "Query Q\n64 × 100 × 88", "offset")
    tensor(s, 13.7, 17.8, 6, 1.2, "Context seed S\n64 × 100 × 88", "size")
    tensor(s, 29.5, 17.8, 6, 1.2, "Gate logits G\n4 × 100 × 88", "yaw")
    s.operation(29.5, 16.1, 6, 1.2, "Softmax over 4 levels\nFP32 · sumlevel w=1", "yellow", 10)
    s.arrow([(29.5, 17.2), (29.5, 16.7)])
    for y, dilation in ((14.9, 1), (12.5, 2), (10.1, 3)):
        s.operation(13.7, y, 6, 1.6,
                    f"Depthwise Conv3×3 · dilation {dilation}\n64 groups · padding {dilation} · SiLU", "peach", 10)
    for upper, lower in ((17.2, 15.7), (14.1, 13.3), (11.7, 10.9)):
        s.arrow([(13.7, upper), (13.7, lower)])
    s.operation(13.7, 7.7, 6, 1.6, "Global spatial mean of S3\nFP32 · 64 × 1 × 1", "yellow", 10)
    s.arrow([(13.7, 9.3), (13.7, 8.5)])
    s.operation(23, 11.3, 6, 8.8,
                "Weighted context fusion\n\nF = w1 S1 + w2 S2\n+ w3 S3 + w4 global(S3)\n\n"
                "Global vector is broadcast\nGates vary per pixel\n64 × 100 × 88", "green", 10)
    for y, label in ((14.9, "S1"), (12.5, "S2"), (10.1, "S3"), (7.7, "global")):
        s.arrow([(16.7, y), (20, y)])
        s.text(18.35, y+.35, label, 9, color="muted")
    s.arrow([(26.5, 16.1), (23, 16.1), (23, 15.7)])
    s.operation(29.5, 11.3, 6, 1.6, "Context Conv1×1\n64→64, bias=False", "peach", 11)
    s.arrow([(26, 11.3), (26.5, 11.3)])
    s.circle(15.5, 5.5, "×")
    s.arrow([(4.5, 17.2), (4.5, 5.5), (15.2, 5.5)])
    s.arrow([(29.5, 10.5), (29.5, 5.5), (15.8, 5.5)])
    s.text(15.5, 6.25, "Q × projected context · FP32", 10)
    s.operation(22.6, 3.7, 8, 1.6, "Conv1×1: 64→96 · BatchNorm\n× gamma · 96 learned channel scales", "peach", 11)
    s.arrow([(15.5, 5.2), (15.5, 3.7), (18.6, 3.7)])
    s.circle(29, 3.7)
    s.arrow([(26.6, 3.7), (28.7, 3.7)])
    s.arrow([(2, 20.5), (1, 20.5), (1, 2.65), (29, 2.65), (29, 3.4)])
    tensor(s, 32, 3.7, 2.5, 1.5, "Output Y\n96×100×88", "offset", 9)
    s.arrow([(29.3, 3.7), (30.75, 3.7)])
    s.text(17, .7, "Y = X + gamma ⊙ BN(Conv1×1(Q ⊙ Conv1×1(F))) · gamma initial=0.001 · spatial size preserved", 11)
    return s, (34, 25)


def simam_scene():
    s = Scene()
    title(s, 34, 20, "SimAM · Spatial Energy Attention + Residual Adapter",
          "Stage 2 / C3 · 48 × 200 × 176 · per-batch, per-channel spatial statistics")
    s.panel(.5, 5.8, 33, 11.7, "SimAMCore · zero learned parameters", "panel")
    tensor(s, 3.8, 15.3, 5.4, 1.3, "Input X\n48 × 200 × 176")
    s.operation(11.5, 15.3, 6, 1.8, "Spatial mean m = meanHW(X)\nFP32 · B × C × 1 × 1", "yellow", 11)
    s.operation(20, 15.3, 7, 1.8, "Squared deviation\nd = (X − m)^2", "blue", 12)
    s.arrow([(6.5, 15.3), (8.5, 15.3)])
    s.arrow([(14.5, 15.3), (16.5, 15.3)])
    s.arrow([(3.8, 15.95), (3.8, 16.5), (20, 16.5), (20, 16.2)])
    s.operation(20, 12.4, 7, 1.8, "Channel-local variance\nv = sumHW(d) / max(HW−1, 1)", "yellow", 11)
    s.arrow([(20, 14.4), (20, 13.3)])
    s.operation(28.8, 12.4, 7, 1.9, "Energy score\ne = d / [4(v + λ)] + 0.5\nλ=0.0001", "green", 11)
    s.arrow([(23.5, 12.4), (25.3, 12.4)])
    s.arrow([(23.5, 15.3), (28.8, 15.3), (28.8, 13.35)])
    s.operation(28.8, 9.5, 6, 1.5, "Spatial gate = sigmoid(e)\nFP32 → cast to input dtype", "purple", 10)
    s.arrow([(28.8, 11.45), (28.8, 10.25)])
    s.circle(21.5, 9.5, "×")
    s.arrow([(25.8, 9.5), (21.8, 9.5)])
    s.arrow([(3.8, 14.65), (3.8, 6.8), (21.5, 6.8), (21.5, 9.2)])
    tensor(s, 15.5, 9.5, 7, 1.5, "T(X) = X ⊙ sigmoid(e)\n48 × 200 × 176", "green", 11)
    s.arrow([(21.2, 9.5), (19, 9.5)])
    s.panel(.5, 1.3, 33, 4.2, "ResidualLightAttention · one learned scalar β", "blue")
    s.operation(17, 3.4, 16, 1.4, "Output Y = X + β [T(X) − X]\n48 × 200 × 176 · input dtype preserved", "offset", 13)
    s.arrow([(15.5, 8.75), (15.5, 4.1)])
    s.arrow([(1.1, 15.3), (.7, 15.3), (.7, 3.4), (9, 3.4)])
    s.operation(29, 3.4, 7, 1.4, "Learned scalar β\ninitial value=0.001", "purple", 11)
    s.arrow([(25.5, 3.4), (25, 3.4)])
    s.text(17, .55, "Placement: after Stage 2 / C3, before Down3 and the C3 lateral projection to the FPN", 11)
    return s, (34, 20)


def focal_loss_scene():
    s = Scene()
    title(s, 30, 15.5, "Gaussian Heatmap · Modified Focal Loss",
          "Classification term in OGA · cls_encoding=gaussian · separate from the backbone Focal Context block")
    s.panel(.5, 2.1, 29, 11.3, "CornerNet / CenterNet-style peak supervision", "panel")
    tensor(s, 5, 11.3, 7, 1.5, "Heatmap logits z\nB × K × H × W", "heat", 12)
    s.operation(15, 11.3, 8.5, 1.5, "p = clamp(sigmoid(z), 1e−4, 1−1e−4)\nprobability and log operations in FP32", "purple", 11)
    tensor(s, 25, 11.3, 7, 1.5, "Gaussian target Y\nB × K × H × W", "offset", 12)
    s.arrow([(8.5, 11.3), (10.75, 11.3)])
    s.arrow([(15, 10.55), (15, 9.4), (8, 9.4), (8, 8.85)])
    s.arrow([(15, 9.4), (23, 9.4), (23, 8.85)])
    s.arrow([(25, 10.55), (25, 9.4), (23, 9.4)])
    s.operation(8, 7.8, 11, 2.1, "Peak centers · Y=1\nLpos = −(1−p)^2 log(p)", "green", 13)
    s.operation(23, 7.8, 12, 2.1, "Background + Gaussian neighbors · Y<1\nLneg = −(1−Y)^4 p^2 log(1−p)", "peach", 12)
    s.operation(15, 4.3, 20, 1.9, "Lcls = [sum(Lpos) + sum(Lneg)] / max(Npeak, 1)\nNo peaks: retain the summed background loss", "purple", 13)
    s.arrow([(8, 6.75), (8, 5.25)])
    s.arrow([(23, 6.75), (23, 5.25)])
    s.text(15, 1.2, "Q-OGA replaces only the peak term with |q^βq−p|^2 BCE(p, q^βq); the negative term remains the same.", 11)
    s.text(15, .45, "At q=1, the implemented quality-coupled focal loss equals this modified focal loss.", 10, color="muted")
    return s, (30, 15.5)


def weighting_scene():
    s = Scene()
    title(s, 30, 18, "T-SBUW · Adaptive Task Weighting with EMA Calibration",
          "Used by both OGA and Q-OGA · learned loss weights are training-only parameters")
    s.panel(.5, 2.4, 29, 13.2, "One separate weighting module per group", "panel")
    tensor(s, 5, 13.4, 7, 1.6, "Raw scalar task losses Li\nM=5, or M=6 with OGA IQA", "offset", 11)
    s.operation(14, 13.4, 8, 2, "Detached running loss means mi\nFirst train call: max(Li, 1e−4)\nThen EMA momentum=0.99", "yellow", 11)
    s.arrow([(8.5, 13.4), (10, 13.4)])
    s.operation(25, 13.4, 7, 2, "Learned logits si\nM parameters · initial=0", "purple", 12)
    s.operation(25, 10.6, 7, 1.7, "Bounded scales\nsbi = 3 tanh(si / 3)", "purple", 12)
    s.operation(25, 8, 7, 1.7, "Temperature softmax\nwi = M softmax(sbi / 2)", "green", 12)
    s.arrow([(25, 12.4), (25, 11.45)])
    s.arrow([(25, 9.75), (25, 8.85)])
    s.operation(11, 9, 14, 1.7, "Calibrated tasks: Lci = Li / max(mi, 1e−4)\nEMA buffers receive no gradient", "blue", 12)
    s.arrow([(5, 12.6), (5, 11.2), (7, 11.2), (7, 9.85)])
    s.arrow([(14, 12.4), (14, 9.85)])
    s.operation(20, 5.4, 16, 1.8, "Lgroup = mean(mi) × sum_i(wi × Lci)\nnominal scale mean(mi) is detached", "peach", 13)
    s.arrow([(11, 8.15), (11, 6.8), (16, 6.8), (16, 6.3)])
    s.arrow([(25, 7.15), (25, 6.3)])
    s.text(15, 3.2, "All wi > 0 · sum_i wi = M · eval mode freezes EMA updates · each group owns its logits and buffers", 11)
    s.text(15, 1.45, "Defaults: temperature=2 · bound=3 · EMA momentum=0.99 · loss calibration floor=1e−4", 11)
    s.text(15, .6, "Task weights act on loss terms; they are distinct from fixed outer group weights (0.5 / 0.5 here).", 11, color="muted")
    return s, (30, 18)


FIGURES = [
    ("oga", "OGA", oga_scene),
    ("q_oga", "Q-OGA", qoga_scene),
    ("focal_context", "Focal Context", focal_context_scene),
    ("simam", "SimAM", simam_scene),
    ("focal_loss", "Gaussian focal loss", focal_loss_scene),
    ("task_weighting", "T-SBUW task weighting", weighting_scene),
]


def documentation(output, config_path):
    cards = []
    for name, label, _ in FIGURES:
        cards.append(f'<section><h2>{html.escape(label)}</h2><a href="{name}.svg"><img src="{name}.svg" '
                     f'alt="{html.escape(label)}"></a><p><a href="{name}.png">PNG</a> · '
                     f'<a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a> · '
                     f'<a href="{name}.tex">TikZ</a></p></section>')
    gallery = ('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" '
               'content="width=device-width,initial-scale=1"><title>OGA / Q-OGA / Focal / SimAM</title>'
               '<style>body{font:16px Georgia,serif;background:#f4f5f7;color:#263340;margin:2rem}'
               'main{max-width:1500px;margin:auto}section{background:white;border-radius:12px;'
               'padding:1rem 1.5rem;margin:1.5rem 0}img{width:100%;height:auto}a{color:#326295}'
               'h1,h2{margin:.5rem 0 1rem}p{line-height:1.6}</style><main>'
               '<h1>OGA, Q-OGA, Focal Context và SimAM</h1><p>Vẽ theo code hiện tại. '
               'Nhấp hình để mở SVG đầy đủ. <a href="loss_attention_atlas.pdf">PDF gồm 6 hình</a>.</p>'
               + ''.join(cards) + '</main></html>')
    (output/"index.html").write_text(gallery)
    readme = f"""# Loss và attention theo code hiện tại

[Mở gallery](index.html) · [PDF gồm 6 hình](loss_attention_atlas.pdf) · [Audit nguồn và số học](audit.json).

| Hình | PNG | SVG chỉnh sửa | PDF vector |
| --- | --- | --- | --- |
"""
    for name, label, _ in FIGURES:
        readme += f"| {label} | [PNG]({name}.png) | [SVG]({name}.svg) | [PDF]({name}.pdf) |\n"
    readme += f"""
Nguồn config: `{config_path}`; bản sao giữ trong [config.resolved.json](config.resolved.json).
Đây là config OGA/IQA và Focal/SimAM. Q-OGA được vẽ như một lựa chọn loss riêng,
không phải loss đang bật cùng OGA trong config này. Hình loss áp dụng cho Gaussian BEV.

- **OGA:** cls focal, ba masked Smooth-L1, và geometry từ offset/size/yaw. Geometry
  gồm projection GIoU loss và corner distance chọn min(direct, đảo 180°), chia đường
  chéo GT. IQA tùy chọn dùng BCE với target quality detached; config này bật IQA
  với MGIoU proxy và weight=1. Mỗi group có module T-SBUW riêng, 6 logits học được
  khi bật IQA, hoặc 5 khi tắt. Masked Smooth-L1 chuẩn hóa trên số tọa độ scalar,
  còn geometry chuẩn hóa trên số box positive.
- **Q-OGA:** thay focal ở peak bằng CQFL với target q^beta, dùng range weights cho
  ba coordinate losses và soft-min corner matching. Default quality_target=mgiou
  dùng clamp(mean GIoU trên 4 trục,0,1), **không phải polygon IoU**. Opt-in
  quality_target=rotated_iou dùng exact area IoU; curriculum optional có
  a=clip(epoch/Ewarm,0,1), qpeak=(1-a)+a IoU. Epoch tính từ 0; Ewarm mặc định 0.
  Nhánh target quality không truyền gradient về regression. Gaussian neighbors
  vẫn được supervision negative, không chuyển thành positive quality targets.
  Full pipeline Q-OGA không dùng IQA head riêng. Q-OGA có 5 adaptive task logits
  cho mỗi group. RDA theo đúng code dùng grid tại (j+0.5,i+0.5), cộng GT offset;
  coordinate loss dùng sum 2 tọa độ rồi mean box, khác normalization OGA.
  Soft-min trong code không cộng tau*log(2); giá trị corner term có thể âm nhỏ.
- **Focal Context:** C4=96×100×88; BN + Conv1×1 chia thành Q64/S64/G4. Ba DWConv
  dilation 1→2→3 nối tiếp, SiLU sau mỗi lớp. Global context là mean của lớp cuối.
  Softmax G trên 4 levels, fusion theo từng pixel rồi Conv1×1, nhân query, Conv1×1
  và BN. Residual dùng 96 channel scales gamma, khởi tạo .001. 25,120 tham số.
- **SimAM:** C3=48×200×176; mean/variance độc lập cho từng batch/channel, trên HW.
  d=(X-meanHW(X))², v=sumHW(d)/max(HW-1,1), e=d/[4(v+lambda)]+.5.
  Core T(X)=X*sigmoid(e) không có tham số; wrapper Y=X+beta(T(X)-X) thêm một
  scalar beta học được, khởi tạo .001. Lambda=.0001; các thống kê dùng FP32.
  Đặt sau Stage2, trước Down3 và lateral C3.
- **Focal loss** là loss heatmap riêng, khác **Focal Context** trong backbone.
- **T-SBUW:** s'=3*tanh(s/3), w=M*softmax(s'/2), loss calibration Li/max(EMAi,1e-4).
  Lgroup=mean(EMA)*sum(w*calibrated loss). EMA dùng detach và update trong train,
  init bằng loss clamp đầu tiên, sau đó momentum .99. Group weights phía ngoài
  được chuẩn hóa thành .5/.5 trong config này; không phải các task weights.

Generator chạy các module production, kiểm tra phép dựng lại Focal/SimAM,
shape, số tham số, forward/backward OGA/Q-OGA, quality detached, curriculum
epochs 0/2/4, và CQFL(q=1)=modified focal loss. Không sửa model/loss hay chạy
huấn luyện dataset. PNG 300 DPI; SVG giữ text; PDF vector; TikZ dùng PlotNeuralNet
từ third_party, cần `.tex` và `layers/` khi tải Overleaf. TikZ chưa biên dịch
vì môi trường thiếu LaTeX compiler; PDF được xuất bằng Matplotlib.

Tạo lại từ folder Code:

```bash
python tools/visualization/draw_loss_attention_diagrams.py
```
"""
    (output/"README.md").write_text(readme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT/"configs/experiments/under1m/hist14_local3_focal_simam_fusion32_shared_grouped_oga_iqa.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"diagrams/mobilepixornext/loss_attention")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT), str(ROOT/"detector"), str(ROOT/"detector/core/datasets")]
    config = json.loads(args.config.read_text())
    report = audit(config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(VENDOR/"PlotNeuralNet/Layers", output/"layers", dirs_exist_ok=True)
    shutil.copy2(VENDOR/"LICENSE", output/"layers/LICENSE")
    (output/"config.resolved.json").write_text(json.dumps(config, indent=2, sort_keys=True)+"\n")
    for name, _, build in FIGURES:
        scene, canvas = build()
        render(scene, output/name, args.dpi, canvas)
        write_tikz(scene, output/name, canvas)
        # The shared renderer escapes labels for text-mode TikZ. These new
        # formula labels also contain literal powers, so protect their carets.
        tex_path = output/f"{name}.tex"
        tex_path.write_text(tex_path.read_text().replace("^", r"\textasciicircum{}"))
        print(f"Exported {name}: PNG / SVG / vector PDF / TikZ", flush=True)
    subprocess.run(["pdfunite", *(str(output/f"{name}.pdf") for name, _, _ in FIGURES),
                    str(output/"loss_attention_atlas.pdf")], check=True)
    report.update(source_config=str(args.config.resolve()),
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  source_hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in SOURCES},
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Source-verified diagrams; production model/loss/config unchanged; Q-OGA is an alternative")
    (output/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
    documentation(output, args.config.resolve())
    print(f"Gallery: {output/'index.html'}", flush=True)


if __name__ == "__main__":
    main()
