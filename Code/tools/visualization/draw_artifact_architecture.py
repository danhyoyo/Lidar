#!/usr/bin/env python3
"""Export a source-verified architecture atlas in the MobilePixorNeXt paper style.

Uses the shared Matplotlib scene and vendored PlotNeuralNet TikZ primitives.
Run from Code with --config /path/to/config.resolved.json --output-dir /path/to/atlas.
"""

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

from draw_mobilepixornext_paper import (
    ROOT, VENDOR, Scene, architecture_scene, gate_inset, inspect_config,
    next_inset, render, shape, write_tikz,
)


def audit_model(config, report, *, local_attention="none"):
    """Record actual operators and verify the full grouped detector on CPU."""
    import torch
    from torch import nn
    sys.path.insert(0, str(ROOT / "detector"))
    from core.models.model import CustomModel
    from core.task_groups import resolve_task_groups

    data, cfg = config["data"], config["model"]
    required = {
        "input_shape": [1, 14, 800, 704], "output_shape": [1, 32, 200, 176],
        "stage_depths": [3, 4, 2], "fusion_channels": 32, "expansion": 2.5,
        "eca": False, "focal": True, "focal_bottleneck": 64,
        "focal_dilations": [1, 2, 3], "grouped": True, "iou": True,
        "head_groups": [{"name": "car", "classes": ["Car"]},
                        {"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"]}],
    }
    for key, expected in required.items():
        if report[key] != expected:
            raise ValueError(f"This atlas targets hist14/focal/none/f32: {key} must be {expected}")
    groups = resolve_task_groups(data["head_groups"], data["kitti"]["objects"])
    torch.manual_seed(config.get("seed", 42))
    model = CustomModel(
        dict(cfg, geometry=data["kitti"]["geometry"], bev_encoding=data["bev_encoding"]),
        num_classes=data["num_classes"], input_channels=report["input_shape"][1],
        task_groups=groups, box_mode=data.get("box_mode", "bev"),
    ).eval()
    operators = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            operators[name] = dict(
                type="Conv2d", input_channels=module.in_channels,
                output_channels=module.out_channels, kernel=list(module.kernel_size),
                stride=list(module.stride), padding=list(module.padding),
                dilation=list(module.dilation), groups=module.groups,
                bias=module.bias is not None,
            )
    backbone = model.backbone
    if local_attention == "none":
        assert isinstance(backbone.c3_light_attention, nn.Identity)
    elif local_attention == "simam":
        from core.models.backbones.light_attention import ResidualLightAttention, SimAMCore
        adapter = backbone.c3_light_attention
        assert isinstance(adapter, ResidualLightAttention) and isinstance(adapter.core, SimAMCore)
        assert sum(p.numel() for p in adapter.core.parameters()) == 0
        assert sum(p.numel() for p in adapter.parameters()) == 1
        assert abs(float(adapter.beta.detach()) - .001) < 1e-9
        assert adapter.core.lambda_value == .0001
    else:
        raise ValueError("This atlas supports local_attention none or simam")
    assert not backbone.use_reparam and not backbone.detail_path
    assert backbone.c4_context.bottleneck == 64
    assert backbone.c4_context.dilations == (1, 2, 3)
    assert backbone.neck_fusion_channels == 32 and backbone.backbone_out_dim == 32
    assert backbone.stage_depths == (3, 4, 2)
    assert cfg["expansion"] == 2.5
    assert abs(float(backbone.c4_context.gamma.flatten()[0].detach()) - .001) < 1e-9
    for stage, c in ((backbone.stage2, 48), (backbone.stage3, 96), (backbone.stage4, 128)):
        for block in stage:
            assert block.dwconv.groups == c and block.dwconv.kernel_size == (7, 7)
            assert block.pw_expand.out_channels == round(c * 2.5)
            assert abs(float(block.layer_scale.flatten()[0].detach()) - 1e-5) < 1e-10
    # This atlas deliberately targets the requested preset. Reject other variants
    # rather than silently keeping their labels at the values used by this atlas.
    for gate in (backbone.gate_c4, backbone.gate_c3):
        assert torch.count_nonzero(gate.weight) == 0
        assert torch.count_nonzero(gate.bias) == 0
    captured = {}
    def capture(name):
        def hook(module, inputs, output):
            captured[name] = list(output.shape)
            assert torch.isfinite(output).all(), name
        return hook
    hooks = [module.register_forward_hook(capture(name))
             for name, module in backbone.named_children()
             if name in {"down2", "down3", "down4", "stage2", "stage3", "stage4",
                         "c3_light_attention", "c4_context", "gate_c4", "gate_c3", "out_conv"}]
    try:
        with torch.inference_mode():
            outputs = model(torch.zeros(report["input_shape"]))
    finally:
        for hook in hooks:
            hook.remove()
    head_shapes = {}
    for group in groups:
        predictions = outputs["groups"][group.name]
        widths = dict(cls=group.num_classes, offset=2, size=2, yaw=2, iou=1)
        assert set(predictions) == set(widths)
        head_shapes[group.name] = {}
        for name, value in predictions.items():
            assert list(value.shape) == [1, widths[name], *report["output_shape"][2:]]
            assert torch.isfinite(value).all()
            head_shapes[group.name][name] = list(value.shape)
    count = lambda module: sum(p.numel() for p in module.parameters())
    assert count(backbone) == report["parameters_including_neck"]
    return dict(
        module_parameters={name: count(module) for name, module in backbone.named_children()},
        head_parameters={name: count(module) for name, module in model.grouped_header.heads.items()},
        total_detector_parameters=count(model), backbone_shapes=captured,
        grouped_output_shapes=head_shapes, operators=operators,
        focal_projection_split=[64, 64, 4], focal_projection_channels=132,
        gate_initial_multiplier=1.0,
        verification="Fresh CPU eval full detector forward, batch 1 at 14×800×704; finite tensors and exact head shapes",
    )


def base(title, subtitle, width=24, height=7, fill="blue"):
    s = Scene()
    s.text(width / 2, height - .48, title, 20, True)
    s.text(width / 2, height - 1.15, subtitle, 10, color="muted")
    s.panel(.45, .8, width - .9, height - 2.45, "", fill)
    return s, (width, height)


def chain(s, nodes, y, start=1.65, gap=.7):
    """Place a series of explicitly connected operations with stable spacing."""
    x = start
    previous = None
    positions = []
    for label, w, fill in nodes:
        center = x + w / 2
        if previous is not None:
            s.arrow([(previous, y), (x, y)])
        s.operation(center, y, w, 1.1, label, fill, 10)
        positions.append((center, w))
        previous = x + w
        x = previous + gap
    return positions


def crop(scene, bounds, canvas, title=None, subtitle=None):
    """Reuse a panel from the shared scene without raster cropping."""
    x0, y0, x1, y1 = bounds
    out = Scene()
    for kind, original in scene.items:
        v = copy.deepcopy(original)
        if kind == "arrow":
            if not all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in v["points"]):
                continue
            v["points"] = [(x - x0 + .25, y - y0 + .7) for x, y in v["points"]]
        else:
            if not (x0 <= v["x"] <= x1 and y0 <= v["y"] <= y1):
                continue
            v["x"] += .25 - x0
            v["y"] += .7 - y0
        out.items.append((kind, v))
    if title:
        out.text(canvas[0] / 2, canvas[1] - .5, title, 20, True)
        out.text(canvas[0] / 2, canvas[1] - 1.15, subtitle, 10, color="muted")
    return out, canvas


def stem_scene():
    s, canvas = base("Spatial stem", "Two ordinary convolutions · output stride 2", 24, 7)
    chain(s, [("Input\n14 × 800 × 704", 2.8, "offset"),
              ("Conv 3×3\n14→32, s=2, p=1", 3.25, "peach"),
              ("BN + SiLU", 2.2, "white"),
              ("Conv 3×3\n32→32, s=1, p=1", 3.25, "peach"),
              ("BN + SiLU", 2.2, "white"),
              ("Output\n32 × 400 × 352", 2.8, "green")], 3.25, start=1.35, gap=.65)
    s.text(12, 1.55, "Both convolutions: groups=1, bias=False", 10)
    return s, canvas


def stage_scene(index, report):
    c_in, c, depth, h, w = ((32, 48, 3, 200, 176), (48, 96, 4, 100, 88),
                           (96, 128, 2, 50, 44))[index]
    focal = index == 1
    s, canvas = base(f"Stage {index+2} / C{index+3}",
                     f"Stride {4 * 2**index} · {depth} NeXt blocks · expansion 2.5", 26, 7)
    nodes = [(f"Input\n{c_in} × {h*2} × {w*2}", 3, "offset"),
             (f"Downsample\nConv 3×3, {c_in}→{c}, s=2\nBN + SiLU", 4, "peach")]
    nodes += [(f"NeXt {i+1}\n{c}→{round(2.5*c)}→{c}", 2.1, "blue") for i in range(depth)]
    if focal:
        nodes += [("Focal Context\n96→64→96", 2.6, "purple")]
    nodes += [(f"C{index+3}\n{c} × {h} × {w}", 2.8, "green")]
    widths = sum(n[1] for n in nodes)
    chain(s, nodes, 3.25, start=(26-widths-.55*(len(nodes)-1))/2, gap=.55)
    s.text(13, 1.55, "Focal output feeds both L4 and Downsample 4" if focal
           else "Residual NeXt blocks preserve channel count and spatial resolution", 10)
    return s, canvas


def next_scene(c, report):
    s = Scene()
    next_inset(s, report)
    s, canvas = crop(s, (0, .3, 18.6, 7.6), (19.1, 8.7))
    for kind, v in s.items:
        if kind == "text":
            if v["label"] == "(A) MobilePixorNeXt block":
                v["label"] = f"NeXt block · C={c}, expanded C={round(2.5*c)}"
                v["size"] = 14
            v["label"] = v["label"].replace("C→2.5C", f"{c}→{round(2.5*c)}").replace(
                "2.5C→C", f"{round(2.5*c)}→{c}").replace("C→C", f"{c}→{c}")
    s.text(9.55, .35, "DW7×7: padding 3, groups=C · PW1×1: groups=1 · all convolutions: bias=False", 9)
    return s, canvas


def gate_scene(c, level, report):
    s = Scene()
    gate_inset(s, report)
    s, canvas = crop(s, (18.7, .3, 35.8, 7.6), (17.6, 8.7))
    for kind, v in s.items:
        if kind == "text" and v["label"] == "(B) Scale-gated fusion":
            v["label"] = f"Scale-gated fusion · P{level}, C={c}"
            v["size"] = 14
    s.text(8.8, .35, f"Gate DWConv: {c}→{c}, groups={c}, bias=True · no BN or activation before sigmoid", 9)
    return s, canvas


def focal_scene():
    s, canvas = base("Focal Context · C4 / stride 8",
                     "C=96 · bottleneck D=64 · sequential dilation levels 1, 2, 3", 30, 20, "blue")
    s.text(1.5, 15.4, "X", 14, True)
    s.text(3.5, 17.0, "96 × 100 × 88", 10)
    s.arrow([(1.8, 15.4), (2.8, 15.4)])
    s.operation(3.5, 15.4, 1.4, 1.1, "BN", "white")
    s.arrow([(4.2, 15.4), (4.8, 15.4)])
    s.operation(6.35, 15.4, 3.1, 1.1, "Conv 1×1\n96→132", "peach")
    s.arrow([(7.9, 15.4), (8.65, 15.4)])
    s.operation(9.75, 15.4, 2.2, 1.1, "Split\n64 / 64 / 4", "green")
    for y, label, fill in [(16.5, "Q · 64 ch", "offset"),
                            (13.5, "S · 64 ch", "offset"), (6.4, "G · 4 ch", "purple")]:
        s.arrow([(10.85, 15.4), (11.3, 15.4), (11.3, y), (11.8, y)])
        s.operation(12.95, y, 2.3, .9, label, fill, 10)
    for i, x in enumerate((16.25, 20.05, 23.85)):
        s.operation(x, 13.5, 3.0, 1.1, f"DWConv 3×3\nd={i+1}, p={i+1}\nSiLU", "purple", 10)
        previous = 14.1 if i == 0 else x-3.8+1.5
        s.arrow([(previous, 13.5), (x-1.5, 13.5)])
        s.text(x, 14.55, f"Z{i+1} · 64 ch", 10, True)
    s.operation(26.5, 11.15, 3.5, 1.1, "Global mean of Z3\nFP32, H/W → 1/1", "yellow", 9)
    s.arrow([(25.35, 13.5), (26.5, 13.5), (26.5, 11.7)])
    s.operation(19.75, 9.0, 5.2, 1.35, "Weighted context sum\ng1Z1 + g2Z2 + g3Z3 + g4Zglobal", "green", 9)
    for x, tx in ((16.25, 18.25), (20.05, 19.75), (23.85, 21.25)):
        s.arrow([(x, 12.95), (x, 10.75), (tx, 9.675)])
    s.arrow([(26.5, 10.6), (26.5, 9), (22.35, 9)])
    s.operation(16.65, 6.4, 3.2, 1.1, "Softmax(G)\nFP32, dim=channels", "purple", 9)
    s.arrow([(14.1, 6.4), (15.05, 6.4)])
    s.arrow([(18.25, 6.4), (19.75, 6.4), (19.75, 8.325)])
    s.text(16.5, 5.25, "g1 + g2 + g3 + g4 = 1 at each pixel", 9)
    s.operation(23.75, 6.4, 2.8, 1.1, "Conv 1×1\n64→64", "peach", 10)
    s.arrow([(19.75, 8.325), (20.9, 7.6), (23.75, 7.6), (23.75, 6.95)])
    s.circle(27.5, 6.4, "×")
    s.arrow([(25.15, 6.4), (27.2, 6.4)])
    s.arrow([(14.1, 16.5), (28.3, 16.5), (28.3, 6.4), (27.8, 6.4)])
    s.text(26.5, 7.35, "Q ⊙ context", 10)
    s.operation(26.65, 3.7, 3.0, 1.1, "Conv 1×1\n64→96", "peach", 10)
    s.arrow([(27.5, 6.1), (27.5, 4.25)])
    s.operation(22.75, 3.7, 1.4, 1.1, "BN", "white")
    s.arrow([(25.15, 3.7), (23.45, 3.7)])
    s.operation(19.75, 3.7, 2.5, 1.1, "LayerScale\nγ₀ = 0.001", "purple", 10)
    s.arrow([(22.05, 3.7), (21.0, 3.7)])
    s.circle(16.4, 3.7)
    s.arrow([(18.5, 3.7), (16.7, 3.7)])
    s.arrow([(1.9, 15.4), (1.9, 1.85), (16.4, 1.85), (16.4, 3.4)])
    s.text(9, 2.15, "Identity shortcut", 10)
    s.arrow([(16.1, 3.7), (13, 3.7)])
    s.text(11.1, 3.7, "Y · 96 × 100 × 88", 11, True)
    s.text(15, .37, "Y = X + γ ⊙ BN(Conv1×1(Q ⊙ Conv1×1(context)))", 11)
    return s, canvas


def neck_scene(report):
    s, canvas = base("Scale-gated feature pyramid", "Top-down path · lateral projections have no BN or activation", 25, 16, "yellow")
    ys = (11.9, 7.9, 3.9)
    for level, c, lc, h, w, y in ((3, 48, 32, 200, 176, ys[0]),
                                 (4, 96, 48, 100, 88, ys[1]), (5, 128, 48, 50, 44, ys[2])):
        s.operation(3.2, y, 4, 1.1, f"C{level}\n{c} × {h} × {w}", "blue")
        s.arrow([(5.2, y), (6.1, y)])
        s.operation(8, y, 3.8, 1.1, f"L{level}: Conv 1×1\n{c}→{lc}", "green")
        s.text(8, y-1.0, f"{lc} × {h} × {w}", 10)
        s.arrow([(9.9, y), (18.7, y)])
        if level != 5:
            s.circle(19, y, "G")
            s.text(20, y, f"P{level}", 11, True, align="left")
    s.tensor(19, ys[2], .65, 1.1, "green", .3)
    for low, high, c, dw in ((3.9, 7.9, 48, True), (7.9, 11.9, 32, False)):
        y = (low+high)/2
        s.arrow([(19, low+.55 if dw else low+.3), (19, y-.6)])
        s.operation(19, y, 6.2, 1.2,
                    f"Bilinear ↑2, align_corners=False\n{'DWConv' if dw else 'Conv'} 3×3, 48→{c}\nBN + SiLU", "peach", 10)
        s.arrow([(19, y+.6), (19, high-.3)])
    s.arrow([(19, 12.2), (19, 13.1)])
    s.operation(19, 13.65, 6.2, 1.1, "Output Conv 3×3, 32→32 · BN + SiLU", "green", 10)
    s.text(12.5, 1.8, "P4: 48 × 100 × 88    ·    P3 and output: 32 × 200 × 176", 11)
    s.text(12.5, .35, "G: P = U + 2σ(DWConv(L + U)) ⊙ L; independent gates at P4 and P3", 10)
    return s, canvas


def projection_scene(name, title, nodes, subtitle, width=24):
    s, canvas = base(title, subtitle, width, 7, "yellow")
    total = sum(n[1] for n in nodes) + .75*(len(nodes)-1)
    chain(s, nodes, 3.25, start=(width-total)/2, gap=.75)
    s.text(width/2, 1.55, "Convolution: bias=False · BatchNorm2d · SiLU", 10)
    return name, (s, canvas)


def heads_scene(report):
    s, canvas = base("Grouped BEV detection heads + IoU quality",
                     "Two independent Header modules · input 32 × 200 × 176", 30, 14, "yellow")
    labels = (("Heatmap", "heat"), ("Offset", "offset"), ("Size", "size"),
              ("Yaw", "yaw"), ("IoU quality", "iou"))
    for group_index, group in enumerate(report["head_groups"]):
        x0 = 1 + group_index*14.5
        s.text(x0+6.75, 10.85, " + ".join(group["classes"]), 14, True)
        s.text(x0+6.75, 10.1, f"{group['name']} · K={len(group['classes'])}", 10)
        for i, (label, fill) in enumerate(labels):
            y = 8.85-i*1.4
            s.operation(x0+4.1, y, 5.4, 1.0, "Conv 3×3 → BN → SiLU\nConv 3×3 → BN → SiLU → Conv 1×1", "peach", 9)
            s.arrow([(x0+.6, y), (x0+1.4, y)])
            s.arrow([(x0+6.8, y), (x0+7.6, y)])
            c = len(group["classes"]) if i == 0 else (1 if i == 4 else 2)
            s.operation(x0+10.2, y, 5.2, 1, f"{label}\n{c} × 200 × 176", fill, 10)
        s.arrow([(x0+.6, 8.85), (x0+.6, 3.25)])
    s.text(15, 1.7, "Each branch: 32→32→32→output channels · separate weights for every branch and every group", 10)
    s.text(15, .35, "Displayed outputs are raw predictions; box_mode=bev", 10, color="muted")
    return s, canvas


def scenes(report):
    overview = architecture_scene(report)
    bundle = {"overview": (overview, (36, 26)),
              "backbone": crop(overview, (0, 7.8, 25.9, 24), (26.4, 18.8),
                               "MobilePixorNeXt backbone",
                               f"hist14 + Focal Context + SG-FPN (fusion 32) · {report['parameters_including_neck']:,} parameters"),
              "stem": stem_scene(), "focal_context": focal_scene(),
              "sg_fpn": neck_scene(report), "grouped_heads": heads_scene(report)}
    for i, c in enumerate((48, 96, 128)):
        bundle[f"stage{i+2}"] = stage_scene(i, report)
        bundle[f"next_block_c{c}"] = next_scene(c, report)
    for c, level in ((48, 4), (32, 3)):
        bundle[f"gate_p{level}"] = gate_scene(c, level, report)
    for name, title, nodes, sub in [
        ("upsample_u4", "U4 · stride 16 → stride 8", [
            ("L5\n48 × 50 × 44", 3, "green"), ("Bilinear ↑2", 3, "yellow"),
            ("DWConv 3×3\n48→48, groups=48", 4, "peach"), ("BN + SiLU", 3, "white"),
            ("U4\n48 × 100 × 88", 3, "green")], "Bilinear interpolation: align_corners=False"),
        ("upsample_u3", "U3 · stride 8 → stride 4", [
            ("P4\n48 × 100 × 88", 3, "green"), ("Bilinear ↑2", 3, "yellow"),
            ("Conv 3×3\n48→32, groups=1", 4, "peach"), ("BN + SiLU", 3, "white"),
            ("U3\n32 × 200 × 176", 3, "green")], "Bilinear interpolation: align_corners=False"),
        ("output_projection", "Backbone output projection", [
            ("P3\n32 × 200 × 176", 3, "green"), ("Conv 3×3\n32→32, s=1, p=1", 4, "peach"),
            ("BN", 2, "white"), ("SiLU", 2, "white"),
            ("Head input\n32 × 200 × 176", 3.5, "offset")], "Final feature map · output stride 4"),
    ]:
        key, value = projection_scene(name, title, nodes, sub)
        bundle[key] = value
    return bundle


def write_gallery(output, report, names):
    local_label = " / SimAM" if report.get("simam") else ""
    cards = "\n".join(
        f'<article><h2>{html.escape(name.replace("_", " "))}</h2>'
        f'<a href="{name}.svg"><img src="{name}.svg" loading="lazy" alt="{name}"></a>'
        + "<p>" + " · ".join(f'<a href="{name}.{ext}">{ext.upper()}</a>'
                              for ext in ("svg", "png", "pdf", "tex")) + "</p></article>"
        for name in names)
    (output/"index.html").write_text(
        '<!doctype html><html lang="vi"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>MobilePixorNeXt architecture atlas</title><style>'
        'body{max-width:1500px;margin:40px auto;padding:0 24px;background:#f6f6f3;color:#242629;'
        'font-family:Georgia,serif}article{background:white;padding:24px;margin:28px 0;'
        'border:1px solid #d9d9d4;border-radius:12px}img{width:100%;height:auto}'
        'h1{font-size:32px}h2{font-size:22px}a{color:#355f77}p{line-height:1.7}</style>'
        f'<h1>MobilePixorNeXt · hist14 / Focal{local_label} / SG-FPN f32</h1>'
        f'<p>Backbone: {report["parameters_including_neck"]:,} parameters · '
        '14 × 800 × 704 → 32 × 200 × 176 · stage depths 3 / 4 / 2</p>'
        '<p>Mỗi hình có SVG chỉnh sửa được, PNG 300 DPI, PDF vector và mã TikZ dùng PlotNeuralNet.</p>'
        + cards + '</html>', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    path = args.config.resolve()
    config = json.loads(path.read_text())
    report = inspect_config(path)
    if not report["grouped"] or not report["iou"]:
        raise ValueError("This atlas requires grouped BEV heads with IoU quality")
    report["audit"] = audit_model(config, report)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if path != output/"config.resolved.json":
        shutil.copy2(path, output/"config.resolved.json")
    shutil.copytree(VENDOR/"PlotNeuralNet/Layers", output/"layers", dirs_exist_ok=True)
    shutil.copy2(VENDOR/"LICENSE", output/"layers/LICENSE")
    bundle = scenes(report)
    for name, (scene, canvas) in bundle.items():
        render(scene, output/name, args.dpi, canvas)
        write_tikz(scene, output/name, canvas)
        print(f"Exported {name}: PNG/SVG/PDF/TikZ", flush=True)
    report["figures"] = list(bundle)
    report["png_dpi"] = args.dpi
    report["source_hashes"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT/"tools/visualization/draw_mobilepixornext_paper.py",
                  ROOT/"detector/core/models/backbones/mobilepixornext.py",
                  ROOT/"detector/core/models/backbones/mobilepixornext_blocks.py",
                  ROOT/"detector/core/models/backbones/focal_context.py",
                  ROOT/"detector/core/models/heads/cnn.py",
                  ROOT/"detector/core/models/heads/grouped.py",
                  VENDOR/"PlotNeuralNet/PyCore/TikzGen.py"]}
    (output/"architecture.json").write_text(json.dumps(report, indent=2)+"\n")
    write_gallery(output, report, bundle)
    print(f"Verified full detector: {report['audit']['total_detector_parameters']:,} parameters")


if __name__ == "__main__":
    main()
