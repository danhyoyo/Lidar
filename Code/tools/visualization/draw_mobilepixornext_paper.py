#!/usr/bin/env python3
"""Draw source-verified MobilePixorNeXt figures in the reference paper style.

PNG/SVG/PDF use Matplotlib; editable .tex uses the vendored PlotNeuralNet
ToConv primitive. Both exports consume the same layout. No TeX compiler is
required to produce the vector figures.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party/PlotNeuralNet"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lidar-paper-matplotlib")

COLORS = {
    "ink": "#242629", "muted": "#59616a", "panel": "#f6f6f3",
    "blue": "#dfeaf5", "peach": "#f7cdae", "green": "#a9ce91",
    "yellow": "#ffe49b", "purple": "#d5c5e8", "white": "#ffffff",
    "heat": "#e89a50", "offset": "#a6bdd8", "size": "#ddc66e",
    "yaw": "#a8c983", "iou": "#b6a1cd",
}


def inspect_config(path, *, allow_simam=False):
    import torch
    from torch import nn

    sys.path.insert(0, str(ROOT / "detector"))
    from core.models.backbones.registry import build_backbone

    config = json.loads(path.read_text())
    cfg = config["model"]
    if cfg["backbone"].lower() != "mobilepixornext":
        raise ValueError("The figure requires MobilePixorNeXt")
    from core.backbone_config import resolve_backbone_features
    if not resolve_backbone_features(cfg).has_legacy_placement:
        raise ValueError("This curated layout requires legacy Focal C4/local C3 placement; use export_backbone_diagram.py for selectable stages")
    if (cfg.get("cls_encoding", "gaussian") != "gaussian" or
            config["data"].get("box_mode", "bev") != "bev" or
            not cfg.get("header_use_bn", True) or cfg.get("header_act", "silu") != "silu"):
        raise ValueError("The head template requires Gaussian BEV heads with BN and SiLU")
    geometry = config["data"]["kitti"]["geometry"]
    # Production channel resolver; figure labels use the dense BEV contract.
    from core.bev_encoding import resolve_input_channels
    encoding = config["data"]["bev_encoding"]
    channels = resolve_input_channels(encoding, geometry)
    h = round((geometry["y_max"] - geometry["y_min"]) / geometry["y_res"])
    w = round((geometry["x_max"] - geometry["x_min"]) / geometry["x_res"])
    model = build_backbone("mobilepixornext", dict(cfg, geometry=geometry), channels).eval()
    if (model.rc_neck is not None or model.use_reparam or model.detail_path or
            not model.scale_gated_fpn or
            cfg.get("local_attention", "none") not in
            ({"none", "eca", "simam"} if allow_simam else {"none", "eca"})):
        raise ValueError("This layout covers standard SG-FPN, optional focal/ECA, without RepConv/detail")
    names = ("stem", "c3_light_attention", "c4_context", "stage4", "lat_c3",
             "lat_c4", "lat_c5", "refine_u4", "proj_u3", "out_conv")
    shapes = {}

    def capture(name):
        def hook(module, inputs, output):
            shapes[name] = list(output.shape)
        return hook

    hooks = [getattr(model, name).register_forward_hook(capture(name)) for name in names]
    try:
        torch.set_num_threads(2)
        with torch.inference_mode():
            output = model(torch.zeros(1, channels, h, w))
        if not torch.isfinite(output).all():
            raise ValueError("Forward output is not finite")
    finally:
        for hook in hooks:
            hook.remove()
    expected = ((32, 2), (48, 4), (96, 8), (128, 16),
                (model.neck_fusion_channels, 4), (48, 8), (48, 16), (48, 8),
                (model.neck_fusion_channels, 4), (model.backbone_out_dim, 4))
    for name, (c, stride) in zip(names, expected):
        if shapes[name] != [1, c, h // stride, w // stride]:
            raise ValueError(f"Unexpected shape at {name}: {shapes[name]}")
    grouped = cfg.get("head_mode", "legacy_single") == "grouped"
    groups = config["data"].get("head_groups", []) if grouped else []
    context = model.c4_context
    local = model.c3_light_attention
    return {
        "config": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "encoding": encoding["name"], "input_shape": [1, channels, h, w],
        "output_shape": list(output.shape), "shapes": shapes,
        "stage_depths": list(model.stage_depths),
        "expansion": model.stage2[0].pw_expand.out_channels / 48,
        "layer_scale_init": float(model.stage2[0].layer_scale.detach().flatten()[0]),
        "fusion_channels": model.neck_fusion_channels,
        "scale_gated_fpn": model.scale_gated_fpn,
        "focal": not isinstance(context, nn.Identity),
        "focal_bottleneck": getattr(context, "bottleneck", None),
        "focal_dilations": list(getattr(context, "dilations", ())),
        "eca": cfg.get("local_attention", "none") == "eca",
        "local_attention": cfg.get("local_attention", "none"),
        "simam": cfg.get("local_attention", "none") == "simam",
        "simam_lambda": getattr(getattr(local, "core", None), "lambda_value", None),
        "local_attention_beta_init": (float(local.beta.detach())
                                      if hasattr(local, "beta") else None),
        "eca_kernel": getattr(getattr(local, "core", None), "kernel_size", None),
        "parameters_including_neck": sum(p.numel() for p in model.parameters()),
        "grouped": grouped, "head_groups": groups,
        "num_classes": config["data"]["num_classes"],
        "iou": cfg.get("header_use_iou", False),
        "verified": "CPU eval forward at full configured resolution; finite output and checked shapes",
    }


def shape(value):
    return " × ".join(map(str, value[1:]))


class Scene:
    """Small shared drawing description for Matplotlib and editable TikZ."""

    def __init__(self):
        self.items = []

    def text(self, x, y, label, size=12, bold=False, color="ink", align="center", rotation=0):
        self.items.append(("text", dict(x=x, y=y, label=label, size=size,
                                        bold=bold, color=color, align=align, rotation=rotation)))

    def box(self, x, y, w, h, fill="blue", dashed=False):
        self.items.append(("box", dict(x=x, y=y, w=w, h=h, fill=fill, dashed=dashed)))

    def arrow(self, points, color="ink", dashed=False):
        self.items.append(("arrow", dict(points=points, color=color, dashed=dashed)))

    def circle(self, x, y, label="+", r=0.30):
        self.items.append(("circle", dict(x=x, y=y, label=label, r=r)))

    def tensor(self, x, y, w, h, fill="blue", depth=0.55):
        self.items.append(("tensor", dict(x=x, y=y, w=w, h=h, fill=fill, depth=depth)))

    def panel(self, x, y, w, h, label, fill="panel"):
        self.box(x + w / 2, y + h / 2, w, h, fill, dashed=True)
        self.text(x + 0.35, y + h - 0.5, label, 15, True, align="left")

    def operation(self, x, y, w, h, label, fill="peach", size=11):
        self.box(x, y, w, h, fill)
        self.text(x, y, label, size)


def architecture_scene(report):
    s = Scene()
    shapes = report["shapes"]
    fusion = report["fusion_channels"]
    out = report["output_shape"][1]
    enhanced = report["focal"] or report["eca"]
    s.text(18, 25.45, "MobilePixorNeXt", 24, True)
    suffix = (" + Focal Context" if report["focal"] else "") + (" + ECA" if report["eca"] else "")
    s.text(18, 24.73, f"{report['encoding']} BEV{suffix}  ·  "
           f"{report['parameters_including_neck']:,} backbone parameters (including neck)",
           12, color="muted")
    s.panel(0.45, 8.0, 25.25, 15.85, "Backbone")
    s.text(6.3, 22.96, "Bottom-up encoder", 12, True, color="muted")
    s.text(21.6, 22.96, "Top-down SG-FPN", 12, True, color="muted")

    # The stem is deliberately two ordinary convolutions, not a depthwise stem.
    s.tensor(1.45, 21.3, 0.55, 1.5, "offset")
    s.text(1.75, 20.05, report["encoding"] + " input", 10, True)
    s.text(1.75, 19.58, shape(report["input_shape"]), 9)
    s.arrow([(2.0, 21.3), (3.15, 21.3)])
    s.operation(6.65, 21.3, 7.0, 1.2,
                f"Stem: Conv 3×3, {report['input_shape'][1]}→32, s=2\n"
                "Conv 3×3, 32→32, s=1  ·  BN + SiLU", size=11)
    s.text(6.65, 20.35, shape(shapes["stem"]) + "  /2", 10)

    # Stages use semantic C3/C4/C5 labels, independent of Python temporary names.
    for idx, (cy, c, stride, name) in enumerate([
            (17.5, 48, 4, "c3_light_attention"),
            (13.55, 96, 8, "c4_context"), (9.6, 128, 16, "stage4")]):
        depth = report["stage_depths"][idx]
        ds_y = cy + 2.0
        previous_bottom = 20.1 if idx == 0 else cy + 2.85
        s.arrow([(6.65, previous_bottom), (6.65, ds_y + 0.30)])
        prev_c = (32, 48, 96)[idx]
        s.operation(6.65, ds_y, 4.45, 0.60,
                    f"Conv 3×3, {prev_c}→{c}, s=2", size=10)
        s.text(9.60, ds_y, "BN + SiLU", 8.5)
        s.arrow([(6.65, ds_y - 0.30), (6.65, cy + 1.05)])
        s.box(6.65, cy, 7.9, 2.1, "blue")
        s.text(6.65, cy + 0.70, f"Stage {idx+2} / C{idx+3}  ·  {depth} NeXt blocks", 11, True)
        has_refine = (idx == 0 and report["eca"]) or (idx == 1 and report["focal"])
        space = 5.6 if has_refine else 7.1
        start = 3.1
        bw = space / depth - 0.08
        for block in range(depth):
            s.operation(start + (block + 0.5) * space / depth, cy, bw, 0.55,
                        "NeXt", "blue", 10)
        if has_refine:
            s.operation(9.62, cy, 1.63, 0.55,
                        "ECA" if idx == 0 else "Focal", "purple", 10)
        s.text(6.65, cy - 0.68, shape(shapes[name]) + f"  /{stride}", 10)
        s.arrow([(10.6, cy), (11.65, cy)])
        lat_name = ("lat_c3", "lat_c4", "lat_c5")[idx]
        lat_c = shapes[lat_name][1]
        s.operation(13.3, cy, 3.3, 1.2, f"L{idx+3}: Conv 1×1\n{c}→{lat_c}", "green", 11)
        s.text(13.3, cy - 0.95, shape(shapes[lat_name]), 10)
        if idx < 2:
            s.arrow([(14.95, cy), (21.98, cy)])
        else:
            s.arrow([(14.95, cy), (21.9, cy)])

    s.tensor(22.3, 9.6, 0.7, 1.2, "green", 0.50)
    s.arrow([(22.3, 10.23), (22.3, 10.86)])
    s.operation(22.3, 11.50, 4.75, 1.28,
                "Bilinear ↑2\nDWConv 3×3, 48→48\nBN + SiLU", "yellow", 10)
    s.arrow([(22.3, 12.14), (22.3, 13.20)])
    s.circle(22.3, 13.55, "G" if report["scale_gated_fpn"] else "+", .35)
    s.text(23.02, 13.55, "P4", 11, True, align="left")
    s.text(18.0, 12.98, "Gated fusion" if report["scale_gated_fpn"] else "Addition", 10)
    s.arrow([(22.3, 13.90), (22.3, 14.79)])
    s.operation(22.3, 15.45, 4.75, 1.32,
                f"Bilinear ↑2\nConv 3×3, 48→{fusion}\nBN + SiLU", "yellow", 10)
    s.arrow([(22.3, 16.11), (22.3, 17.15)])
    s.circle(22.3, 17.50, "G" if report["scale_gated_fpn"] else "+", .35)
    s.text(23.02, 17.50, "P3", 11, True, align="left")
    s.text(18.0, 16.93, "Gated fusion" if report["scale_gated_fpn"] else "Addition", 10)
    s.arrow([(22.3, 17.85), (22.3, 20.70)])
    s.text(23.0, 19.25, f"{fusion} channels\nstride 4", 10, align="left")
    s.operation(22.3, 21.3, 4.75, 1.2, f"Conv 3×3, {fusion}→{out}\nBN + SiLU", "green", 11)
    s.text(19.3, 20.30, shape(report["output_shape"]), 10, True)
    s.arrow([(24.68, 21.30), (27.25, 21.30)])

    # Heads are a template, explicitly repeated for each group when selected.
    s.panel(26.4, 16.1, 9.1, 7.75, "Detection heads", "yellow")
    s.text(30.95, 22.83, "Per branch: 3×3 + 3×3 + 1×1", 10)
    branches = [("Heatmap", "K" if report["grouped"] else str(report["num_classes"]), "heat"),
                ("Offset", "2", "offset"), ("Size", "2", "size"), ("Yaw", "2", "yaw")]
    if report["iou"]:
        branches.append(("IoU quality", "1", "iou"))
    ys = [21.90 - i * .95 for i in range(len(branches))]
    bus_x = 27.25
    s.arrow([(bus_x, ys[0]), (bus_x, ys[-1])])
    for cy, (name, channels, fill) in zip(ys, branches):
        s.arrow([(bus_x, cy), (27.85, cy)])
        s.operation(28.65, cy, 1.6, .60, "Conv", "peach", 11)
        s.arrow([(29.45, cy), (30.3, cy)])
        s.tensor(30.6, cy, .20, .60, fill, .20)
        s.text(31.3, cy, f"{name}\n{report['output_shape'][2]}×{report['output_shape'][3]}×{channels}",
               10, align="left")
    if report["grouped"]:
        desc = "; ".join(f"{'+'.join(g['classes'])}: K={len(g['classes'])}" for g in report["head_groups"])
        s.text(30.95, 16.63, "Independent template per group\n" + desc, 9)
    else:
        s.text(30.95, 16.63, "Gaussian center heatmap · shared class set", 10)

    s.panel(26.4, 8.0, 9.1, 7.55,
            "Feature refinement" if enhanced else "Architecture notes", "blue")
    if enhanced:
        s.text(30.95, 14.48, "C3 / stride 4: residual ECA" if report["eca"]
               else "C3 / stride 4: convolutional features", 11, True)
        eca = (f"GAP → Conv1D k={report['eca_kernel']} → sigmoid" if report["eca"]
               else f"{report['stage_depths'][0]} NeXt blocks → C3")
        s.text(30.95, 13.94, eca, 10)
        if report["eca"]:
            s.text(30.95, 13.40, "Y = X + β [ECA(X) − X], β₀ = 0.001", 10)
        if not report["focal"]:
            s.text(30.95, 11.8, "C4 Focal Context: disabled", 11)
            next_inset(s, report)
            gate_inset(s, report)
            return s
        s.text(30.95, 12.56, "C4 / stride 8: Focal Context", 11, True)
        s.text(30.95, 12.00, f"BN → Conv1×1 → Q, S, G  (D={report['focal_bottleneck']})", 10)
        for i, d in enumerate(report["focal_dilations"]):
            cx = 27.95 + 2.9 * i
            s.operation(cx, 11.20, 2.3, .65, f"DW3×3 d={d}", "purple", 9)
            if i < 2:
                s.arrow([(cx+1.15, 11.2), (cx+1.75, 11.2)])
        s.text(30.95, 10.45, "Sequential local levels + global mean", 10)
        s.text(30.95, 9.87, "Softmax(G) → context mix → Conv1×1", 10)
        s.text(30.95, 9.28, "Q ⊙ context → projection + BN", 10)
        s.text(30.95, 8.67, "Y = X + γ · residual, γ₀ = 0.001", 10)
    else:
        lines = ["DW7×7 spatial mixing before expansion", f"Expansion ratio: {report['expansion']:g} · BatchNorm + SiLU",
                 "Stage depths: " + " / ".join(map(str, report['stage_depths'])), "C5: stride 16 → output: stride 4",
                 "Bilinear upsampling + convolution", "Scale gates initialize to additive fusion",
                 "Focal / ECA / RepConv / detail: disabled"]
        for i, line in enumerate(lines):
            s.text(30.95, 14.25 - i*.78, line, 10)

    next_inset(s, report)
    gate_inset(s, report)
    s.text(18, .13, "Shapes use C × H × W (batch omitted).  G = scale-gated fusion.  "
           "Cuboid dimensions are schematic.", 9, color="muted")
    return s


def next_inset(s, report):
    s.panel(.45, .55, 18.0, 6.85, "(A) MobilePixorNeXt block", "blue")
    s.text(9.45, 5.94, "Depthwise spatial mixing → pointwise channel expansion → residual", 10)
    y = 3.93
    nodes = [(2.95, 1.65, "DW 7×7\nC→C", "blue"),
             (5.05, 1.15, "BN", "white"),
             (7.30, 2.30, f"PW 1×1\nC→{report['expansion']:g}C", "peach"),
             (9.88, 1.50, "SiLU", "white"),
             (12.40, 2.30, f"PW 1×1\n{report['expansion']:g}C→C", "peach"),
             (14.62, 1.10, "BN", "white"),
             (16.02, .80, "γ", "purple")]
    s.text(1.13, y, "X", 14, True)
    last = 1.45
    for x, w, label, fill in nodes:
        s.arrow([(last, y), (x-w/2, y)])
        if "PW" in label or "DW" in label:
            s.tensor(x, y, w, 1.18, fill, .24)
            s.text(x, y, label, 10)
        else:
            s.operation(x, y, w, 1.18, label, fill, 11)
        last = x+w/2 + (.10 if "PW" in label or "DW" in label else 0)
    s.arrow([(last, y), (17.12, y)])
    s.circle(17.42, y)
    s.arrow([(17.72, y), (18.10, y)])
    s.text(18.05, 4.7, "Y", 12, True)
    s.arrow([(1.45, y), (1.45, 2.00), (17.42, 2.00), (17.42, y-.30)])
    s.text(9.45, 2.27, "Identity shortcut", 10)
    s.text(9.45, 1.16, "Y = X + γ ⊙ F(X)    ·    γ is learned per channel; initialized to 10⁻⁵", 10)


def gate_inset(s, report):
    s.panel(19.0, .55, 16.5, 6.85, "(B) Scale-gated fusion", "yellow")
    s.text(27.25, 6.07, "L: lateral feature    U: upsampled feature", 10)
    s.text(20.02, 4.60, "L", 13, True)
    s.text(20.02, 2.32, "U", 13, True)
    s.circle(22.25, 4.60)
    s.arrow([(20.35, 4.60), (21.95, 4.60)])
    s.arrow([(20.35, 2.32), (22.25, 2.32), (22.25, 4.30)])
    s.arrow([(22.55, 4.60), (23.20, 4.60)])
    s.operation(24.54, 4.60, 2.68, 1.0, "DWConv 3×3\nzero-init", "green", 10)
    s.arrow([(25.88, 4.60), (26.58, 4.60)])
    s.operation(27.50, 4.60, 1.84, 1.0, "2 · sigmoid", "purple", 10)
    s.arrow([(28.42, 4.60), (30.08, 4.60)])
    s.circle(30.38, 4.60, "×")
    s.arrow([(20.70, 4.60), (20.70, 5.48), (30.38, 5.48), (30.38, 4.90)])
    s.arrow([(30.68, 4.60), (33.10, 4.60), (33.10, 3.55)])
    s.circle(33.10, 3.25)
    s.arrow([(22.25, 2.32), (33.10, 2.32), (33.10, 2.95)])
    s.arrow([(33.40, 3.25), (34.60, 3.25)])
    s.text(34.70, 3.25, "P", 13, True)
    s.text(27.25, 1.61, "P = U + 2σ(DWConv(L + U)) ⊙ L", 12)
    s.text(27.25, 1.00, "At initialization: gate = 1, so P = U + L", 10)


def render(scene, target, dpi, canvas=(36, 26)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
    from matplotlib.colors import to_rgb

    plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    width, height = canvas
    fig, ax = plt.subplots(figsize=(width * .6, height * .6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0,width), ylim=(0,height), aspect="equal")
    ax.axis("off")
    texts = []
    for kind, v in scene.items:
        if kind == "text":
            t = ax.text(v["x"], v["y"], v["label"], fontsize=v["size"],
                        fontweight="bold" if v["bold"] else "normal",
                        color=COLORS[v["color"]], ha=v["align"], va="center",
                        rotation=v.get("rotation", 0), zorder=5)
            texts.append(t)
        elif kind == "box":
            ax.add_patch(FancyBboxPatch((v["x"]-v["w"]/2, v["y"]-v["h"]/2),
                v["w"], v["h"], boxstyle="round,pad=0,rounding_size=0.16",
                facecolor=COLORS[v["fill"]], edgecolor=COLORS["ink"],
                linewidth=1.2 if v["dashed"] else .9,
                linestyle=(0,(5,4)) if v["dashed"] else "solid", zorder=1 if v["dashed"] else 3))
        elif kind == "arrow":
            points = v["points"]
            xs, ys = zip(*points)
            ax.plot(xs[:-1], ys[:-1], color=COLORS[v["color"]], lw=1.15, zorder=2)
            ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                mutation_scale=10, linewidth=1.15, color=COLORS[v["color"]],
                shrinkA=0, shrinkB=0, zorder=2,
                linestyle="dashed" if v["dashed"] else "solid"))
        elif kind == "circle":
            ax.add_patch(Circle((v["x"],v["y"]),v["r"],facecolor="white",
                                edgecolor=COLORS["ink"],lw=1.2,zorder=4))
            ax.text(v["x"],v["y"],v["label"],ha="center",va="center",fontsize=13,zorder=5)
        else:
            x,y,w,h,d = (v[k] for k in ("x","y","w","h","depth"))
            l,r,b,t = x-w/2,x+w/2,y-h/2,y+h/2
            dx,dy = d*.65,d*.48
            color = to_rgb(COLORS[v["fill"]])
            top = tuple(.25+.75*c for c in color)
            side = tuple(.82*c for c in color)
            for points, face in [([(l,t),(l+dx,t+dy),(r+dx,t+dy),(r,t)],top),
                                 ([(r,b),(r+dx,b+dy),(r+dx,t+dy),(r,t)],side),
                                 ([(l,b),(r,b),(r,t),(l,t)],color)]:
                ax.add_patch(Polygon(points,closed=True,facecolor=face,
                                     edgecolor=COLORS["ink"],linewidth=.8,zorder=3))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_bounds = ax.get_window_extent(renderer)
    clipped = [t.get_text() for t in texts if not canvas_bounds.contains(*t.get_window_extent(renderer).p0)
               or not canvas_bounds.contains(*t.get_window_extent(renderer).p1)]
    if clipped:
        raise ValueError(f"Text outside figure canvas: {clipped}")
    overlaps = []
    for i, a in enumerate(texts):
        for b in texts[i+1:]:
            if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
                overlaps.append([a.get_text(),b.get_text()])
    if overlaps:
        raise ValueError(f"Overlapping text labels: {overlaps}")
    for suffix in ("png","svg","pdf"):
        fig.savefig(target.with_suffix("."+suffix), dpi=dpi, facecolor="white")
    plt.close(fig)


def tex_escape(text):
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
                    "#": r"\#", "{": r"\{", "}": r"\}", "×": r"$\times$",
                    "→": r"$\rightarrow$", "↑": r"$\uparrow$", "·": r"$\cdot$",
                    "γ": r"$\gamma$", "β": r"$\beta$", "σ": r"$\sigma$", "λ": r"$\lambda$", "⊙": r"$\odot$",
                    "−": "$-$", "₀": "$_0$", "10⁻⁵": "$10^{-5}$",
                    "β₀": r"$\beta_0$", "γ₀": r"$\gamma_0$"}
    pattern = "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True))
    return re.sub(pattern, lambda match: replacements[match.group()], text).replace("\n", r"\\")


def write_tikz(scene, target, canvas=(36, 26)):
    sys.path.insert(0,str(VENDOR))
    from PlotNeuralNet.PyCore.TikzGen import ToConv

    tex = [r"\documentclass[tikz,border=6pt]{standalone}",
           r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}",
           r"\usepackage{import}", r"\subimport{layers/}{init.tex}",
           r"\usetikzlibrary{arrows.meta,positioning}"]
    for name,color in COLORS.items():
        tex.append(r"\definecolor{"+name+r"}{HTML}{"+color[1:]+"}")
    tex += [r"\begin{document}",r"\begin{tikzpicture}[x=1.524cm,y=1.524cm,z={(0.9906cm,0.73152cm)}]",
            rf"\path[use as bounding box] (0,0) rectangle ({canvas[0]},{canvas[1]});"]
    for i,(kind,v) in enumerate(scene.items):
        if kind=="text":
            anchor={"left":"west","right":"east","center":"center"}[v["align"]]
            weight=r"\bfseries " if v["bold"] else ""
            tex.append(rf"\node[anchor={anchor},align={v['align']},rotate={v.get('rotation', 0)},text={v['color']},font=\fontsize{{{v['size']}}}{{{v['size']*1.2}}}\selectfont {weight}] at ({v['x']},{v['y']}) {{{tex_escape(v['label'])}}};")
        elif kind=="box":
            dash=",dashed" if v["dashed"] else ""
            tex.append(rf"\draw[draw=ink,fill={v['fill']},rounded corners=3pt,line width=.7pt{dash}] ({v['x']-v['w']/2},{v['y']-v['h']/2}) rectangle ({v['x']+v['w']/2},{v['y']+v['h']/2});")
        elif kind=="arrow":
            route=" -- ".join(f"({x},{y})" for x,y in v["points"])
            tex.append(rf"\draw[-{{Stealth[length=4pt]}},draw={v['color']},line width=.7pt] {route};")
        elif kind=="circle":
            tex.append(rf"\draw[fill=white,draw=ink,line width=.8pt] ({v['x']},{v['y']}) circle ({v['r']});")
            tex.append(rf"\node at ({v['x']},{v['y']}) {{{tex_escape(v['label'])}}};")
        else:
            # Genuine PlotNeuralNet primitive, with schematic dimensions and
            # labels kept outside its automatic caption/feature-label slots.
            block=ToConv(f"tensor{i}",sFilter="",nFilter='""',
                         to=f"({v['x']-v['w']/2},{v['y']},0)",
                         width=v["w"]/.2,height=v["h"]/.2,depth=v["depth"]/.2,caption="")
            tex.append(block.replace(r"\ConvColor",v["fill"]).replace("Box={","Box={opacity=1,"))
    tex += [r"\end{tikzpicture}",r"\end{document}"]
    target.with_suffix(".tex").write_text("\n".join(tex)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,action="append",help="Repeat for multiple configurations")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"diagrams/mobilepixornext")
    parser.add_argument("--dpi",type=int,default=300)
    args=parser.parse_args()
    configs=args.config or [ROOT/"configs/config.json",
        ROOT/"configs/experiments/under1m/hist14_local3_focal_eca_grouped_oga_iqa.json"]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    layers=args.output_dir/"layers"
    layers.mkdir(exist_ok=True)
    for file in (VENDOR/"PlotNeuralNet/Layers").iterdir():
        if file.is_file():
            shutil.copy2(file,layers/file.name)
    shutil.copy2(VENDOR/"LICENSE",layers/"LICENSE")
    for config in configs:
        report=inspect_config(config.resolve())
        name="mobilepixornext_"+report["encoding"]
        if report["focal"]:
            name+="_focal"
        if report["eca"]:
            name+="_eca"
        target=args.output_dir/name
        scene=architecture_scene(report)
        render(scene,target,args.dpi)
        write_tikz(scene,target)
        target.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
        print(f"{target}: {report['parameters_including_neck']:,} parameters; "
              f"shape={report['output_shape']}; PNG/SVG/PDF/TikZ/JSON exported")


if __name__=="__main__":
    main()
