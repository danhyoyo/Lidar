#!/usr/bin/env python3
"""Compare the production independent and per-group shared head architectures."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

from draw_mobilepixornext_paper import ROOT, VENDOR, Scene, render, write_tikz


BRANCHES = [("cls", "Heatmap", "heat"), ("offset", "Offset", "offset"),
            ("size", "Size", "size"), ("yaw", "Yaw", "yaw"),
            ("iou", "IoU quality", "iou")]


def audit_heads(config):
    import torch
    from torch import nn
    sys.path.insert(0, str(ROOT/"detector"))
    from core.models.heads.grouped import GroupedHeader
    from core.task_groups import resolve_task_groups

    cfg, data = config["model"], config["data"]
    if (cfg.get("head_mode") != "grouped" or cfg.get("backbone_out_dim") != 32
            or not cfg.get("header_use_bn") or cfg.get("header_act") != "silu"
            or not cfg.get("header_use_iou") or data.get("box_mode") != "bev"
            or cfg.get("cls_encoding") != "gaussian"):
        raise ValueError("Comparison requires grouped Gaussian BEV heads, width 32, BN/SiLU and IoU quality")
    groups = resolve_task_groups(data["head_groups"], data["kitti"]["objects"])
    if [(g.name, g.num_classes) for g in groups] != [("car", 1), ("ped_cyc", 2)]:
        raise ValueError("Comparison expects Car and Pedestrian/Cyclist groups")
    geometry = data["kitti"]["geometry"]
    spatial = [round((geometry[f"{axis}_max"]-geometry[f"{axis}_min"])
                     /geometry[f"{axis}_res"]/data["out_size_factor"])
               for axis in ("y", "x")]
    if spatial != [200, 176]:
        raise ValueError("Comparison expects a 200×176 output grid")

    current = GroupedHeader(groups, 32, cls_encoding="gaussian", use_bn=True,
                            act="silu", use_iou=True, box_mode="bev",
                            header_architecture="independent").eval()
    proposed = GroupedHeader(groups, 32, cls_encoding="gaussian", use_bn=True,
                             act="silu", use_iou=True, box_mode="bev",
                             header_architecture="shared").eval()
    torch.set_num_threads(2)
    torch.manual_seed(config.get("seed", 42))
    feature = torch.randn(1, 32, *spatial)
    with torch.inference_mode():
        predictions = {
            "current": current(feature)["groups"],
            "proposed": proposed(feature)["groups"],
        }
    report = {}
    for kind, model in (("current", current), ("proposed", proposed)):
        shapes = {}
        for group in groups:
            shapes[group.name] = {}
            for name, _, _ in BRANCHES:
                tensor = predictions[kind][group.name][name]
                channels = group.num_classes if name == "cls" else (1 if name == "iou" else 2)
                assert list(tensor.shape) == [1, channels, *spatial]
                assert torch.isfinite(tensor).all()
                shapes[group.name][name] = list(tensor.shape)
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        report[kind] = dict(
            parameters=sum(p.numel() for p in model.parameters()),
            conv3x3_count=sum(m.kernel_size == (3, 3) for m in convs),
            conv1x1_count=sum(m.kernel_size == (1, 1) for m in convs),
            convolution_mac=sum(spatial[0]*spatial[1]*m.weight.numel() for m in convs),
            output_shapes=shapes,
        )
    assert report["current"]["parameters"] == 186161
    assert report["proposed"]["parameters"] == 111921
    assert report["current"]["conv3x3_count"] == 20
    assert report["proposed"]["conv3x3_count"] == 12
    assert report["current"]["output_shapes"] == report["proposed"]["output_shapes"]
    return report


def variant_scene(kind, audit):
    shared = kind == "proposed"
    s = Scene()
    s.text(10.5, 22.35, "B · Proposed head" if shared else "A · Current code", 18, True)
    s.text(10.5, 21.65, "One shared Conv3×3 per group + one private Conv3×3 per branch"
           if shared else "Two private Conv3×3 layers in every prediction branch", 9)
    s.tensor(10.5, 20.2, 6.2, 1.05, "offset", .3)
    s.text(10.5, 20.2, "Backbone output\n32 × 200 × 176", 12)
    # Both groups receive the entire feature tensor. The input bus stays outside
    # their panels so it cannot be mistaken for a group-specific shared tower.
    s.arrow([(10.5, 19.675), (10.5, 19.2), (.25, 19.2), (.25, 7.1)])
    for group, k, bottom, first_y in (("Group 1: Car", 1, 11.8, 17.4),
                                     ("Group 2: Pedestrian / Cyclist", 2, 3.7, 9.3)):
        s.panel(.65, bottom, 19.95, 7.0, group, "yellow" if k == 2 else "panel")
        middle = first_y-2.2
        if shared:
            s.arrow([(.25, middle), (1.55, middle)])
            s.operation(3.15, middle, 3.2, 2.6,
                        "Shared\nConv3×3\n32→32\nBN + SiLU", "green", 11)
            s.arrow([(4.75, middle), (5.6, middle)])
            bus = 5.6
        else:
            bus = 2.55
            s.arrow([(.25, middle), (bus, middle)])
        s.arrow([(bus, middle), (bus, first_y)])
        s.arrow([(bus, first_y), (bus, first_y-4.4)])
        for i, (name, label, fill) in enumerate(BRANCHES):
            y = first_y-1.1*i
            channels = k if name == "cls" else (1 if name == "iou" else 2)
            if shared:
                s.arrow([(bus, y), (6.65, y)])
                s.operation(8.3, y, 3.3, .75, "Conv3×3, 32→32\nBN + SiLU", "peach", 9)
                last = 9.95
            else:
                s.arrow([(bus, y), (3.45, y)])
                s.operation(5.1, y, 3.3, .75, "Conv3×3, 32→32\nBN + SiLU", "peach", 9)
                s.arrow([(6.75, y), (7.65, y)])
                s.operation(9.3, y, 3.3, .75, "Conv3×3, 32→32\nBN + SiLU", "peach", 9)
                last = 10.95
            s.arrow([(last, y), (11.8, y)])
            s.operation(13.25, y, 2.9, .75, f"Conv1×1\n32→{channels}", "purple", 9)
            s.arrow([(14.7, y), (15.55, y)])
            s.operation(17.7, y, 4.3, .75, f"{label}\n{channels} × 200 × 176", fill, 10)
        s.text(10.5, bottom+.55, "Shared Conv has separate weights in each group; prediction branches remain independent"
               if shared else "Every branch has its own Conv3×3 / BN / SiLU weights", 8)
    stats = audit[kind]
    s.text(10.5, 2.55, f"{stats['conv3x3_count']} Conv3×3   ·   {stats['conv1x1_count']} Conv1×1   ·   "
           f"{stats['parameters']:,} head parameters", 11, True)
    s.text(10.5, 1.77, f"Convolution cost: {stats['convolution_mac']/1e9:.2f} G MAC / sample", 10)
    s.text(10.5, .94, "Implemented: header_architecture=shared; training results pending"
           if shared else "Topology taken directly from GroupedHeader and Header / Head", 9, color="muted")
    return s


def translated(scene, offset):
    out = Scene()
    for kind, values in scene.items:
        v = copy.deepcopy(values)
        if kind == "arrow":
            v["points"] = [(x+offset, y) for x, y in v["points"]]
        else:
            v["x"] += offset
        out.items.append((kind, v))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"diagrams/mobilepixornext/hist14_focal_simam_f32_s42/config.resolved.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"diagrams/mobilepixornext/head_current_vs_proposed")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    audit = audit_heads(config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(VENDOR/"PlotNeuralNet/Layers", output/"layers", dirs_exist_ok=True)
    shutil.copy2(VENDOR/"LICENSE", output/"layers/LICENSE")
    views = {kind: variant_scene(kind, audit) for kind in ("current", "proposed")}
    comparison = Scene()
    comparison.text(21, 23.22, "Grouped detection heads · current vs proposed", 22, True)
    comparison.items += views["current"].items + translated(views["proposed"], 21).items
    comparison.text(21, .25, "Shapes: C × H × W. Both groups receive all 32 channels. "
                    "MAC counts include convolutions only; backbone is excluded.", 9, color="muted")
    for name, scene, canvas in (("current_heads", views["current"], (21, 22.9)),
                                ("proposed_heads", views["proposed"], (21, 22.9)),
                                ("comparison", comparison, (42, 23.8))):
        render(scene, output/name, args.dpi, canvas)
        write_tikz(scene, output/name, canvas)
        print(f"Exported {name}: PNG/SVG/PDF/TikZ", flush=True)
    report = dict(
        source_config=str(args.config.resolve()),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        input_shape=[1, 32, 200, 176], audit=audit,
        proposed_status="Implemented in production Header/GroupedHeader; opt in with model.header_architecture=shared",
        verification="Fresh CPU eval forwards of production independent and shared GroupedHeader; exact matching shapes and finite outputs",
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__), ROOT/"detector/core/models/heads/cnn.py",
                                 ROOT/"detector/core/models/heads/grouped.py")},
    )
    (output/"comparison.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
