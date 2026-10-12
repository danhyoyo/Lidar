#!/usr/bin/env python3
"""One paper-style BEV → backbone → heads figure for the artifact preset.

Rows of vertical encoder blocks and an ascending SG-FPN follow other_paper.png.
The scene remains vector/editable; tensor cuboids use PlotNeuralNet in TikZ.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from draw_artifact_architecture import audit_model, write_gallery
from draw_mobilepixornext_paper import (
    ROOT, VENDOR, Scene, inspect_config, render, write_tikz,
)


def vertical(s, x, y, label, fill="blue", w=1.05, h=4.0, size=10):
    s.box(x, y, w, h, fill)
    s.text(x, y, label, size, rotation=90)


def next_detail(s):
    s.panel(1.0, 1.15, 12.4, 5.65, "(A) NeXt block", "blue")
    s.text(7.2, 5.62, "C = 48 / 96 / 128    ·    expansion 2.5", 9)
    y = 4.1
    nodes = [(2.5, 1.35, "DW7×7\nC→C", "blue"),
             (4.02, .8, "BN", "white"),
             (5.56, 1.65, "PW1×1\nC→2.5C", "peach"),
             (7.16, 1.0, "SiLU", "white"),
             (8.88, 1.65, "PW1×1\n2.5C→C", "peach"),
             (10.4, .8, "BN", "white"), (11.6, .65, "γ", "purple")]
    s.text(1.35, y, "X", 11, True)
    last = 1.6
    for x, w, label, fill in nodes:
        s.arrow([(last, y), (x-w/2, y)])
        if "PW" in label or "DW" in label:
            s.tensor(x, y, w, 1.1, fill, .22)
            s.text(x, y, label, 8)
            last = x+w/2+.15
        else:
            s.operation(x, y, w, 1.1, label, fill, 8.5)
            last = x+w/2
    s.arrow([(last, y), (12.25, y)])
    s.circle(12.55, y)
    s.arrow([(12.85, y), (13.15, y)])
    s.text(13.1, 4.9, "Y", 10, True)
    s.arrow([(1.6, y), (1.6, 2.62), (12.55, 2.62), (12.55, y-.3)])
    s.text(7.2, 2.93, "Identity shortcut", 9)
    s.text(7.2, 1.76, "Y = X + γ ⊙ F(X)    ·    γ₀ = 10⁻⁵", 9)


def simam_detail(s, report):
    s.panel(1.0, 7.25, 6.4, 6.4, "(B) Residual SimAM", "purple")
    s.text(4.2, 12.53, "λ = 0.0001    ·    β₀ = 0.001", 9)
    s.operation(4.2, 11.55, 5.4, .95,
                "Squared deviation d; spatial variance v\nper batch / channel, FP32", "blue", 8)
    s.arrow([(4.2, 11.075), (4.2, 10.675)])
    s.operation(4.2, 10.2, 5.4, .95, "e = d / [4(v + λ)] + 0.5", "yellow", 9)
    s.arrow([(4.2, 9.725), (4.2, 9.325)])
    s.operation(4.2, 8.85, 5.4, .95, "SimAM(X) = X ⊙ sigmoid(e)", "green", 9)
    s.text(4.2, 7.87, "Y = X + β [SimAM(X) − X]", 9)


def grouped_heads(s):
    """Draw both independent Header modules fed by the same backbone features."""
    groups = [("Group 1: Car", 1, 10.65, 18.15),
              ("Group 2: Pedestrian / Cyclist", 2, .7, 8.2)]
    s.arrow([(24.475, 17), (25.75, 17)])
    s.arrow([(25.75, 17), (25.75, 6.5)])
    s.text(30.4, 10.37, "Shared backbone features: 32 × 200 × 176", 9, True)
    for title, k, bottom, top_y in groups:
        s.panel(25.25, bottom, 10.35, 9.75 if k == 1 else 9.4,
                title, "white" if k == 1 else "yellow")
        s.text(30.425, bottom+8.57, f"K={k}   ·   five independent prediction branches", 9)
        inlet_y = 17 if k == 1 else 6.5
        s.arrow([(25.75, inlet_y), (27.2, inlet_y)])
        s.arrow([(27.2, inlet_y), (27.2, top_y)])
        s.arrow([(27.2, top_y), (27.2, top_y-5.4)])
        branches = [("Heatmap", k, "heat"), ("Offset", 2, "offset"),
                    ("Size", 2, "size"), ("Yaw", 2, "yaw"),
                    ("IoU quality", 1, "iou")]
        for i, (name, channels, fill) in enumerate(branches):
            cy = top_y - 1.35*i
            s.arrow([(27.2, cy), (27.95, cy)])
            s.tensor(28.6, cy, 1.3, .65, "peach", .22)
            s.text(28.6, cy, "Conv", 10)
            s.arrow([(29.393, cy), (30.04, cy)])
            s.tensor(30.15, cy, .22, .9, fill, .22)
            s.text(30.92, cy, f"{name}\n{channels} × 200 × 176", 10, align="left")
        s.text(30.425, bottom+.62, "Per branch: Conv3×3 → Conv3×3 → Conv1×1", 8.5)


def make_scene(report):
    s = Scene()
    simam = report.get("simam", False)
    suffix = " + SimAM" if simam else ""
    s.text(18, 21.52, "MobilePixorNeXt" + suffix + " · BEV to grouped detection heads", 20, True)
    s.panel(.45, .7, 24.25, 19.7, "Backbone", "panel")

    # Row 1: dense BEV, two ordinary stem convolutions, C3 stage.
    y = 17.0
    s.tensor(1.55, y, .65, 3.1, "offset", .48)
    s.text(1.65, 14.55, "hist14 BEV\n14 × 800 × 704", 9, True)
    s.arrow([(2.15, y), (2.8, y)])
    vertical(s, 3.25, y, "Conv3×3, 14→32, s=2\nBN + SiLU", "peach", .9)
    s.arrow([(3.7, y), (3.9, y)])
    vertical(s, 4.35, y, "Conv3×3, 32→32, s=1\nBN + SiLU", "peach", .9)
    s.text(3.8, 14.55, "Stem /2\n32 × 400 × 352", 9)
    s.arrow([(4.8, y), (5.475, y)])
    vertical(s, 6.0, y, "Downsample 2\nConv3×3, 32→48, s=2\nBN + SiLU")
    last = 6.525
    for i, x in enumerate((7.2, 8.4, 9.6)):
        s.arrow([(last, y), (x-.525, y)])
        vertical(s, x, y, f"NeXt block {i+1}\n48→120→48")
        last = x+.525
    if simam:
        s.arrow([(10.125, y), (10.4, y)])
        vertical(s, 11.05, y, "Residual SimAM\n48 × 200 × 176\nλ=0.0001; β₀=0.001", "purple", 1.3)
    s.text(8.8 if simam else 8.4, 14.55,
           "Stage 2 / C3" + (" + SimAM" if simam else "") + "\n48 × 200 × 176   /4", 10, True)

    # C3 continues down into stage 3 as well as across to L3.
    branch_x = 11.85 if simam else 10.25
    s.arrow([(branch_x, y), (branch_x, 13.35), (8.25, 13.35), (8.25, 12.7)])
    y = 10.7
    vertical(s, 8.25, y, "Downsample 3\nConv3×3, 48→96, s=2\nBN + SiLU", w=1.1)
    last = 8.8
    for i, x in enumerate((9.45, 10.65, 11.85, 13.05)):
        s.arrow([(last, y), (x-.525, y)])
        vertical(s, x, y, f"NeXt block {i+1}\n96→240→96")
        last = x+.525
    s.arrow([(last, y), (13.85, y)])
    vertical(s, 14.5, y, "Focal Context\nD=64; dilation 1,2,3\nγ₀=0.001", "purple", 1.3)
    s.text(11.4, 8.15, "Stage 3 / C4\n96 × 100 × 88   /8", 10, True)
    s.arrow([(15.65, y), (15.65, 7.5), (14.5, 7.5), (14.5, 6.8)])

    # Final encoder row; the lower-left area stays available for the block inset.
    y = 4.8
    vertical(s, 14.5, y, "Downsample 4\nConv3×3, 96→128, s=2\nBN + SiLU", w=1.1)
    last = 15.05
    for i, x in enumerate((15.75, 17.0)):
        s.arrow([(last, y), (x-.525, y)])
        vertical(s, x, y, f"NeXt block {i+1}\n128→320→128")
        last = x+.525
    s.text(15.75, 2.05, "Stage 4 / C5\n128 × 50 × 44   /16", 10, True)

    # The neck matches the real forward: lateral 1×1, bilinear upsample,
    # convolution, then gated fusion. G4 is depthwise; U3 uses a regular conv.
    for row_y, input_c, c, level, start in (
            (17.0, 48, 32, 3, 11.7 if simam else 10.125), (10.7, 96, 48, 4, 15.15),
            (4.8, 128, 48, 5, 17.525)):
        s.arrow([(start, row_y), (18.75, row_y)])
        vertical(s, 19.25, row_y, f"L{level}: Conv1×1\n{input_c}→{c}", "green", 1.0, 2.9, 11)
        s.arrow([(19.75, row_y), (22, row_y)])
        if level == 5:
            s.tensor(22.3, row_y, .55, 1.1, "green", .3)
        else:
            s.circle(22.3, row_y, "G")
            s.text(21.65, row_y+.75, f"P{level}", 10, True)
    for low, middle, high, label in (
        (4.8, 7.65, 10.7, "Bilinear ↑2\nDWConv3×3, 48→48\nBN + SiLU"),
        (10.7, 13.8, 17.0, "Bilinear ↑2\nConv3×3, 48→32\nBN + SiLU")):
        s.arrow([(22.3, low+(.65 if low == 4.8 else .3)), (22.3, middle-.85)])
        s.operation(22.3, middle, 3.5, 1.7, label, "yellow", 9)
        s.arrow([(22.3, middle+.85), (22.3, high-.3)])
    s.arrow([(22.6, 17), (23.425, 17)])
    vertical(s, 23.95, 17, "Output Conv3×3, 32→32\nBN + SiLU", "green", 1.05, 3.7, 10)

    grouped_heads(s)
    next_detail(s)
    if simam:
        simam_detail(s, report)
    s.text(15.75, 1.24, "G: P = U + 2σ(DWConv(L + U)) ⊙ L", 8)
    s.text(18, .25, "Shapes: C × H × W (batch omitted).  G: scale-gated fusion.  "
           "Cuboids are schematic; head outputs are raw predictions.", 9, color="muted")
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"diagrams/mobilepixornext/hist14_focal_none_f32_s42/config.resolved.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--simam", action="store_true", help="Derive a SimAM-at-C3 variant of the artifact config")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    source = args.config.resolve()
    source_bytes = source.read_bytes()
    config = json.loads(source_bytes)
    effective_simam = args.simam or config["model"].get("local_attention", "none") == "simam"
    output = (args.output_dir or ROOT/"diagrams/mobilepixornext"/
              ("hist14_focal_simam_f32_s42" if effective_simam else "hist14_focal_none_f32_s42")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.simam:
        config = copy.deepcopy(config)
        config["model"].update(local_attention="simam", local_attention_simam_lambda=.0001,
                               local_attention_layer_scale_init=.001)
        variant = output/"config.resolved.json"
        variant.write_text(json.dumps(config, indent=2)+"\n")
        config_path = variant
    else:
        config_path = source
    report = inspect_config(config_path, allow_simam=True)
    local_attention = config["model"].get("local_attention", "none")
    report["audit"] = audit_model(config, report, local_attention=local_attention)
    scene = make_scene(report)
    shutil.copytree(VENDOR/"PlotNeuralNet/Layers", output/"layers", dirs_exist_ok=True)
    shutil.copy2(VENDOR/"LICENSE", output/"layers/LICENSE")
    target = output/"architecture_bev_backbone_heads"
    render(scene, target, args.dpi, canvas=(36, 22.1))
    write_tikz(scene, target, canvas=(36, 22.1))
    report["figure"] = target.name
    report["png_dpi"] = args.dpi
    report["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["layout"] = "BEV → descending encoder stages → ascending SG-FPN → two independent grouped heads"
    report["source_artifact_config"] = str(source)
    report["source_artifact_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    report["config_overrides"] = ({k: config["model"][k] for k in
                                   ("local_attention", "local_attention_simam_lambda", "local_attention_layer_scale_init")}
                                  if args.simam else {})
    target.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    atlas_report = output/"architecture.json"
    if atlas_report.exists():
        names = json.loads(atlas_report.read_text()).get("figures", [])
        write_gallery(output, report, [target.name, *names])
    else:
        write_gallery(output, report, [target.name])
    print(f"{target}: PNG/SVG/PDF/PlotNeuralNet TikZ/JSON; full detector forward verified", flush=True)


if __name__ == "__main__":
    main()
