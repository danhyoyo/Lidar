#!/usr/bin/env python3
"""Draw the MobilePixorNeXt backbone architecture with a hybrid 3D Isometric + 2D Flow layout.

- Top Section: 3D Isometric Tensor Feature Maps for the Multi-Scale Encoder (PlotNeuralNet style)
- Middle Section: Detailed 2D Technical Flow for the Top-Down Scale-Gated FPN Neck
- Bottom Section: Architectural Insets detailing NeXt Block, Scale-Gated Fusion, and ECA/FocalContext

Produces:
1. LaTeX/TikZ code (.tex) adhering to the PlotNeuralNet standard for paper inclusion/Overleaf.
2. Publication-quality vector PDF, SVG, and 300 DPI PNG previews rendered directly via Matplotlib.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party" / "PlotNeuralNet"
DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "under1m" / "hist14_local3_focal_eca_grouped_oga_iqa.json"

# Academic & Publication Palette (IEEE / CVPR / Nature style)
PALETTE = {
    "bg_main": "#ffffff",
    "panel_bg": "#f8fafc",
    "panel_border": "#e2e8f0",
    "text_dark": "#0f172a",
    "text_muted": "#475569",
    "text_light": "#64748b",
    "input": "#b8c9dc",          # Soft steel blue (BEV Input)
    "stem": "#f3ba6f",           # Warm amber / gold (Spatial Stem)
    "stage2": "#66b7ab",         # Mint teal (Stage 2)
    "stage3": "#47a396",         # Deep ocean teal (Stage 3)
    "stage4": "#2d8a7c",         # Forest teal (Stage 4)
    "attention": "#9c7ec5",      # Iris purple (ECA / FocalContext bands)
    "lateral": "#e09f58",        # Ochre / amber for lateral 1x1 convs
    "upsample": "#5c92c4",       # Slate blue for upsample DW/Conv
    "fusion": "#7bbd6c",         # Apple green for Scale-Gated Fusion
    "fusion_bg": "#f0f8ed",      # Soft green tint
    "output": "#ea7568",         # Coral / salmon for backbone output
    "output_bg": "#fdf1ef",      # Soft coral tint
    "edge_flow": "#334155",      # Slate dark for main arrows
    "edge_lateral": "#d97706",   # Amber for lateral skip connections
    "card_bg": "#ffffff",
    "card_border": "#cbd5e1",
}


def inspect_backbone(config_path):
    """Resolve the selected config through production code and check tensor shapes."""
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(ROOT / "detector"))
    from core.bev_encoding import resolve_input_channels
    from core.models.backbones.registry import build_backbone
    from core.models.backbones.focal_context import FocalContext
    from core.models.backbones.light_attention import ECACore, ResidualLightAttention

    config = json.loads(config_path.read_text())
    cfg = config["model"]
    if cfg["backbone"].lower() != "mobilepixornext":
        raise ValueError("This diagram requires the MobilePixorNeXt backbone")
    geometry = config["data"]["kitti"]["geometry"]
    encoding = config["data"].get("bev_encoding")
    channels = resolve_input_channels(encoding, geometry)
    height = round((geometry["y_max"] - geometry["y_min"]) / geometry["y_res"])
    width = round((geometry["x_max"] - geometry["x_min"]) / geometry["x_res"])
    if height % 16 or width % 16:
        raise ValueError("The diagram requires a BEV grid divisible by 16")
    backbone = build_backbone("mobilepixornext", dict(cfg, geometry=geometry), channels).eval()
    local = backbone.c3_light_attention
    context = backbone.c4_context
    has_eca = isinstance(local, ResidualLightAttention) and isinstance(local.core, ECACore)
    has_focal = isinstance(context, FocalContext)
    if (backbone.rc_neck is not None or backbone.use_reparam or backbone.detail_path
            or not isinstance(context, (nn.Identity, FocalContext))
            or not (isinstance(local, nn.Identity) or has_eca)):
        raise ValueError("This drawing covers the standard backbone; RC neck, RepConv, "
                         "detail path and local attention other than ECA need separate drawings")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    shapes = {}
    names = ("stem", "down2", "stage2", "c3_light_attention", "down3", "stage3", "c4_attention", "c4_context",
             "down4", "stage4", "lat_c5", "lat_c4", "refine_u4", "lat_c3",
             "proj_u3", "out_conv")

    def capture(name):
        def hook(module, inputs, output):
            shapes[name] = list(output.shape)
        return hook

    hooks = [getattr(backbone, name).register_forward_hook(capture(name)) for name in names]
    try:
        with torch.inference_mode():
            output = backbone(torch.zeros(1, channels, height, width))
    finally:
        for hook in hooks:
            hook.remove()
    expected_channels = (32, 48, 48, 48, 96, 96, 96, 96, 128, 128, 48, 48, 48,
                         backbone.neck_fusion_channels, backbone.neck_fusion_channels,
                         backbone.backbone_out_dim)
    strides = (2, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 8, 8, 4, 4, 4)
    for name, out_channels, stride in zip(names, expected_channels, strides):
        expected = [1, out_channels, height // stride, width // stride]
        if shapes[name] != expected:
            raise ValueError(f"{name}: forward returned {shapes[name]}, expected {expected}")
    if list(output.shape) != shapes["out_conv"]:
        raise ValueError("Backbone output must match the final projection")

    attention = backbone.c4_attention
    block = backbone.stage2[0]
    report = {
        "config": str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "backbone": type(backbone).__name__,
        "encoding": (encoding or {}).get("name", "binary_slices"),
        "input_shape": [1, channels, height, width],
        "output_shape": list(output.shape),
        "stage_depths": list(backbone.stage_depths),
        "expansion": block.pw_expand.out_channels / block.channels,
        "attention": backbone.c4_attention_name,
        "attention_scales": list(getattr(attention, "scales", ())),
        "head_dim": getattr(attention, "head_dim", None),
        "qk_norm": getattr(attention, "qk_norm_name", None),
        "local_attention": "eca" if has_eca else "none",
        "eca_kernel_size": local.core.kernel_size if has_eca else None,
        "local_attention_layer_scale_init": cfg.get("local_attention_layer_scale_init", 0.001) if has_eca else None,
        "c4_context": "focal" if has_focal else "none",
        "focal_bottleneck": context.bottleneck if has_focal else None,
        "focal_dilations": list(context.dilations) if has_focal else [],
        "focal_layer_scale_init": context.layer_scale_init if has_focal else None,
        "scale_gated_fpn": backbone.scale_gated_fpn,
        "fusion_channels": backbone.neck_fusion_channels,
        "parameters": sum(p.numel() for p in backbone.parameters()),
        "shapes": shapes,
        "verified": "CPU eval forward pass, batch=1, production registry and BEV channel resolver",
    }
    return report


def shape_str(shape):
    """Format shape as C x H x W string."""
    return f"{shape[1]} × {shape[2]} × {shape[3]}"


def render_preview(report, target_path):
    """Render high-resolution hybrid 3D + 2D diagram using Matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(26, 17), dpi=300)
    fig.patch.set_facecolor(PALETTE["bg_main"])
    ax.set_facecolor(PALETTE["bg_main"])
    ax.set_xlim(-0.5, 25.5)
    ax.set_ylim(-9.4, 7.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    shapes = report["shapes"]
    depths = report["stage_depths"]
    eca = report["local_attention"] == "eca"
    focal = report["c4_context"] == "focal"
    gated = report["scale_gated_fpn"]
    fusion_ch = report["fusion_channels"]
    out_ch = report["output_shape"][1]

    # -------------------------------------------------------------------------
    # Helper: Draw Isometric 3D Cuboid
    # -------------------------------------------------------------------------
    def draw_cuboid(x, y, w, h, d, face_color, edge_color=PALETTE["text_dark"],
                    slices=1, band_color=None, band_ratio=0.28):
        """Draw an isometric cuboid with shaded top/right faces, block slices, and optional band."""
        dx, dy = d * 0.40, d * 0.30
        rgb = to_rgb(face_color)
        top_color = tuple(min(1.0, 0.28 + 0.72 * c) for c in rgb)
        right_color = tuple(0.78 * c for c in rgb)

        left = x - w / 2.0
        right = x + w / 2.0
        bottom = y - h / 2.0
        top = y + h / 2.0

        # Top Face
        top_pts = [(left, top), (left + dx, top + dy), (right + dx, top + dy), (right, top)]
        ax.add_patch(Polygon(top_pts, closed=True, facecolor=top_color, edgecolor=edge_color,
                             linewidth=0.75, zorder=3))

        # Right Face
        right_pts = [(right, bottom), (right + dx, bottom + dy), (right + dx, top + dy), (right, top)]
        ax.add_patch(Polygon(right_pts, closed=True, facecolor=right_color, edgecolor=edge_color,
                             linewidth=0.75, zorder=3))

        # Front Face
        front_pts = [(left, bottom), (right, bottom), (right, top), (left, top)]
        ax.add_patch(Polygon(front_pts, closed=True, facecolor=face_color, edgecolor=edge_color,
                             linewidth=0.85, zorder=3))

        # Internal block slices (representation of repetitive blocks)
        if slices > 1:
            for s in range(1, slices):
                sx = left + (w * s) / slices
                # Front vertical line
                ax.plot([sx, sx], [bottom, top], color=edge_color, alpha=0.35, linewidth=0.6, zorder=4)
                # Top skewed line
                ax.plot([sx, sx + dx], [top, top + dy], color=edge_color, alpha=0.35, linewidth=0.6, zorder=4)

        # Right Attention Band (ECA / FocalContext)
        if band_color:
            band_w = w * band_ratio
            b_left = right - band_w
            b_rgb = to_rgb(band_color)
            b_top_color = tuple(min(1.0, 0.25 + 0.75 * c) for c in b_rgb)
            b_right_color = tuple(0.78 * c for c in b_rgb)

            # Band Front
            ax.add_patch(Polygon([(b_left, bottom), (right, bottom), (right, top), (b_left, top)],
                                 closed=True, facecolor=band_color, edgecolor=edge_color,
                                 linewidth=0.7, zorder=5))
            # Band Top
            ax.add_patch(Polygon([(b_left, top), (b_left + dx, top + dy), (right + dx, top + dy), (right, top)],
                                 closed=True, facecolor=b_top_color, edgecolor=edge_color,
                                 linewidth=0.6, zorder=5))
            # Band Right
            ax.add_patch(Polygon([(right, bottom), (right + dx, bottom + dy), (right + dx, top + dy), (right, top)],
                                 closed=True, facecolor=b_right_color, edgecolor=edge_color,
                                 linewidth=0.6, zorder=5))

    # -------------------------------------------------------------------------
    # Helper: Text & Badges
    # -------------------------------------------------------------------------
    def text(x, y, s, size=10, weight="normal", color=PALETTE["text_dark"],
             ha="center", va="center", **kwargs):
        return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                       ha=ha, va=va, **kwargs)

    def draw_badge(x, y, label, bg="#e2e8f0", fg=PALETTE["text_dark"], pad=(0.5, 0.25), size=9.5, weight="bold"):
        ax.add_patch(FancyBboxPatch((x - pad[0], y - pad[1]), pad[0] * 2, pad[1] * 2,
                                    boxstyle="round,pad=0.08", facecolor=bg, edgecolor="#cbd5e1",
                                    linewidth=0.8, zorder=6))
        text(x, y, label, size=size, weight=weight, color=fg, zorder=7)

    # =========================================================================
    # 1. HEADER SECTION
    # =========================================================================
    text(12.5, 7.15, "MobilePixorNeXt Backbone Architecture", size=23, weight="bold",
         color=PALETTE["text_dark"])
    sub_title = (f"Lightweight 3D LiDAR Object Detection  •  Input: {report['encoding']} (14 ch)  •  "
                 f"{report['parameters']:,} Backbone Parameters  •  Stride 4 Multi-Scale Output")
    text(12.5, 6.62, sub_title, size=11.5, weight="normal", color=PALETTE["text_muted"])

    # Info pills
    pills = [
        ("Encoder: BEVNeXt Blocks (DW 7×7)", PALETTE["stage2"]),
        ("Local Attention: ECA (k=3, β=1e-3)", PALETTE["attention"]),
        ("Context: FocalContext (D=64, d={1,2,3})", PALETTE["attention"]),
        ("Neck: Scale-Gated FPN (48 → 24 → 32)", PALETTE["fusion"]),
    ]
    pill_xs = [4.2, 9.7, 15.3, 20.8]
    for px, (label, col) in zip(pill_xs, pills):
        ax.add_patch(FancyBboxPatch((px - 2.5, 6.05), 5.0, 0.42, boxstyle="round,pad=0.06",
                                    facecolor="#f1f5f9", edgecolor=col, linewidth=1.2, zorder=2))
        text(px, 6.26, label, size=9.2, weight="bold", color=PALETTE["text_dark"], zorder=3)

    # =========================================================================
    # 2. PANEL 1: MULTI-SCALE 3D ENCODER
    # =========================================================================
    # Background panel
    ax.add_patch(FancyBboxPatch((0.2, 1.8), 24.6, 3.95, boxstyle="round,pad=0.2",
                                facecolor=PALETTE["panel_bg"], edgecolor=PALETTE["panel_border"],
                                linewidth=1.2, zorder=1))
    text(0.65, 5.5, "1. BOTTOM-UP MULTI-SCALE 3D ENCODER", size=11, weight="bold",
         color=PALETTE["text_dark"], ha="left")
    text(24.3, 5.5, "Geometric Feature Extraction & Progressive Downsampling (Stride 1 → 16)",
         size=9.5, weight="normal", color=PALETTE["text_muted"], ha="right")
    ax.plot([0.5, 24.5], [5.3, 5.3], color="#e2e8f0", linewidth=1.0, zorder=2)

    # Positions of 3D Cuboids along X axis
    # BEV: x=1.8 | Stem: x=5.6 | Stage2: x=10.2 | Stage3: x=15.2 | Stage4: x=20.5
    enc_y = 3.65

    # A. BEV Input
    draw_cuboid(1.8, enc_y, w=0.55, h=2.5, d=0.75, face_color=PALETTE["input"], slices=1)
    text(1.8, 2.22, "BEV Input", size=11, weight="bold")
    text(1.8, 1.95, "Hist14 Slices", size=9.2, color=PALETTE["text_muted"])
    draw_badge(1.8, 1.62, shape_str(report["input_shape"]), bg="#e2e8f0", pad=(0.75, 0.18), size=8.5)

    # Transition Arrow: BEV -> Stem
    ax.annotate("", xy=(4.65, enc_y), xytext=(2.45, enc_y),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=1.6))
    text(3.55, enc_y + 0.35, "Downsample", size=8.8, weight="bold", color=PALETTE["text_dark"])
    text(3.55, enc_y + 0.12, "Conv 3×3, s=2", size=8.2, color=PALETTE["text_muted"])

    # B. Spatial Stem
    draw_cuboid(5.6, enc_y, w=0.85, h=2.2, d=0.75, face_color=PALETTE["stem"], slices=2)
    text(5.6, 2.22, "Fast Spatial Stem", size=11, weight="bold")
    text(5.6, 1.95, "2× Conv 3×3 + BN + SiLU", size=9.2, color=PALETTE["text_muted"])
    draw_badge(5.6, 1.62, f"{shape_str(shapes['stem'])}  (/2)", bg="#fef3c7", fg="#92400e", pad=(0.85, 0.18), size=8.5)

    # Transition Arrow: Stem -> Stage 2
    ax.annotate("", xy=(8.7, enc_y), xytext=(6.5, enc_y),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=1.6))
    text(7.6, enc_y + 0.35, "Downsample", size=8.8, weight="bold", color=PALETTE["text_dark"])
    text(7.6, enc_y + 0.12, "Conv 3×3, s=2", size=8.2, color=PALETTE["text_muted"])

    # C. Stage 2 (with ECA Band)
    draw_cuboid(10.2, enc_y, w=1.5, h=1.85, d=0.75, face_color=PALETTE["stage2"],
                slices=depths[0], band_color=PALETTE["attention"] if eca else None)
    text(10.2, 2.22, "Stage 2 (C2)", size=11, weight="bold")
    text(10.2, 1.95, f"{depths[0]}× NeXt Blocks" + (" + ECA Attention" if eca else ""), size=9.2, color=PALETTE["text_muted"])
    draw_badge(10.2, 1.62, f"{shape_str(shapes['c3_light_attention'])}  (/4)", bg="#ccfbf1", fg="#0f766e", pad=(0.85, 0.18), size=8.5)

    # Transition Arrow: Stage 2 -> Stage 3
    ax.annotate("", xy=(13.3, enc_y), xytext=(11.5, enc_y),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=1.6))
    text(12.4, enc_y + 0.35, "Downsample", size=8.8, weight="bold", color=PALETTE["text_dark"])
    text(12.4, enc_y + 0.12, "Conv 3×3, s=2", size=8.2, color=PALETTE["text_muted"])

    # D. Stage 3 (with FocalContext Band)
    draw_cuboid(15.2, enc_y, w=1.9, h=1.45, d=0.75, face_color=PALETTE["stage3"],
                slices=depths[1], band_color=PALETTE["attention"] if focal else None)
    text(15.2, 2.22, "Stage 3 (C3 / C4)", size=11, weight="bold")
    text(15.2, 1.95, f"{depths[1]}× NeXt Blocks" + (" + FocalContext" if focal else ""), size=9.2, color=PALETTE["text_muted"])
    draw_badge(15.2, 1.62, f"{shape_str(shapes['c4_context'])}  (/8)", bg="#e0e7ff", fg="#3730a3", pad=(0.85, 0.18), size=8.5)

    # Transition Arrow: Stage 3 -> Stage 4
    ax.annotate("", xy=(18.8, enc_y), xytext=(16.8, enc_y),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=1.6))
    text(17.8, enc_y + 0.35, "Downsample", size=8.8, weight="bold", color=PALETTE["text_dark"])
    text(17.8, enc_y + 0.12, "Conv 3×3, s=2", size=8.2, color=PALETTE["text_muted"])

    # E. Stage 4
    draw_cuboid(20.5, enc_y, w=1.45, h=1.1, d=0.75, face_color=PALETTE["stage4"], slices=depths[2])
    text(20.5, 2.22, "Stage 4 (C5 Apex)", size=11, weight="bold")
    text(20.5, 1.95, f"{depths[2]}× NeXt Blocks (Pure Conv)", size=9.2, color=PALETTE["text_muted"])
    draw_badge(20.5, 1.62, f"{shape_str(shapes['stage4'])}  (/16)", bg="#dcfce7", fg="#166534", pad=(0.85, 0.18), size=8.5)

    # =========================================================================
    # 3. PANEL 2: SCALE-GATED TOP-DOWN FPN NECK (DETAILED 2D FLOW)
    # =========================================================================
    ax.add_patch(FancyBboxPatch((0.2, -4.15), 24.6, 5.6, boxstyle="round,pad=0.2",
                                facecolor=PALETTE["panel_bg"], edgecolor=PALETTE["panel_border"],
                                linewidth=1.2, zorder=1))
    text(0.65, 1.15, "2. TOP-DOWN SCALE-GATED FPN NECK (SG-FPN)", size=11, weight="bold",
         color=PALETTE["text_dark"], ha="left")
    text(24.3, 1.15, "Multi-Scale Top-Down Guidance with Dynamic Scale-Gated Residual Fusion",
         size=9.5, weight="normal", color=PALETTE["text_muted"], ha="right")
    ax.plot([0.5, 24.5], [0.95, 0.95], color="#e2e8f0", linewidth=1.0, zorder=2)

    # -------------------------------------------------------------------------
    # Lateral Connectors descending from Encoder
    # -------------------------------------------------------------------------
    # Drop points: Stage 4 (x=20.5), Stage 3 (x=15.2), Stage 2 (x=10.2)
    # Drop down to Lateral conv nodes:
    # L5 at x=20.5, y=-0.25
    # L4 at x=15.2, y=-0.25
    # L3 at x=10.2, y=-0.25

    for drop_x, label, ch in [(20.5, "C5 Output", "128 ch"),
                              (15.2, "C4 Output (Post-Focal)", "96 ch"),
                              (10.2, "C2 Output (Post-ECA)", "48 ch")]:
        ax.annotate("", xy=(drop_x, 0.35), xytext=(drop_x, 1.35),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_lateral"], lw=1.6))
        text(drop_x + 0.1, 0.85, f"{label}  ({ch})", size=8.2, weight="bold",
             color=PALETTE["edge_lateral"], ha="left")

    # -------------------------------------------------------------------------
    # Flow Nodes in FPN Neck
    # -------------------------------------------------------------------------
    # LEVEL 5 (Stride 16): Col x = 20.5
    # L5 Node
    ax.add_patch(FancyBboxPatch((19.3, -0.45), 2.4, 0.75, boxstyle="round,pad=0.08",
                                facecolor="#fef3c7", edgecolor=PALETTE["lateral"], linewidth=1.2, zorder=3))
    text(20.5, -0.15, "Lateral L5", size=10, weight="bold", color="#92400e")
    text(20.5, -0.32, "Conv 1×1 (128 → 48)", size=8.4, color=PALETTE["text_muted"])
    text(20.5, -0.48, shape_str(shapes["lat_c5"]), size=8.0, weight="bold", color=PALETTE["text_dark"])

    # Path from L5 to U4 (Upsample 2x)
    # Drops from L5 bottom (20.5, -0.45) down to y = -1.8, then goes left to U4 (18.6, -1.8)
    ax.plot([20.5, 20.5, 18.6], [-0.45, -1.8, -1.8], color=PALETTE["edge_flow"], lw=1.6, zorder=2)
    ax.annotate("", xy=(18.6, -1.8), xytext=(19.0, -1.8),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_flow"], lw=1.6))
    text(19.6, -1.55, "Bilinear 2×", size=8.5, weight="bold", color=PALETTE["text_muted"])

    # LEVEL 4 (Stride 8): Col x = 15.2
    # L4 Node
    ax.add_patch(FancyBboxPatch((14.0, -0.45), 2.4, 0.75, boxstyle="round,pad=0.08",
                                facecolor="#fef3c7", edgecolor=PALETTE["lateral"], linewidth=1.2, zorder=3))
    text(15.2, -0.15, "Lateral L4", size=10, weight="bold", color="#92400e")
    text(15.2, -0.32, "Conv 1×1 (96 → 48)", size=8.4, color=PALETTE["text_muted"])
    text(15.2, -0.48, shape_str(shapes["lat_c4"]), size=8.0, weight="bold", color=PALETTE["text_dark"])

    # U4 Node (Upsample refinement)
    ax.add_patch(FancyBboxPatch((16.4, -2.15), 2.2, 0.70, boxstyle="round,pad=0.08",
                                facecolor="#eff6ff", edgecolor=PALETTE["upsample"], linewidth=1.2, zorder=3))
    text(17.5, -1.88, "Upsample U4", size=9.8, weight="bold", color="#1e40af")
    text(17.5, -2.03, "DWConv 3×3 + BN + SiLU", size=8.2, color=PALETTE["text_muted"])
    text(17.5, -2.18, shape_str(shapes["refine_u4"]), size=8.0, weight="bold", color=PALETTE["text_dark"])

    # Arrow from U4 left into P4
    ax.annotate("", xy=(15.75, -1.8), xytext=(16.4, -1.8),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_flow"], lw=1.6))

    # Arrow from L4 down into P4
    ax.annotate("", xy=(14.7, -1.35), xytext=(14.7, -0.55),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_lateral"], lw=1.6))
    text(14.35, -0.95, "L4", size=8.5, weight="bold", color=PALETTE["edge_lateral"], ha="right")

    # P4 Fusion Node (Prominent Featured Card)
    ax.add_patch(FancyBboxPatch((13.6, -2.55), 2.2, 1.25, boxstyle="round,pad=0.1",
                                facecolor=PALETTE["fusion_bg"], edgecolor=PALETTE["fusion"], linewidth=1.5, zorder=3))
    text(14.7, -1.52, "Scale-Gated Fusion (P4)" if gated else "FPN Addition (P4)", size=10, weight="bold", color="#166534")
    text(14.7, -1.72, "Gate4 = 2·σ(DW(L4 + U4))" if gated else "P4 = L4 + U4", size=8.0, color="#15803d")
    text(14.7, -1.90, "P4 = U4 + Gate4 ⊙ L4" if gated else "Channel: 48 ch", size=8.2, weight="bold", color="#166534")
    draw_badge(14.7, -2.25, f"{shape_str(shapes['refine_u4'])}  (/8)", bg="#ffffff", fg="#166534", pad=(0.75, 0.16), size=8.0)

    # Path from P4 to U3 (Upsample 2x)
    # Drops from P4 bottom (14.7, -2.55) down to y = -3.2, then goes left to U3 (13.2, -3.2)
    ax.plot([14.7, 14.7, 13.2], [-2.55, -3.2, -3.2], color=PALETTE["edge_flow"], lw=1.6, zorder=2)
    ax.annotate("", xy=(13.2, -3.2), xytext=(13.6, -3.2),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_flow"], lw=1.6))
    text(14.0, -2.90, "Bilinear 2×", size=8.5, weight="bold", color=PALETTE["text_muted"])

    # LEVEL 3 (Stride 4): Col x = 10.2
    # L3 Node
    ax.add_patch(FancyBboxPatch((9.0, -0.45), 2.4, 0.75, boxstyle="round,pad=0.08",
                                facecolor="#fef3c7", edgecolor=PALETTE["lateral"], linewidth=1.2, zorder=3))
    text(10.2, -0.15, "Lateral L3", size=10, weight="bold", color="#92400e")
    text(10.2, -0.32, f"Conv 1×1 (48 → {fusion_ch})", size=8.4, color=PALETTE["text_muted"])
    text(10.2, -0.48, shape_str(shapes["lat_c3"]), size=8.0, weight="bold", color=PALETTE["text_dark"])

    # U3 Node (Upsample projection 48 -> fusion_ch)
    ax.add_patch(FancyBboxPatch((11.1, -3.55), 2.1, 0.70, boxstyle="round,pad=0.08",
                                facecolor="#eff6ff", edgecolor=PALETTE["upsample"], linewidth=1.2, zorder=3))
    text(12.15, -3.28, "Upsample U3", size=9.8, weight="bold", color="#1e40af")
    text(12.15, -3.43, f"Conv 3×3 (48 → {fusion_ch})", size=8.2, color=PALETTE["text_muted"])
    text(12.15, -3.58, shape_str(shapes["proj_u3"]), size=8.0, weight="bold", color=PALETTE["text_dark"])

    # Arrow from U3 left into P3
    ax.annotate("", xy=(10.75, -3.2), xytext=(11.1, -3.2),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_flow"], lw=1.6))

    # Arrow from L3 down into P3
    ax.annotate("", xy=(9.7, -2.75), xytext=(9.7, -0.55),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_lateral"], lw=1.6))
    text(9.35, -1.8, "L3", size=8.5, weight="bold", color=PALETTE["edge_lateral"], ha="right")

    # P3 Fusion Node (Prominent Featured Card)
    ax.add_patch(FancyBboxPatch((8.6, -3.95), 2.2, 1.25, boxstyle="round,pad=0.1",
                                facecolor=PALETTE["fusion_bg"], edgecolor=PALETTE["fusion"], linewidth=1.5, zorder=3))
    text(9.7, -2.92, "Scale-Gated Fusion (P3)" if gated else "FPN Addition (P3)", size=10, weight="bold", color="#166534")
    text(9.7, -3.12, "Gate3 = 2·σ(DW(L3 + U3))" if gated else "P3 = L3 + U3", size=8.0, color="#15803d")
    text(9.7, -3.30, "P3 = U3 + Gate3 ⊙ L3" if gated else f"Channel: {fusion_ch} ch", size=8.2, weight="bold", color="#166534")
    draw_badge(9.7, -3.65, f"{shape_str(shapes['lat_c3'])}  (/4)", bg="#ffffff", fg="#166534", pad=(0.75, 0.16), size=8.0)

    # -------------------------------------------------------------------------
    # OUTPUT COLUMN: Stride 4 (Col x = 5.2 and x = 1.8)
    # -------------------------------------------------------------------------
    # Arrow from P3 left into Out Conv
    ax.annotate("", xy=(6.55, -3.4), xytext=(8.6, -3.4),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=1.8))

    # Output Conv Node
    ax.add_patch(FancyBboxPatch((4.5, -3.9), 2.05, 1.0, boxstyle="round,pad=0.08",
                                facecolor="#fff7ed", edgecolor="#ea580c", linewidth=1.3, zorder=3))
    text(5.52, -3.15, "Output Projection", size=10, weight="bold", color="#9a3412")
    text(5.52, -3.38, f"Conv 3×3: {fusion_ch} → {out_ch}", size=8.5, color=PALETTE["text_muted"])
    text(5.52, -3.55, "BN + SiLU", size=8.2, color=PALETTE["text_muted"])
    text(5.52, -3.72, shape_str(shapes["out_conv"]), size=8.0, weight="bold", color=PALETTE["text_dark"])

    # Arrow from Out Conv left into Final Backbone Output
    ax.annotate("", xy=(3.5, -3.4), xytext=(4.5, -3.4),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=2.0))

    # Final Backbone Output Card (Coral Highlight)
    ax.add_patch(FancyBboxPatch((0.6, -4.0), 2.9, 1.25, boxstyle="round,pad=0.1",
                                facecolor=PALETTE["output_bg"], edgecolor=PALETTE["output"], linewidth=1.8, zorder=3))
    text(2.05, -3.05, "BACKBONE OUTPUT", size=11, weight="bold", color="#b91c1c")
    text(2.05, -3.32, f"Shape: (1, {shape_str(report['output_shape'])})", size=9.2, weight="bold", color=PALETTE["text_dark"])
    text(2.05, -3.55, "Spatial Stride 4 (200 × 176)", size=8.5, color=PALETTE["text_muted"])
    text(2.05, -3.78, "To 4 Parallel Detection Heads", size=8.5, weight="bold", color="#c2410c")

    # =========================================================================
    # 4. PANEL 3: ARCHITECTURAL INSETS (TECHNICAL CARDS)
    # =========================================================================
    ax.add_patch(FancyBboxPatch((0.2, -9.1), 24.6, 4.65, boxstyle="round,pad=0.2",
                                facecolor=PALETTE["panel_bg"], edgecolor=PALETTE["panel_border"],
                                linewidth=1.2, zorder=1))
    text(0.65, -4.68, "3. CORE MICRO-ARCHITECTURE MODULES & FORMULATIONS", size=11, weight="bold",
         color=PALETTE["text_dark"], ha="left")
    ax.plot([0.5, 24.5], [-4.88, -4.88], color="#e2e8f0", linewidth=1.0, zorder=2)

    # -------------------------------------------------------------------------
    # Card A: NeXt Residual Block (Left: x=0.5 to 8.2)
    # -------------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.5, -8.85), 7.6, 3.75, boxstyle="round,pad=0.1",
                                facecolor=PALETTE["card_bg"], edgecolor=PALETTE["card_border"], linewidth=1.0, zorder=2))
    text(0.8, -5.15, "A. NeXt Residual Block (Depthwise Inverted Expansion)", size=10.5, weight="bold",
         color=PALETTE["text_dark"], ha="left")

    # Sub-flow diagram inside Card A
    # Sequence: x -> DW 7x7 (BN) -> PW 1x1 (exp 2.5, SiLU) -> PW 1x1 (BN) -> LayerScale -> (+) -> y
    flow_steps = [
        ("Input x", "#e2e8f0", PALETTE["text_dark"]),
        ("DWConv 7×7\n(BN)", "#ccfbf1", "#0f766e"),
        ("PWConv 1×1\n(exp 2.5, SiLU)", "#fef3c7", "#92400e"),
        ("PWConv 1×1\n(BN)", "#fef3c7", "#92400e"),
        ("LayerScale (γ)\ninit=1e-3", "#e0e7ff", "#3730a3"),
        ("Output y\n= x + γ·F(x)", "#dcfce7", "#166534"),
    ]
    step_xs = [1.1, 2.3, 3.7, 5.0, 6.3, 7.5]
    for i, (label, bg, fg) in enumerate(flow_steps):
        sx = step_xs[i]
        ax.add_patch(FancyBboxPatch((sx - 0.48, -6.6), 0.96, 0.85, boxstyle="round,pad=0.06",
                                    facecolor=bg, edgecolor=fg, linewidth=0.9, zorder=3))
        text(sx, -6.18, label, size=7.8, weight="bold", color=fg, linespacing=1.2)
        if i < len(flow_steps) - 1:
            next_x = step_xs[i + 1]
            ax.annotate("", xy=(next_x - 0.48, -6.18), xytext=(sx + 0.48, -6.18),
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=9, color=PALETTE["edge_flow"], lw=1.2))

    # Skip connection arrow for residual
    ax.annotate("", xy=(7.02, -6.6), xytext=(1.1, -6.6),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=10, color=PALETTE["edge_lateral"],
                                lw=1.2, connectionstyle="arc3,rad=-0.4"))
    text(4.3, -7.25, "Residual Skip: + x", size=8.0, weight="bold", color=PALETTE["edge_lateral"])

    # Technical bullet points for Card A
    text(0.8, -7.70, "• Metric Receptive Field: 7×7 kernel covers 0.7m in BEV grid, matching car dimensions.",
         size=8.2, color=PALETTE["text_muted"], ha="left")
    text(0.8, -8.10, "• Inverted Expansion (t=2.5): Expands channel width in depthwise space for representation power.",
         size=8.2, color=PALETTE["text_muted"], ha="left")
    text(0.8, -8.50, "• LayerScale Stability: Multiplies block output by γ (init 0.001) for robust deep residual convergence.",
         size=8.2, color=PALETTE["text_muted"], ha="left")

    # -------------------------------------------------------------------------
    # Card B: Scale-Gated Fusion Mechanism (Middle: x=8.5 to 16.5)
    # -------------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((8.5, -8.85), 7.8, 3.75, boxstyle="round,pad=0.1",
                                facecolor=PALETTE["card_bg"], edgecolor=PALETTE["card_border"], linewidth=1.0, zorder=2))
    text(8.8, -5.15, "B. Scale-Gated Feature Fusion Mechanism", size=10.5, weight="bold",
         color=PALETTE["text_dark"], ha="left")

    # Mathematical formulation box
    ax.add_patch(FancyBboxPatch((8.8, -6.6), 7.2, 1.25, boxstyle="round,pad=0.08",
                                facecolor="#f0fdf4", edgecolor=PALETTE["fusion"], linewidth=1.2, zorder=3))
    text(12.4, -5.60, r"$\mathbf{P}_i = \mathbf{U}_i + \mathbf{Gate}_i \odot \mathbf{L}_i$",
         size=11.5, weight="bold", color="#166534")
    text(12.4, -6.00, r"$\mathbf{Gate}_i = 2 \cdot \sigma\left(\mathrm{DWConv}_{3\times3}(\mathbf{L}_i + \mathbf{U}_i)\right)$",
         size=10.5, weight="bold", color="#15803d")
    text(12.4, -6.35, "Zero-initialized gate: 2·σ(0) = 1.0 (smoothly converges from standard addition)",
         size=8.2, color="#166534")

    # Technical bullet points for Card B
    text(8.8, -7.15, "• Dynamic Resolution Balancing: Fuses high-resolution spatial details (Li) with semantically",
         size=8.2, color=PALETTE["text_muted"], ha="left")
    text(8.8, -7.45, "  rich top-down context (Ui) through adaptive spatial-channel confidence weights.",
         size=8.2, color=PALETTE["text_muted"], ha="left")
    text(8.8, -7.85, "• Metric Gating: Depthwise 3×3 convolution captures local neighborhood alignment before",
         size=8.2, color=PALETTE["text_muted"], ha="left")
    text(8.8, -8.15, "  modulating lateral features, preventing semantic dilution across metric scales.",
         size=8.2, color=PALETTE["text_muted"], ha="left")
    text(8.8, -8.55, "• Gradient Flow Preservation: Additive identity highway ensures stable backpropagation.",
         size=8.2, color=PALETTE["text_muted"], ha="left")

    # -------------------------------------------------------------------------
    # Card C: Attention & Multi-Scale Context (Right: x=16.7 to 24.5)
    # -------------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((16.7, -8.85), 7.8, 3.75, boxstyle="round,pad=0.1",
                                facecolor=PALETTE["card_bg"], edgecolor=PALETTE["card_border"], linewidth=1.0, zorder=2))
    text(17.0, -5.15, "C. Attention & Global Context Modules", size=10.5, weight="bold",
         color=PALETTE["text_dark"], ha="left")

    # Module C1: ECA
    ax.add_patch(FancyBboxPatch((17.0, -6.75), 7.2, 1.45, boxstyle="round,pad=0.08",
                                facecolor="#faf5ff", edgecolor=PALETTE["attention"], linewidth=1.0, zorder=3))
    text(17.2, -5.50, "ECA (Efficient Channel Attention) @ Stride 4", size=9.5, weight="bold",
         color="#6b21a8", ha="left")
    text(17.2, -5.75, "• Global Average Pooling  →  Conv1D (k=3 across channels)  →  Sigmoid",
         size=8.2, color=PALETTE["text_muted"], ha="left")
    text(17.2, -6.05, "• Cross-Channel Interaction: Captures channel dependencies without dimensionality reduction.",
         size=8.2, color=PALETTE["text_muted"], ha="left")
    text(17.2, -6.35, f"• Residual Adapter: x + β · (ECA(x) - x) with initial scaling β = {report.get('local_attention_layer_scale_init', 0.001):g}.",
         size=8.2, weight="bold", color="#581c87", ha="left")

    # Module C2: FocalContext
    ax.add_patch(FancyBboxPatch((17.0, -8.6), 7.2, 1.65, boxstyle="round,pad=0.08",
                                facecolor="#eef2ff", edgecolor="#6366f1", linewidth=1.0, zorder=3))
    text(17.2, -7.15, "FocalContext (Multi-Granularity Context) @ Stride 8", size=9.5, weight="bold",
         color="#3730a3", ha="left")
    text(17.2, -7.40, f"• Bottleneck Dimension: D = {report.get('focal_bottleneck', 64)} channels.",
         size=8.2, color=PALETTE["text_muted"], ha="left")
    text(17.2, -7.70, f"• Multi-Scale Dilated DWConv: Sequential dilations d = {report.get('focal_dilations', [1,2,3])} + Global Pooling.",
         size=8.2, color=PALETTE["text_muted"], ha="left")
    text(17.2, -8.00, "• Softmax Multi-Level Modulation: Dynamically gates local and broad spatial context.",
         size=8.2, color=PALETTE["text_muted"], ha="left")
    text(17.2, -8.30, f"• Residual Highway: LayerScale initialized with γ = {report.get('focal_layer_scale_init', 0.001):g}.",
         size=8.2, weight="bold", color="#312e81", ha="left")

    # Save to disk
    for ext in ("png", "svg", "pdf"):
        fig.savefig(target_path.with_suffix("." + ext), dpi=300, facecolor=PALETTE["bg_main"])
    plt.close(fig)


def write_tex(report, target_path):
    """Generate standalone LaTeX/TikZ code adhering to PlotNeuralNet standard."""
    import os
    layers_rel = os.path.relpath(VENDOR / "PlotNeuralNet" / "Layers", target_path.parent).replace(os.sep, "/")

    def esc(val):
        return (str(val).replace("_", r"\_").replace("->", r"$\to$")
                .replace("gamma", r"$\gamma$").replace(" x ", r" $\times$ ")
                .replace("×", r"$\times$"))

    shapes = report["shapes"]
    depths = report["stage_depths"]
    eca = report["local_attention"] == "eca"
    focal = report["c4_context"] == "focal"

    tex = [
        r"\documentclass[tikz,border=14pt]{standalone}" + "\n",
        r"\usepackage{import}" + "\n",
        rf"\subimport{{{layers_rel}/}}{{init.tex}}" + "\n",
        r"\usetikzlibrary{3d,shapes.geometric,arrows.meta,positioning,fit,backgrounds}" + "\n",
        r"\definecolor{inputCol}{HTML}{b8c9dc}" + "\n",
        r"\definecolor{stemCol}{HTML}{f3ba6f}" + "\n",
        r"\definecolor{stage2Col}{HTML}{66b7ab}" + "\n",
        r"\definecolor{stage3Col}{HTML}{47a396}" + "\n",
        r"\definecolor{stage4Col}{HTML}{2d8a7c}" + "\n",
        r"\definecolor{attnCol}{HTML}{9c7ec5}" + "\n",
        r"\definecolor{latCol}{HTML}{fef3c7}" + "\n",
        r"\definecolor{latBorder}{HTML}{d97706}" + "\n",
        r"\definecolor{upCol}{HTML}{eff6ff}" + "\n",
        r"\definecolor{upBorder}{HTML}{2563eb}" + "\n",
        r"\definecolor{fuseCol}{HTML}{f0f8ed}" + "\n",
        r"\definecolor{fuseBorder}{HTML}{16a34a}" + "\n",
        r"\definecolor{outCol}{HTML}{fdf1ef}" + "\n",
        r"\definecolor{outBorder}{HTML}{dc2626}" + "\n",
        r"\begin{document}" + "\n",
        r"\begin{tikzpicture}[font=\sffamily," + "\n",
        r"  flowArrow/.style={-Stealth,line width=1.2pt,draw=black!75}," + "\n",
        r"  latArrow/.style={-Stealth,line width=1.2pt,draw=orange!80!black}," + "\n",
        r"  blockNode/.style={draw=black!40,rounded corners=4pt,inner sep=6pt,align=center}," + "\n",
        r"  titleNode/.style={font=\sffamily\bfseries\Large}," + "\n",
        r"]" + "\n",
    ]

    # Title
    title = f"MobilePixorNeXt Backbone Architecture"
    sub = (f"Hist14 BEV Input  |  {depths[0]}/{depths[1]}/{depths[2]} NeXt Depths  |  "
           f"{report['parameters']:,} Parameters  |  Stride 4 Multi-Scale Output")
    tex.append(rf"\node[titleNode] at (12.5,7.2) {{{title}}};" + "\n")
    tex.append(rf"\node[font=\sffamily\small,text=black!70] at (12.5,6.65) {{{sub}}};" + "\n")

    # Section 1 Label
    tex.append(r"\node[anchor=west,font=\sffamily\bfseries\normalsize] at (0.5,5.6) {1. BOTTOM-UP MULTI-SCALE 3D ENCODER};" + "\n")

    # 3D Boxes using PlotNeuralNet primitives
    # BEV Input
    tex.append(r"""
\pic[shift={(1.8,3.6,0)}] at (0,0,0) {
    Box={name=bev,caption=,xlabel={" "},zlabel=,fill=inputCol,height=12.0,width=1.6,depth=3.5}
};
\node[anchor=north,align=center,font=\sffamily\scriptsize] at (1.8,1.8) {\textbf{BEV Input}\\Hist14 Slices\\14 $\times$ 800 $\times$ 704};
""")

    # Stem
    tex.append(r"""
\pic[shift={(5.6,3.6,0)}] at (0,0,0) {
    Box={name=stem,caption=,xlabel={" "," "},zlabel=,fill=stemCol,height=10.5,width={1.6,1.6},depth=3.5}
};
\node[anchor=north,align=center,font=\sffamily\scriptsize] at (5.6,1.8) {\textbf{Spatial Stem}\\2x Conv 3$\times$3, s=2; s=1\\32 $\times$ 400 $\times$ 352 (s=2)};
\draw[flowArrow] (bev-east) -- (stem-west) node[midway,above,font=\sffamily\tiny] {Downsample};
""")

    # Stage 2
    if eca:
        tex.append(r"""
\pic[shift={(10.2,3.6,0)}] at (0,0,0) {
    RightBandedBox={name=stage2,caption=,xlabel={" "," "},zlabel=,fill=stage2Col,bandfill=attnCol,height=8.8,width=5.0,depth=3.5}
};
""")
    else:
        tex.append(r"""
\pic[shift={(10.2,3.6,0)}] at (0,0,0) {
    Box={name=stage2,caption=,xlabel={" "},zlabel=,fill=stage2Col,height=8.8,width=5.0,depth=3.5}
};
""")
    tex.append(rf"\node[anchor=north,align=center,font=\sffamily\scriptsize] at (10.2,1.8) {{\textbf{{Stage 2 (C2)}}\\{depths[0]}x NeXt Blocks + ECA\\48 $\times$ 200 $\times$ 176 (s=4)}};" + "\n")
    tex.append(r"\draw[flowArrow] (stem-east) -- (stage2-west) node[midway,above,font=\sffamily\tiny] {Downsample};" + "\n")

    # Stage 3
    if focal:
        tex.append(r"""
\pic[shift={(15.2,3.6,0)}] at (0,0,0) {
    RightBandedBox={name=stage3,caption=,xlabel={" "," "},zlabel=,fill=stage3Col,bandfill=attnCol,height=7.0,width=6.5,depth=3.5}
};
""")
    else:
        tex.append(r"""
\pic[shift={(15.2,3.6,0)}] at (0,0,0) {
    Box={name=stage3,caption=,xlabel={" "},zlabel=,fill=stage3Col,height=7.0,width=6.5,depth=3.5}
};
""")
    tex.append(rf"\node[anchor=north,align=center,font=\sffamily\scriptsize] at (15.2,1.8) {{\textbf{{Stage 3 (C3/C4)}}\\{depths[1]}x NeXt Blocks + FocalContext\\96 $\times$ 100 $\times$ 88 (s=8)}};" + "\n")
    tex.append(r"\draw[flowArrow] (stage2-east) -- (stage3-west) node[midway,above,font=\sffamily\tiny] {Downsample};" + "\n")

    # Stage 4
    tex.append(r"""
\pic[shift={(20.5,3.6,0)}] at (0,0,0) {
    Box={name=stage4,caption=,xlabel={" "," "},zlabel=,fill=stage4Col,height=5.2,width={2.5,2.5},depth=3.5}
};
\node[anchor=north,align=center,font=\sffamily\scriptsize] at (20.5,1.8) {\textbf{Stage 4 (C5)}\\2x NeXt Blocks\\128 $\times$ 50 $\times$ 44 (s=16)};
\draw[flowArrow] (stage3-east) -- (stage4-west) node[midway,above,font=\sffamily\tiny] {Downsample};
""")

    # Section 2 Label: FPN
    tex.append(r"""
\node[anchor=west,font=\sffamily\bfseries\normalsize] at (0.5,1.0) {2. TOP-DOWN SCALE-GATED FPN NECK (SG-FPN)};

% 2D Flow Nodes for FPN
\node[blockNode,fill=latCol,draw=latBorder] (lat5) at (20.5,-0.2) {\textbf{Lateral L5}\\1x1 Conv: 128 $\to$ 48\\48 $\times$ 50 $\times$ 44};
\node[blockNode,fill=latCol,draw=latBorder] (lat4) at (15.2,-0.2) {\textbf{Lateral L4}\\1x1 Conv: 96 $\to$ 48\\48 $\times$ 100 $\times$ 88};
\node[blockNode,fill=latCol,draw=latBorder] (lat3) at (10.2,-0.2) {\textbf{Lateral L3}\\1x1 Conv: 48 $\to$ 24\\24 $\times$ 200 $\times$ 176};

\node[blockNode,fill=upCol,draw=upBorder] (u4) at (17.5,-1.8) {\textbf{Upsample U4}\\Bilinear 2x + DWConv 3x3\\48 $\times$ 100 $\times$ 88};
\node[blockNode,fill=fuseCol,draw=fuseBorder] (p4) at (14.5,-1.8) {\textbf{Scale-Gated Fusion P4}\\$P_4 = U_4 + Gate_4 \odot L_4$\\48 $\times$ 100 $\times$ 88};

\node[blockNode,fill=upCol,draw=upBorder] (u3) at (12.2,-3.2) {\textbf{Upsample U3}\\Bilinear 2x + Conv 3x3\\24 $\times$ 200 $\times$ 176};
\node[blockNode,fill=fuseCol,draw=fuseBorder] (p3) at (9.5,-3.2) {\textbf{Scale-Gated Fusion P3}\\$P_3 = U_3 + Gate_3 \odot L_3$\\24 $\times$ 200 $\times$ 176};

\node[blockNode,fill=latCol,draw=latBorder] (outconv) at (5.5,-3.2) {\textbf{Output Projection}\\Conv 3x3: 24 $\to$ 32\\32 $\times$ 200 $\times$ 176};
\node[blockNode,fill=outCol,draw=outBorder] (out) at (1.8,-3.2) {\textbf{BACKBONE OUTPUT}\\(1, 32, 200, 176) [s=4]\\To 3D Detection Heads};

% Connector Arrows
\draw[latArrow] (stage4-south) -- (lat5.north) node[midway,right,font=\sffamily\tiny] {C5};
\draw[latArrow] (stage3-south) -- (lat4.north) node[midway,right,font=\sffamily\tiny] {C4};
\draw[latArrow] (stage2-south) -- (lat3.north) node[midway,right,font=\sffamily\tiny] {C2};

\draw[flowArrow] (lat5.south) |- (u4.east);
\draw[flowArrow] (u4.west) -- (p4.east);
\draw[latArrow] (lat4.south) |- (p4.north);

\draw[flowArrow] (p4.south) |- (u3.east);
\draw[flowArrow] (u3.west) -- (p3.east);
\draw[latArrow] (lat3.south) |- (p3.north);

\draw[flowArrow] (p3.west) -- (outconv.east);
\draw[flowArrow] (outconv.west) -- (out.east);

% Section 3: Inset Cards
\node[anchor=west,font=\sffamily\bfseries\normalsize] at (0.5,-4.8) {3. CORE ARCHITECTURAL MODULES};

\node[blockNode,anchor=north west,text width=7.2cm,align=left,fill=white] at (0.5,-5.2) {
  \textbf{A. NeXt Residual Block}\\[2pt]
  $\bullet$ DWConv 7$\times$7 $\to$ BN $\to$ PWConv 1$\times$1 (exp 2.5, SiLU) $\to$ PWConv 1$\times$1 (BN) $\to$ LayerScale ($\gamma=10^{-3}$) $\to$ Add Residual\\[2pt]
  $\bullet$ 0.7m BEV metric receptive field captures full vehicle geometry.
};

\node[blockNode,anchor=north west,text width=7.2cm,align=left,fill=white] at (8.5,-5.2) {
  \textbf{B. Scale-Gated FPN Fusion}\\[2pt]
  $\mathbf{P}_i = \mathbf{U}_i + \mathbf{Gate}_i \odot \mathbf{L}_i$\\[2pt]
  $\mathbf{Gate}_i = 2 \cdot \sigma(\mathrm{DWConv}_{3\times3}(\mathbf{L}_i + \mathbf{U}_i))$\\[2pt]
  $\bullet$ Zero-init: $2\sigma(0)=1.0$, smoothly learns metric scale gating.
};

\node[blockNode,anchor=north west,text width=7.2cm,align=left,fill=white] at (16.5,-5.2) {
  \textbf{C. Attention \& Global Context}\\[2pt]
  $\bullet$ \textbf{ECA @ Stride 4}: Global Pool $\to$ Conv1D ($k=3$) $\to$ Sigmoid modulation with $\beta=10^{-3}$ residual.\\[2pt]
  $\bullet$ \textbf{FocalContext @ Stride 8}: Bottleneck $D=64$, dilated DWConv ($d=\{1,2,3\}$) + global context with Softmax gate.
};
""")

    tex.append(r"\end{tikzpicture}" + "\n")
    tex.append(r"\end{document}" + "\n")

    target_path.write_text("".join(tex))


def compile_tex(tex_path):
    """Compile generated TeX file with pdflatex if available."""
    compiler = shutil.which("pdflatex")
    if compiler is None:
        return False
    try:
        subprocess.run(
            [compiler, "-interaction=nonstopmode", "-halt-on-error",
             "-jobname=" + tex_path.stem + "_latex", tex_path.name],
            cwd=tex_path.parent, check=True, capture_output=True
        )
        for ext in ("aux", "log"):
            tex_path.with_name(tex_path.stem + "_latex." + ext).unlink(missing_ok=True)
        return True
    except subprocess.CalledProcessError:
        return False


def update_readme(report, out_dir):
    """Write concise documentation for the generated diagram."""
    readme_path = out_dir / "README.md"
    readme_content = f"""# MobilePixorNeXt Backbone Architecture Diagram

![MobilePixorNeXt Backbone](mobilepixornext_backbone.png)

Diagram architecture based on configuration [`{Path(report['config']).name}`](../../{report['config']}) and source code `detector/core/models/backbones/mobilepixornext.py`.

## Architectural Summary

- **Total Backbone Parameters**: **{report['parameters']:,}**
- **Input Tensor**: `(1, {shape_str(report['input_shape'])})` (Hist14 BEV encoding, Stride 1)
- **Output Tensor**: `(1, {shape_str(report['output_shape'])})` (Stride 4 feature map, feeding Detection Heads)
- **Multi-Scale Encoder**: 
  - Stem (Stride 2): 2× Conv 3×3 + BN + SiLU
  - Stage 2 (Stride 4): {report['stage_depths'][0]}× NeXt Blocks (DW 7×7, exp {report['expansion']:g}) + ECA Channel Attention
  - Stage 3 (Stride 8): {report['stage_depths'][1]}× NeXt Blocks + FocalContext (dilations [1, 2, 3] + global context)
  - Stage 4 (Stride 16): {report['stage_depths'][2]}× NeXt Blocks (apex metric context)
- **Top-Down Scale-Gated FPN Neck**:
  - Laterals $L_5, L_4, L_3$ projecting features into fusion width ({report['fusion_channels']})
  - Bilinear 2× upsamplers with refinement convs ($U_4, U_3$)
  - Scale-Gated fusion gates $P = U + 2\\sigma(\\text{{DW3x3}}(L+U)) \\odot L$
  - Final projection producing 32-channel Stride 4 representation

## Generated Files

| File | Purpose |
| --- | --- |
| `mobilepixornext_backbone.png` | 300 DPI high-resolution raster image (paper/report ready) |
| `mobilepixornext_backbone.svg` | Scalable vector graphic (editable in Inkscape / Illustrator) |
| `mobilepixornext_backbone.pdf` | Single-page publication-quality vector PDF |
| `mobilepixornext_backbone.tex` | LaTeX/TikZ source code using PlotNeuralNet layers for Overleaf / TeXLive |
| `mobilepixornext_backbone.json` | Exact tensor dimensions and parameter verified report |

## Reproduction

Run from repository root using Python environment with PyTorch and Matplotlib:

```bash
MPLCONFIGDIR=/tmp/lidar-matplotlib \\
  /home/duyennh/miniconda3/envs/AI_env/bin/python \\
  tools/visualization/plot_mobilepixornext.py
```
"""
    readme_path.write_text(readme_content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to experiment config JSON")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagrams" / "plotneuralnet",
                        help="Directory to save generated diagram files")
    parser.add_argument("--compile", action="store_true",
                        help="Attempt compiling TeX file with pdflatex")
    args = parser.parse_args()

    print(f"Resolving model configuration: {args.config}")
    report = inspect_backbone(args.config.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir.resolve() / "mobilepixornext_backbone"

    print("Generating Matplotlib high-resolution preview (PNG, SVG, PDF)...")
    render_preview(report, target)

    print("Generating PlotNeuralNet LaTeX/TikZ code (.tex)...")
    write_tex(report, target.with_suffix(".tex"))

    if args.compile:
        print("Compiling TeX with pdflatex...")
        ok = compile_tex(target.with_suffix(".tex"))
        if ok:
            print(f"Successfully compiled: {target.parent / (target.stem + '_latex.pdf')}")
        else:
            print("Note: pdflatex not found or compilation skipped; local PDF/PNG/SVG previews are fully available.")

    report["plotneuralnet_vendor"] = str(VENDOR)
    target.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    update_readme(report, args.output_dir)

    print(f"Verification successful! Backbone input {report['input_shape']} -> output {report['output_shape']}")
    print(f"Artifacts saved to {args.output_dir}:")
    print(f"  • {target.name}.png (300 DPI)")
    print(f"  • {target.name}.svg (Vector)")
    print(f"  • {target.name}.pdf (Vector PDF)")
    print(f"  • {target.name}.tex (PlotNeuralNet LaTeX)")
    print(f"  • {target.name}.json (Report)")


if __name__ == "__main__":
    main()
