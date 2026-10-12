import re
from pathlib import Path

path = Path("tools/visualization/plot_mobilepixornext.py")
content = path.read_text()

# We need to replace the content of render_preview between PANEL 1 and PANEL 3
new_panels = """
    # =========================================================================
    # 2. PANEL 1: MULTI-SCALE 3D ENCODER (LEFT COLUMN)
    # =========================================================================
    # Background panel for Encoder
    ax.add_patch(FancyBboxPatch((0.2, -8.0), 9.6, 13.8, boxstyle="round,pad=0.2",
                                facecolor=PALETTE["panel_bg"], edgecolor=PALETTE["panel_border"],
                                linewidth=1.2, zorder=1))
    text(0.65, 5.5, "1. BOTTOM-UP ENCODER", size=11, weight="bold", color=PALETTE["text_dark"], ha="left")
    ax.plot([0.5, 9.5], [5.3, 5.3], color="#e2e8f0", linewidth=1.0, zorder=2)

    enc_x = 3.5

    # A. BEV Input
    draw_cuboid(enc_x, 4.2, w=0.55, h=2.5, d=0.75, face_color=PALETTE["input"], slices=1)
    text(enc_x, 2.75, "BEV Input", size=11, weight="bold")
    text(enc_x, 2.45, "Hist14 Slices", size=9.2, color=PALETTE["text_muted"])
    draw_badge(enc_x, 2.15, shape_str(report["input_shape"]), bg="#e2e8f0", pad=(0.75, 0.18), size=8.5)

    # Transition Arrow: BEV -> Stem
    ax.annotate("", xy=(enc_x, 1.3), xytext=(enc_x, 1.9),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=1.6))
    text(enc_x + 0.3, 1.6, "Downsample", size=8.8, weight="bold", color=PALETTE["text_dark"], ha="left")
    text(enc_x + 0.3, 1.35, "Conv 3×3, s=2", size=8.2, color=PALETTE["text_muted"], ha="left")

    # B. Spatial Stem
    draw_cuboid(enc_x, 0.2, w=0.85, h=2.2, d=0.75, face_color=PALETTE["stem"], slices=2)
    text(enc_x, -1.25, "Fast Spatial Stem", size=11, weight="bold")
    draw_badge(enc_x, -1.65, f"{shape_str(shapes['stem'])}  (/2)", bg="#fef3c7", fg="#92400e", pad=(0.85, 0.18), size=8.5)

    # Transition Arrow: Stem -> Stage 2
    ax.annotate("", xy=(enc_x, -2.4), xytext=(enc_x, -1.9),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=1.6))
    text(enc_x + 0.3, -2.15, "Downsample", size=8.8, weight="bold", color=PALETTE["text_dark"], ha="left")

    # C. Stage 2 (with ECA Band)
    draw_cuboid(enc_x, -3.5, w=1.5, h=1.85, d=0.75, face_color=PALETTE["stage2"],
                slices=depths[0], band_color=PALETTE["attention"] if eca else None)
    text(enc_x, -4.75, f"Stage 2 (C2) - {depths[0]}× NeXt" + (" + ECA" if eca else ""), size=10, weight="bold")
    draw_badge(enc_x, -5.15, f"{shape_str(shapes['c3_light_attention'])}  (/4)", bg="#ccfbf1", fg="#0f766e", pad=(0.85, 0.18), size=8.5)

    # Transition Arrow: Stage 2 -> Stage 3
    ax.annotate("", xy=(enc_x, -5.8), xytext=(enc_x, -5.4),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=1.6))
    text(enc_x + 0.3, -5.6, "Downsample", size=8.8, weight="bold", color=PALETTE["text_dark"], ha="left")

    # D. Stage 3 (with FocalContext Band)
    draw_cuboid(enc_x, -6.65, w=1.9, h=1.45, d=0.75, face_color=PALETTE["stage3"],
                slices=depths[1], band_color=PALETTE["attention"] if focal else None)
    text(enc_x, -7.7, f"Stage 3 (C3/C4) - {depths[1]}× NeXt" + (" + FocalContext" if focal else ""), size=10, weight="bold")
    draw_badge(enc_x, -8.1, f"{shape_str(shapes['c4_context'])}  (/8)", bg="#e0e7ff", fg="#3730a3", pad=(0.85, 0.18), size=8.5)

    # Transition Arrow: Stage 3 -> Stage 4
    # Note: Stage 4 is moved to the bottom right for the YOLO U-shape, 
    # but we can also place it at the bottom left. Let's place it at bottom left.
    ax.annotate("", xy=(enc_x, -8.7), xytext=(enc_x, -8.35),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=1.6))
    
    # Need more space, let's adjust Y scales globally for left column.
    # We will adjust everything upwards by 1.0 unit.
"""

# Wait, let's just write the full python script and replace it. 
