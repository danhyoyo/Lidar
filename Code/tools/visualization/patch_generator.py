import json
from pathlib import Path

def patch_file():
    path = Path("tools/visualization/plot_mobilepixornext.py")
    content = path.read_text()

    start_marker = "    # =========================================================================\n    # 2. PANEL 1: MULTI-SCALE 3D ENCODER"
    end_marker = "    # =========================================================================\n    # 4. PANEL 3: ARCHITECTURAL INSETS (TECHNICAL CARDS)"
    
    idx_start = content.find(start_marker)
    idx_end = content.find(end_marker)
    
    if idx_start == -1 or idx_end == -1:
        print("Markers not found!")
        return

    new_panels = """    # =========================================================================
    # 2. LEFT PANEL: MULTI-SCALE 3D ENCODER (DOWNSAMPLE)
    # =========================================================================
    # Background panel
    ax.add_patch(FancyBboxPatch((0.2, -8.7), 8.5, 14.5, boxstyle="round,pad=0.2",
                                facecolor=PALETTE["panel_bg"], edgecolor=PALETTE["panel_border"],
                                linewidth=1.2, zorder=1))
    text(0.65, 5.4, "1. BOTTOM-UP ENCODER (Left Column)", size=11, weight="bold",
         color=PALETTE["text_dark"], ha="left")
    ax.plot([0.5, 8.4], [5.1, 5.1], color="#e2e8f0", linewidth=1.0, zorder=2)

    enc_x = 4.3

    # A. BEV Input
    draw_cuboid(enc_x, 4.0, w=0.55, h=2.0, d=0.75, face_color=PALETTE["input"], slices=1)
    text(enc_x, 2.75, "BEV Input", size=10, weight="bold")
    draw_badge(enc_x, 2.35, shape_str(report["input_shape"]), bg="#e2e8f0", pad=(0.6, 0.15), size=8)

    # Transition
    ax.annotate("", xy=(enc_x, 1.7), xytext=(enc_x, 2.15),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=PALETTE["edge_flow"], lw=1.6))
    text(enc_x + 0.3, 1.95, "Conv 3×3, s=2", size=8, color=PALETTE["text_muted"], ha="left")

    # B. Spatial Stem
    draw_cuboid(enc_x, 0.9, w=0.85, h=1.8, d=0.75, face_color=PALETTE["stem"], slices=2)
    text(enc_x, -0.25, "Fast Spatial Stem", size=10, weight="bold")
    draw_badge(enc_x, -0.65, f"{shape_str(shapes['stem'])}  (/2)", bg="#fef3c7", fg="#92400e", pad=(0.6, 0.15), size=8)

    # Transition
    ax.annotate("", xy=(enc_x, -1.3), xytext=(enc_x, -0.85),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=PALETTE["edge_flow"], lw=1.6))
    text(enc_x + 0.3, -1.1, "Conv 3×3, s=2", size=8, color=PALETTE["text_muted"], ha="left")

    # C. Stage 2 (with ECA Band)
    draw_cuboid(enc_x, -2.15, w=1.5, h=1.5, d=0.75, face_color=PALETTE["stage2"],
                slices=depths[0], band_color=PALETTE["attention"] if eca else None)
    text(enc_x, -3.25, f"Stage 2 (C2)", size=10, weight="bold")
    draw_badge(enc_x, -3.65, f"{shape_str(shapes['c3_light_attention'])}  (/4)", bg="#ccfbf1", fg="#0f766e", pad=(0.6, 0.15), size=8)

    # Transition
    ax.annotate("", xy=(enc_x, -4.35), xytext=(enc_x, -3.85),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=PALETTE["edge_flow"], lw=1.6))
    text(enc_x + 0.3, -4.1, "Conv 3×3, s=2", size=8, color=PALETTE["text_muted"], ha="left")

    # D. Stage 3 (with FocalContext Band)
    draw_cuboid(enc_x, -5.15, w=1.9, h=1.3, d=0.75, face_color=PALETTE["stage3"],
                slices=depths[1], band_color=PALETTE["attention"] if focal else None)
    text(enc_x, -6.15, f"Stage 3 (C3/C4)", size=10, weight="bold")
    draw_badge(enc_x, -6.55, f"{shape_str(shapes['c4_context'])}  (/8)", bg="#e0e7ff", fg="#3730a3", pad=(0.6, 0.15), size=8)

    # Transition
    ax.annotate("", xy=(enc_x, -7.25), xytext=(enc_x, -6.75),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=PALETTE["edge_flow"], lw=1.6))
    text(enc_x + 0.3, -7.0, "Conv 3×3, s=2", size=8, color=PALETTE["text_muted"], ha="left")

    # E. Stage 4
    draw_cuboid(enc_x, -7.75, w=1.45, h=0.8, d=0.75, face_color=PALETTE["stage4"], slices=depths[2])
    text(enc_x, -8.45, f"Stage 4 (C5)", size=10, weight="bold")
    #draw_badge(enc_x, -8.85, f"{shape_str(shapes['stage4'])}  (/16)", bg="#dcfce7", fg="#166534", pad=(0.6, 0.15), size=8)


    # =========================================================================
    # 3. RIGHT PANEL: SCALE-GATED TOP-DOWN FPN NECK (UPSAMPLE)
    # =========================================================================
    ax.add_patch(FancyBboxPatch((9.2, -8.7), 15.6, 14.5, boxstyle="round,pad=0.2",
                                facecolor=PALETTE["panel_bg"], edgecolor=PALETTE["panel_border"],
                                linewidth=1.2, zorder=1))
    text(9.65, 5.4, "2. TOP-DOWN FPN NECK (Right Column)", size=11, weight="bold",
         color=PALETTE["text_dark"], ha="left")
    ax.plot([9.5, 24.5], [5.1, 5.1], color="#e2e8f0", linewidth=1.0, zorder=2)

    neck_x = 18.5
    lat_x = 11.5

    # -------------------------------------------------------------------------
    # Lateral Connectors descending from Left Encoder to Right Neck
    # -------------------------------------------------------------------------
    
    # 1) Stage 4 -> L5 (Level 5)
    ax.annotate("", xy=(lat_x-1.0, -7.75), xytext=(enc_x+1.5, -7.75),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_lateral"], lw=1.6))
    
    ax.add_patch(FancyBboxPatch((lat_x-1.0, -8.1), 2.2, 0.7, boxstyle="round,pad=0.08",
                                facecolor="#fef3c7", edgecolor=PALETTE["lateral"], linewidth=1.2, zorder=3))
    text(lat_x+0.1, -7.75, "Lateral L5\\n1×1 Conv", size=8.5, weight="bold", color="#92400e")
    
    ax.annotate("", xy=(neck_x-1.2, -7.75), xytext=(lat_x+1.3, -7.75),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_flow"], lw=1.6))
    
    ax.add_patch(FancyBboxPatch((neck_x-1.1, -8.1), 2.2, 0.7, boxstyle="round,pad=0.08",
                                facecolor="#eff6ff", edgecolor=PALETTE["upsample"], linewidth=1.2, zorder=3))
    text(neck_x, -7.75, shape_str(shapes["lat_c5"]), size=8.5, weight="bold", color=PALETTE["text_dark"])

    # Path from L5 to U4 (Upsample 2x)
    ax.annotate("", xy=(neck_x, -6.1), xytext=(neck_x, -7.3),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_flow"], lw=1.6))
    text(neck_x+0.1, -6.7, "Upsample U4\\nBilinear 2×", size=8.5, weight="bold", color=PALETTE["text_muted"], ha="left")

    # 2) Stage 3 -> L4 (Level 4)
    ax.annotate("", xy=(lat_x-1.0, -5.35), xytext=(enc_x+1.8, -5.35),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_lateral"], lw=1.6))
    
    ax.add_patch(FancyBboxPatch((lat_x-1.0, -5.7), 2.2, 0.7, boxstyle="round,pad=0.08",
                                facecolor="#fef3c7", edgecolor=PALETTE["lateral"], linewidth=1.2, zorder=3))
    text(lat_x+0.1, -5.35, "Lateral L4\\n1×1 Conv", size=8.5, weight="bold", color="#92400e")
    
    # Path from L4 to P4
    ax.annotate("", xy=(neck_x-1.3, -5.35), xytext=(lat_x+1.3, -5.35),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_lateral"], lw=1.6))

    # P4 Fusion Node
    ax.add_patch(FancyBboxPatch((neck_x-1.3, -5.9), 2.6, 1.1, boxstyle="round,pad=0.1",
                                facecolor=PALETTE["fusion_bg"], edgecolor=PALETTE["fusion"], linewidth=1.5, zorder=3))
    text(neck_x, -5.2, "Fusion (P4)" if not gated else "Scale-Gate (P4)", size=9, weight="bold", color="#166534")
    text(neck_x, -5.55, shape_str(shapes['refine_u4']), size=8.5, weight="bold", color="#166534")

    # Path from P4 to U3 (Upsample 2x)
    ax.annotate("", xy=(neck_x, -3.6), xytext=(neck_x, -4.7),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_flow"], lw=1.6))
    text(neck_x+0.1, -4.15, "Upsample U3\\nBilinear 2×", size=8.5, weight="bold", color=PALETTE["text_muted"], ha="left")

    # 3) Stage 2 -> L3 (Level 3)
    ax.annotate("", xy=(lat_x-1.0, -2.75), xytext=(enc_x+1.6, -2.75),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_lateral"], lw=1.6))
    
    ax.add_patch(FancyBboxPatch((lat_x-1.0, -3.1), 2.2, 0.7, boxstyle="round,pad=0.08",
                                facecolor="#fef3c7", edgecolor=PALETTE["lateral"], linewidth=1.2, zorder=3))
    text(lat_x+0.1, -2.75, "Lateral L3\\n1×1 Conv", size=8.5, weight="bold", color="#92400e")

    # Path from L3 to P3
    ax.annotate("", xy=(neck_x-1.3, -2.75), xytext=(lat_x+1.3, -2.75),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=PALETTE["edge_lateral"], lw=1.6))

    # P3 Fusion Node
    ax.add_patch(FancyBboxPatch((neck_x-1.3, -3.3), 2.6, 1.1, boxstyle="round,pad=0.1",
                                facecolor=PALETTE["fusion_bg"], edgecolor=PALETTE["fusion"], linewidth=1.5, zorder=3))
    text(neck_x, -2.6, "Fusion (P3)" if not gated else "Scale-Gate (P3)", size=9, weight="bold", color="#166534")
    text(neck_x, -2.95, shape_str(shapes['lat_c3']), size=8.5, weight="bold", color="#166534")

    # Output Conv
    ax.annotate("", xy=(neck_x, -1.0), xytext=(neck_x, -2.1),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=1.8))
    
    ax.add_patch(FancyBboxPatch((neck_x-1.3, -0.9), 2.6, 0.9, boxstyle="round,pad=0.08",
                                facecolor="#fff7ed", edgecolor="#ea580c", linewidth=1.3, zorder=3))
    text(neck_x, -0.45, "Output Conv 3×3", size=9.5, weight="bold", color="#9a3412")

    # Final Backbone Output Card
    ax.annotate("", xy=(neck_x, 1.6), xytext=(neck_x, 0.1),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge_flow"], lw=2.0))
    
    ax.add_patch(FancyBboxPatch((neck_x-2.0, 1.7), 4.0, 1.6, boxstyle="round,pad=0.1",
                                facecolor=PALETTE["output_bg"], edgecolor=PALETTE["output"], linewidth=1.8, zorder=3))
    text(neck_x, 2.85, "BACKBONE OUTPUT", size=11, weight="bold", color="#b91c1c")
    text(neck_x, 2.45, f"Shape: (1, {shape_str(report['output_shape'])})", size=9.2, weight="bold", color=PALETTE["text_dark"])
    text(neck_x, 2.1, "To Detection Heads", size=8.5, weight="bold", color="#c2410c")

"""

    content = content[:idx_start] + new_panels + content[idx_end:]
    
    tex_start_marker = '    # Section 1 Label\n    tex.append(r"\\node[anchor=west,font=\\sffamily\\bfseries\\normalsize] at (0.5,5.6) {1. BOTTOM-UP MULTI-SCALE 3D ENCODER};" + "\\n")'
    tex_end_marker = '    # Section 3: Inset Cards\n    tex.append(r"\\node[anchor=west,font=\\sffamily\\bfseries\\normalsize] at (0.5,-4.8) {3. CORE ARCHITECTURAL MODULES};" + "\\n")'

    idx_tex_start = content.find(tex_start_marker)
    idx_tex_end = content.find(tex_end_marker)
    
    if idx_tex_start != -1 and idx_tex_end != -1:
        new_tex = """    # Section 1 Label
    tex.append(r"\\node[anchor=west,font=\\sffamily\\bfseries\\normalsize] at (0.5,5.6) {1. BOTTOM-UP ENCODER (Left)};" + "\\n")
    tex.append(r"\\node[anchor=west,font=\\sffamily\\bfseries\\normalsize] at (12.5,5.6) {2. TOP-DOWN NECK (Right)};" + "\\n")

    # 3D Boxes using PlotNeuralNet primitives (YOLO layout)
    tex.append(r\"\"\"
\\pic[shift={(3.5,3.0,0)}] at (0,0,0) {
    Box={name=bev,caption=,xlabel={" "},zlabel=,fill=inputCol,height=8.0,width=1.6,depth=2.5}
};
\\node[anchor=east,align=right,font=\\sffamily\\scriptsize] at (1.5,3.0) {\\textbf{BEV Input}\\\\14 $\\times$ 800 $\\times$ 704};

\\pic[shift={(3.5,0.0,0)}] at (0,0,0) {
    Box={name=stem,caption=,xlabel={" "," "},zlabel=,fill=stemCol,height=6.5,width={1.6,1.6},depth=2.5}
};
\\node[anchor=east,align=right,font=\\sffamily\\scriptsize] at (1.5,0.0) {\\textbf{Stem (s=2)}\\\\32 $\\times$ 400 $\\times$ 352};
\\draw[flowArrow] (bev-south) -- (stem-north);
\"\"\")

    if eca:
        tex.append(r\"\"\"
\\pic[shift={(3.5,-3.0,0)}] at (0,0,0) {
    RightBandedBox={name=stage2,caption=,xlabel={" "," "},zlabel=,fill=stage2Col,bandfill=attnCol,height=5.0,width=4.0,depth=2.5}
};
\"\"\")
    else:
        tex.append(r\"\"\"
\\pic[shift={(3.5,-3.0,0)}] at (0,0,0) {
    Box={name=stage2,caption=,xlabel={" "},zlabel=,fill=stage2Col,height=5.0,width=4.0,depth=2.5}
};
\"\"\")
    tex.append(r\"\"\"
\\node[anchor=east,align=right,font=\\sffamily\\scriptsize] at (1.5,-3.0) {\\textbf{Stage 2 (s=4)}\\\\48 $\\times$ 200 $\\times$ 176};
\\draw[flowArrow] (stem-south) -- (stage2-north);
\"\"\")

    if focal:
        tex.append(r\"\"\"
\\pic[shift={(3.5,-6.0,0)}] at (0,0,0) {
    RightBandedBox={name=stage3,caption=,xlabel={" "," "},zlabel=,fill=stage3Col,bandfill=attnCol,height=3.5,width=5.5,depth=2.5}
};
\"\"\")
    else:
        tex.append(r\"\"\"
\\pic[shift={(3.5,-6.0,0)}] at (0,0,0) {
    Box={name=stage3,caption=,xlabel={" "},zlabel=,fill=stage3Col,height=3.5,width=5.5,depth=2.5}
};
\"\"\")
    tex.append(r\"\"\"
\\node[anchor=east,align=right,font=\\sffamily\\scriptsize] at (1.5,-6.0) {\\textbf{Stage 3 (s=8)}\\\\96 $\\times$ 100 $\\times$ 88};
\\draw[flowArrow] (stage2-south) -- (stage3-north);
\"\"\")

    tex.append(r\"\"\"
\\pic[shift={(3.5,-8.5,0)}] at (0,0,0) {
    Box={name=stage4,caption=,xlabel={" "," "},zlabel=,fill=stage4Col,height=2.0,width={2.5,2.5},depth=2.5}
};
\\node[anchor=east,align=right,font=\\sffamily\\scriptsize] at (1.5,-8.5) {\\textbf{Stage 4 (s=16)}\\\\128 $\\times$ 50 $\\times$ 44};
\\draw[flowArrow] (stage3-south) -- (stage4-north);
\"\"\")

    tex.append(r\"\"\"
% 2D Flow Nodes for FPN (Right Column)
\\node[blockNode,fill=latCol,draw=latBorder] (lat5) at (10.0,-8.5) {L5 (1x1)};
\\node[blockNode,fill=upCol,draw=upBorder] (u5) at (14.0,-8.5) {48 $\\times$ 50 $\\times$ 44};

\\node[blockNode,fill=upCol,draw=upBorder] (u4_up) at (14.0,-7.25) {Upsample 2x};
\\node[blockNode,fill=latCol,draw=latBorder] (lat4) at (10.0,-6.0) {L4 (1x1)};
\\node[blockNode,fill=fuseCol,draw=fuseBorder] (p4) at (14.0,-6.0) {Fusion P4};

\\node[blockNode,fill=upCol,draw=upBorder] (u3_up) at (14.0,-4.5) {Upsample 2x};
\\node[blockNode,fill=latCol,draw=latBorder] (lat3) at (10.0,-3.0) {L3 (1x1)};
\\node[blockNode,fill=fuseCol,draw=fuseBorder] (p3) at (14.0,-3.0) {Fusion P3};

\\node[blockNode,fill=latCol,draw=latBorder] (outconv) at (14.0,-1.0) {Out Conv 3x3};
\\node[blockNode,fill=outCol,draw=outBorder] (out) at (14.0,1.0) {\\textbf{BACKBONE OUTPUT}};

% Connector Arrows (Left to Right)
\\draw[latArrow] (stage4-east) -- (lat5.west);
\\draw[flowArrow] (lat5.east) -- (u5.west);

\\draw[latArrow] (stage3-east) -- (lat4.west);
\\draw[flowArrow] (lat4.east) -- (p4.west);

\\draw[latArrow] (stage2-east) -- (lat3.west);
\\draw[flowArrow] (lat3.east) -- (p3.west);

% Connector Arrows (Bottom to Top)
\\draw[flowArrow] (u5.north) -- (u4_up.south);
\\draw[flowArrow] (u4_up.north) -- (p4.south);

\\draw[flowArrow] (p4.north) -- (u3_up.south);
\\draw[flowArrow] (u3_up.north) -- (p3.south);

\\draw[flowArrow] (p3.north) -- (outconv.south);
\\draw[flowArrow] (outconv.north) -- (out.south);

"""
        content = content[:idx_tex_start] + new_tex + content[idx_tex_end:]

    path.write_text(content)
    print("Patch applied.")

if __name__ == "__main__":
    patch_file()
