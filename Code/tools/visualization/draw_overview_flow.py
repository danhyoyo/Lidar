#!/usr/bin/env python3
"""Draw the compact Figure 1 middle panel from the existing inference metadata.

No generated point clouds or heatmaps. SVG text remains editable; raster and
PDF exports use librsvg. Run from any working directory.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
INK = "#252525"
ORANGE = "#dd8946"


def draw(metadata_path: Path, output: Path, heatmap_path: Path) -> None:
    metadata = json.loads(metadata_path.read_text())
    inference = metadata["inference"]
    identity = inference["checkpoint_identity"]
    features = identity["model"]["features"]["config"]
    groups = identity["groups"]
    # This is a source-verified figure preset, not a generic architecture claim.
    expected = {
        "encoding": inference["encoding"] == "hist14",
        "BEV outputs": inference["box_mode"] == "bev",
        "IoU branch": inference["use_iou"] is True,
        "backbone": identity["model"]["backbone"] == "mobilepixornext",
        "Focal Context": features["c4_context"] == "focal",
        "SG-FPN": identity["model"]["neck_type"] == "scale_gated_fpn",
        "groups": [g["classes"] for g in groups] == [["Car"], ["Pedestrian", "Cyclist"]],
        "classification": identity["model"]["cls_encoding"] == "gaussian",
    }
    if not all(expected.values()):
        raise ValueError(f"Unsupported figure preset: {[k for k, v in expected.items() if not v]}")
    output.mkdir(parents=True, exist_ok=True)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="95mm" height="76.3mm" '
        'viewBox="0 0 660 530">',
        '<title>Compact overview of the Hist14 MobilePixorNeXt detector</title>',
        '<desc>LiDAR to Hist14 BEV encoding, MobilePixorNeXt with Focal Context '
        'and SG-FPN (collapsed into the detector box), grouped heatmap, metric '
        'offset, log-size, doubled-angle and IoU-quality heads. The right arrow '
        'denotes BEV detection after decoding and NMS. Five downward arrows '
        'denote OGA training supervision. Any 3D cuboids require subsequent '
        'LiDAR-based vertical lifting outside this panel.</desc>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>'
        '<marker id="orange-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ORANGE}"/></marker></defs>',
        '<g font-family="Times New Roman, Liberation Serif, DejaVu Serif, serif" '
        f'fill="{INK}" stroke-linejoin="round" stroke-linecap="round">',
    ]

    def box(x, y, w, h, fill="white", dashed=False, radius=15):
        dash = ' stroke-dasharray="9 7"' if dashed else ""
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                     f'rx="{radius}" fill="{fill}" stroke="{INK}" stroke-width="2"{dash}/>')

    def text(x, y, label, size=24, bold=False, color=INK, anchor="middle"):
        weight = ' font-weight="600"' if bold else ""
        parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                     f'font-size="{size}" fill="{color}"{weight}>{escape(label)}</text>')

    def path(points, arrow=False, training=False, orange=False):
        color = ORANGE if orange else ("#777777" if training else INK)
        marker = ' marker-end="url(#orange-arrow)"' if arrow and orange else (
            ' marker-end="url(#arrow)"' if arrow else "")
        dash = ' stroke-dasharray="7 6"' if training else ""
        coords = " L ".join(f"{x},{y}" for x, y in points)
        parts.append(f'<path d="M {coords}" fill="none" stroke="{color}" '
                     f'stroke-width="{3 if orange else 2}"{marker}{dash}/>')

    def block_arrow(x, y, length=32, vertical=False):
        points = [(0, -7), (length-13, -7), (length-13, -14),
                  (length, 0), (length-13, 14), (length-13, 7), (0, 7)]
        if vertical:
            points = [(-dy, dx) for dx, dy in points]
        coords = " ".join(f"{x+dx},{y+dy}" for dx, dy in points)
        parts.append(f'<polygon points="{coords}" fill="#eea05b" '
                     f'stroke="{INK}" stroke-width="1.8"/>')

    # Match the reference: two upper blocks, one aligned output row, one loss bar.
    box(18, 12, 622, 506, fill="none", dashed=True, radius=12)
    box(53, 32, 172, 124, radius=22)
    text(139, 70, "Hist14", 27)
    text(139, 106, "BEV Feature", 27)
    text(139, 142, "Encoder", 27)
    box(284, 32, 302, 124, fill="#fce8da", radius=22)
    text(435, 70, "Proposed", 27)
    text(435, 106, "(MobilePixorNeXt)", 27)
    text(435, 142, "Detector", 27)
    block_arrow(0, 95, length=51)
    block_arrow(227, 95, length=55)
    block_arrow(435, 158, length=35, vertical=True)

    box(28, 194, 598, 248, fill="#f8f8f8", dashed=True, radius=9)
    centers = [92, 200, 300, 427, 561]
    path([(435, 194), (435, 204)])
    path([(92, 213), (92, 204), (561, 204), (561, 213)])
    for cx in centers[1:-1]:
        path([(cx, 204), (cx, 213)])
    specs = [
        (38, 108, ("Gaussian", "Heatmap"), "#ffe7a0"),
        (154, 92, ("Local", "Offset"), "#f5f5f5"),
        (254, 92, ("Object", "Size"), "#f5f5f5"),
        (354, 146, ("Orientation",), "#f5f5f5"),
        (508, 106, ("IoU", "Quality"), "#f5f5f5"),
    ]
    for cx, (x, width, lines, fill) in zip(centers, specs):
        box(x, 214, width, 68, fill=fill, radius=12)
        if len(lines) == 1:
            text(cx, 256, lines[0], 25)
        else:
            text(cx, 241, lines[0], 25)
            text(cx, 269, lines[1], 25)

    # Embed the user's existing prediction heatmap byte-for-byte, without edits.
    image_data = base64.b64encode(heatmap_path.read_bytes()).decode("ascii")
    image_element = (f'<image x="41" y="292" width="102" height="136" '
                     f'preserveAspectRatio="xMidYMid meet" '
                     f'href="data:image/png;base64,{image_data}"/>')
    parts.append(image_element)
    text(200, 320, "[Δx, Δy]", 25)
    text(300, 311, "[log w,", 25)
    text(300, 342, "log l]", 25)
    text(427, 311, "[cos 2θ,", 25)
    text(427, 342, "sin 2θ]", 25)
    text(561, 320, "σ(i)", 25)
    for cx, start in zip(centers, [417, 333, 355, 355, 333]):
        path([(cx, start), (cx, 467)], arrow=True)
    block_arrow(627, 348, length=32)
    box(28, 469, 598, 39, fill="#e7f2fc", radius=9)
    text(327, 498, "OGA Loss", 29)
    parts.extend(["</g>", "</svg>"])
    svg = output / "overview_flow.svg"
    svg.write_text("\n".join(parts) + "\n", encoding="utf-8")
    # A second source lets the user assemble their own heatmap in the blank slot.
    without_heatmap = output / "overview_flow_no_heatmap.svg"
    without_heatmap.write_text("\n".join(p for p in parts if p != image_element) + "\n")
    for name, options in (
        ("overview_flow.png", ["--background-color=white", "--dpi-x=600", "--dpi-y=600"]),
        ("overview_flow_transparent.png", ["--dpi-x=600", "--dpi-y=600"]),
        ("overview_flow.pdf", ["--format=pdf"]),
    ):
        subprocess.run(["rsvg-convert", *options, "--output", str(output / name), str(svg)], check=True)
    subprocess.run(["rsvg-convert", "--format=pdf", "--output",
                    str(output / "overview_flow_no_heatmap.pdf"), str(without_heatmap)], check=True)
    provenance = {
        "metadata_source": str(metadata_path.relative_to(REPO)) if metadata_path.is_relative_to(REPO) else str(metadata_path),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": inference["checkpoint_sha256"],
        "preset_checks": expected,
        "box_mode": "bev",
        "vertical_geometry": "Postprocessing: local ground and measured point heights; no learned vertical head",
        "offset_units": "Metric residuals in meters, not fractional cell offsets",
        "loss_source": "Code/detector/core/losses/strategies/oga.py",
        "png_dpi": 600,
        "width_mm": 95,
        "height_mm": 76.3,
        "layout": "Reference-matched compact two-block / five-output / loss-bar panel",
        "collapsed_detector_modules": ["Backbone", "Focal Context", "SG-FPN"],
        "inference_arrow": "Decoded BEV boxes after IoU-aware scoring and NMS",
        "heatmap_source": str(heatmap_path),
        "heatmap_sha256": hashlib.sha256(heatmap_path.read_bytes()).hexdigest(),
        "heatmap_processing": "Original PNG embedded byte-for-byte; no image modification",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported SVG, white/transparent PNG and vector PDF to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REPO / "Demo/outputs/bev/005663/metadata.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagrams/figure1_flow")
    parser.add_argument("--heatmap", type=Path,
                        default=REPO / "Demo/outputs/heatmap/005663/gaussian_heatmap.png")
    args = parser.parse_args()
    draw(args.metadata.resolve(), args.output_dir, args.heatmap.resolve())


if __name__ == "__main__":
    main()
