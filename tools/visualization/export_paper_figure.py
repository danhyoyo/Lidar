#!/usr/bin/env python3
"""Export a two-column 2D backbone figure as editable SVG, vector PDF and PNG.

Requires Node.js and rsvg-convert (librsvg). No model weights, GPU or PyTorch.
Uses the same configuration adapter and graph as the NN-SVG viewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.visualization.export_backbone_diagram import VIEWER, SOURCE_FILES, build_diagram, export_diagram


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_figure(config_path, output_dir, dpi=600, details=False):
    for executable in ("node", "rsvg-convert"):
        if not shutil.which(executable):
            raise ValueError(f"{executable} is required to export SVG/PDF/PNG")
    if not isinstance(dpi, int) or not 72 <= dpi <= 1200:
        raise ValueError("dpi must be an integer between 72 and 1200")
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir)
    source = str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else config_path.name
    diagram = build_diagram(json.loads(config_path.read_text(encoding="utf-8")), source=source)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lidar-paper-figure-") as temp:
        temp = Path(temp)
        export_diagram(diagram, temp)
        subprocess.run(["node", str(VIEWER / "export-svg.cjs"), str(temp / "mobilepixornext.json"),
                        str(temp / "svg"), "--style", "paper"], check=True)
        svg = output_dir / "backbone.svg"
        shutil.copyfile(temp / "svg/overview.svg", svg)
        if details:
            detail_dir = output_dir / "details"
            detail_dir.mkdir(exist_ok=True)
            for file in sorted((temp / "svg").glob("*.svg")):
                if file.name != "overview.svg":
                    shutil.copyfile(file, detail_dir / file.name)
    subprocess.run(["rsvg-convert", "--format=pdf", "--output", str(output_dir / "backbone.pdf"),
                    str(svg)], check=True)
    subprocess.run(["rsvg-convert", "--dpi-x", str(dpi), "--dpi-y", str(dpi),
                    "--output", str(output_dir / "backbone.png"), str(svg)], check=True)
    (output_dir / "architecture.json").write_text(
        json.dumps(diagram, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    provenance = {
        "config": source,
        "config_sha256": sha256(config_path),
        "shape_mode": diagram["shape_mode"],
        "input_encoding": diagram["input_encoding"],
        "options": diagram["options"],
        "width_mm": 182,
        "png_dpi": dpi,
        "svg_sha256": sha256(svg),
        "source_sha256": {path: sha256(ROOT / path) for path in SOURCE_FILES},
        "renderer_sha256": {path: sha256(VIEWER / path) for path in ("Backbone.js", "Paper.js")},
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return diagram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagrams/paper_2d")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution; PDF and SVG remain vector")
    parser.add_argument("--details", action="store_true", help="Also export the expanded primitive views as SVG")
    args = parser.parse_args()
    try:
        diagram = export_figure(args.config, args.output_dir, args.dpi, args.details)
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Cannot export paper figure: {exc}\n")
    print(f"Exported {diagram['name']} paper SVG/PDF/{args.dpi}-dpi PNG to {args.output_dir}")


if __name__ == "__main__":
    main()
