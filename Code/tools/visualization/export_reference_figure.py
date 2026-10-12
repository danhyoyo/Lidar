#!/usr/bin/env python3
"""Export an editable paper-style backbone from a model config or custom JSON.

Reference style follows diagrams/other_paper*.png. Requires Node.js and librsvg.
Custom drawing imports use their saved geometry and topology as supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.visualization.export_backbone_diagram import VIEWER, SOURCE_FILES, build_diagram


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_reference(output_dir, config_path=None, document_path=None, dpi=600):
    if config_path is not None and document_path is not None:
        raise ValueError("choose either config_path or document_path")
    if not isinstance(dpi, int) or not 72 <= dpi <= 1200:
        raise ValueError("dpi must be an integer between 72 and 1200")
    for executable in ("node", "rsvg-convert"):
        if not shutil.which(executable):
            raise ValueError(f"{executable} is required")
    source = Path(document_path or config_path or ROOT / "configs/config.json").resolve()
    source_name = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name
    content = json.loads(source.read_text(encoding="utf-8"))
    custom = document_path is not None
    diagram = content if custom else build_diagram(content, source=source_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lidar-reference-figure-") as folder:
        payload = Path(folder) / "document.json"
        payload.write_text(json.dumps(diagram, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        subprocess.run(["node", str(VIEWER / "export-reference.cjs"), str(payload), str(output)], check=True)
    svg = output / "backbone.svg"
    subprocess.run(["rsvg-convert", "--format=pdf", "--output", str(output / "backbone.pdf"), str(svg)], check=True)
    subprocess.run(["rsvg-convert", "--dpi-x", str(dpi), "--dpi-y", str(dpi),
                    "--output", str(output / "backbone.png"), str(svg)], check=True)
    document = json.loads((output / "backbone.editor.json").read_text(encoding="utf-8"))
    provenance = {
        "input_kind": "editable_document" if custom else "model_config",
        "source": source_name,
        "source_sha256": sha256(source),
        "shape_mode": document["shape_mode"],
        "width_mm": 182,
        "png_dpi": dpi,
        "svg_sha256": sha256(svg),
        "document_sha256": sha256(output / "backbone.editor.json"),
        "renderer_sha256": {name: sha256(VIEWER / name) for name in ("Backbone.js", "Editor.js", "Reference.js")},
    }
    if not custom:
        provenance["architecture_source_sha256"] = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument("--document", type=Path, help="JSON downloaded from the editor")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagrams/reference_2d")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    try:
        document = export_reference(args.output_dir, args.config, args.document, args.dpi)
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Cannot export reference figure: {exc}\n")
    print(f"Exported {document['name']} editable JSON/SVG/vector PDF/{args.dpi}-dpi PNG to {args.output_dir}")


if __name__ == "__main__":
    main()
