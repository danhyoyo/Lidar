"""Publication export and conditional topology checks, without model execution."""
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.visualization.export_backbone_diagram import build_diagram
from tools.visualization.export_paper_figure import export_figure

SVG = {"s": "http://www.w3.org/2000/svg"}


@pytest.mark.skipif(not shutil.which("node") or not shutil.which("rsvg-convert"),
                    reason="paper export requires Node.js and librsvg")
def test_export_embeds_vector_pdf_and_records_exact_config(tmp_path):
    config = ROOT / "configs/config.json"
    diagram = export_figure(config, tmp_path, dpi=120, details=True)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert provenance["svg_sha256"] == hashlib.sha256((tmp_path / "backbone.svg").read_bytes()).hexdigest()
    assert provenance["width_mm"] == 182
    assert provenance["png_dpi"] == 120
    assert (tmp_path / "backbone.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path / "backbone.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(list((tmp_path / "details").glob("*.svg"))) == len(diagram["views"]) - 1
    assert json.loads((tmp_path / "architecture.json").read_text()) == diagram
    if shutil.which("pdfimages"):
        result = subprocess.run(["pdfimages", "-list", str(tmp_path / "backbone.pdf")],
                                check=True, capture_output=True, text=True).stdout
        assert not any(line.strip()[:1].isdigit() for line in result.splitlines())
    if shutil.which("pdftotext"):
        text = subprocess.run(["pdftotext", str(tmp_path / "backbone.pdf"), "-"],
                              check=True, capture_output=True, text=True).stdout
        assert "Scale-gated fusion" in text
        assert "120 → 48" in text


@pytest.mark.skipif(not shutil.which("node"), reason="SVG renderer requires Node.js")
@pytest.mark.parametrize("variant", ["focal_eca_detail", "rep_training", "rep_deploy", "sum_only"])
def test_publication_renderer_tracks_resolved_options(tmp_path, variant):
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["model"]["expansion"] = 3
    if variant == "focal_eca_detail":
        config["data"]["bev_encoding"] = {"name": "hist14"}
        config["model"].update(stage_depths=[3, 4, 2], backbone_out_dim=32,
                              neck_fusion_channels=32, c4_context="focal",
                              local_attention="eca", detail_path=True)
    elif variant.startswith("rep_"):
        config["model"].update(use_reparam=True, deploy=variant == "rep_deploy")
    else:
        config["model"]["scale_gated_fpn"] = False
    bundle = build_diagram(config)
    input_file = tmp_path / "architecture.json"
    input_file.write_text(json.dumps(bundle))
    subprocess.run(["node", str(ROOT / "tools/visualization/nn_svg/export-svg.cjs"),
                    str(input_file), str(tmp_path / "svg"), "--style", "paper"], check=True)
    svg = ET.parse(tmp_path / "svg/overview.svg").getroot()
    assert svg.get("width") == "182mm"
    text = " ".join(svg.itertext())
    assert "48 → 144" in text
    ids = {g.get("data-node") for g in svg.findall(".//s:g", SVG)}
    assert {n["id"] for n in bundle["views"]["overview"]["nodes"]} <= ids
    actual_edges = {(p.get("data-source"), p.get("data-target")) for p in svg.findall(".//s:path", SVG)}
    assert {(e["source"], e["target"]) for e in bundle["views"]["overview"]["edges"]} <= actual_edges
    if variant == "focal_eca_detail":
        assert "14 × 800 × 704" in text and "32 × 200 × 176" in text
        assert "FocalContext" in text and "Residual ECA" in text
    elif variant == "rep_training":
        assert "3 branches" in text and "RepDW" in text
    elif variant == "rep_deploy":
        assert "fused" in text and "3 branches" not in text
    else:
        p4 = svg.find(".//s:g[@data-node='p4']/s:text", SVG)
        assert p4.text == "+"
        assert "Additive fusion" in text and "2σ" not in text
