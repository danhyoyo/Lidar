"""Portable editable figure exports use the same renderer as the browser."""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.visualization.export_backbone_diagram import build_diagram
from tools.visualization.export_reference_figure import export_reference

pytestmark = pytest.mark.skipif(not shutil.which("node") or not shutil.which("rsvg-convert"),
                                reason="reference export requires Node.js and librsvg")


def test_config_exports_reference_layout_and_lossless_editable_document(tmp_path):
    config = ROOT / "configs/config.json"
    document = export_reference(tmp_path, config_path=config, dpi=100)
    source = build_diagram(json.loads(config.read_text()))
    for name, graph in source["views"].items():
        assert [n["shape"] for n in document["views"][name]["nodes"]] == [n["shape"] for n in graph["nodes"]]
        assert [(e["source"], e["target"]) for e in document["views"][name]["edges"]] == [(e["source"], e["target"]) for e in graph["edges"]]
    svg = (tmp_path / "backbone.svg").read_text()
    assert 'stroke-dasharray="7 5"' in svg
    assert "48 → 24" in svg and "128 → 48" in svg
    assert len(list((tmp_path / "details").glob("*.svg"))) == len(source["views"]) - 1
    assert (tmp_path / "backbone.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path / "backbone.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["input_kind"] == "model_config"
    assert provenance["source_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    # Saved geometry and labels reproduce the exact same SVG after JSON import.
    restored = export_reference(tmp_path / "restored", document_path=tmp_path / "backbone.editor.json", dpi=100)
    assert restored == document
    assert (tmp_path / "restored/backbone.svg").read_bytes() == (tmp_path / "backbone.svg").read_bytes()


def test_custom_document_changes_geometry_labels_topology_and_shapes(tmp_path):
    original = export_reference(tmp_path / "original", dpi=100)
    custom = copy.deepcopy(original)
    custom["shape_mode"] = "manual drawing"
    custom["editor"]["modified"] = True
    graph = custom["views"]["overview"]
    block = next(n for n in graph["nodes"] if n["id"] == "stage2")
    block.update(label="My encoder", caption="My encoder", shape=[64,200,176], repeat=5)
    block["layout"].update(x=470,y=340,width=250,height=150,fill="#abcdee")
    graph["nodes"].append(dict(id="my_layer",label="My layer",kind="conv",shape=[17,40,50],col=8,row=1,
                                layout=dict(x=1100,y=380,width=200,height=90,fill="#ffeecc",font_size=18)))
    graph["edges"].append(dict(id="my_edge",source="stage2",target="my_layer",kind="skip",label="custom branch",
                                source_port="right",target_port="left",waypoints=[dict(x=790,y=720)]))
    file = tmp_path / "custom.json"
    file.write_text(json.dumps(custom))
    result = export_reference(tmp_path / "custom", document_path=file, dpi=100)
    rendered = (tmp_path / "custom/backbone.svg").read_text()
    assert "My encoder" in rendered and "NeXt × 5" in rendered
    assert "64 → 24" in rendered and "17 × 40 × 50" in rendered
    assert '#abcdee' in rendered and 'data-source="stage2" data-target="my_layer"' in rendered
    assert "790 720" in rendered
    assert next(n for n in result["views"]["overview"]["nodes"] if n["id"] == "stage2")["layout"] == block["layout"]
    assert "architecture_source_sha256" not in json.loads((tmp_path / "custom/provenance.json").read_text())


def test_cli_reports_invalid_layout_without_rendering_it(tmp_path):
    original = export_reference(tmp_path / "original", dpi=100)
    original["views"]["overview"]["nodes"][0]["layout"]["width"] = -100
    file = tmp_path / "invalid.json"
    file.write_text(json.dumps(original))
    result = subprocess.run(["node", str(ROOT / "tools/visualization/nn_svg/export-reference.cjs"),
                             str(file), str(tmp_path / "invalid")], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Invalid layout" in result.stderr
    assert not (tmp_path / "invalid/backbone.svg").exists()
