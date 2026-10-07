"""Notebook import setup must work in a fresh kernel, before any model imports."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "3D_Lidar_Object_Detection_Notebook_standard.ipynb"


def notebook_bootstrap():
    sources = ["".join(item["source"]) for item in json.loads(NOTEBOOK.read_text())["cells"]
               if item["cell_type"] == "code"]
    source = next(source for source in sources if 'os.chdir(REPO_DIR)' in source)
    # Only kernel path setup remains; models are built by direct CLI commands.
    return source.split('os.chdir(REPO_DIR)', 1)[1]


def test_notebook_exposes_detector_and_legacy_utils_in_a_fresh_process(tmp_path):
    script = """
import importlib.util
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "tools/kitti_training_pipeline")]
assert importlib.util.find_spec("core") is None
assert importlib.util.find_spec("utils_1") is None
namespace = {"REPO_DIR": root, "sys": sys}
exec(sys.argv[2], namespace)
import core.backbone_config
import core.bev_encoding
import utils_1
assert Path(core.backbone_config.__file__).resolve() == root / "detector/core/backbone_config.py"
assert root / "detector/core/datasets/utils_1" in map(Path, utils_1.__path__)
assert not {"torch", "numpy", "numba"} & set(sys.modules)
for path in (root / "detector", root / "detector/core/datasets"):
    assert sys.path.count(str(path)) == 1
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT), notebook_bootstrap()],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_notebook_reports_incomplete_detector_source_before_model_import(tmp_path):
    script = """
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "tools/kitti_training_pipeline")]
namespace = {"REPO_DIR": Path(sys.argv[2]), "sys": sys}
exec(sys.argv[3], namespace)
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT), str(tmp_path),
                             notebook_bootstrap()], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Detector source is incomplete" in result.stderr
    assert "No module named 'core'" not in result.stderr
