import sys
import json
import subprocess
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT if (REPO_ROOT / "configs/config.json").exists() else REPO_ROOT / "Code"

# Setup sys.path for detector modules
sys.path.insert(0, str(CODE_DIR / "tools/kitti_training_pipeline"))
sys.path.insert(0, str(CODE_DIR / "detector"))
sys.path.insert(0, str(CODE_DIR))

from common import configure_detector_imports, build_model, input_shape
configure_detector_imports(CODE_DIR / "detector")
from core.losses.loss_fn import build_loss_function


def test_config_validity():
    """Verify primary configuration file exists and has expected top-level sections."""
    cfg_path = CODE_DIR / "configs/config.json"
    assert cfg_path.exists(), f"Missing config file at {cfg_path}"
    with open(cfg_path) as f:
        cfg = json.load(f)
    for required in ["model", "train", "data", "loss", "augmentation"]:
        assert required in cfg, f"Key '{required}' missing in {cfg_path}"


def test_model_forward_pass():
    """Verify model can be built and executes a forward pass on dummy BEV tensor."""
    with open(CODE_DIR / "configs/config.json") as f:
        cfg = json.load(f)
    model = build_model(cfg).eval()
    shape = input_shape(cfg)
    dummy_input = torch.zeros(shape, dtype=torch.float32)

    with torch.no_grad():
        out = model(dummy_input)

    for head in ["cls", "offset", "size", "yaw"]:
        assert head in out, f"Output head '{head}' missing in model predictions"
        assert out[head].shape[0] == shape[0], f"Batch size mismatch in {head}"


@pytest.mark.parametrize("loss_name", ["baseline", "oga", "q_oga"])
def test_loss_function_instantiation(loss_name):
    """Verify loss function factory successfully constructs criteria for all supported losses."""
    with open(CODE_DIR / "configs/config.json") as f:
        cfg = json.load(f)
    loss_cfg = dict(cfg.get("loss", {}))
    loss_cfg["name"] = loss_name
    criterion = build_loss_function("one_hot", loss_cfg, box_mode="bev")
    assert criterion is not None, f"Failed to instantiate {loss_name}"


@pytest.mark.parametrize("script_name", [
    "train.py",
    "evaluate_kitti_bev.py",
    "export_onnx.py",
    "checkpoint_selection.py",
])
def test_pipeline_cli_help(script_name):
    """Verify training & evaluation CLI tools load without import/syntax errors."""
    script_path = CODE_DIR / f"tools/kitti_training_pipeline/{script_name}"
    assert script_path.exists(), f"Script {script_path} not found"
    res = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Script {script_name} failed with: {res.stderr}"
