"""Direct notebook commands and readable training output."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "3D_Lidar_Object_Detection_Notebook_standard.ipynb"
sys.path[:0] = [str(ROOT), str(ROOT / "tools/kitti_training_pipeline")]


def test_notebook_runs_commands_directly_without_a_python_process_wrapper():
    notebook = json.loads(NOTEBOOK.read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "run_command" not in code and "subprocess" not in code
    assert "!" in code and " -u " in code
    assert "require_training_ready" not in code
    assert "freeze_protocol" not in code


def test_train_losses_are_logged_before_ap_and_ap_results_name_each_class(tmp_path, capsys):
    import train
    from test_ap_training import setup

    _, _, _, run, args = setup(tmp_path, epochs=1, interval=1)
    train.main(args)
    output = capsys.readouterr().out
    assert output.index("Train Loss:") < output.index("AP validation:")
    assert "Val Loss:" in output and "LR:" in output and "Time:" in output
    ap_summary = next(line for line in output.splitlines() if "Validation AP:" in line)
    for name in ("Car", "Pedestrian", "Cyclist"):
        assert name in ap_summary
    assert "R40 Moderate" in ap_summary
    log = (run / "train.log").read_text()
    assert log.index("Train Loss:") < log.index("AP validation:")
    assert (run / "tensorboard").is_dir()
    assert any((run / "tensorboard").glob("events.out.tfevents.*"))
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(run / "tensorboard"))
        ea.Reload()
        tags = set(ea.Tags().get("scalars", []))
        assert {"train/loss", "val/loss", "train/learning_rate", "val_ap/score_r40_moderate"} <= tags
    except ImportError:
        pass


def test_omitted_runtime_flags_do_not_overwrite_saved_notebook_settings():
    import train

    args = train.build_parser().parse_args([
        "--config", "config.json", "--detector-root", "detector", "--output-root", "artifacts",
    ])
    assert args.num_workers is None
    assert args.target_backend is None
    assert args.compile_model is None


def test_direct_training_uses_runtime_settings_from_config(tmp_path):
    import train
    from test_ap_training import setup, load

    config, path, _, run, args = setup(tmp_path, primary="loss", epochs=1)
    config["train"].update(num_workers=0, target_backend="numba", compile_model=False)
    path.write_text(json.dumps(config))
    for flag in ("--num-workers", "--target-backend"):
        index = args.index(flag)
        del args[index:index + 2]
    train.main(args)
    saved = load(run / "checkpoints/last.pt")["config"]["train"]
    assert saved["num_workers"] == 0
    assert saved["target_backend"] == "numba"
    assert saved["compile_model"] is False


def run_notebook_training(tmp_path, *, box_mode="bev", device="cpu"):
    import hashlib
    import math
    import shlex
    import tempfile
    import torch
    from benchmark_fixtures import asset_fixture
    from common import read_json, write_json
    from notebook_test_utils import notebook_cell, execute_shell_cell
    from test_under1m_notebook_controls import execute_cell

    config, _, processed, raw = asset_fixture(tmp_path / "assets", box_mode=box_mode)
    namespace = dict(REPO_DIR=ROOT, ARTIFACT_ROOT=tmp_path / "runs with spaces",
        PROCESSED_DATASET_DIR=processed, RAW_KITTI_ROOT=raw.parent,
        SEED=42, EPOCHS=2, WARMUP_EPOCHS=0, LEARNING_RATE=.0007, NUM_WORKERS=0,
        PHYSICAL_BATCH_SIZE=1, ACCUMULATION_STEPS=1, VAL_BATCH_SIZE=1,
        PRECISION="fp32", TARGET_BACKEND="python", COMPILE_MODEL=False,
        torch=torch, Path=Path, json=json, math=math, sys=sys, hashlib=hashlib,
        q=shlex.quote, tempfile=tempfile, read_json=read_json, write_json=write_json,
        DEVICE=device, BRANCH="research/mobilepixornext-under1m", COMMIT="synthetic-test",
        WARM_START_PATH=None, ALLOW_LEGACY_RESUME=False, RUN_SMOKE_TEST=True,
        SMOKE_TRAIN_BATCHES=1, SMOKE_VAL_BATCHES=1, RUN_EVALUATION=True)
    modes = ["local_bev"] if box_mode == "bev" else ["3d", "bev"]
    execute_cell(tmp_path, {"BOX_MODE": box_mode, "EVALUATION_MODES": modes,
        "TRAIN_SPLIT": config["train"]["data"], "VAL_SPLIT": config["val"]["data"],
        "GEOMETRY_OVERRIDES": config["data"]["kitti"]["geometry"],
        "CUSTOM_RUN_NAME": "notebook_synthetic"}, namespace)
    for marker in ("SMOKE_ROOT =", "RUN_METADATA_PATH =", "comparison.to_csv"):
        execute_shell_cell(notebook_cell(marker), namespace)
    run = namespace["RUN_DIR"]
    before = (run / "metrics.jsonl").read_bytes()
    execute_shell_cell(notebook_cell("RUN_METADATA_PATH ="), namespace)
    assert (run / "metrics.jsonl").read_bytes() == before
    assert json.loads((run / "config.resolved.json").read_text()) == namespace["config_dict"]
    return run


def test_actual_notebook_smoke_train_evaluate_and_completed_resume(tmp_path):
    run = run_notebook_training(tmp_path)
    comparison = json.loads((run / "comparison.json").read_text())
    assert {r["checkpoint_selection"] for r in comparison["rows"]} == {"ap", "loss"}
    assert (run / "selected/best_ap.pt").is_file() and (run / "selected/best_loss.pt").is_file()
    assert (run / "tensorboard").is_dir() and any((run / "tensorboard").glob("events.out.tfevents.*"))
