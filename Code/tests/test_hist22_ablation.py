"""The runnable pair freezes every user-selected condition outside encoder metadata."""
import copy
import hashlib
import json
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/kitti_training_pipeline")]
from common import bev_encoding_spec
from notebook_config import resolve_notebook_config


@pytest.fixture
def baseline(tmp_path):
    config = resolve_notebook_config(ROOT, preset="custom", augmentation="hybrid_gt",
        custom_overrides={"data": {"bev_encoding": {"name": "hist14", "backend": "numba",
            "density_norm": 47, "intensity_scale": 2}}}, runtime_overrides={"seed": 42,
            "train": {"epochs": 17, "warmup_epochs": 2, "checkpoint_selection": {"primary": "ap", "ap_every": 3}}})
    path = tmp_path / "user baseline.json"
    path.write_text(json.dumps(config))
    return path, config


def test_prepare_exact_pair_freezes_user_protocol_and_does_not_train(baseline, tmp_path, monkeypatch):
    import hist22_ablation as ablation
    monkeypatch.setattr(ablation.subprocess, "run", lambda *a, **kw: pytest.fail("prepare launched training"))
    path, source = baseline
    before = path.read_bytes()
    root = tmp_path / "pair with spaces"
    manifest = ablation.prepare_pair(path, root, repo_root=ROOT, device="cpu")
    assert path.read_bytes() == before
    assert manifest["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert len(manifest["runs"]) == 2
    configs = [json.loads(Path(run["config_path"]).read_text()) for run in manifest["runs"]]
    assert [bev_encoding_spec(c).name for c in configs] == ["hist14", "hist22"]
    assert [bev_encoding_spec(c).channels for c in configs] == [14, 22]
    assert configs[0]["initialization"] == configs[1]["initialization"] == {
        "mode": "hist14_hist22_paired_random", "version": 1, "seed": 42}
    for config in configs:
        assert "initialization_provenance" not in config
        for key in source.keys() - {"data", "experiment", "initialization"}:
            assert config[key] == source[key]
        assert {k:v for k,v in config["data"].items() if k != "bev_encoding"} == {
            k:v for k,v in source["data"].items() if k != "bev_encoding"}
        assert config["train"]["epochs"] == 17 and config["train"]["warmup_epochs"] == 2
        assert bev_encoding_spec(config).density_norm == 47 and bev_encoding_spec(config).intensity_scale == 2
    assert ablation.leaf_differences(*configs) == manifest["config_differences"]
    assert set(manifest["config_differences"]) <= set(manifest["allowed_difference_paths"])
    assert json.loads((root / "baseline.source.json").read_text()) == source
    assert json.loads((root / "ap_report.pending.json").read_text())["status"] == "pending_full_training"
    for run in manifest["runs"]:
        shell = shlex.split(run["train_command"])
        assert shell[:3] == ["cd", str(ROOT), "&&"]
        assert shell[3:] == run["train_argv"]
        assert "--resume" not in shell and "--warm-start" not in shell
        assert run["run_name"] in run["train_argv"]
        assert not Path(run["run_dir"]).exists()
        assert "evaluate" in run["evaluation_argv"] and "--run-dir" in run["evaluation_argv"]
        assert run["checkpoint_selection"]["primary"] == "ap"


@pytest.mark.parametrize("mutation", [
    {"data": {"bev_encoding": {"name": "rich8"}}},
    {"data": {"kitti": {"geometry": {"x_res": .2}}}},
    {"data": {"kitti": {"geometry": {"y_res": .2}}}},
    {"model": {"backbone": "mobilepixor"}},
    {"initialization_provenance": {}},
    {"initialization": {"mode": "backbone_warm_start", "checkpoint": "old.pt"}},
    {"resume": "last.pt"}, {"train": {"warm_start": "trained.pt"}},
    {"initialization": {"mode": "hist14_hist22_paired_random", "version": 1, "seed": 7}},
])
def test_invalid_source_rejected_before_any_artifact_write(baseline, tmp_path, mutation):
    import hist22_ablation as ablation
    from common import create_experiment_config
    path, config = baseline
    path.write_text(json.dumps(create_experiment_config(config, mutation)))
    output = tmp_path / "absent parent" / "pair"
    with pytest.raises(ValueError): ablation.prepare_pair(path, output, repo_root=ROOT)
    assert not output.parent.exists()


def test_existing_output_never_overwritten_and_protocol_fingerprint_changes(baseline, tmp_path):
    import hist22_ablation as ablation
    path, config = baseline
    root = tmp_path / "pair"
    first = ablation.prepare_pair(path, root, repo_root=ROOT)
    original = (root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError): ablation.prepare_pair(path, root, repo_root=ROOT)
    assert (root / "manifest.json").read_bytes() == original
    config["train"]["epochs"] += 1
    path.write_text(json.dumps(config))
    second = ablation.prepare_pair(path, tmp_path / "different", repo_root=ROOT)
    assert {r["run_name"] for r in first["runs"]}.isdisjoint(r["run_name"] for r in second["runs"])


@pytest.mark.parametrize("box,primary,mode", [("bev", "loss", "local_bev"), ("bev", "ap", "local_bev"),
    ("3d", "ap", "3d"), ("3d", "ap", "bev")])
def test_evaluation_uses_verified_selected_checkpoint_and_policy_metric(baseline, tmp_path, monkeypatch, box, primary, mode):
    import hist22_ablation as ablation
    path, config = baseline
    config["data"]["box_mode"] = box
    if box == "3d": config["loss"]["vertical_loss_weight"] = 1.0
    config["train"]["checkpoint_selection"] = {"primary": primary, "metric_mode": mode}
    config.setdefault("evaluation", {})["kitti_root"] = str(tmp_path / "raw data")
    path.write_text(json.dumps(config))
    manifest = ablation.prepare_pair(path, tmp_path / "pair", repo_root=ROOT)
    run = manifest["runs"][0]
    saved_config = copy.deepcopy(config)
    selected = {"run_dir": Path(run["run_dir"]), "config": saved_config,
        "config_path": Path(run["run_dir"]) / "config.resolved.json",
        "checkpoint": Path(run["run_dir"]) / "selected" / f"best_{primary}.pt"}
    seen = []
    monkeypatch.setattr(ablation, "selected_run", lambda directory: (seen.append(directory) or selected))
    command = ablation.selected_evaluation_command(run["run_dir"], tmp_path / "result.json", repo_root=ROOT, device="cpu")
    assert seen == [run["run_dir"]]
    assert str(selected["checkpoint"]) in command
    assert ("evaluate_kitti_bev.py" in command[1]) == (box == "bev")
    if box == "3d": assert command[command.index("--metric-mode") + 1] == mode
