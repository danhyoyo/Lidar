import importlib
import itertools
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]


def _audit():
    return importlib.import_module("tools.visualization.audit_gt_sampler")


def _audit_fixture(tmp_path):
    data_dir = tmp_path / "processed"
    (data_dir / "pointcloud").mkdir(parents=True)
    (data_dir / "label").mkdir()
    # At the deterministic pose (10, 0, -1.6), remove one interior point and
    # one shadow point; preserve the point outside the box silhouette.
    points = np.array(
        [[10, 0, -1.1, 0.1], [20, 0, -1.1, 0.2], [20, 3, -1.1, 0.3]],
        dtype=np.float32,
    )
    points.tofile(data_dir / "pointcloud/000001.bin")
    (data_dir / "label/000001.txt").write_text("", encoding="utf-8")
    np.zeros((0, 4), dtype=np.float32).tofile(data_dir / "pointcloud/000002.bin")
    # Covers every proposed pose: each of 20 trials must fail box collision.
    (data_dir / "label/000002.txt").write_text(
        "Car 4 200 200 30 0 -2 0\n", encoding="utf-8"
    )
    database_path = tmp_path / "database.pkl"
    object_points = np.tile(np.array([0, 0, 0.5, 0.8], dtype=np.float32), (8, 1))
    with database_path.open("wb") as stream:
        pickle.dump(
            {"Car": [{"box": np.array([0, 2, 2, 4, 10, 0, -1.6, 0], dtype=np.float32),
                      "points": object_points, "r_origin": 10.0, "num_points": 8}]},
            stream,
        )
    config = {
        "augmentation": {
            "use_pcu_aug": True,
            "pcu_aug": {
                "enable_gt_sampling": True,
                "gt_database_path": str(database_path),
                "sample_counts": {"Car": 1},
                "enable_physics": False,
                "enable_shadow_masking": True,
                "enable_density_subsample": False,
                "enable_radiometric_calibration": False,
            },
            "rotation": {"use": True, "limit_angle": 30, "p": 1},
            "translation": {"use": True, "scale": 20, "p": 1},
        },
        "data": {"kitti": {"location": str(data_dir)}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    frames_path = tmp_path / "frames.txt"
    frames_path.write_text("000001;kitti\n000002;kitti\n", encoding="utf-8")
    return config_path, frames_path


def _fixed_pose(monkeypatch):
    # Control proposals only; geometry, visibility and point removal remain real.
    monkeypatch.setattr(np.random, "uniform", lambda low, high: 10.0 if low >= 0 and high > 5 else 0.0)


def test_audit_aggregates_measured_counters_and_plots_actual_output(tmp_path, monkeypatch):
    config_path, frames_path = _audit_fixture(tmp_path)
    audit = _audit()
    _fixed_pose(monkeypatch)
    monkeypatch.setattr(audit.time, "perf_counter", itertools.count(step=0.001).__next__)
    captured = []
    save_example = audit._save_example

    def capture_example(path, frame_id, original_points, original_boxes, points, boxes):
        captured.append((frame_id, points.copy(), boxes.copy()))
        save_example(path, frame_id, original_points, original_boxes, points, boxes)

    monkeypatch.setattr(audit, "_save_example", capture_example)
    output_dir = tmp_path / "audit"
    result = audit.run_audit(config_path, frames_path, 17, output_dir, warmup=3, max_examples=1)

    diagnostics = result["totals"]
    assert diagnostics["attempted_by_class"] == {"Car": 21}
    assert diagnostics["accepted_by_class"] == {"Car": 1}
    assert diagnostics["rejected_by_reason"] == {"box_collision": 20}
    assert diagnostics["removed_interior_points"] == 1
    assert diagnostics["removed_shadow_points"] == 1
    assert result["latency_ms"]["num_samples"] == 2
    assert result["latency_ms"]["p50"] == pytest.approx(1.0)
    assert result["latency_ms"]["p95"] == pytest.approx(1.0)
    assert "diagnostics" in result["latency_ms"]["scope"]
    assert result["warmup_calls"] == 3
    assert result["seed"] == 17
    assert result["effective_sampler_settings"]["enable_shadow_masking"] is True
    assert result["effective_sampler_settings"]["enable_ground_validation"] is False
    assert result["effective_sampler_settings"]["enable_density_subsample"] is False
    first, second = result["frames"]
    assert (first["points_before"], first["points_after"]) == (3, 9)
    assert (second["boxes_before"], second["boxes_after"]) == (1, 1)
    assert first["diagnostics"]["inserted_positions"] == [{"class_name": "Car", "x": 10.0, "range": 10.0}]
    assert first["diagnostics"]["box_visibility"] == [{"box_index": 0, "is_inserted": True, "baseline_count": 8, "final_count": 8}]
    assert second["diagnostics"]["box_visibility"][0]["final_count"] == 0
    assert len(captured) == 1
    assert captured[0][0] == "000001"
    assert len(captured[0][1]) == first["points_after"]
    assert captured[0][2][0, 4] == 10.0  # no translation/rotation jitter
    assert (output_dir / first["example_image"]).stat().st_size > 1000
    saved = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert saved["totals"] == diagnostics
    for key in ("config", "database", "manifest"):
        assert len(saved["identities"][key]["sha256"]) == 64
    assert saved["environment"]["logical_cpus"] > 0


def test_warmup_does_not_change_seeded_measured_results(tmp_path):
    config_path, frames_path = _audit_fixture(tmp_path)
    audit = _audit()
    cold = audit.run_audit(config_path, frames_path, 42, tmp_path / "cold", warmup=0, max_examples=0)
    warm = audit.run_audit(config_path, frames_path, 42, tmp_path / "warm", warmup=4, max_examples=0)
    assert cold["totals"] == warm["totals"]
    assert [f["diagnostics"] for f in cold["frames"]] == [f["diagnostics"] for f in warm["frames"]]
    assert [f["points_after"] for f in cold["frames"]] == [f["points_after"] for f in warm["frames"]]


@pytest.mark.parametrize("disabled_setting", ["use_pcu_aug", "enable_gt_sampling"])
def test_audit_rejects_disabled_sampling_config(tmp_path, disabled_setting):
    config_path, frames_path = _audit_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if disabled_setting == "use_pcu_aug":
        config["augmentation"][disabled_setting] = False
    else:
        config["augmentation"]["pcu_aug"][disabled_setting] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling must be enabled"):
        _audit().run_audit(config_path, frames_path, 42, tmp_path / "audit", max_examples=0)


@pytest.mark.parametrize("manifest", ["\n \n", ";kitti\n", "000001;nuscenes\n", "000001;kitti;extra\n", "../000001;kitti\n"])
def test_audit_rejects_invalid_manifest(tmp_path, manifest):
    config_path, frames_path = _audit_fixture(tmp_path)
    frames_path.write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        _audit().run_audit(config_path, frames_path, 42, tmp_path / "audit", max_examples=0)


def test_audit_cli_accepts_bare_ids(tmp_path, monkeypatch):
    config_path, frames_path = _audit_fixture(tmp_path)
    frames_path.write_text("000001\n", encoding="utf-8")
    _fixed_pose(monkeypatch)
    output_dir = tmp_path / "audit"
    _audit().main(["--config", str(config_path), "--frames", str(frames_path), "--seed", "23",
                   "--output-dir", str(output_dir), "--warmup", "0", "--max-examples", "0"])
    result = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert result["seed"] == 23
    assert len(result["frames"]) == 1
    assert result["totals"]["accepted_by_class"] == {"Car": 1}
