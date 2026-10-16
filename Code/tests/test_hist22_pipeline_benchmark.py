"""Benchmark contracts: real stage execution and reproducible sampling."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.benchmarks.benchmark_hist22_pipeline import (
    main, run_encoding_benchmark, run_pipeline_benchmark, select_frames,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frame_selection_is_seeded_stratified_and_excludes_showcase(tmp_path):
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text("000804;kitti\n" + "\n".join(f"{i:06d};kitti" for i in range(10)))
    val.write_text("\n".join(f"{i:06d};kitti" for i in range(10, 20)))
    selected = select_frames(train, val, per_split=3, seed=42)
    assert selected == select_frames(train, val, per_split=3, seed=42)
    assert len(selected) == len({row["id"] for row in selected}) == 10
    assert [row["selection"] for row in selected].count("train_random") == 3
    assert [row["selection"] for row in selected].count("val_random") == 3
    assert selected != select_frames(train, val, per_split=3, seed=43)


def test_encoding_executes_both_chw_paths_with_reference_and_raw_samples():
    geometry = dict(x_min=0., x_max=4.8, x_res=.1, y_min=-1.6, y_max=1.6,
                    y_res=.1, z_min=-2.5, z_max=1., z_res=.1)
    rng = np.random.default_rng(42)
    points = rng.uniform([0., -1., -2., 0.], [4., 1., .5, 1.], (120, 4)).astype(np.float32)
    report = run_encoding_benchmark([{"id": "fixture", "points": points}], geometry,
                                    warmup=1, iterations=2, seed=42, reference_count=1)
    assert report["frames"][0]["reference_parity"]["passed"]
    for result in report["frames"][0]["results"].values():
        assert len(result["samples_ms"]) == 2
        assert result["p50_ms"] > 0
        assert result["contiguous"] and result["shape"][1:] == [32, 48]
    assert len(report["execution_order"]) == 2


def test_cli_rejects_invalid_counts_and_existing_output(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("preserve")
    with pytest.raises(SystemExit):
        main(["--output", str(output), "--iterations", "0"])
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])
    assert output.read_text() == "preserve"


@pytest.mark.parametrize("tamper", ["digest", "schema", "loss", "augmentation", "budget", "duplicate", "policy"])
def test_cli_rejects_tampered_pair_before_creating_any_artifacts(tmp_path, tamper):
    geometry = dict(x_min=0., x_max=70.4, x_res=.1, y_min=-40., y_max=40.,
                    y_res=.1, z_min=-2.5, z_max=1., z_res=.1)
    from core.bev_encoding import resolve_bev_encoding
    runs = []
    for label, name, width in (("A", "hist14", 14), ("B", "hist22", 22)):
        config = {"seed": 42, "initialization": {"mode": "hist14_hist22_paired_random", "version": 1, "seed": 42},
            "data": {"bev_encoding": dict(name=name, backend="numba", out_channels=width), "kitti": {"geometry": geometry}},
            "experiment": {"name": name}, "loss": {"name": "oga"}, "augmentation": {"mode": "openpcdet"}, "train": {"epochs": 50}}
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        original_digest = digest(config)
        if tamper == "digest":
            config["train"]["epochs"] = 51
        if tamper == "policy":
            config.pop("initialization")
        if name == "hist22":
            if tamper == "loss":
                config["loss"]["name"] = "different"
            elif tamper == "augmentation":
                config["augmentation"]["mode"] = "different"
            elif tamper == "budget":
                config["train"]["epochs"] = 51
        checksum = original_digest if tamper == "digest" else digest(config)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(config))
        runs.append(dict(label=label, encoding=name, config_path=str(path), config_sha256=checksum,
            schema_sha256=resolve_bev_encoding(config["data"]["bev_encoding"], geometry).semantic_hash,
            run_name=name + "_" + checksum[:20]))
    if tamper == "schema":
        runs[1]["schema_sha256"] = "0" * 64
    if tamper == "duplicate":
        runs[1] = copy.deepcopy(runs[0])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(version=1, runs=runs)))
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="pair|config|schema|manifest|initialization"):
        main(["--mode", "pipeline", "--pair-manifest", str(manifest), "--output", str(output)])
    assert not output.exists()
    assert not output.with_suffix(".artifacts").exists()


def test_small_real_pipeline_uses_targets_backward_decode_and_strict_restore(tmp_path):
    # Camera->Velodyne axis conversion in the real KITTI label converter.
    raw = tmp_path / "raw"
    for name in ("velodyne", "label_2", "calib"):
        (raw / name).mkdir(parents=True)
    rng = np.random.default_rng(14)
    points = rng.uniform([.1, -.8, -1., 0.], [4.5, .8, .5, 1.], (900, 4)).astype(np.float32)
    points.tofile(raw / "velodyne/000001.bin")
    (raw / "label_2/000001.txt").write_text("Car 0 0 0 0 0 20 20 1.4 1.0 1.5 0 0 2 0\n")
    (raw / "calib/000001.txt").write_text(
        "R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n")
    for name, suffix in (("velodyne", "bin"), ("label_2", "txt"), ("calib", "txt")):
        (raw / name / f"000002.{suffix}").write_bytes((raw / name / f"000001.{suffix}").read_bytes())
    base = json.loads((ROOT / "configs/config.json").read_text())
    base["data"]["kitti"]["geometry"].update(
        x_min=0., x_max=4.8, x_res=.1, y_min=-1.6, y_max=1.6, y_res=.1)
    base["data"]["min_radius"] = 0
    base["model"].update(backbone="mobilepixornext", stage_depths=[1, 1, 1],
                          expansion=1., backbone_out_dim=16, c4_context="none",
                          c4_attention="none", c4_attention_scales=[],
                          neck_type="scale_gated_fpn", scale_gated_fpn=True,
                          head_mode="grouped", header_use_iou=True)
    base["data"]["head_groups"] = [dict(name="car", classes=["Car"]),
                                         dict(name="ped_cyc", classes=["Pedestrian", "Cyclist"])]
    base["loss"] = {"name": "oga", "use_iou": True}
    base["initialization"] = {"mode": "hist14_hist22_paired_random", "version": 1, "seed": 42}
    configs = {}
    for name, width in (("hist14", 14), ("hist22", 22)):
        configs[name] = copy.deepcopy(base)
        configs[name]["data"]["bev_encoding"] = dict(name=name, backend="numba", out_channels=width)
    report = run_pipeline_benchmark(configs, raw / "velodyne", raw / "label_2", raw / "calib",
                                    ["000001"], tmp_path / "smoke", warmup=0, iterations=1,
                                    training_iterations=1, seed=42, training_frame_id="000002")
    assert report["initial_output_parity"]["passed"]
    assert report["training_frame_id"] not in report["frame_ids"]
    assert (tmp_path / "smoke/train.txt").read_text().strip() == "000002"
    assert (tmp_path / "smoke/val.txt").read_text().strip() == "000001"
    for name in configs:
        row = report["models"][name]
        assert row["checkpoint"]["strict_output_identity"]
        assert row["training"]["losses"][0] > 0
        assert row["stages"]["decode_nms"]["p50_ms"] > 0
        assert row["training"]["positive_target_cells"] > 0
    assert all(report["models"]["hist22"]["training"]["extra_slice_gradient_nonzero"])
    assert all(report["models"]["hist22"]["training"]["extra_slice_updated"])
