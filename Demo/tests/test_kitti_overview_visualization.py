"""Geometry/provenance checks for real KITTI paper illustrations."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "Code"
sys.path.insert(0, str(ROOT))


def api():
    path = ROOT / "tools/visualization/export_kitti_overview.py"
    assert path.is_file(), "The KITTI overview exporter is missing"
    return importlib.import_module("tools.visualization.export_kitti_overview")


def calibration(tmp_path):
    # Real KITTI geometry: nonzero rectification roll/pitch and P2 depth offset.
    path = tmp_path / "calib.txt"
    path.write_text(
        "P2: 707.0493 0 604.0814 45.75831 0 707.0493 180.5066 -.3454157 0 0 1 .004981016\n"
        "R0_rect: .9999128 .01009263 -.008511932 -.01012729 .9999406 -.004037671 .008470675 .004123522 .9999556\n"
        "Tr_velo_to_cam: .006927964 -.9999722 -.002757829 -.02457729 -.001162982 .002749836 -.9999955 -.06127237 .9999753 .006931141 -.001143899 -.3321029\n"
    )
    return path


def labels(tmp_path, text, name="label.txt"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_full_calibrated_corners_and_membership_preserve_pitch_roll(tmp_path):
    module = api()
    cal = calibration(tmp_path)
    objects = module.read_objects(labels(tmp_path,
        "Car 0 0 0 10 20 100 120 1.5 1.8 4.2 2 1.7 15 .8\n"), cal)
    obj = objects[0]
    # Independently defined corners for KITTI camera [x,y,z], bottom center.
    base = np.array([[-2.1, 0, -.9], [2.1, 0, -.9], [2.1, 0, .9], [-2.1, 0, .9]])
    local = np.r_[base, base + [0, -1.5, 0]]
    c, s = np.cos(.8), np.sin(.8)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    expected_camera = local @ rotation.T + [2, 1.7, 15]
    from tools.kitti_training_pipeline.prepare_kitti import parse_calibration
    transform = parse_calibration(cal)
    recovered_camera = np.c_[obj.corners, np.ones(8)] @ transform.T
    np.testing.assert_allclose(recovered_camera[:, :3], expected_camera, atol=1e-10)
    center = obj.corners.mean(axis=0)
    outside = center + 2 * (obj.corners[1] - obj.corners[0])
    points = np.r_[center[None], obj.corners, outside[None]]
    assert module.points_in_object(points, obj).tolist() == [True] * 9 + [False]


def test_real_scores_are_required_only_for_prediction_files(tmp_path):
    module = api()
    cal = calibration(tmp_path)
    text = "Cyclist 0 0 0 10 20 100 120 1.8 .7 1.9 2 1.7 15 .8"
    path = labels(tmp_path, text + "\n")
    assert module.read_objects(path, cal)[0].score is None
    with pytest.raises(ValueError, match="score"):
        module.read_objects(path, cal, predictions=True)
    path.write_text(text + " .87\n")
    assert module.read_objects(path, cal, predictions=True)[0].score == .87
    path.write_text(text + " nan\n")
    with pytest.raises(ValueError):
        module.read_objects(path, cal, predictions=True)


def test_binary_reader_returns_exact_samples_and_rejects_partial_records(tmp_path):
    module = api()
    path = tmp_path / "000000.bin"
    original = np.array([[1, 2, 3, .4], [4, 5, 6, .7]], dtype="<f4")
    original.tofile(path)
    np.testing.assert_array_equal(module.read_points(path), original)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="16"):
        module.read_points(path)


def test_selection_requires_three_supported_classes_before_total_box_count():
    module = api()
    rows = [
        {"frame_id": "000001", "usable_counts": {"Car": 30, "Cyclist": 0, "Pedestrian": 0},
         "counts": {"Car": 30, "Cyclist": 0, "Pedestrian": 0},
         "best_point_counts": {"Car": 1000, "Cyclist": 0, "Pedestrian": 0}},
        {"frame_id": "000002", "usable_counts": {"Car": 2, "Cyclist": 2, "Pedestrian": 2},
         "counts": {"Car": 2, "Cyclist": 2, "Pedestrian": 2},
         "best_point_counts": {"Car": 100, "Cyclist": 80, "Pedestrian": 50}},
    ]
    assert module.select_candidate(rows, min_points=20)["frame_id"] == "000002"
    with pytest.raises(ValueError, match="three classes"):
        module.select_candidate(rows[:1], min_points=20)


def test_separate_archive_layout_is_not_mistaken_for_flat_training_root(tmp_path):
    module = api()
    for name, extension in (("velodyne", ".bin"), ("label_2", ".txt"), ("calib", ".txt")):
        directory = tmp_path / name / "training" / name
        directory.mkdir(parents=True)
        (directory / f"000001{extension}").touch()
    paths = module.dataset_paths(tmp_path)
    assert paths.velodyne == tmp_path / "velodyne/training/velodyne"
    assert paths.labels == tmp_path / "label_2/training/label_2"


def test_predictions_allow_unknown_occlusion_and_truncation(tmp_path):
    module = api()
    obj = module.read_objects(labels(tmp_path,
        "Cyclist -1 -1 -10 10 20 100 120 1.8 .7 1.9 2 1.7 15 .8 .87\n"),
        calibration(tmp_path), predictions=True)[0]
    assert module.is_usable(obj, max_distance=45)


def test_export_preserves_original_points_indices_and_source_kind(tmp_path):
    import json
    from PIL import Image
    module = api()
    root = tmp_path / "kitti/training"
    for name in ("velodyne", "label_2", "calib"):
        (root / name).mkdir(parents=True)
    cal = calibration(tmp_path)
    (root / "calib/000001.txt").write_text(cal.read_text())
    label_path = root / "label_2/000001.txt"
    label_path.write_text("\n".join(
        f"{name} 0 0 0 10 20 100 120 1.8 .8 2 {x} 1.7 15 .8"
        for name, x in zip(module.CLASSES, [-4, 0, 4])) + "\n")
    objects = module.read_objects(label_path, cal)
    original = np.array([np.r_[obj.center, .5] for obj in objects], dtype="<f4")
    original.tofile(root / "velodyne/000001.bin")
    output = tmp_path / "export"
    module.main(["--kitti-root", str(root.parent), "--frame", "1", "--output-dir", str(output),
                 "--min-object-points", "1", "--zoom-padding", "0", "--dpi", "72",
                 "--scene-size", "180", "120", "--zoom-size", "150"])
    report = json.loads((output / "metadata.json").read_text())
    assert report["box_source"] == "KITTI ground truth"
    assert report["generated_points"] == 0
    assert report["counts"] == dict.fromkeys(module.CLASSES, 1)
    for name in module.CLASSES:
        data = np.load(output / f"zoom_{name.lower()}_points.npz")
        np.testing.assert_array_equal(data["points"], original[data["source_indices"]])
        assert len(data["inside_indices"]) == 1
        with Image.open(output / f"zoom_{name.lower()}.png") as panel:
            assert panel.size == (150, 150)
        assert (output / f"zoom_{name.lower()}.svg").stat().st_size > 0
