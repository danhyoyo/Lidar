import sys
from pathlib import Path
import pickle
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
    str(ROOT / "tools" / "visualization"),
]

from tools.visualization.visualize_pcu_aug import plot_pcu_augmentation_sample


def test_visualize_pcu_aug_mock_scene(tmp_path):
    processed_dir = tmp_path / "processed"
    pc_dir = processed_dir / "pointcloud"
    lbl_dir = processed_dir / "label"
    pc_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    sid = "000001"
    pts = np.zeros((200, 4), dtype=np.float32)
    pts[:, 0] = np.linspace(5.0, 30.0, 200)
    pts[:, 1] = np.linspace(-10.0, 10.0, 200)
    pts[:, 2] = -1.6
    pts[:, 3] = 0.5
    pts.tofile(str(pc_dir / f"{sid}.bin"))

    label_str = "Car 1.5 1.8 4.5 15.0 0.0 -1.6 0.0\n"
    (lbl_dir / f"{sid}.txt").write_text(label_str, encoding="utf-8")

    db_file = tmp_path / "gt_database.pkl"
    mock_db = {
        "Car": [
            {
                "box": np.array(
                    [0.0, 1.5, 1.8, 4.5, 10.0, 0.0, -1.6, 0.0],
                    dtype=np.float32,
                ),
                "points": np.random.uniform(-0.5, 0.5, size=(30, 4)).astype(
                    np.float32
                ),
                "r_origin": 10.0,
                "num_points": 30,
            }
        ]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    out_img = tmp_path / "test_vis.png"
    fig = plot_pcu_augmentation_sample(
        data_dir=processed_dir,
        gt_database_path=db_file,
        frame_id=sid,
        sample_counts={"Car": 1},
        output_path=out_img,
    )

    assert out_img.is_file()
    assert out_img.stat().st_size > 1000
    matplotlib.pyplot.close(fig)
