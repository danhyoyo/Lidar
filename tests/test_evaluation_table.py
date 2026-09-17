#!/usr/bin/env python3
import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))

import evaluation_table


def evaluation(name, accuracy):
    return {
        "status": "ok",
        "name": name,
        "model": {"parameters": 600337},
        "latency": {"model": {"fps": 92.00085611926869}},
        "accuracy": accuracy,
    }


def accuracy(mean_ap, car, pedestrian, cyclist, moderate):
    return {
        "mean_ap_9_percent": mean_ap,
        "map_moderate_percent": moderate,
        "per_class": {
            "Car": {"mean_ap_r40_percent": car},
            "Pedestrian": {"mean_ap_r40_percent": pedestrian},
            "Cyclist": {"mean_ap_r40_percent": cyclist},
        },
    }


def main():
    default_args = evaluation_table.parser().parse_args([])
    assert default_args.output_prefix == ROOT / "evaluation" / "evaluation_table"

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary) / "evaluation_json"
        directory.mkdir()
        (directory / "bev.json").write_text(json.dumps(evaluation(
            "bev_run", accuracy(77.281, 91.946, 64.748, 75.150, 76.874)
        )), encoding="utf-8")
        (directory / "center3d.json").write_text(json.dumps(evaluation(
            "center3d_run", {
                "bev": accuracy(60, 61, 62, 63, 64),
                "3d": accuracy(70, 71, 72, 73, 74),
            }
        )), encoding="utf-8")

        prefix = Path(temporary) / "table22"
        evaluation_table.main([str(directory), "--output-prefix", str(prefix)])

        markdown = prefix.with_suffix(".md").read_text(encoding="utf-8")
        assert "| bev_run | 600,337 | 92.00 | 77.28 | 91.95 | 64.75 | 75.15 | 76.87 |" in markdown
        assert "| center3d_run | 600,337 | 92.00 | 60.00 | 61.00 | 62.00 | 63.00 | 64.00 |" in markdown

        with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert rows[0] == {
            "Method": "bev_run",
            "Number of parameters": "600337",
            "FPS": "92.00085611926869",
            "Mean AP-9": "77.281",
            "Car": "91.946",
            "Ped": "64.748",
            "Cyc": "75.15",
            "KITTI mAP@Moderate": "76.874",
        }
        assert rows[1]["Mean AP-9"] == "60"

        three_d_markdown = prefix.with_name("table22_3d").with_suffix(".md").read_text(
            encoding="utf-8"
        )
        assert "| center3d_run | 600,337 | 92.00 | 70.00 | 71.00 | 72.00 | 73.00 | 74.00 |" in three_d_markdown
        with prefix.with_name("table22_3d").with_suffix(".csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            three_d_rows = list(csv.DictReader(stream))
        assert three_d_rows == [{
            "Method": "center3d_run",
            "Number of parameters": "600337",
            "FPS": "92.00085611926869",
            "Mean 3D AP-9": "70",
            "3D Car": "71",
            "3D Ped": "72",
            "3D Cyc": "73",
            "3D KITTI mAP@Moderate": "74",
        }]

    print("PASS evaluation_table")


if __name__ == "__main__":
    main()
