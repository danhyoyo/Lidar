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
        prefix = Path(temporary) / "table22"
        evaluation_table.main([str(directory), "--output-prefix", str(prefix)])

        markdown = prefix.with_suffix(".md").read_text(encoding="utf-8")
        assert "| bev_run | 600,337 | 92.00 | 77.28 | 91.95 | 64.75 | 75.15 | 76.87 |" in markdown

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
        assert len(rows) == 1

    print("PASS evaluation_table")


if __name__ == "__main__":
    main()
