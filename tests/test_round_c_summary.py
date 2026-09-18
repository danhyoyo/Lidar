#!/usr/bin/env python3
"""Small deterministic checks for the Round C result summarizer."""

import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))

import summarize_round_c as summary


def score_metric(predictions, labels):
    del labels
    return float(np.mean([rows[0, 1] for rows in predictions.values()]))


def main():
    labels = {name: object() for name in ("a", "b", "c")}
    profiles = {
        profile: {
            seed: {
                name: np.array([[0, base + seed / 1000 + index]], dtype=np.float32)
                for index, name in enumerate(labels)
            }
            for seed in (42, 43, 44)
        }
        for profile, base in (("C0", 0.0), ("C1", 1.0), ("C2", 1.5))
    }
    first = summary.paired_bootstrap(
        profiles, labels, samples=20, rng_seed=42, metric=score_metric
    )
    second = summary.paired_bootstrap(
        profiles, labels, samples=20, rng_seed=42, metric=score_metric
    )
    assert first == second
    assert np.isclose(first["c1_minus_c0"]["estimate"], 1.0)
    assert np.isclose(first["c2_minus_c1"]["estimate"], 0.5)

    values = summary.describe([1.0, 2.0, 3.0])
    assert values == {"mean": 2.0, "sd": 1.0, "values": [1.0, 2.0, 3.0]}
    report = {
        "schema_version": 1,
        "profiles": {"C0": {"accuracy": values}},
        "gates": {"c1_accuracy": {"passed": True}},
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        summary.write_outputs(report, root / "one.json", root / "one.md")
        summary.write_outputs(report, root / "two.json", root / "two.md")
        assert (root / "one.json").read_bytes() == (root / "two.json").read_bytes()
        assert (root / "one.md").read_bytes() == (root / "two.md").read_bytes()

    print("PASS round_c_summary")


if __name__ == "__main__":
    main()
