#!/usr/bin/env python3
"""Collect evaluation JSON files into Table 22-style Markdown and CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COMMON_HEADERS = ("Method", "Number of parameters", "FPS")
HEADERS = COMMON_HEADERS + (
    "Mean AP-9", "Car", "Ped", "Cyc", "KITTI mAP@Moderate",
)
DEFAULT_OUTPUT_PREFIX = (
    Path(__file__).resolve().parents[2] / "evaluation" / "evaluation_table"
)


def find_json_files(inputs: list[Path]) -> list[Path]:
    files = []
    for path in inputs or [Path("evaluation_json")]:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError("No JSON files found")
    return files


def metric_row(result: dict, path: Path, primary: dict, headers: tuple[str, ...]) -> dict:
    per_class = primary["per_class"]
    return {
        "Method": result.get("name", path.stem),
        "Number of parameters": result.get("model", {}).get("parameters"),
        "FPS": result.get("latency", {}).get("model", {}).get("fps"),
        headers[3]: primary["mean_ap_9_percent"],
        headers[4]: per_class["Car"]["mean_ap_r40_percent"],
        headers[5]: per_class["Pedestrian"]["mean_ap_r40_percent"],
        headers[6]: per_class["Cyclist"]["mean_ap_r40_percent"],
        headers[7]: primary["map_moderate_percent"],
    }


def evaluation_row(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    return metric_row(result, path, result["accuracy"], HEADERS)


def display(value, integer=False) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}" if integer else f"{float(value):.2f}"


def markdown_table(rows: list[dict], headers: tuple[str, ...] = HEADERS) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [str(row["Method"]).replace("|", "\\|")]
        values.append(display(row["Number of parameters"], integer=True))
        values.extend(display(row[header]) for header in headers[2:])
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "inputs", nargs="*", type=Path,
        help="Evaluation JSON files or directories (default: evaluation_json)",
    )
    value.add_argument(
        "--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX,
        help="Output path without extension (default: <repo>/evaluation/evaluation_table)",
    )
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    rows = [evaluation_row(path) for path in find_json_files(args.inputs)]

    markdown = markdown_table(rows)
    markdown_path = args.output_prefix.with_suffix(".md")
    csv_path = args.output_prefix.with_suffix(".csv")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(markdown, end="")
    print(f"Wrote {markdown_path.resolve()}")
    print(f"Wrote {csv_path.resolve()}")

if __name__ == "__main__":
    main()
