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
THREE_D_HEADERS = COMMON_HEADERS + (
    "Mean 3D AP-9", "3D Car", "3D Ped", "3D Cyc", "3D KITTI mAP@Moderate",
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


def evaluation_rows(path: Path) -> tuple[dict, dict | None]:
    result = json.loads(path.read_text(encoding="utf-8"))
    accuracy = result["accuracy"]
    bev = accuracy.get("bev", accuracy)
    three_d = accuracy.get("3d")
    return (
        metric_row(result, path, bev, HEADERS),
        metric_row(result, path, three_d, THREE_D_HEADERS) if three_d else None,
    )


def evaluation_row(path: Path) -> dict:
    return evaluation_rows(path)[0]


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
    bev_rows, three_d_rows = [], []
    for path in find_json_files(args.inputs):
        bev, three_d = evaluation_rows(path)
        bev_rows.append(bev)
        if three_d:
            three_d_rows.append(three_d)

    markdown = markdown_table(bev_rows)
    markdown_path = args.output_prefix.with_suffix(".md")
    csv_path = args.output_prefix.with_suffix(".csv")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(bev_rows)
    print(markdown, end="")
    print(f"Wrote {markdown_path.resolve()}")
    print(f"Wrote {csv_path.resolve()}")

    if three_d_rows:
        three_d_prefix = args.output_prefix.with_name(f"{args.output_prefix.name}_3d")
        three_d_markdown = markdown_table(three_d_rows, THREE_D_HEADERS)
        three_d_markdown_path = three_d_prefix.with_suffix(".md")
        three_d_csv_path = three_d_prefix.with_suffix(".csv")
        three_d_markdown_path.write_text(three_d_markdown, encoding="utf-8")
        with three_d_csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=THREE_D_HEADERS)
            writer.writeheader()
            writer.writerows(three_d_rows)
        print(three_d_markdown, end="")
        print(f"Wrote {three_d_markdown_path.resolve()}")
        print(f"Wrote {three_d_csv_path.resolve()}")


if __name__ == "__main__":
    main()
