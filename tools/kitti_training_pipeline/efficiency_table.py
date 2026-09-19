#!/usr/bin/env python3
"""Collect inference-efficiency fields from evaluation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HEADERS = (
    "Method", "Parameters", "Model FPS", "End-to-end FPS",
    "End-to-end mean ms", "End-to-end p95 ms", "Preprocess mean ms",
    "Model mean ms", "Decode/NMS mean ms", "Peak GPU MB", "Samples",
)


def find_json_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.rglob("report.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError("No evaluation JSON files found")
    return files


def efficiency_row(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    latency = result["latency"]
    end_to_end = latency["input_to_detections"]
    model = latency["model"]
    preprocess = latency["preprocess"]
    decode_nms = latency["decode_nms"]
    return {
        "Method": result.get("name", path.stem),
        "Parameters": result.get("model", {}).get("parameters"),
        "Model FPS": model.get("fps"),
        "End-to-end FPS": end_to_end.get("fps"),
        "End-to-end mean ms": end_to_end.get("mean_ms"),
        "End-to-end p95 ms": end_to_end.get("p95_ms"),
        "Preprocess mean ms": preprocess.get("mean_ms"),
        "Model mean ms": model.get("mean_ms"),
        "Decode/NMS mean ms": decode_nms.get("mean_ms"),
        "Peak GPU MB": result.get("runtime", {}).get("torch_peak_memory_mb"),
        "Samples": end_to_end.get("samples"),
    }


def display(value, integer=False) -> str:
    if value is None:
        return "n/a"
    if integer:
        return f"{int(value):,}"
    return f"{float(value):.2f}"


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| " + " | ".join(HEADERS) + " |",
        "|" + "|".join("---:" if index else "---" for index in range(len(HEADERS))) + "|",
    ]
    integer_headers = {"Parameters", "Samples"}
    for row in rows:
        values = [str(row["Method"]).replace("|", "\\|")]
        values.extend(
            display(row[header], integer=header in integer_headers)
            for header in HEADERS[1:]
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("inputs", nargs="+", type=Path,
                       help="Evaluation report JSON files or directories")
    value.add_argument("--output-prefix", required=True, type=Path,
                       help="Output path without extension")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    rows = [efficiency_row(path) for path in find_json_files(args.inputs)]
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
