#!/usr/bin/env python3
"""Install a TensorRT plan atomically while retaining the previous engine."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from common import sha256, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--confirm", action="store_true",
                        help="Acknowledge replacement of the destination engine")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    source, destination = args.engine.expanduser().resolve(), args.destination.expanduser().resolve()
    if not args.confirm:
        raise SystemExit(f"Refusing replacement without --confirm: {destination}")
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    if source == destination:
        raise ValueError("Source and destination are the same file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if destination.exists():
        backup = destination.with_name(destination.name +
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(destination, backup)
    temporary = destination.with_name(destination.name + ".new")
    shutil.copy2(source, temporary)
    if sha256(temporary) != sha256(source):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Copied engine failed SHA-256 verification")
    os.replace(temporary, destination)
    write_json(destination.with_suffix(destination.suffix + ".deployment.json"), {
        "source": str(source), "source_sha256": sha256(source),
        "destination": str(destination), "backup": str(backup) if backup else None
    })
    print(f"Deployed: {destination}")
    if backup:
        print(f"Previous engine backup: {backup}")


if __name__ == "__main__":
    main()
