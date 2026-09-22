#!/usr/bin/env python3
"""Shared helpers for the KITTI training/export pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


SUPPORTED_BACKBONES = {"mobilepixor", "mobilepixor_coordatt", "pixor", "rpn"}


def read_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def configure_detector_imports(detector_root: Path | str) -> Path:
    """Expose the copied detector package and its legacy utils_1 imports."""
    detector_root = Path(detector_root).expanduser().resolve()
    core_root = detector_root / "core"
    datasets_root = core_root / "datasets"
    required = [
        core_root / "models" / "model.py",
        core_root / "datasets" / "dataset.py",
        core_root / "losses" / "loss_fn.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Detector source is incomplete. Missing:\n  " + "\n  ".join(missing)
        )
    for path in (detector_root, datasets_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return detector_root


def validate_backbone(config: Dict[str, Any]) -> None:
    backbone = config["model"]["backbone"]
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone {backbone!r}; expected one of "
            f"{sorted(SUPPORTED_BACKBONES)}"
        )


def build_model(config: Dict[str, Any]):
    validate_backbone(config)
    from core.models.model import CustomModel

    return CustomModel(
        config["model"],
        config["data"]["num_classes"],
        input_channels=input_shape(config)[1],
    )


def input_shape(config: Dict[str, Any], dataset_name: str = "kitti") -> Tuple[int, ...]:
    geometry = config["data"][dataset_name]["geometry"]

    def bins(axis: str) -> int:
        return int(
            round(
                (geometry[f"{axis}_max"] - geometry[f"{axis}_min"])
                / geometry[f"{axis}_res"]
            )
        )

    encoding = config["data"].get("bev_encoding", {"name": "binary_slices"})
    name = encoding.get("name", "binary_slices")
    if name not in {"binary_slices", "rich8"}:
        raise ValueError(f"unsupported BEV encoding: {name!r}")
    channels = 8 if name == "rich8" else bins("z")
    return (1, channels, bins("y"), bins("x"))


def normalize_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def atomic_torch_save(value: Any, path: Path | str) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata(path: Path | str) -> Dict[str, Any]:
    try:
        root = Path(path).resolve()
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
