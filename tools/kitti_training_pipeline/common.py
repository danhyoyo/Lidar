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


def read_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def generate_run_name(
    config: Dict[str, Any],
    seed: int | None = None,
    effective_batch_size: int | None = None,
    extra_tags: list[str] | None = None,
) -> str:
    """Generate standardized run name following:
    {backbone}-{augmentation}-{loss}-{bev_encoder}-{iou}[-{extra}]
    """
    model_cfg = config.get("model", {})
    backbone = str(model_cfg.get("backbone", "unknown")).lower()

    # Augmentation
    aug_cfg = config.get("augmentation", {})
    if aug_cfg.get("use_pcu_aug", False):
        augmentation = "pcu"
    elif aug_cfg.get("p", 0.0) > 0.0 and any(
        isinstance(v, dict) and v.get("use", False) for v in aug_cfg.values()
    ):
        augmentation = "standard_aug"
    elif aug_cfg.get("p", 0.0) > 0.0:
        augmentation = "standard_aug"
    else:
        augmentation = "noaug"

    # Loss
    loss_cfg = config.get("loss", {})
    loss_raw = str(loss_cfg.get("name", "baseline")).lower()
    loss_name = loss_raw if loss_raw.endswith("_loss") else f"{loss_raw}_loss"

    # BEV encoder
    bev_cfg = config.get("data", {}).get("bev_encoding", {})
    bev_name = bev_cfg.get("name")
    if bev_name in {"rich8", "rich10", "rich12", "rich_mamba"}:
        out_ch = int(bev_cfg.get("out_channels", 8 if bev_name in {"rich8", "rich_mamba"} else (10 if bev_name == "rich10" else 12)))
        if bev_name == "rich_mamba" and out_ch != 8:
            bev_encoder = f"rich_mamba_{out_ch}ch"
        else:
            bev_encoder = str(bev_name)
    else:
        bev_encoder = "legacy35"

    # IOU Quality Aware
    if model_cfg.get("header_use_iou", False) or loss_cfg.get("use_iou", False):
        iou = "iqa"
    else:
        iou = "baseline_iou"

    parts = [backbone, augmentation, loss_name, bev_encoder, iou]

    extras: list[str] = []
    neck_type = model_cfg.get("neck_type")
    if neck_type in ("rc_sgfpn", "rc_bisgfpn"):
        extras.append(neck_type)
    elif model_cfg.get("scale_gated_fpn", False):
        extras.append("sgfpn")
    if model_cfg.get("use_reparam", False):
        extras.append("reparam")
    if seed is not None:
        extras.append(f"s{seed}")
    if effective_batch_size is not None:
        extras.append(f"eb{effective_batch_size}")
    if extra_tags:
        extras.extend(extra_tags)

    if extras:
        return f"{'-'.join(parts)}-{'-'.join(extras)}"
    return "-".join(parts)


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
    for path in (detector_root.parent, detector_root, datasets_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return detector_root


def validate_backbone(config: Dict[str, Any]) -> None:
    from core.models.backbones.registry import get_available_backbones

    backbone = config["model"]["backbone"]
    available = get_available_backbones()
    if str(backbone).lower() not in available:
        raise ValueError(
            f"Unsupported backbone: {backbone!r}. "
            f"Available backbones: {', '.join(available)}"
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
    if name not in {"binary_slices", "rich8", "rich10", "rich12", "rich_mamba"}:
        raise ValueError(f"unsupported BEV encoding: {name!r}")
    if name in {"rich8", "rich_mamba"}:
        channels = int(encoding.get("out_channels", 8))
    elif name == "rich10":
        channels = int(encoding.get("out_channels", 10))
    elif name == "rich12":
        channels = int(encoding.get("out_channels", 12))
    else:
        channels = bins("z")
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
