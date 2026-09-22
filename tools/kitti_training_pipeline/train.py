#!/usr/bin/env python3
"""Train a configured KITTI MobilePIXOR variant with best-checkpoint selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (atomic_torch_save, build_model, configure_detector_imports,
                    git_metadata, normalize_state_dict, read_json, sha256,
                    write_json)


def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    """Append one epoch to a CSV file, preserving the run's column schema."""
    fieldnames = list(row)
    if path.exists() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as stream:
            existing_fieldnames = next(csv.reader(stream), [])
        if existing_fieldnames != fieldnames:
            raise RuntimeError(
                f"CSV columns in {path} do not match this run. "
                "Resume into the original run directory or choose a new run name."
            )
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def move_tensor_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def loader_kwargs(num_workers: int, pin_memory: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory,
                              "worker_init_fn": seed_worker}
    if num_workers > 0:
        result.update(persistent_workers=True, prefetch_factor=2)
    return result


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return torch.amp.autocast(
        device_type=device.type, dtype=dtype, enabled=True
    )


def planned_optimizer_update_attempts(
    epochs: int, batches_per_epoch: int, accumulation_steps: int
) -> int:
    """Return the number of accumulation-group update attempts in a run.

    The last, incomplete accumulation group is deliberately an attempt.  B3's
    BD-to-Hellinger schedule is indexed by these attempts, rather than by
    successful optimizer steps, so a skipped non-finite group cannot move its
    switch point.
    """
    if epochs <= 0 or batches_per_epoch <= 0 or accumulation_steps <= 0:
        raise ValueError(
            "epochs, batches_per_epoch, and accumulation_steps must be positive"
        )
    return int(epochs) * math.ceil(int(batches_per_epoch) / int(accumulation_steps))


def normalized_training_progress(
    global_update_attempts: int, total_planned_attempts: int
) -> float:
    """Normalize an attempt counter to the closed unit interval."""
    if total_planned_attempts <= 0:
        raise ValueError("total_planned_attempts must be positive")
    value = float(global_update_attempts) / float(total_planned_attempts)
    if not math.isfinite(value):
        raise ValueError("global_update_attempts must produce finite progress")
    return min(1.0, max(0.0, value))


def all_gradients_finite(parameters) -> bool:
    """Check every present gradient without treating a missing gradient as bad."""
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in parameters
    )


def gradient_l2_norm(parameters) -> torch.Tensor:
    """Compute the global L2 norm of present gradients without masking NaN/Inf."""
    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        contribution = parameter.grad.detach().float().square().sum()
        squared_norm = (
            contribution if squared_norm is None else squared_norm + contribution
        )
    if squared_norm is None:
        return torch.tensor(0.0)
    return squared_norm.sqrt()


LOSS_COUNT_DIAGNOSTICS = frozenset(("clamp_count", "yaw_fallback_count"))


def accumulate_loss_outputs(component_sums: Dict[str, float],
                            diagnostic_counts: Dict[str, int],
                            losses: Dict[str, Any], *, batch_size: int) -> None:
    """Add batch losses and integer diagnostics using their distinct reductions.

    Loss components are sample-weighted before epoch/validation means are
    calculated. Counts describe events observed in a batch, so multiplying
    them by batch size would over-report clamping or MGIoU yaw fallbacks.
    """
    for name, value in losses.items():
        scalar = float(value.detach().item() if torch.is_tensor(value) else value)
        if name in LOSS_COUNT_DIAGNOSTICS:
            diagnostic_counts[name] = diagnostic_counts.get(name, 0) + int(scalar)
        elif math.isfinite(scalar):
            component_sums[name] = component_sums.get(name, 0.0) + scalar * batch_size


def selection_checkpoint_filename(config: Dict[str, Any]) -> str:
    """Map the opt-in B-series selection policy to its selected filename."""
    policy = config.get("train", {}).get("selection_policy", "best")
    if policy == "best":
        return "best.pt"
    if policy == "final_epoch":
        return "final.pt"
    raise ValueError(
        "train.selection_policy must be 'best' or 'final_epoch', "
        f"got {policy!r}"
    )


def validate_run_directory_policy(run_dir: Path | str,
                                  resume_path: Path | str | None) -> tuple[Path, Path | None]:
    """Enforce one immutable run directory for fresh and resumed training.

    A fresh run must claim a directory that does not exist yet, preventing an
    accidental append to prior metrics/checkpoints. A resume instead requires
    an existing selected run and a real checkpoint resolved beneath it; this
    prevents importing weights from another experiment into its provenance.
    """
    selected_run = Path(run_dir).expanduser().resolve()
    if resume_path is None:
        if selected_run.exists():
            raise ValueError(
                f"Fresh run directory already exists: {selected_run}. "
                "Choose a new --run-name or resume that exact run."
            )
        return selected_run, None

    checkpoint = Path(resume_path).expanduser().resolve()
    if not selected_run.is_dir():
        raise ValueError(
            f"Resume requires an existing run directory: {selected_run}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    try:
        checkpoint.relative_to(selected_run)
    except ValueError as error:
        raise ValueError(
            "Resume checkpoint must be contained in the selected run directory: "
            f"{checkpoint} is outside {selected_run}"
        ) from error
    return selected_run, checkpoint


def require_successful_epoch(epoch: int, successful_updates: int) -> None:
    """Block validation/checkpointing if an epoch never updated trainable weights."""
    if int(successful_updates) <= 0:
        raise ValueError(
            f"Epoch {epoch} had no successful optimizer updates; refusing to "
            "validate or save stale weights"
        )


def save_checkpoint_with_sha256(payload: Dict[str, Any], path: Path | str) -> str:
    """Atomically save a checkpoint and its adjacent immutable-content record."""
    path = Path(path)
    atomic_torch_save(payload, path)
    digest = sha256(path)
    write_json(path.with_suffix(path.suffix + ".sha256.json"), {
        "checkpoint": str(path.resolve()), "sha256": digest
    })
    return digest


def verify_checkpoint_sha256(checkpoint_path: Path | str) -> str:
    """Verify a checkpoint against the sidecar emitted by ``save_checkpoint``."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Checkpoint SHA256 sidecar does not exist: {sidecar}")
    record = read_json(sidecar)
    if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
        raise ValueError(f"Checkpoint SHA256 sidecar is invalid: {sidecar}")
    recorded_path = record.get("checkpoint")
    if recorded_path is not None:
        try:
            sidecar_target = Path(recorded_path).expanduser().resolve()
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Checkpoint SHA256 sidecar has invalid checkpoint path: {sidecar}"
            ) from error
        if sidecar_target != checkpoint:
            raise ValueError(
                "Checkpoint SHA256 sidecar belongs to a different target: "
                f"{sidecar_target} != {checkpoint}"
            )
    digest = sha256(checkpoint)
    if digest != record["sha256"]:
        raise ValueError(
            f"Checkpoint SHA256 mismatch for {checkpoint}: expected "
            f"{record['sha256']}, got {digest}"
        )
    return digest


def resolved_runtime_controls(device: torch.device, num_workers: int,
                              max_train_batches: int,
                              max_val_batches: int) -> Dict[str, Any]:
    """Return JSON-safe CLI controls that change runtime training behavior."""
    return {
        "device": str(device),
        "num_workers": int(num_workers),
        "max_train_batches": int(max_train_batches),
        "max_val_batches": int(max_val_batches),
    }


def verify_run_manifest_inputs(manifest: Dict[str, Any], source_config: Path | str,
                               train_split: Path | str,
                               val_split: Path | str) -> None:
    """Verify immutable manifest input paths and hashes before a resume load."""
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must be a JSON object")

    def verify_record(record, path, label):
        if not isinstance(record, dict):
            raise ValueError(f"Run manifest {label} record is missing or malformed")
        recorded_path = record.get("path")
        recorded_digest = record.get("sha256")
        if not isinstance(recorded_path, str) or not isinstance(recorded_digest, str):
            raise ValueError(f"Run manifest {label} path/SHA256 is missing or malformed")
        if len(recorded_digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in recorded_digest
        ):
            raise ValueError(f"Run manifest {label} SHA256 is malformed")
        expected_path = Path(path).expanduser().resolve()
        if not expected_path.is_file():
            raise ValueError(f"Current {label} does not exist: {expected_path}")
        if Path(recorded_path).expanduser().resolve() != expected_path:
            raise ValueError(
                f"Run manifest {label} path does not match current input: "
                f"{recorded_path} != {expected_path}"
            )
        actual_digest = sha256(expected_path)
        if actual_digest != recorded_digest:
            raise ValueError(
                f"Run manifest {label} SHA256 mismatch: expected "
                f"{recorded_digest}, got {actual_digest}"
            )

    verify_record(manifest.get("source_config"), source_config, "source config")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Run manifest splits record is missing or malformed")
    verify_record(splits.get("train"), train_split, "train split")
    verify_record(splits.get("val"), val_split, "validation split")


def record_resume_event(run_dir: Path | str, *, command_argv,
                        checkpoint_path: Path | str, checkpoint_sha256: str,
                        runtime_controls: Dict[str, Any],
                        initial_manifest: Dict[str, Any],
                        device: torch.device) -> Path:
    """Append one immutable resume-provenance event without rewriting manifest."""
    events_dir = Path(run_dir).resolve() / "resume_events"
    events_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    event_path = events_dir / (
        timestamp.strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex + ".json"
    )
    event = {
        "command_argv": [str(value) for value in command_argv],
        "checkpoint": {
            "path": str(Path(checkpoint_path).expanduser().resolve()),
            "sha256": checkpoint_sha256,
        },
        "resumed_at_utc": timestamp.isoformat(),
        "runtime_controls": runtime_controls,
        "initial_manifest": {
            "git": initial_manifest.get("git"),
            "runtime": initial_manifest.get("runtime"),
        },
        "current_environment": {
            "git": git_metadata(Path(__file__).resolve().parents[2]),
            "runtime": runtime_manifest(device),
        },
    }
    # Exclusive creation makes an already-existing event a hard failure rather
    # than silently overwriting provenance. UUID names make collisions remote.
    with event_path.open("x", encoding="utf-8") as stream:
        json.dump(event, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return event_path


def runtime_manifest(device: torch.device) -> Dict[str, Any]:
    """Capture host/runtime fields needed to interpret a reproducibility run."""
    gpu = None
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        gpu = torch.cuda.get_device_name(index)
    return {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "device": str(device),
    }


def build_run_manifest(*, command_argv, source_config: Path,
                       resolved_config: Path, device: torch.device, seed: int,
                       train_split: Path, val_split: Path,
                       total_planned_attempts: int, effective_batch_size: int,
                       selection_policy: str,
                       runtime_controls: Dict[str, Any]) -> Dict[str, Any]:
    """Build the immutable provenance record written only for a fresh run."""
    repository_root = Path(__file__).resolve().parents[2]
    return {
        "command_argv": [str(value) for value in command_argv],
        "source_config": {
            "path": str(source_config.resolve()), "sha256": sha256(source_config)
        },
        "resolved_config_path": str(resolved_config.resolve()),
        "git": git_metadata(repository_root),
        "runtime": runtime_manifest(device),
        "seed": int(seed),
        "splits": {
            "train": {"path": str(train_split.resolve()), "sha256": sha256(train_split)},
            "val": {"path": str(val_split.resolve()), "sha256": sha256(val_split)},
        },
        "total_planned_attempts": int(total_planned_attempts),
        "effective_batch_size": int(effective_batch_size),
        "selection_policy": selection_policy,
        "runtime_controls": runtime_controls,
    }


def checkpoint_payload(model, criterion, optimizer, scheduler, scaler, epoch: int,
                       validation: Dict[str, float], best_val: float,
                       config: Dict[str, Any], *, global_update_attempts: int = 0,
                       successful_updates: int = 0, skipped_updates: int = 0,
                       total_planned_attempts: int = 0) -> Dict[str, Any]:
    return {
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(), "validation": validation,
        "best_validation_objective": best_val, "config": config,
        "global_update_attempts": int(global_update_attempts),
        "successful_updates": int(successful_updates),
        "skipped_updates": int(skipped_updates),
        "total_planned_attempts": int(total_planned_attempts),
        "saved_at_utc": datetime.now(timezone.utc).isoformat()
    }


@torch.no_grad()
def validate(model, criterion, loader, device, precision, max_batches=0,
             progress=0.0):
    model.eval()
    criterion.eval()
    sums: Dict[str, float] = {}
    diagnostic_counts: Dict[str, int] = {}
    samples = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_tensor_batch(batch, device)
        batch_size = int(batch["voxel"].shape[0])
        with autocast_context(device, precision):
            outputs = model(batch["voxel"])
            losses = criterion(outputs, batch, progress=progress)
        accumulate_loss_outputs(sums, diagnostic_counts, losses, batch_size=batch_size)
        samples += batch_size
        if max_batches and batch_index >= max_batches:
            break
    if not samples:
        raise RuntimeError("Validation loader produced no samples")
    metrics = {name: value / samples for name, value in sums.items()}
    metrics.update(diagnostic_counts)
    metrics.update(seconds=time.perf_counter() - started, samples=samples)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--detector-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--physical-batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--amp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    configure_detector_imports(args.detector_root)
    from core.datasets.dataset import Dataset
    from core.losses.loss_fn import LossFunction

    source_config_path = args.config.expanduser().resolve()
    if not source_config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {source_config_path}")
    config = read_json(source_config_path)
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.max_train_batches < 0 or args.max_val_batches < 0:
        raise ValueError("max batch limits must be non-negative")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.physical_batch_size is not None and args.physical_batch_size < 1:
        raise ValueError("physical_batch_size must be positive")
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    config["seed"] = seed
    seed_everything(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    precision = str(
        args.precision or config["train"].get("precision", "fp32")
    ).lower()
    if args.amp:
        if args.precision and args.precision != "fp16":
            raise ValueError("--amp is a legacy alias for --precision fp16")
        precision = "fp16"
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported training precision: {precision!r}")
    if precision != "fp32" and device.type != "cuda":
        raise ValueError(f"{precision} training requires a CUDA device")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16")
    config["train"]["precision"] = precision
    scaler_enabled = precision == "fp16"

    physical_batch_size = int(args.physical_batch_size if args.physical_batch_size is not None else
        config["train"].get("physical_batch_size") or config["train"].get("batch_size", 2))
    accumulation_steps = int(args.accumulation_steps if args.accumulation_steps is not None else
        config["train"].get("accumulation_steps", 1))
    if physical_batch_size < 1 or accumulation_steps < 1:
        raise ValueError("Batch size and accumulation steps must be positive")
    epochs = int(args.epochs if args.epochs is not None else config["train"]["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    save_every = int(config["train"].get("save_every", 5))
    if save_every < 1:
        raise ValueError("save_every must be positive")
    # The resolved configuration must describe the actual CLI overrides, not
    # merely the source JSON, so a resume can prove it is the same experiment.
    config["train"]["epochs"] = epochs
    config["train"]["physical_batch_size"] = physical_batch_size
    config["train"]["accumulation_steps"] = accumulation_steps
    runtime_controls = resolved_runtime_controls(
        device,
        args.num_workers,
        args.max_train_batches,
        args.max_val_batches,
    )
    config["runtime_controls"] = runtime_controls

    train_dataset = Dataset(config["train"]["data"], config["data"],
        config["augmentation"], config["model"]["cls_encoding"], "train")
    # A non-special task name disables augmentation and list-valued visualisation data.
    val_dataset = Dataset(config["val"]["data"], config["data"],
        config["augmentation"], config["model"]["cls_encoding"], "validation")
    generator = torch.Generator().manual_seed(seed)
    common_loader = loader_kwargs(args.num_workers, device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=physical_batch_size,
        shuffle=True, generator=generator, **common_loader)
    validation_batch_size = int(
        config["val"].get("physical_batch_size", physical_batch_size)
    )
    if validation_batch_size < 1:
        raise ValueError("validation physical_batch_size must be positive")
    val_loader = DataLoader(val_dataset, batch_size=validation_batch_size,
        shuffle=False, **common_loader)
    total_loader_batches = len(train_loader)
    batches_this_epoch = (
        min(total_loader_batches, args.max_train_batches)
        if args.max_train_batches else total_loader_batches
    )
    if batches_this_epoch == 0:
        raise RuntimeError("Training loader produced no batches")
    total_planned_attempts = planned_optimizer_update_attempts(
        epochs, batches_this_epoch, accumulation_steps
    )
    selection_policy = config["train"].get("selection_policy", "best")
    selected_filename = selection_checkpoint_filename(config)

    model = build_model(config).to(device)
    criterion = LossFunction(
        config["model"]["cls_encoding"], config.get("loss")
    ).to(device)
    weight_decay = float(config["train"]["weight_decay"])
    model_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    criterion_parameters = [
        parameter for parameter in criterion.parameters() if parameter.requires_grad
    ]
    optimizer_groups = [
        {"params": model_parameters, "weight_decay": weight_decay}
    ]
    if criterion_parameters:
        # Do not regularize UWAG log-scales; their additive term is the
        # uncertainty-weighting regularizer from Equation 48.
        optimizer_groups.append(
            {"params": criterion_parameters, "weight_decay": 0.0}
        )
    optimizer = torch.optim.Adam(
        optimizer_groups, lr=float(config["train"]["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=list(config["train"]["lr_decay_at"]), gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    loss_name = config.get("loss", {}).get(
        "name", config.get("loss", {}).get("regression", "baseline")
    )
    default_name = (
        f"{config['model']['backbone']}_{loss_name}_{precision}_"
        f"seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = args.output_root.expanduser().resolve() / (args.run_name or default_name)
    run_dir, resume_path = validate_run_directory_policy(run_dir, args.resume)
    checkpoints_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best_checkpoints"
    loss_selection_dir = run_dir / "selected"
    resolved_config_path = run_dir / "config.resolved.json"
    manifest_path = run_dir / "run.manifest.json"
    train_split_path = Path(config["train"]["data"]).expanduser().resolve()
    val_split_path = Path(config["val"]["data"]).expanduser().resolve()
    command_argv = (
        list(sys.argv) if argv is None else [str(Path(__file__).resolve()), *argv]
    )
    resume_manifest = None
    resume_digest = None
    if resume_path is None:
        for path in (checkpoints_dir, best_dir, loss_selection_dir):
            path.mkdir(parents=True, exist_ok=False)
        write_json(resolved_config_path, config)
        write_json(manifest_path, build_run_manifest(
            command_argv=command_argv,
            source_config=source_config_path,
            resolved_config=resolved_config_path,
            device=device,
            seed=seed,
            train_split=train_split_path,
            val_split=val_split_path,
            total_planned_attempts=total_planned_attempts,
            effective_batch_size=physical_batch_size * accumulation_steps,
            selection_policy=selection_policy,
            runtime_controls=runtime_controls,
        ))
    else:
        if not resolved_config_path.is_file():
            raise FileNotFoundError(
                f"Resume run is missing resolved config: {resolved_config_path}"
            )
        if read_json(resolved_config_path) != config:
            raise ValueError(
                "Current resolved configuration does not match the existing run; "
                "resume would invalidate its provenance"
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Resume run is missing immutable manifest: {manifest_path}"
            )
        # Read and verify the immutable manifest before deserializing weights.
        # Resume never rewrites either provenance artifact.
        resume_manifest = read_json(manifest_path)
        verify_run_manifest_inputs(
            resume_manifest,
            source_config_path,
            train_split_path,
            val_split_path,
        )
        if resume_manifest.get("resolved_config_path") != str(
            resolved_config_path.resolve()
        ):
            raise ValueError(
                "Run manifest resolved config path does not match selected run: "
                f"{resume_manifest.get('resolved_config_path')!r} != "
                f"{resolved_config_path.resolve()}"
            )
        resume_digest = verify_checkpoint_sha256(resume_path)

    start_epoch, best_val = 0, math.inf
    global_update_attempts = successful_updates = skipped_updates = 0
    if resume_path is not None:
        # Never deserialize an unverified artifact. The event is only recorded
        # once all compatibility checks and state restoration below succeed.
        resume = torch.load(resume_path, map_location=device)
        if isinstance(resume, dict) and "total_planned_attempts" in resume:
            saved_total = int(resume["total_planned_attempts"])
            if saved_total != total_planned_attempts:
                raise ValueError(
                    "Resume checkpoint total_planned_attempts does not match "
                    f"this run ({saved_total} != {total_planned_attempts}); "
                    "changing the plan would move the ProbIoU switch point"
                )
        model.load_state_dict(normalize_state_dict(resume), strict=True)
        if isinstance(resume, dict) and resume.get("criterion_state_dict") is not None:
            criterion.load_state_dict(resume["criterion_state_dict"], strict=True)
        elif any(parameter.requires_grad for parameter in criterion.parameters()):
            print("WARNING: resume checkpoint has no UWAG criterion state; "
                  "log-scales start from configured initial values")
        if isinstance(resume, dict) and "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            if "scheduler_state_dict" in resume:
                scheduler.load_state_dict(resume["scheduler_state_dict"])
            else:
                print("WARNING: resume checkpoint has no scheduler state; "
                      "the learning-rate schedule restarts")
            if resume.get("scaler_state_dict"):
                scaler.load_state_dict(resume["scaler_state_dict"])
            start_epoch = int(resume.get("epoch", 0))
            best_val = float(resume.get("best_validation_objective", math.inf))
            # Older checkpoints predate these counters.  Starting them at zero
            # is explicit and safe: their old training objective had no B3
            # attempt-indexed schedule to preserve.
            global_update_attempts = int(resume.get("global_update_attempts", 0))
            successful_updates = int(resume.get("successful_updates", 0))
            skipped_updates = int(resume.get("skipped_updates", 0))
        record_resume_event(
            run_dir,
            command_argv=command_argv,
            checkpoint_path=resume_path,
            checkpoint_sha256=resume_digest,
            runtime_controls=runtime_controls,
            initial_manifest=resume_manifest,
            device=device,
        )
        print(f"Resumed from epoch {start_epoch}: {resume_path}")

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is required for training logs. Install dependencies with "
            "`python -m pip install -r requirements-kitti.txt`."
        ) from exc
    tensorboard_dir = run_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir), purge_step=start_epoch + 1)
    log_path = run_dir / "metrics.jsonl"
    csv_path = run_dir / "metrics.csv"
    print(f"Run directory: {run_dir}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Epoch CSV: {csv_path}")
    print(f"Backbone={config['model']['backbone']}; loss={loss_name}; "
          f"precision={precision}")
    print(f"Frames train={len(train_dataset)} val={len(val_dataset)}; "
          f"physical batch={physical_batch_size}; accumulation={accumulation_steps}; "
          f"effective batch={physical_batch_size * accumulation_steps}")
    print(f"Planned optimizer attempts={total_planned_attempts}; "
          f"selection={selection_policy}")

    all_trainable_parameters = model_parameters + criterion_parameters
    loss_regression = config.get("loss", {}).get("regression")

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        training_sum = 0.0
        training_samples = 0
        component_sums: Dict[str, float] = {}
        diagnostic_counts: Dict[str, int] = {}
        nonfinite_loss_count = 0
        epoch_attempts = epoch_successful_updates = epoch_skipped_updates = 0
        gradient_norms = []
        nonfinite_gradient_norm_count = 0
        uwag_gradient_norms = []
        b3_forms = set()
        group_has_nonfinite_loss = group_had_backward = False
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_tensor_batch(batch, device)
            batch_size = int(batch["voxel"].shape[0])
            progress = normalized_training_progress(
                global_update_attempts, total_planned_attempts
            )
            if loss_regression == "probiou":
                b3_forms.add(
                    "bd" if progress < float(config["loss"]["probiou_switch_fraction"])
                    else "hellinger"
                )
            group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, batches_this_epoch - group_start)
            try:
                with autocast_context(device, precision):
                    outputs = model(batch["voxel"])
                    losses = criterion(outputs, batch, progress=progress)
                    objective = losses["loss"]
                    backward_objective = objective / group_size
                if not torch.isfinite(objective).all():
                    raise FloatingPointError("non-finite training objective")
            except FloatingPointError as error:
                # Do not turn a bad loss into a finite value.  The entire
                # accumulation group is discarded at its boundary and counted.
                nonfinite_loss_count += 1
                group_has_nonfinite_loss = True
                print(
                    f"WARNING: non-finite loss at epoch {epoch}, batch {batch_index}; "
                    f"discarding its accumulation group ({error})"
                )
            else:
                scaler.scale(backward_objective).backward()
                group_had_backward = True
                objective_value = float(objective.detach().item())
                training_sum += objective_value * batch_size
                training_samples += batch_size
                accumulate_loss_outputs(
                    component_sums, diagnostic_counts, losses, batch_size=batch_size
                )
            should_update = (batch_index % accumulation_steps == 0
                             or batch_index == batches_this_epoch)
            if should_update:
                global_update_attempts += 1
                epoch_attempts += 1
                if group_had_backward:
                    # ``unscale_`` must precede both our finite-gradient guard
                    # and the norm diagnostic.  It also records FP16 inf state
                    # for ``GradScaler.update`` when a group is skipped.
                    scaler.unscale_(optimizer)
                    gradient_norm = gradient_l2_norm(all_trainable_parameters)
                    gradient_value = float(gradient_norm.detach().cpu())
                    if math.isfinite(gradient_value):
                        gradient_norms.append(gradient_value)
                    else:
                        nonfinite_gradient_norm_count += 1
                    if criterion_parameters:
                        uwag_norm = gradient_l2_norm(criterion_parameters)
                        uwag_value = float(uwag_norm.detach().cpu())
                        if math.isfinite(uwag_value):
                            uwag_gradient_norms.append(uwag_value)
                    gradients_are_finite = all_gradients_finite(all_trainable_parameters)
                else:
                    gradients_are_finite = True

                can_step = (
                    group_had_backward
                    and not group_has_nonfinite_loss
                    and gradients_are_finite
                )
                if can_step:
                    scaler.step(optimizer)
                    scaler.update()
                    successful_updates += 1
                    epoch_successful_updates += 1
                else:
                    skipped_updates += 1
                    epoch_skipped_updates += 1
                    if group_had_backward:
                        # With a manual ``unscale_`` this clears the scaler's
                        # per-optimizer state and reduces FP16 scale on inf.
                        scaler.update()
                optimizer.zero_grad(set_to_none=True)
                group_has_nonfinite_loss = group_had_backward = False
            if batch_index >= batches_this_epoch:
                break

        training_seconds = time.perf_counter() - started
        # A fully skipped epoch has not produced new weights.  Do not validate
        # or checkpoint those stale parameters, especially not as final.pt.
        require_successful_epoch(epoch, epoch_successful_updates)
        validation = validate(model, criterion, val_loader, device, precision,
                              args.max_val_batches,
                              progress=normalized_training_progress(
                                  global_update_attempts, total_planned_attempts
                              ))
        scheduler.step()
        train_objective = training_sum / max(training_samples, 1)
        current_val = float(validation["loss"])
        retained = current_val < best_val
        if retained:
            best_val = current_val
        payload = checkpoint_payload(
            model, criterion, optimizer, scheduler, scaler, epoch,
            validation, best_val, config,
            global_update_attempts=global_update_attempts,
            successful_updates=successful_updates,
            skipped_updates=skipped_updates,
            total_planned_attempts=total_planned_attempts,
        )
        save_checkpoint_with_sha256(payload, checkpoints_dir / "last.pt")
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint_with_sha256(payload, checkpoints_dir / f"{epoch}epoch.pt")
        if retained:
            retained_path = best_dir / f"{epoch}epoch.pt"
            retained_hash = save_checkpoint_with_sha256(payload, retained_path)
            if selection_policy == "best":
                selected_path = loss_selection_dir / "best.pt"
                selected_hash = save_checkpoint_with_sha256(payload, selected_path)
                write_json(loss_selection_dir / "selection.json", {
                    "epoch": epoch, "validation_objective": current_val,
                    "checkpoint": str(retained_path),
                    "checkpoint_sha256": retained_hash,
                    "selected_checkpoint": str(selected_path),
                    "selected_checkpoint_sha256": selected_hash,
                    "criterion": "minimum mean validation loss"
                })
        if selection_policy == "final_epoch" and epoch == epochs:
            selected_path = loss_selection_dir / "final.pt"
            selected_hash = save_checkpoint_with_sha256(payload, selected_path)
            write_json(loss_selection_dir / "selection.json", {
                "epoch": epoch, "validation_objective": current_val,
                "checkpoint": str(selected_path),
                "checkpoint_sha256": selected_hash,
                "selected_checkpoint": str(selected_path),
                "selected_checkpoint_sha256": selected_hash,
                "criterion": "final configured epoch"
            })
        train_components = {
            name: value / max(training_samples, 1)
            for name, value in component_sums.items()
        }
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
               "train_objective": train_objective, "validation": validation,
               "train_components": train_components,
               "training_seconds": training_seconds,
               "optimizer_update_attempts": epoch_attempts,
               "global_update_attempts": global_update_attempts,
               "successful_updates": successful_updates,
               "skipped_updates": skipped_updates,
               "epoch_successful_updates": epoch_successful_updates,
               "epoch_skipped_updates": epoch_skipped_updates,
               "nonfinite_loss_count": nonfinite_loss_count,
               "log_size_clamp_count": diagnostic_counts.get("clamp_count", 0),
               "gradient_l2_norm_mean": (
                   sum(gradient_norms) / len(gradient_norms) if gradient_norms else None
               ),
               "gradient_l2_norm_max": max(gradient_norms) if gradient_norms else None,
               "nonfinite_gradient_norm_count": nonfinite_gradient_norm_count,
               "uwag_gradient_l2_norm_mean": (
                   sum(uwag_gradient_norms) / len(uwag_gradient_norms)
                   if uwag_gradient_norms else None
               ),
               "probiou_form": "/".join(sorted(b3_forms)) if b3_forms else None,
               "precision": precision, "loss": loss_name, "retained": retained}
        if device.type == "cuda":
            row["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            row["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        if getattr(criterion, "name", None) == "uwag":
            log_scales = criterion.log_scales.detach().cpu()
            row["uwag_log_scales"] = [float(value) for value in log_scales]
            row["uwag_weights"] = [float(value) for value in torch.exp(-log_scales)]
        if loss_regression == "mgiou":
            row["yaw_fallback_count"] = diagnostic_counts.get("yaw_fallback_count", 0)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        csv_row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value for name, value in sorted(train_components.items())},
            **{f"val_{name}": value for name, value in sorted(validation.items())},
            "training_seconds": training_seconds,
            "optimizer_update_attempts": epoch_attempts,
            "global_update_attempts": global_update_attempts,
            "successful_updates": successful_updates,
            "skipped_updates": skipped_updates,
            "best_validation_objective": best_val,
            "retained": retained,
        }
        append_csv_row(csv_path, csv_row)
        for name, value in train_components.items():
            writer.add_scalar(f"train/{name}", value, epoch)
        for name, value in validation.items():
            scalar = float(value)
            if math.isfinite(scalar):
                writer.add_scalar(f"validation/{name}", scalar, epoch)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("train/optimizer_update_attempts", epoch_attempts, epoch)
        writer.add_scalar("train/successful_updates", successful_updates, epoch)
        writer.add_scalar("train/skipped_updates", skipped_updates, epoch)
        writer.add_scalar("checkpoint/best_validation_objective", best_val, epoch)
        writer.flush()
        print(f"Epoch {epoch:03d}/{epochs} train={train_objective:.6f} "
              f"val={current_val:.6f} retained={retained} "
              f"train_s={training_seconds:.1f} val_s={validation['seconds']:.1f}")

    writer.close()
    print(f"Selected checkpoint: {loss_selection_dir / selected_filename}")
    print(f"Selection record: {loss_selection_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
