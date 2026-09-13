#!/usr/bin/env python3
"""Train a configured KITTI MobilePIXOR variant with best-checkpoint selection."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (atomic_torch_save, build_model, configure_detector_imports,
                    normalize_state_dict, read_json, write_json)


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


def checkpoint_payload(model, criterion, optimizer, scheduler, scaler, epoch: int,
                       validation: Dict[str, float], best_val: float,
                       config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(), "validation": validation,
        "best_validation_objective": best_val, "config": config,
        "saved_at_utc": datetime.now(timezone.utc).isoformat()
    }


@torch.no_grad()
def validate(model, criterion, loader, device, precision, max_batches=0):
    model.eval()
    criterion.eval()
    sums: Dict[str, float] = {}
    samples = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_tensor_batch(batch, device)
        batch_size = int(batch["voxel"].shape[0])
        with autocast_context(device, precision):
            outputs = model(batch["voxel"])
            losses = criterion(outputs, batch)
        for name, value in losses.items():
            scalar = float(value.item() if torch.is_tensor(value) else value)
            sums[name] = sums.get(name, 0.0) + scalar * batch_size
        samples += batch_size
        if max_batches and batch_index >= max_batches:
            break
    if not samples:
        raise RuntimeError("Validation loader produced no samples")
    metrics = {name: value / samples for name, value in sums.items()}
    metrics.update(seconds=time.perf_counter() - started, samples=samples)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--detector-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-name")
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

    config = read_json(args.config)
    seed = int(config.get("seed", 42))
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

    physical_batch_size = int(args.physical_batch_size or
        config["train"].get("physical_batch_size") or config["train"].get("batch_size", 2))
    accumulation_steps = int(args.accumulation_steps or
        config["train"].get("accumulation_steps", 1))
    if physical_batch_size < 1 or accumulation_steps < 1:
        raise ValueError("Batch size and accumulation steps must be positive")
    epochs = int(args.epochs or config["train"]["epochs"])

    train_dataset = Dataset(config["train"]["data"], config["data"],
        config["augmentation"], config["model"]["cls_encoding"], "train")
    # A non-special task name disables augmentation and list-valued visualisation data.
    val_dataset = Dataset(config["val"]["data"], config["data"],
        config["augmentation"], config["model"]["cls_encoding"], "validation")
    generator = torch.Generator().manual_seed(seed)
    common_loader = loader_kwargs(args.num_workers, device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=physical_batch_size,
        shuffle=True, generator=generator, **common_loader)
    val_loader = DataLoader(val_dataset,
        batch_size=int(config["val"].get("physical_batch_size", physical_batch_size)),
        shuffle=False, **common_loader)

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

    loss_name = config.get("loss", {}).get("name", "baseline")
    default_name = (
        f"{config['model']['backbone']}_{loss_name}_{precision}_"
        f"seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = args.output_root.expanduser().resolve() / (args.run_name or default_name)
    checkpoints_dir, best_dir, selected_dir = (run_dir / "checkpoints",
                                                run_dir / "best_checkpoints",
                                                run_dir / "selected")
    for path in (checkpoints_dir, best_dir, selected_dir):
        path.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.resolved.json", config)

    start_epoch, best_val = 0, math.inf
    if args.resume:
        resume = torch.load(args.resume, map_location=device)
        model.load_state_dict(normalize_state_dict(resume), strict=True)
        if isinstance(resume, dict) and resume.get("criterion_state_dict") is not None:
            criterion.load_state_dict(resume["criterion_state_dict"], strict=True)
        elif any(parameter.requires_grad for parameter in criterion.parameters()):
            print("WARNING: resume checkpoint has no UWAG criterion state; "
                  "log-scales start from configured initial values")
        if isinstance(resume, dict) and "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            scheduler.load_state_dict(resume["scheduler_state_dict"])
            if resume.get("scaler_state_dict"):
                scaler.load_state_dict(resume["scaler_state_dict"])
            start_epoch = int(resume.get("epoch", 0))
            best_val = float(resume.get("best_validation_objective", math.inf))
        print(f"Resumed from epoch {start_epoch}: {args.resume}")

    log_path = run_dir / "metrics.jsonl"
    print(f"Run directory: {run_dir}")
    print(f"Backbone={config['model']['backbone']}; loss={loss_name}; "
          f"precision={precision}")
    print(f"Frames train={len(train_dataset)} val={len(val_dataset)}; "
          f"physical batch={physical_batch_size}; accumulation={accumulation_steps}; "
          f"effective batch={physical_batch_size * accumulation_steps}")

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        training_sum = 0.0
        training_samples = update_count = 0
        started = time.perf_counter()
        total_batches = len(train_loader)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_tensor_batch(batch, device)
            batch_size = int(batch["voxel"].shape[0])
            with autocast_context(device, precision):
                outputs = model(batch["voxel"])
                losses = criterion(outputs, batch)
                objective = losses["loss"]
                backward_objective = objective / accumulation_steps
            scaler.scale(backward_objective).backward()
            should_update = batch_index % accumulation_steps == 0 or batch_index == total_batches
            if should_update:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # A reduced scale means GradScaler detected non-finite gradients
                # and deliberately skipped optimizer.step().
                if not scaler_enabled or scaler.get_scale() >= previous_scale:
                    update_count += 1
            training_sum += float(objective.detach().item()) * batch_size
            training_samples += batch_size
            if args.max_train_batches and batch_index >= args.max_train_batches:
                if not should_update:
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if not scaler_enabled or scaler.get_scale() >= previous_scale:
                        update_count += 1
                break

        training_seconds = time.perf_counter() - started
        validation = validate(model, criterion, val_loader, device, precision,
                              args.max_val_batches)
        if update_count:
            scheduler.step()
        else:
            print(f"WARNING: epoch {epoch} had no finite optimizer update; LR scheduler not advanced")
        train_objective = training_sum / max(training_samples, 1)
        current_val = float(validation["loss"])
        retained = current_val < best_val
        if retained:
            best_val = current_val
        payload = checkpoint_payload(
            model, criterion, optimizer, scheduler, scaler, epoch,
            validation, best_val, config
        )
        save_every = int(config["train"].get("save_every", 5))
        if epoch % save_every == 0 or epoch == epochs:
            atomic_torch_save(payload, checkpoints_dir / f"{epoch}epoch.pt")
        if retained:
            retained_path = best_dir / f"{epoch}epoch.pt"
            atomic_torch_save(payload, retained_path)
            atomic_torch_save(payload, selected_dir / "best.pt")
            write_json(selected_dir / "selection.json", {
                "epoch": epoch, "validation_objective": current_val,
                "checkpoint": str(retained_path),
                "criterion": "minimum mean validation loss"
            })
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
               "train_objective": train_objective, "validation": validation,
               "training_seconds": training_seconds, "optimizer_updates": update_count,
               "precision": precision, "loss": loss_name, "retained": retained}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"Epoch {epoch:03d}/{epochs} train={train_objective:.6f} "
              f"val={current_val:.6f} retained={retained} "
              f"train_s={training_seconds:.1f} val_s={validation['seconds']:.1f}")

    print(f"Selected checkpoint: {selected_dir / 'best.pt'}")
    print(f"Selection record: {selected_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
