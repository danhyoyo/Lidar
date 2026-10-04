#!/usr/bin/env python3
"""Train a configured KITTI MobilePIXOR variant with best-checkpoint selection."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (
    atomic_torch_save,
    build_model,
    configure_detector_imports,
    generate_run_name,
    normalize_state_dict,
    read_json,
    write_json,
)


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
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def loader_kwargs(num_workers: int, pin_memory: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        result.update(persistent_workers=True, prefetch_factor=2)
    return result


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True)


def synchronize_device(device: torch.device) -> None:
    """Make wall-clock measurements include queued CUDA work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpoint_payload(
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    validation: Dict[str, float],
    best_val: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    checkpoint_model = getattr(model, "_orig_mod", model)
    return {
        "epoch": epoch,
        # torch.compile wraps the model.  Persist the original state-dict so a
        # checkpoint remains resumable with and without --compile-model.
        "model_state_dict": checkpoint_model.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "validation": validation,
        "best_validation_objective": best_val,
        "config": config,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@torch.no_grad()
def validate(
    model,
    criterion,
    loader,
    device,
    precision,
    max_batches: int = 0,
) -> Dict[str, Any]:
    model.eval()
    criterion.eval()
    sums: Dict[str, torch.Tensor] = {}
    samples = 0
    synchronize_device(device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_tensor_batch(batch, device)
        batch_size = int(batch["voxel"].shape[0])
        with autocast_context(device, precision):
            outputs = model(batch["voxel"])
            losses = criterion(outputs, batch)
        for name, value in losses.items():
            scalar = value.detach() if torch.is_tensor(value) else torch.as_tensor(
                value, device=device
            )
            sums[name] = sums.get(name, torch.zeros_like(scalar)) + scalar * batch_size
        samples += batch_size
        if max_batches and batch_index >= max_batches:
            break
    if not samples:
        raise RuntimeError("Validation loader produced no samples")
    synchronize_device(device)
    metrics = {name: float((value / samples).cpu()) for name, value in sums.items()}
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
    parser.add_argument("--stop-after-epoch", type=int,
                        help="stop at an absolute epoch without shortening the LR schedule; override train.stop_after_epoch")
    parser.add_argument("--grad-clip-norm", type=float,
                        help="override train.grad_clip_norm (default 10); zero disables clipping")
    parser.add_argument("--physical-batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument(
        "--target-backend",
        choices=("python", "numba"),
        default="python",
        help="target-map backend; Python is the compatibility default",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="opt in to torch.compile after checkpoint loading",
    )
    parser.add_argument("--amp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--val-physical-batch-size", type=int, default=None, help="override validation physical batch size")
    return parser


def resolve_training_epochs(train_cfg, epochs_override=None, stop_override=None):
    epochs = int(epochs_override if epochs_override is not None else train_cfg["epochs"])
    stop_epoch = (int(stop_override) if stop_override is not None
                  else min(int(train_cfg.get("stop_after_epoch", epochs)), epochs))
    if epochs < 1 or not 1 <= stop_epoch <= epochs:
        raise ValueError("Require 1 <= stop_after_epoch <= epochs")
    return epochs, stop_epoch


def clip_optimizer_gradients(optimizer, scaler, max_norm):
    if max_norm > 0:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        parameters = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def configure_matmul_precision() -> None:
    if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def build_optimizer(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
    train_cfg = config.get("train", {})
    opt_type = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0001))

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    crit_params = [p for p in criterion.parameters() if p.requires_grad]
    if crit_params:
        groups.append({"params": crit_params, "weight_decay": 0.0})

    if opt_type == "adam":
        return torch.optim.Adam(groups, lr=lr)
    elif opt_type == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    train_cfg = config.get("train", {})
    sched_type = train_cfg.get("scheduler", "cosine").lower()
    base_lr = float(train_cfg["learning_rate"])

    if sched_type == "cosine":
        warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
        min_lr = float(train_cfg.get("min_lr", 1e-6))
        if warmup_epochs > 0:
            start_factor = min(1.0, max(1e-4, min_lr / base_lr))
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=min_lr,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=min_lr,
            )
    elif sched_type == "multistep":
        milestones = list(train_cfg.get("lr_decay_at", [65, 85]))
        gamma = float(train_cfg.get("lr_decay_gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")


def main(argv=None) -> None:
    configure_matmul_precision()
    args = build_parser().parse_args(argv)
    configure_detector_imports(args.detector_root)
    from core.datasets.dataset import Dataset
    from core.losses.loss_fn import LossFunction

    config = read_json(args.config)
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

    precision = str(args.precision or config["train"].get("precision", "fp32")).lower()
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
    config["train"]["target_backend"] = args.target_backend
    config["train"]["compile_model"] = args.compile_model
    config["train"]["num_workers"] = args.num_workers
    scaler_enabled = precision == "fp16"
    grad_clip_norm = float(args.grad_clip_norm if args.grad_clip_norm is not None
                           else config["train"].get("grad_clip_norm", 10.0))
    if not math.isfinite(grad_clip_norm) or grad_clip_norm < 0:
        raise ValueError("grad_clip_norm must be finite and non-negative")
    config["train"]["grad_clip_norm"] = grad_clip_norm

    physical_batch_size = int(
        args.physical_batch_size
        if args.physical_batch_size is not None
        else config["train"].get("physical_batch_size")
        or config["train"].get("batch_size", 2)
    )
    accumulation_steps = int(
        args.accumulation_steps
        if args.accumulation_steps is not None
        else config["train"].get("accumulation_steps", 1)
    )
    if physical_batch_size < 1 or accumulation_steps < 1:
        raise ValueError("Batch size and accumulation steps must be positive")
    epochs, stop_epoch = resolve_training_epochs(config["train"], args.epochs, args.stop_after_epoch)
    config["train"].update(epochs=epochs, stop_after_epoch=stop_epoch,
                           physical_batch_size=physical_batch_size, accumulation_steps=accumulation_steps)
    save_every = int(config["train"].get("save_every", 5))
    if save_every < 1:
        raise ValueError("save_every must be positive")

    train_dataset = Dataset(
        config["train"]["data"],
        config["data"],
        config["augmentation"],
        config["model"]["cls_encoding"],
        "train",
        args.target_backend,
    )
    # A non-special task name disables augmentation and list-valued visualisation data.
    val_dataset = Dataset(
        config["val"]["data"],
        config["data"],
        config["augmentation"],
        config["model"]["cls_encoding"],
        "validation",
        args.target_backend,
    )
    generator = torch.Generator().manual_seed(seed)
    common_loader = loader_kwargs(args.num_workers, device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=physical_batch_size,
        shuffle=True,
        generator=generator,
        **common_loader,
    )
    val_bs_config = config.get("val", {}).get("physical_batch_size", physical_batch_size)
    validation_batch_size = int(
        args.val_physical_batch_size
        if args.val_physical_batch_size is not None
        else val_bs_config
    )
    if validation_batch_size < 1:
        raise ValueError("validation physical_batch_size must be positive")
    val_loader = DataLoader(
        val_dataset, batch_size=validation_batch_size, shuffle=False, **common_loader
    )

    model = build_model(config).to(device)
    criterion = LossFunction(config["model"]["cls_encoding"], config.get("loss")).to(
        device
    )
    optimizer = build_optimizer(model, criterion, config)
    scheduler = build_scheduler(optimizer, config, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    loss_name = config.get("loss", {}).get("name", "baseline")
    default_name = generate_run_name(config, seed=seed)
    run_dir = args.output_root.expanduser().resolve() / (args.run_name or default_name)
    checkpoints_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best_checkpoints"
    loss_selection_dir = run_dir / "selected"
    for path in (checkpoints_dir, best_dir, loss_selection_dir):
        path.mkdir(parents=True, exist_ok=True)
    train_log_path = run_dir / "train.log"

    def log_line(text: str) -> None:
        print(text, flush=True)
        with train_log_path.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    start_epoch, best_val = 0, math.inf
    if args.resume:
        resume = torch.load(args.resume, map_location=device)
        if isinstance(resume, dict):
            saved_train = resume.get("config", {}).get("train", {})
            if "stop_after_epoch" in saved_train and int(saved_train["epochs"]) != epochs:
                raise ValueError("Resume must preserve the checkpoint's LR schedule horizon; change stop_after_epoch instead")
        model.load_state_dict(normalize_state_dict(resume), strict=True)
        if isinstance(resume, dict) and resume.get("criterion_state_dict") is not None:
            criterion.load_state_dict(resume["criterion_state_dict"], strict=True)
        elif any(parameter.requires_grad for parameter in criterion.parameters()):
            print(
                "WARNING: resume checkpoint has no UWAG criterion state; "
                "log-scales start from configured initial values"
            )
        if isinstance(resume, dict) and "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            if "scheduler_state_dict" in resume:
                scheduler.load_state_dict(resume["scheduler_state_dict"])
            else:
                print(
                    "WARNING: resume checkpoint has no scheduler state; "
                    "the learning-rate schedule restarts"
                )
            if resume.get("scaler_state_dict"):
                scaler.load_state_dict(resume["scaler_state_dict"])
            start_epoch = int(resume.get("epoch", 0))
            best_val = float(resume.get("best_validation_objective", math.inf))
        log_line(f"Resumed from epoch {start_epoch}: {args.resume}")

    write_json(run_dir / "config.resolved.json", config)

    if args.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile-model requires PyTorch 2.0 or newer")
        logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
        model = torch.compile(model)
        try:
            print("Warming up torch.compile kernels...")
            geom = config["data"]["kitti"]["geometry"]
            in_ch = 8 if config.get("data", {}).get("bev_encoding", {}).get("name") == "rich8" else 35
            h = int(round((geom["x_max"] - geom["x_min"]) / geom["x_res"]))
            w = int(round((geom["y_max"] - geom["y_min"]) / geom["y_res"]))
            dummy_voxel = torch.zeros((physical_batch_size, in_ch, h, w), device=device)
            with autocast_context(device, precision):
                dummy_out = model(dummy_voxel)
                if isinstance(dummy_out, dict) and "cls" in dummy_out:
                    dummy_loss = dummy_out["cls"].sum()
                    dummy_loss.backward()
            optimizer.zero_grad(set_to_none=True)
            synchronize_device(device)
            print("Warmup torch.compile completed.")
        except Exception as exc:
            print(f"torch.compile warmup skipped: {exc}")

    log_path = run_dir / "metrics.jsonl"
    log_line(f"Run directory: {run_dir}")
    log_line(
        f"Backbone={config['model']['backbone']}; loss={loss_name}; "
        f"precision={precision}; grad_clip_norm={grad_clip_norm}; "
        f"schedule_epochs={epochs}; stop_after_epoch={stop_epoch}"
    )
    log_line(
        f"Frames train={len(train_dataset)} val={len(val_dataset)}; "
        f"physical batch={physical_batch_size}; accumulation={accumulation_steps}; "
        f"effective batch={physical_batch_size * accumulation_steps}"
    )

    for epoch in range(start_epoch + 1, stop_epoch + 1):
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        training_sum = torch.zeros((), device=device)
        training_samples = update_count = 0
        synchronize_device(device)
        started = time.perf_counter()
        total_batches = len(train_loader)
        batches_this_epoch = (
            min(total_batches, args.max_train_batches)
            if args.max_train_batches
            else total_batches
        )
        if batches_this_epoch == 0:
            raise RuntimeError("Training loader produced no batches")
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_tensor_batch(batch, device)
            batch_size = int(batch["voxel"].shape[0])
            with autocast_context(device, precision):
                outputs = model(batch["voxel"])
                losses = criterion(outputs, batch)
                objective = losses["loss"]
                group_start = (
                    (batch_index - 1) // accumulation_steps
                ) * accumulation_steps
                group_size = min(accumulation_steps, batches_this_epoch - group_start)
                backward_objective = objective / group_size
            scaler.scale(backward_objective).backward()
            should_update = (
                batch_index % accumulation_steps == 0
                or batch_index == batches_this_epoch
            )
            if should_update:
                clip_optimizer_gradients(optimizer, scaler, grad_clip_norm)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # A reduced scale means GradScaler detected non-finite gradients
                # and deliberately skipped optimizer.step().
                if not scaler_enabled or scaler.get_scale() >= previous_scale:
                    update_count += 1
            training_sum = training_sum + objective.detach() * batch_size
            training_samples += batch_size
            if batch_index >= batches_this_epoch:
                break

        synchronize_device(device)
        training_seconds = time.perf_counter() - started
        validation = validate(
            model,
            criterion,
            val_loader,
            device,
            precision,
            args.max_val_batches,
        )
        if update_count:
            scheduler.step()
        else:
            print(
                f"WARNING: epoch {epoch} had no finite optimizer update; LR scheduler not advanced",
                flush=True,
            )
        train_objective = float((training_sum / max(training_samples, 1)).cpu())
        current_val = float(validation["loss"])
        retained = current_val < best_val
        if retained:
            best_val = current_val
        payload = checkpoint_payload(
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            epoch,
            validation,
            best_val,
            config,
        )
        atomic_torch_save(payload, checkpoints_dir / "last.pt")
        if epoch % save_every == 0 or epoch == stop_epoch:
            atomic_torch_save(payload, checkpoints_dir / f"{epoch}epoch.pt")
        if retained:
            retained_path = best_dir / f"{epoch}epoch.pt"
            atomic_torch_save(payload, retained_path)
            atomic_torch_save(payload, loss_selection_dir / "best.pt")
            write_json(
                loss_selection_dir / "selection.json",
                {
                    "epoch": epoch,
                    "validation_objective": current_val,
                    "checkpoint": str(retained_path),
                    "criterion": "minimum mean validation loss",
                },
            )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_objective": train_objective,
            "validation": validation,
            "training_seconds": training_seconds,
            "optimizer_updates": update_count,
            "precision": precision,
            "loss": loss_name,
            "retained": retained,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        retained_flag = " [BEST]" if retained else ""
        epoch_summary = (
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: {train_objective:.4f} | "
            f"Val Loss: {current_val:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: train={training_seconds:.1f}s, val={validation['seconds']:.1f}s{retained_flag}"
        )
        log_line(epoch_summary)

    log_line(f"Selected checkpoint: {loss_selection_dir / 'best.pt'}")
    log_line(f"Selection record: {loss_selection_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
