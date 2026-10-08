#!/usr/bin/env python3
"""Train a configured KITTI MobilePIXOR variant with best-checkpoint selection."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
import tempfile
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
    checkpoint_backends,
    checkpoint_identity,
    detection_spec,
    generate_run_name,
    input_shape,
    dummy_model_input,
    model_input_batch_size,
    normalize_state_dict,
    read_json,
    sha256,
    warm_start_backbone,
    write_json,
)
from checkpoint_selection import (selection_settings, ap_due, ap_measurement,
                                  advance_ap_state, save_selections)


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
    """Transfer tensor leaves without converting or flattening sample metadata."""
    import copy
    from collections.abc import Mapping, MutableMapping

    def move(value):
        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)
        if isinstance(value, Mapping):
            items = {key: move(leaf) for key, leaf in value.items()}
            if isinstance(value, MutableMapping):
                result = copy.copy(value)
                result.update(items)
                return result
            return type(value)(items)
        if isinstance(value, list):
            return [move(leaf) for leaf in value]
        if isinstance(value, tuple):
            items = tuple(move(leaf) for leaf in value)
            return type(value)(*items) if hasattr(value, "_fields") else items
        return value

    return move(batch)


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


def set_loss_epoch(criterion: torch.nn.Module, epoch: int) -> None:
    """Set zero-based curriculum epoch when supported by the loss strategy."""
    setter = getattr(criterion, "set_epoch", None)
    if setter is not None:
        setter(epoch)


def build_training_criterion(config, device):
    """Resolve the objective's group metadata and place it before the optimizer."""
    from core.losses.loss_fn import build_loss_function

    spec = detection_spec(config)
    return build_loss_function(spec.cls_encoding, config.get("loss"),
                               task_groups=spec.groups if spec.head_mode == "grouped" else None,
                               head_mode=spec.head_mode, box_mode=spec.box_mode).to(device)


def warmup_model(model, voxel, optimizer, device, precision):
    """Compile every prediction branch without changing BN or criterion state."""
    from collections.abc import Mapping

    def tensor_sums(value):
        if torch.is_tensor(value):
            yield value.float().sum()
        elif isinstance(value, Mapping):
            for leaf in value.values():
                yield from tensor_sums(leaf)

    was_training = model.training
    model.eval()
    try:
        with autocast_context(device, precision):
            terms = list(tensor_sums(model(voxel)))
            if not terms:
                raise ValueError("Compile warmup requires tensor predictions")
            sum(terms).backward()
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.train(was_training)


class QualityMetrics:
    """Aggregate global/group quality by peak or regression-cell counts."""

    MEANS = ("quality_iou_mean", "quality_iou_zero_fraction", "quality_target_mean")

    def __init__(self):
        self.streams = {}

    @staticmethod
    def is_quality_metric(name):
        leaf = name.rsplit("/", 1)[-1]
        return leaf.startswith("quality_") or leaf in ("mean_iou_target", "iou_target_count")

    def update(self, losses, targets=None):
        # The legacy OGA facade exposes a mean but no sufficient count.
        # Derive it from its flat assigned mask without changing loss outputs.
        if ("mean_iou_target" in losses and "iou_target_count" not in losses
                and targets is not None and "reg_mask" in targets):
            losses = {**losses, "iou_target_count": targets["reg_mask"].bool().sum().float()}
        for key, value in losses.items():
            leaf = key.rsplit("/", 1)[-1]
            if leaf not in ("quality_peak_count", "iou_target_count"):
                continue
            prefix = key[:-len(leaf)]
            means = self.MEANS if leaf == "quality_peak_count" else ("mean_iou_target",)
            count = value.detach().float()
            stream = self.streams.setdefault(key, {"count": torch.zeros_like(count), "sums": {}, "mix": None})
            if leaf == "quality_peak_count":
                mix = losses[prefix + "quality_iou_mix"].detach()
                if stream["mix"] is not None and not torch.equal(stream["mix"], mix):
                    raise ValueError("Q-OGA curriculum mix must agree within an epoch")
                stream["mix"] = mix
            stream["count"] = stream["count"] + count
            for name in means:
                weighted = losses[prefix + name].detach().float() * count
                stream["sums"][name] = stream["sums"].get(name, torch.zeros_like(weighted)) + weighted

    def summarize(self):
        result = {}
        for key, stream in self.streams.items():
            leaf = key.rsplit("/", 1)[-1]
            prefix = key[:-len(leaf)]
            result[key] = float(stream["count"].cpu())
            result.update({prefix + name: float((value / stream["count"].clamp_min(1.)).cpu())
                           for name, value in stream["sums"].items()})
            if stream["mix"] is not None:
                result[prefix + "quality_iou_mix"] = float(stream["mix"].cpu())
        return result


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
    *,
    loader_generator=None,
    ap_selection=None,
) -> Dict[str, Any]:
    checkpoint_model = getattr(model, "_orig_mod", model)
    payload = {
        "epoch": epoch,
        # torch.compile wraps the model.  Persist the original state-dict so a
        # checkpoint remains resumable with and without --compile-model.
        "model_state_dict": checkpoint_model.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_initialized_parameters": sorted(optimizer.state_dict()["state"]),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "validation": validation,
        "best_validation_objective": best_val,
        "config": config,
        "checkpoint_identity": checkpoint_identity(config),
        "backends": checkpoint_backends(config),
        "training_state_types": {"optimizer": type(optimizer).__name__,
                                 "scheduler": type(scheduler).__name__},
        "rng_state": capture_rng_state(loader_generator),
        "replay": {"boundary": "epoch_end", "num_workers": config.get("train", {}).get("num_workers", 0),
                   "deterministic_worker_replay": config.get("train", {}).get("num_workers", 0) == 0},
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if ap_selection is not None:
        payload['ap_selection'] = ap_selection
    return payload


def evaluate_training_ap(payload, config_path, detector_root, device, loader_generator, run_dir):
    """Use the existing full-split FP32 evaluator without advancing training RNG."""
    rng = capture_rng_state(loader_generator)
    started = time.perf_counter()
    try:
        from evaluate_kitti_bev import read_ids
        config = payload['config']; settings = selection_settings(config)
        evaluation = config.get('evaluation', {})
        raw = Path(evaluation['kitti_root']).expanduser().resolve()
        if raw.name == 'training':
            raw = raw.parent
        split = Path(config['val']['data']).expanduser()
        if not split.is_absolute():
            split = Path(__file__).resolve().parents[2] / split
        ids = read_ids(split)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('AP validation requires a nonempty unique full validation split')
        if settings['metric_mode'] == 'local_bev':
            from evaluate_kitti_bev import run_evaluation
            options = {}
        else:
            from evaluate_kitti_3d import run_evaluation
            raw = raw / 'training'
            options = {'metric_mode': settings['metric_mode']}
        with tempfile.TemporaryDirectory(prefix='ap-validation-', dir=run_dir) as temporary:
            candidate = Path(temporary) / 'candidate.pt'
            atomic_torch_save(payload, candidate)
            report = run_evaluation(name='validation_ap', backend='pytorch', model_path=candidate,
                config_path=config_path, detector_root=detector_root, kitti_root=raw,
                split_path=split, device=str(device), warmup_frames=0, progress_every=50,
                score_threshold=evaluation.get('score_threshold', .05),
                nms_threshold=evaluation.get('nms_threshold', .10),
                max_detections=evaluation.get('max_detections', 500), **options)
        measurement = ap_measurement(report, settings, full_frames=len(ids), split_sha256=sha256(split))
        return measurement, time.perf_counter() - started
    finally:
        restore_rng_state(rng, loader_generator)


def capture_rng_state(loader_generator):
    """Use tensors/primitives so checkpoints support weights-only torch.load."""
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": numpy_state[0],
                      "keys": torch.tensor(numpy_state[1].astype(np.int64)),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_generator": loader_generator.get_state() if loader_generator is not None else None}


def validate_rng_state(state, loader_generator):
    """Preflight with private RNGs, without advancing any global stream."""
    try:
        random.Random().setstate(state["python"])
        numpy = state["numpy"]
        numpy_state = (numpy["algorithm"], numpy["keys"].cpu().numpy().astype(np.uint32),
                       numpy["position"], numpy["has_gauss"], numpy["cached_gaussian"])
        np.random.RandomState().set_state(numpy_state)
        torch.Generator().set_state(state["torch_cpu"].cpu())
        if loader_generator is None:
            raise ValueError("Full resume requires the train DataLoader generator")
        torch.Generator().set_state(state["loader_generator"].cpu())
        cuda_states = state["torch_cuda"]
        if not isinstance(cuda_states, list):
            raise ValueError("Invalid CUDA RNG list")
        if cuda_states:
            if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
                raise ValueError("CUDA RNG restoration requires matching visible CUDA devices")
            for index, value in enumerate(cuda_states):
                torch.Generator(device=f"cuda:{index}").set_state(value.cpu())
        elif torch.cuda.is_available():
            raise ValueError("CUDA resume requires saved CUDA RNG state")
    except (KeyError, TypeError, AttributeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid or incompatible checkpoint RNG state: {exc}") from exc
    return numpy_state


def restore_rng_state(state, loader_generator):
    numpy_state = validate_rng_state(state, loader_generator)
    random.setstate(state["python"])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    loader_generator.set_state(state["loader_generator"].cpu())


def _validate_module_state(module, state, label):
    """Require complete buffers even when legacy module loaders allow omissions."""
    expected = module.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError(f"Incomplete or incompatible {label} state keys")
    for key, value in expected.items():
        saved = state[key]
        if (not torch.is_tensor(saved) or saved.shape != value.shape or saved.dtype != value.dtype):
            raise ValueError(f"Incompatible {label} tensor: {key}")


def restore_checkpoint(checkpoint, config, model, criterion, optimizer, scheduler, scaler,
                       *, loader_generator=None, backend_parity_verified=False):
    """Validate semantics and all training state before changing live modules.

    Pure legacy weights are supported with a matching legacy topology and start
    at epoch zero. Incomplete training checkpoints cannot claim full resume.
    """
    original_model = getattr(model, "_orig_mod", model)
    model_state = normalize_state_dict(checkpoint)
    training_keys = {"optimizer_state_dict", "criterion_state_dict", "scheduler_state_dict",
                     "scaler_state_dict", "rng_state", "checkpoint_identity"}
    full_resume = isinstance(checkpoint, dict) and bool(training_keys & set(checkpoint))
    if not full_resume:
        if detection_spec(config).head_mode != "legacy_single":
            raise ValueError("Grouped resume requires checkpoint identity; legacy weights need legacy config")
        if isinstance(checkpoint.get("config"), dict):
            if checkpoint_identity(checkpoint["config"]) != checkpoint_identity(config):
                raise ValueError("Legacy checkpoint identity does not match config")
        _validate_module_state(original_model, model_state, "legacy model")
        original_model.load_state_dict(model_state, strict=True)
        return {"mode": "legacy_weights", "epoch": 0, "best_val": math.inf, "backend_changes": {}}

    required = training_keys | {"epoch", "best_validation_objective", "config", "backends",
                                "training_state_types", "optimizer_initialized_parameters"}
    if not required <= set(checkpoint):
        raise ValueError(f"Incomplete full-resume checkpoint: missing {sorted(required - set(checkpoint))}")
    current_identity = checkpoint_identity(config)
    if (checkpoint["checkpoint_identity"] != current_identity or
            checkpoint_identity(checkpoint["config"]) != checkpoint["checkpoint_identity"]):
        raise ValueError("Checkpoint architecture/objective/training identity does not match config")
    current_backends = checkpoint_backends(config)
    if checkpoint["backends"] != checkpoint_backends(checkpoint["config"]):
        raise ValueError("Incomplete or inconsistent checkpoint backend metadata")
    backend_changes = {name: {"saved": checkpoint["backends"][name], "current": backend}
                       for name, backend in current_backends.items() if checkpoint["backends"][name] != backend}
    if backend_changes and not backend_parity_verified:
        raise ValueError("Backend change requires explicitly verified numerical parity")
    if checkpoint["training_state_types"] != {"optimizer": type(optimizer).__name__,
                                               "scheduler": type(scheduler).__name__}:
        raise ValueError("Incompatible optimizer/scheduler resume mode")
    if type(checkpoint["epoch"]) is not int or checkpoint["epoch"] < 0:
        raise ValueError("Invalid checkpoint epoch")
    best_val = float(checkpoint["best_validation_objective"])
    if math.isnan(best_val):
        raise ValueError("Invalid checkpoint best validation objective")
    settings = selection_settings(config)
    if settings['primary'] == 'ap':
        state = checkpoint.get('ap_selection')
        if not isinstance(state, dict):
            raise ValueError('AP resume requires complete checkpoint AP selection state')
        advance_ap_state(state, None, checkpoint['epoch'], settings)
        best_epoch = state.get('best_epoch')
        if best_epoch is not None:
            if type(best_epoch) is not int or not 1 <= best_epoch <= checkpoint['epoch'] or not ap_due(best_epoch, config['train']['epochs'], settings):
                raise ValueError('Invalid checkpoint best AP epoch')
            checked, _ = advance_ap_state(None, state.get('best_measurement'), best_epoch, settings)
            if checked != state:
                raise ValueError('Invalid checkpoint best AP measurement/state')
        elif state.get('best_score') is not None or state.get('best_measurement') is not None or any(
                ap_due(e, config['train']['epochs'], settings) for e in range(1, checkpoint['epoch'] + 1)):
            raise ValueError('Checkpoint is missing scheduled AP selection state')
    _validate_module_state(original_model, model_state, "model")
    _validate_module_state(criterion, checkpoint["criterion_state_dict"], "criterion")
    for key, value in checkpoint["criterion_state_dict"].items():
        if key.endswith("quality_epoch") and value.item() != checkpoint["epoch"] - 1:
            raise ValueError("Checkpoint curriculum epoch does not match saved training epoch")
    validate_rng_state(checkpoint["rng_state"], loader_generator)
    # Some PyTorch loaders accept incomplete dictionaries. Validate topology
    # and exercise loaders on copies before touching the actual optimizer.
    saved_optimizer = checkpoint["optimizer_state_dict"]
    current_optimizer = optimizer.state_dict()
    if set(saved_optimizer) != set(current_optimizer):
        raise ValueError("Incomplete optimizer state")
    if sorted(saved_optimizer["state"]) != checkpoint["optimizer_initialized_parameters"]:
        raise ValueError("Incomplete optimizer initialized parameter state")
    parameter_ids = {key for group in current_optimizer["param_groups"] for key in group["params"]}
    if not set(saved_optimizer["state"]) <= parameter_ids:
        raise ValueError("Unexpected optimizer parameter state")
    if len(saved_optimizer["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Incompatible optimizer group count")
    for saved_group, group, current_group in zip(saved_optimizer["param_groups"], optimizer.param_groups,
                                                current_optimizer["param_groups"]):
        if set(saved_group) != set(current_group) or saved_group["params"] != current_group["params"]:
            raise ValueError("Incompatible optimizer parameter groups")
        for key, parameter in zip(saved_group["params"], group["params"]):
            moments = saved_optimizer["state"].get(key, {})
            expected_moments = {"step", "exp_avg", "exp_avg_sq"}
            if saved_group.get("amsgrad", False):
                expected_moments.add("max_exp_avg_sq")
            if moments and set(moments) != expected_moments:
                raise ValueError("Incomplete optimizer moments")
            for name, value in moments.items():
                if torch.is_tensor(value) and name != "step" and value.shape != parameter.shape:
                    raise ValueError(f"Incompatible optimizer tensor: {name}")
    if set(checkpoint["scheduler_state_dict"]) != set(scheduler.state_dict()):
        raise ValueError("Incomplete scheduler state")
    if set(checkpoint["scaler_state_dict"]) != set(scaler.state_dict()):
        raise ValueError("Incompatible scaler state/precision")
    try:
        copy.deepcopy(criterion).load_state_dict(copy.deepcopy(checkpoint["criterion_state_dict"]), strict=True)
        copy.deepcopy(optimizer).load_state_dict(copy.deepcopy(saved_optimizer))
        copy.deepcopy(scheduler).load_state_dict(copy.deepcopy(checkpoint["scheduler_state_dict"]))
        copy.deepcopy(scaler).load_state_dict(copy.deepcopy(checkpoint["scaler_state_dict"]))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid checkpoint training state: {exc}") from exc
    original_model.load_state_dict(model_state, strict=True)
    criterion.load_state_dict(checkpoint["criterion_state_dict"], strict=True)
    optimizer.load_state_dict(saved_optimizer)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint["rng_state"], loader_generator)
    return {"mode": "resume", "epoch": checkpoint["epoch"], "best_val": best_val,
            "backend_changes": backend_changes}


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
    quality_metrics = QualityMetrics()
    samples = 0
    synchronize_device(device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_tensor_batch(batch, device)
        batch_size = model_input_batch_size(batch["voxel"])
        with autocast_context(device, precision):
            outputs = model(batch["voxel"])
            losses = criterion(outputs, batch)
        quality_metrics.update(losses, batch)
        for name, value in losses.items():
            if quality_metrics.is_quality_metric(name):
                continue
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
    metrics.update(quality_metrics.summarize())
    metrics.update(seconds=time.perf_counter() - started, samples=samples)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--detector-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--warm-start", type=Path,
                                help="load compatible backbone tensors only, with fresh heads and training state")
    parser.add_argument("--backend-parity-verified", action="store_true",
                        help="allow resume backend changes after numerical parity has been verified")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--physical-batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int, help="override train.num_workers (default: config or 2)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument(
        "--target-backend",
        choices=("python", "numba"),
        default=None,
        help="override train.target_backend (default: config or python)",
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override train.compile_model; compile after checkpoint loading",
    )
    parser.add_argument("--amp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="maximum norm for gradient clipping (default: 10.0 or train.grad_clip_norm in config; <=0 disables)",
    )
    parser.add_argument("--val-physical-batch-size", type=int, default=None, help="override validation physical batch size")
    parser.add_argument(
        "--override-json",
        type=str,
        default=None,
        help="JSON string of config keys to override dynamically (e.g. '{\"loss\": {\"name\": \"gw_qal\"}}')",
    )
    return parser


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

    all_params = decay_params + no_decay_params + crit_params
    use_fused = (
        torch.cuda.is_available()
        and len(all_params) > 0
        and all_params[0].is_cuda
    )
    opt_kwargs = {"fused": True} if use_fused else {}

    if opt_type == "adam":
        return torch.optim.Adam(groups, lr=lr, **opt_kwargs)
    elif opt_type == "adamw":
        return torch.optim.AdamW(groups, lr=lr, **opt_kwargs)
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
    from core.datasets.dataset import Dataset, collate_detector_batch

    config = read_json(args.config)
    if args.override_json:
        try:
            overrides = json.loads(args.override_json)
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON string passed to --override-json: {err}")

        def _deep_update(base, update):
            for k, v in update.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    _deep_update(base[k], v)
                else:
                    base[k] = v

        _deep_update(config, overrides)
    # Direct notebook/CLI calls inherit the run snapshot unless explicitly overridden.
    training_config = config["train"]
    if args.num_workers is None:
        args.num_workers = training_config.get("num_workers", 2)
    if args.target_backend is None:
        args.target_backend = training_config.get("target_backend", "python")
    if args.compile_model is None:
        args.compile_model = training_config.get("compile_model", False)
    if args.target_backend not in {"python", "numba"}:
        raise ValueError("target_backend must be python or numba")
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
    grad_clip_norm = float(
        args.grad_clip_norm
        if args.grad_clip_norm is not None
        else config["train"].get("grad_clip_norm", 10.0)
    )
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
    epochs = int(args.epochs if args.epochs is not None else config["train"]["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    config["train"].update(epochs=epochs, physical_batch_size=physical_batch_size,
                          accumulation_steps=accumulation_steps)
    save_every = int(config["train"].get("save_every", 5))
    if save_every < 1:
        raise ValueError("save_every must be positive")
    spec = detection_spec(config)
    selection = selection_settings(config)
    if selection['primary'] == 'ap':
        if args.max_train_batches or args.max_val_batches:
            raise ValueError('AP selection requires full training/validation; use loss selection for a partial smoke run')
        raw = config.get('evaluation', {}).get('kitti_root')
        if not raw:
            raise ValueError('AP selection requires evaluation.kitti_root pointing to raw KITTI labels/calibration')
        raw = Path(raw).expanduser().resolve()
        if (raw / 'training').is_dir():
            raw = raw / 'training'
        for folder in ('label_2', 'calib') + (('image_2',) if selection['metric_mode'] != 'local_bev' else ()):
            if not (raw / folder).is_dir():
                raise FileNotFoundError(f'AP validation asset directory is missing: {raw / folder}')

    dataset_groups = spec.groups if spec.head_mode == "grouped" else None
    train_dataset = Dataset(
        config["train"]["data"],
        config["data"],
        config["augmentation"],
        config["model"]["cls_encoding"],
        "train",
        args.target_backend,
        task_groups=dataset_groups,
    )
    # A non-special task name disables augmentation and list-valued visualisation data.
    val_dataset = Dataset(
        config["val"]["data"],
        config["data"],
        config["augmentation"],
        config["model"]["cls_encoding"],
        "validation",
        args.target_backend,
        task_groups=dataset_groups,
    )
    generator = torch.Generator().manual_seed(seed)
    common_loader = loader_kwargs(args.num_workers, device.type == "cuda")
    if config["data"].get("bev_encoding", {}).get("name") == "pillar32":
        common_loader["collate_fn"] = collate_detector_batch
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
    criterion = build_training_criterion(config, device)
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
    ap_state, _ = advance_ap_state(None, None, 0, selection)
    if args.warm_start:
        source_path = args.warm_start.expanduser().resolve()
        source = torch.load(source_path, map_location=device)
        report = warm_start_backbone(model, source)
        config["initialization"] = {"mode": "backbone_warm_start", "checkpoint": str(source_path),
                                    "sha256": sha256(source_path)}
        write_json(run_dir / "warm_start.json", {**config["initialization"], **report})
        log_line(f"Backbone warm-start: loaded={len(report['loaded'])}, skipped={len(report['skipped'])}, "
                 f"missing={len(report['missing'])}; fresh training state at epoch 0; source={source_path}")
    if args.resume:
        resume = torch.load(args.resume, map_location=device)
        restored = restore_checkpoint(resume, config, model, criterion, optimizer, scheduler, scaler,
                                      loader_generator=generator,
                                      backend_parity_verified=args.backend_parity_verified)
        start_epoch, best_val = restored["epoch"], restored["best_val"]
        ap_state, _ = advance_ap_state(resume.get('ap_selection'), None, start_epoch, selection)
        log_line(f"Checkpoint mode={restored['mode']}; start epoch={start_epoch}: {args.resume}")
        if restored["backend_changes"]:
            log_line(f"Numerical backend parity asserted: {restored['backend_changes']}")
        if args.num_workers or resume.get("replay", {}).get("num_workers", 0):
            log_line("Persistent-worker augmentation RNG is not restored; bitwise worker replay is unsupported")

    write_json(run_dir / "config.resolved.json", config)

    if args.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile-model requires PyTorch 2.0 or newer")
        logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
        model = torch.compile(model)
        try:
            print("Warming up torch.compile kernels...")
            dummy_voxel = dummy_model_input(config, physical_batch_size, device)
            warmup_model(model, dummy_voxel, optimizer, device, precision)
            synchronize_device(device)
            print("Warmup torch.compile completed.")
        except Exception as exc:
            print(f"torch.compile warmup skipped: {exc}")

    if args.resume and restored["mode"] == "resume":
        # Compiler discovery/warmup may consume RNG. Start the next epoch from
        # the saved boundary after setup is finished.
        restore_rng_state(resume["rng_state"], generator)

    log_path = run_dir / "metrics.jsonl"
    log_line(f"Run directory: {run_dir}")
    tensorboard_dir = run_dir / "tensorboard"
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        purge_step = (start_epoch + 1) if start_epoch > 0 else None
        writer = SummaryWriter(log_dir=str(tensorboard_dir), purge_step=purge_step)
        log_line(f"TensorBoard directory: {tensorboard_dir}")
    except (ImportError, Exception):
        writer = None
    log_line(
        f"Backbone={config['model']['backbone']}; loss={loss_name}; "
        f"precision={precision}; grad_clip_norm={grad_clip_norm}"
    )
    log_line(
        f"Frames train={len(train_dataset)} val={len(val_dataset)}; "
        f"physical batch={physical_batch_size}; accumulation={accumulation_steps}; "
        f"effective batch={physical_batch_size * accumulation_steps}"
    )
    log_line(f"Checkpoint selection={selection['primary']}; validation loss every epoch; "
             f"AP every {selection['ap_every']} epoch(s) plus final, mode={selection['metric_mode']}, "
             "R40 Moderate macro of Car/Pedestrian/Cyclist" if selection['primary'] == 'ap'
             else 'Checkpoint selection=loss; AP validation disabled')

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        criterion.train()
        set_loss_epoch(criterion, epoch - 1)
        optimizer.zero_grad(set_to_none=True)
        training_sum = torch.zeros((), device=device)
        training_quality = QualityMetrics()
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
        log_line(f"Epoch {epoch:03d}/{epochs:03d}: training {batches_this_epoch} batches")
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_tensor_batch(batch, device)
            batch_size = model_input_batch_size(batch["voxel"])
            with autocast_context(device, precision):
                outputs = model(batch["voxel"])
                losses = criterion(outputs, batch)
                objective = losses["loss"]
                group_start = (
                    (batch_index - 1) // accumulation_steps
                ) * accumulation_steps
                group_size = min(accumulation_steps, batches_this_epoch - group_start)
                backward_objective = objective / group_size
            training_quality.update(losses, batch)
            scaler.scale(backward_objective).backward()
            should_update = (
                batch_index % accumulation_steps == 0
                or batch_index == batches_this_epoch
            )
            if should_update:
                if grad_clip_norm > 0.0:
                    if scaler_enabled:
                        scaler.unscale_(optimizer)
                    grad_params = [p for p in model.parameters() if p.requires_grad] + [
                        p for p in criterion.parameters() if p.requires_grad
                    ]
                    torch.nn.utils.clip_grad_norm_(grad_params, max_norm=grad_clip_norm)
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
        log_line(f"Epoch {epoch:03d}/{epochs:03d}: validating loss")
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
        # Show completed train/val results before the additional full-split AP pass.
        retained_flag = " [BEST LOSS]" if retained else ""
        epoch_summary = (
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: {train_objective:.4f} | "
            f"Val Loss: {current_val:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: train={training_seconds:.1f}s, val={validation['seconds']:.1f}s{retained_flag}"
        )
        log_line(epoch_summary)
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
            loader_generator=generator,
            ap_selection=ap_state,
        )
        ap_validation, ap_seconds, retained_ap = None, 0., False
        if ap_due(epoch, epochs, selection):
            log_line(f"AP validation: epoch {epoch}/{epochs}; full validation split; {selection['metric_mode']} R40 Moderate")
            ap_validation, ap_seconds = evaluate_training_ap(payload, run_dir / 'config.resolved.json',
                args.detector_root, device, generator, run_dir)
            ap_state, retained_ap = advance_ap_state(ap_state, ap_validation, epoch, selection)
            class_scores = " | ".join(
                f"{name}: {score:.2f}%"
                for name, score in ap_validation['per_class_moderate_percent'].items()
            )
            log_line(f"Validation AP: {ap_validation['score']:.4f}% (R40 Moderate) | "
                     f"{class_scores} | Time: {ap_seconds:.1f}s" +
                     (' [BEST AP]' if retained_ap else ''))
        payload['ap_selection'] = ap_state
        payload['ap_validation'] = ap_validation
        atomic_torch_save(payload, checkpoints_dir / 'last.pt')
        if epoch % save_every == 0 or epoch == epochs:
            atomic_torch_save(payload, checkpoints_dir / f"{epoch}epoch.pt")
        save_selections(run_dir, payload, retained_loss=retained, retained_ap=retained_ap)
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
            "retained_ap": retained_ap,
            "ap_validation": ap_validation,
            "ap_seconds": ap_seconds,
        }
        quality_summary = training_quality.summarize()
        if quality_summary:
            row["training_quality"] = quality_summary
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if writer is not None:
            writer.add_scalar("train/loss", train_objective, epoch)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("train/training_seconds", training_seconds, epoch)
            writer.add_scalar("train/optimizer_updates", update_count, epoch)
            if quality_summary:
                for q_name, q_val in quality_summary.items():
                    if isinstance(q_val, (int, float)) and math.isfinite(q_val):
                        writer.add_scalar(f"train_quality/{q_name}", float(q_val), epoch)
            writer.add_scalar("val/loss", current_val, epoch)
            for v_name, v_val in validation.items():
                if v_name == "loss":
                    continue
                if isinstance(v_val, (int, float)) and math.isfinite(v_val):
                    writer.add_scalar(f"val/{v_name}", float(v_val), epoch)
            if ap_validation is not None:
                writer.add_scalar("val_ap/score_r40_moderate", float(ap_validation["score"]), epoch)
                if "per_class_moderate_percent" in ap_validation:
                    for cls_name, cls_score in ap_validation["per_class_moderate_percent"].items():
                        writer.add_scalar(f"val_ap/{cls_name}_moderate_percent", float(cls_score), epoch)
                if ap_seconds:
                    writer.add_scalar("val_ap/seconds", float(ap_seconds), epoch)
            writer.add_scalar("checkpoint/best_val_loss", best_val, epoch)
            if selection["primary"] == "ap" and ap_state.get("best_score") is not None:
                writer.add_scalar("checkpoint/best_val_ap", float(ap_state["best_score"]), epoch)
            if torch.cuda.is_available():
                writer.add_scalar("system/gpu_max_memory_gb", torch.cuda.max_memory_allocated() / (1024 ** 3), epoch)
            writer.flush()

    if writer is not None:
        writer.close()
    log_line(f"Selected checkpoint: {loss_selection_dir / ('best_ap.pt' if selection['primary'] == 'ap' else 'best_loss.pt')}")
    log_line(f"Best loss checkpoint: {loss_selection_dir / 'best_loss.pt'}")
    log_line(f"Selection record: {loss_selection_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
