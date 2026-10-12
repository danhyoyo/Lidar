"""Independent criteria with fixed, normalized group objective weights.

Core components use objective weights; quality means use supervised counts.
"""

import copy
from collections.abc import Mapping
from numbers import Integral

import torch
import torch.nn as nn

from ..detection_config import resolve_detection_config
from ..task_groups import validate_resolved_groups
from .loss_fn import LossFunction
from .iou_targets import compute_iou_targets


class GroupedLossFunction(nn.Module):
    COMPONENTS = ("cls", "offset", "size", "yaw", "geo", "iou", "vertical")

    def __init__(self, cls_encoding, config=None, *, task_groups, box_mode="bev"):
        super().__init__()
        groups = validate_resolved_groups(task_groups)
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Grouped loss config must be a mapping")
        self.config = copy.deepcopy(dict(config or {}))
        # Reuse pure capabilities/weight validation at the direct loss boundary.
        contract = resolve_detection_config({
            "model": {"head_mode": "grouped", "cls_encoding": cls_encoding,
                      "header_use_iou": self.config.get("use_iou", False)},
            "data": {"num_classes": sum(g.num_classes for g in groups),
                     "box_mode": box_mode,
                     "head_groups": [{"name": g.name, "classes": list(g.classes)} for g in groups],
                     "kitti": {"objects": {cls: i for g in groups for cls, i in zip(g.classes, g.global_ids)}}},
            "loss": self.config,
        })
        self.task_groups = contract.groups
        self.group_weights = contract.group_weights
        self.cls_encoding = contract.cls_encoding
        self.use_iou = contract.use_iou
        self.box_mode = contract.box_mode
        strategy_config = {k: v for k, v in self.config.items() if k != "group_weights"}
        self.criteria = nn.ModuleDict({
            g.name: LossFunction(self.cls_encoding, copy.deepcopy(strategy_config), box_mode=self.box_mode)
            for g in self.task_groups
        })

    def set_epoch(self, epoch):
        """Broadcast one zero-based epoch without adding wrapper state."""
        if not isinstance(epoch, Integral) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("set_epoch requires a non-negative integer epoch")
        for criterion in self.criteria.values():
            setter = getattr(criterion, "set_epoch", None)
            if setter is not None:
                setter(epoch)

    def _validate_groups(self, value, label):
        if not isinstance(value, Mapping) or not isinstance(value.get("groups"), Mapping):
            raise ValueError(f"Grouped {label} must contain a groups mapping")
        groups = value["groups"]
        if set(groups) != {g.name for g in self.task_groups}:
            raise ValueError(f"Grouped {label} names must match resolved task_groups exactly")
        return groups

    def _validate_inputs(self, predictions, targets):
        common_shape = None
        curriculum_epoch = None
        for group in self.task_groups:
            pred, target = predictions[group.name], targets[group.name]
            if not isinstance(pred, Mapping) or not isinstance(target, Mapping):
                raise ValueError(f"Group {group.name} prediction/target must be mappings")
            for value, keys in ((pred, ("cls", "offset", "size", "yaw")),
                                (target, ("cls", "offset", "size", "yaw", "reg_mask"))):
                for key in keys:
                    if key not in value:
                        raise KeyError(f"Group {group.name} is missing {key!r}")
                    if not torch.is_tensor(value[key]):
                        raise ValueError(f"Group {group.name}/{key} must be a tensor")
            channels = group.num_classes + int(self.cls_encoding == "binary")
            if pred["cls"].ndim != 4 or pred["cls"].shape[1] != channels:
                raise ValueError(f"Group {group.name} cls width must be {channels}")
            shape = (pred["cls"].shape[0], *pred["cls"].shape[2:])
            if common_shape is not None and shape != common_shape:
                raise ValueError("All groups must share batch size and spatial grid")
            common_shape = shape
            expected = (shape[0], 2, *shape[1:])
            for key in ("offset", "size", "yaw"):
                if tuple(pred[key].shape) != expected:
                    raise ValueError(f"Group {group.name}/{key} must have BEV shape {expected}")
            if ("iou" in pred) != self.use_iou:
                raise ValueError(f"Group {group.name} IQA prediction and supervision must agree")
            if self.use_iou and (not torch.is_tensor(pred["iou"]) or
                                 tuple(pred["iou"].shape) != (shape[0], 1, *shape[1:])):
                raise ValueError(f"Group {group.name} IQA must have shape [B,1,H,W]")
            # Check every route before any strategy can update adaptive state.
            self.criteria[group.name].validate_vertical_inputs(pred, target)
            self.criteria[group.name].strategy._validate_inputs(pred, target)
            strategy = self.criteria[group.name].strategy
            if getattr(strategy, "quality_target", None) == "rotated_iou":
                peaks = target["cls"].ge(1.).any(dim=1)
                if (peaks & ~target["reg_mask"].bool()).any():
                    raise ValueError("classification peaks must have an assigned regression target")
                if getattr(strategy, "quality_warmup_epochs", 0):
                    epoch = int(strategy.quality_epoch.item())
                    if epoch < 0:
                        raise ValueError("call set_epoch before using Q-OGA quality curriculum")
                    if curriculum_epoch is not None and epoch != curriculum_epoch:
                        raise ValueError("Q-OGA groups must use the same epoch")
                    curriculum_epoch = epoch

    def forward(self, prediction, target):
        predictions = self._validate_groups(prediction, "prediction")
        targets = self._validate_groups(target, "target")
        self._validate_inputs(predictions, targets)
        losses = {group.name: self.criteria[group.name](predictions[group.name], targets[group.name])
                  for group in self.task_groups}
        if self.use_iou:
            for name, values in losses.items():
                mask = targets[name]["reg_mask"].bool()
                values["iou_target_count"] = mask.sum().float().detach()
                if "mean_iou_target" not in values:
                    # Baseline does not expose its target mean. Preserve the
                    # legacy facade and derive this diagnostic without a graph.
                    strategy = self.criteria[name].strategy
                    quality = compute_iou_targets(predictions[name], targets[name],
                                                  method=strategy.iou_target_type,
                                                  epsilon=strategy.eps,
                                                  max_abs_log_size=strategy.max_abs_log_size)
                    values["mean_iou_target"] = quality[mask].mean() if mask.any() else quality.new_zeros(())
        total = sum(weight * losses[name]["loss"] for name, weight in self.group_weights)
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite grouped loss")
        result = {"loss": total}
        for name, weight in self.group_weights:
            for key, value in losses[name].items():
                scalar = value if torch.is_tensor(value) else total.new_tensor(value)
                if scalar.ndim != 0:
                    raise ValueError(f"Group {name} diagnostic {key} must be scalar")
                result[f"group/{name}/{key}"] = scalar.detach()
            result[f"group/{name}/group_weight"] = total.detach().new_tensor(weight)
        for key in self.COMPONENTS:
            if all(key in losses[g.name] for g in self.task_groups):
                result[key] = sum(weight * result[f"group/{name}/{key}"]
                                  for name, weight in self.group_weights)
        for count_key, mean_keys in (
            ("quality_peak_count", ("quality_iou_mean", "quality_iou_zero_fraction", "quality_target_mean")),
            ("iou_target_count", ("mean_iou_target",)),
        ):
            if all(count_key in values for values in losses.values()):
                count = sum(result[f"group/{name}/{count_key}"] for name in losses)
                result[count_key] = count
                for key in mean_keys:
                    result[key] = sum(result[f"group/{name}/{key}"] * result[f"group/{name}/{count_key}"]
                                      for name in losses) / count.clamp_min(1.)
        if "quality_peak_count" in result:
            mixes = [result[f"group/{name}/quality_iou_mix"] for name in losses]
            if any(not torch.equal(mix, mixes[0]) for mix in mixes[1:]):
                raise ValueError("Q-OGA curriculum mix must agree across groups")
            result["quality_iou_mix"] = mixes[0]
        return result
