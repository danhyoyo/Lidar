"""OpenPCDet-style augmentation queue for this detector's bottom-centered boxes.

Points: [x, y, z, ...]. Boxes: [class_id, h, w, l, x, y, z_bottom, yaw].
Configuration follows AUG_CONFIG_LIST / DISABLE_AUG_LIST. Rotation angles are
in radians; 'flip along x' negates y, as in OpenPCDet. Implemented with NumPy
so DataLoader workers need no OpenPCDet installation or CUDA extensions.

Design reference: https://github.com/open-mmlab/OpenPCDet/tree/master/pcdet/datasets/augmentor
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np


def _range(value, name, *, positive=False):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all() or array[0] > array[1]:
        raise ValueError(f"{name} must be a finite ordered [min, max] pair")
    if positive and array[0] <= 0:
        raise ValueError(f"{name} must contain positive scales")
    return array


class DataAugmentor:
    def __init__(self, root_path, augmentor_configs, class_names, *, rng=None,
                 geometry=None, allowed_frame_ids=None):
        self.root_path = Path(root_path)
        self.class_names = dict(class_names)
        # None uses the NumPy state seeded by PyTorch in each DataLoader worker.
        self.rng = rng
        self.geometry = geometry
        configs = copy.deepcopy(augmentor_configs)
        if isinstance(configs, dict):
            if "AUG_CONFIG_LIST" not in configs:
                raise ValueError("OpenPCDet mode requires AUG_CONFIG_LIST")
            disabled = set(configs.get("DISABLE_AUG_LIST", []))
            configs = configs["AUG_CONFIG_LIST"]
        else:
            disabled = set()
        if not isinstance(configs, list):
            raise ValueError("AUG_CONFIG_LIST must be a list")
        self.queue = []
        for config in configs:
            if not isinstance(config, dict):
                raise ValueError("Each AUG_CONFIG_LIST entry must be a config object")
            name = config.get("NAME")
            if name in disabled:
                continue
            required_keys = {
                "random_world_rotation": "WORLD_ROT_ANGLE",
                "random_world_scaling": "WORLD_SCALE_RANGE",
                "random_world_translation": "NOISE_TRANSLATE_STD",
            }
            required = required_keys.get(name)
            if required is not None and required not in config:
                raise ValueError(f"{name} requires {required}")
            default_probability = .5 if name == "random_world_flip" else 1.
            probability = float(config.get("PROBABILITY", default_probability))
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("PROBABILITY must be between 0 and 1")
            if name == "random_world_flip":
                axes = config.get("ALONG_AXIS_LIST", ["x"])
                if not axes or any(axis not in ("x", "y") for axis in axes):
                    raise ValueError("ALONG_AXIS_LIST supports x and y")
                value = tuple(axes)
            elif name == "random_world_rotation":
                value = config["WORLD_ROT_ANGLE"]
                if np.isscalar(value):
                    value = [-float(value), float(value)]
                value = _range(value, "WORLD_ROT_ANGLE")
            elif name == "random_world_scaling":
                value = _range(config["WORLD_SCALE_RANGE"], "WORLD_SCALE_RANGE", positive=True)
            elif name == "random_world_translation":
                value = np.asarray(config["NOISE_TRANSLATE_STD"], dtype=np.float64)
                if value.shape != (3,) or not np.isfinite(value).all() or (value < 0).any():
                    raise ValueError("NOISE_TRANSLATE_STD must contain three nonnegative stds")
            elif name == "gt_sampling":
                from .database_sampler import DataBaseSampler
                value = DataBaseSampler(
                    self.root_path, config, self.class_names, geometry=geometry,
                    allowed_frame_ids=allowed_frame_ids,
                )
            elif name == "omni_gt_sampling":
                from .omni_sampler import OmniDataBaseSampler
                value = OmniDataBaseSampler(
                    self.root_path, config, self.class_names, geometry=geometry,
                    allowed_frame_ids=allowed_frame_ids,
                )
            else:
                raise ValueError(f"Unsupported augmentation NAME: {name!r}")
            self.queue.append((name, probability, value))

    def set_epoch(self, epoch: int) -> None:
        """Update epoch for augmentors that support scheduling (e.g. curriculum)."""
        for _, _, value in self.queue:
            if hasattr(value, "set_epoch"):
                value.set_epoch(epoch)

    def __call__(self, points, boxes, road_plane=None):
        points = np.asarray(points, dtype=np.float32).copy()
        boxes = np.asarray(boxes, dtype=np.float32).copy()
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points must have shape [N, 3 + features]")
        if boxes.ndim != 2 or boxes.shape[1] != 8:
            raise ValueError("boxes must have shape [N, 8]")
        if not np.isfinite(points).all() or not np.isfinite(boxes).all():
            raise ValueError("Augmentation requires finite points and boxes")
        if (boxes[:, 1:4] <= 0).any():
            raise ValueError("Box dimensions must be positive")
        rng = self.rng if self.rng is not None else np.random
        for name, probability, value in self.queue:
            if name == "random_world_flip":
                for axis in value:
                    if probability == 0 or (probability < 1 and rng.random() >= probability):
                        continue
                    dimension = 1 if axis == "x" else 0
                    points[:, dimension] *= -1
                    boxes[:, 4 + dimension] *= -1
                    boxes[:, 7] = -boxes[:, 7] if axis == "x" else -boxes[:, 7] - np.pi
                continue
            if probability == 0 or (probability < 1 and rng.random() >= probability):
                continue
            if name == "random_world_rotation":
                angle = rng.uniform(*value)
                cosine, sine = np.cos(angle), np.sin(angle)
                rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
                points[:, :2] = points[:, :2] @ rotation.T
                boxes[:, 4:6] = boxes[:, 4:6] @ rotation.T
                boxes[:, 7] += angle
            elif name == "random_world_scaling":
                scale = rng.uniform(*value)
                points[:, :3] *= scale
                boxes[:, 1:7] *= scale
            elif name == "random_world_translation":
                translation = rng.normal(0, value, size=3).astype(np.float32)
                points[:, :3] += translation
                boxes[:, 4:7] += translation
            elif name == "gt_sampling":
                points, boxes = value(points, boxes, rng)
            elif name == "omni_gt_sampling":
                points, boxes = value(points, boxes, road_plane=road_plane, rng=rng)
        boxes[:, 7] = (boxes[:, 7] + np.pi) % (2 * np.pi) - np.pi
        return np.ascontiguousarray(points), np.ascontiguousarray(boxes)
