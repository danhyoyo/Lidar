"""CPU GT database sampling with class quotas and rotated BEV collisions.

Inspired by OpenPCDet's database_sampler.py. This implementation consumes the
JSON database produced by build_gt_database.py. Metadata boxes use geometric
center Z; dataset boxes use bottom Z. Object points retain their original yaw
and are stored relative to their geometric center (translation only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import bev_polygon, points_in_box


def _class_counts(values, class_names):
    counts = {}
    for value in values:
        try:
            name, raw = value.split(":")
            count = int(raw)
        except (ValueError, AttributeError) as error:
            raise ValueError("Class counts must use 'Class:integer'") from error
        if name not in class_names or count < 0 or name in counts:
            raise ValueError(f"Invalid class count: {value!r}")
        counts[name] = count
    return counts


class DataBaseSampler:
    def __init__(self, root_path, config, class_names, *, geometry=None,
                 allowed_frame_ids=None):
        self.root_path = Path(root_path)
        self.class_names = dict(class_names)
        self.geometry = geometry
        self.feature_count = int(config.get("NUM_POINT_FEATURES", 4))
        if self.feature_count < 3:
            raise ValueError("NUM_POINT_FEATURES must be at least 3")
        for name in ("USE_ROAD_PLANE", "DATABASE_WITH_FAKELIDAR", "USE_SHARED_MEMORY"):
            if config.get(name, False):
                raise ValueError(f"{name} is not supported by this LiDAR-only sampler")
        self.extra_width = np.asarray(config.get("REMOVE_EXTRA_WIDTH", [0, 0, 0]), dtype=np.float64)
        if (self.extra_width.shape != (3,) or not np.isfinite(self.extra_width).all()
                or (self.extra_width < 0).any()):
            raise ValueError("REMOVE_EXTRA_WIDTH must contain three nonnegative margins")
        self.limit_whole_scene = bool(config.get("LIMIT_WHOLE_SCENE", False))
        self.quotas = _class_counts(config.get("SAMPLE_GROUPS", []), class_names)
        paths = config.get("DB_INFO_PATH", [])
        if not isinstance(paths, list) or not paths:
            raise ValueError("DB_INFO_PATH must be a nonempty list of JSON paths")
        self.db_infos = {name: [] for name in class_names}
        allowed = set(allowed_frame_ids) if allowed_frame_ids is not None else None
        for path in paths:
            with (self.root_path / path).open(encoding="utf-8") as stream:
                metadata = json.load(stream)
            if metadata.get("format") != "lidar_gt_database_v1":
                raise ValueError("Unsupported GT database format; run build_gt_database.py")
            if metadata.get("num_point_features") != self.feature_count:
                raise ValueError("GT database NUM_POINT_FEATURES mismatch")
            source_ids = set(metadata["source_frame_ids"])
            if allowed is not None and not source_ids <= allowed:
                raise ValueError("GT database contains frames outside the current training split")
            for name in class_names:
                for entry in metadata["db_infos"].get(name, []):
                    if entry["image_idx"] not in source_ids:
                        raise ValueError("GT database entry is outside its declared training split")
                    box = np.asarray(entry["box3d_lidar"], dtype=np.float64)
                    if (box.shape != (7,) or not np.isfinite(box).all()
                            or (box[3:6] <= 0).any() or entry["num_points_in_gt"] < 1):
                        raise ValueError("Malformed GT database box or point count")
                    if entry.get("name", name) != name:
                        raise ValueError("GT database class mismatch")
                    self.db_infos[name].append(entry)
        prepare = config.get("PREPARE", {})
        unknown = set(prepare) - {"filter_by_min_points", "filter_by_difficulty"}
        if unknown:
            raise ValueError(f"Unsupported database PREPARE filters: {sorted(unknown)}")
        minima = _class_counts(prepare.get("filter_by_min_points", []), class_names)
        excluded_difficulties = set(prepare.get("filter_by_difficulty", []))
        for name, entries in self.db_infos.items():
            self.db_infos[name] = [entry for entry in entries
                                   if entry["num_points_in_gt"] >= minima.get(name, 0)
                                   and entry.get("difficulty") not in excluded_difficulties]
        # Shuffle lazily, after PyTorch seeds each worker, rather than in parent.
        self.orders = {name: None for name in self.db_infos}
        self.pointers = {name: 0 for name in self.db_infos}

    def _sample_entries(self, name, count, rng):
        entries = self.db_infos[name]
        count = min(count, len(entries))
        if count <= 0:
            return []
        # At most one pass over the pool per scene; do not repeat an object in
        # one scene when a class has a tiny database.
        if self.orders[name] is None or self.pointers[name] + count > len(entries):
            self.orders[name] = rng.permutation(len(entries))
            self.pointers[name] = 0
        start = self.pointers[name]
        self.pointers[name] += count
        return [entries[index] for index in self.orders[name][start:start + count]]

    def __call__(self, points, boxes, rng):
        if points.shape[1] != self.feature_count:
            raise ValueError("Scene point feature count does not match GT database")
        occupied = [bev_polygon(box) for box in boxes]
        sampled_boxes, sampled_points = [], []
        for name, quota in self.quotas.items():
            if self.limit_whole_scene:
                quota = max(0, quota - int((boxes[:, 0] == self.class_names[name]).sum()))
            for entry in self._sample_entries(name, quota, rng):
                x, y, z_center, length, width, height, yaw = entry["box3d_lidar"]
                box = np.array([self.class_names[name], height, width, length,
                                x, y, z_center - height/2, yaw], dtype=np.float32)
                if self.geometry is not None:
                    geom = self.geometry
                    if not (geom["x_min"] < x < geom["x_max"] and
                            geom["y_min"] < y < geom["y_max"]):
                        continue
                polygon = bev_polygon(box)
                if any(polygon.intersection(other).area > 0 for other in occupied):
                    continue
                local = np.fromfile(self.root_path / entry["path"], dtype=np.float32)
                expected = entry["num_points_in_gt"] * self.feature_count
                if local.size != expected:
                    raise ValueError(f"GT object point count mismatch: {entry['path']}")
                local = local.reshape(-1, self.feature_count)
                if not np.isfinite(local).all():
                    raise ValueError(f"Non-finite GT object points: {entry['path']}")
                local[:, :3] += np.array([x, y, z_center], dtype=np.float32)
                occupied.append(polygon)
                sampled_boxes.append(box)
                sampled_points.append(local)
        if not sampled_boxes:
            return points, boxes
        keep = np.ones(len(points), dtype=bool)
        for box in sampled_boxes:
            keep &= ~points_in_box(points, box, self.extra_width)
        return (np.concatenate([points[keep], *sampled_points]),
                np.concatenate([boxes, np.stack(sampled_boxes)]))
