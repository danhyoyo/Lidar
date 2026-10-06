"""Transactional GT sampling: cached object IO, JIT geometry, opt-in PCU heuristics.

References:
- https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/transforms/dbsampler.py
- This repository's feature/pillar5-pcu-augmentation (9b99d92).
Uses the existing train-provenance-checked JSON database; no framework dependency.
"""
from __future__ import annotations

from collections import OrderedDict
from numbers import Integral, Real
import os
from pathlib import Path

import numpy as np

from .database_sampler import DataBaseSampler
from .hybrid_geometry import collision_matrix, points_in_boxes_union, resolve_backend
from .hybrid_physics import (ground_height, line_of_sight_blockers, shadow_mask,
                             static_obstacle_count, transform_object)


class ObjectPointCache:
    """Bounded LRU, immutable arrays, file-change detection and empty worker copies."""
    def __init__(self, max_bytes):
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 0:
            raise ValueError("cache max_bytes must be nonnegative")
        self._items = OrderedDict()
        self._bytes = self.hits = self.misses = self.disk_reads = self.evictions = 0
        self._owner_pid = os.getpid()

    def load(self, path, feature_count, expected_points):
        if self._owner_pid != os.getpid():
            self.__init__(self.max_bytes)
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, feature_count, expected_points)
        if key in self._items:
            self.hits += 1
            self._items.move_to_end(key)
            return self._items[key]
        self.misses += 1
        # Discard old versions of this path so changed files cannot fill the cache.
        for old in list(self._items):
            if old[0] == key[0]:
                self._bytes -= self._items.pop(old).nbytes
        values = np.fromfile(path, dtype=np.float32)
        self.disk_reads += 1
        if values.size != expected_points * feature_count or not np.isfinite(values).all():
            raise ValueError(f"Malformed GT object points: {path}")
        values = values.reshape(-1, feature_count)
        values.setflags(write=False)
        if values.nbytes <= self.max_bytes:
            while self._items and self._bytes + values.nbytes > self.max_bytes:
                self._bytes -= self._items.popitem(last=False)[1].nbytes
                self.evictions += 1
            self._items[key] = values
            self._bytes += values.nbytes
        return values

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "disk_reads": self.disk_reads,
                "evictions": self.evictions, "bytes": self._bytes, "entries": len(self._items)}

    def __getstate__(self):
        return {"max_bytes": self.max_bytes}

    def __setstate__(self, state):
        self.__init__(state["max_bytes"])


def _number(config, key, default, *, lower=0, upper=None, integer=False):
    value = config.get(key, default)
    kind = Integral if integer else Real
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, kind)
            or not np.isfinite(value) or value < lower or (upper is not None and value > upper)):
        raise ValueError(f"Invalid {key}: {value!r}")
    return int(value) if integer else float(value)


class HybridDataBaseSampler(DataBaseSampler):
    def __init__(self, root_path, config, class_names, *, geometry=None, allowed_frame_ids=None):
        # Reuse format validation, class filters, worker-seeded pools and provenance.
        super().__init__(root_path, config, class_names, geometry=geometry,
                         allowed_frame_ids=allowed_frame_ids)
        self.backend = resolve_backend(config.get("GEOMETRY_BACKEND", "auto"))
        self.cache = ObjectPointCache(int(_number(config, "CACHE_SIZE_MB", 64) * 1024**2))
        self.placement = config.get("PLACEMENT_MODE", "source_relative")
        if self.placement not in ("source", "source_relative", "random"):
            raise ValueError("PLACEMENT_MODE must be source, source_relative or random")
        self.range_scale = np.asarray(config.get("RANGE_SCALE", [.9, 1.1]), np.float64)
        if (self.range_scale.shape != (2,) or not np.isfinite(self.range_scale).all()
                or self.range_scale[0] <= 0 or self.range_scale[1] < self.range_scale[0]):
            raise ValueError("RANGE_SCALE must contain ordered positive scales")
        self.azimuth_jitter = _number(config, "AZIMUTH_JITTER_DEG", 5, upper=180)
        self.attempts = _number(config, "MAX_PLACEMENT_ATTEMPTS", 6, lower=1, upper=100, integer=True)
        self.multiplier = _number(config, "CANDIDATE_MULTIPLIER", 3, lower=1, upper=20, integer=True)
        self.rate = _number(config, "SAMPLE_RATE", 1, upper=1)
        self.margin = _number(config, "COLLISION_MARGIN", 0.1)
        self.min_sample_points = _number(config, "MIN_SAMPLE_POINTS", 5, lower=1, integer=True)
        self.min_visible_points = _number(config, "MIN_VISIBLE_POINTS", 5, integer=True)
        self.min_visible_ratio = _number(config, "MIN_VISIBLE_RATIO", .5, upper=1)
        self.max_static_points = _number(config, "MAX_STATIC_OBSTACLE_POINTS", 2, integer=True)
        self.max_blockers = _number(config, "MAX_LOS_BLOCKING_POINTS", 5, lower=1, integer=True)
        self.gamma = _number(config, "INTENSITY_GAMMA", 1.7)
        self.flags = {}
        for name in ("GROUND_VALIDATION", "STATIC_COLLISION", "LINE_OF_SIGHT", "SHADOW_MASKING",
                     "DENSITY_SUBSAMPLE", "RADIOMETRIC_CALIBRATION", "VISIBILITY_PROTECTION"):
            key = f"ENABLE_{name}"
            value = config.get(key, name == "VISIBILITY_PROTECTION")
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} must be bool")
            self.flags[name] = bool(value)
        self.geometry = geometry or {"x_min": 0., "x_max": 70.4, "y_min": -40., "y_max": 40.}
        self._owner_pid = os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["orders"] = {name: None for name in self.orders}
        state["pointers"] = {name: 0 for name in self.pointers}
        # ObjectPointCache clears its contents on serialization.
        return state

    def _pose(self, entry, rng):
        x, y, zc, length, width, height, yaw = entry["box3d_lidar"]
        if self.placement == "source_relative":
            scale = rng.uniform(*self.range_scale)
            angle = np.deg2rad(rng.uniform(-self.azimuth_jitter, self.azimuth_jitter))
            c, s = np.cos(angle), np.sin(angle)
            x, y, yaw = scale * (x*c - y*s), scale * (x*s + y*c), yaw + angle
        elif self.placement == "random":
            g = self.geometry
            x, y = rng.uniform(g["x_min"], g["x_max"]), rng.uniform(g["y_min"], g["y_max"])
            yaw = rng.uniform(-np.pi, np.pi)
        return np.array([self.class_names[entry["name"]], height, width, length,
                         x, y, zc-height/2, (yaw+np.pi) % (2*np.pi)-np.pi], np.float32)

    def _visibility_indices(self, points, boxes):
        return [np.flatnonzero(points_in_boxes_union(points, b[None], backend=self.backend)) for b in boxes]

    def __call__(self, points, boxes, rng, *, return_metadata=False):
        if self._owner_pid != os.getpid():
            self.orders = {name: None for name in self.orders}
            self.pointers = {name: 0 for name in self.pointers}
            self.cache.__init__(self.cache.max_bytes)
            self._owner_pid = os.getpid()
        if (points.ndim != 2 or points.shape[1] != self.feature_count or boxes.ndim != 2
                or boxes.shape[1] != 8 or not np.isfinite(points).all() or not np.isfinite(boxes).all()
                or (boxes[:, 1:4] <= 0).any()):
            raise ValueError("Hybrid sampling requires finite points and (M,8) positive-dimension boxes")
        # Early quota exit avoids geometry/membership work on scenes already at quota.
        requested = {}
        for name, quota in self.quotas.items():
            existing = int((boxes[:, 0] == self.class_names[name]).sum()) if self.limit_whole_scene else 0
            requested[name] = max(0, int(np.round(self.rate * max(0, quota-existing))))
        meta = {"attempted_by_class": {name: 0 for name in self.quotas},
                "accepted_by_class": {name: 0 for name in self.quotas},
                "rejected_by_reason": {}, "removed_interior_points": 0,
                "removed_shadow_points": 0, "inserted_sources": [], "num_inserted": 0,
                "geometry_backend": self.backend}
        if not any(requested.values()):
            if return_metadata:
                meta.update(point_is_sampled=np.zeros(len(points), bool),
                            box_is_sampled=np.zeros(len(boxes), bool), cache=self.cache.stats())
                return points, boxes, meta
            return points, boxes

        chunks = [(points, np.ones(len(points), bool))]
        accepted = []
        original_indices = None
        # Positive collision clearance prevents deleting another box's points.
        # At zero margin, touching boxes can share points on their boundaries.
        protect = self.flags["VISIBILITY_PROTECTION"] and (
            self.flags["SHADOW_MASKING"] or np.any(self.extra_width > 0) or self.margin <= 1e-8)

        def reject(reason):
            meta["rejected_by_reason"][reason] = meta["rejected_by_reason"].get(reason, 0) + 1

        for name, quota in requested.items():
            if quota == 0:
                continue
            entries = self._sample_entries(name, quota * self.multiplier, rng)
            if not entries:
                reject("empty_pool")
                continue
            first_poses = np.stack([self._pose(entry, rng) for entry in entries])
            occupied = np.concatenate([boxes, np.asarray(accepted, np.float32).reshape(-1, 8)])
            blocked = collision_matrix(first_poses, occupied, self.margin, backend=self.backend).any(axis=1)
            for index, entry in enumerate(entries):
                if meta["accepted_by_class"][name] >= quota:
                    break
                attempts = 1 if self.placement == "source" else self.attempts
                for attempt in range(attempts):
                    meta["attempted_by_class"][name] += 1
                    candidate = first_poses[index].copy() if attempt == 0 else self._pose(entry, rng)
                    g = self.geometry
                    if not (g["x_min"] < candidate[4] < g["x_max"] and g["y_min"] < candidate[5] < g["y_max"]):
                        reject("bounds")
                        continue
                    original_collision = blocked[index] if attempt == 0 else collision_matrix(
                        candidate[None], occupied, self.margin, backend=self.backend).any()
                    accepted_collision = bool(accepted) and collision_matrix(
                        candidate[None], np.stack(accepted), self.margin, backend=self.backend).any()
                    if original_collision or accepted_collision:
                        reject("box_collision")
                        continue
                    if self.flags["GROUND_VALIDATION"]:
                        supported, ground = ground_height(chunks, candidate)
                        if not supported:
                            reject("ground")
                            continue
                        candidate[6] = ground
                    if self.flags["STATIC_COLLISION"] and static_obstacle_count(
                            chunks, candidate, backend=self.backend) > self.max_static_points:
                        reject("static_collision")
                        continue
                    if self.flags["LINE_OF_SIGHT"] and line_of_sight_blockers(
                            chunks, candidate, backend=self.backend) >= self.max_blockers:
                        reject("line_of_sight")
                        continue
                    local = self.cache.load(self.root_path / entry["path"], self.feature_count, entry["num_points_in_gt"])
                    world = transform_object(local, entry, candidate, rng,
                        density=self.flags["DENSITY_SUBSAMPLE"], intensity=self.flags["RADIOMETRIC_CALIBRATION"],
                        minimum_points=self.min_sample_points, gamma=self.gamma)
                    if len(world) < self.min_sample_points:
                        reject("insufficient_points")
                        continue

                    deletions, inside_count, shadow_count = [], 0, 0
                    # Ground/static/LOS operate on surviving points, as do shadow masks.
                    for cloud, keep in chunks:
                        interior = points_in_boxes_union(cloud, candidate[None], self.extra_width, backend=self.backend) & keep
                        shadow = (shadow_mask(cloud, candidate, backend=self.backend) & keep & ~interior
                                  if self.flags["SHADOW_MASKING"] else np.zeros(len(cloud), bool))
                        deletions.append(interior | shadow)
                        inside_count += int(interior.sum())
                        shadow_count += int(shadow.sum())
                    if protect:
                        if original_indices is None:
                            original_indices = self._visibility_indices(points, boxes)
                        original_keep = chunks[0][1] & ~deletions[0]
                        counts = [(len(indices), int(original_keep[indices].sum())) for indices in original_indices]
                        counts += [(len(cloud), int((keep & ~delete).sum()))
                                   for (cloud, keep), delete in zip(chunks[1:], deletions[1:])]
                        if any(base > 0 and (after < min(base, self.min_visible_points)
                                or after/base < self.min_visible_ratio) for base, after in counts):
                            reject("visibility")
                            continue
                    # Commit atomically after every check. No candidate mutates inputs.
                    for (_, keep), delete in zip(chunks, deletions):
                        keep &= ~delete
                    chunks.append((world, np.ones(len(world), bool)))
                    accepted.append(candidate)
                    meta["accepted_by_class"][name] += 1
                    meta["removed_interior_points"] += inside_count
                    meta["removed_shadow_points"] += shadow_count
                    meta["inserted_sources"].append({"class": name, "frame_id": entry["image_idx"],
                                                      "gt_idx": entry["gt_idx"]})
                    break
        if accepted:
            result_points = np.concatenate([cloud[keep] for cloud, keep in chunks])
            result_boxes = np.concatenate([boxes, np.stack(accepted)])
        else:
            result_points, result_boxes = points, boxes
        if return_metadata:
            meta.update(num_inserted=len(accepted), cache=self.cache.stats(),
                point_is_sampled=np.concatenate([np.full(int(keep.sum()), i > 0, bool)
                                                for i, (_, keep) in enumerate(chunks)]),
                box_is_sampled=np.r_[np.zeros(len(boxes), bool), np.ones(len(accepted), bool)])
            return result_points, result_boxes, meta
        return result_points, result_boxes
