"""Omni-PCU State-of-the-Art 3D LiDAR Database Sampler."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .omni_cache import BoundedPointCache
from .omni_geometry import (
    check_collision_2d_vectorized,
    points_in_oriented_box_3d,
    project_to_road_plane,
)


def _parse_class_counts(values: list[str], class_names: dict[str, int]) -> dict[str, int]:
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


class OmniDataBaseSampler:
    """SOTA Physics-Consistent & Curricular GT Sampler.
    
    Unifies OpenPCDet split integrity, MMDetection3D road planes,
    PCU physics-consistent simulation, and COM curriculum learning.
    Compatible with both modern and legacy configuration profiles.
    """

    def __init__(
        self,
        root_path: Path | str,
        config: dict,
        class_names: dict[str, int],
        *,
        geometry: dict | None = None,
        allowed_frame_ids: list[str] | None = None,
    ):
        self.root_path = Path(root_path)
        self.class_names = dict(class_names)
        self.geometry = geometry
        self.config = config
        self.feature_count = int(config.get("NUM_POINT_FEATURES", 4))
        self.limit_whole_scene = bool(config.get("LIMIT_WHOLE_SCENE", True))
        self.quotas = _parse_class_counts(config.get("SAMPLE_GROUPS", []), class_names)

        # Placement parameters (with graceful defaults for legacy configs)
        placement = config.get("PLACEMENT", {})
        self.placement_mode = placement.get("MODE", "stratified_range")
        self.range_tiers = placement.get("RANGE_TIERS", [[8.0, 25.0], [25.0, 45.0], [45.0, 65.5]])
        self.tier_weights = placement.get("TIER_WEIGHTS", [0.30, 0.40, 0.30])
        self.max_attempts = int(placement.get("MAX_ATTEMPTS", 15))

        # Physics heuristics (with graceful defaults)
        physics = config.get("PHYSICS", {})
        self.enable_density_subsample = bool(physics.get("ENABLE_DENSITY_SUBSAMPLE", True))
        self.enable_anti_wall = bool(physics.get("ENABLE_ANTI_WALL", True))
        self.anti_wall_height = float(physics.get("ANTI_WALL_HEIGHT_THRESH", 0.35))
        self.max_obstacle_points = int(physics.get("MAX_OBSTACLE_POINTS", 3))
        self.min_visible_ratio = float(physics.get("MIN_VISIBLE_RATIO", 0.50))
        self.min_visible_points = int(physics.get("MIN_VISIBLE_POINTS", 5))

        # Curriculum scheduling (COM) - opt-in or defaults
        curr_cfg = config.get("CURRICULUM", {})
        self.curriculum_enabled = bool(curr_cfg.get("ENABLED", False))
        self.warmup_epochs = max(1, int(curr_cfg.get("WARMUP_EPOCHS", 10)))
        self.hard_ratio_base = float(curr_cfg.get("HARD_RATIO_BASE", 0.15))
        self.hard_ratio_target = float(curr_cfg.get("HARD_RATIO_TARGET", 0.70))
        self.diff_threshold = float(curr_cfg.get("DIFFICULTY_THRESHOLD", 0.45))
        self.epoch = 0
        self.current_hard_ratio = self.hard_ratio_base

        # Cache (PERFORMANCE)
        cache_cfg = config.get("PERFORMANCE", {})
        cache_size = float(cache_cfg.get("CACHE_SIZE_MB", 128.0))
        self.cache = BoundedPointCache(cache_size) if cache_cfg.get("CACHE_ENABLED", True) else None

        # Load and filter database
        self.db_infos = {name: [] for name in class_names}
        self.easy_pools = {name: [] for name in class_names}
        self.hard_pools = {name: [] for name in class_names}
        self._load_database(config.get("DB_INFO_PATH", []), allowed_frame_ids)
        self._apply_prepare_filters(config.get("PREPARE", {}))
        self._partition_curriculum()

    def _load_database(self, paths: list[str], allowed_frames: list[str] | None) -> None:
        if not isinstance(paths, list) or not paths:
            raise ValueError("DB_INFO_PATH must be a nonempty list of JSON paths")
        allowed = set(allowed_frames) if allowed_frames is not None else None
        for rel_path in paths:
            with (self.root_path / rel_path).open(encoding="utf-8") as stream:
                meta = json.load(stream)
            if meta.get("format") != "lidar_gt_database_v1":
                raise ValueError("Unsupported GT database format; requires lidar_gt_database_v1")
            source_ids = set(meta.get("source_frame_ids", []))
            if allowed is not None and not source_ids <= allowed:
                raise ValueError("GT database contains frames outside the current training split")
            for name in self.class_names:
                for entry in meta.get("db_infos", {}).get(name, []):
                    # Automatic backward compatibility: derive r_origin and density if absent
                    if "r_origin" not in entry:
                        entry["r_origin"] = float(np.hypot(entry["box3d_lidar"][0], entry["box3d_lidar"][1]))
                    if "density" not in entry:
                        box = entry["box3d_lidar"]
                        vol = max(1e-4, box[3] * box[4] * box[5])
                        entry["density"] = float(entry.get("num_points_in_gt", 1) / vol)
                    self.db_infos[name].append(entry)

    def _apply_prepare_filters(self, prepare: dict) -> None:
        if not prepare:
            return
        minima = _parse_class_counts(prepare.get("filter_by_min_points", []), self.class_names)
        excluded_difficulties = set(prepare.get("filter_by_difficulty", []))
        for name in self.class_names:
            min_pts = minima.get(name, 0)
            self.db_infos[name] = [
                e for e in self.db_infos[name]
                if e["num_points_in_gt"] >= min_pts
                and e.get("difficulty") not in excluded_difficulties
            ]

    def _partition_curriculum(self) -> None:
        ref_counts = {"Pedestrian": 50.0, "Cyclist": 100.0, "Car": 300.0}
        for name, entries in self.db_infos.items():
            ref_pts = ref_counts.get(name, 100.0)
            for entry in entries:
                r = entry.get("r_origin", 20.0)
                pts = entry.get("num_points_in_gt", 50)
                diff = 0.5 * (min(1.0, r / 70.0) + max(0.0, 1.0 - pts / ref_pts))
                if diff > self.diff_threshold:
                    self.hard_pools[name].append(entry)
                else:
                    self.easy_pools[name].append(entry)
            # Ensure pools are never empty
            if not self.easy_pools[name]:
                self.easy_pools[name] = list(entries)
            if not self.hard_pools[name]:
                self.hard_pools[name] = list(entries)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, epoch)
        progress = min(1.0, self.epoch / max(1, self.warmup_epochs))
        self.current_hard_ratio = progress * self.hard_ratio_target + (1.0 - progress) * self.hard_ratio_base

    def _sample_candidates(self, class_name: str, count: int, rng: np.random.Generator) -> list[dict]:
        entries = self.db_infos[class_name]
        if count <= 0 or not entries:
            return []
        if not self.curriculum_enabled:
            indices = rng.choice(len(entries), size=min(count, len(entries)), replace=False)
            return [entries[i] for i in indices]

        sampled = []
        for _ in range(count):
            draw_hard = (rng.uniform(0.0, 1.0) < self.current_hard_ratio)
            pool = self.hard_pools[class_name] if draw_hard else self.easy_pools[class_name]
            idx = rng.integers(0, len(pool))
            sampled.append(pool[idx])
        return sampled

    def _propose_pose(
        self, class_name: str, attempt: int, sample: dict, rng: np.random.Generator
    ) -> tuple[float, float, float]:
        if self.placement_mode == "source":
            # Direct placement as in baseline OpenPCDet
            box = sample["box3d_lidar"]
            return float(box[0]), float(box[1]), float(box[6])

        # Stratified range corridor proposal
        if attempt < 10:
            tier_idx = rng.choice(len(self.range_tiers), p=self.tier_weights)
            r_min, r_max = self.range_tiers[tier_idx]
            x = rng.uniform(r_min, r_max)
            limit_y = {"Pedestrian": 15.0, "Cyclist": 13.0}.get(class_name, 9.5)
            y = rng.uniform(-limit_y, limit_y)
        else:
            x = rng.uniform(5.0, 65.0)
            y = rng.uniform(-25.0, 25.0)
        yaw = rng.uniform(-np.pi, np.pi)
        return float(x), float(y), float(yaw)

    def _load_points(self, entry: dict) -> np.ndarray:
        key = entry["path"]
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        pts = np.fromfile(self.root_path / key, dtype=np.float32).reshape(-1, self.feature_count)
        if self.cache is not None:
            self.cache.put(key, pts)
        return pts.copy()

    def __call__(
        self,
        points: np.ndarray,
        boxes: np.ndarray,
        road_plane: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(road_plane, (np.random.Generator, np.random.RandomState)) and rng is None:
            rng = road_plane
            road_plane = None

        if rng is None:
            rng = np.random.default_rng()

        cur_points = points.copy()
        new_boxes = list(boxes.copy())

        # Baseline point membership for pre-existing boxes
        memberships = [points_in_oriented_box_3d(cur_points, b) for b in boxes]
        baselines = [int(m.sum()) for m in memberships]

        inserted_points_list = []

        for class_name, quota in self.quotas.items():
            cls_id = self.class_names[class_name]
            if self.limit_whole_scene and len(boxes) > 0:
                existing_count = int(np.sum(boxes[:, 0] == cls_id))
                quota = max(0, quota - existing_count)
            if quota <= 0:
                continue

            candidates = self._sample_candidates(class_name, quota, rng)
            for entry in candidates:
                length = float(entry["box3d_lidar"][3])
                width = float(entry["box3d_lidar"][4])
                height = float(entry["box3d_lidar"][5])
                src_r = float(entry.get("r_origin", np.hypot(entry["box3d_lidar"][0], entry["box3d_lidar"][1])))

                committed = False
                for attempt in range(self.max_attempts):
                    x, y, yaw = self._propose_pose(class_name, attempt, entry, rng)
                    target_r = float(np.hypot(x, y))

                    # Spatial boundary check
                    if self.geometry is not None:
                        geom = self.geometry
                        if not (geom["x_min"] < x < geom["x_max"] and geom["y_min"] < y < geom["y_max"]):
                            continue

                    # 1. Height adjustment
                    if road_plane is not None:
                        z_ground = float(project_to_road_plane(x, y, road_plane))
                    else:
                        # Fallback: estimate ground from 5th percentile of local points
                        dist_sq = (cur_points[:, 0] - x) ** 2 + (cur_points[:, 1] - y) ** 2
                        local_pts = cur_points[dist_sq <= 9.0]
                        z_ground = float(np.percentile(local_pts[:, 2], 5)) if len(local_pts) >= 5 else -1.65

                    candidate_box = np.array(
                        [cls_id, height, width, length, x, y, z_ground, yaw], dtype=np.float32
                    )

                    # 2. Collision test with existing and inserted boxes
                    existing_all = np.array(new_boxes, dtype=np.float32) if new_boxes else np.empty((0, 8))
                    if check_collision_2d_vectorized(candidate_box[None, :], existing_all, min_margin=0.3)[0]:
                        continue

                    # 3. Anti-wall static obstacle check
                    if self.enable_anti_wall and len(cur_points) > 0:
                        in_box = points_in_oriented_box_3d(cur_points, candidate_box)
                        obstacles = cur_points[in_box & (cur_points[:, 2] > (z_ground + self.anti_wall_height))]
                        if len(obstacles) > self.max_obstacle_points:
                            continue

                    # 4. Point loading & physical density subsampling
                    local_pts = self._load_points(entry)
                    if self.enable_density_subsample and target_r > src_r:
                        ratio = min(1.0, (src_r / target_r) ** 2)
                        keep_mask = rng.uniform(0.0, 1.0, size=len(local_pts)) <= ratio
                        if keep_mask.sum() < 5 and len(local_pts) >= 5:
                            keep_idx = rng.choice(len(local_pts), size=5, replace=False)
                            local_pts = local_pts[keep_idx]
                        else:
                            local_pts = local_pts[keep_mask]

                    if len(local_pts) == 0:
                        continue

                    # Transform local points to world frame
                    c, s = np.cos(yaw), np.sin(yaw)
                    world_pts = local_pts.copy()
                    world_pts[:, 0] = local_pts[:, 0] * c - local_pts[:, 1] * s + x
                    world_pts[:, 1] = local_pts[:, 0] * s + local_pts[:, 1] * c + y
                    world_pts[:, 2] = local_pts[:, 2] + (z_ground + height * 0.5)

                    # 5. Visibility check on pre-existing objects
                    inside_cand = points_in_oriented_box_3d(cur_points, candidate_box)
                    violated = False
                    for b_idx, (base_cnt, mem_mask) in enumerate(zip(baselines, memberships)):
                        surviving = np.count_nonzero(mem_mask & ~inside_cand)
                        if base_cnt > 0 and (
                            surviving < min(base_cnt, self.min_visible_points)
                            or (surviving / base_cnt) < self.min_visible_ratio
                        ):
                            violated = True
                            break
                    if violated:
                        continue

                    # 6. Commit transaction
                    surv_mask = ~inside_cand
                    cur_points = cur_points[surv_mask]
                    memberships = [m[surv_mask] for m in memberships]

                    new_zeros = np.zeros(len(world_pts), dtype=bool)
                    memberships = [np.concatenate([m, new_zeros]) for m in memberships]

                    new_mem = np.concatenate([
                        np.zeros(len(cur_points), dtype=bool),
                        np.ones(len(world_pts), dtype=bool),
                    ])
                    cur_points = np.vstack([cur_points, world_pts])
                    memberships.append(new_mem)
                    baselines.append(len(world_pts))

                    new_boxes.append(candidate_box)
                    committed = True
                    break

                if not committed:
                    continue

        final_boxes = np.array(new_boxes, dtype=np.float32) if new_boxes else np.empty((0, 8), dtype=np.float32)
        return cur_points, final_boxes
