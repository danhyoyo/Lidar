"""Transactional GT copy-paste with optional physics heuristics.

Counts are maximum additions, not target scene totals. Box collision and volume
clearing always apply, including enable_physics=False. Explicit keyword flags
win over the legacy enable_physics preset. Ground/LOS/density and solid-box
shadows are heuristics; these checks do not establish sensor realism or mAP.
"""
import pickle
from numbers import Integral, Real
from typing import Dict, Optional

import numpy as np
from core.datasets.utils_1.box_geometry import points_in_box
from core.datasets.utils_1.collision_ground import (
    check_box_collision_2d, check_ground_support,
    check_static_obstacle_collision, snap_box_to_ground,
)
from core.datasets.utils_1.physics_aug import (
    check_line_of_sight_occlusion, distance_adaptive_subsample,
    shadow_point_mask, radiometric_intensity_calibrate,
)

DEFAULT_SAMPLE_COUNTS = {'Pedestrian': 6, 'Cyclist': 5, 'Car': 3}
PHYSICS_FLAGS = (
    'enable_ground_validation', 'enable_static_collision', 'enable_line_of_sight',
    'enable_shadow_masking', 'enable_density_subsample', 'enable_radiometric_calibration',
)


def _probability(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return float(value)


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def build_gt_sampler(pcu_cfg):
    """Create a sampler with the same effective settings in Dataset and audit."""
    kwargs = {name: pcu_cfg.get(name) for name in PHYSICS_FLAGS}
    return GTSampler(
        database_path=pcu_cfg.get('gt_database_path', ''),
        sample_counts=pcu_cfg.get('sample_counts'), p=pcu_cfg.get('p', 1.0),
        enable_physics=pcu_cfg.get('enable_physics', True),
        min_visible_points=pcu_cfg.get('min_visible_points', 5),
        min_visible_ratio=pcu_cfg.get('min_visible_ratio', .5), **kwargs,
    )


class GTSampler:
    def __init__(self, database_path: str, sample_counts: Optional[Dict[str, int]] = None,
                 p: float = 1.0, enable_physics: bool = True, *,
                 enable_ground_validation=None, enable_static_collision=None,
                 enable_line_of_sight=None, enable_shadow_masking=None,
                 enable_density_subsample=None, enable_radiometric_calibration=None,
                 min_visible_points=5, min_visible_ratio=.5):
        self.p = _probability(p, 'p')
        if not isinstance(enable_physics, (bool, np.bool_)):
            raise ValueError('enable_physics must be bool')
        self.enable_physics = bool(enable_physics)
        explicit_flags = locals()
        for name in PHYSICS_FLAGS:
            value = explicit_flags[name]
            if value is not None and not isinstance(value, (bool, np.bool_)):
                raise ValueError(f'{name} must be bool or None')
            setattr(self, name, self.enable_physics if value is None else bool(value))
        self.min_visible_points = _nonnegative_integer(min_visible_points, 'min_visible_points')
        self.min_visible_ratio = _probability(min_visible_ratio, 'min_visible_ratio')
        counts = DEFAULT_SAMPLE_COUNTS if sample_counts is None else sample_counts
        if not isinstance(counts, dict) or not all(isinstance(name, str) for name in counts):
            raise ValueError('sample_counts must map class names to nonnegative integers')
        self.sample_counts = {name: _nonnegative_integer(count, f'sample_counts[{name}]')
                              for name, count in counts.items()}
        try:
            with open(database_path, 'rb') as stream:
                self.database = pickle.load(stream)
        except OSError:
            raise
        except Exception as error:
            raise ValueError(f'Failed to load GT database {database_path}: {error}') from error
        self._validate_database(database_path)

    def _validate_database(self, path):
        if not isinstance(self.database, dict):
            raise ValueError(f'GT database {path}: expected class mapping')
        for name, samples in self.database.items():
            if not isinstance(name, str) or not isinstance(samples, (list, tuple)):
                raise ValueError(f'GT database {path}: class {name} must contain a sample list')
            for index, sample in enumerate(samples):
                context = f'GT database {path}: {name} sample {index}'
                try:
                    if not isinstance(sample, dict):
                        raise ValueError('sample must be a mapping')
                    box, points = np.asarray(sample['box']), np.asarray(sample['points'])
                    if box.shape != (8,) or not np.isfinite(box).all() or np.any(box[1:4] <= 0):
                        raise ValueError('box must have 8 finite values with positive dimensions')
                    if points.ndim != 2 or points.shape[1] != 4 or len(points) == 0 or not np.isfinite(points).all():
                        raise ValueError('points must be nonempty finite (N, 4)')
                    count = _nonnegative_integer(sample['num_points'], 'num_points')
                    if count != len(points):
                        raise ValueError('num_points does not match points')
                    origin = sample['r_origin']
                    if isinstance(origin, (bool, np.bool_)) or not isinstance(origin, Real) or not np.isfinite(origin) or origin <= 0:
                        raise ValueError('r_origin must be finite and positive')
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f'{context}: {error}') from error
        for name, count in self.sample_counts.items():
            if count > 0 and not self.database.get(name):
                raise ValueError(f'GT database {path}: requested class {name} has no samples')

    @staticmethod
    def _validate_scene(lidar, boxes):
        if not isinstance(lidar, np.ndarray) or lidar.ndim != 2 or lidar.shape[1] != 4:
            raise ValueError('lidar must have shape (N, 4)')
        if not isinstance(boxes, np.ndarray) or boxes.ndim != 2 or boxes.shape[1] not in (7, 8):
            raise ValueError('boxes must have shape (M, 7) or (M, 8)')
        try:
            finite = np.isfinite(lidar).all() and np.isfinite(boxes).all()
            dimensions = boxes[:, -7:-4]
            valid_dimensions = np.all(dimensions > 0)
        except TypeError as error:
            raise ValueError('scene arrays must be numeric') from error
        if not finite or not valid_dimensions:
            raise ValueError('scene values must be finite and box dimensions positive')

    @staticmethod
    def _propose_pose(cls_name, attempt):
        # 14 corridor attempts, then 6 full-rectangle attempts. X (not radial
        # range) is stratified near/mid/far with the existing 30/40/30 weights.
        if attempt < 14:
            tier = np.random.random()
            if tier < .30:
                x = np.random.uniform(8., 25.)
            elif tier < .70:
                x = np.random.uniform(25., 45.)
            else:
                x = np.random.uniform(45., 64. if cls_name == 'Car' else 65.5)
            limit = {'Pedestrian': 15., 'Cyclist': 13.}.get(cls_name, 9.5)
            y = np.random.uniform(-limit, limit)
        else:
            x, y = np.random.uniform(6., 65.), np.random.uniform(-25., 25.)
        return x, y, np.random.uniform(-np.pi, np.pi)

    def __call__(self, lidar, boxes, return_metadata=False, *, return_diagnostics=False):
        self._validate_scene(lidar, boxes)
        return_metadata = return_metadata or return_diagnostics
        n_original = len(boxes)
        meta = {'inserted_boxes': [], 'inserted_points': [], 'inserted_class_names': [],
                'num_original_points': len(lidar), 'num_original_boxes': n_original,
                'final_boxes': boxes.copy(), 'num_inserted': 0}
        diag = None
        if return_diagnostics:
            diag = {'attempted_by_class': {name: 0 for name in self.sample_counts},
                    'accepted_by_class': {name: 0 for name in self.sample_counts},
                    'rejected_by_reason': {}, 'removed_interior_points': 0,
                    'removed_shadow_points': 0, 'box_visibility': [], 'inserted_positions': []}
            meta['diagnostics'] = diag

        cur_points = lidar.copy()
        new_boxes = [box.copy() for box in boxes]
        # Membership is fixed to original/source identities, can overlap for
        # original boxes, and shrinks only on committed scene deletions.
        memberships = [points_in_box(cur_points, box) for box in boxes]
        baselines = [int(mask.sum()) for mask in memberships]

        def reject(reason):
            if diag is not None:
                rejected = diag['rejected_by_reason']
                rejected[reason] = rejected.get(reason, 0) + 1

        active = self.p > 0 and any(self.sample_counts.values())
        if active and self.p < 1:
            active = np.random.random() < self.p
        if active:
            for cls_name, count in self.sample_counts.items():
                if count == 0:
                    continue
                samples = self.database[cls_name]
                chosen = np.random.choice(len(samples), size=min(count * 3, len(samples)), replace=False)
                inserted = 0
                for index in chosen:
                    if inserted >= count:
                        break
                    sample = samples[index]
                    for attempt in range(20):
                        if diag is not None:
                            diag['attempted_by_class'][cls_name] += 1
                        x, y, yaw = self._propose_pose(cls_name, attempt)
                        if not (2 <= x <= 66 and -35 <= y <= 35):
                            reject('bounds')
                            continue
                        candidate = np.asarray(sample['box'], dtype=np.result_type(boxes.dtype, np.float32)).copy()
                        if boxes.shape[1] == 7:
                            candidate = candidate[1:]
                        candidate[-4], candidate[-3], candidate[-1] = x, y, yaw
                        existing = np.asarray(new_boxes).reshape(-1, boxes.shape[1])
                        if check_box_collision_2d(candidate, existing, min_margin=1. if cls_name == 'Car' else .8):
                            reject('box_collision')
                            continue
                        if self.enable_ground_validation:
                            support, ground = check_ground_support(candidate, cur_points)
                            if not support:
                                reject('ground')
                                continue
                            candidate[-2] = ground
                        else:
                            candidate = snap_box_to_ground(candidate, cur_points)
                        if self.enable_static_collision and check_static_obstacle_collision(candidate, cur_points, max_obstacle_points=2):
                            reject('static_collision')
                            continue
                        if self.enable_line_of_sight and check_line_of_sight_occlusion(candidate, cur_points):
                            reject('line_of_sight')
                            continue
                        target_range = float(np.hypot(x, y))
                        object_points = np.asarray(sample['points']).copy()
                        if self.enable_density_subsample:
                            object_points = distance_adaptive_subsample(object_points, candidate, sample['r_origin'], target_range)
                        if len(object_points) == 0:
                            reject('empty_candidate')
                            continue
                        if self.enable_radiometric_calibration:
                            object_points = radiometric_intensity_calibrate(object_points, sample['r_origin'], target_range)
                        c, s = np.cos(yaw), np.sin(yaw)
                        world = object_points.copy()
                        world[:, 0] = object_points[:, 0] * c - object_points[:, 1] * s + x
                        world[:, 1] = object_points[:, 0] * s + object_points[:, 1] * c + y
                        world[:, 2] = object_points[:, 2] + candidate[-2]
                        world = world.astype(np.result_type(lidar.dtype, np.float32), copy=False)
                        interior = points_in_box(cur_points, candidate)
                        shadow = (shadow_point_mask(cur_points, candidate) & ~interior
                                  if self.enable_shadow_masking else np.zeros(len(cur_points), bool))
                        keep = ~(interior | shadow)
                        remaining = [int(np.count_nonzero(mask & keep)) for mask in memberships]
                        if any(base > 0 and (after < min(base, self.min_visible_points)
                                             or after / base < self.min_visible_ratio)
                               for base, after in zip(baselines, remaining)):
                            reject('visibility')
                            continue
                        # Commit only after every check. No rejected attempt can
                        # touch points, labels, memberships, removal counters or metadata.
                        survivors = cur_points[keep]
                        memberships = [np.r_[mask[keep], np.zeros(len(world), bool)] for mask in memberships]
                        memberships.append(np.r_[np.zeros(len(survivors), bool), np.ones(len(world), bool)])
                        baselines.append(len(world))
                        cur_points = np.vstack((survivors, world))
                        new_boxes.append(candidate)
                        inserted += 1
                        if return_metadata:
                            meta['inserted_boxes'].append(candidate.copy())
                            meta['inserted_class_names'].append(cls_name)
                        if diag is not None:
                            diag['accepted_by_class'][cls_name] += 1
                            diag['removed_interior_points'] += int(interior.sum())
                            diag['removed_shadow_points'] += int(shadow.sum())
                            diag['inserted_positions'].append({'class_name': cls_name, 'x': float(x), 'range': target_range})
                        break

        final_boxes = (np.asarray(new_boxes, dtype=np.result_type(boxes.dtype, np.float32))
                       if len(new_boxes) > n_original else boxes.copy())
        if return_metadata:
            meta['final_boxes'] = final_boxes.copy()
            meta['num_inserted'] = len(new_boxes) - n_original
            meta['inserted_points'] = [cur_points[mask].copy() for mask in memberships[n_original:]]
            if diag is not None:
                diag['box_visibility'] = [{'box_index': index, 'is_inserted': index >= n_original,
                                          'baseline_count': base, 'final_count': int(mask.sum())}
                                         for index, (base, mask) in enumerate(zip(baselines, memberships))]
            return cur_points, final_boxes, meta
        return cur_points, final_boxes
