"""Explicit bottom-center LiDAR/KITTI boxes and real camera projections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Calibration:
    velo_to_rect: np.ndarray
    p2: np.ndarray

    def __post_init__(self):
        transform, projection = np.array(self.velo_to_rect, dtype=np.float64), np.array(self.p2, dtype=np.float64)
        if (transform.shape != (4, 4) or projection.shape != (3, 4) or
                not np.isfinite(transform).all() or not np.isfinite(projection).all() or
                not np.allclose(transform[3], [0, 0, 0, 1]) or
                not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3) or
                np.linalg.det(transform[:3, :3]) <= 0 or
                not np.allclose(projection[2], [0, 0, 1, 0])):
            raise ValueError("Invalid KITTI calibration matrices")
        # Camera X/Z heading plane -> LiDAR X/Y, matching prepare_kitti.
        inverse = np.linalg.inv(transform)[:3, :3]
        heading = inverse[:2, [0, 2]].copy()
        heading[:, 1] *= -1
        if abs(np.linalg.det(heading)) < 1e-8:
            raise ValueError("Calibration cannot map upright camera/LiDAR headings")
        transform.setflags(write=False)
        projection.setflags(write=False)
        object.__setattr__(self, "velo_to_rect", transform)
        object.__setattr__(self, "p2", projection)


def read_calibration(path):
    try:
        from .prepare_kitti import parse_calibration
    except ImportError:
        from prepare_kitti import parse_calibration
    path = Path(path)
    values = {key.strip(): np.fromstring(payload, sep=" ")
              for line in path.read_text().splitlines() if ":" in line
              for key, payload in [line.split(":", 1)]}
    if "P2" not in values or values["P2"].size != 12:
        raise ValueError(f"Missing valid P2 calibration: {path}")
    return Calibration(parse_calibration(path), values["P2"].reshape(3, 4))


def _box(box):
    value = np.asarray(box, dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all() or (value[3:6] <= 0).any():
        raise ValueError("Box must contain finite [x,y,z_bottom,l,w,h,yaw] with positive dimensions")
    return value


def camera_box_to_lidar(box, calibration):
    """Camera [x,y,z_bottom,l,w,h,rotation_y] -> upright LiDAR box."""
    box = _box(box)
    inverse = np.linalg.inv(calibration.velo_to_rect)
    center = (inverse @ np.r_[box[:3], 1.])[:3]
    direction = inverse[:3, :3] @ np.array([np.cos(box[6]), 0., -np.sin(box[6])])
    return np.r_[center, box[3:6], np.arctan2(direction[1], direction[0])]


def lidar_box_to_camera(box, calibration):
    """Invert prepare_kitti's upright heading projection, using full calibration.

    KITTI boxes and this detector encode yaw only. Inverting the 2D heading map
    preserves camera label yaw through nonzero rectification pitch/roll; no
    front/back orientation or independently predicted pitch/roll is claimed.
    """
    box = _box(box)
    center = (calibration.velo_to_rect @ np.r_[box[:3], 1.])[:3]
    inverse = np.linalg.inv(calibration.velo_to_rect)[:3, :3]
    heading = inverse[:2, [0, 2]].copy()
    heading[:, 1] *= -1
    direction = np.linalg.solve(heading, [np.cos(box[6]), np.sin(box[6])])
    return np.r_[center, box[3:6], np.arctan2(direction[1], direction[0])]


def camera_box_corners(box):
    box = _box(box)
    x, y, z, length, width, height, yaw = box
    base = np.array([[-length/2, 0, -width/2], [length/2, 0, -width/2],
                     [length/2, 0, width/2], [-length/2, 0, width/2]])
    corners = np.concatenate([base, base + [0, -height, 0]])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return corners @ rotation.T + [x, y, z]


def project_camera_box(box, calibration, *, image_size=None, near_plane=.1):
    """Clip the 12 box edges to camera z>=near, then to the image rectangle.

    Return None for entirely hidden/off-image/degenerate boxes; never dummy bboxes.
    image_size is (width,height). Without it, retain the genuine unbounded bbox.
    """
    if not np.isfinite(near_plane) or near_plane <= 0:
        raise ValueError("near_plane must be finite and positive")
    corners = camera_box_corners(box)
    points = list(corners[corners[:, 2] >= near_plane])
    edges = [(i, (i+1) % 4) for i in range(4)] + [(i+4, (i+1) % 4+4) for i in range(4)] + [(i,i+4) for i in range(4)]
    for a, b in edges:
        first, second = corners[a], corners[b]
        if (first[2] >= near_plane) != (second[2] >= near_plane):
            points.append(first + (second-first) * (near_plane-first[2])/(second[2]-first[2]))
    if not points:
        return None
    projected = np.c_[np.array(points), np.ones(len(points))] @ calibration.p2.T
    pixels = projected[:, :2] / projected[:, 2:3]
    bbox = np.r_[pixels.min(axis=0), pixels.max(axis=0)]
    if image_size is not None:
        width, height = image_size
        if not np.isfinite([width,height]).all() or width <= 0 or height <= 0:
            raise ValueError("image_size must be positive (width,height)")
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width-1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height-1)
    if not np.isfinite(bbox).all():
        raise FloatingPointError("Non-finite projected image box")
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def empty_annotation():
    return {"name": np.empty(0, dtype=str), "bbox": np.empty((0, 4)),
            "dimensions": np.empty((0, 3)), "location": np.empty((0, 3)),
            "rotation_y": np.empty(0), "alpha": np.empty(0), "score": np.empty(0),
            "truncated": np.empty(0), "occluded": np.empty(0, dtype=np.int64)}


def predictions_to_annotation(rows, calibration, objects, *, image_size=None, near_plane=.1):
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 9 or not np.isfinite(rows).all():
        raise ValueError("3D rows must be finite [N,9]")
    names = {identifier: name for name, identifier in objects.items()}
    data = empty_annotation()
    collected = {key: [] for key in data}
    for row in rows:
        identifier = int(row[0])
        if row[0] != identifier or identifier not in names or not 0 <= row[1] <= 1:
            raise ValueError("Invalid prediction class ID or score")
        camera = lidar_box_to_camera(row[2:], calibration)
        bbox = project_camera_box(camera, calibration, image_size=image_size, near_plane=near_plane)
        if bbox is None:
            continue
        values = {"name": names[identifier], "bbox": bbox, "dimensions": camera[[3,5,4]],
                  "location": camera[:3], "rotation_y": camera[6], "alpha": -10.,
                  "score": row[1], "truncated": 0., "occluded": 0}
        for key, value in values.items():
            collected[key].append(value)
    if collected["name"]:
        data = {key: np.asarray(value, dtype=str if key == "name" else data[key].dtype).reshape((-1, *data[key].shape[1:]))
                for key,value in collected.items()}
    return data
