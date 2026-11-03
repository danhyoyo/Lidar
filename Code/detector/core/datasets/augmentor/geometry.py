"""Geometry for [class, h, w, l, x, y, z_bottom, yaw] boxes."""

import numpy as np
from shapely.geometry import Polygon


def points_in_box(points, box, extra_width=(0., 0., 0.)):
    """Select points in a rotated 3D box; extra_width is per-side [x,y,z]."""
    xyz = points[:, :3].astype(np.float64)
    delta = xyz[:, :2] - box[4:6]
    cosine, sine = np.cos(float(box[7])), np.sin(float(box[7]))
    local_x = delta[:, 0] * cosine + delta[:, 1] * sine
    local_y = -delta[:, 0] * sine + delta[:, 1] * cosine
    ex, ey, ez = extra_width
    return ((np.abs(local_x) <= float(box[3]) / 2 + ex) &
            (np.abs(local_y) <= float(box[2]) / 2 + ey) &
            (xyz[:, 2] >= float(box[6]) - ez) &
            (xyz[:, 2] <= float(box[6]) + float(box[1]) + ez))


def bev_polygon(box):
    length, width = float(box[3]), float(box[2])
    corners = np.array([[-length/2, -width/2], [length/2, -width/2],
                        [length/2, width/2], [-length/2, width/2]])
    cosine, sine = np.cos(float(box[7])), np.sin(float(box[7]))
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return Polygon(corners @ rotation.T + box[4:6])
