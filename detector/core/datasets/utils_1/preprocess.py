import numpy as np
import math


def _grid_shape(geometry):
    shape = []
    for axis in "xyz":
        lower = float(geometry[f"{axis}_min"])
        upper = float(geometry[f"{axis}_max"])
        resolution = float(geometry[f"{axis}_res"])
        cells = (upper - lower) / resolution
        if not np.isfinite([lower, upper, resolution, cells]).all() or resolution <= 0 or cells <= 0:
            raise ValueError(f"invalid {axis}-axis geometry")
        if not np.isclose(cells, round(cells), atol=1e-6):
            raise ValueError(f"{axis}-axis range must be divisible by its resolution")
        shape.append(int(round(cells)))
    return tuple(shape)


def _rich_channels(encoding, name):
    """Number of height bands for a rich encoding; ``rich8``=3, ``rich11``=6."""
    default = 6 if name == "rich11" else 3
    n_bands = int(encoding.get("n_bands", default))
    if n_bands < 1:
        raise ValueError("n_bands must be a positive integer")
    return n_bands


def encode_bev(points, geometry, bev_encoding=None):
    """Encode KITTI ``(x, y, z, intensity)`` points as legacy or RichBEV.

    ``binary_slices`` keeps the 35-channel z-occupancy tensor. ``rich8``/``rich11``
    emit ``n_bands + 5`` channels: ``n_bands`` normalized height-band occupancies
    followed by ``[z_max, z_mean, i_max, i_mean, density]``.
    """
    encoding = bev_encoding or {"name": "binary_slices"}
    name = encoding.get("name", "binary_slices")
    if name == "binary_slices":
        return voxelize(points, geometry)
    if name not in {"rich8", "rich11"}:
        raise ValueError(f"unsupported BEV encoding: {name!r}")
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"{name} expects points shaped (N, >=4)")

    density_norm = float(encoding.get("density_norm", 32.0))
    intensity_scale = float(encoding.get("intensity_scale", 1.0))
    if not np.isfinite(density_norm) or density_norm <= 1:
        raise ValueError("density_norm must be finite and greater than 1")
    if not np.isfinite(intensity_scale) or intensity_scale <= 0:
        raise ValueError("intensity_scale must be finite and greater than 0")
    n_bands = _rich_channels(encoding, name)
    n_channels = n_bands + 5

    x_size, y_size, _ = _grid_shape(geometry)
    output = np.zeros((n_channels, y_size * x_size), dtype=np.float32)
    eps = 0.001
    keep = np.isfinite(points[:, :4]).all(axis=1)
    for column, axis in enumerate("xyz"):
        keep &= points[:, column] > float(geometry[f"{axis}_min"]) + eps
        keep &= points[:, column] < float(geometry[f"{axis}_max"]) - eps
    pts = points[keep]
    if not pts.size:
        return output.reshape(n_channels, y_size, x_size).transpose(1, 2, 0)

    x_index = ((pts[:, 0] - geometry["x_min"]) // geometry["x_res"]).astype(np.int32)
    y_index = ((pts[:, 1] - geometry["y_min"]) // geometry["y_res"]).astype(np.int32)
    flat = y_index * x_size + x_index
    count = np.bincount(flat, minlength=y_size * x_size).astype(np.float32)

    z_norm = np.clip(
        (pts[:, 2] - geometry["z_min"]) / (geometry["z_max"] - geometry["z_min"]),
        0.0,
        1.0,
    ).astype(np.float32)
    band = np.minimum((z_norm * n_bands).astype(np.int32), n_bands - 1)
    output[band, flat] = 1.0
    np.maximum.at(output[n_bands], flat, z_norm)
    output[n_bands + 1] = np.bincount(flat, weights=z_norm, minlength=output.shape[1]) / np.maximum(count, 1)

    intensity = np.clip(pts[:, 3] * intensity_scale, 0.0, 1.0).astype(np.float32)
    np.maximum.at(output[n_bands + 2], flat, intensity)
    output[n_bands + 3] = np.bincount(flat, weights=intensity, minlength=output.shape[1]) / np.maximum(count, 1)
    output[n_bands + 4] = np.minimum(1.0, np.log1p(count) / np.log1p(density_norm))
    return output.reshape(n_channels, y_size, x_size).transpose(1, 2, 0).astype(np.float32, copy=False)

def voxelize(points, geometry):
    x_min = geometry["x_min"]
    x_max = geometry["x_max"]
    y_min = geometry["y_min"]
    y_max = geometry["y_max"]
    z_min = geometry["z_min"]
    z_max = geometry["z_max"]
    x_res = geometry["x_res"]
    y_res = geometry["y_res"]
    z_res = geometry["z_res"]

    x_size = int((x_max - x_min) / x_res)
    y_size = int((y_max - y_min) / y_res)
    z_size = int((z_max - z_min) / z_res)

    eps = 0.001

    #clip points
    x_indexes = np.logical_and(points[:, 0] > x_min + eps, points[:, 0] < x_max - eps)
    y_indexes = np.logical_and(points[:, 1] > y_min + eps, points[:, 1] < y_max - eps)
    z_indexes = np.logical_and(points[:, 2] > z_min + eps, points[:, 2] < z_max - eps)
    pts = points[np.logical_and(np.logical_and(x_indexes, y_indexes), z_indexes)]

    occupancy_mask = np.zeros((pts.shape[0], 3), dtype = np.int32)
    voxels = np.zeros((x_size, y_size, z_size), dtype = np.float32)
    occupancy_mask[:, 0] = (pts[:, 0] - x_min) // x_res
    occupancy_mask[:, 1] = (pts[:, 1] - y_min) // y_res
    occupancy_mask[:, 2] = (pts[:, 2] - z_min) // z_res

    idxs = np.array([occupancy_mask[:, 0].reshape(-1), occupancy_mask[:, 1].reshape(-1), occupancy_mask[:, 2].reshape( -1)])

    voxels[idxs[0], idxs[1], idxs[2]] = 1
    return np.swapaxes(voxels, 0, 1)


def voxel_to_points(voxel, geometry):
    x_res = geometry["x_res"]
    y_res = geometry["y_res"]
    z_res = geometry["z_res"]

    # voxelize() returns axes in Y, X, Z order.
    ys, xs, zs = np.where(voxel.astype(bool))
    points_x = (xs + 0.5) * x_res + geometry["x_min"]
    points_y = (ys + 0.5) * y_res + geometry["y_min"]
    points_z = (zs + 0.5) * z_res + geometry["z_min"]
    #centers = np.array(np.where(voxel.astype(int) == 1)) + np.array([[x_res / 2], [y_res / 2], [z_res / 2]])
    return np.transpose(np.array([points_x, points_y, points_z]))

def trasform_label2metric(label, geometry, ratio=4):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in label map space
    :return: numpy array of shape [..., 2] of the same coordinates in metric space
    '''

    metric = np.copy(label).astype(np.float32)
    metric[..., 0] = metric[..., 0] * ratio * geometry["x_res"] + geometry["x_min"]
    metric[..., 1] = metric[..., 1] * ratio * geometry["y_res"] + geometry["y_min"]

    return metric

def transform_metric2label(metric, ratio=4, grid_size=0.1, base_height=100):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in metric space
    :return: numpy array of shape [..., 2] of the same coordinates in label_map space
    '''

    label = (metric / ratio ) / grid_size
    label[..., 1] += base_height
    return label

def get_points_in_a_rotated_box(corners, label_shape=[200, 175]):
    def minY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y0 is lowest
            return int(math.floor(y0))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # lowest point is at left edge of pixel column
            return int(math.floor(y0 + m * (x - x0)))
        else:
            # lowest point is at right edge of pixel column
            return int(math.floor(y0 + m * ((x + 1.0) - x0)))


    def maxY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y1 is highest
            return int(math.ceil(y1))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # highest point is at right edge of pixel column
            return int(math.ceil(y0 + m * ((x + 1.0) - x0)))
        else:
            # highest point is at left edge of pixel column
            return int(math.ceil(y0 + m * (x - x0)))


    # view_bl, view_tl, view_tr, view_br are the corners of the rectangle
    view = [(corners[i, 0], corners[i, 1]) for i in range(4)]

    pixels = []

    # find l,r,t,b,m1,m2
    l, m1, m2, r = sorted(view, key=lambda p: (p[0], p[1]))
    b, t = sorted([m1, m2], key=lambda p: (p[1], p[0]))

    lx, ly = l
    rx, ry = r
    bx, by = b
    tx, ty = t
    m1x, m1y = m1
    m2x, m2y = m2

    xmin = 0
    ymin = 0
    xmax = label_shape[1]
    ymax = label_shape[0]

    # inward-rounded integer bounds
    # note that we're clamping the area of interest to (xmin,ymin)-(xmax,ymax)
    lxi = max(int(math.ceil(lx)), xmin)
    rxi = min(int(math.floor(rx)), xmax)
    byi = max(int(math.ceil(by)), ymin)
    tyi = min(int(math.floor(ty)), ymax)

    x1 = lxi
    x2 = rxi

    for x in range(x1, x2):
        xf = float(x)

        if xf < m1x:
            # Phase I: left to top and bottom
            y1 = minY(lx, ly, bx, by, xf)
            y2 = maxY(lx, ly, tx, ty, xf)

        elif xf < m2x:
            if m1y < m2y:
                # Phase IIa: left/bottom --> top/right
                y1 = minY(bx, by, rx, ry, xf)
                y2 = maxY(lx, ly, tx, ty, xf)

            else:
                # Phase IIb: left/top --> bottom/right
                y1 = minY(lx, ly, bx, by, xf)
                y2 = maxY(tx, ty, rx, ry, xf)

        else:
            # Phase III: bottom/top --> right
            y1 = minY(bx, by, rx, ry, xf)
            y2 = maxY(tx, ty, rx, ry, xf)

        y1 = max(y1, byi)
        y2 = min(y2, tyi)

        for y in range(y1, y2):
            pixels.append((x, y))

    return pixels


def voxel_to_img(voxel):
    import torch

    voxel = voxel.permute(1, 2, 0)
    max_inds = torch.argmax(voxel, axis = 2)
    img = np.zeros((voxel.shape[0], voxel.shape[1]))
    for i in range(voxel.shape[0]):
        for j in range(voxel.shape[1]):
            idx = max_inds[i][j]
            img[i][j] = int(idx / voxel.shape[2] * 255)

    return img
