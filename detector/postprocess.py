import torch
import numpy as np
from shapely.geometry import Polygon
import json

import torch.nn.functional as F

try:
    from torchvision.ops import nms_rotated as _torchvision_nms_rotated
except (ImportError, RuntimeError):
    _torchvision_nms_rotated = None

def convert_format(boxes_array):
    """

    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.

    return an numpy array of the positions of picks
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    polygons = convert_format(boxes)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)



def filter_pred(pred, config, out_size_factor, thres, nms_thres = None):
    geom = config["geometry"]

    cls_pred = pred["cls"].squeeze().detach()
    offset_pred = pred["offset"].squeeze().detach()
    size_pred = pred["size"].squeeze().detach()
    yaw_pred = pred["yaw"].squeeze().detach()
    log_var_pred = pred.get("log_var")
    if log_var_pred is not None:
        log_var_pred = log_var_pred.squeeze().detach()
    if not torch.stack([
        torch.isfinite(value).all()
        for value in (cls_pred, offset_pred, size_pred, yaw_pred)
        + (() if log_var_pred is None else (log_var_pred,))
    ]).all():
        raise FloatingPointError("non-finite detector output")

    if offset_pred.shape[0] not in {2, 3} or size_pred.shape[0] != offset_pred.shape[0]:
        raise ValueError("offset and size must both have 2 or 3 channels")
    is_3d = offset_pred.shape[0] == 3
    if log_var_pred is not None and (not is_3d or log_var_pred.shape[0] != 6):
        raise ValueError("log_var requires six Center3D channels")
    cos_t, sin_t = yaw_pred[0:1], yaw_pred[1:2]
    dx, dy = offset_pred[0:1], offset_pred[1:2]
    log_w, log_l = size_pred[0:1], size_pred[1:2]
    if is_3d:
        center_z = offset_pred[2].squeeze()
        h = torch.exp(size_pred[2]).squeeze()

    cls_pred = torch.sigmoid(cls_pred)
    cls_probs, cls_ids = torch.max(cls_pred, dim = 0)

    output_shape = [int((geom["y_max"] - geom["y_min"]) / geom["y_res"] / out_size_factor),
                    int((geom["x_max"] - geom["x_min"]) / geom["x_res"] / out_size_factor)]

    y = torch.arange(output_shape[0])
    x = torch.arange(output_shape[1])

    xx, yy = torch.meshgrid(x, y, indexing="xy")
    xx = xx.to(offset_pred.device)
    yy = yy.to(offset_pred.device)

    center_y = dy + yy *  geom["y_res"] * out_size_factor + geom["y_min"]
    center_x = dx + xx *  geom["x_res"] * out_size_factor + geom["x_min"]
    center_x = center_x.squeeze()
    center_y = center_y.squeeze()
    l = torch.exp(log_l).squeeze()
    w = torch.exp(log_w).squeeze()
    yaw2 = torch.atan2(sin_t, cos_t).squeeze()
    yaw = yaw2 / 2
    decoded = [center_x, center_y, l, w, yaw]
    if is_3d:
        decoded.extend([center_z, h])
    if not torch.stack([torch.isfinite(value).all() for value in decoded]).all():
        raise FloatingPointError("non-finite decoded box")

    if nms_thres is None:
        pooled = F.max_pool2d(cls_probs.unsqueeze(0), 3, 1, 1).squeeze()
        selected_idxs = torch.logical_and(cls_probs == pooled, cls_probs > thres)
        if not selected_idxs.any():
            return np.empty((0, 15 if log_var_pred is not None else (9 if is_3d else 7)), dtype=np.float32)
    else:
        pooled = F.max_pool2d(cls_probs.unsqueeze(0), 3, 1, 1).squeeze()
        candidate_mask = torch.logical_and(cls_probs == pooled, cls_probs > thres)
        if not candidate_mask.any():
            return np.empty((0, 15 if log_var_pred is not None else (9 if is_3d else 7)), dtype=np.float32)
        cos_t = torch.cos(yaw)
        sin_t = torch.sin(yaw)

        # Keep decode and candidate selection on CUDA. Only the final small
        # index vector crosses to CPU for ROS message construction.
        rear_left_x = center_x - l/2 * cos_t - w/2 * sin_t
        rear_left_y = center_y - l/2 * sin_t + w/2 * cos_t
        rear_right_x = center_x - l/2 * cos_t + w/2 * sin_t
        rear_right_y = center_y - l/2 * sin_t - w/2 * cos_t
        front_right_x = center_x + l/2 * cos_t + w/2 * sin_t
        front_right_y = center_y + l/2 * sin_t - w/2 * cos_t
        front_left_x = center_x + l/2 * cos_t - w/2 * sin_t
        front_left_y = center_y + l/2 * sin_t + w/2 * cos_t


        if (pred["cls"].is_cuda and _torchvision_nms_rotated is not None):
            candidate_boxes = torch.stack(
                [center_x[candidate_mask], center_y[candidate_mask],
                 w[candidate_mask], l[candidate_mask],
                 torch.rad2deg(yaw[candidate_mask])], dim=1
            )
            selected_idxs = _torchvision_nms_rotated(
                candidate_boxes, cls_probs[candidate_mask], nms_thres
            ).cpu().numpy()
        else:
            decoded_reg = torch.cat([
                rear_left_x.unsqueeze(0), rear_left_y.unsqueeze(0),
                rear_right_x.unsqueeze(0), rear_right_y.unsqueeze(0),
                front_right_x.unsqueeze(0), front_right_y.unsqueeze(0),
                front_left_x.unsqueeze(0), front_left_y.unsqueeze(0)], axis=0)
            decoded_reg = decoded_reg.permute(1, 2, 0)[candidate_mask]
            corners = np.reshape(decoded_reg.cpu().numpy(), (-1, 4, 2))
            selected_idxs = non_max_suppression(
                corners, cls_probs[candidate_mask].cpu().numpy(), nms_thres
            )

        cls_ids = cls_ids[candidate_mask]
        cls_probs = cls_probs[candidate_mask]
        center_x = center_x[candidate_mask]
        center_y = center_y[candidate_mask]
        l = l[candidate_mask]
        w = w[candidate_mask]
        yaw = yaw[candidate_mask]
        if is_3d:
            center_z = center_z[candidate_mask]
            h = h[candidate_mask]
        if log_var_pred is not None:
            log_var_pred = log_var_pred[:, candidate_mask]


    fields = [cls_ids[selected_idxs].cpu().numpy(),
              cls_probs[selected_idxs].cpu().numpy(),
              center_x[selected_idxs].cpu().numpy(),
              center_y[selected_idxs].cpu().numpy()]
    if is_3d:
        fields.extend([center_z[selected_idxs].cpu().numpy(),
                       w[selected_idxs].cpu().numpy(),
                       l[selected_idxs].cpu().numpy(),
                       h[selected_idxs].cpu().numpy()])
    else:
        fields.extend([l[selected_idxs].cpu().numpy(),
                       w[selected_idxs].cpu().numpy()])
    fields.append(yaw[selected_idxs].cpu().numpy())
    if log_var_pred is not None:
        fields.extend(log_var_pred[index][selected_idxs].cpu().numpy() for index in range(6))
    boxes = np.stack(fields)

    boxes = np.swapaxes(boxes, 0, 1)

    return boxes
