import torch
import math
import numpy as np
from shapely.geometry import Polygon
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
    iou = []
    for candidate in boxes:
        union = box.union(candidate).area
        iou.append(0.0 if union <= 0 else box.intersection(candidate).area / union)

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


def _empty_detections():
    return np.empty((0, 7), dtype=np.float32)


def decode_candidates(pred, config, out_size_factor, thres, *, cls_encoding="gaussian", grouped=False,
                      box_mode="bev"):
    """Decode foreground candidates without sorting or NMS.

    Explicit binary mode uses local softmax with background-winner masking.
    The flat wrapper retains its historical sigmoid behavior by default.
    """
    if box_mode not in {"bev", "3d"}:
        raise ValueError("box_mode must be bev or 3d")
    if "vertical" in pred and box_mode == "bev":
        raise ValueError("3D decoding requires box_mode=3d; BEV decoding cannot drop vertical output")
    if box_mode == "3d":
        if "vertical" not in pred:
            raise KeyError("3D prediction requires vertical")
        if not torch.is_tensor(pred["vertical"]) or pred["vertical"].shape != pred["offset"].shape:
            raise ValueError("vertical must match two-channel offset shape")
        # New 3D decode always uses FP32; historical BEV arithmetic is preserved.
        pred = {key: value.float() if torch.is_tensor(value) else value for key, value in pred.items()}
    geom = config["geometry"]

    required = {"cls", "offset", "size", "yaw"}
    missing = required.difference(pred)
    if missing:
        raise KeyError(f"Missing prediction heads: {sorted(missing)}")
    if any(pred[name].ndim != 4 or pred[name].shape[0] != 1 for name in required):
        raise ValueError("filter_pred expects prediction heads with shape [1, C, H, W]")
    spatial_shape = pred["cls"].shape[-2:]
    if out_size_factor <= 0:
        raise ValueError("out_size_factor must be positive")
    if geom["x_res"] <= 0 or geom["y_res"] <= 0:
        raise ValueError("geometry resolutions must be positive")
    if any(pred[name].shape[-2:] != spatial_shape for name in required):
        raise ValueError("All prediction heads must have the same spatial shape")
    if pred["cls"].shape[1] < 1:
        raise ValueError("cls head must contain at least one class")
    if pred["offset"].shape[1] != 2 or pred["size"].shape[1] != 2:
        raise ValueError("offset and size must both have exactly two channels")
    if pred["yaw"].shape[1] != 2:
        raise ValueError("yaw head must contain exactly two channels")

    has_iou = "iou" in pred and pred["iou"] is not None
    if has_iou:
        if pred["iou"].ndim != 4 or pred["iou"].shape[0] != 1 or pred["iou"].shape[1] != 1:
            raise ValueError("iou head must have shape [1, 1, H, W]")
        if pred["iou"].shape[-2:] != spatial_shape:
            raise ValueError("All prediction heads must have the same spatial shape")

    if not 0.0 <= thres <= 1.0:
        raise ValueError("score threshold must be between 0 and 1")

    # Remove only the known batch dimension. A plain squeeze() also removes the
    # class dimension for single-class models and makes torch.max use the wrong axis.
    cls_pred = pred["cls"].squeeze(0).detach()
    offset_pred = pred["offset"].squeeze(0).detach()
    size_pred = pred["size"].squeeze(0).detach()
    yaw_pred = pred["yaw"].squeeze(0).detach()
    tensors_to_check = [cls_pred, offset_pred, size_pred, yaw_pred]
    if has_iou:
        tensors_to_check.append(pred["iou"].squeeze(0).detach())
    if not torch.stack([
        torch.isfinite(value).all()
        for value in tensors_to_check
    ]).all():
        raise FloatingPointError("non-finite detector output")

    cos_t, sin_t = yaw_pred.unbind(dim=0)
    dx, dy = offset_pred
    log_w, log_l = size_pred

    foreground_cells = None
    if cls_encoding == "binary":
        if cls_pred.shape[0] < 2:
            raise ValueError("Binary classification requires background plus foreground channels")
        probabilities = torch.softmax(cls_pred, dim=0)
        foreground_cells = probabilities.argmax(dim=0) != 0
        cls_pred = probabilities[1:]
    elif cls_encoding == "gaussian":
        cls_pred = torch.sigmoid(cls_pred)
    else:
        raise ValueError("cls_encoding must be gaussian or binary")
    peak_mode = config.get("peak_mode", "per_class")
    if peak_mode not in {"per_class", "legacy"}:
        raise ValueError("peak_mode must be 'per_class' or 'legacy'")
    if peak_mode == "legacy":
        cls_probs, cls_ids = torch.max(cls_pred, dim=0)
    else:
        cls_probs = cls_pred

    # Pillar 4: Joint IoU-Aware Quality Scoring
    # Note:
    #   nms_alpha == 0.0: Reverts to standard / old NMS based purely on cls_probs.
    #   nms_alpha == 0.5: Calibrated joint NMS S_final = (P_cls)^(1-alpha) * (S_iou)^alpha.
    if has_iou:
        alpha = float(config.get("nms_alpha", 0.5))
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"nms_alpha must be between 0 and 1, got {alpha}")
    else:
        alpha = 0.0

    if has_iou and alpha > 0.0:
        iou_logit = pred["iou"][0, 0].detach()
        iou_score = torch.sigmoid(iou_logit)
        ranking_scores = (cls_probs ** (1.0 - alpha)) * (iou_score ** alpha)
    else:
        ranking_scores = cls_probs
    if foreground_cells is not None:
        # Background-winning cells must not emit candidates or suppress nearby
        # foreground peaks, including the alpha=1 quality-only ranking case.
        ranking_scores = torch.where(foreground_cells, ranking_scores, 0.)

    raw_grid = [(geom[f"{axis}_max"] - geom[f"{axis}_min"]) / geom[f"{axis}_res"] / out_size_factor
                for axis in ("y", "x")]
    if grouped:
        output_shape = [round(cells) for cells in raw_grid]
        if any(size < 1 or not math.isclose(cells, size, rel_tol=1e-5, abs_tol=1e-6)
               for cells, size in zip(raw_grid, output_shape)):
            raise ValueError("Grouped prediction grid must have positive integral output dimensions")
    else:
        # Historical flat decode keeps its original integer truncation.
        output_shape = [int(cells) for cells in raw_grid]

    y = torch.arange(output_shape[0])
    x = torch.arange(output_shape[1])

    xx, yy = torch.meshgrid(x, y, indexing="xy")
    if tuple(spatial_shape) != tuple(output_shape):
        raise ValueError(
            f"Prediction spatial shape {tuple(spatial_shape)} != expected {tuple(output_shape)}")
    xx = xx.to(offset_pred.device)
    yy = yy.to(offset_pred.device)

    center_y = dy + yy *  geom["y_res"] * out_size_factor + geom["y_min"]
    center_x = dx + xx *  geom["x_res"] * out_size_factor + geom["x_min"]
    l = torch.exp(log_l)
    w = torch.exp(log_w)
    yaw2 = torch.atan2(sin_t, cos_t)
    yaw = yaw2 / 2
    decoded = [center_x, center_y, l, w, yaw]
    if box_mode == "3d":
        z_bottom, log_height = pred["vertical"][0].detach().unbind(dim=0)
        height = torch.exp(log_height)
        if (height <= 0).any() or (l <= 0).any() or (w <= 0).any():
            raise FloatingPointError("3D decoded dimensions/height must be positive")
        decoded.extend([z_bottom, height])
    if not torch.stack([torch.isfinite(value).all() for value in decoded]).all():
        raise FloatingPointError("non-finite decoded box")

    # Pool each class separately before flattening candidates. Taking max over
    # classes first lets a nearby Car suppress a Pedestrian/Cyclist peak.
    if peak_mode == "legacy":
        pooled = F.max_pool2d(ranking_scores[None, None], 3, 1, 1)[0, 0]
    else:
        pooled = F.max_pool2d(ranking_scores[None], 3, 1, 1)[0]
    candidate_mask = (ranking_scores == pooled) & (ranking_scores > thres) & (cls_probs > thres)
    if not candidate_mask.any():
        return offset_pred.new_empty((0, 9 if box_mode == "3d" else 7))
    if peak_mode == "legacy":
        ys, xs = torch.nonzero(candidate_mask, as_tuple=True)
        candidate_cls_ids = cls_ids[ys, xs]
    else:
        candidate_cls_ids, ys, xs = torch.nonzero(candidate_mask, as_tuple=True)
    candidate_scores = ranking_scores[candidate_mask]
    center_x, center_y = center_x[ys, xs], center_y[ys, xs]
    l, w, yaw = l[ys, xs], w[ys, xs], yaw[ys, xs]

    if box_mode == "3d":
        return torch.stack([candidate_cls_ids, candidate_scores, center_x, center_y,
                            z_bottom[ys, xs], l, w, height[ys, xs], yaw], dim=1)
    return torch.stack([candidate_cls_ids, candidate_scores, center_x, center_y, l, w, yaw], dim=1)


def finalize_detections(candidates, nms_thres=None, *, max_detections=None, box_mode="bev"):
    """Apply the historical classwise rotated NMS and descending score order."""
    if nms_thres is not None and not 0.0 <= nms_thres <= 1.0:
        raise ValueError("NMS threshold must be between 0 and 1")
    if max_detections is not None and (type(max_detections) is not int or max_detections < 1):
        raise ValueError("max_detections must be a positive integer or None")
    if box_mode not in {"bev", "3d"}:
        raise ValueError("box_mode must be bev or 3d")
    columns = 9 if box_mode == "3d" else 7
    if candidates.ndim != 2 or candidates.shape[1] != columns:
        raise ValueError(f"Candidates must have shape [N, {columns}]")
    if not len(candidates):
        return np.empty((0, 9), dtype=np.float32) if box_mode == "3d" else _empty_detections()
    candidates = candidates.detach()
    candidate_cls_ids = candidates[:, 0].long()
    candidate_scores = candidates[:, 1]
    footprint = candidates[:, [2, 3, 5, 6, 8]] if box_mode == "3d" else candidates[:, 2:]
    center_x, center_y, l, w, yaw = footprint.unbind(dim=1)

    if nms_thres is not None:
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

        kept_by_class = []
        if candidates.is_cuda and _torchvision_nms_rotated is not None:
            candidate_boxes = torch.stack(
                [center_x, center_y, w, l, torch.rad2deg(yaw)], dim=1
            )
            for class_id in torch.unique(candidate_cls_ids):
                class_mask = candidate_cls_ids == class_id
                class_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
                local_indices = _torchvision_nms_rotated(
                    candidate_boxes[class_mask], candidate_scores[class_mask], nms_thres
                )
                kept_by_class.append(class_indices[local_indices])
            selected_idxs = torch.cat(kept_by_class)
            selected_idxs = selected_idxs[
                torch.argsort(candidate_scores[selected_idxs], descending=True)
            ].cpu().numpy()
        else:
            decoded_reg = torch.stack([
                rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                front_right_x, front_right_y, front_left_x, front_left_y], dim=1)
            corners = decoded_reg.cpu().numpy().reshape(-1, 4, 2)
            candidate_classes_np = candidate_cls_ids.cpu().numpy()
            candidate_scores_np = candidate_scores.cpu().numpy()
            for class_id in np.unique(candidate_classes_np):
                class_indices = np.flatnonzero(candidate_classes_np == class_id)
                local_indices = non_max_suppression(
                    corners[class_indices], candidate_scores_np[class_indices], nms_thres
                )
                kept_by_class.extend(class_indices[local_indices])
            selected_idxs = np.asarray(kept_by_class, dtype=np.int64)
            selected_idxs = selected_idxs[
                np.argsort(candidate_scores_np[selected_idxs])[::-1]
            ]

    else:
        selected_idxs = torch.argsort(candidate_scores, descending=True)

    return candidates[selected_idxs[:max_detections]].cpu().numpy().astype(np.float32, copy=False)


def filter_pred(pred, config, out_size_factor, thres, nms_thres=None, *,
                task_groups=None, cls_encoding="gaussian", use_iou=None, max_detections=None,
                box_mode="bev"):
    """Decode one frame; grouped branches require resolved metadata.

    The original positional flat interface and its sigmoid/peak/NMS policies
    remain available. Grouped binary branches use explicit local softmax.
    """
    if "groups" not in pred:
        if task_groups is not None:
            raise ValueError("Resolved task_groups require grouped predictions")
        return finalize_detections(decode_candidates(pred, config, out_size_factor, thres,
                                   cls_encoding=cls_encoding if box_mode == "3d" else "gaussian",
                                   box_mode=box_mode), nms_thres,
                                   max_detections=max_detections, box_mode=box_mode)
    from collections.abc import Mapping
    try:
        from core.task_groups import validate_resolved_groups, resolve_task_groups
    except ImportError:
        from detector.core.task_groups import validate_resolved_groups, resolve_task_groups
    if task_groups is None:
        raise ValueError("Grouped decode requires resolved task_groups")
    groups = validate_resolved_groups(task_groups)
    if cls_encoding not in {"gaussian", "binary"}:
        raise ValueError("cls_encoding must be gaussian or binary")
    if set(pred) != {"groups"} or not isinstance(pred["groups"], Mapping):
        raise ValueError("Grouped predictions must contain only a groups mapping")
    if set(pred["groups"]) != {group.name for group in groups}:
        raise ValueError("Prediction groups must match task_groups exactly")
    if "objects" in config:
        checked = resolve_task_groups([{"name": group.name, "classes": list(group.classes)}
                                      for group in groups], config["objects"])
        if checked != groups:
            raise ValueError("Global class mapping does not match task_groups")
    quality_presence = []
    for group in groups:
        heads = pred["groups"][group.name]
        allowed = {"cls", "offset", "size", "yaw", "iou"} | ({"vertical"} if box_mode == "3d" else set())
        if not isinstance(heads, Mapping) or set(heads) - allowed:
            raise ValueError(f"Malformed prediction heads for group {group.name}")
        if any(not torch.is_tensor(value) for key, value in heads.items()
               if key != "iou" or value is not None):
            raise ValueError(f"Prediction heads must be tensors for group {group.name}")
        if "cls" not in heads:
            raise KeyError(f"Missing cls head for group {group.name}")
        expected_width = group.num_classes + int(cls_encoding == "binary")
        if not torch.is_tensor(heads["cls"]) or heads["cls"].ndim != 4 or heads["cls"].shape[1] != expected_width:
            raise ValueError(f"{cls_encoding} cls width must match group {group.name}")
        quality_presence.append("iou" in heads and heads["iou"] is not None)
    if len(set(quality_presence)) != 1 or (use_iou is not None and quality_presence[0] != use_iou):
        raise ValueError("IQA presence must agree across groups and the resolved head setting")
    candidates = []
    for group in groups:
        decoded = decode_candidates(pred["groups"][group.name], config, out_size_factor, thres,
                                    cls_encoding=cls_encoding, grouped=True, box_mode=box_mode)
        if len(decoded):
            ids = torch.tensor(group.global_ids, device=decoded.device)
            decoded[:, 0] = ids[decoded[:, 0].long()]
        candidates.append(decoded)
    return finalize_detections(torch.cat(candidates, dim=0), nms_thres,
                               max_detections=max_detections, box_mode=box_mode)


def filter_pred_3d(pred, config, out_size_factor, thres, nms_thres=None, **kwargs):
    """Return float32 [class,score,x,y,z_bottom,length,width,height,yaw] rows.

    NMS remains classwise BEV footprint NMS, retaining the complete selected row.
    IQA still ranks BEV quality; it is not a new 3D IoU score.
    """
    return filter_pred(pred, config, out_size_factor, thres, nms_thres, box_mode="3d", **kwargs)
