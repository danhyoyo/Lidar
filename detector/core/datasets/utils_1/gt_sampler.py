import pickle
from typing import Dict, Optional, Tuple
import numpy as np
from core.datasets.utils_1.collision_ground import (
    check_box_collision_2d,
    check_ground_support,
    check_static_obstacle_collision,
    snap_box_to_ground,
)
from core.datasets.utils_1.physics_aug import (
    check_line_of_sight_occlusion,
    distance_adaptive_subsample,
    mask_shadow_points,
    radiometric_intensity_calibrate,
)


class GTSampler:

    def __init__(
        self,
        database_path: str,
        sample_counts: Optional[Dict[str, int]] = None,
        p: float = 1.0,
        enable_physics: bool = True,
    ):
        self.p = p
        self.sample_counts = sample_counts or {
            "Pedestrian": 6,
            "Cyclist": 5,
            "Car": 3,
        }
        self.enable_physics = enable_physics
        with open(database_path, "rb") as f:
            self.database = pickle.load(f)

    def __call__(
        self,
        lidar: np.ndarray,
        boxes: np.ndarray,
        return_metadata: bool = False,
    ):
        meta = {
            "inserted_boxes": [],
            "inserted_points": [],
            "inserted_class_names": [],
            "num_original_points": len(lidar),
            "num_original_boxes": len(boxes),
        }
        if np.random.random() > self.p or len(self.database) == 0:
            if return_metadata:
                return lidar, boxes, meta
            return lidar, boxes

        new_boxes_list = [b.copy() for b in boxes]
        cur_points = lidar.copy()
        is_8col = boxes.shape[1] >= 8 if len(boxes) > 0 else True

        # Check if the scene contains spatial background points
        has_spatial_points = len(cur_points) > 0 and (
            float(np.ptp(cur_points[:, 0])) > 2.0
            or float(np.ptp(cur_points[:, 1])) > 2.0
        )

        for cls_name, count in self.sample_counts.items():
            if (
                cls_name not in self.database
                or len(self.database[cls_name]) == 0
            ):
                continue

            candidates = self.database[cls_name]
            chosen_indices = np.random.choice(
                len(candidates),
                size=min(count * 3, len(candidates)),
                replace=False,
            )

            inserted = 0
            for idx in chosen_indices:
                if inserted >= count:
                    break
                sample = candidates[idx]
                r_orig = sample["r_origin"]

                placed = False
                for attempt in range(20):
                    # Tries 0-9: Corridor sampling (|Y| <= 14.0m)
                    # Tries 10-19: Fallback to full FOV cone
                    if attempt < 14:
                        # Depth X: spread evenly along road depth (8m to 52m) to avoid clumping near Ego
                        t_x = np.random.uniform(8.0, 52.0)
                        # Lateral Y: spread across driving lanes, bike paths, and sidewalks
                        if cls_name == "Pedestrian":
                            t_y = np.random.uniform(-15.0, 15.0)
                        elif cls_name == "Cyclist":
                            t_y = np.random.uniform(-13.0, 13.0)
                        else:  # Car
                            t_y = np.random.uniform(-9.5, 9.5)
                    else:
                        t_x = np.random.uniform(6.0, 60.0)
                        t_y = np.random.uniform(-25.0, 25.0)

                    cand_box = sample["box"].copy()
                    if not is_8col and len(cand_box) >= 8:
                        cand_box = cand_box[1:]  # match 7-col format if needed

                    x_idx = 4 if len(cand_box) >= 8 else 3
                    y_idx = 5 if len(cand_box) >= 8 else 4
                    z_idx = 6 if len(cand_box) >= 8 else 5
                    yaw_idx = 7 if len(cand_box) >= 8 else 6

                    if not (2.0 <= t_x <= 65.0 and -35.0 <= t_y <= 35.0):
                        continue

                    cand_box[x_idx] = t_x
                    cand_box[y_idx] = t_y
                    cand_box[yaw_idx] = np.random.uniform(-np.pi, np.pi)

                    # 1. Box-to-box collision check with realistic traffic safety margin
                    # (1.0m for cars, 0.8m for pedestrians & cyclists to avoid clumping)
                    box_margin = 1.0 if cls_name == "Car" else 0.8
                    existing_arr = (
                        np.array(new_boxes_list)
                        if len(new_boxes_list) > 0
                        else np.zeros((0, len(cand_box)))
                    )
                    if check_box_collision_2d(
                        cand_box, existing_arr, min_margin=box_margin
                    ):
                        continue

                    if self.enable_physics:
                        # 2. Ground surface support verification (range-adaptive ground check)
                        if has_spatial_points:
                            has_support, ground_z = check_ground_support(
                                cand_box,
                                cur_points,
                            )
                            if not has_support:
                                continue
                            cand_box[z_idx] = ground_z
                        else:
                            cand_box = snap_box_to_ground(cand_box, cur_points)

                        # 3. Static obstacle collision check (walls, trees, poles)
                        if check_static_obstacle_collision(
                            cand_box, cur_points, max_obstacle_points=2
                        ):
                            continue

                        # 4. Foreground line-of-sight occlusion check (behind buildings/walls)
                        if check_line_of_sight_occlusion(
                            cand_box, cur_points
                        ):
                            continue
                    else:
                        cand_box = snap_box_to_ground(cand_box, cur_points)

                    placed = True
                    placed_r_target = float(np.sqrt(t_x**2 + t_y**2))
                    break

                if not placed:
                    continue

                obj_pts = sample["points"].copy()
                if self.enable_physics:
                    obj_pts = distance_adaptive_subsample(
                        obj_pts, cand_box, r_orig, placed_r_target
                    )
                    obj_pts = radiometric_intensity_calibrate(
                        obj_pts, r_orig, placed_r_target
                    )

                # Transform canonical points to target world coordinates
                yaw = cand_box[yaw_idx]
                bx = cand_box[x_idx]
                by = cand_box[y_idx]
                bz = cand_box[z_idx]

                cos_y = np.cos(yaw)
                sin_y = np.sin(yaw)
                x_world = obj_pts[:, 0] * cos_y - obj_pts[:, 1] * sin_y + bx
                y_world = obj_pts[:, 0] * sin_y + obj_pts[:, 1] * cos_y + by
                z_world = obj_pts[:, 2] + bz

                world_pts = np.hstack(
                    [
                        np.stack([x_world, y_world, z_world], axis=1),
                        (
                            obj_pts[:, 3:]
                            if obj_pts.shape[1] > 3
                            else np.ones((len(obj_pts), 1), dtype=np.float32)
                        ),
                    ]
                )

                if self.enable_physics and len(cur_points) > 0:
                    # 1. Remove background points inside newly placed box volume
                    pts_trans = cur_points[:, :3] - np.array([bx, by, bz])
                    cos_y_inv = np.cos(-yaw)
                    sin_y_inv = np.sin(-yaw)
                    x_rot = (
                        pts_trans[:, 0] * cos_y_inv
                        - pts_trans[:, 1] * sin_y_inv
                    )
                    y_rot = (
                        pts_trans[:, 0] * sin_y_inv
                        + pts_trans[:, 1] * cos_y_inv
                    )
                    z_rot = pts_trans[:, 2]

                    b_dim = (
                        cand_box[1:4]
                        if len(cand_box) >= 8
                        else cand_box[:3]
                    )
                    h_val, w_val, l_val = b_dim[0], b_dim[1], b_dim[2]
                    inside_box = (
                        (np.abs(x_rot) <= l_val / 2.0)
                        & (np.abs(y_rot) <= w_val / 2.0)
                        & (z_rot >= 0.0)
                        & (z_rot <= h_val)
                    )
                    cur_points = cur_points[~inside_box]

                    # 2. Mask line-of-sight shadow points behind box
                    cur_points = mask_shadow_points(cur_points, cand_box)

                cur_points = np.vstack([cur_points, world_pts])
                new_boxes_list.append(cand_box)
                inserted += 1

                if return_metadata:
                    meta["inserted_boxes"].append(cand_box.copy())
                    meta["inserted_points"].append(world_pts.copy())
                    meta["inserted_class_names"].append(cls_name)

        target_cols = 8 if is_8col else 7
        final_boxes = (
            np.array(new_boxes_list, dtype=np.float32)
            if len(new_boxes_list) > 0
            else np.zeros((0, target_cols), dtype=np.float32)
        )
        if return_metadata:
            meta["final_boxes"] = final_boxes
            meta["num_inserted"] = len(meta["inserted_boxes"])
            return cur_points.astype(np.float32), final_boxes, meta
        return cur_points.astype(np.float32), final_boxes
