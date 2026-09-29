import pickle
from typing import Dict, Optional, Tuple
import numpy as np
from core.datasets.utils_1.collision_ground import (
    check_box_collision_2d,
    snap_box_to_ground,
)
from core.datasets.utils_1.physics_aug import (
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
            "Car": 8,
            "Pedestrian": 6,
            "Cyclist": 6,
        }
        self.enable_physics = enable_physics
        with open(database_path, "rb") as f:
            self.database = pickle.load(f)

    def __call__(
        self, lidar: np.ndarray, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > self.p or len(self.database) == 0:
            return lidar, boxes

        new_boxes_list = [b.copy() for b in boxes]
        cur_points = lidar.copy()
        is_8col = boxes.shape[1] >= 8 if len(boxes) > 0 else True

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
                for _ in range(10):
                    # Pick valid target location in sensor FOV (5m to 65m, azimuth [-45 deg, +45 deg])
                    r_target = np.clip(
                        r_orig + np.random.uniform(-5.0, 5.0), 5.0, 65.0
                    )
                    azimuth_target = np.random.uniform(-np.pi / 4.0, np.pi / 4.0)

                    cand_box = sample["box"].copy()
                    if not is_8col and len(cand_box) >= 8:
                        cand_box = cand_box[1:]  # match 7-col format if needed

                    x_idx = 4 if len(cand_box) >= 8 else 3
                    y_idx = 5 if len(cand_box) >= 8 else 4
                    z_idx = 6 if len(cand_box) >= 8 else 5
                    yaw_idx = 7 if len(cand_box) >= 8 else 6

                    cand_box[x_idx] = r_target * np.cos(azimuth_target)
                    cand_box[y_idx] = r_target * np.sin(azimuth_target)
                    cand_box[yaw_idx] = np.random.uniform(-np.pi, np.pi)

                    existing_arr = (
                        np.array(new_boxes_list)
                        if len(new_boxes_list) > 0
                        else np.zeros((0, len(cand_box)))
                    )
                    if not check_box_collision_2d(
                        cand_box, existing_arr, min_margin=0.3
                    ):
                        placed = True
                        break

                if not placed:
                    continue

                cand_box = snap_box_to_ground(cand_box, cur_points)

                obj_pts = sample["points"].copy()
                if self.enable_physics:
                    obj_pts = distance_adaptive_subsample(
                        obj_pts, cand_box, r_orig, r_target
                    )
                    obj_pts = radiometric_intensity_calibrate(
                        obj_pts, r_orig, r_target
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

                if self.enable_physics:
                    cur_points = mask_shadow_points(cur_points, cand_box)

                cur_points = np.vstack([cur_points, world_pts])
                new_boxes_list.append(cand_box)
                inserted += 1

        target_cols = 8 if is_8col else 7
        final_boxes = (
            np.array(new_boxes_list, dtype=np.float32)
            if len(new_boxes_list) > 0
            else np.zeros((0, target_cols), dtype=np.float32)
        )
        return cur_points.astype(np.float32), final_boxes
