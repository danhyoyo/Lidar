import numpy as np
import math


def center_to_corner_box3d(boxes_center):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    for i in range(N):
        box = boxes_center[i]
        translation = box[3:6]
        size = box[0:3]
        h, w, l = size[0], size[1], size[2]
        local_x = np.array([-l / 2, -l / 2, l / 2, l / 2,
                            -l / 2, -l / 2, l / 2, l / 2])
        local_y = np.array([w / 2, -w / 2, -w / 2, w / 2,
                            w / 2, -w / 2, -w / 2, w / 2])
        local_z = np.array([0, 0, 0, 0, h, h, h, h])

        # This is the same z-axis rotation as the previous 3x3 matrix
        # multiplication.  Keeping it elementwise avoids dispatching these tiny
        # tensors to NumPy's BLAS backend, which can load a second OpenMP runtime
        # alongside PyTorch on Windows.
        cos_yaw, sin_yaw = np.cos(box[-1]), np.sin(box[-1])
        ret[i, :, 0] = local_x * cos_yaw - local_y * sin_yaw + translation[0]
        ret[i, :, 1] = local_x * sin_yaw + local_y * cos_yaw + translation[1]
        ret[i, :, 2] = local_z + translation[2]

    return ret

def corner_to_center_box3d(boxes_corner):
    # (N, 8, 3) -> (N, 7)
    ret = []
    for roi in boxes_corner:
        roi = np.array(roi)
        h = abs(np.sum(roi[:4, 2] - roi[4:, 2]) / 4)

        l = np.sum(
            np.sqrt(np.sum((roi[0, [0, 1]] - roi[3, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[1, [0, 1]] - roi[2, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[4, [0, 1]] - roi[7, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[5, [0, 1]] - roi[6, [0, 1]]) ** 2))
        ) / 4
        w = np.sum(
            np.sqrt(np.sum((roi[0, [0, 1]] - roi[1, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[2, [0, 1]] - roi[3, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[4, [0, 1]] - roi[5, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[6, [0, 1]] - roi[7, [0, 1]]) ** 2))
        ) / 4
        x = np.sum(roi[:, 0], axis=0) / 8
        y = np.sum(roi[:, 1], axis=0) / 8
        z = np.sum(roi[0:4, 2], axis=0) / 4
        rz = np.sum(
            math.atan2(roi[2, 0] - roi[1, 0], -roi[2, 1] + roi[1, 1]) +
            math.atan2(roi[6, 0] - roi[5, 0], -roi[6, 1] + roi[5, 1]) +
            math.atan2(roi[3, 0] - roi[0, 0], -roi[3, 1] + roi[0, 1]) +
            math.atan2(roi[7, 0] - roi[4, 0], -roi[7, 1] + roi[4, 1])

        ) / 4

        rz = rz - np.pi / 2
        ret.append([h, w, l, x, y, z, rz])

    return np.array(ret)

def corner_to_center_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 5)
    ret = []
    for roi in boxes_corner:
        roi = np.array(roi)

        l = np.sum(
            np.sqrt(np.sum((roi[0, [0, 1]] - roi[3, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[1, [0, 1]] - roi[2, [0, 1]]) ** 2))
        ) / 2
        w = np.sum(
            np.sqrt(np.sum((roi[0, [0, 1]] - roi[1, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[2, [0, 1]] - roi[3, [0, 1]]) ** 2))
        ) / 2
        x = np.sum(roi[:, 0], axis=0) / 4
        y = np.sum(roi[:, 1], axis=0) / 4

        rz = np.sum(
            math.atan2(roi[2, 0] - roi[1, 0], -roi[2, 1] + roi[1, 1]) +
            math.atan2(roi[3, 0] - roi[0, 0], -roi[3, 1] + roi[0, 1])
        ) / 2

        rz = rz - np.pi / 2
        ret.append([x, y, l, w, rz])

    return np.array(ret)


def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
    # Input:
    #   points: (N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (N, 3)
    # The legacy implementation expressed these as homogeneous 4x4 row-vector
    # matrix products.  These direct formulas preserve that convention exactly
    # without calling NumPy BLAS/MKL for very small matrices.
    points = np.asarray(points, dtype=np.float64).copy()
    points[:, 0] += tx
    points[:, 1] += ty
    points[:, 2] += tz

    if rx != 0:
        y, z = points[:, 1].copy(), points[:, 2].copy()
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        points[:, 1] = y * cos_x + z * sin_x
        points[:, 2] = -y * sin_x + z * cos_x

    if ry != 0:
        x, z = points[:, 0].copy(), points[:, 2].copy()
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        points[:, 0] = x * cos_y - z * sin_y
        points[:, 2] = x * sin_y + z * cos_y

    if rz != 0:
        x, y = points[:, 0].copy(), points[:, 1].copy()
        cos_z, sin_z = np.cos(rz), np.sin(rz)
        points[:, 0] = x * cos_z + y * sin_z
        points[:, 1] = -x * sin_z + y * cos_z

    return points


def box_transform(boxes, tx, ty, tz, r=0):
    # Input:
    #   boxes: (N, 7) h w l x y z rz/y
    # Output:
    #   boxes: (N, 7) h w l x y z rz/y
    boxes_corner = center_to_corner_box3d(boxes)  # (N, 8, 3)
    for idx in range(len(boxes_corner)):
        boxes_corner[idx] = point_transform(
            boxes_corner[idx], tx, ty, tz, rz=r)


    return corner_to_center_box3d(boxes_corner)


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    rotation = inv_Tr[0:3, 0:3]
    translation = Tr[0:3, 3]
    inv_Tr[0:3, 3] = -(
        rotation[:, 0] * translation[0]
        + rotation[:, 1] * translation[1]
        + rotation[:, 2] * translation[2]
    )
    return inv_Tr


class Compose(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, lidar, labels):
        if np.random.random() <= self.p:
            for t in self.transforms:
                lidar, labels = t(lidar, labels)
        return lidar, labels


class OneOf(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, lidar, labels):
        if np.random.random() <= self.p and len(self.transforms) > 0:
            choice = np.random.randint(low=0, high=len(self.transforms))
            lidar, labels = self.transforms[choice](lidar, labels)

        return lidar, labels


class Random_Rotation(object):
    def __init__(self, limit_angle=20., p=0.5):
        self.limit_angle = limit_angle / 180. * np.pi
        self.p = p

    def __call__(self, lidar, labels):
        """
        :param labels: # (N', 7) h, w, l, x, y, z,  r
        :return:
        """
        if np.random.random() <= self.p:
            angle = np.random.uniform(-self.limit_angle, self.limit_angle)
            lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
            labels = box_transform(labels, 0, 0, 0, r=angle)

        return lidar, labels


class Random_Scaling(object):
    def __init__(self, scaling_range=(0.95, 1.05), p=0.5):
        self.scaling_range = scaling_range
        self.p = p

    def __call__(self, lidar, labels):
        """
        :param labels: # (N', 7) h, w, l, x, y, z,  r
        :return:
        """
        if np.random.random() <= self.p:
            factor = np.random.uniform(self.scaling_range[0], self.scaling_range[1])
            lidar[:, 0:3] = lidar[:, 0:3] * factor
            labels[:, 0:6] = labels[:, 0:6] * factor

        return lidar, labels


class Random_Translation(object):
    def __init__(self, scale = 0.1, p = 0.5):
        self.scale = scale
        self.p = p

    def __call__(self, lidar, labels):
        """
        :param labels: # (N', 7) h, w, l, x, y, z,  r
        :return:
        """
        if np.random.random() <= self.p:
            dx = np.random.normal(loc = 0.0, scale = self.scale)
            dy = np.random.normal(loc = 0.0, scale = self.scale)
            dz = np.random.normal(loc = 0.0, scale = self.scale)

            lidar[:, 0] += dx
            lidar[:, 1] += dy
            lidar[:, 2] += dz
            labels[:, 3] +=dx
            labels[:, 4] +=dy
            labels[:, 5] +=dz

        return lidar, labels
