import numpy as np
import math


def center_to_corner_box3d(boxes_center):
    # (N, 7) -> (N, 8, 3), in Velodyne coordinates.
    ret = np.zeros((boxes_center.shape[0], 8, 3), dtype=np.float32)
    for index, box in enumerate(boxes_center):
        h, w, length, x, y, z, yaw = box
        local_x = np.array([-length / 2, -length / 2, length / 2,
                            length / 2, -length / 2, -length / 2,
                            length / 2, length / 2])
        local_y = np.array([w / 2, -w / 2, -w / 2, w / 2,
                            w / 2, -w / 2, -w / 2, w / 2])
        c, s = np.cos(yaw), np.sin(yaw)
        ret[index, :, 0] = c * local_x - s * local_y + x
        ret[index, :, 1] = s * local_x + c * local_y + y
        ret[index, :, 2] = np.array([0, 0, 0, 0, h, h, h, h]) + z
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
    # Preserve the original row-vector transform order: translation, X, Y, Z.
    transformed = np.asarray(points, dtype=np.float64).copy()
    transformed[:, 0] += tx
    transformed[:, 1] += ty
    transformed[:, 2] += tz

    if rx != 0:
        c, s = np.cos(rx), np.sin(rx)
        y, z = transformed[:, 1].copy(), transformed[:, 2].copy()
        transformed[:, 1] = c * y + s * z
        transformed[:, 2] = -s * y + c * z
    if ry != 0:
        c, s = np.cos(ry), np.sin(ry)
        x, z = transformed[:, 0].copy(), transformed[:, 2].copy()
        transformed[:, 0] = c * x - s * z
        transformed[:, 2] = s * x + c * z
    if rz != 0:
        c, s = np.cos(rz), np.sin(rz)
        x, y = transformed[:, 0].copy(), transformed[:, 1].copy()
        transformed[:, 0] = c * x + s * y
        transformed[:, 1] = -s * x + c * y
    return transformed


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
    inv_Tr[0:3, 3] = -np.sum(inv_Tr[0:3, 0:3] * Tr[None, 0:3, 3], axis=1)
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