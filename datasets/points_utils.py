# Created by zenn at 2021/5/6
import torch
import os
import copy
import numpy as np
from pyquaternion import Quaternion
from datasets.data_classes import PointCloud
from scipy.spatial.distance import cdist


def random_choice(num_samples, size, replacement=False, seed=None):
    if seed is not None:
        generator = torch.random.manual_seed(seed)
    else:
        generator = None
    return torch.multinomial(
        torch.ones((size), dtype=torch.float32),
        num_samples=num_samples,
        replacement=replacement,
        generator=generator
    )


def regularize_pc(points, sample_size, seed=None):
    # random sampling from points
    num_points = points.shape[0]
    new_pts_idx = None
    rng = np.random if seed is None else np.random.default_rng(seed)
    if num_points > 2:
        if num_points != sample_size:
            new_pts_idx = rng.choice(num_points, size=sample_size, replace=sample_size > num_points)
            # new_pts_idx = random_choice(num_samples=sample_size, size=num_points,
            #                             replacement=sample_size > num_points, seed=seed).numpy()
        else:
            new_pts_idx = np.arange(num_points)
    if new_pts_idx is not None:
        points = points[new_pts_idx, :]
    else:
        points = np.zeros((sample_size, 3), dtype='float32')
    return points, new_pts_idx


def getOffsetBB(box, offset, degrees=True, use_z=False, limit_box=True):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)
    if len(offset) == 3:
        use_z = False
    # REMOVE TRANSfORM
    if degrees:
        if len(offset) == 3:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], degrees=offset[2]))
        elif len(offset) == 4:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], degrees=offset[3]))
    else:
        if len(offset) == 3:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], radians=offset[2]))
        elif len(offset) == 4:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], radians=offset[3]))
    if limit_box:
        if offset[0] > new_box.wlh[0]:
            offset[0] = np.random.uniform(-1, 1)
        if offset[1] > min(new_box.wlh[1], 2):
            offset[1] = np.random.uniform(-1, 1)
        if use_z and offset[2] > new_box.wlh[2]:
            offset[2] = 0
    if use_z:
        new_box.translate(np.array([offset[0], offset[1], offset[2]]))
    else:
        new_box.translate(np.array([offset[0], offset[1], 0]))

    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box


def getModel(PCs, boxes, offset=0, scale=1.0, normalize=False):
    """center and merge the object pcs in boxes"""
    if len(PCs) == 0:
        return PointCloud(np.ones((3, 0)))
    points = [np.ones((PCs[0].points.shape[0], 0), dtype='float32')]
    for PC, box in zip(PCs, boxes):
        cropped_PC, new_box = cropAndCenterPC(PC, box, offset=offset, scale=scale, normalize=normalize)
        # try:
        if cropped_PC.nbr_points() > 0:
            points.append(cropped_PC.points)

    PC = PointCloud(np.concatenate(points, axis=1))
    return PC, new_box


def cropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):
    """
    crop and center the pc using the given box
    """
    new_PC = crop_pc_axis_aligned(PC, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    new_PC = crop_pc_axis_aligned(new_PC, new_box, offset=offset, scale=scale)

    if normalize:
        new_PC.normalize(box.wlh)
    return new_PC, new_box


def get_point_to_box_distance(pc, box):
    """
    generate the BoxCloud for the given pc and box
    :param pc: Pointcloud object or numpy array
    :param box:
    :return:
    """
    if isinstance(pc, PointCloud):
        points = pc.points.T  # N,3
    else:
        points = pc  # N,3
        assert points.shape[1] == 3
    box_corners = box.corners()  # 3,8
    box_centers = box.center.reshape(-1, 1)  # 3,1
    box_points = np.concatenate([box_centers, box_corners], axis=1)  # 3,9
    points2cc_dist = cdist(points, box_points.T)  # N,9
    return points2cc_dist


def crop_pc_axis_aligned(PC, box, offset=0, scale=1.0, return_mask=False):
    """
    crop the pc using the box in the axis-aligned manner
    """
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    if return_mask:
        return new_PC, close
    return new_PC


def crop_pc_oriented(PC, box, offset=0, scale=1.0, return_mask=False):
    """
    crop the pc using the exact box.
    slower than 'crop_pc_axis_aligned' but more accurate
    """

    box_tmp = copy.deepcopy(box)
    new_PC = PointCloud(PC.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_PC.translate(trans)
    box_tmp.translate(trans)
    new_PC.rotate(rot_mat)
    box_tmp.rotate(Quaternion(matrix=rot_mat))

    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = new_PC.points[0, :] < maxi[0]
    x_filt_min = new_PC.points[0, :] > mini[0]
    y_filt_max = new_PC.points[1, :] < maxi[1]
    y_filt_min = new_PC.points[1, :] > mini[1]
    z_filt_max = new_PC.points[2, :] < maxi[2]
    z_filt_min = new_PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(new_PC.points[:, close])

    # transform back to the original coordinate system
    new_PC.rotate(np.transpose(rot_mat))
    new_PC.translate(-trans)
    if return_mask:
        return new_PC, close
    return new_PC


def generate_subwindow(pc, sample_bb, scale, offset=2, oriented=True):
    """
    generating the search area using the sample_bb

    :param pc:
    :param sample_bb:
    :param scale:
    :param offset:
    :param oriented: use oriented or axis-aligned cropping
    :return:
    """
    rot_mat = np.transpose(sample_bb.rotation_matrix)
    trans = -sample_bb.center
    if oriented:
        new_pc = PointCloud(pc.points.copy())
        box_tmp = copy.deepcopy(sample_bb)

        # transform to the coordinate system of sample_bb
        new_pc.translate(trans)
        box_tmp.translate(trans)
        new_pc.rotate(rot_mat)
        box_tmp.rotate(Quaternion(matrix=rot_mat))
        new_pc = crop_pc_axis_aligned(new_pc, box_tmp, scale=scale, offset=offset)


    else:
        new_pc = crop_pc_axis_aligned(pc, sample_bb, scale=scale, offset=offset)

        # transform to the coordinate system of sample_bb
        new_pc.translate(trans)
        new_pc.rotate(rot_mat)

    return new_pc


def transform_box(box, ref_box):
    box = copy.deepcopy(box)
    box.translate(-ref_box.center)
    box.rotate(Quaternion(matrix=ref_box.rotation_matrix.T))
    return box


def get_in_box_mask(PC, box):
    """check which points of PC are inside the box"""
    box_tmp = copy.deepcopy(box)
    new_PC = PointCloud(PC.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_PC.translate(trans)
    box_tmp.translate(trans)
    new_PC.rotate(rot_mat)
    box_tmp.rotate(Quaternion(matrix=rot_mat))
    maxi = np.max(box_tmp.corners(), 1)
    mini = np.min(box_tmp.corners(), 1)

    x_filt_max = new_PC.points[0, :] < maxi[0]
    x_filt_min = new_PC.points[0, :] > mini[0]
    y_filt_max = new_PC.points[1, :] < maxi[1]
    y_filt_min = new_PC.points[1, :] > mini[1]
    z_filt_max = new_PC.points[2, :] < maxi[2]
    z_filt_min = new_PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)
    return close

