# Created by Xu Yan at 2021/10/17

import copy
import random

from torch.utils.data import Dataset
from datasets.data_classes import PointCloud, Box
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import os
import warnings
import pickle
from functools import reduce
from tqdm import tqdm
from datasets.generate_waymo_sot import generate_waymo_data
from collections import defaultdict
from datasets import points_utils, base_dataset


class WaymoDataset(base_dataset.BaseDataset):
    def __init__(self, path, split, category_name="VEHICLE", **kwargs):
        super().__init__(path, split, category_name, **kwargs)
        self.Waymo_Folder = path
        self.category_name = category_name
        self.Waymo_velo = os.path.join(self.Waymo_Folder, split, "velodyne")
        self.Waymo_label = os.path.join(self.Waymo_Folder, split, "label_02")
        self.Waymo_calib = os.path.join(self.Waymo_Folder, split, "calib")
        self.velos = defaultdict(dict)
        self.calibs = {}

        self.split = self.split.lower()
        self.category_name = self.category_name.lower()
        self.split = 'val' if self.split == 'test' else self.split
        assert self.split in ['train', 'val']
        assert self.category_name in ['vehicle', 'pedestrian', 'cyclist']

        self.tiny = kwargs.get('tiny', False)
        self.tracklet_anno_list, self.tracklet_len_list = self._build_tracklet_anno()
        if self.tiny:
            self.tracklet_anno_list = self.tracklet_anno_list[:100]
            self.tracklet_len_list = self.tracklet_len_list[:100]

        self.preload_offset = kwargs.get('preload_offset', 10)
        if self.preloading:
            self.training_samples = self._load_data()

    def _load_data(self):
        print('preloading data into memory')
        if self.tiny:
            preload_data_path = os.path.join(self.Waymo_Folder,
                                             f"preload_{self.split}_{self.category_name}_{self.preload_offset}_tiny.dat")
        else:
            preload_data_path = os.path.join(self.Waymo_Folder,
                                             f"preload_{self.split}_{self.category_name}_{self.preload_offset}.dat")

        print(preload_data_path)

        if os.path.isfile(preload_data_path):
            print(f'loading from saved file {preload_data_path}.')
            with open(preload_data_path, 'rb') as f:
                training_samples = pickle.load(f)
        else:
            print('reading from annos')
            training_samples = []
            for i in tqdm(range(len(self.tracklet_anno_list)), total=len(self.tracklet_anno_list)):
                frames = []
                for anno in self.tracklet_anno_list[i]:
                    frames.append(self._get_frame_from_anno(anno, i))

                training_samples.append(frames)
            with open(preload_data_path, 'wb') as f:
                print(f'saving loaded data to {preload_data_path}')
                pickle.dump(training_samples, f)
        return training_samples

    def get_num_scenes(self):
        return len(self.scene_list)

    def get_num_tracklets(self):
        return len(self.tracklet_anno_list)

    def get_num_frames_total(self):
        return sum(self.tracklet_len_list)

    def get_num_frames_tracklet(self, tracklet_id):
        return self.tracklet_len_list[tracklet_id]

    def _build_tracklet_anno(self):
        preload_data_path = os.path.join(self.Waymo_Folder,
                                         f"sot_infos_{self.category_name.lower()}_{self.split.lower()}.pkl")
        if not os.path.exists(preload_data_path):
            print('Prepare %s' % preload_data_path)
            generate_waymo_data(self.Waymo_Folder, self.category_name, self.split)

        with open(preload_data_path, 'rb') as f:
            infos = pickle.load(f)

        list_of_tracklet_anno = []
        list_of_tracklet_len = []

        for scene in list(infos.keys()):
            anno = infos[scene]
            list_of_tracklet_anno.append(anno)
            list_of_tracklet_len.append(len(anno))

        return list_of_tracklet_anno, list_of_tracklet_len

    def get_frames(self, seq_id, frame_ids):
        if self.preloading:
            frames = [self.training_samples[seq_id][f_id] for f_id in frame_ids]
        else:
            seq_annos = self.tracklet_anno_list[seq_id]
            frames = [self._get_frame_from_anno(seq_annos[f_id]) for f_id in frame_ids]

        return frames

    def _get_frame_from_anno(self, anno, track_id=None, check=False):
        '''
        'box': np.array([box.center_x, box.center_y, box.center_z,
                         box.length, box.width, box.height, ref_velocity[0],
                         ref_velocity[1], box.heading], dtype=np.float32),
        '''
        sample_data_lidar = anno['PC']
        gt_boxes = anno['Box']

        with open(sample_data_lidar, 'rb') as f:
            pc_info = pickle.load(f)

        pointcloud = pc_info['lidars']['points_xyz'].transpose((1, 0))

        with open(sample_data_lidar.replace('lidar', 'annos'), 'rb') as f:
            ref_obj = pickle.load(f)

        ref_pose = np.reshape(ref_obj['veh_to_global'], [4, 4])
        global_from_car, _ = self.veh_pos_to_transform(ref_pose)
        nbr_points = pointcloud.shape[1]
        pointcloud[:3, :] = global_from_car.dot(
            np.vstack((pointcloud[:3, :], np.ones(nbr_points)))
        )[:3, :]

        # transform from Waymo to KITTI coordinate
        # Waymo: x, y, z, length, width, height, rotation from positive x axis clockwisely
        # KITTI: x, y, z, width, length, height, rotation from negative y axis counterclockwisely
        gt_boxes[[3, 4]] = gt_boxes[[4, 3]]

        pc = PointCloud(pointcloud)
        bb = Box(gt_boxes[0:3], gt_boxes[3:6], Quaternion(axis=[0, 0, 1], radians=-gt_boxes[-1]),
                 velocity=gt_boxes[6:9], name=anno['Class'])
        bb.rotate(Quaternion(matrix=global_from_car))
        bb.translate(global_from_car[:3, -1])
        if self.preload_offset > 0:
            pc = points_utils.crop_pc_axis_aligned(pc, bb, offset=self.preload_offset)

        if check:
            from datasets.utils import write_bbox, write_obj, get_3d_box, box2obj
            print('check', pc_info['frame_id'])
            path = 'visual_%s_track%d/' % (pc_info['scene_name'], track_id)
            os.makedirs(path, exist_ok=True)
            if pc_info['frame_id'] % 50 == 0:
                write_obj(pc.points.transpose((1, 0)), path + 'frames_%d' % pc_info['frame_id'])
                # write_bbox(get_3d_box(bb.wlh, bb.orientation.radians * bb.orientation.axis[-1], bb.center), 0, path + 'box_%d.ply' % pc_info['frame_id'])
                box2obj(bb, path + 'box_%d.obj' % pc_info['frame_id'])
            print(path + 'box_%d.obj' % pc_info['frame_id'])
            # exit()

        return {"pc": pc, "3d_bbox": bb, 'meta': anno}

    @staticmethod
    def veh_pos_to_transform(veh_pos):
        def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                             rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                             inverse: bool = False) -> np.ndarray:
            """
            Convert pose to transformation matrix.
            :param translation: <np.float32: 3>. Translation in x, y, z.
            :param rotation: Rotation in quaternions (w ri rj rk).
            :param inverse: Whether to compute inverse transform matrix.
            :return: <np.float32: 4, 4>. Transformation matrix.
            """
            tm = np.eye(4)

            if inverse:
                rot_inv = rotation.rotation_matrix.T
                trans = np.transpose(-np.array(translation))
                tm[:3, :3] = rot_inv
                tm[:3, 3] = rot_inv.dot(trans)
            else:
                tm[:3, :3] = rotation.rotation_matrix
                tm[:3, 3] = np.transpose(np.array(translation))

            return tm

        "convert vehicle pose to two transformation matrix"
        rotation = veh_pos[:3, :3]
        tran = veh_pos[:3, 3]

        global_from_car = transform_matrix(
            tran, Quaternion(matrix=rotation), inverse=False
        )

        car_from_global = transform_matrix(
            tran, Quaternion(matrix=rotation), inverse=True
        )

        return global_from_car, car_from_global


if __name__ == '__main__':
    WaymoDataset('/raid/databases/Waymo/', 'train', tiny=True)
    # WaymoDataset('/raid/databases/Waymo/', 'val')
