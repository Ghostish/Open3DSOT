# Created by zenn at 2021/4/27

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
from collections import defaultdict
from datasets import points_utils


class kittiDataset():
    def __init__(self, path, split, category_name="Car", **kwargs):
        self.KITTI_Folder = path
        self.category_name = category_name
        self.split = split
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_image = os.path.join(self.KITTI_Folder, "image_02")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")
        self.KITTI_calib = os.path.join(self.KITTI_Folder, "calib")
        self.scene_list = self._build_scene_list(split)
        self.velos = defaultdict(dict)
        self.calibs = {}
        self.tracklet_anno_list, self.tracklet_len_list = self._build_tracklet_anno()
        self.coordinate_mode = kwargs.get('coordinate_mode', 'velodyne')
        self.preload_offset = kwargs.get('preload_offset', -1)
        self.training_samples = self._load_data()

    @staticmethod
    def _build_scene_list(split):
        if "TRAIN" in split.upper():  # Training SET
            if "TINY" in split.upper():
                scene_names = [0]
            else:
                scene_names = list(range(0, 17))
        elif "VALID" in split.upper():  # Validation Set
            if "TINY" in split.upper():
                scene_names = [18]
            else:
                scene_names = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set
            if "TINY" in split.upper():
                scene_names = [19]
            else:
                scene_names = list(range(19, 21))

        else:  # Full Dataset
            scene_names = list(range(21))
        scene_names = ['%04d' % scene_name for scene_name in scene_names]
        return scene_names

    def _load_data(self):
        print('preloading data into memory')
        preload_data_path = os.path.join(self.KITTI_Folder,
                                         f"preload_kitti_{self.category_name}_{self.split}_{self.coordinate_mode}_{self.preload_offset}.dat")
        if os.path.isfile(preload_data_path):
            print(f'loading from saved file {preload_data_path}.')
            with open(preload_data_path, 'rb') as f:
                training_samples = pickle.load(f)
        else:
            print('reading from annos')
            training_samples = []
            for i in range(len(self.tracklet_anno_list)):
                frames = []
                for anno in self.tracklet_anno_list[i]:
                    frames.append(self._get_frame_from_anno(anno))
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

        list_of_tracklet_anno = []
        list_of_tracklet_len = []
        for scene in self.scene_list:

            label_file = os.path.join(self.KITTI_label, scene + ".txt")

            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            if self.category_name in ['Car', 'Van', 'Truck',
                                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                      'Misc']:
                df = df[df["type"] == self.category_name]
            elif self.category_name == 'All':
                df = df[(df["type"] == 'Car') |
                        (df["type"] == 'Van') |
                        (df["type"] == 'Pedestrian') |
                        (df["type"] == 'Cyclist')]
            else:
                df = df[df["type"] != 'DontCare']
            df.insert(loc=0, column="scene", value=scene)
            for track_id in df.track_id.unique():
                df_tracklet = df[df["track_id"] == track_id]
                df_tracklet = df_tracklet.sort_values(by=['frame'])
                df_tracklet = df_tracklet.reset_index(drop=True)
                tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                list_of_tracklet_anno.append(tracklet_anno)
                list_of_tracklet_len.append((len(tracklet_anno)))

        return list_of_tracklet_anno, list_of_tracklet_len

    def get_frames(self, seq_id, frame_ids):
        frames = [self.training_samples[seq_id][f_id] for f_id in frame_ids]
        return frames

    def _get_frame_from_anno(self, anno):
        scene_id = anno['scene']
        frame_id = anno['frame']
        try:
            calib = self.calibs[scene_id]
        except KeyError:
            calib_path = os.path.join(self.KITTI_calib, scene_id + ".txt")
            calib = self._read_calib_file(calib_path)
            self.calibs[scene_id] = calib
        velo_to_cam = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))

        if self.coordinate_mode == 'velodyne':
            box_center_cam = np.array([anno["x"], anno["y"] - anno["height"] / 2, anno["z"], 1])
            # transform bb from camera coordinate into velo coordinates
            box_center_velo = np.dot(np.linalg.inv(velo_to_cam), box_center_cam)
            box_center_velo = box_center_velo[:3]
            size = [anno["width"], anno["length"], anno["height"]]
            orientation = Quaternion(
                axis=[0, 0, -1], radians=anno["rotation_y"]) * Quaternion(axis=[0, 0, -1], degrees=90)
            bb = Box(box_center_velo, size, orientation)
        else:
            center = [anno["x"], anno["y"] - anno["height"] / 2, anno["z"]]
            size = [anno["width"], anno["length"], anno["height"]]
            orientation = Quaternion(
                axis=[0, 1, 0], radians=anno["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
            bb = Box(center, size, orientation)

        try:
            try:
                pc = self.velos[scene_id][frame_id]
            except KeyError:
                # VELODYNE PointCloud
                velodyne_path = os.path.join(self.KITTI_velo, scene_id,
                                             '{:06}.bin'.format(frame_id))

                pc = PointCloud(
                    np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
                if self.coordinate_mode == "camera":
                    pc.transform(velo_to_cam)
                self.velos[scene_id][frame_id] = pc
            if self.preload_offset > 0:
                pc = points_utils.crop_pc_axis_aligned(pc, bb, offset=self.preload_offset)
        except:
            # in case the Point cloud is missing
            # (0001/[000177-000180].bin)
            # msg = f"The point cloud at scene {scene_id} frame {frame_id} is missing."
            # warnings.warn(msg)
            pc = PointCloud(np.array([[0, 0, 0]]).T)
        # todo add image
        return {"pc": pc, "3d_bbox": bb, 'meta': anno}

    @staticmethod
    def _read_calib_file(filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data
