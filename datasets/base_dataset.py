""" 
base_dataset.py
Created by zenn at 2021/9/1 22:16
"""


class BaseDataset:
    def __init__(self, path, split, category_name="Car", **kwargs):
        self.path = path
        self.split = split
        self.category_name = category_name

    def get_num_tracklets(self):
        raise NotImplementedError

    def get_num_frames_total(self):
        raise NotImplementedError

    def get_num_frames_tracklet(self, tracklet_id):
        raise NotImplementedError

    def get_frames(self, seq_id, frame_ids):
        raise NotImplementedError
