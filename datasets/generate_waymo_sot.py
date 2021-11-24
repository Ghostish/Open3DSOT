#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: generate_waymo_sot.py
@time: 2021/6/17 13:17
'''
import os
import pickle
from collections import defaultdict

from tqdm import tqdm


def lood_pickle(root):
    with open(root, "rb") as f:
        file = pickle.load(f)
    return file


def generate_waymo_data(root, cla, split):
    TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

    print('Generate %s class for %s set' % (cla, split))
    waymo_infos_all = lood_pickle(os.path.join(root, 'infos_%s_01sweeps_filter_zero_gt.pkl' % split))

    DATA = defaultdict(list)

    for idx, frame in tqdm(enumerate(waymo_infos_all), total=len(waymo_infos_all)):
        anno = lood_pickle(os.path.join(root, frame['anno_path']))
        
        for obj in anno['objects']:
            if TYPE_LIST[obj['label']] == cla:
                if not obj['name'] in DATA:
                    DATA[obj['name']] = [
                        {
                            'PC': frame['path'],
                            'Box': obj['box'],
                            'Class': cla
                        }
                    ]
                else:
                    DATA[obj['name']] += [
                        {
                            'PC': frame['path'],
                            'Box': obj['box'],
                            'Class': cla
                        }
                    ]

    print('Save data...')
    with open(os.path.join(root, 'sot_infos_%s_%s.pkl' % (cla.lower(), split)), "wb") as f:
        pickle.dump(DATA, f)


if __name__ == '__main__':
    splits = ['train', 'val']
    classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
    root = '/raid/databases/Waymo/'
    for split in splits:
        for cla in classes:
            generate_waymo_data(root, cla, split)
