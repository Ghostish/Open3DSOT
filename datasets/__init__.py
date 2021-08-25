""" 
___init__.py
Created by zenn at 2021/7/18 15:50
"""
from datasets import kitti, sampler


def get_dataset(config, type='train', **kwargs):
    if config.dataset == 'kitti':
        data = kitti.kittiDataset(path=config.path,
                                  split=kwargs.get('split', 'train'),
                                  category_name=config.category_name,
                                  coordinate_mode=config.coordinate_mode)
    else:
        data = None

    if type == 'train':
        return sampler.PointTrackingSampler(dataset=data,
                                            random_sample=config.random_sample,
                                            sample_per_epoch=config.sample_per_epoch,
                                            config=config)
    else:
        return sampler.TestTrackingSampler(dataset=data, config=config)
