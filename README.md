# Box-Aware Tracker (BAT)
Pytorch-Lightning implementation of the Box-Aware Tracker.  

[Box-Aware Feature Enhancement for Single Object Tracking on Point Clouds](https://arxiv.org/abs/2108.04728). **ICCV 2021** 

[Chaoda Zheng](https://github.com/Ghostish/), [Xu Yan](https://yanx27.github.io/), Jiantao Gao, Weibing Zhao, Wei Zhang, [Zhen Li*](https://mypage.cuhk.edu.cn/academics/lizhen/), Shuguang Cui

### Citation
```bibtex
@inproceedings{zheng2021box,
  title={Box-Aware Feature Enhancement for Single Object Tracking on Point Clouds},
  author={Zheng, Chaoda and Yan, Xu and Gao, Jiantao and Zhao, Weibing and Zhang, Wei and Li, Zhen and Cui, Shuguang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13199--13208},
  year={2021}
}
```
<img src="figures/results.gif" width="1000"/>

### Recent Updates
+ Update experiment results in the main paper, see [arxiv link](https://arxiv.org/abs/2108.04728)
+ Add `--preloading` option for traning (enabled by default in previous version). see [Training](#training)
+ Add support for Waymo
+ Add support for NuScenes
+ ...
### Features
+ Modular design. It is easy to config the model and training/testing behaviors through just a `.yaml` file.
+ DDP support for both training and testing.
+ Provide a 3rd party implementation of [P2B](https://github.com/HaozheQi/P2B).
### Setup
Installation
+ Create the environment
  ```
  git clone https://github.com/Ghostish/BAT.git
  cd BAT
  conda create -n bat  python=3.6
  conda activate bat
  ```
+ Install pytorch
  ```
  conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
  ```
  Our code is well tested with pytorch 1.4.0 and CUDA 10.1. But other platforms may also work. Follow [this](https://pytorch.org/get-started/locally/) to install another version of pytorch. **Note: In order to reproduce the reported results with the provided checkpoints, please use CUDA 10.x.** 

+ Install other dependencies:
  ```
  pip install -r requirement.txt
  ```
  Install the nuscenes-devkit if you use want to use NuScenes dataset:
  ```
  pip install nuscenes-devkit
  ```

KITTI dataset
+ Download the data for [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).
+ Unzip the downloaded files.
+ Put the unzipped files under the same folder as following.
  ```
  [Parent Folder]
  --> [calib]
      --> {0000-0020}.txt
  --> [label_02]
      --> {0000-0020}.txt
  --> [velodyne]
      --> [0000-0020] folders with velodynes .bin files
  ```

NuScenes dataset
+ Download the dataset from the [download page](https://www.nuscenes.org/download)
+ Extract the downloaded files and make sure you have the following structure:
  ```
  [Parent Folder]
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    maps	        -	Folder for all map files: rasterized .png images and vectorized .json files.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
  ```
>Note: We use the **train_track** split to train our model and test it with the **val** split. Both splits are officially provided by NuScenes. During testing, we ignore the sequences where there is no point in the first given bbox.

Waymo dataset
+ Download and prepare dataset by the instruction of [CenterPoint](https://github.com/tianweiy/CenterPoint/blob/master/docs/WAYMO.md).
  ```
  [Parent Folder]
    tfrecord_training	                    
    tfrecord_validation	                 
    train 	                                    -	all training frames and annotations 
    val   	                                    -	all validation frames and annotations 
    infos_train_01sweeps_filter_zero_gt.pkl
    infos_val_01sweeps_filter_zero_gt.pkl
  ```
+ Prepare SOT dataset. Data from specific category and split will be merged (e.g., sot_infos_vehicle_train.pkl).
```bash
  python datasets/generate_waymo_sot.py
```

### Quick Start
#### Training
To train a model, you must specify the `.yaml` file with `--cfg` argument. The `.yaml` file contains all the configurations of the dataset and the model. Currently, we provide four `.yaml` files under the [*cfgs*](./cfgs) directory. **Note:** Before running the code, you will need to edit the `.yaml` file by setting the `path` argument as the correct root of the dataset.
```bash
python main.py --gpu 0 1 --cfg cfgs/BAT_Car.yaml  --batch_size 50 --epoch 60 --preloading
```
After you start training, you can start Tensorboard to monitor the training process:
```
tensorboard --logdir=./ --port=6006
```
By default, the trainer runs a full evaluation on the full test split after training every epoch. You can set `--check_val_every_n_epoch` to a larger number to speed up the training. The `--preloading` flag is used to preload the training samples into the memory to save traning time. Remove this flag if you don't have enough memory.
#### Testing
To test a trained model, specify the checkpoint location with `--checkpoint` argument and send the `--test` flag to the command.
```bash
python main.py --gpu 0 1 --cfg cfgs/BAT_Car.yaml  --checkpoint /path/to/checkpoint/xxx.ckpt --test
```

### Reproduction
This codebase produces better results than those we report in our original paper.
| Model | Category | Success| Precision| Checkpoint
|--|--|--|--|--|
| BAT-KITTI | Car	|65.37 | 78.88|pretrained_models/bat_kitti_car.ckpt
| BAT-NuScenes | Car	|40.73 | 43.29|pretrained_models/bat_nuscenes_car.ckpt
| BAT-KITTI | Pedestrian | 45.74| 74.53| pretrained_models/bat_kitti_pedestrian.ckpt

Three trained BAT models for KITTI and NuScenes datasets are provided in the  [*pretrained_models*](./pretrained_models) directory. To reproduce the results, simply run the code with the corresponding `.yaml` file and checkpoint. For example, to reproduce the tracking results on KITTI Car, just run:
```bash
python main.py --gpu 0 1 --cfg cfgs/BAT_Car.yaml  --checkpoint ./pretrained_models/bat_kitti_car.ckpt --test
```

### To-dos
- [x] DDP support
- [x] Multi-gpus testing
- [x] Add NuScenes dataset
- [ ] Add codes for visualization
- [ ] Add support for more methods

### Acknowledgment
+ This repo is built upon [P2B](https://github.com/HaozheQi/P2B) and [SC3D](https://github.com/SilvioGiancola/ShapeCompletion3DTracking).
+ Thank Erik Wijmans for his pytorch implementation of [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)

### License
This repository is released under MIT License (see LICENSE file for details).
