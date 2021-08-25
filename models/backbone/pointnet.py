""" 
pointnet.py
Created by zenn at 2021/5/9 13:41
"""

import torch
import torch.nn.functional as F

import torch.nn as nn
from pointnet2.utils.pointnet2_modules import PointnetSAModule


class Pointnet_Backbone(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, use_fps=False, normalize_xyz=False, return_intermediate=False):
        super(Pointnet_Backbone, self).__init__()
        self.return_intermediate = return_intermediate
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.3,
                nsample=32,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
                use_fps=use_fps, normalize_xyz=normalize_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.5,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                use_fps=False, normalize_xyz=normalize_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.7,
                nsample=32,
                mlp=[256, 256, 256, 256],
                use_xyz=True,
                use_fps=False, normalize_xyz=normalize_xyz
            )
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, numpoints):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features, l_idxs = [xyz], [features], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, sample_idxs = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i], True)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_idxs.append(sample_idxs)
        if self.return_intermediate:
            return l_xyz[1:], l_features[1:], l_idxs[0]
        return l_xyz[-1], l_features[-1], l_idxs[0]
