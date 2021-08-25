""" PointNet++ Layers
Modified by Zenn
Date: Feb 2021
"""
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2.utils import pytorch_utils as pt_utils

from pointnet2.utils import pointnet2_utils

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class _PointnetSAModuleBase(nn.Module):
    def __init__(self, use_fps=False):
        super(_PointnetSAModuleBase, self).__init__()
        self.groupers = None
        self.mlps = None
        self.use_fps = use_fps

    def forward(self, xyz, features, npoint, return_idx=False):
        # modified to return sample idxs
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        self.npoint = npoint
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if self.use_fps:
            sample_idxs = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        else:
            sample_idxs = torch.arange(self.npoint).repeat(xyz.size(0), 1).int().cuda()

        new_xyz = (
            pointnet2_utils.gather_operation(xyz_flipped, sample_idxs)
                .transpose(1, 2)
                .contiguous()
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)
        if return_idx:
            return new_xyz, torch.cat(new_features_list, dim=1), sample_idxs
        else:
            return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, radii, nsamples, mlps, bn=True, use_xyz=True, use_fps=False, normalize_xyz=False):
        # type: (PointnetSAModuleMSG, List[float],List[int], List[List[int]], bool, bool,bool) -> None
        super(PointnetSAModuleMSG, self).__init__(use_fps=use_fps)

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, normalize_xyz=normalize_xyz))

            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
            self, mlp, radius=None, nsample=None, bn=True, use_xyz=True, use_fps=False, normalize_xyz=False
    ):
        # type: (PointnetSAModule, List[int], float, int, bool, bool, bool,bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            use_fps=use_fps,
            normalize_xyz=normalize_xyz
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


class FlowEmbedding(nn.Module):
    """Modified from https://github.com/hyangwinter/flownet3d_pytorch/blob/master/util.py"""

    def __init__(self, radius, nsample, in_channel, mlp, pooling='max', corr_func='concat', knn=True):
        super(FlowEmbedding, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.pooling = pooling
        self.corr_func = corr_func
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if corr_func is 'concat':
            last_channel = in_channel * 2 + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, feature1, feature2):
        """
        Input:
            xyz1: (batch_size, npoint, 3)
            xyz2: (batch_size, npoint, 3)
            feat1: (batch_size, channel, npoint)
            feat2: (batch_size, channel, npoint)
        Output:
            xyz1: (batch_size, npoint, 3)
            feat1_new: (batch_size, mlp[-1], npoint)
        """
        xyz1_t = xyz1.permute(0, 2, 1).contiguous()
        xyz2_t = xyz2.permute(0, 2, 1).contiguous()
        # feature1 = feature1.permute(0, 2, 1).contiguous()
        # feature2 = feature2.permute(0, 2, 1).contiguous()

        B, N, C = xyz1.shape
        if self.knn:
            idx = pointnet2_utils.knn_point(self.nsample, xyz1, xyz2)  # (B, npoint, nsample)
        else:
            idx, cnt = pointnet2_utils.ball_query(self.radius, self.nsample, xyz2, xyz1)  # (B, npoint, nsample)

        xyz2_grouped = pointnet2_utils.grouping_operation(xyz2_t, idx)  # (B, 3, npoint, nsample)
        pos_diff = xyz2_grouped - xyz1_t.view(B, -1, N, 1)  # (B, 3, npoint, nsample)

        feat2_grouped = pointnet2_utils.grouping_operation(feature2, idx)  # [B, C, npoint, nsample]
        if self.corr_func == 'concat':
            feat_diff = torch.cat([feat2_grouped, feature1.view(B, -1, N, 1).repeat(1, 1, 1, self.nsample)], dim=1)

        feat1_new = torch.cat([pos_diff, feat_diff], dim=1)  # [B, 2*C+3,npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat1_new = F.relu(bn(conv(feat1_new)))

        feat1_new = torch.max(feat1_new, -1)[0]  # [B, mlp[-1], npoint]
        return xyz1, feat1_new


class PointNetSetUpConv(nn.Module):
    def __init__(self, nsample, radius, f1_channel, f2_channel, mlp, mlp2, knn=True):
        super(PointNetSetUpConv, self).__init__()
        self.nsample = nsample
        self.radius = radius
        self.knn = knn
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = f2_channel + 3
        for out_channel in mlp:
            self.mlp1_convs.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
        if len(mlp) is not 0:
            last_channel = mlp[-1] + f1_channel
        else:
            last_channel = last_channel + f1_channel
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm1d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, feature1, feature2):
        """
            Feature propagation from xyz2 (less points) to xyz1 (more points)
        Inputs:
            xyz1: (batch_size, npoint1, 3)
            xyz2: (batch_size, npoint2, 3)
            feat1: (batch_size, channel1, npoint1) features for xyz1 points (earlier layers)
            feat2: (batch_size, channel2, npoint2) features for xyz2 points
        Output:
            feat1_new: (batch_size, mlp[-1] or mlp2[-1] or channel1+3, npoint2)
            TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
        """
        xyz1_t = xyz1.permute(0, 2, 1).contiguous()
        xyz2_t = xyz2.permute(0, 2, 1).contiguous()
        # feature1 = feature1.permute(0, 2, 1).contiguous()
        # feature2 = feature2.permute(0, 2, 1).contiguous()
        B, C, N = xyz1_t.shape
        if self.knn:
            idx = pointnet2_utils.knn_point(self.nsample, xyz1, xyz2)  # (B, npoint1, nsample)
        else:
            idx, cnt = pointnet2_utils.ball_query(self.radius, self.nsample, xyz2, xyz1)  # (B, npoint1, nsample)

        xyz2_grouped = pointnet2_utils.grouping_operation(xyz2_t, idx)
        pos_diff = xyz2_grouped - xyz1_t.view(B, -1, N, 1)  # [B,3,N1,S]

        feat2_grouped = pointnet2_utils.grouping_operation(feature2, idx)
        feat_new = torch.cat([feat2_grouped, pos_diff], dim=1)  # [B,C1+3,N1,S]
        for conv in self.mlp1_convs:
            feat_new = conv(feat_new)
        # max pooling
        feat_new = feat_new.max(-1)[0]  # [B,mlp1[-1],N1]
        # concatenate feature in early layer
        if feature1 is not None:
            feat_new = torch.cat([feat_new, feature1], dim=1)
        # feat_new = feat_new.view(B,-1,N,1)
        for conv in self.mlp2_convs:
            feat_new = conv(feat_new)

        return feat_new


if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    #  test_module = PointnetFPModule(mlp=[6, 6])
    #  test_module.cuda()
    #  from torch.autograd import gradcheck
    #  inputs = (xyz, xyz, None, xyz_feats)
    #  test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    #  print(test)

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
