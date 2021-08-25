""" 
rpn.py
Created by zenn at 2021/5/8 20:55
"""
import torch
from torch import nn
from pointnet2.utils import pytorch_utils as pt_utils

from pointnet2.utils.pointnet2_modules import PointnetSAModule


class P2BVoteNetRPN(nn.Module):

    def __init__(self, feature_channel, vote_channel=256, num_proposal=64, normalize_xyz=False):
        super().__init__()
        self.num_proposal = num_proposal
        self.FC_layer_cla = (
            pt_utils.Seq(feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(feature_channel, bn=True)
                .conv1d(1, activation=None))
        self.vote_layer = (
            pt_utils.Seq(3 + feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(feature_channel, bn=True)
                .conv1d(3 + feature_channel, activation=None))

        self.vote_aggregation = PointnetSAModule(
            radius=0.3,
            nsample=16,
            mlp=[1 + feature_channel, vote_channel, vote_channel, vote_channel],
            use_xyz=True,
            normalize_xyz=normalize_xyz)

        self.FC_proposal = (
            pt_utils.Seq(vote_channel)
                .conv1d(vote_channel, bn=True)
                .conv1d(vote_channel, bn=True)
                .conv1d(3 + 1 + 1, activation=None))

    def forward(self, xyz, feature):
        """

        :param xyz: B,N,3
        :param feature: B,f,N
        :return: B,N,4+1 (xyz,theta,targetnessscore)
        """
        estimation_cla = self.FC_layer_cla(feature).squeeze(1)
        score = estimation_cla.sigmoid()

        xyz_feature = torch.cat((xyz.transpose(1, 2).contiguous(), feature), dim=1)

        offset = self.vote_layer(xyz_feature)
        vote = xyz_feature + offset
        vote_xyz = vote[:, 0:3, :].transpose(1, 2).contiguous()
        vote_feature = vote[:, 3:, :]

        vote_feature = torch.cat((score.unsqueeze(1), vote_feature), dim=1)

        center_xyzs, proposal_features = self.vote_aggregation(vote_xyz, vote_feature, self.num_proposal)
        proposal_offsets = self.FC_proposal(proposal_features)
        estimation_boxes = torch.cat(
            (proposal_offsets[:, 0:3, :] + center_xyzs.transpose(1, 2).contiguous(), proposal_offsets[:, 3:5, :]),
            dim=1)

        estimation_boxes = estimation_boxes.transpose(1, 2).contiguous()
        return estimation_boxes, estimation_cla, vote_xyz, center_xyzs
