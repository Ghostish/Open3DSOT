"""
pointnet.py
Created by zenn at 2021/5/9 13:41
"""

import torch
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

    def __init__(self, use_fps=False, normalize_xyz=False, return_intermediate=False, input_channels=0):
        super(Pointnet_Backbone, self).__init__()
        self.return_intermediate = return_intermediate
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.3,
                nsample=32,
                mlp=[input_channels, 64, 64, 128],
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


class MiniPointNet(nn.Module):

    def __init__(self, input_channel, per_point_mlp, hidden_mlp, output_size=0):
        """

        :param input_channel: int
        :param per_point_mlp: list
        :param hidden_mlp: list
        :param output_size: int, if output_size <=0, then the final fc will not be used
        """
        super(MiniPointNet, self).__init__()
        seq_per_point = []
        in_channel = input_channel
        for out_channel in per_point_mlp:
            seq_per_point.append(nn.Conv1d(in_channel, out_channel, 1))
            seq_per_point.append(nn.BatchNorm1d(out_channel))
            seq_per_point.append(nn.ReLU())
            in_channel = out_channel
        seq_hidden = []
        for out_channel in hidden_mlp:
            seq_hidden.append(nn.Linear(in_channel, out_channel))
            seq_hidden.append(nn.BatchNorm1d(out_channel))
            seq_hidden.append(nn.ReLU())
            in_channel = out_channel

        # self.per_point_mlp = nn.Sequential(*seq)
        # self.pooling = nn.AdaptiveMaxPool1d(output_size=1)
        # self.hidden_mlp = nn.Sequential(*seq_hidden)

        self.features = nn.Sequential(*seq_per_point,
                                      nn.AdaptiveMaxPool1d(output_size=1),
                                      nn.Flatten(),
                                      *seq_hidden)
        self.output_size = output_size
        if output_size >= 0:
            self.fc = nn.Linear(in_channel, output_size)

    def forward(self, x):
        """

        :param x: B,C,N
        :return: B,output_size
        """

        # x = self.per_point_mlp(x)
        # x = self.pooling(x)
        # x = self.hidden_mlp(x)
        x = self.features(x)
        if self.output_size > 0:
            x = self.fc(x)
        return x


class SegPointNet(nn.Module):

    def __init__(self, input_channel, per_point_mlp1, per_point_mlp2, output_size=0, return_intermediate=False):
        """

        :param input_channel: int
        :param per_point_mlp: list
        :param hidden_mlp: list
        :param output_size: int, if output_size <=0, then the final fc will not be used
        """
        super(SegPointNet, self).__init__()
        self.return_intermediate = return_intermediate
        self.seq_per_point = nn.ModuleList()
        in_channel = input_channel
        for out_channel in per_point_mlp1:
            self.seq_per_point.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, 1),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU()
                ))
            in_channel = out_channel

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.seq_per_point2 = nn.ModuleList()
        in_channel = in_channel + per_point_mlp1[1]
        for out_channel in per_point_mlp2:
            self.seq_per_point2.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, 1),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU()
                ))
            in_channel = out_channel

        self.output_size = output_size
        if output_size >= 0:
            self.fc = nn.Conv1d(in_channel, output_size, 1)

    def forward(self, x):
        """

        :param x: B,C,N
        :return: B,output_size,N
        """
        second_layer_out = None
        for i, mlp in enumerate(self.seq_per_point):
            x = mlp(x)
            if i == 1:
                second_layer_out = x
        pooled_feature = self.pool(x)  # B,C,1
        pooled_feature_expand = pooled_feature.expand_as(x)
        x = torch.cat([second_layer_out, pooled_feature_expand], dim=1)
        for mlp in self.seq_per_point2:
            x = mlp(x)
        if self.output_size > 0:
            x = self.fc(x)
        if self.return_intermediate:
            return x, pooled_feature.squeeze(dim=-1)
        return x

