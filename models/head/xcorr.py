# Created by zenn at 2021/5/8
import torch
from torch import nn
from pointnet2.utils import pytorch_utils as pt_utils
from pointnet2.utils import pointnet2_utils

import torch.nn.functional as F


class BaseXCorr(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=1)
        self.mlp = pt_utils.SharedMLP([in_channel, hidden_channel, hidden_channel, hidden_channel], bn=True)
        self.fea_layer = (pt_utils.Seq(hidden_channel)
                          .conv1d(hidden_channel, bn=True)
                          .conv1d(out_channel, activation=None))


class P2B_XCorr(BaseXCorr):
    def __init__(self, feature_channel, hidden_channel, out_channel):
        mlp_in_channel = feature_channel + 4
        super().__init__(mlp_in_channel, hidden_channel, out_channel)

    def forward(self, template_feature, search_feature, template_xyz):
        """

        :param template_feature: B,f,M
        :param search_feature: B,f,N
        :param template_xyz: B,M,3
        :return:
        """
        B = template_feature.size(0)
        f = template_feature.size(1)
        n1 = template_feature.size(2)
        n2 = search_feature.size(2)
        final_out_cla = self.cosine(template_feature.unsqueeze(-1).expand(B, f, n1, n2),
                                    search_feature.unsqueeze(2).expand(B, f, n1, n2))  # B,n1,n2

        fusion_feature = torch.cat(
            (final_out_cla.unsqueeze(1), template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(B, 3, n1, n2)),
            dim=1)  # B,1+3,n1,n2

        fusion_feature = torch.cat((fusion_feature, template_feature.unsqueeze(-1).expand(B, f, n1, n2)),
                                   dim=1)  # B,1+3+f,n1,n2

        fusion_feature = self.mlp(fusion_feature)

        fusion_feature = F.max_pool2d(fusion_feature, kernel_size=[fusion_feature.size(2), 1])  # B, f, 1, n2
        fusion_feature = fusion_feature.squeeze(2)  # B, f, n2
        fusion_feature = self.fea_layer(fusion_feature)

        return fusion_feature


class BoxAwareXCorr(BaseXCorr):
    def __init__(self, feature_channel, hidden_channel, out_channel, k=8, use_search_bc=False, use_search_feature=False,
                 bc_channel=9):
        self.k = k
        self.use_search_bc = use_search_bc
        self.use_search_feature = use_search_feature
        mlp_in_channel = feature_channel + 3 + bc_channel
        if use_search_bc: mlp_in_channel += bc_channel
        if use_search_feature: mlp_in_channel += feature_channel
        super(BoxAwareXCorr, self).__init__(mlp_in_channel, hidden_channel, out_channel)

    def forward(self, template_feature, search_feature, template_xyz,
                search_xyz=None, template_bc=None, search_bc=None):
        """

        :param template_feature: B,f,M
        :param search_feature: B,f,N
        :param template_xyz: B,M,3
        :param search_xyz: B.N,3
        :param template_bc: B,M,9
        :param search_bc: B.N,9
        :param args:
        :param kwargs:
        :return:
        """
        dist_matrix = torch.cdist(template_bc, search_bc)  # B, M, N
        template_xyz_feature_box = torch.cat([template_xyz.transpose(1, 2).contiguous(),
                                              template_bc.transpose(1, 2).contiguous(),
                                              template_feature], dim=1)
        # search_xyz_feature = torch.cat([search_xyz.transpose(1, 2).contiguous(), search_feature], dim=1)

        top_k_nearest_idx_b = torch.argsort(dist_matrix, dim=1)[:, :self.k, :]  # B, K, N
        top_k_nearest_idx_b = top_k_nearest_idx_b.transpose(1, 2).contiguous().int()  # B, N, K
        correspondences_b = pointnet2_utils.grouping_operation(template_xyz_feature_box,
                                                               top_k_nearest_idx_b)  # B,3+9+D,N,K
        if self.use_search_bc:
            search_bc_expand = search_bc.transpose(1, 2).unsqueeze(dim=-1).repeat(1, 1, 1, self.K)  # B,9,N,K
            correspondences_b = torch.cat([search_bc_expand, correspondences_b], dim=1)
        if self.use_search_feature:
            search_feature_expand = search_feature.unsqueeze(dim=-1).repeat(1, 1, 1, self.K)  # B,D,N,K
            correspondences_b = torch.cat([search_feature_expand, correspondences_b], dim=1)

        ## correspondences fusion head
        fusion_feature = self.mlp(correspondences_b)  # B,D,N,K
        fusion_feature, _ = torch.max(fusion_feature, dim=-1)  # B,D,N,1
        fusion_feature = self.fea_layer(fusion_feature.squeeze(dim=-1))  # B,D,N

        return fusion_feature
