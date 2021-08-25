""" 
baseModel.py
Created by zenn at 2021/5/9 14:40
"""

import torch
from easydict import EasyDict
import pytorch_lightning as pl
from datasets import points_utils
from utils.metrics import TorchSuccess, TorchPrecision
from utils.metrics import estimateOverlap, estimateAccuracy
import torch.nn.functional as F
import numpy as np


class BaseModel(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()

    def configure_optimizers(self):
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, data, output):
        """

        :param data: input data
        :param output:
        :return:
        """
        estimation_boxes = output['estimation_boxes']  # B,num_proposal,5
        estimation_cla = output['estimation_cla']  # B,N
        seg_label = data['seg_label']
        box_label = data['box_label']  # B,4
        proposal_center = output["center_xyz"]  # B,num_proposal,3
        vote_xyz = output["vote_xyz"]

        loss_seg = F.binary_cross_entropy_with_logits(estimation_cla, seg_label)

        loss_vote = F.smooth_l1_loss(vote_xyz, box_label[:, None, :3].expand_as(vote_xyz), reduction='none')  # B,N,3
        loss_vote = (loss_vote.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)

        dist = torch.sum((proposal_center - box_label[:, None, :3]) ** 2, dim=-1)

        dist = torch.sqrt(dist + 1e-6)  # B, K
        objectness_label = torch.zeros_like(dist, dtype=torch.float)
        objectness_label[dist < 0.3] = 1
        objectness_score = estimation_boxes[:, :, 4]  # B, K
        objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
        objectness_mask[dist < 0.3] = 1
        objectness_mask[dist > 0.6] = 1
        loss_objective = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                            pos_weight=torch.tensor([2.0]).cuda())
        loss_objective = torch.sum(loss_objective * objectness_mask) / (
                torch.sum(objectness_mask) + 1e-6)
        loss_box = F.smooth_l1_loss(estimation_boxes[:, :, :4],
                                    box_label[:, None, :4].expand_as(estimation_boxes[:, :, :4]),
                                    reduction='none')
        loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)

        return {
            "loss_objective": loss_objective,
            "loss_box": loss_box,
            "loss_seg": loss_seg,
            "loss_vote": loss_vote,
        }

    def generate_template(self, sequence, current_frame_id, results_bbs):
        """
        generate template for evaluating.
        the template can be updated using the previous predictions.
        :param sequence: the list of the whole sequence
        :param current_frame_id:
        :param results_bbs: predicted box for previous frames
        :return:
        """
        first_pc = sequence[0]['pc']
        previous_pc = sequence[current_frame_id - 1]['pc']
        if "firstandprevious".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([first_pc, previous_pc],
                                                               [results_bbs[0], results_bbs[current_frame_id - 1]],
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        elif "first".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(first_pc, results_bbs[0],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "previous".upper() in self.config.hape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(previous_pc, results_bbs[current_frame_id - 1],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "all".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([frame["pc"] for frame in sequence[:current_frame_id]],
                                                               results_bbs,
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        return template_pc, canonical_box

    def generate_search_area(self, sequence, current_frame_id, results_bbs):
        """
        generate search area for evaluating.

        :param sequence:
        :param current_frame_id:
        :param results_bbs:
        :return:
        """
        this_bb = sequence[current_frame_id]["3d_bbox"]
        this_pc = sequence[current_frame_id]["pc"]
        if ("previous_result".upper() in self.config.reference_BB.upper()):
            ref_bb = results_bbs[-1]
        elif ("previous_gt".upper() in self.config.reference_BB.upper()):
            previous_bb = sequence[current_frame_id - 1]["3d_bbox"]
            ref_bb = previous_bb
        elif ("current_gt".upper() in self.config.reference_BB.upper()):
            ref_bb = this_bb
        search_pc_crop = points_utils.generate_subwindow(this_pc, ref_bb,
                                                         scale=self.config.search_bb_scale,
                                                         offset=self.config.search_bb_offset)
        return search_pc_crop, ref_bb

    def prepare_input(self, template_pc, search_pc, template_box, *args, **kwargs):
        """
        construct input dict for evaluating
        :param template_pc:
        :param search_pc:
        :param template_box:
        :return:
        """
        template_points, idx_t = points_utils.regularize_pc(template_pc.points.T, self.config.template_size,
                                                            seed=1)
        search_points, idx_s = points_utils.regularize_pc(search_pc.points.T, self.config.search_size,
                                                          seed=1)
        template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
        search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32)
        data_dict = {
            'template_points': template_points_torch[None, ...],
            'search_points': search_points_torch[None, ...],
        }
        return data_dict

    def evaluate_one_sequence(self, sequence):
        """

        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(sequence)):  # tracklet
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
            else:

                # preparing search area
                search_pc_crop, ref_bb = self.generate_search_area(sequence, frame_id, results_bbs)
                # update template
                template_pc, canonical_box = self.generate_template(sequence, frame_id, results_bbs)
                # construct input dict
                data_dict = self.prepare_input(template_pc, search_pc_crop, canonical_box)

                end_points = self(data_dict)
                estimation_box = end_points['estimation_boxes']
                estimation_boxes_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
                best_box_idx = estimation_boxes_cpu[:, 4].argmax()
                estimation_box_cpu = estimation_boxes_cpu[best_box_idx, 0:4]
                candidate_box = points_utils.getOffsetBB(ref_bb, estimation_box_cpu, degrees=self.config.degrees,
                                                         use_z=self.config.use_z,
                                                         limit_box=self.config.limit_box)
                results_bbs.append(candidate_box)

            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)
        return ious, distances

    def validation_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        ious, distances = self.evaluate_one_sequence(sequence)
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success, on_step=True, on_epoch=True)
        self.log('precision/test', self.prec, on_step=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)
