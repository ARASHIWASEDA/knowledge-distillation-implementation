import torch

from .registry import register_distiller
from ._base import BaseDistiller
from .utils import init_weights

import torch.nn as nn
import torch.nn.functional as F


@register_distiller
class FITNET(BaseDistiller):
    requires_feature = True

    def __init__(self, student, teacher, criterion, args):
        super(FITNET, self).__init__(student, teacher, criterion, args)

        assert self.student.is_cnn_model() and self.teacher.is_cnn_model(), 'FitNet only support CNN models'

        self.projector = nn.ModuleDict()

        for stage in self.args.fitnet_stages:
            _, shape_s = self.student.stage_info(stage)
            _, shape_t = self.teacher.stage_info(stage)

            chans_s, _, _ = shape_s
            chans_t, _, _ = shape_t

            projector = nn.Conv2d(chans_s, chans_t, 1, 1, 0, bias=False)
            self.projector[stage] = projector
        self.projector.apply(init_weights)

    def forward(self, image, labels):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, features_teacher = self.teacher(image, self.requires_feature)

        logits_student, featurea_student = self.student(image, self.requires_feature)

        losses_fitnet = []
        for stage in self.args.fitnet_stages:
            idx_s, _ = self.student.stage_info(stage)
            idx_t, _ = self.teacher.stage_info(stage)

            outs_teacher = features_teacher[idx_t]
            outs_student = self.projector[stage](featurea_student[idx_s])

            losses_fitnet.append(F.mse_loss(outs_student, outs_teacher))

        loss_fitnet = self.args.kd_loss_weight * sum(losses_fitnet)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, labels)
        losses_dict = {
            'loss_gt': loss_gt,
            'loss_fitnet': loss_fitnet
        }
        return logits_student, losses_dict
