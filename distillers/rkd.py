from .registry import register_distiller
from ._base import BaseDistiller

import torch
import torch.nn.functional as F


@register_distiller
class RKD(BaseDistiller):
    requires_feature = True

    def __init__(self, student, teacher, criterion, args):
        super(RKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, features_teacher = self.teacher(image, self.requires_feature)

        logits_student, features_student = self.student(image, self.requires_feature)

        features_student = features_student[-1]
        features_teacher = features_teacher[-1]

        stu = features_student.view(features_student.shape[0], -1)
        tea = features_teacher.view(features_teacher.shape[0], -1)

        with torch.no_grad():
            t_d = _pdist(tea, self.args.rkd_squared, self.rkd_eps)
            mend_td = t_d[t_d > 0].mean()
            t_d = t_d / mend_td
        d = _pdist(stu, self.args.rkd_squared, self.args.rkd_eps)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = tea.unsqueeze(0) - tea.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = stu.unsqueeze(0) - stu.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * (
                self.args.rkd_distance_weight * loss_d + self.args.rkd_angle_weight * loss_a)
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd
        }
        return logits_student, losses_dict


def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res
