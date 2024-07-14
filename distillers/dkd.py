import torch

from .registry import register_distiller
from ._base import BaseDistiller
from .distillation_losses import dkd_loss


@register_distiller
class DKD(BaseDistiller):
    requires_feature = False

    def __init__(self, student, teacher, criterion, args):
        super(DKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * dkd_loss(logits_student, logits_teacher, label, self.args.dkd_alpha,
                                                      self.args.dkd_beta, self.args.dkd_temperature)
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd
        }
        return logits_student, losses_dict
