from .distillation_losses import kd_loss
from ._base import BaseDistiller
from .registry import register_distiller
import torch


@register_distiller
class KD(BaseDistiller):
    requires_feature = False

    def __init__(self, student, teacher, criterion, args):
        super(KD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)
        logits_student = self.student(image)
        loss_kd = self.args.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.args.kd_temperature)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        losses_dict = {
            'loss_kd': loss_kd,
            'loss_gt': loss_gt
        }
        return logits_student, losses_dict
