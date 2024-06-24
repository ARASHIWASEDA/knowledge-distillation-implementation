from ._base import BaseDistiller
from .registry import register_distiller


@register_distiller
class Vanilla(BaseDistiller):
    requires_feature = False

    def __init__(self, student, teacher, criterion, args):
        super(Vanilla, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label):
        logits_student = self.student(image)
        loss_gt = self.criterion(logits_student, label)
        losses_dict = {
            'loss_gt': loss_gt
        }
        return logits_student, losses_dict
