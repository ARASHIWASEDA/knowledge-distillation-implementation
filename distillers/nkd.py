from .registry import register_distiller
from ._base import BaseDistiller


@register_distiller
class NKD(BaseDistiller):
    requires_feature = False

    def __init__(self, student, teacher, criterion, args):
        super(NKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, labels):
        raise NotImplementedError
