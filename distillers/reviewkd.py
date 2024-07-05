from .registry import register_distiller
from ._base import BaseDistiller


@register_distiller
class REVIEWKD(BaseDistiller):
    def __init__(self, student, teacher, criterion, args):
        super(REVIEWKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label):
        raise NotImplementedError
