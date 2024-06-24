import torch.nn as nn


class BaseDistiller(nn.Module):
    def __init__(self, student, teacher, criterion, args):
        super(BaseDistiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.args = args

    def forward(self, x):
        raise NotImplementedError

    def compute_parameters(self):
        student_params = 0
        teacher_params = 0
        extra_params = 0
        for n, p in self.named_parameters():
            if n.startswith('student'):
                student_params += p.numel()
            elif n.startswith('teacher'):
                teacher_params += p.numel()
            else:
                if p.requires_grad:
                    extra_params += p.numel()
        return student_params, teacher_params, extra_params
