import torch.nn.functional as F


def kd_loss(logits_student, logits_teacher, kd_temperature):
    log_pred_student = F.log_softmax(logits_student / kd_temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / kd_temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
    loss_kd *= kd_temperature ** 2
    return loss_kd
