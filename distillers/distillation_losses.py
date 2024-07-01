import torch
import torch.nn.functional as F


def kd_loss(logits_student, logits_teacher, kd_temperature):
    log_pred_student = F.log_softmax(logits_student / kd_temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / kd_temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
    loss_kd *= kd_temperature ** 2
    return loss_kd


def dkd_loss(logits_student, logits_teacher, label, alpha, beta, kd_temperature):
    gt_mask, other_mask = _get_mask(logits_student, label)
    pred_student = F.softmax(logits_student / kd_temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / kd_temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * (kd_temperature ** 2)
    pred_teacher_second = F.softmax(logits_teacher / kd_temperature - 1000.0 * gt_mask, dim=1)
    log_pred_student_second = F.log_softmax(logits_student / kd_temperature - 1000.0 * gt_mask, dim=1)
    nckd_loss = F.kl_div(log_pred_student_second, pred_teacher_second, reduction='batchmean') * (kd_temperature ** 2)
    return alpha * tckd_loss + beta * nckd_loss


def _get_mask(logits, label):
    target = label.reshape(-1)
    target = target.unsqueeze(1)
    other_mask = torch.ones_like(logits).scatter_(1, target, 0).bool()  # 1 for non-target classes
    gt_mask = torch.zeros_like(logits).scatter_(1, target, 1).bool()  # 1 for target classes
    return gt_mask, other_mask


def cat_mask(logits, gt_mask, other_mask):
    target_pred = (logits * gt_mask).sum(dim=1, keepdims=True)  # target_pred:(batch_size,1)
    other_pred = (logits * other_mask).sum(dim=1, keepdim=True)
    pred = torch.cat([target_pred, other_pred], dim=1)  # (batch_size, 2)
    return pred
