import torch
import torch.nn.functional as F


# kd loss
def kd_loss(logits_student, logits_teacher, kd_temperature):
    log_pred_student = F.log_softmax(logits_student / kd_temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / kd_temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
    loss_kd *= kd_temperature ** 2
    return loss_kd


# dkd loss
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


# dist loss
def dist_loss(logits_student, logits_teacher, beta=1., gamma=1., kd_temperature=1.):
    pred_student = F.softmax(logits_student / kd_temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / kd_temperature, dim=1)
    inter_loss = kd_temperature ** 2 * inter_class_relation(pred_student, pred_teacher)
    intra_loss = kd_temperature ** 2 * intra_class_relation(pred_student, pred_teacher)
    return beta * inter_loss + gamma * intra_loss


def inter_class_relation(pred_student, pred_teacher):
    return 1 - pearson_correlation(pred_student, pred_teacher).mean()


def intra_class_relation(pred_student, pred_teacher):
    return inter_class_relation(pred_student.transpose(0, 1), pred_teacher.transpose(0, 1))


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps)


# nkd loss
def nkd_loss(logits_student, logits_teacher, label, alpha, beta, temperature=1.):
    # todo：nkd和dkd有什么区别
    target = label.reshape(-1)
    target = target.unsqueeze(1)

    N, c = logits_student.shape
    log_pred_student = F.log_softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher, dim=1)  # todo：这里为什么要先计算softmax，如果直接计算target看看效果

    target_student = torch.gather(log_pred_student, 1, label)  # get the score of target class
    target_teacher = torch.gather(pred_teacher, 1, label)  # shape: (batch_size,1)
    tckd_loss = -(target_teacher * target_teacher).mean()

    mask = torch.ones_like(logits_student).scatter_(1, label, 0).bool()
    logits_student = logits_student[mask].reshape(N, -1)
    logits_teacher = logits_teacher[mask].reshape(N, -1)

    non_target_student = F.log_softmax(logits_student / temperature, dim=1)  # todo:这里的损失函数改成ranking
    non_target_teacher = F.softmax(logits_teacher / temperature, dim=1)

    nckd_loss = -(non_target_student, non_target_teacher).sum(dim=1).mean()
    return alpha * tckd_loss + beta * (temperature ** 2) * nckd_loss


# ofa loss
def ofa_loss(logits_student, logits_teacher, target_mask, eps, temperature=1.):
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps
    loss = torch.sum(-(prod - target_mask) * torch.log(pred_student), dim=-1)
    return loss.mean()
