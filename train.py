import os
import logging
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from timm.utils.log import setup_default_logging
from timm.utils.metrics import AverageMeter, accuracy
from timm.utils.summary import get_outdir
from timm.utils.checkpoint_saver import CheckpointSaver
from timm.utils.random import random_seed
from timm.models import create_model, load_checkpoint
from distillers import get_distiller

from utils import TimePredictor, _load_config, get_dataset, get_optimizer
from custom_forward import apply_new_method

# logger setting
_logger = logging.getLogger("train")


# train and evaluation loop
def train_and_eval(epoch,
                   distiller,
                   train_loader,
                   eval_loader,
                   optimizer,
                   scheduler,
                   scaler,
                   device,
                   loss_func,
                   saver,
                   args):
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    losses_meter_dict = defaultdict(AverageMeter)
    eval_loss_meter = AverageMeter()
    eval_top1_meter = AverageMeter()
    eval_top5_meter = AverageMeter()

    train_iters = len(train_loader)
    eval_iters = len(eval_loader)
    distiller.train()
    for batch_idx, (images, labels, *additional_input) in enumerate(train_loader):
        image = images.to(device)
        labels = labels.to(device)
        additional_input = [i.to(device) for i in additional_input]
        with autocast():
            if args.distiller == 'crd':
                logits_student, losses_dict = distiller(image, labels, *additional_input)
            else:
                logits_student, losses_dict = distiller(image, labels)
            loss = sum(losses_dict.values())
        top1, top5 = accuracy(logits_student.detach(), labels, topk=(1, 5))
        top1_meter.update(top1.item(), image.size(0))
        top5_meter.update(top5.item(), image.size(0))

        loss_meter.update(loss.item(), image.size(0))
        for k in losses_dict:
            losses_meter_dict[k].update(losses_dict[k].item(), image.size(0))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.clip_grad:
            nn.utils.clip_grad_norm_(distiller.parameters(), max_norm=args.clip_grad, norm_type=args.clip_type)
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % args.log_interval == 0 or batch_idx == train_iters - 1:
            lr_list = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lr_list) / len(lr_list)
            losses_info = []
            for k, v in losses_meter_dict.items():
                info = f'{k.capitalize()}: {v.avg:.4f}'
                losses_info.append(info)
            losses_info = '  '.join(losses_info)
            _logger.info(
                f'Train-{epoch}: [{batch_idx}/{train_iters}], '
                f'Loss: {loss_meter.avg:.4f} {losses_info}, '
                f'Acc@1: {top1_meter.avg:.2f} Acc@5: {top5_meter.avg:.2f}, '
                f'LR: {lr:.2e}'
            )

    best_metric = None
    best_epoch = None
    with torch.no_grad():
        distiller.student.eval()
        for batch_idx, (image, labels) in enumerate(eval_loader):
            image = image.to(device)
            labels = labels.to(device)
            preds = distiller.student(image)
            loss = loss_func(preds, labels)
            top1, top5 = accuracy(preds, labels, topk=(1, 5))
            eval_loss_meter.update(loss.item(), image.size(0))
            eval_top1_meter.update(top1.item(), image.size(0))
            eval_top5_meter.update(top5.item(), image.size(0))
            if batch_idx % args.log_interval == 0 or batch_idx == eval_iters - 1:
                _logger.info(
                    f'\tEval-{epoch}: [{batch_idx}/{eval_iters}], '
                    f'Loss: {eval_loss_meter.avg:.4f}, Acc@1: {eval_top1_meter.avg:.2f} Acc@5: {eval_top5_meter.avg:.2f}'
                )
        best_metric, best_epoch = saver.save_checkpoint(epoch, metric=eval_top1_meter.avg)
    if args.sched == 'plateau':
        scheduler.step(eval_top1_meter.avg)
    else:
        scheduler.step()
    return best_metric, best_epoch


# main function
def main():
    # process args and some paths
    args, args_text = _load_config()
    output_path = args.output if args.output is not None else './output/train'
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model
        ])
    output_dir = get_outdir(output_path, exp_name)
    log_path = os.path.join(output_dir, 'train.log')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir)

    # setup logging
    setup_default_logging(log_path=log_path)
    random_seed(args.seed)

    # choose device
    assert torch.cuda.is_available(), 'This code only support cuda because the usage of amp.'
    device = torch.device("cuda")

    # get dataset
    train_dataset, eval_dataset, train_loader, eval_loader = get_dataset(args)
    # create model
    Distiller = get_distiller(args.distiller.lower())
    if args.weight is not None:
        model = create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes,
                             pretrained_cfg_overlay=dict(file=args.weight))
    else:
        model = create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes)
    if Distiller.requires_feature:
        apply_new_method(model)

    # create teacher model
    teacher = None
    if args.teacher:
        teacher = create_model(args.teacher, pretrained=args.teacher_pretrained, num_classes=args.num_classes)
        if args.teacher_weight:
            load_checkpoint(teacher, args.teacher_weight)
        if Distiller.requires_feature:
            apply_new_method(teacher)
        teacher.requires_grad_(False)
        teacher.eval()

    # create optimizer
    loss_func = nn.CrossEntropyLoss()

    # create distiller
    if args.distiller == 'crd':
        distiller = Distiller(model, teacher, loss_func, args, num_data=len(train_dataset))
    else:
        distiller = Distiller(model, teacher, loss_func, args)

    # print parameter amounts
    student_params, teacher_params, extra_params = distiller.compute_parameters()
    _logger.info(f'======================================================================\n'
                 f'Student params: {student_params / 1e6:.2f}M\n'
                 f'Teacher params: {teacher_params / 1e6:.2f}M\n'
                 f'Extra params: {extra_params / 1e6:.2f}M\n'
                 f'======================================================================\n')
    distiller.to(device)
    optimizer, scheduler = get_optimizer(distiller, args)
    # amp setting
    scaler = GradScaler()

    # Train and validation
    best_metric = None
    best_epoch = None
    saver = CheckpointSaver(
        model=model,
        optimizer=optimizer,
        args=args,
        checkpoint_dir=checkpoint_dir,
        recovery_dir=checkpoint_dir,
        decreasing=False,
        max_history=3
    )
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
    try:
        timer = TimePredictor(args.epochs)
        for epoch in range(1, args.epochs + 1):
            best_metric, best_epoch = train_and_eval(epoch=epoch,
                                                     distiller=distiller,
                                                     train_loader=train_loader,
                                                     eval_loader=eval_loader,
                                                     optimizer=optimizer,
                                                     scheduler=scheduler,
                                                     scaler=scaler,
                                                     device=device,
                                                     loss_func=loss_func,
                                                     saver=saver,
                                                     args=args)

            timer.update()
            _logger.info(f'Average running time of lateset {timer.history} epochs is {timer.average_duration:.2f}s, '
                         f'predicting finish time is {timer.get_predict()}')

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info(f'***** Best metric is {best_metric:.2f} at epoch {best_epoch}. *****.')


if __name__ == "__main__":
    main()
