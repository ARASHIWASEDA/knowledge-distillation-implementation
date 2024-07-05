import os
import argparse
import logging
from datetime import datetime
from collections import defaultdict

import torch
import timm
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from timm.utils.log import setup_default_logging
from timm.utils.metrics import AverageMeter, accuracy
from timm.utils.summary import get_outdir
from timm.utils.checkpoint_saver import CheckpointSaver
from timm.models import create_model, load_checkpoint
from distillers import get_distiller

_logger = logging.getLogger("eval")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet18", type=str, help='model name')
parser.add_argument("--device", default="cuda", type=str, help='training device')
parser.add_argument("--batch-size", default=128, type=int, help='batch size')
parser.add_argument("--log-interval", default=100, type=int, help='to print log info every designated iters')
parser.add_argument("--weight", default=None, type=str)
parser.add_argument("--dataset-download", default=False, action='store_true')
parser.add_argument("--dataset", default='cifar100', type=str, help='support cifar10 or cifar100 now')
parser.add_argument("--num-classes", default=10, type=int)


def eval(model, eval_loader, device, args):
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    eval_iters = len(eval_loader)
    with torch.no_grad():
        model.eval()
        for batch_idx, (images, labels) in enumerate(eval_loader):
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            top1, top5 = accuracy(preds, labels, topk=(1, 5))
            top1_meter.update(top1.item(), images.size(0))
            top5_meter.update(top5.item(), images.size(0))
            if batch_idx % args.log_interval == 0 or batch_idx == eval_iters - 1:
                _logger.info(
                    f'Eval: [{batch_idx}/{eval_iters}], '
                    f'Acc@1: {top1_meter.avg:.2f} Acc@5: {top5_meter.avg:.2f}'
                )
    return top1_meter.avg, top5_meter.avg


def main():
    # args
    args = parser.parse_args()
    # choose device
    assert torch.cuda.is_available(), 'This code only support cuda because the usage of amp.'
    device = torch.device("cuda")

    # load dataset
    if args.dataset == 'cifar10':
        eval_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        eval_dataset = datasets.CIFAR10(root="datasets", train=False, transform=eval_transforms,
                                        download=args.dataset_download)
    else:
        eval_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=((0.5071, 0.4867, 0.4408)), std=(0.2675, 0.2565, 0.2761))
        ])
        eval_dataset = datasets.CIFAR100(root="datasets", train=False, transform=eval_transforms,
                                         download=args.dataset_download)

    # create dataloader
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes)
    load_checkpoint(model, args.weight)
    model.to(device)

    # evaluation
    top1, top5 = eval(model, eval_loader, device, args)
    _logger.info(f'****** Evaluation finished. ACC@1 - {top1}, ACC@5 - {top5}. ******')


if __name__ == "__main__":
    main()
