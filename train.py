import argparse
import yaml
import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transform

from timm.utils.log import setup_default_logging
from timm.utils.summary import get_outdir
from timm.utils.checkpoint_saver import CheckpointSaver
from timm.models import create_model, load_checkpoint
from distillers import get_distiller

# logger setting
_logger = logging.getLogger("train")

# configs setting
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet18", type=str, help='model name')
parser.add_argument("--learning-rate", default=0.01, type=float, help='learning rate')
parser.add_argument("--epochs", default=3, type=int, help='epochs to train')
parser.add_argument("--device", default="cuda", type=str, help='training device')
parser.add_argument("--batch-size", default=32, type=int, help='batch size')
parser.add_argument("--log-interval", default=500, type=int, help='to print log info every designated iters')
parser.add_argument("--output", default=None, type=str, help='output path to save results')
parser.add_argument("--experiment", default=None, type=str, help='name of subfolder of outpur dir')
parser.add_argument("--num-classes", default=10, type=int, help='number of classes of the dataset')
parser.add_argument("--pretrained", default=False, action='store_true')
parser.add_argument("--dataset-download", default=False, action='store_true')
parser.add_argument("--dataset", default='cifar10', type=str, help='support cifar10 or cifar100 now')

# distillation setting
parser.add_argument("--distiller", default='vanilla', type=str)
parser.add_argument("--teacher", default=None, type=str, help='name of teacher model')
parser.add_argument("--teacher-pretrained", default=False, action='store_true')
parser.add_argument("--teacher-weight", default='./weights/resnet/resnet50_cifar10.pth.tar', type=str,
                    help='path of pre-trained teacher weight')
parser.add_argument("--kd-temperature", default=4., type=float, help='distillation temperature')
parser.add_argument("--kd-loss-weight", default=1., type=float)
parser.add_argument("--gt-loss-weight", default=1., type=float)


def train(epoch, distiller, loader, optimizer, args, device, total_iters):
    correct = 0
    total = 0
    distiller.train()
    for batch_idx, (images, labels) in enumerate(loader):
        image = images.to(device)
        labels = labels.to(device)
        logits_student, losses_dict = distiller(image, labels)
        loss = sum(losses_dict.values())
        preds = torch.argmax(logits_student.detach(), dim=1)
        correct += (preds == labels).sum()
        total += preds.shape[0]
        acc = 100.0 * correct / total
        if batch_idx % args.log_interval == 0 or batch_idx == total_iters - 1:
            losses_infos = []
            for k, v in losses_dict.items():
                info = f'{k.capitalize()} - {v.item():.4f}'
                losses_infos.append(info)
            losses_info = '  '.join(losses_infos)
            _logger.info(
                f'Train-{epoch}: [{batch_idx}/{total_iters}], '
                f'Loss: {losses_info}, Acc: {acc:.2f}'
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval(epoch, model, loader, args, device, total_iters, loss_func, saver):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for batch_idx, (image, labels) in enumerate(loader):
            image = image.to(device)
            labels = labels.to(device)
            preds = model(image)
            loss = loss_func(preds, labels)
            pred_labels = torch.argmax(preds, dim=1)
            correct += (pred_labels == labels).sum()
            total += pred_labels.shape[0]
            acc = 100.0 * correct / total
            if batch_idx % args.log_interval == 0 or batch_idx == total_iters - 1:
                _logger.info(
                    f'\tEval-{epoch}: [{batch_idx}/{total_iters}], '
                    f'Loss: {loss:.4f}, Acc: {acc:.2f}'
                )
        best_metric, best_epoch = saver.save_checkpoint(epoch, metric=acc)
    return best_metric, best_epoch


# main function
def main():
    # process args and some paths
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
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

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataset
    if args.dataset == 'cifar10':
        transforms = transform.Compose([
            transform.Resize(224),
            transform.ToTensor(),
            transform.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root="datasets", train=True, transform=transforms,
                                         download=args.dataset_download)
        eval_dataset = datasets.CIFAR10(root="datasets", train=False, transform=transforms,
                                        download=args.dataset_download)
    else:
        transforms = transform.Compose([
            transform.Resize(224),
            transform.ToTensor(),
            transform.Normalize(mean=((0.5071, 0.4867, 0.4408)), std=(0.2675, 0.2565, 0.2761))
        ])
        train_dataset = datasets.CIFAR100(root="datasets", train=True, transform=transforms,
                                          download=args.dataset_download)
        eval_dataset = datasets.CIFAR100(root="datasets", train=False, transform=transforms,
                                         download=args.dataset_download)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # create model
    model = create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes)

    # create teacher model
    teacher = None
    if args.teacher:
        teacher = create_model(args.teacher, pretrained=args.teacher_pretrained, num_classes=args.num_classes)
        if args.teacher_weight:
            load_checkpoint(teacher, args.teacher_weight)
        teacher.requires_grad_(False)
        teacher.eval()

    # create optimizer
    loss_func = nn.CrossEntropyLoss()

    # create distiller
    Distiller = get_distiller(args.distiller.lower())
    distiller = Distiller(model, teacher, loss_func, args)
    distiller.to(device)
    optimizer = torch.optim.SGD(distiller.parameters(), lr=args.learning_rate)

    # Train and validation
    total_train_iters = len(train_loader)
    total_eval_iters = len(eval_loader)
    best_metric = None

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
        for epoch in range(1, args.epochs + 1):
            train(epoch=epoch,
                  distiller=distiller,
                  loader=train_loader,
                  optimizer=optimizer,
                  args=args,
                  device=device,
                  total_iters=total_train_iters)

            best_metric, best_epoch = eval(epoch=epoch,
                                           model=model,
                                           loader=eval_loader,
                                           args=args,
                                           device=device,
                                           total_iters=total_eval_iters,
                                           loss_func=loss_func,
                                           saver=saver)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info(f'***** Best metric is {best_metric:.2f}({best_epoch}) *****.')


if __name__ == "__main__":
    main()
