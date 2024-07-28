import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='', type=str, help='path to config file')
parser.add_argument("--model", default="resnet18", type=str, help='model name')
parser.add_argument("--lr", default=0.01, type=float, help='learning rate')
parser.add_argument("--min-lr", default=1e-6, type=float, help='min learning rate')
parser.add_argument("--epochs", default=3, type=int, help='epochs to train')
parser.add_argument("--device", default="cuda", type=str, help='training device')
parser.add_argument("--batch-size", default=32, type=int, help='batch size')
parser.add_argument("--log-interval", default=500, type=int, help='to print log info every designated iters')
parser.add_argument("--output", default=None, type=str, help='output path to save results')
parser.add_argument("--experiment", default=None, type=str, help='name of subfolder of outpur dir')
parser.add_argument("--num-classes", default=10, type=int, help='number of classes of the dataset')
parser.add_argument("--pretrained", default=False, action='store_true')
parser.add_argument("--weight", default=None, type=str)
parser.add_argument("--dataset-download", default=False, action='store_true')
parser.add_argument("--dataset", default='cifar10', type=str, help='support cifar10 or cifar100 now')
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--seed", default=42, type=int)

# optimizer settings
parser.add_argument("--opt", type=str, default='sgd')
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--sched", type=str, default='plateau')
parser.add_argument("--step-size", type=int, default=50)
parser.add_argument("--sched-gamma", type=float, default=0.1)
parser.add_argument("--sched-mode", type=str, default="max")
parser.add_argument("--sched-patience", type=int, default=30)
parser.add_argument("--cosine-tmax", type=int, default=100)
parser.add_argument("--clip-grad", type=float, default=None)
parser.add_argument("--clip-type", type=int, default=2)
parser.add_argument("--multistep-milestones", nargs='+', type=int, default=[150, 180, 210])

# distillation settings
parser.add_argument("--distiller", default='vanilla', type=str)
parser.add_argument("--teacher", default=None, type=str, help='name of teacher model')
parser.add_argument("--teacher-pretrained", default=False, action='store_true')
parser.add_argument("--teacher-weight", default='./weights/resnet/resnet50_cifar10.pth.tar', type=str,
                    help='path of pre-trained teacher weight')
parser.add_argument("--kd-loss-weight", default=1., type=float)
parser.add_argument("--gt-loss-weight", default=1., type=float)

# KD settings
parser.add_argument("--kd-temperature", default=4., type=float, help='distillation temperature')

# NKD settings
parser.add_argument("--nkd-gamma", default=1., type=float)
parser.add_argument("--nkd-temperature", default=1., type=float)

# DKD settings
parser.add_argument("--dkd-alpha", default=1., type=float)
parser.add_argument("--dkd-beta", default=1., type=float)
parser.add_argument("--dkd-temperature", default=1., type=float)

# DIST settings
parser.add_argument("--dist-beta", default=1., type=float)
parser.add_argument("--dist-gamma", default=1., type=float)
parser.add_argument("--dist-temperature", default=1., type=float)

# RKD settings
parser.add_argument("--rkd-squared", action='store_true', default=False)
parser.add_argument("--rkd-eps", default=1e-12, type=float)
parser.add_argument("--rkd-distance-weight", default=25, type=int)
parser.add_argument("--rkd-angle-weight", default=50, type=int)

# OFAKD settings
parser.add_argument("--ofa-stages", default=[1, 2, 3, 4], nargs='+', type=int)
parser.add_argument("--ofa-loss-weight", default=1.0, type=float)
parser.add_argument("--ofa-temperature", default=1.0, type=float)
parser.add_argument("--ofa-eps", default=1.0, type=float)

# CRD settings
parser.add_argument('--crd-feature-dim', default=128, type=int)
parser.add_argument('--crd-k', default=16384, type=int)
parser.add_argument('--crd-momentum', default=0.5, type=float)
parser.add_argument('--crd-temperature', default=0.07, type=float)

# FITNET settings
parser.add_argument("--fitnet-stages", nargs='+', default=[1, 2, 3, 4], type=int)

# THKD settings
parser.add_argument("--thkd-stages", nargs='+', default=[1, 2, 3], type=int)
parser.add_argument("--thkd-shallow-stages", nargs='+', default=[1], type=int)
parser.add_argument("--thkd-temperature", default=4.0, type=float)
parser.add_argument("--thkd-blk-num", default=1, type=int)
parser.add_argument("--thkd-gamma", default=1.5, type=float)
parser.add_argument("--thkd-loss-weight", default=1., type=float)

# CMTHKD settings
parser.add_argument("--cmthkd-stages", nargs='+', default=[1, 2, 3, 4], type=int)
parser.add_argument("--cmthkd-shallow-stages", nargs='+', default=[1], type=int)
parser.add_argument("--cmthkd-temperature", default=4.0, type=float)
parser.add_argument("--cmthkd-gamma", default=1.5, type=float)
parser.add_argument("--cmthkd-loss-weight", default=1., type=float)
parser.add_argument("--cmthkd-blk-num", default=2, type=int)


def _load_config():
    config_parser = parser
    args_known, parser_unknown = parser.parse_known_args()  # args_config are args specified in terminal
    if args_known.config:
        with open(args_known.config, 'r') as f:
            cfg = yaml.safe_load(f)
            config_parser.set_defaults(**cfg)

    args = config_parser.parse_args(parser_unknown)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
