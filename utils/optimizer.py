import torch


def get_optimizer(model, args):
    assert args.opt in ['sgd', 'adam', 'adamw'], 'Only support sgd, adam and adamw now.'
    assert args.sched in ['step', 'cosine', 'plateau'], 'Only support steplr, cosinelr and reduceonplateau'
    optimizer = None
    scheduler = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.sched == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_tmax, eta_min=args.min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.sched_mode, factor=args.sched_gamma,
                                                               patience=args.sched_patience, min_lr=args.min_lr)
    return optimizer, scheduler
