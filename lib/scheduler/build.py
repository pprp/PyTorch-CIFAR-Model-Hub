from torch.optim import lr_scheduler

from .schdulers import CyclicLR, GradualWarmupScheduler, WarmupMultiStepLR


def build_scheduler(args, optimizer):

    if args.optims in ['sam', 'asam']:
        optimizer = optimizer.optimizer

    if args.sched == 'warmup':
        scheduler = WarmupMultiStepLR(
            optimizer=optimizer,
            milestones=[int(e) for e in args.milestones.split(',')],
            gamma=0.5,
            warmup_epochs=int(args.epochs * 0.1),
        )
    elif args.sched == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in args.milestones.split(',')],
            gamma=args.gamma,
        )
    elif args.sched == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=args.epochs,
                                                   eta_min=0)
    elif args.sched == 'warmcosine':
        # cifar slow / total = 20 / 300
        # tmp_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        # scheduler = GradualWarmupScheduler(
        #     optimizer, 1, total_epoch=args.epochs * 0.1, after_scheduler=tmp_scheduler
        # )
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=10,
                                                             T_mult=2,
                                                             eta_min=0.0001)
    elif args.sched == 'custom':
        # tmp_scheduler = lr_scheduler.LambdaLR(
        #     optimizer, lambda step: (1.0 - step / args.epochs), last_epoch=-1
        # )
        # scheduler = GradualWarmupScheduler(
        #     optimizer,
        #     1,
        #     total_epoch=int(args.epochs * 0.5),
        #     after_scheduler=tmp_scheduler,
        # )
        scheduler = CyclicLR(
            optimizer,
            base_lr=0,
            max_lr=args.lr,
            step_size_up=15,
            step_size_down=args.epochs - 15,
        )
    else:
        raise 'Not Implemented.'

    return scheduler
