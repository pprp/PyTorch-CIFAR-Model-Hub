import datetime
import random

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler, optimizer
from torch.utils.data.dataset import random_split

from utils.adabound import AdaBound, AdaBoundW
from utils.labelsmoothing import LSR
from utils.schd import GradualWarmupScheduler
from utils.warmup import WarmupMultiStepLR
from ASAM.asam import SAM, ASAM

"""
Generate optimizer and scheduler
"""


def build_optimizer(model, args):
    if args.optims == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optims == "nesterov":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    elif args.optims == "adabound":
        optimizer = AdaBound(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    elif args.optims == "adaboundw":
        optimizer = AdaBoundW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    elif args.optims == "sam":
        opt = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        optimizer = SAM(
            optimizer=opt,
            model=model,
            rho=0.5,
            eta=0,
        )
    elif args.optims == "asam":
        opt = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        optimizer = ASAM(
            optimizer=opt,
            model=model,
            rho=0.5,
            eta=0,
        )
    else:
        raise "Not Implemented."

    return optimizer


def build_scheduler(args, optimizer):

    if args.optims in ["sam", "asam"]:
        optimizer = optimizer.optimizer

    if args.sched == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer=optimizer, milestones=[int(e) for e in args.milestones.split(",")]
        )
    elif args.sched == "multistep":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in args.milestones.split(",")],
            gamma=args.gamma,
        )
    elif args.sched == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=0
        )
    elif args.sched == "warmcosine":
        # cifar slow / total = 20 / 300
        tmp_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler = GradualWarmupScheduler(
            optimizer, 1, total_epoch=20, after_scheduler=tmp_scheduler
        )
    else:
        raise "Not Implemented."

    return scheduler


def build_criterion(args):
    if args.crit == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.crit == "lsr":
        criterion = LSR()
    else:
        raise "Not Implemented."
    return criterion


def build_expname(args):
    # model and dataset
    args.name = "e-%s_m-%s_d-%s__" % (args.name, args.model, args.dataset)
    # running time
    args.name += datetime.datetime.now().strftime("%mM_%dD_%HH")
    # random number
    args.name += "__{:02d}".format(random.randint(0, 99))
    return args.name
