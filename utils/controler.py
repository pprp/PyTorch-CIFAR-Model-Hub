import datetime
import random

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler, optimizer
from torch.utils.data.dataset import random_split

from adabound import AdaBound, AdaBoundW
from labelsmoothing import LSR
from warmup import WarmupMultiStepLR

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
    elif args.optims == "adabound":
        optimizer = AdaBound(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    elif args.optims == "adaboundw":
        optimizer = AdaBoundW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    else:
        raise "Not Implemented."

    return optimizer


def build_scheduler(args):
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
    args.name = "e-%s_m-%s_d-%s__" % (args.name, args.model, args.datasest)
    # running time
    args.name += datetime.datetime.now().strftime("%mM_%dD_%HH")
    # random number
    args.name += "__{:02d}".format(random.randint(0, 99))
    return args.name
