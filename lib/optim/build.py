import torch.optim as optim

from .adabound import AdaBound, AdaBoundW
from .adamw import AdamW
from .asam import ASAM, SAM


def build_optimizer(model, args):
    if args.optims == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optims == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
        )
    elif args.optims == 'adamw':
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
        )
    elif args.optims == 'nesterov':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    elif args.optims == 'adabound':
        optimizer = AdaBound(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=args.lr)
    elif args.optims == 'adaboundw':
        optimizer = AdaBoundW(filter(lambda p: p.requires_grad,
                                     model.parameters()),
                              lr=args.lr)
    elif args.optims == 'sam':
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
    elif args.optims == 'asam':
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
        raise 'Not Implemented.'

    return optimizer
