import argparse
import inspect
import json
import logging
import os
import traceback
import warnings
from functools import wraps
from importlib.util import find_spec
from typing import List, Sequence

import colorlog
import numpy as np
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import _LRScheduler


def rank_zero_only(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            s = traceback.extract_stack()
            filename, lineno, name, line = s[-2]
            args = list(args)
            args[0] = f'[{filename}:{lineno}] - {args[0]}'
            args = tuple(args)
            return fn(*args, **kwargs)

    return wrapped_fn


# TODO: this should be part of the cluster environment
def _get_rank() -> int:
    rank_keys = ('RANK', 'SLURM_PROCID', 'LOCAL_RANK')
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, 'rank', _get_rank())


def get_logger(name=__name__,
               level=logging.INFO,
               rank_zero=True) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = (
        '%(cyan)s[%(asctime)s%(reset)s] [%(green)s%(levelname)-4s%(reset)s] %(message)s'
    )
    sh = colorlog.StreamHandler()
    sh.setFormatter(colorlog.ColoredFormatter(formatter))

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    if rank_zero:
        for level in (
                'debug',
                'info',
                'warning',
                'error',
                'exception',
                'fatal',
                'critical',
        ):
            setattr(logger, level, rank_zero_only(getattr(logger, level)))
            setattr(sh, level, rank_zero_only(getattr(logger, level)))

    logger.addHandler(sh)
    sh.close()
    return logger


def _module_available(module_path: str) -> bool:
    """
    Check if a path is available in your environment

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False


class TorchTensorEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, torch.Tensor):
            olist = o.tolist()
            if 'bool' not in o.type().lower() and all(
                    map(lambda d: d == 0 or d == 1, olist)):
                print(
                    'Every element in %s is either 0 or 1. '
                    'You might consider convert it into bool.',
                    olist,
                )
            return olist
        return super().default(o)


def save_arch_to_json(mask: dict, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(mask, f, indent=4, sort_keys=True, cls=TorchTensorEncoder)


def load_json(filename):
    if filename is None:
        data = None
    elif isinstance(filename, str):
        with open(filename, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            data[key] = torch.tensor(value)
    elif isinstance(filename, dict):
        data = filename
    else:
        raise 'Wrong argument value for %s in `load_json` function' % filename
    return data


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get('ignore_warnings'):
        log.info('Disabling python warnings! <config.ignore_warnings=True>')
        warnings.filterwarnings('ignore')

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get('debug'):
        log.info('Running in debug mode! <config.debug=True>')
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get('fast_dev_run'):
        log.info(
            'Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>'
        )
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get('gpus') > 1:
            config.trainer.gpus = 1
        if config.datamodule.get('pin_memory'):
            config.datamodule.pin_memory = False
        if config.datamodule.get('num_workers'):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        'trainer',
        'model',
        'datamodule',
        'callbacks',
        'logger',
        'seed',
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = 'dim'
    tree = rich.tree.Tree(':gear: CONFIG', style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))

    with open('config_tree.txt', 'w') as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams['trainer'] = config['trainer']
    hparams['model'] = config['model']
    hparams['datamodule'] = config['datamodule']
    if 'seed' in config:
        hparams['seed'] = config['seed']
    if 'callbacks' in config:
        hparams['callbacks'] = config['callbacks']

    # save number of model parameters
    hparams['model/params_total'] = sum(p.numel() for p in model.parameters())
    hparams['model/params_trainable'] = sum(p.numel()
                                            for p in model.parameters()
                                            if p.requires_grad)
    hparams['model/params_not_trainable'] = sum(p.numel()
                                                for p in model.parameters()
                                                if not p.requires_grad)

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def hparams_wrapper(cls):
    """Obtain the input arguments and values of __init__ func of a class

    Example:
    >>> @hparams_wrapper
        class A:
            def __init__(self, a, b, c=2, d=4):
                print(self.hparams)
    >>> a = A(2,4,5,8)
    >>> output: {'c': 5, 'd': 8, 'a': 2, 'b': 4}
    """
    origin__new__ = cls.__new__

    def __new__(cls, *args, **kwargs):
        signature = inspect.signature(cls.__init__)
        _hparams = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        _args_name = inspect.getfullargspec(cls.__init__).args[1:]
        for i, arg in enumerate(args):
            _hparams[_args_name[i]] = arg
        _hparams.update(kwargs)
        self = origin__new__(cls)
        self._hparams = _hparams
        for key, value in self._hparams.items():
            try:
                setattr(self, key, value)
            except Exception as e:
                print(f'Error occurs when setting value for {key} due to {e}')
        cls.hparams = property(lambda self: self._hparams
                               )  # generate a `hparams` property function
        return self

    cls.__new__ = __new__
    return cls


class FindLR(_LRScheduler):
    """exponentially increasing learning rate
    Args:
        optimizer: optimzier(e.g. SGD)
        num_iter: totoal_iters
        max_lr: maximum  learning rate
    """
    def __init__(self, optimizer, max_lr=10, num_iter=100, last_epoch=-1):

        self.total_iters = num_iter
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        return [
            base_lr * (self.max_lr / base_lr)**(self.last_epoch /
                                                (self.total_iters + 1e-32))
            for base_lr in self.base_lrs
        ]


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]
