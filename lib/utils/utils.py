import argparse
import math
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler


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
            base_lr
            * (self.max_lr / base_lr) ** (self.last_epoch / (self.total_iters + 1e-32))
            for base_lr in self.base_lrs
        ]


def str2bool(v):
    if v.lower() in ["true", 1]:
        return True
    elif v.lower() in ["false", 0]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RandomErase(object):
    def __init__(self, prob, sl, sh, r):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r = r

    def __call__(self, img):
        if random.uniform(0, 1) < self.prob:
            return img

        while True:
            area = random.uniform(self.sl, self.sh) * img.size[0] * img.size[1]
            ratio = random.uniform(self.r, 1 / self.r)

            h = int(round(math.sqrt(area * ratio)))
            w = int(round(math.sqrt(area / ratio)))

            if h < img.size[0] and w < img.size[1]:
                x = random.randint(0, img.size[0] - h)
                y = random.randint(0, img.size[1] - w)
                img = np.array(img)
                if len(img.shape) == 3:
                    for c in range(img.shape[2]):
                        img[x : x + h, y : y + w, c] = random.uniform(0, 1)
                else:
                    img[x : x + h, y : y + w] = random.uniform(0, 1)
                img = Image.fromarray(img)

                return img


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
            if hasattr(m, "weight"):
                no_decay.append(m.weight)
            if hasattr(m, "bias"):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]
