import copy
import datetime
import random
import time
from collections import namedtuple
from functools import singledispatch

import numpy as np
import torch


def build_expname(args):
    # model and dataset
    args.name = 'e-%s_m-%s_d-%s__' % (args.name, args.model, args.dataset)
    # running time
    args.name += datetime.datetime.now().strftime('%mM_%dD_%HH')
    # random number
    args.name += '__{:02d}'.format(random.randint(0, 99))
    return args.name


class Timer:
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.perf_counter()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.synch()
        self.times.append(time.perf_counter())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t


## data preprocessing
def preprocess(dataset, transforms):
    dataset = copy.copy(dataset)  # shallow copy
    for transform in transforms:
        dataset['data'] = transform(dataset['data'])
    return dataset


# 泛函数，根据类型分发
@singledispatch
def normalise(x, mean, std):
    return (x - mean) / std


@normalise.register(np.ndarray)
def _(x, mean, std):
    # faster inplace for numpy arrays
    x = np.array(x, np.float32)
    x -= mean
    x *= 1.0 / std
    return x


@singledispatch
def pad(x, border):
    raise NotImplementedError


@pad.register(np.ndarray)
def _(x, border):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)],
                  mode='reflect')


@singledispatch
def transpose(x, source, target):
    raise NotImplementedError


@transpose.register(np.ndarray)
def _(x, source, target):
    return x.transpose([source.index(d) for d in target])


## data augmentation
class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[..., y0:y0 + self.h, x0:x0 + self.w]

    def options(self, shape):
        *_, H, W = shape
        return [{
            'x0': x0,
            'y0': y0
        } for x0 in range(W + 1 - self.w) for y0 in range(H + 1 - self.h)]

    def output_shape(self, shape):
        *_, H, W = shape
        return (*_, self.h, self.w)


@singledispatch
def flip_lr(x):
    raise NotImplementedError


@flip_lr.register(np.ndarray)
def _(x):
    return x[..., ::-1].copy()


@flip_lr.register(torch.Tensor)
def _(x):
    return torch.flip(x, [-1])


class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return flip_lr(x) if choice else x

    def options(self, shape):
        return [{'choice': b} for b in [True, False]]
