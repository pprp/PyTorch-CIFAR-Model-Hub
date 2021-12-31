from typing import Union, Optional
from collections import OrderedDict

import torch
import torch.nn as nn

from lib.utils.utils import hparams_wrapper
from hyperbox.networks.pytorch_modules import Hsigmoid 
from hyperbox.networks.utils import build_activation, make_divisible

from lib.mutables import ops
from lib.mutables.spaces import ValueSpace

__all__ = [
    'Base2DLayer', 'SELayer', 'MBConvLayer',
]


@hparams_wrapper
class Base2DLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Base2DLayer, self).__init__()
        # The decorator @hparams_wrapper can automatically save all input arguments to
        # ``hparams`` attribute
        self.value_spaces = self.getValueSpaces(self.hparams)

    def getValueSpaces(self, kwargs):
        value_spaces = nn.ModuleDict()
        for key, value in kwargs.items():
            if isinstance(value, ValueSpace):
                value_spaces[key] = value
                if value.index is not None:
                    _v = value.candidates[value.index]
                elif len(value.mask) != 0:
                    if isinstance(value.mask, torch.Tensor):
                        index = value.mask.clone().detach().argmax()
                    else:
                        index = torch.tensor(value.mask).argmax()
                    _v = value.candidates[index]
                else:
                    _v = value.max_value
                setattr(self, key, _v)
            else:
                setattr(self, key, value)
        return value_spaces


class SELayer(Base2DLayer):
    REDUCTION = 4
    CHANNEL_DIVISIBLE = 8

    def __init__(
        self,
        channel: Union[int, ValueSpace],
        reduction=None,
        prefix=None
    ):
        super(SELayer, self).__init__()

        self.reduction = SELayer.REDUCTION if reduction is None else reduction

        if isinstance(channel, ValueSpace):
            num_mid = []
            for c in channel.candidates:
                num_mid.append(
                    make_divisible(c // self.reduction, divisor=self.CHANNEL_DIVISIBLE)
                )
            if prefix is None:
                key = channel.key + '_subSE_mc'
            else:
                key = prefix + '_subSE_mc'
            num_mid = ValueSpace(num_mid, key=key)
        else:
            num_mid = make_divisible(channel // self.reduction, divisor=self.CHANNEL_DIVISIBLE)

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', ops.Conv2d(channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', ops.Conv2d(num_mid, channel, 1, 1, 0, bias=True)),
            ('h_sigmoid', Hsigmoid(inplace=True)),
        ]))

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y


class MBConvLayer(Base2DLayer):
    CHANNEL_DIVISIBLE = 8

    def __init__(
        self,
        in_channels: Union[int, ValueSpace],
        out_channels: Union[int, ValueSpace],
        kernel_size: Union[int, ValueSpace] = 3,
        stride: Union[int, ValueSpace] = 1,
        groups: Union[int, ValueSpace] = 1,
        expand_ratio: Union[int, ValueSpace] = 6,
        act_func: str = 'relu6',
        use_se: bool = False,
        prefix=None, # prefix for key of ValueSpace
    ):
        flag = isinstance(in_channels, ValueSpace) and isinstance(expand_ratio, ValueSpace)
        assert not flag, "in_channels and expand_ratio cannot both be ValueSpace"
        super(MBConvLayer, self).__init__()

        # build modules
        if isinstance(in_channels, ValueSpace):
            middle_channels = []
            for c in in_channels.candidates:
                middle_channels.append(
                    make_divisible(round(self.in_channels * self.expand_ratio), self.CHANNEL_DIVISIBLE)
                )
            if prefix is None:
                key = in_channels.key + '_subMB_mc'
            else:
                key = prefix + '_subMB_mc'
            middle_channels = ValueSpace(middle_channels, key=key)
        elif isinstance(expand_ratio, ValueSpace):
            middle_channels = []
            for e in expand_ratio.candidates:
                middle_channels.append(
                    make_divisible(round(self.in_channels * e), self.CHANNEL_DIVISIBLE)
                )
            if prefix is None:
                key = expand_ratio.key + '_subMB_mc'
            else:
                key = prefix + '_subMB_mc'
            middle_channels = ValueSpace(middle_channels, key=key)
        else:
            middle_channels = make_divisible(
                round(self.in_channels * self.expand_ratio), self.CHANNEL_DIVISIBLE)
        if (isinstance(expand_ratio, ValueSpace) and expand_ratio.max_value==1) or self.expand_ratio== 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', ops.Conv2d(in_channels, middle_channels, kernel_size=1, stride=1, bias=False)),
                ('bn', ops.BatchNorm2d(middle_channels)),
                ('act', build_activation(self.act_func)),
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', ops.Conv2d(middle_channels, middle_channels, kernel_size=kernel_size, stride=stride, groups=middle_channels, auto_padding=True, bias=False)),
            ('bn', ops.BatchNorm2d(middle_channels)),
            ('act', build_activation(self.act_func))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', SELayer(middle_channels))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', ops.Conv2d(middle_channels, out_channels, kernel_size=1, stride=1, bias=False)),
            ('bn', ops.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x