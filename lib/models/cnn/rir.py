"""resnet in resnet in pytorch



[1] Sasha Targ, Diogo Almeida, Kevin Lyman.

    Resnet in Resnet: Generalizing Residual Architectures
    https://arxiv.org/abs/1603.08029v1
"""

import torch
import torch.nn as nn

from ..registry import register_model

__all__ = ['rir']


#geralized
class ResnetInit(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        self.residual_stream_conv = nn.Conv2d(in_channel,
                                              out_channel,
                                              3,
                                              padding=1,
                                              stride=stride)

        self.transient_stream_conv = nn.Conv2d(in_channel,
                                               out_channel,
                                               3,
                                               padding=1,
                                               stride=stride)

        self.residual_stream_conv_across = nn.Conv2d(in_channel,
                                                     out_channel,
                                                     3,
                                                     padding=1,
                                                     stride=stride)

        self.transient_stream_conv_across = nn.Conv2d(in_channel,
                                                      out_channel,
                                                      3,
                                                      padding=1,
                                                      stride=stride)

        self.residual_bn_relu = nn.Sequential(nn.BatchNorm2d(out_channel),
                                              nn.ReLU(inplace=True))

        self.transient_bn_relu = nn.Sequential(nn.BatchNorm2d(out_channel),
                                               nn.ReLU(inplace=True))

        self.short_cut = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channel,
                          out_channel,
                          kernel_size=1,
                          stride=stride))

    def forward(self, x):
        x_residual, x_transient = x
        residual_r_r = self.residual_stream_conv(x_residual)
        residual_r_t = self.residual_stream_conv_across(x_residual)
        residual_shortcut = self.short_cut(x_residual)

        transient_t_t = self.transient_stream_conv(x_transient)
        transient_t_r = self.transient_stream_conv_across(x_transient)

        x_residual = self.residual_bn_relu(residual_r_r + transient_t_r +
                                           residual_shortcut)
        x_transient = self.transient_bn_relu(residual_r_t + transient_t_t)

        return x_residual, x_transient


class RiRBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 layer_num,
                 stride,
                 layer=ResnetInit):
        super().__init__()
        self.resnetinit = self._make_layers(in_channel, out_channel, layer_num,
                                            stride)

    def forward(self, x):
        x_residual, x_transient = self.resnetinit(x)
        return (x_residual, x_transient)

    def _make_layers(self,
                     in_channel,
                     out_channel,
                     layer_num,
                     stride,
                     layer=ResnetInit):
        strides = [stride] + [1] * (layer_num - 1)
        layers = nn.Sequential()
        for index, s in enumerate(strides):
            layers.add_module('generalized layers{}'.format(index),
                              layer(in_channel, out_channel, s))
            in_channel = out_channel

        return layers


class ResnetInResneet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = int(96 / 2)
        self.residual_pre_conv = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.BatchNorm2d(base),
            nn.ReLU(inplace=True))
        self.transient_pre_conv = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.BatchNorm2d(base),
            nn.ReLU(inplace=True))

        self.rir1 = RiRBlock(base, base, 2, 1)
        self.rir2 = RiRBlock(base, base, 2, 1)
        self.rir3 = RiRBlock(base, base * 2, 2, 2)
        self.rir4 = RiRBlock(base * 2, base * 2, 2, 1)
        self.rir5 = RiRBlock(base * 2, base * 2, 2, 1)
        self.rir6 = RiRBlock(base * 2, base * 4, 2, 2)
        self.rir7 = RiRBlock(base * 4, base * 4, 2, 1)
        self.rir8 = RiRBlock(base * 4, base * 4, 2, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                384, num_classes * 10, kernel_size=3,
                stride=2),  #without this convolution, loss will soon be nan
            nn.BatchNorm2d(num_classes * 10),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(900, 450),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(450, 100),
        )

        self._weight_init()

    def forward(self, x):
        x_residual = self.residual_pre_conv(x)
        x_transient = self.transient_pre_conv(x)

        x_residual, x_transient = self.rir1((x_residual, x_transient))
        x_residual, x_transient = self.rir2((x_residual, x_transient))
        x_residual, x_transient = self.rir3((x_residual, x_transient))
        x_residual, x_transient = self.rir4((x_residual, x_transient))
        x_residual, x_transient = self.rir5((x_residual, x_transient))
        x_residual, x_transient = self.rir6((x_residual, x_transient))
        x_residual, x_transient = self.rir7((x_residual, x_transient))
        x_residual, x_transient = self.rir8((x_residual, x_transient))
        h = torch.cat([x_residual, x_transient], 1)
        h = self.conv1(h)
        h = h.view(h.size()[0], -1)
        h = self.classifier(h)

        return h

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)


@register_model
def rir(num_classes=10):
    return ResnetInResneet(num_classes=num_classes)
