from collections import namedtuple

import torch
import torch.nn as nn

from ..attention_structure import *  # noqa: F401, F403
from ..operations import *  # noqa: F401, F403

Genotype = namedtuple('Genotype', 'normal normal_concat')


class CifarRFBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, step, genotype):
        super(CifarRFBasicBlock, self).__init__()
        self._steps = step
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.genotype = genotype
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

        self.attention = ReceptiveFieldAttention(planes,
                                                 genotype=self.genotype)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        out = out + residual
        out = self.relu(out)

        return out


class CifarAttentionBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, step, genotype):
        super(CifarAttentionBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.genotype = genotype
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride
        self._step = step
        # print(f"line 158: planes: {planes}")
        self.attention = LAAttention(self._step, planes, self.genotype)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        out = residual + out
        out = self.relu(out)
        return out


class CifarAttentionResNet(nn.Module):
    def __init__(self, block, n_size, num_classes, genotype):
        super(CifarAttentionResNet, self).__init__()
        self.inplane = 16
        self.genotype = genotype
        self.channel_in = 16
        self.conv1 = nn.Conv2d(3,
                               self.inplane,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self._step = 4
        self.layer1 = self._make_layer(
            block,
            self.channel_in,
            blocks=n_size,
            stride=1,
            step=self._step,
            genotype=self.genotype,
        )
        self.layer2 = self._make_layer(
            block,
            self.channel_in * 2,
            blocks=n_size,
            stride=2,
            step=self._step,
            genotype=self.genotype,
        )
        self.layer3 = self._make_layer(
            block,
            self.channel_in * 4,
            blocks=n_size,
            stride=2,
            step=self._step,
            genotype=self.genotype,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.channel_in * 4, num_classes)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, step, genotype):
        strides = [stride] + [1] * (blocks - 1)
        self.layers = nn.ModuleList()
        for stride in strides:
            Block = block(self.inplane, planes, stride, step, genotype)
            self.layers += [Block]
            self.inplane = planes
        return self.layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for i, layer in enumerate(self.layer1):
            x = layer(x)
        for i, layer in enumerate(self.layer2):
            x = layer(x)
        for i, layer in enumerate(self.layer3):
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def rf_resnet20(**kwargs):
    model = CifarAttentionResNet(CifarRFBasicBlock, 3, **kwargs)
    return model


def attention_resnet20(num_classes, genotype, **kwargs):
    """Constructs a ResNet-20 model."""
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 3, num_classes,
                                 genotype, **kwargs)
    return model


def attention_resnet32(**kwargs):
    """Constructs a ResNet-32 model."""
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 5, **kwargs)
    return model


if __name__ == '__main__':
    g = Genotype(
        normal=[
            ('max_pool_3x3', 0),
            ('max_pool_3x3', 0),
            ('noise', 1),
            ('avg_pool_5x5', 0),
            ('noise', 1),
            ('noise', 2),
        ],
        normal_concat=range(0, 4),
    )

    m = CifarRFBasicBlock(16, 8, 1, 3, g)
    i = torch.zeros(3, 16, 32, 32)
    o = m(i)
    print(o.shape)
