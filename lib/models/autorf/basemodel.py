from utils.utils import drop_path
from torchvision.models import ResNet
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch.autograd import Variable
from .spaces import OPS
from .operations import *
import torch.nn as nn
import torch
import os
import pdb
import sys
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


Genotype = namedtuple("Genotype", "normal normal_concat")


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    """3x3 convolution"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


def conv7x7(in_channels, out_channels, stride=1, padding=3, dilation=1):
    """7x7 convolution"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


class ReceptiveFieldAttention(nn.Module):
    def __init__(self, C, steps=3, reduction=False, se=False, genotype=None):
        super(ReceptiveFieldAttention, self).__init__()
        assert genotype is not None
        self._ops = nn.ModuleList()
        self._C = C
        self._steps = steps
        self._stride = 1
        self._se = se
        self.C_in = C

        self.genotype = genotype
        op_names, indices = zip(*self.genotype.normal)
        concat = genotype.normal_concat

        self.conv1x1 = nn.Conv2d(
            C * self._steps, C, kernel_size=1, stride=1, padding=0, bias=False)

        if self._se:
            self.se = SE(self.C_in, reduction=4)

        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](self.C_in, 1, True)
            self._ops += [op]

        self.indices = indices

    def forward(self, x):
        states = [x]
        offset = 0

        total_step = (1+self._steps) * self._steps // 2

        for i in range(total_step):
            h = states[self.indices[i]]
            ops = self._ops[i]
            s = ops(h)
            states.append(s)

        # concate all released nodes
        node_out = torch.cat(states[-self._steps:], dim=1)
        node_out = self.conv1x1(node_out)
        # shortcut
        node_out = node_out + x
        if self._se:
            node_out = self.se(node_out)

        return node_out


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
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

        self.attention = ReceptiveFieldAttention(
            planes, genotype=self.genotype)

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


class Attention(nn.Module):
    def __init__(self, step, C, genotype):
        super(Attention, self).__init__()
        self._steps = step
        self._C = C
        self._ops = nn.ModuleList()
        self.C_in = self._C // 4
        self.C_out = self._C
        self.width = 4
        self.se = SE(self.C_in, reduction=2)  # 8
        self.se2 = SE(self.C_in * 4, reduction=2)  # 8
        self.channel_back = nn.Sequential(
            nn.Conv2d(
                self.C_in * 5, self._C, kernel_size=1, padding=0, groups=1, bias=False
            ),
            nn.BatchNorm2d(self._C),
            nn.ReLU(inplace=False),
            nn.Conv2d(self._C, self._C, kernel_size=1,
                      padding=0, groups=1, bias=False),
            nn.BatchNorm2d(self._C),
        )
        self.genotype = genotype
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](self.C_in, 1, True)
            self._ops += [op]
        self.indices = indices

    def forward(self, x):
        states = [x]
        C_num = x.shape[1]
        length = C_num // 4
        spx = torch.split(x, length, 1)
        spx_sum = sum(spx)
        spx_sum = self.se(spx_sum)
        states[0] = spx[0]
        h01 = states[self.indices[0]]
        op01 = self._ops[0]
        h01_out = op01(h01)
        s = h01_out
        states += [s]

        states[0] = spx[1]
        h02 = states[self.indices[1]]
        h12 = states[self.indices[2]]
        op02 = self._ops[1]
        op12 = self._ops[2]
        h02_out = op02(h02)
        h12_out = op12(h12)
        s = h02_out + h12_out
        states += [s]

        states[0] = spx[2]
        h03 = states[self.indices[3]]
        h13 = states[self.indices[4]]
        h23 = states[self.indices[5]]
        op03 = self._ops[3]
        op13 = self._ops[4]
        op23 = self._ops[5]
        h03_out = op03(h03)
        h13_out = op13(h13)
        h23_out = op23(h23)
        s = h03_out + h13_out + h23_out
        states += [s]

        states[0] = spx[3]
        h04 = states[self.indices[6]]
        h14 = states[self.indices[7]]
        h24 = states[self.indices[8]]
        h34 = states[self.indices[9]]

        op04 = self._ops[6]
        op14 = self._ops[7]
        op24 = self._ops[8]
        op34 = self._ops[9]

        h04_out = op04(h04)
        h14_out = op14(h14)
        h24_out = op24(h24)
        h34_out = op34(h34)
        s = h04_out + h14_out + h24_out + h34_out
        states += [s]

        node_concat = torch.cat(states[-4:], dim=1)
        node_concat = torch.cat((node_concat, spx_sum), dim=1)
        attention_out = self.channel_back(node_concat) + x
        attention_out = self.se2(attention_out)
        return attention_out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


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
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride
        self._step = step
        # print(f"line 158: planes: {planes}")
        self.attention = Attention(self._step, planes, self.genotype)

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
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False
        )
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
    model = CifarAttentionResNet(
        CifarAttentionBasicBlock, 3, num_classes, genotype, **kwargs
    )
    return model


def attention_resnet32(**kwargs):
    """Constructs a ResNet-32 model."""
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 5, **kwargs)
    return model


if __name__ == "__main__":
    g = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0), (
        'noise', 1), ('avg_pool_5x5', 0), ('noise', 1), ('noise', 2)], normal_concat=range(0, 4))

    m = CifarRFBasicBlock(16, 8, 1, 3, g)
    i = torch.zeros(3, 16, 32, 32)
    o = m(i)
    print(o.shape)
