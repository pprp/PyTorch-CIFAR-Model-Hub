import torch.nn as nn
import torch.nn.functional as F
from models.att_overfit.cbam import *


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
        use_att=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (nn.BatchNorm2d(
            out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        cardinality,
        base_width,
        expansion,
        use_att=False,
    ):

        super(Bottleneck, self).__init__()
        width_ratio = out_channels / (expansion * 64.0)
        D = cardinality * int(base_width * width_ratio)

        self.relu = nn.ReLU(inplace=True)
        if use_att:
            self.cbam_module = CBAM(out_channels)
        else:
            self.cbam_module = None
        self.conv_reduce = nn.Conv2d(in_channels,
                                     D,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(
            D,
            D,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D,
                                     out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'shortcut_conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
            )
            self.shortcut.add_module('shortcut_bn',
                                     nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv_reduce.forward(x)
        out = self.relu(self.bn_reduce.forward(out))
        out = self.conv_conv.forward(out)
        out = self.relu(self.bn.forward(out))
        out = self.conv_expand.forward(out)
        out = self.bn_expand.forward(out)

        residual = self.shortcut.forward(x)

        if self.cbam_module is not None:
            out = self.cbam_module(out) + residual
        else:
            out += residual
        out = self.relu(out)
        return out


class CBAMResNeXt(nn.Module):
    def __init__(self,
                 cardinality,
                 depth,
                 num_classes,
                 base_width,
                 expansion=4,
                 use_att=False):
        super(CBAMResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.expansion = expansion
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [
            64,
            64 * self.expansion,
            128 * self.expansion,
            256 * self.expansion,
        ]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1',
                                  self.stages[0],
                                  self.stages[1],
                                  1,
                                  use_att=use_att)
        self.stage_2 = self.block('stage_2',
                                  self.stages[1],
                                  self.stages[2],
                                  2,
                                  use_att=use_att)
        self.stage_3 = self.block('stage_3',
                                  self.stages[2],
                                  self.stages[3],
                                  2,
                                  use_att=use_att)
        self.fc = nn.Linear(self.stages[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def block(self,
              name,
              in_channels,
              out_channels,
              pool_stride=2,
              use_att=False):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(
                    name_,
                    Bottleneck(
                        in_channels,
                        out_channels,
                        pool_stride,
                        self.cardinality,
                        self.base_width,
                        self.expansion,
                        use_att,
                    ),
                )
            else:
                block.add_module(
                    name_,
                    Bottleneck(
                        out_channels,
                        out_channels,
                        1,
                        self.cardinality,
                        self.base_width,
                        self.expansion,
                        use_att,
                    ),
                )
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.fc(x)


def cbam_resnext29_8x64d(num_classes):
    return CBAMResNeXt(cardinality=8,
                       depth=29,
                       num_classes=num_classes,
                       base_width=64,
                       use_att=True)


def cbam_resnext29_16x64d(num_classes):
    return CBAMResNeXt(cardinality=16,
                       depth=29,
                       num_classes=num_classes,
                       base_width=64,
                       use_att=True)


# New model to test attention
def cbam_resnext29_16x8d(num_classes):
    return CBAMResNeXt(cardinality=16,
                       depth=29,
                       num_classes=num_classes,
                       base_width=8,
                       use_att=True)


def cbam_resnext29_16x16d(num_classes):
    return CBAMResNeXt(cardinality=16,
                       depth=29,
                       num_classes=num_classes,
                       base_width=16,
                       use_att=True)


def cbam_resnext29_16x32d(num_classes):
    return CBAMResNeXt(cardinality=16,
                       depth=29,
                       num_classes=num_classes,
                       base_width=32,
                       use_att=True)


def norm_resnext29_16x8d(num_classes):
    return CBAMResNeXt(cardinality=16,
                       depth=29,
                       num_classes=num_classes,
                       base_width=8,
                       use_att=False)


def norm_resnext29_16x16d(num_classes):
    return CBAMResNeXt(cardinality=16,
                       depth=29,
                       num_classes=num_classes,
                       base_width=16,
                       use_att=False)


def norm_resnext29_16x32d(num_classes):
    return CBAMResNeXt(cardinality=16,
                       depth=29,
                       num_classes=num_classes,
                       base_width=32,
                       use_att=False)


def norm_resnext29_16x64d(num_classes):
    return CBAMResNeXt(cardinality=16,
                       depth=29,
                       num_classes=num_classes,
                       base_width=64,
                       use_att=False)


att_family = {
    'cbam_resnext29_16x8d': cbam_resnext29_16x8d,
    'cbam_resnext29_16x16d': cbam_resnext29_16x16d,
    'cbam_resnext29_16x32d': cbam_resnext29_16x32d,
    'cbam_resnext29_16x64d': cbam_resnext29_16x64d,
    'norm_resnext29_16x8d': norm_resnext29_16x8d,
    'norm_resnext29_16x16d': norm_resnext29_16x16d,
    'norm_resnext29_16x32d': norm_resnext29_16x32d,
    'norm_resnext29_16x64d': norm_resnext29_16x64d,
}

if __name__ == '__main__':
    m = norm_resnext29_16x16d(10)
    # m = cbam_resnext29_16x8d(10)
    print(m)
