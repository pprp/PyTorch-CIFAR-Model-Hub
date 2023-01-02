import torch
import torch.nn as nn
import torch.nn.functional as F
# from bam import *
# from cbam import *
from models.spp_depth.poolings import *

# from model.att_overfit.cbam import *

# from poolings import *


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
            kernel_list=(3, 5, 7),
    ):

        super(Bottleneck, self).__init__()
        width_ratio = out_channels / (expansion * 64.0)
        D = cardinality * int(base_width * width_ratio)

        self.relu = nn.ReLU(inplace=True)
        if use_att:
            self.spp_module = SPP(out_channels, out_channels, kernel_list)
        else:
            self.spp_module = None
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

        if self.spp_module is not None:
            out = self.spp_module(out) + residual
        else:
            out += residual
        out = self.relu(out)
        return out


class SPPResNeXt(nn.Module):
    def __init__(
        self,
        cardinality,
        depth,
        num_classes,
        base_width,
        expansion=4,
        use_att=False,
        kernel_list=[3, 5, 7],
    ):
        super(SPPResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.expansion = expansion
        self.num_classes = num_classes
        self.output_size = 64
        self.kernel_list = kernel_list
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
                        self.kernel_list,
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
                        self.kernel_list,
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


def build_spp_models(num_classes,
                     depth: int = 11,
                     spp: bool = True,
                     kernel_list: list = [3]):
    """
    depth: 11 20 29
    spp: [3] [5] [7] [3,5,7]
    """
    return SPPResNeXt(
        cardinality=16,
        depth=depth,
        num_classes=num_classes,
        base_width=8,
        use_att=spp,
        kernel_list=kernel_list,
    )


"""
N: without spp
A: [3]
B: [5]
C: [7]
D: [3,5,7]
"""


# depth = 11
def spp_d11_pN(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=11,
                            spp=False,
                            kernel_list=[])


def spp_d11_pA(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=11,
                            spp=False,
                            kernel_list=[3])


def spp_d11_pB(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=11,
                            spp=False,
                            kernel_list=[5])


def spp_d11_pC(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=11,
                            spp=False,
                            kernel_list=[7])


def spp_d11_pD(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=11,
                            spp=False,
                            kernel_list=[3, 5, 7])


# depth = 20
def spp_d20_pN(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=20,
                            spp=False,
                            kernel_list=[])


def spp_d20_pA(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=20,
                            spp=False,
                            kernel_list=[3])


def spp_d20_pB(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=20,
                            spp=False,
                            kernel_list=[5])


def spp_d20_pC(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=20,
                            spp=False,
                            kernel_list=[7])


def spp_d20_pD(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=20,
                            spp=False,
                            kernel_list=[3, 5, 7])


# depth = 29
def spp_d29_pN(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=29,
                            spp=False,
                            kernel_list=[])


def spp_d29_pA(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=29,
                            spp=False,
                            kernel_list=[3])


def spp_d29_pB(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=29,
                            spp=False,
                            kernel_list=[5])


def spp_d29_pC(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=29,
                            spp=False,
                            kernel_list=[7])


def spp_d29_pD(num_classes=10):
    return build_spp_models(num_classes=num_classes,
                            depth=29,
                            spp=False,
                            kernel_list=[3, 5, 7])


spp_family = {
    'spp_d11_pN': spp_d11_pN,
    'spp_d11_pA': spp_d11_pA,
    'spp_d11_pB': spp_d11_pB,
    'spp_d11_pC': spp_d11_pC,
    'spp_d11_pD': spp_d11_pD,
    'spp_d20_pN': spp_d20_pN,
    'spp_d20_pA': spp_d20_pA,
    'spp_d20_pB': spp_d20_pB,
    'spp_d20_pC': spp_d20_pC,
    'spp_d20_pD': spp_d20_pD,
    'spp_d29_pN': spp_d29_pN,
    'spp_d29_pA': spp_d29_pA,
    'spp_d29_pB': spp_d29_pB,
    'spp_d29_pC': spp_d29_pC,
    'spp_d29_pD': spp_d29_pD,
}

if __name__ == '__main__':
    # m = norm_resnext29_16x8d(10)
    m = spp_d11_pN(10)
    print(m)
