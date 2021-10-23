from __future__ import absolute_import

from .cbam_resnext import cbam_resnext29_8x64d, cbam_resnext29_16x64d
from .genet import ge_resnext29_8x64d, ge_resnext29_16x64d
from .shake_shake import shake_resnet26_2x32d, shake_resnet26_2x64d
from .sknet import sk_resnext29_16x32d, sk_resnext29_16x64d
from .squeezenet import squeezenet
from .stochasticdepth import (
    stochastic_depth_resnet18,
    stochastic_depth_resnet34,
    stochastic_depth_resnet50,
    stochastic_depth_resnet101,
    stochastic_depth_resnet152,
)
from .xception import xception

from .attention import *
from .densenet import *
from .dla import *
from .dpn import DPN26
from .efficientnetb0 import *
from .googlenet import *
from .inceptionv3 import *
from .inceptionv4 import *
from .lenet import *
from .mobilenet import *
from .mobilenetv2 import *
from .nasnet import *
from .pnasnet import *
from .preact_resnet import *
from .regnet import *
from .resnet import *
from .resnet20 import *
from .resnext import *
from .rir import *
from .sample_resnet20 import *
from .senet import *
from .shufflenet import *
from .shufflenetv2 import *
from .stochasticdepth import *
from .vgg import *
from .xception import *
from .toymodel import ToyNet
from .wide_resnet import WideResNet
from .pyramidnet import pyramidnet164, pyramidnet272

__model_factory = {
    "wideresnet": WideResNet,
    "resnet20": resnet20,
    "densenet": densenet_cifar,
    "senet": senet18_cifar,
    "googlenet": GoogLeNet,
    "dla": DLA,
    "shufflenet": ShuffleNetG2,
    "shufflenetv2": ShuffleNetV2,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "efficientnetb0": EfficientNetB0,
    "lenet": LeNet,
    "mobilenet": MobileNet,
    "mobilenetv2": MobileNetV2,
    "pnasnet": PNASNetB,
    "preact_resnet": PreActResNet18,
    "regnet": RegNetX_200MF,
    "resnext": ResNeXt29_2x64d,
    "vgg": vgg11,
    "attention56": attention56,
    "attention92": attention92,
    "inceptionv3": inceptionv3,
    "inceptionv4": inceptionv4,
    "inception_resnet_v2": inception_resnet_v2,
    "nasnet": nasnet,
    "rir": resnet_in_resnet,
    "squeezenet": squeezenet,
    "stochastic_depth_resnet18": stochastic_depth_resnet18,
    "stochastic_depth_resnet34": stochastic_depth_resnet34,
    "stochastic_depth_resnet50": stochastic_depth_resnet50,
    "stochastic_depth_resnet101": stochastic_depth_resnet101,
    "stochastic_depth_resnet152": stochastic_depth_resnet152,
    "xception": xception,
    "dpn": DPN26,
    "shake_resnet26_2x32d": shake_resnet26_2x32d,
    "shake_resnet26_2x64d": shake_resnet26_2x64d,
    "ge_resnext29_8x64d": ge_resnext29_8x64d,
    "ge_resnext29_16x64d": ge_resnext29_16x64d,
    "sk_resnext29_16x32d": sk_resnext29_16x32d,
    "sk_resnext29_16x64d": sk_resnext29_16x64d,
    "cbam_resnext29_16x64d": cbam_resnext29_16x64d,
    "cbam_resnext29_8x64d": cbam_resnext29_8x64d,
    "toynet": ToyNet,
    "pyramidnet272": pyramidnet272,
    "pyramidnet164": pyramidnet164,
}


def show_available_models():
    """Displays available models"""
    print(list(__model_factory.keys()))


def build_model(name, num_classes=10):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError("Unknown model: {}. Must be one of {}".format(name, avai_models))
    return __model_factory[name](num_classes=num_classes)
