from __future__ import absolute_import

from .att_overfit.att_net import *
from .cnn.attention import *
from .cnn.cbam_resnext import cbam_resnext29_8x64d, cbam_resnext29_16x64d
from .cnn.densenet import *
from .cnn.dla import *
from .cnn.dpn import DPN26
from .cnn.googlenet import *
from .cnn.squeezenet import squeezenet
from .cnn.stochasticdepth import *
from .cnn.stochasticdepth import (stochastic_depth_resnet18,
                                  stochastic_depth_resnet34,
                                  stochastic_depth_resnet50,
                                  stochastic_depth_resnet101,
                                  stochastic_depth_resnet152)
from .cnn.vgg import *
from .cnn.wide_resnet import WideResNet
from .cnn.xception import *
from .cnn.xception import xception
from .dawnnet import resnet_dawn
from .cnn.efficientnetb0 import *
from .cnn.genet import ge_resnext29_8x64d, ge_resnext29_16x64d
from .cnn.inceptionv3 import *
from .cnn.inceptionv4 import *
from .cnn.lenet import *
from .cnn.mobilenet import *
from .cnn.mobilenetv2 import *
from .cnn.nasnet import *
from .cnn.pnasnet import *
from .cnn.preact_resnet import *
from .cnn.pyramidnet import pyramidnet164, pyramidnet272
from .cnn.regnet import *
from .cnn.resnet import *
from .cnn.resnet20 import *
from .cnn.resnext import *
from .cnn.rir import *
from .sample_resnet20 import *
from .cnn.senet import *
from .cnn.shake_shake import shake_resnet26_2x32d, shake_resnet26_2x64d
from .cnn.shufflenet import *
from .cnn.shufflenetv2 import *
from .cnn.sknet import sk_resnext29_16x32d, sk_resnext29_16x64d
from .spp_depth.spp_resnet import *
from .toymodel import ToyNet
from .vit.cct import *
from .vit.coatnet import *
from .vit.cotnet import *
from .vit.cvt import CvT
from .vit.mobile_vit import *
from .vit.poolformer import *
from .vit.swin_transformer import *
from .vit.vision_transformer import ViT

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
    "vision_transformer": ViT,
    "mobilevit_s": mobilevit_s,
    "mobilevit_xs": mobilevit_xs,
    "mobilevit_xxs": mobilevit_xxs,
    "coatnet_0": coatnet_0,
    "coatnet_1": coatnet_1,
    "coatnet_2": coatnet_2,
    "coatnet_3": coatnet_3,
    "coatnet_4": coatnet_4,
    "cvt": CvT,
    "swin_t": swin_t,
    "swin_s": swin_s,
    "swin_b": swin_b,
    "swin_l": swin_l,
    "poolformer_s12": poolformer_s12,
    "poolformer_s24": poolformer_s24,
    "poolformer_s36": poolformer_s36,
    "convit_tiny": convit_tiny,
    "convit_small": convit_small,
    "cct_2": cct_2_3x2_32,
    "cct_4": cct_4_3x2_32,
    "dawnnet": resnet_dawn,
    **spp_family,
    **att_family,
}


def show_available_models():
    """Displays available models"""
    print(list(__model_factory.keys()))


def build_model(name, num_classes=10):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError("Unknown model: {}. Must be one of {}".format(name, avai_models))
    return __model_factory[name](num_classes=num_classes)
