import torch
import torch.nn as nn

from lib.mutables.spaces import OperationSpace, ValueSpace
from lib.utils.utils import load_json

from lib.models.base_nas_network import BaseNASNetwork
from lib.models.nas.spos.ops import ShuffleBlock, Shuffle_Xception
from lib.models.registry import register_model


__all__ = [
    'ShuffleNASNetV2'
]

class ShuffleNASNetV2(BaseNASNetwork):

    def __init__(
        self,
        stage_repeats: list=[4, 4, 8, 4],
        stage_out_channels: list=[-1, 16, 64, 160, 320, 640, 1024],
        n_class: int=10,
        mask=None
    ):
        super(ShuffleNASNetV2, self).__init__()
        self.mask = load_json(mask)

        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, affine=False),
            nn.ReLU(inplace=True),
        )

        self.features = torch.nn.ModuleList()
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                base_mid_channels = outp // 2
                mid_channels = int(base_mid_channels)

                key = f"stage{idxstage}_block{i}"
                candidate_ops = OperationSpace([
                    ShuffleBlock(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride),
                    ShuffleBlock(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride),
                    ShuffleBlock(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride),
                    Shuffle_Xception(inp, outp, mid_channels=mid_channels, stride=stride)
                    ], key=key, mask=self.mask
                )
                self.features.append(candidate_ops)
                input_channel = output_channel
        # self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                input_channel, self.stage_out_channels[
                    -1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1], affine=False),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)

        for idx, archs in enumerate(self.features):
            x = archs(x)

        x = self.conv_last(x)

        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

@register_model 
def shufflenetnasnet(num_classes=10, **kwargs):
    return ShuffleNASNetV2(n_class=num_classes, **kwargs)

if __name__ == "__main__":
    model = ShuffleNASNetV2()
    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())
