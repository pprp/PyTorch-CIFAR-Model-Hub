"""Toy in PyTorch."""
import torch
import torch.nn as nn

from .registry import register_model

__all__ = ['ToyNet']


class CBR(nn.Module):
    def __init__(self, inplane, outplane, kernel):
        super(CBR, self).__init__()
        inplane = inplane
        outplane = outplane

        self.stem = nn.Sequential(
            nn.Conv2d(inplane,
                      outplane,
                      kernel_size=kernel,
                      padding=kernel // 2),
            nn.BatchNorm2d(outplane),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.stem(x)


class ToyNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ToyNet, self).__init__()
        base_c = 6

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_c, 3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(),
            CBR(base_c, base_c * 2, kernel=3),
            nn.MaxPool2d(2, stride=2),
            CBR(base_c * 2, base_c * 4, kernel=3),
            CBR(base_c * 4, base_c * 4, kernel=3),
            nn.MaxPool2d(2, stride=2),
            CBR(base_c * 4, base_c * 8, kernel=3),
            CBR(base_c * 8, base_c * 16, kernel=3),
            CBR(base_c * 16, base_c * 32, kernel=3),
            nn.MaxPool2d(2, stride=2),
            CBR(base_c * 32, base_c * 64, kernel=3),
            CBR(base_c * 64, base_c * 64, kernel=3),
            CBR(base_c * 64, base_c * 64, kernel=3),
            CBR(base_c * 64, base_c * 64, kernel=3),
            nn.MaxPool2d(2, stride=2),
            CBR(base_c * 64, base_c * 128, kernel=3),
            CBR(base_c * 128, base_c * 128, kernel=3),
            CBR(base_c * 128, base_c * 128, kernel=3),
            CBR(base_c * 128, base_c * 128, kernel=3),
            CBR(base_c * 128, base_c * 128, kernel=3),
            nn.MaxPool2d(2, stride=2),
            CBR(base_c * 128, base_c * 128, kernel=3),
            CBR(base_c * 128, base_c * 128, kernel=3),
            CBR(base_c * 128, base_c * 128, kernel=3),
            CBR(base_c * 128, base_c * 128, kernel=3),
            CBR(base_c * 128, base_c * 128, kernel=3),
            CBR(base_c * 128, base_c * 128, kernel=3),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(base_c * 128, num_classes)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.stem(x)
        # out = out.view(out.size(0), -1)
        out = self.gap(out)
        out = out.view(out.size(0), -1)

        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        return self.fc1(out)


@register_model
def ToyNet_S(**kwargs):
    return ToyNet(**kwargs)


if __name__ == '__main__':
    input = torch.zeros(5, 3, 32, 32)

    model = ToyNet(scale=1, kernel=3, num_classes=10)

    output = model(input)

    print(output.shape)

    from torchsummary import summary

    summary(model, input_size=(3, 32, 32), batch_size=-1)
