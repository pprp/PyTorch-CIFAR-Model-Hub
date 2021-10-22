"""Toy in PyTorch."""
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules import padding

__all__ = ["ToyNet"]


class ToyNet(nn.Module):
    def __init__(self, scale=1, kernel=5, num_classes=10):
        super(ToyNet, self).__init__()

        if kernel == 5:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 6 * scale, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(6 * scale, 16 * scale, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 6 * scale, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(6 * scale, 16 * scale, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.fc1 = nn.Linear(16 * scale * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.stem(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


if __name__ == "__main__":
    input = torch.zeros(5, 3, 32, 32)

    model = ToyNet(scale=1, kernel=3, num_classes=10)

    output = model(input)

    print(output.shape)

    from torchsummary import summary

    summary(model, input_size=(3, 32, 32), batch_size=-1)
