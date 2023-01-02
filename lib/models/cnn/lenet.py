'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model

__all__ = ['LeNet']


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, num_classes)

        # get the gradient results of the last layer
        self.gradient = None

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.classifier(out)
        out.register_hook(self.save_gradient)
        return out

    def save_gradient(self, grad):
        self.gradient = grad


@register_model
def lenet(num_classes=10):
    return LeNet(num_classes)
