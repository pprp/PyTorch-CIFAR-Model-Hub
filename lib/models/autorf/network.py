import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .basemodel import *
from .genotypes import Genotype

# from space.operations import *
from .operations import *
from .spaces import PRIMITIVES


class Network(nn.Module):
    def __init__(self, num_classes, genotype):
        super(Network, self).__init__()
        self._num_classes = num_classes
        self.genotype = genotype
        model = rf_resnet20(num_classes=self._num_classes, genotype=self.genotype)
        self.model = model

    def forward(self, x):
        return self.model(x)
