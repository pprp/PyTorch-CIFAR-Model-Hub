# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.mutables.spaces import InputSpace, OperationSpace, ValueSpace

from .default_mutator import Mutator
from .darts_mutator import DartsMutator

__all__ = [
    'OnehotMutator',
]


class OnehotMutator(DartsMutator):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def sample_search(self):
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, OperationSpace):
                result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1)
                mutable.mask = torch.zeros_like(result[mutable.key])
                mutable.mask[result[mutable.key].cpu().detach().numpy().argmax()] = 1
            elif isinstance(mutable, ValueSpace):
                result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1)
                mutable.mask.data = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1).data
            elif isinstance(mutable, InputSpace):
                result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1)
                mutable.mask = torch.zeros_like(result[mutable.key])
                mutable.mask[result[mutable.key].cpu().detach().numpy().argmax()] = 1
        return result

    def sample_final(self):
        super().sample_final()
