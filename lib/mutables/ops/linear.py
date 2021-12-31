
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .base_module import FinegrainedModule
from .utils import is_searchable

__all__ = [
    'Linear'
]


class Linear(FinegrainedModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False
    ) -> None:
        super(Linear, self).__init__()
        linear_kwargs = {key: getattr(self, key, None) for key in ['in_features', 'out_features', 'bias']}
        self.init_ops(**linear_kwargs)
        self.is_search = self.isSearchLinear()

    def init_ops(self, in_features: int, out_features: int, bias: bool) -> None:
        '''Generate linear operation'''
        self.weight = Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def isSearchLinear(self):
        '''search flag
            search
            search_in_features
            search_out_features
        '''
        self.search_in_features = False
        self.search_out_features = False
        if all([not vs.is_search for vs in self.value_spaces.values()]):
            return False
        if  is_searchable(getattr(self.value_spaces, 'in_features', None)):
            self.search_in_features = True
        if  is_searchable(getattr(self.value_spaces, 'out_features', None)):
            self.search_out_features = True
        return True

    ###########################################
    # forward implementation
    # - forward_linear
    #   - get_active_weight_bias
    ###########################################

    def forward(self, x):
        out = None
        if not self.is_search:
            out = F.linear(x, self.weight, self.bias)
        else:
            out = self.forward_linear(x)
        return out

    def forward_linear(self, x):
        weight, bias = self.get_active_weight_bias()
        y = F.linear(x, weight, bias)
        return y

    def get_active_weight_bias(self):
        weight = self.weight.contiguous()
        bias = self.bias
        in_features = None
        out_features = None
        if self.search_in_features:
            in_features = self.value_spaces['in_features'].value
        if self.search_out_features:
            out_features = self.value_spaces['out_features'].value
            if self.bias is not None:
                bias = bias[:out_features]
        weight = weight[:out_features, :in_features]
        return weight, bias

    def sort_weight_bias(self, module):
        if self.search_in_features:
            vc = self.value_spaces['in_features']
            module.weight.data = torch.index_select(module.weight.data, 1, vc.sortIdx)
        if self.search_out_features:
            vc = self.value_spaces['out_features']
            module.weight.data = torch.index_select(module.weight.data, 0, vc.sortIdx)
            if self.bias:
                module.bias.data = torch.index_select(module.bias.data, 0, vc.sortIdx)

    ###########################################
    # property
    ###########################################

    @property
    def params(self):
        '''The number of the trainable parameters'''
        # linear
        weight = self.weight
        bias = self.bias
        in_features = None
        out_features = None

        if self.search_in_features:
            in_features = self.value_spaces['in_features'].value
        if self.search_out_features:
            out_features = self.value_spaces['out_features'].value
        weight = weight[:in_features, :out_features]
        if bias is not None: bias = bias[:out_features]
        parameters = [weight, bias]
        size = sum([p.numel() for p in parameters if p is not None])
        return size
