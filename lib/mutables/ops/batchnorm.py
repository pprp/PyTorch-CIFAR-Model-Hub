
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from lib.mutables.spaces import ValueSpace

from .base_module import FinegrainedModule
from .utils import is_searchable


__all__ = [
    'BaseBatchNorm',
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d'
]


class BaseBatchNorm(_BatchNorm, FinegrainedModule):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        FinegrainedModule.__init__(self)
        _BatchNorm.__init__(
            self, self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats)
        self.is_search = is_searchable(getattr(self.value_spaces, 'num_features', None))

    ###########################################
    # property
    ###########################################

    @property
    def params(self):
        '''The number of the trainable parameters'''
        # bn
        bn_weight = self.weight
        bn_bias = self.bias
        if 'num_features' in self.value_spaces:
            num_features = self.value_spaces['num_features'].value
            bn_weight = bn_weight[:num_features]
            bn_bias = bn_bias[:num_features]
        size = sum([p.numel() for p in [bn_weight, bn_bias] if p is not None])
        return size

    def forward(self, x):
        out = None
        if not self.is_search:
            out = _BatchNorm.forward(self, x)
        else:
            out = self.forward_bn(x)
        return out

    def forward_bn(self, x):
        num_features = getattr(self.value_spaces, 'num_features', self.num_features)
        if isinstance(num_features, ValueSpace):
            num_features = num_features.value
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        return F.batch_norm(
            x, self.running_mean[:num_features], self.running_var[:num_features], self.weight[:num_features],
            self.bias[:num_features], self.training or not self.track_running_stats,
            exponential_average_factor, self.eps,
        )

    def sort_weight_bias(self, module):
        vc = self.value_spaces['num_features']
        module.weight.data = torch.index_select(module.weight.data, 0, vc.sortIdx)
        module.bias.data = torch.index_select(module.bias.data, 0, vc.sortIdx)
        if type(module) in [nn.BatchNorm1d, nn.BatchNorm2d]:
            module.running_mean.data = torch.index_select(module.running_mean.data, 0, idx)
            module.running_var.data = torch.index_select(module.running_var.data, 0, idx)


class BatchNorm1d(BaseBatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )

class BatchNorm2d(BaseBatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

class BatchNorm3d(BaseBatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
