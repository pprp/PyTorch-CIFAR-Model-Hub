import torch
import torch.nn as nn


__all__ = [
    '__MASKERS__',
    'BaseMasker',
    'L1Masker'
]


class BaseMasker:
    def __init__(self,):
        pass

    def get_channel_sortedIdx(self, module):
        raise NotImplementedError

    def __call__(self, module, is_in_feat=True):
        return self.get_channel_sortedIdx(module, is_in_feat)


class L1Masker(BaseMasker):
    def get_channel_sortedIdx(self, module, is_in_feat=True):
        num_dim = len(module.weight.shape)
        dim = list(range(num_dim))
        if num_dim > 1: # conv or linear
            if is_in_feat:
                del dim[1] # [0, 2, 3] for conv2d, [0, 2, 3, 4] for conv3, [0] for linear
            else:
                del dim[0] # [1, 2, 3]
            importance = torch.sum(torch.abs(module.weight.data), dim=dim)
        else: # BN
            importance = torch.abs(module.weight.data)
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        return sorted_idx


__MASKERS__ = {
    'l1': L1Masker,
}
