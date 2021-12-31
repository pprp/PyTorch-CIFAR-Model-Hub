from typing import Optional, Union

import torch
import torch.nn as nn

from ..spaces import Mutable

def is_searchable(
    obj: Optional[Union[None, Mutable]]
    ):
    '''Check whether the Space obj is searchable'''
    if (obj is None) or (not obj.is_search):
        return False
    return True

def sub_filter_start_end(kernel_size, sub_kernel_size):
    if isinstance(kernel_size, (list, tuple)):
        kernel_size = kernel_size[0]
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end
