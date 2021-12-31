
from typing import Union, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter

from ..spaces import ValueSpace
from .base_module import FinegrainedModule
from .utils import sub_filter_start_end, is_searchable


__all__ = [
    'BaseConvNd',
    'Conv1d',
    'Conv2d',
    'Conv3d'
]


class BaseConvNd(_ConvNd, FinegrainedModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple, ValueSpace],
        stride: Union[int, tuple, ValueSpace],
        padding: Union[str, int, tuple, ValueSpace],
        dilation: Union[int, tuple, ValueSpace],
        groups: Union[int, tuple, ValueSpace],
        bias: bool,
        padding_mode: str = 'zeros',
        auto_padding: bool = False,
        *args,
        **kwargs
    ):
        '''Base Conv Module
        Args:
            auto_padding: if set to true, will set a proper padding size to make output size same as the input size.
                For example, if kernel size is 3, the padding size is 1;
                if kernel_size is (3,7), the padding size is (1, 3)
        '''
        self.conv_dim = self.__class__.__name__[-2:]
        # first initialize by FinegrainedModule
        FinegrainedModule.__init__(self)
        conv_kwargs = {
            key: getattr(self, key, None) for key in [
                'in_channels', 'out_channels', 'kernel_size',
                'stride', 'padding', 'dilation', 'groups', 'bias'
                ]
        }
        self.init_ops(**conv_kwargs)
        # then initialized by _ConvNd
        _ConvNd.__init__(
            self, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
            self.dilation, False, self.output_padding, self.groups, True, self.padding_mode)
        if not bias:
            del self.bias
            self.register_parameter('bias', None)
        self.is_search = self.isSearchConv()

    def init_ops(self, *args, **kwargs):
        '''Generate Conv operation'''
        raise NotImplementedError

    def isSearchConv(self):
        '''Search flag
        Supported arguments
            - search_in_channel
            - search_out_channel
            - search_kernel_size
            - search_stride
            - search_dilation
            - search_groups
        '''
        self.search_in_channel = False
        self.search_out_channel = False
        self.search_kernel_size = False
        self.search_stride = False
        self.search_dilation = False
        self.search_groups = False
        # self.search_bias = False

        if all([not vs.is_search for vs in self.value_spaces.values()]):
            return False

        if  is_searchable(getattr(self.value_spaces, 'in_channels', None)):
            self.search_in_channel = True
        if  is_searchable(getattr(self.value_spaces, 'out_channels', None)):
            self.search_out_channel = True
        if  is_searchable(getattr(self.value_spaces, 'kernel_size', None)):
            kernel_candidates = self.value_spaces['kernel_size'].candidates
            max_k = self.kernel_size
            # Todo: 与`transform_kernel_size`搭配使用，目前未使用
            # for i, k in enumerate(sorted(kernel_candidates)[:-1]):
            #     self.register_parameter(f'{max_k}to{k}_kernelMatrix', Parameter(torch.rand(max_k**2, k**2)))
            self.search_kernel_size = True
        if  is_searchable(getattr(self.value_spaces, 'stride', None)):
            self.search_stride = True
        if  is_searchable(getattr(self.value_spaces, 'dilation', None)):
            self.search_dilation = True
        if  is_searchable(getattr(self.value_spaces, 'groups', None)):
            self.search_groups = True
        # if  is_searchable(getattr(self.value_spaces, 'bias', None)):
        #     self.search_bias = True

        return True

    ###########################################
    # forward implementation
    # - forward_conv
    #   - transform_kernel_size
    ###########################################

    def forward(self, x):
        out = None
        if not self.is_search:
            padding = self.padding
            if self.auto_padding:
                kernel_size = self.weight.shape[2:]
                padding = []
                for k in kernel_size:
                    padding.append(k//2)
            out = self.conv(x, self.weight, self.bias, self.stride,
                padding, self.dilation, self.groups)
        else:
            out = self.forward_conv(x)
        return out

    def forward_conv(self, x):
        filters = self.weight.contiguous()
        bias = self.bias
        in_channels = self.in_channels
        out_channels = self.out_channels
        stride = self.value_spaces['stride'].value if self.search_stride else self.stride
        groups = self.value_spaces['groups'].value if self.search_groups else self.groups
        dilation = self.value_spaces['dilation'].value if self.search_dilation else self.dilation
        padding = self.padding

        if self.search_in_channel:
            in_channels = self.value_spaces['in_channels'].value
            filters = filters[:, :in_channels, ...]
        if self.search_out_channel:
            out_channels = self.value_spaces['out_channels'].value
            if self.bias is not None:
                bias = bias[:out_channels]
            filters = filters[:out_channels, ...]
        if self.search_kernel_size:
            filters = self.transform_kernel_size(filters)
        if self.search_groups:
            filters = self.get_filters_by_groups(filters, in_channels, groups).contiguous()
        if self.auto_padding:
            kernel_size = filters.shape[2:]
            padding = []
            for k in kernel_size:
                padding.append(k//2)
        return self.conv(x, filters, bias, stride, padding, dilation, groups)

    def get_filters_by_groups(self, filters, in_channels, groups):
        '''Get filters when searching for #of groups'''
        sub_filters = torch.chunk(filters, groups, dim=0)
        sub_in_channels = in_channels // groups
        sub_ratio = filters.size(1) // sub_in_channels

        filter_crops = []
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start = part_id * sub_in_channels
            filter_crops.append(sub_filter[:, start:start + sub_in_channels, :, :])
        filters = torch.cat(filter_crops, dim=0)
        return filters

    def transform_kernel_size(self, filters):
        # Todo: support different types of kernel size transformation methods by `transform_kernel_size` function
        sub_kernel_size = self.value_spaces['kernel_size'].value
        start, end = sub_filter_start_end(self.kernel_size, sub_kernel_size)
        if self.conv_dim=='1d': filters = filters[:, :, start:end]
        if self.conv_dim=='2d': filters = filters[:, :, start:end, start:end]
        if self.conv_dim=='3d': filters = filters[:, :, start:end, start:end, start:end]
        return filters

    def sort_weight_bias(self, module):
        if self.search_in_channel:
            vc = self.value_spaces['in_channels']
            module.weight.data = torch.index_select(module.weight.data, 1, vc.sortIdx)
        if self.search_out_channel:
            vc = self.value_spaces['out_channels']
            module.weight.data = torch.index_select(module.weight.data, 0, vc.sortIdx)
            if self.bias is not None:
                module.bias.data = torch.index_select(module.bias.data, 0, vc.sortIdx)

    ###########################################
    # property
    ###########################################

    @property
    def params(self):
        '''The number of the trainable parameters'''
        # conv
        weight = self.weight
        bias = self.bias

        if self.search_in_channel:
            in_channels = self.value_spaces['in_channels'].value
            weight = weight[:, :in_channels, ...]
        if self.search_out_channel:
            out_channels = self.value_spaces['out_channels'].value
            weight = weight[:out_channels, :, ...]
            if bias is not None: bias = bias[:out_channels]
        if self.search_kernel_size:
            kernel_size = self.value_spaces['kernel_size'].value
            start, end = sub_filter_start_end(self.kernel_size, kernel_size)
            shape_size = len(weight.shape)
            if shape_size == 3:
                # 1D conv
                weight = weight[:, :, start:end]
            elif shape_size == 4:
                # 2D conv
                weight = weight[:, :, start:end, start:end]
            else:
                # 3D conv
                weight = weight[:, :, start:end, start:end, start:end]
        parameters = [weight, bias]
        params = sum([p.numel() for p in parameters if p is not None])
        return params


class Conv1d(BaseConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[_size_1_t, ValueSpace],
        stride: Union[_size_1_t, ValueSpace] = 1,
        padding: Union[str, _size_1_t, ValueSpace] = 0,
        dilation: Union[_size_1_t, ValueSpace] = 1,
        groups: Union[int, ValueSpace] = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        auto_padding: bool = False
    ):
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)

    def init_ops(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True
    ):
        '''Generate Conv operation'''
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = padding if isinstance(padding, str) else _single(padding)
        self.dilation = _single(dilation)
        self.output_padding = _single(0)
        self.conv = F.conv1d


class Conv2d(BaseConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[str, int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        auto_padding: bool = False
        ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)

    def init_ops(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        '''Generate Conv operation'''
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)
        self.output_padding = _pair(0)
        self.conv = F.conv2d


class Conv3d(BaseConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[str, int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        auto_padding: bool = False
    ):
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)

    def init_ops(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True
    ):
        '''Generate Conv operation'''
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = padding if isinstance(padding, str) else _triple(padding)
        self.dilation = _triple(dilation)
        self.output_padding = _triple(0)
        self.conv = F.conv3d
