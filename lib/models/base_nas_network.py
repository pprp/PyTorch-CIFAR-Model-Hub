from typing import Optional, Union, Tuple, List
import torch.nn as nn

from lib.mutables import spaces, ops
from lib.utils.utils import load_json, hparams_wrapper
from hyperbox.utils.calc_model_size import flops_size_counter


@hparams_wrapper
class BaseNASNetwork(nn.Module):
    def __init__(self, mask: Optional[Union[str, dict]]=None):
        super(BaseNASNetwork, self).__init__()
        self._mask = None
        if mask is None or mask == '':
            self.is_search=True
        elif isinstance(mask, str):
            self._mask_file = mask
            self._mask = load_json(mask)
            self.is_search = False
        elif isinstance(mask, dict):
            self._mask = mask
            self.is_search = False
        for key, value in self.hparams.items():
            if key != 'mask':
                setattr(self, key, value)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, new_mask):
        self._mask = new_mask

    @property
    def arch(self):
        '''return the current arch encoding'''
        arch = ''
        for name, module in self.named_modules():
            if isinstance(module, spaces.Mutable):
                key = module.key
                mask = module.mask
                arch += f"{name}-{key}:{mask}\n"
        return arch

    def arch_size(
        self,
        datasize: Optional[Union[Tuple, List]] = None,
        convert: bool = True,
        verbose: bool = False
    ):
        size = datasize
        assert size is not None, \
            "Please specify valid data size, e.g., size=self.arch_size(datasize=(1,3,32,32))"
        result = flops_size_counter(self, size, convert, verbose)
        mflops, mb_size = list(result.values())
        return mflops, mb_size

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            if 'total_ops' in key or \
                'total_params' in key or \
                'module_used' in key:
                continue
            else:
                model_dict[key] = state_dict[key]
        super(BaseNASNetwork, self).load_state_dict(model_dict, **kwargs)

    def build_subnet(self, mask, preserve_weight=True):
        '''build subnet by the given mask'''
        hparams = self.hparams
        hparams['mask'] = mask
        new_cls = self.__class__(**hparams)
        if preserve_weight:
            new_cls.load_from_supernet(self.state_dict())
        return new_cls

    def copy(self, mask=None):
        if mask is None and self.mask is not None: 
            mask = self.mask 
        new_net = self.build_subnet(mask, False)
        new_net.load_state_dict(self.state_dict())
        return new_net

    def get_module_by_name(self, name):
        def is_int(item):
            try:
                item = int(item)
                return True
            except:
                return False
        full_name = 'self'
        for item in name.split('.'):
            if is_int(item):
                full_name += f"[{item}]"
            else:
                full_name += f".{item}"
        return eval(full_name)

    def load_from_supernet(self, state_dict, **kwargs):
        '''load subnet state dict from the given state_dict'''
        def sub_filter_start_end(kernel_size, sub_kernel_size):
            center = kernel_size // 2
            dev = sub_kernel_size // 2
            start, end = center - dev, center + dev + 1
            assert end - start == sub_kernel_size
            return start, end

        model_dict = self.state_dict()
        for key in model_dict:
            if 'total_ops' in key or \
                'total_params' in key or \
                'module_used' in key or \
                'mask' in key:
                continue
            if '.candidates' in key:
                # OperationSpace
                name = ''.join(key.split('.candidates')[0])
                module = self.get_module_by_name(name)
                if isinstance(module, spaces.OperationSpace):
                    cand_indices = {}
                    for idx, cand in enumerate(module.candidates_original):
                        cand_indices[cand.__class__.__name__] = idx
                    index = module.index
                    if index is None:
                        cand_index = int(key.split('.candidates.')[1].split('.')[0])
                        index = cand_indices[module.candidates[cand_index].__class__.__name__]
                    prefix, suffix = key.split('.candidates.')
                    prefix += '.candidates'
                    suffix = '.'.join(suffix.split('.')[1:])
                    fullname = f"{prefix}.{index}.{suffix}"
                    model_dict[key] = state_dict[fullname]
            else:
                name = '.'.join(key.split('.')[:-1])
                module = self.get_module_by_name(name)
                if not isinstance(module, ops.FinegrainedModule):
                    assert model_dict[key].shape == state_dict[key].shape, f"the shape of {key} not match"
                    model_dict[key] = state_dict[key]
                else:
                    # ValueSpace-based operation
                    shape = model_dict[key].shape
                    dim = len(shape)
                    if dim == 1:
                        # e.g., bias
                        model_dict[key].data = state_dict[key].data[:shape[0]]
                    if dim == 2:
                        # e.g., linear weight
                        _out, _in = shape
                        model_dict[key].data = state_dict[key].data[:_out, :_in]
                    if dim >= 3:
                        # e.g., conv weight
                        _out, _in, k = shape[:3]
                        k_larger = state_dict[key].shape[-1]
                        start, end = sub_filter_start_end(k_larger, k)
                        if dim == 3: # conv1d
                            model_dict[key].data = state_dict[key].data[:_out, :_in, start:end]
                        elif dim == 4: #conv2d
                            model_dict[key].data = state_dict[key].data[:_out, :_in, start:end, start:end]
                        else:
                            model_dict[key].data = state_dict[key].data[:_out, :_in, start:end, start:end, start:end]
        super(BaseNASNetwork, self).load_state_dict(model_dict, **kwargs, strict=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
