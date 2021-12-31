

import torch
import torch.nn as nn

from lib.mutables.spaces import ValueSpace
from lib.utils.utils import hparams_wrapper


__all__ = [
    'FinegrainedModule'
]


@hparams_wrapper
class FinegrainedModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FinegrainedModule, self).__init__()
        # The decorator @hparams_wrapper can automatically save all input arguments to
        # ``hparams`` attribute
        self.value_spaces = self.getValueSpaces(self.hparams)

    def getValueSpaces(self, kwargs):
        value_spaces = nn.ModuleDict()
        for key, value in kwargs.items():
            if isinstance(value, ValueSpace):
                value_spaces[key] = value
                if value.index is not None:
                    _v = value.candidates[value.index]
                elif len(value.mask) != 0:
                    if isinstance(value.mask, torch.Tensor):
                        index = value.mask.clone().detach().argmax()
                    else:
                        index = torch.tensor(value.mask).argmax()
                    _v = value.candidates[index]
                else:
                    _v = value.max_value
                setattr(self, key, _v)
            else:
                setattr(self, key, value)
        return value_spaces

    def __deepcopy__(self, memo):
        try:
            new_instance = self.__class__(**self.hparams)
            device = next(self.parameters()).device
            new_instance.load_state_dict(self.state_dict())
            return new_instance.to(device)
        except Exception as e:
            print(str(e))
