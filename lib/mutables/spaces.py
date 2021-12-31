# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Optional, Union, List, Tuple

import torch
import torch.nn as nn

from lib.utils.utils import hparams_wrapper


__all__ = [
    'Mutable',
    'OperationSpace',
    'InputSpace',
    'MutableScope',
    'ValueSpace',
    'global_mutable_counting'
]


class global_mutable_counting:
    _counter = 0
    
    @classmethod
    def __call__(cls):
        cls._counter += 1
        return cls._counter


@hparams_wrapper
class Mutable(nn.Module):
    """
    Mutable is designed to function as a normal layer, with all necessary operators' weights.
    States and weights of architectures should be included in mutator, instead of the layer itself.

    Mutable has a key, which marks the identity of the mutable. This key can be used by users to share
    decisions among different mutables. In mutator's implementation, mutators should use the key to
    distinguish different mutables. Mutables that share the same key should be "similar" to each other.

    Currently the default scope for keys is global.
    """

    def __init__(self, key=None):
        super().__init__()
        self.is_search =True
        if key is not None:
            if not isinstance(key, str):
                key = str(key)
                logger.warning("Warning: key \"%s\" is not string, converted to string.", key)
            self._key = key
        else:
            self._key = self.__class__.__name__ + str(global_mutable_counting()())
        self.init_hook = self.forward_hook = None

    def __deepcopy__(self, memodict=None):
        raise NotImplementedError("Deep copy doesn't work for mutables.")

    def __call__(self, *args, **kwargs):
        if self.is_search:
            self._check_built()
        return super().__call__(*args, **kwargs)

    def set_mutator(self, mutator):
        if "mutator" in self.__dict__:
            raise RuntimeError("`set_mutator` is called more than once. Did you parse the search space multiple times? "
                               "Or did you apply multiple fixed architectures?")
        self.__dict__["mutator"] = mutator

    def forward(self, *inputs):
        raise NotImplementedError

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._name if hasattr(self, "_name") else "_key"

    @name.setter
    def name(self, name):
        self._name = name

    def _check_built(self):
        if not hasattr(self, "mutator"):
            raise ValueError(
                "Mutator not set for {}. You might have forgotten to initialize and apply your mutator. "
                "Or did you initialize a mutable on the fly in forward pass? Move to `__init__` "
                "so that trainer can locate all your mutables. See NNI docs for more details.".format(self))

    def __repr__(self):
        return "{} ({})".format(self.name, self.key)

    def __setattr__(self, attribute, value):
        if getattr(self, 'is_freeze', False) and \
            attribute in getattr(self, 'frozen_attributes', []):
            if getattr(self, 'verbose_freeze', False):
                print(f"{attribute} has been forzen, you should call `defrost` function before you modify it.")
        else:
            super(Mutable, self).__setattr__(attribute, value)

    def freeze(self, attribute=None, verbose=False):
        self.is_freeze = True
        self.verbose_freeze = verbose
        if attribute is None:
            self.frozen_attributes = ['mask', 'index', 'candidates', 'key']
        else:
            self.frozen_attributes = [attribute]

    def defrost(self):
        self.is_freeze = False


class MutableScope(Mutable):
    """
    Mutable scope marks a subgraph/submodule to help mutators make better decisions.
    Mutators get notified when a mutable scope is entered and exited. Mutators can override ``enter_mutable_scope``
    and ``exit_mutable_scope`` to catch corresponding events, and do status dump or update.
    MutableScope are also mutables that are listed in the mutables (search space).
    """

    def __init__(self, key):
        super().__init__(key=key)

    def __call__(self, *args, **kwargs):
        try:
            if self.is_search:
                self._check_built()
                self.mutator.enter_mutable_scope(self)
            return super().__call__(*args, **kwargs)
        finally:
            self.mutator.exit_mutable_scope(self)


class CategoricalSpace(Mutable):
    def __init__(self,
                 candidates: List[Any],
                 mask: Optional[Union[dict, list]] = None,
                 index: int = None,
                 key: Optional[str] = None):
        '''CategoricalSpace search space
        Examples:
            candidates: [conv3x3, conv5x5, Identity]
            key: default value is empty string "". suppose key is 'key0'
            mask: (sopports many types of masks)
                - [True, False, False]
                - [0, 1, 0]
                - {'key0': [0,1,0], 'key2': [1,0,0,0]} will automatically match 'key0'
        '''
        super().__init__(key)
        self.is_search = True
        self.candidates = candidates
        self.candidates_original = candidates
        self.length = len(candidates)
        self.dtype = type(candidates[0])
        self.index = index

        if index is not None:
            if mask is not None and len(mask)!=0:
                print('You only need to specify the valye of mask or index. Index is used by default.')
            self.mask = torch.zeros(self.length)
            self.mask[index] = 1
            self.is_search = False
        elif mask is not None:
            self.is_search = False
            if isinstance(mask, dict):
                if isinstance(mask[self.key], torch.Tensor):
                    self.mask = mask[self.key].cpu().clone().detach()
                else:
                    self.mask = torch.tensor(mask[self.key])
            elif isinstance(mask, list):
                assert len(mask) == len(candidates), \
                    f"The length of the mask ({len(mask)}) should be equal to #candidates ({len(candidates)})"
                self.mask = torch.tensor(mask)
            if 'int' in str(self.mask.dtype).lower():
                if self.mask.sum()==1:
                    # converting one-hot mask to bool type
                    self.mask = self.mask.bool()
                else:
                    # converting non-one-hot mask to float type
                    self.mask = self.mask.float()
            self.convert_index_by_mask(self.mask)
        else:
            if isinstance(candidates[0], (int, float)):
                # random a mask based on biggest value
                index = torch.tensor(candidates).argmax()
            else:
                index = -1
            self.mask = torch.zeros(self.length)
            self.mask[index] = 1
        self.device = 'cpu'

    def convert_index_by_mask(self, mask):
        '''Only convert index for one-hot mask'''
        if "BoolTensor" in mask.type() and mask.int().sum()==1:
            self.index = mask.int().argmax()

    @property
    def value(self):
        if self.index is not None:
            return self.candidates_original[self.index]
        else:
            value = [self.candidates_original[idx] for idx, is_selected in enumerate(self.mask) if is_selected]
            if len(value) == 1: value=value[0]
            return value

    @property
    def max_value(self):
        try:
            value = max(self.candidates_original)
            return value
        except Expection as e:
            print(str(e))
            print("The candidates cannot be compared to get max value, return the first item instead.")
            return self.candidates_original[0]

    def forward(self):
        warnings.warn(f'You should not run forward of {self.__class__.__name__} directly.')
        return self.value

    def __getitem__(self, index):
        return self.candidates[index]

    def __setitem__(self, index, data):
        self.candidates[index] = data

    def __len__(self):
        return len(self.candidates)

    def __iter__(self):
        for elem in self.candidates:
            yield elem

    def __repr__(self):
        name = self.__class__.__name__
        _repr = f'{name}({self.candidates}, key={repr(self.key)}, value={self.value})'
        return _repr

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        if self.hparams['key'] is None:
            global_mutable_counting._counter -= 1
        new_instance = self.__class__(self.candidates_original)
        new_instance._key = self.key
        new_instance.mask = self.mask
        return new_instance

    def _select_with_mask(self, map_fn, candidates, mask):
        if "BoolTensor" in mask.type():
            out = [map_fn(*cand) for cand, m in zip(candidates, mask) if m]
        elif "FloatTensor" in mask.type():
            out = [map_fn(*cand) * m for cand, m in zip(candidates, mask) if m]
        else:
            raise ValueError("Unrecognized mask")
        return out

    def _tensor_reduction(self, reduction_type, tensor_list):
        if reduction_type == "none":
            return tensor_list
        if not tensor_list:
            return None  # empty. return None for now
        if len(tensor_list) == 1:
            return tensor_list[0]
        if reduction_type == "sum":
            return sum(tensor_list)
        if reduction_type == "mean":
            return sum(tensor_list) / len(tensor_list)
        if reduction_type == "concat":
            return torch.cat(tensor_list, dim=1)
        raise ValueError("Unrecognized reduction policy: \"{}\"".format(reduction_type))


class OperationSpace(CategoricalSpace):
    def __init__(self,
                 candidates: Union[list],
                 mask: Optional[Union[dict, list]] = None,
                 index: int = None,
                 reduction: str = "sum",
                 return_mask: bool = False,
                 key: Optional[str] = None):
        super().__init__(candidates, mask, index, key)
        self.reduction = reduction
        self.return_mask = return_mask
        if self.is_search:
            self.candidates = nn.ModuleList(candidates)
        else:
            self.candidates = nn.ModuleList()
            for idx, is_selected in enumerate(self.mask):
                if is_selected: self.candidates.append(candidates[idx])

    def forward(self, *inputs, reduction='sum'):
        if self.is_search and hasattr(self, "mutator") and self.mutator._cache:
            out, mask = self.mutator.on_forward_operation_space(self, *inputs)
            self.mask = mask
        else:
            def _map_fn(op, *inputs):
                return op(*inputs)

            mask = self.mask
            if "BoolTensor" in self.mask.type():
                mask = torch.tensor([True for i in range(len(self.candidates))])
            assert len(mask) == len(self.candidates), \
                "Invalid mask, expected {} to be of length {}.".format(mask, len(self.candidates))
            out = self._select_with_mask(_map_fn, [(choice, *inputs) for choice in self.candidates], mask)
            out = self._tensor_reduction(self.reduction, out)
        if self.return_mask:
            return out, self.mask
        else:
            return out

    def __repr__(self):
        if self.is_search:
            return super(Mutable, self).__repr__()
        else:
            name = self.__class__.__name__
            _repr = f'{name}(key={repr(self.key)}, value={self.value})'
            return _repr


class InputSpace(CategoricalSpace):
    """
    Description:
        Input choice selects `n_chosen` inputs from `choose_from` (contains `n_candidates` keys). For beginners,
        use `n_candidates` instead of `choose_from` is a safe option. To get the most power out of it, you might want to
        know about `choose_from`.

        The keys in `choose_from` can be keys that appear in past mutables, or ``NO_KEY`` if there are no suitable ones.
        The keys are designed to be the keys of the sources. To help mutators make better decisions,
        mutators might be interested in how the tensors to choose from come into place. For example, the tensor is the
        output of some operator, some node, some cell, or some module. If this operator happens to be a mutable (e.g.,
        ``OperationSpace`` or ``InputSpace``), it has a key naturally that can be used as a source key. If it's a
        module/submodule, it needs to be annotated with a key: that's where a ``MutableScope`` is needed.
    """

    NO_KEY = ""

    def __init__(self,
                 n_candidates: Optional[int] = None,
                 choose_from: Optional[List[str]] = None,
                 n_chosen: Optional[int] = None,
                 mask: Optional[Union[dict, list]] = None,
                 index: Optional[int] = None,
                 reduction: str = "sum",
                 return_mask: bool = False,
                 key: Optional[str] = None):
        """
        Initialization.

            Parameters
            ----------
            n_candidates : int
                Number of inputs to choose from.
            choose_from : list of str
                List of source keys to choose from. At least of one of `choose_from` and `n_candidates` must be fulfilled.
                If `n_candidates` has a value but `choose_from` is None, it will be automatically treated as `n_candidates`
                number of empty string.
            n_chosen : int
                Recommended inputs to choose. If None, mutator is instructed to select any.
            reduction : str
                `mean`, `concat`, `sum` or `none`.
            return_mask : bool
                If `return_mask`, return output tensor and a mask. Otherwise return tensor only.
            key : str
                Key of the input choice.

            Examples
            ----------
                >>>    # first example
                >>>    inputs = [out1, out2, out3]
                >>>    input_choice = InputSpace(n_candidates=3, n_chosen=1)
                >>>    out, mask = InputSpace(inputs, return_mask=True)
                >>>    #
                >>>    # second example
                >>>    inputs = {'key1':out1, 'key2':out2, 'key3':out3}
                >>>    input_choice = InputSpace(choose_from=['key1', 'key2', 'key3'], n_chosen=1)
                >>>    out, mask = InputSpace(inputs, return_mask=True)
        """
        # precondition check
        assert n_candidates is not None or choose_from is not None, "At least one of `n_candidates` and `choose_from`" \
                                                                    "must be not None."
        if choose_from is not None and n_candidates is None:
            n_candidates = len(choose_from)
        elif choose_from is None and n_candidates is not None:
            choose_from = [self.NO_KEY] * n_candidates
        assert n_candidates == len(choose_from), "Number of candidates must be equal to the length of `choose_from`."
        assert n_candidates > 0, "Number of candidates must be greater than 0."
        assert n_chosen is None or 0 <= n_chosen <= n_candidates, "Expected selected number must be None or no more " \
                                                                  "than number of candidates."

        super().__init__(candidates=choose_from, mask=mask, index=index, key=key)

        self.n_candidates = n_candidates
        self.choose_from = choose_from
        self.n_chosen = n_chosen
        self.reduction = reduction
        self.return_mask = return_mask

    def forward(self, optional_inputs: Union[list, dict]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward method of OperationSpace.

        Parameters
        ----------
        optional_inputs : list or dict
            Recommended to be a dict. As a dict, inputs will be converted to a list that follows the order of
            `choose_from` in initialization. As a list, inputs must follow the semantic order that is the same as
            `choose_from`.
        Returns
        -------
        tuple of torch.Tensor and torch.Tensor or torch.Tensor
        """
        if self.is_search and hasattr(self, "mutator") and self.mutator._cache:
            optional_input_list = optional_inputs
            if isinstance(optional_inputs, dict):
                optional_input_list = [optional_inputs[tag] for tag in self.choose_from]
            assert isinstance(optional_input_list, list), \
                "Optional input list must be a list, not a {}.".format(type(optional_input_list))
            assert len(optional_inputs) == self.n_candidates, \
                "Length of the input list must be equal to number of candidates."
            out, mask = self.mutator.on_forward_input_space(self, optional_input_list)
        else:
            mask = self.mask
            out = self._select_with_mask(lambda x: x, [(t,) for t in optional_inputs], mask)
            out = self._tensor_reduction(self.reduction, out)
        if self.return_mask:
            return out, mask
        else:
            return out

    def __repr__(self):
        if self.is_search:
            return super(Mutable, self).__repr__()
        else:
            name = self.__class__.__name__
            _repr = f'{name}({self.choose_from}, key={repr(self.key)}, value={self.value})'
            return _repr

class ValueSpace(CategoricalSpace):
    def __init__(self,
                 candidates: List[Any],
                 mask: Optional[Union[dict, list]] = None,
                 index: int = None,
                 key: Optional[str] = None):
        '''
        Examples:
            >>> ValueSpace([8,16,24], index=1)
            ValueSpace([8, 16, 24], key='ValueChoice1', value=16)
            >>> ValueSpace([3,5,7], index=0)
            ValueSpace([3, 5, 7], key='ValueChoice2', value=3)
            >>> ValueSpace([3,5,7], mask={'key0': [1,0,0], 'key1':[0,1]}, key='key0')
            ValueSpace([3, 5, 7], key='key0', value=3)
        '''
        super().__init__(candidates, mask, index, key)
        self._sortIdx = None # sorted indices for module weights when pruning
        self.bindModuleNames = []

    @property
    def lastBindModuleName(self):
        assert len(self.bindModuleNames)>0, \
            "Please apply `bind_module_to_ValueSpace` function to bind modules to ValueSpace.\
            There is currently no module name bound."            
        return self.bindModuleNames[-1]

    @property
    def sortIdx(self):
        if self._sortIdx is None:
            self._sortIdx = torch.tensor(list(range(self.max_value))).to(self.device)
            # print(f"Use {self._sortIdx} instead. Please use `sortChannels` to get sortIdx.")
            return self._sortIdx
        return self._sortIdx

    @sortIdx.setter
    def sortIdx(self, indices):
        if isinstance(indices, list):
            indices = torch.tensor(indices)
        self._sortIdx = indices

if __name__ == '__main__':
    # mask = {'test': torch.tensor([0.5,0.3,0.2,0.1])}
    mask = {
        'test1': torch.tensor([True, False, True, False]),
        'test2': torch.tensor([0.5,0.3,0.2,0.1])
    }
    op1 = OperationSpace([
            nn.Linear(10,1),
            nn.Linear(10,1),
            nn.Linear(10,1),
            nn.Linear(10,1)
        ], key='test1', mask=mask
    )
    op2 = OperationSpace([
            nn.Linear(10,1),
            nn.Linear(10,1),
            nn.Linear(10,1),
            nn.Linear(10,1)
        ], key='test2', mask=mask
    )
    print(op1,op2)
    x = torch.rand(2,10)
    y = op1(x)
    print(y)
    y = op2(x)
    print(y)
