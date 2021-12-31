# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import json

import torch

from lib.mutables import spaces
from lib.utils.utils import TorchTensorEncoder

from .base_mutator import BaseMutator

logger = logging.getLogger(__name__)


class Mutator(BaseMutator):

    def __init__(self, model):
        super().__init__(model)
        self._cache = dict()
        default_mask = dict()
        for m in model.modules():
            if isinstance(m, spaces.Mutable):
                default_mask[m.key] = m.mask
        self.default_mask = default_mask

    def sample_by_mask(self, mask: dict):
        '''
        Sample an architecture by the mask
        '''
        self._cache = mask
        for mutable in self.mutables:
            assert mutable.mask.shape==mask[mutable.key].shape,\
                f"the given mask ({mask[mutable.key].shape}) cannot match the original size [{mutable.key}]{mutable.mask.shape}"
            mutable.mask = mask[mutable.key]

    def build_archs_for_valid(self, *args, **kwargs):
        '''
        Build a list of archs for validation

        Returns
        -------
        a list of dict
        '''
        if hasattr(self.model, 'build_archs_for_valid'):
            archs_to_valid = self.model.build_archs_for_valid(*args, **kwargs)
        else:
            archs_to_valid = [self.default_mask]
        self.archs_to_valid = archs_to_valid

    def sample_search(self):
        """
        Override to implement this method to iterate over mutables and make decisions.

        Returns
        -------
        dict
            A mapping from key of mutables to decisions.
        """
        raise NotImplementedError

    def sample_final(self):
        """
        Override to implement this method to iterate over mutables and make decisions that is final
        for export and retraining.

        Returns
        -------
        dict
            A mapping from key of mutables to decisions.
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        """
        Reset the mutator by call the `sample_search` to resample (for search). Stores the result in a local
        variable so that `on_forward_operation_space` and `on_forward_input_space` can use the decision directly.

        Returns
        -------
        None
        """
        if not hasattr(self, 'sample_func'):
            self._cache = self.sample_search()
        else:
            self._cache = self.sample_func(self, *args, **kwargs)
            del self.sample_func
        self._cache = self.check_freeze_mutable(self._cache)

    def check_freeze_mutable(self, mask):
        for mutable in self.mutables:
            if getattr(mutable, 'is_freeze', False):
                mask[mutable.key] = mutable.mask
        return mask

    def export(self):
        """
        Resample (for final) and return results.

        Returns
        -------
        dict
        """
        return self.sample_final()

    def save_arch(self, file_path):
        mask = self._cache
        with open(file_path, "w") as f:
            json.dump(mask, f, indent=4, sort_keys=True, cls=TorchTensorEncoder)

    def on_forward_operation_space(self, mutable, *inputs):
        """
        On default, this method calls :meth:`on_calc_layer_choice_mask` to get a mask on how to choose between layers
        (either by switch or by weights), then it will reduce the list of all tensor outputs with the policy specified
        in `mutable.reduction`. It will also cache the mask with corresponding `mutable.key`.

        Parameters
        ----------
        mutable : OperationSpace
        inputs : list of torch.Tensor

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
        """

        def _map_fn(op, *inputs):
            return op(*inputs)

        mask = self._get_decision(mutable) # 从mutable中获取建议，比如随机采样
        assert len(mask) == len(mutable.candidates), \
            "Invalid mask, expected {} to be of length {}.".format(mask, len(mutable.candidates))
        out = self._select_with_mask(_map_fn, [(choice, *inputs) for choice in mutable.candidates], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def on_forward_input_space(self, mutable, tensor_list):
        """
        On default, this method calls :meth:`on_calc_input_choice_mask` with `tags`
        to get a mask on how to choose between inputs (either by switch or by weights), then it will reduce
        the list of all tensor outputs with the policy specified in `mutable.reduction`. It will also cache the
        mask with corresponding `mutable.key`.

        Parameters
        ----------
        mutable : InputSpace
        tensor_list : list of torch.Tensor
        tags : list of string

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
        """
        mask = self._get_decision(mutable)
        assert len(mask) == mutable.n_candidates, \
            "Invalid mask, expected {} to be of length {}.".format(mask, mutable.n_candidates)
        out = self._select_with_mask(lambda x: x, [(t,) for t in tensor_list], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

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

    def _get_decision(self, mutable):
        """
        By default, this method checks whether `mutable.key` is already in the decision cache,
        and returns the result without double-check.

        Parameters
        ----------
        mutable : Mutable

        Returns
        -------
        object
        """
        if mutable.key not in self._cache:
            raise ValueError("\"{}\" not found in decision cache.".format(mutable.key))
        result = self._cache[mutable.key]
        logger.debug("Decision %s: %s", mutable.key, result)
        return result
