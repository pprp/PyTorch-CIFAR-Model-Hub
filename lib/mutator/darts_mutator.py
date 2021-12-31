# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.mutables.spaces import InputSpace, OperationSpace, ValueSpace

from .default_mutator import Mutator

__all__ = [
    'DartsMutator',
]


class DartsMutator(Mutator):
    """
    Doc:
        Connects the model in a DARTS (differentiable) way.

        An extra connection is automatically inserted for each OperationSpace, when this connection is selected, there is no
        op on this OperationSpace (namely a ``ZeroOp``), in which case, every element in the exported choice list is ``false``
        (not chosen).

        All input choice will be fully connected in the search phase. On exporting, the input choice will choose inputs based
        on keys in ``choose_from``. If the keys were to be keys of LayerChoices, the top logit of the corresponding OperationSpace
        will join the competition of input choice to compete against other logits. Otherwise, the logit will be assumed 0.

        It's possible to cut branches by setting parameter ``choices`` in a particular position to ``-inf``. After softmax, the
        value would be 0. Framework will ignore 0 values and not connect. Note that the gradient on the ``-inf`` location will
        be 0. Since manipulations with ``-inf`` will be ``nan``, you need to handle the gradient update phase carefully.

        Attributes
        ----------
        choices: ParameterDict
            dict that maps keys of LayerChoices to weighted-connection float tensors.
    """
    def __init__(self, model, *args, **kwargs):
        super().__init__(model)
        self.choices = nn.ParameterDict()
        for mutable in self.mutables:
            if isinstance(mutable, OperationSpace):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length))
            if isinstance(mutable, ValueSpace):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length))
                mutable.mask = self.choices[mutable.key].data
            elif isinstance(mutable, InputSpace):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.n_candidates))

    @property
    def device(self):
        for v in self.choices.values():
            return v.device

    def sample_search(self):
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, OperationSpace):
                # slicing zero operation. if zero operation is chosen, then a list of all 'False' will be returned
                result[mutable.key] = F.softmax(self.choices[mutable.key], dim=-1)
                mutable.mask = torch.zeros_like(result[mutable.key])
                mutable.mask.data = result[mutable.key].data
            elif isinstance(mutable, ValueSpace):
                result[mutable.key] = F.softmax(self.choices[mutable.key], dim=-1)
                mutable.mask.data = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1).data
            elif isinstance(mutable, InputSpace):
                result[mutable.key] = torch.ones(mutable.n_candidates, dtype=torch.bool, device=self.device)
                mutable.mask = torch.ones_like(result[mutable.key]) # Todo: 搜索阶段全为1
        return result

    def sample_final(self):
        result = dict()
        edges_max = dict()
        for mutable in self.mutables:
            if isinstance(mutable, (OperationSpace, ValueSpace)):
                max_val, index = torch.max(F.softmax(self.choices[mutable.key], dim=-1), 0)
                edges_max[mutable.key] = max_val
                result[mutable.key] = F.one_hot(index, num_classes=mutable.length).view(-1).bool()
                mutable.mask = torch.zeros_like(result[mutable.key])
                mutable.mask[result[mutable.key].cpu().detach().numpy().argmax()] = 1
        for mutable in self.mutables:
            if isinstance(mutable, InputSpace):
                if mutable.n_chosen is not None:
                    weights = []
                    for src_key in mutable.choose_from:
                        # todo: figure out this issue
                        if src_key not in edges_max:
                            print("InputSpace.NO_KEY in '%s' is weighted 0 when selecting inputs.", mutable.key)
                        weights.append(edges_max.get(src_key, 0.))
                    weights = torch.tensor(weights)  # pylint: disable=not-callable
                    _, topk_edge_indices = torch.topk(weights, mutable.n_chosen)
                    selected_multihot = []
                    for i, src_key in enumerate(mutable.choose_from):
                        if i not in topk_edge_indices and src_key in result:
                            # If an edge is never selected, there is no need to calculate any op on this edge.
                            # This is to eliminate redundant calculation.
                            result[src_key] = torch.zeros_like(result[src_key])
                        selected_multihot.append(i in topk_edge_indices)
                    result[mutable.key] = torch.tensor(selected_multihot, dtype=torch.bool, device=self.device)  # pylint: disable=not-callable
                    mutable.mask = torch.zeros_like(result[mutable.key]) # Todo: 搜索阶段全为1
                    mutable.mask[result[mutable.key].cpu().detach().numpy().argmax()] = 1
                else:
                    result[mutable.key] = torch.ones(mutable.n_candidates, dtype=torch.bool, device=self.device)  # pylint: disable=not-callable
        return result
