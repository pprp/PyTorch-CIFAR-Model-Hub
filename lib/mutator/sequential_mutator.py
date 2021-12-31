import numpy as np
import json
import torch
import torch.nn.functional as F

from lib.mutables.spaces import InputSpace, OperationSpace, ValueSpace

from .default_mutator import Mutator

__all__ = [
    'SequentialMutator',
]

class SequentialMutator(Mutator):
    def __init__(self, model, start_idx: int):
        super().__init__(model)
        with open('./mutator/Track1_final_archs.json', 'r') as f:
            self.masks = json.load(f)
            self.crt_index = start_idx
            self.max_num = len(self.masks)
            assert self.crt_index > 0, 'Index should start at 1'

    def sample_search(self):
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, OperationSpace):
                gen_index = torch.randint(high=mutable.length, size=(1, ))
                result[mutable.key] = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
                mutable.mask = torch.zeros_like(result[mutable.key])
                mutable.mask[result[mutable.key].detach().numpy().argmax()] = 1
            elif isinstance(mutable, InputSpace):
                if mutable.n_chosen is None:
                    result[mutable.key] = torch.randint(high=2, size=(mutable.n_candidates,)).view(-1).bool()
                else:
                    perm = torch.randperm(mutable.n_candidates)
                    mask = [i in perm[:mutable.n_chosen] for i in range(mutable.n_candidates)]
                    result[mutable.key] = torch.tensor(mask, dtype=torch.bool)  # pylint: disable=not-callable
                mutable.mask = torch.zeros_like(result[mutable.key])
                mutable.mask[result[mutable.key].detach().numpy().argmax()] = 1
            elif isinstance(mutable, ValueSpace):
                index_choice = int(mutable.key.split('ValueSpace')[-1]) - 1
                value = self.masks[f'arch{self.crt_index}']['arch'].split('-')[index_choice]
                gen_index = np.argwhere(np.array(mutable.candidates)==int(value))[0][0]
                gen_index = torch.tensor(gen_index)
                result[mutable.key] = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
                mutable.mask = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
        if self.crt_index >= self.max_num:
            self.crt_index = 1
        else:
            self.crt_index += 1
        return result

    def sample_final(self):
        return self.sample_search()
