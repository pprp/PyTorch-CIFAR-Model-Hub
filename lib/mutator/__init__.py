from .base_mutator import BaseMutator
from .default_mutator import Mutator
from .darts_mutator import DartsMutator
from .enas_mutator import EnasMutator
from .onehot_mutator import OnehotMutator
from .random_mutator import RandomMutator
from .sequential_mutator import SequentialMutator
from .proxyless_mutator import ProxylessMutator
from .fixed_mutator import apply_fixed_architecture, FixedArchitecture


_mutator_factory = {
    "darts": DartsMutator,
    "enas": EnasMutator, 
    "gdas": OnehotMutator,
    "random": RandomMutator,
}

def build_mutator(mutator_name, model):
    if mutator_name not in list(_mutator_factory.keys()):
        raise "mutator name is not available"
    return _mutator_factory[mutator_name](model)


