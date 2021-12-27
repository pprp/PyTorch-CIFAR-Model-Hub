from .registry import model_entrypoints
from .registry import is_model


def build_model(model_name, **kwargs):
    if not is_model(model_name):
        raise ValueError(f"Unkown model: {model_name}")

    return model_entrypoints(model_name)(**kwargs)
