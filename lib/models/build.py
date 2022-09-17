from .registry import _model_entrypoints, is_model, model_entrypoints


def show_available_models():
    """Displays available models"""
    print(list(model_entrypoints.keys()))


def build_model(model_name, **kwargs):
    if not is_model(model_name):
        raise ValueError(
            f'Unkown model: {model_name} not in {list(_model_entrypoints.keys())}'
        )

    return model_entrypoints(model_name)(**kwargs)
