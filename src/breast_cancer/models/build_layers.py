from breast_cancer.utils.configs import LAYERS, resolve_regularizers


def build_layers(x, layers_cfg):
    for spec in layers_cfg:
        name, params = next(iter(spec.items()))
        if name not in LAYERS:
            raise ValueError(f"Unsupported layer `{name}`.")
        params = resolve_regularizers(params) if params else {}
        x = LAYERS[name](**params)(x)
    return x
