from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from breast_cancer.models.build_layers import build_layers


def build_from_config(cfg):
    # Read config specifications
    input_shape = cfg.get("input_shape")
    layers_cfg = cfg.get("layers", [])
    if input_shape is None:
        raise ValueError("`input_shape` must be provided in cfg.")

    # Collect extra model kwargs, compose name with backbone.name
    reserved = {"input_shape", "layers"}
    model_args = {k: v for k, v in cfg.items() if k not in reserved}

    # Setup model layers
    inputs = Input(shape=input_shape)
    x = build_layers(inputs, layers_cfg)
    
    return Model(inputs=inputs, outputs=x, **model_args)
