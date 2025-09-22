from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from breast_cancer.models.build_layers import build_layers


def build_from_pretrained(backbone, cfg, backbone_trainable=False):
    # Read config specifications
    input_shape = cfg.get("input_shape")
    layers_cfg = cfg.get("layers", [])
    if input_shape is None:
        raise ValueError("`input_shape` must be provided in cfg.")

    # Collect extra model kwargs, compose name with backbone.name
    reserved = {"input_shape", "layers"}
    model_args = {k: v for k, v in cfg.items() if k not in reserved}

    # Freeze/unfreeze backbone weights during training
    backbone.trainable = bool(backbone_trainable)

    # Setup inputs and shared features
    inputs = Input(shape=input_shape)
    x = backbone(inputs)
    x = build_layers(x, layers_cfg)

    return Model(inputs=inputs, outputs=x, **model_args)

def stack_models(backbone, head, transition_layer = None, name = "final_model"):
    from tensorflow.keras import Input, Model
    from tensorflow.keras import layers
    if transition_layer is None:
        transition_layer = layers.GlobalAveragePooling2D(name=f"{name}_gap")

    inputs = Input(shape=backbone.input_shape[1:], name=f"{name}_input")
    x = backbone(inputs)
    x = transition_layer(x)
    outputs = head(x)
    return Model(inputs=inputs, outputs=outputs, name=name)