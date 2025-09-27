from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import Input

from breast_cancer.models.build_layers import build_layers


def build_from_pretrained(
    backbone: tf.keras.Model,
    cfg: Dict[str, Any],
    backbone_trainable: bool = False,
) -> Model:
    """Build a model with a pretrained backbone followed by config-defined layers.

    Args:
        backbone: Pretrained Keras backbone model.
        cfg: Configuration dict (must include input_shape and layers).
        backbone_trainable: Whether to keep backbone weights trainable.
    Returns:
        Model: Keras model with backbone and custom head.

    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
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


def stack_models(
    backbone: tf.keras.Model,
    head: tf.keras.Model,
    transition_layer: tf.keras.layers.Layer | None = None,
    name: str = "final_model",
) -> Model:
    """Stack a backbone and head model with an optional transition layer.
    
    Args:
        backbone: Backbone model for feature extraction.
        head: Head model for classification/regression.
        transition_layer: Optional transition layer (default GlobalAveragePooling2D).
        name: Name of the final model.
    Returns:
        Model: Combined Keras model.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    # Set default transition_layer
    if transition_layer is None:
        transition_layer = layers.GlobalAveragePooling2D(name=f"{name}_gap")

    # Stack backbone and head into one model
    inputs = Input(shape=backbone.input_shape[1:], name=f"{name}_input")
    x = backbone(inputs)
    x = transition_layer(x)
    outputs = head(x)

    return Model(inputs=inputs, outputs=outputs, name=name)
