from typing import Any, Dict, List

import tensorflow as tf

from breast_cancer.utils.configs import LAYERS, resolve_regularizers


def build_layers(x: tf.Tensor, layers_cfg: List[Dict[str, Any]]) -> tf.Tensor:
    """Build sequential layers from config specs and apply them to input tensor.

    Args:
        x: Input tensor to transform.
        layers_cfg: List of layer specifications {layer_name: params}.
    Returns:
        tf.Tensor: Output tensor after applying configured layers.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    for spec in layers_cfg:
        name, params = next(iter(spec.items()))
        if name not in LAYERS:
            raise ValueError(f"Unsupported layer `{name}`.")
        params = resolve_regularizers(params) if params else {}
        x = LAYERS[name](**params)(x)
    return x
