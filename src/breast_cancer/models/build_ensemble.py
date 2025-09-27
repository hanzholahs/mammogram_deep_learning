from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Input

from breast_cancer.models.build_layers import build_layers
from breast_cancer.utils.configs import LAYERS, VOTE_METHODS


def build_ensemble_shared_backbone(
    backbone: tf.keras.Model,
    cfg: Dict[str, Any],
    backbone_trainable: bool = False,
) -> Model:
    """Build a shared-backbone ensemble with N classifier heads and a voting layer.

    Args:
        backbone: Keras backbone model applied once to inputs.
        cfg: Configuration dict (e.g., input_shape, n_classifiers, layers, transition, vote_method).
        backbone_trainable: Whether to keep backbone weights trainable.
    Returns:
        Model: Compiled Keras model producing a single probabilistic output.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    # Read config specifications
    n_classifiers = int(cfg.get("n_classifiers", 3))
    vote_method = cfg.get("vote_method", "BinaryMajorityVote")
    input_shape = cfg.get("input_shape")
    layers_cfg = cfg.get("layers", [])
    transition = cfg.get("transition", "GlobalAveragePooling2D")

    # Perform basic validation
    if n_classifiers < 1:
        raise ValueError("`n_classifiers` must be >= 1.")
    if vote_method not in VOTE_METHODS:
        raise ValueError(f"Unsupported vote_method `{vote_method}`.")
    if input_shape is None:
        raise ValueError("`input_shape` must be provided in cfg.")
    if transition not in LAYERS:
        raise ValueError(f"Unsupported layer `{transition}` as transition.")

    # Collect extra model kwargs, compose name with backbone name
    reserved = {"input_shape", "layers", "n_classifiers", "vote_method", "transition"}
    model_args = {k: v for k, v in cfg.items() if k not in reserved}

    # Freeze/unfreeze backbone weights during training
    backbone.trainable = bool(backbone_trainable)

    # Setup inputs and shared features
    inputs = Input(shape=input_shape)
    features = backbone(inputs)
    features = LAYERS[transition](name="transition_layer")(features)

    # Build n independent classifier heads on top of shared features
    votes = []
    for _ in range(n_classifiers):
        x = features
        x = build_layers(x, layers_cfg)
        votes.append(x)

    votes_tensor = tf.stack(votes, axis=-1)
    outputs = VOTE_METHODS[vote_method]()(votes_tensor)

    return Model(inputs=inputs, outputs=outputs, **model_args)


def build_ensemble_dual_backbone(
    backbone_a: tf.keras.Model,
    backbone_b: tf.keras.Model,
    cfg: Dict[str, Any],
    backbone_trainable: bool = False,
) -> Model:
    """Build a dual-backbone model that fuses two branches before head layers.
    
    Args:
        backbone_a: First backbone model.
        backbone_b: Second backbone model.
        cfg: Configuration dict (e.g., input_shape, branch_layers, head_layers).
        backbone_trainable: Whether to keep backbone weights trainable.
    Returns:
        Model: Keras model with concatenated branches and a common head.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    # Read config specifications
    input_shape = cfg.get("input_shape")
    branch_layers = cfg.get("branch_layers", [])
    head_layers = cfg.get("head_layers", [])

    # Perform basic validations
    if input_shape is None:
        raise ValueError("`input_shape` must be provided in cfg.")

    # Collect extra model kwargs, compose name with backbone name
    reserved = {"input_shape", "branch_layers", "head_layers", "layers"}
    model_args = {k: v for k, v in cfg.items() if k not in reserved}

    # Freeze/unfreeze backbone weights during training
    backbone_a.trainable = backbone_trainable
    backbone_b.trainable = backbone_trainable

    # Setup inputs
    inputs = Input(shape=input_shape)

    # Build branch A (backbone, transition, dense layers)
    xa = backbone_a(inputs)
    xa = build_layers(xa, branch_layers)

    # Build branch B (backbone, transition, dense layers)
    xb = backbone_b(inputs)
    xb = build_layers(xb, branch_layers)

    # Fuse and extend to the head layers
    x = Concatenate()([xa, xb])
    outputs = build_layers(x, head_layers)

    return Model(inputs=inputs, outputs=outputs, **model_args)


def build_ensemble_multi_backbone(
    *backbones: tf.keras.Model,
    cfg: Dict[str, Any],
    backbone_trainable: bool = False,
) -> Model:
    """Build a multi-backbone ensemble that concatenates branch outputs into a head.
    
    Args:
        *backbones: One or more backbone models.
        cfg: Configuration dict (e.g., input_shape, branch_layers, head_layers).
        backbone_trainable: Whether to keep backbone weights trainable.
    Returns:
        Model: Keras model combining multiple backbones into a unified head.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    # Read config specifications
    input_shape = cfg.get("input_shape")
    branch_layers = cfg.get("branch_layers", [])
    head_layers = cfg.get("head_layers", [])

    # Perform basic validations
    if input_shape is None:
        raise ValueError("`input_shape` must be provided in cfg.")

    # Collect extra model kwargs, compose name with backbone name
    reserved = {"input_shape", "branch_layers", "head_layers", "layers"}
    model_args = {k: v for k, v in cfg.items() if k not in reserved}

    # Setup inputs
    inputs = Input(shape=input_shape)
    b_outs = []

    for backbone in backbones:
        backbone.trainable = backbone_trainable
        b_out = backbone(inputs)
        b_out = build_layers(b_out, branch_layers)
        b_outs.append(b_out)

    x = Concatenate()(b_outs)
    outputs = build_layers(x, head_layers)

    return Model(inputs=inputs, outputs=outputs, **model_args)
