import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate

from breast_cancer.models.build_layers import build_layers
from breast_cancer.utils.configs import LAYERS, VOTE_METHODS


def build_ensemble_shared_backbone(backbone, cfg, backbone_trainable=False):
    """Build a Keras model from a dictionary config with a backbone.

    Voting by default layers expect (n_batches, n_classifiers) of probabilities
       apply tf.sigmoid before voting if using logits
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
    backbone_a,
    backbone_b,
    cfg,
    backbone_trainable=False,
):
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
    *backbones,
    cfg,
    backbone_trainable=False,
):
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
