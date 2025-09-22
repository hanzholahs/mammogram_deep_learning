from tensorflow.keras import layers, regularizers

from breast_cancer.models.binary_average_vote import BinaryAverageVote
from breast_cancer.models.binary_majority_vote import BinaryMajorityVote


VOTE_METHODS = {
    "BinaryMajorityVote": BinaryMajorityVote,
    "BinaryAverageVote": BinaryAverageVote,
}


REGULARIZERS = {
    "L1": regularizers.L1,
    "L2": regularizers.L2,
    "L1L2": regularizers.L1L2,
}


LAYERS = {
    "Input": layers.Input,
    "Conv2D": layers.Conv2D,
    "MaxPool2D": layers.MaxPooling2D,
    "MaxPooling2D": layers.MaxPooling2D,
    "BatchNormalization": layers.BatchNormalization,
    "Dropout": layers.Dropout,
    "SpatialDropout2D": layers.SpatialDropout2D,
    "Flatten": layers.Flatten,
    "Dense": layers.Dense,
    "GlobalAveragePooling2D": layers.GlobalAveragePooling2D,
}


def resolve_regularizers(params: dict) -> dict:
    """Replace any *regularizer* entries with real keras.regularizers objects."""
    out = params.copy()

    for param_key in ("kernel_regularizer", "bias_regularizer", "activity_regularizer"):
        param_val = out.get(param_key)
        if param_val is None:
            continue

        if isinstance(param_val, str):
            out[param_key] = REGULARIZERS[param_val]()
            continue

        if isinstance(param_val, dict):
            reg_name, reg_arg = next(iter(param_val.items()))
            fn = REGULARIZERS[reg_name]
            # `reg_arg` can be scalar, list/tuple, or dict
            if isinstance(reg_arg, dict):
                sorted_vals = [reg_arg[k] for k in sorted(reg_arg)]
                out[param_key] = fn(*sorted_vals)
            elif isinstance(reg_arg, (list, tuple)):
                out[param_key] = fn(*reg_arg)
            else:
                out[param_key] = fn(reg_arg)
            continue

        raise ValueError(f"Unsupported regularizer spec: `{param_val!r}`.")

    return out
