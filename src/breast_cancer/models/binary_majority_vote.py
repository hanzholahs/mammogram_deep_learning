from typing import Any

import tensorflow as tf


class BinaryMajorityVote(tf.keras.layers.Layer):
    """Hard voting layer that outputs 1 if fraction of positive votes ≥ threshold.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """

    def __init__(
        self, threshold: float = 0.5, dtype: tf.dtypes.DType = tf.float32, **kwargs: Any
    ) -> None:
        super().__init__(dtype=dtype, **kwargs)
        self.threshold = threshold

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Round inputs and apply majority voting based on threshold."""
        votes = tf.round(inputs)
        mean_votes = tf.reduce_mean(votes, axis=-1)
        return tf.cast(mean_votes >= self.threshold, self.compute_dtype)

    def get_config(self) -> dict:
        """Return layer configuration including threshold."""
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config
