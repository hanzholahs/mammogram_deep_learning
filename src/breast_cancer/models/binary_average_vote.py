from typing import Any

import tensorflow as tf


class BinaryAverageVote(tf.keras.layers.Layer):
    """Soft voting layer that returns the mean probability across classifiers.

    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """

    def __init__(self, dtype: tf.dtypes.DType = tf.float32, **kwargs: Any) -> None:
        super().__init__(dtype=dtype, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute average probability across classifiers."""
        mean_votes = tf.reduce_mean(inputs, axis=-1)
        return tf.cast(mean_votes, self.compute_dtype)
