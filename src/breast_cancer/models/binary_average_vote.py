import tensorflow as tf

class BinaryAverageVote(tf.keras.layers.Layer):
    """Soft voting layer.
    
    Returns the mean probability across classifiers.
    """
    def __init__(self, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

    def call(self, inputs):
        mean_votes = tf.reduce_mean(inputs, axis=-1)
        return tf.cast(mean_votes, self.compute_dtype)