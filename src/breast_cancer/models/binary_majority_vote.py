import tensorflow as tf

class BinaryMajorityVote(tf.keras.layers.Layer):
    """Hard voting (majority vote) layer.
    
    Rounds each classifier's probability to 0/1, then outputs 1 if the fraction of 1s >= threshold.
    """
    def __init__(self, threshold=0.5, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.threshold = threshold

    def call(self, inputs):
        votes = tf.round(inputs)
        mean_votes = tf.reduce_mean(votes, axis=-1)
        return tf.cast(mean_votes >= self.threshold, self.compute_dtype)
    
    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config