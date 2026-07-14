import tensorflow as tf
import numpy as np

class FourierFeatureProjection(tf.keras.layers.Layer):
    def __init__(self, num_features, scale=10.0):
        super(FourierFeatureProjection, self).__init__()
        self.num_features = num_features
        self.scale = scale

    def build(self, input_shape):
        # Generate random projection frequencies (non-trainable)
        self.B = self.add_weight(
            shape=(input_shape[-1], self.num_features),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.scale),
            trainable=False,
            name="fourier_projection_matrix"
        )

    def call(self, inputs):
        # Project spatial coordinates into high-frequency sine/cosine space
        projected = tf.matmul(inputs, self.B)
        return tf.concat([tf.sin(projected), tf.cos(projected)], axis=-1)

print("Fourier Feature Projection Layer initialized for high-frequency physics mapping.")
