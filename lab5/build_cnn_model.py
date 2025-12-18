import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from train import NORM_MEAN, NORM_VAR


def build_cnn_model(hp: kt.HyperParameters) -> keras.Model:
    """Convolutional architecture for Keras Tuner."""
    model = keras.Sequential([
        keras.layers.Normalization(
            input_shape=(28, 28),
            mean=NORM_MEAN,
            variance=NORM_VAR,
            axis=None
        ),
        keras.layers.Reshape((28, 28, 1)),
        keras.layers.Conv2D(
            filters=hp.Int("conv_filters", min_value=16, max_value=64, step=16),
            kernel_size=hp.Choice("kernel_size", [3, 5]),
            activation="relu",
        ),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(
            units=hp.Int("cnn_fc_units", min_value=64, max_value=256, step=64),
            activation="relu",
        ),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice("cnn_learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

