import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from train import NORM_MEAN, NORM_VAR


def build_fc_model(hp: kt.HyperParameters) -> keras.Model:
    """Fully connected architecture for Keras Tuner."""
    model = keras.Sequential([
        keras.layers.Normalization(
            input_shape=(28, 28),
            mean=NORM_MEAN,
            variance=NORM_VAR,
            axis=None
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(
            units=hp.Int("fc_units", min_value=64, max_value=256, step=64),
            activation="relu",
        ),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

