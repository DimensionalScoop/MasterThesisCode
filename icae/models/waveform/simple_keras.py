raise DeprecationWarning("Keras")

from keras.models import Sequential
import sys

from icae.tools import EMD


def optimal_simple(latent_dim=3, loss_method="mean_squared_error"):
    """A very simple model that is near-optimal on toy data."""
    from keras.layers import Reshape, Dense, Flatten, Conv1D

    params = {"kernel": 4, "strides": 2}
    m = Sequential()

    m.add(Reshape((128, 1), input_shape=(128,)))
    m.add(Conv1D(1, params["kernel"], strides=params["strides"]))
    m.add(Flatten())
    m.add(Dense(latent_dim))

    model_encoder = Sequential(m.layers)
    m.add(Dense(87, input_shape=(latent_dim,), activation="relu"))
    m.add(Dense(128, activation="sigmoid"))

    m.compile(optimizer="adam", loss=loss_method, metrics=[EMD.keras_no_norm])
    return m, model_encoder


# +
# m = optimal_simple()
# m[0].summary()
# -

from keras.layers import Reshape, Dense, Flatten, Conv1D, BatchNormalization


def optimal_NB(latent_dim=3, loss_method="mean_squared_error"):
    """A very simple model that is near-optimal on toy data."""
    from keras.layers import Reshape, Dense, Flatten, Conv1D, BatchNormalization

    params = {"kernel": 4, "strides": 2}
    m = Sequential()

    m.add(Reshape((128, 1), input_shape=(128,)))
    m.add(Conv1D(1, params["kernel"], strides=params["strides"]))
    m.add(Flatten())
    m.add(BatchNormalization())
    m.add(Dense(latent_dim))
    m.add(BatchNormalization())

    model_encoder = Sequential(m.layers)
    m.add(Dense(87, input_shape=(latent_dim,), activation="relu"))
    m.add(Dense(128, activation="sigmoid"))

    m.compile(optimizer="adam", loss=loss_method, metrics=[EMD.keras_no_norm])
    return m, model_encoder


def several_preception(latent_dim=5):
    """not optimal, somewhat more complex model."""
    from keras.layers import Reshape, Dense, Flatten, Conv1D, AveragePooling1D

    params = {"kernel": 4, "strides": 2}

    m = Sequential()

    m.add(Reshape((128, 1), input_shape=(128,)))
    m.add(Conv1D(1, params["kernel"], strides=params["strides"]))
    m.add(AveragePooling1D())
    m.add(Conv1D(1, 6))
    m.add(AveragePooling1D())
    m.add(Flatten())
    m.add(Dense(latent_dim))

    model_encoder = Sequential(m.layers)
    m.add(Dense(87, input_shape=(2,), activation="relu"))
    m.add(Dense(128, activation="sigmoid"))

    m.compile(optimizer="adam", loss="mean_squared_error", metrics=[EMD.keras_no_norm])
    return m, model_encoder