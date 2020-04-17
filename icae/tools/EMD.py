"""
Earth Mover's Distance
"""
raise DeprecationWarning()
import keras.backend as K


def keras_no_norm(yTrue, yPred):
    t = K.reshape(yTrue, (K.shape(yTrue)[0], 128))
    p = K.reshape(yPred, (K.shape(yPred)[0], 128))
    return K.mean(K.sum(K.abs(K.cumsum(t - p, axis=1)), axis=1))


def keras_normed(yTrue, yPred):
    """Converts `yTrue` and `yPred` to PDFs by scaling them so their integral is 1."""
    t = K.reshape(yTrue, (K.shape(yTrue)[0], 128))
    p = K.reshape(yPred, (K.shape(yPred)[0], 128))

    t_norm = K.sum(t, axis=1)
    p_norm = K.sum(p, axis=1)

    t /= t_norm[:, None]
    p /= p_norm[:, None]
    return keras_no_norm(t, p)

