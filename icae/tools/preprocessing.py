# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


data_cols = ["t=" + str(i) for i in range(128)]


def remove_deep_core(df):
    df = df.reset_index(level="string")
    drop = df["string"] >= 79  # deep core strings
    df = df[~drop]
    return df


def fit_transform_single(data, scaler):
    try:
        d = data.values
    except AttributeError:
        d = data
    return scaler.fit_transform(d.reshape(-1, 1))


def quantile_partitions(data, n):
    """
    Sorts data in n partitions so that each partition has
    about the same number of elements.
    """
    q = [i / n for i in range(n)]
    quantiles = np.quantile(data, q)
    data = np.digitize(data, quantiles)
    # np.unique(t_d, return_counts=True)
    return data, quantiles


class preprocess_scalers:
    x = MinMaxScaler()
    y = MinMaxScaler()
    z = MinMaxScaler()
    t = MinMaxScaler()


def norm_space(df: pd.DataFrame):
    """
    ['x_norm', 'y_norm', 'z_norm']
    :param df:
    :return:
    """
    df["x_norm"] = fit_transform_single(df.x, preprocess_scalers.x)
    df["y_norm"] = fit_transform_single(df.y, preprocess_scalers.y)
    df["z_norm"] = fit_transform_single(df.z, preprocess_scalers.z)


# TODO: use [r,z,t] for binning with [~20,60,many] and use 3DConv.
# TODO: use only the time space with the most events (but the same for every event)
# TODO: reconstruct only this as there should be a cylindrical symmetry.
# TODO: take about 4 losses of outlier(outlier(outlier...))


def pandas_to_nd_array(df: pd.DataFrame, resolution):
    """
    adapted from https://stackoverflow.com/questions/35047882/transform-pandas-dataframe-with-n-level-hierarchical-index-into-n-d-numpy-array
    :param df:
    :return:
    """
    # create an empty array of NaN of the right dimensions
    n_cols = len(df.columns)
    shape = list(map(len, df.index.levels))  # index dimensions
    shape.append(n_cols)
    shape = np.array(shape)
    shape[
        1:5
    ] = (
        resolution
    )  # XXX: need to manually set resolution, algorithm underestimates this (why?)
    arr = np.full(shape, 0, dtype="float32")

    indices = np.asarray(df.index)
    values = np.reshape(np.fromiter(df.values.flat, "float32"), (-1, n_cols))
    assert len(indices) == len(values)
    for i, j in zip(indices, values):
        arr[i] = j

    print("Size", arr.nbytes / 1e9, "GB")
    return arr

