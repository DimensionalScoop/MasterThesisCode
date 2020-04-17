# FIXME: This file is never imported.

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from tqdm import tqdm

import sys
import dask
import dask.dataframe as dd

import icae.tools.loss
import icae.tools.loss.EMD

from icae.tools import EMD
from icae.tools import nn
from icae.tools import status_report

from sklearn.preprocessing import MinMaxScaler
from hilbertcurve.hilbertcurve import HilbertCurve

import icae.tools.preprocessing.general as preproc


# -
def cylindircal(df):
    r = np.sqrt(df.x ** 2 + df.y ** 2)
    df["r_xy"] = MinMaxScaler().fit_transform(r)


def hilbert(df, iterations=12, dim=2, delete_temp_features=True, use_grouping=True):
    """iterations = 4: i.e. 256 cells (2^(p*n)). Frame cannot use multiindexing when use_grouping is true."""
    cells = 2 ** (iterations * dim)

    xy_transform = HilbertCurve(iterations, dim)
    mm = MinMaxScaler((0, xy_transform.max_x))
    minmax = lambda x: mm.fit_transform(np.rint(x.values.reshape(-1, 1)))
    df["_x_hnorm"] = minmax(np.rint(df.x))
    df["_y_hnorm"] = minmax(np.rint(df.y))
    unique_xy = df[["_x_hnorm", "_y_hnorm"]].drop_duplicates().values
    print("Found", len(unique_xy), "unique positions.")

    distance_by_xy = {}
    for i in unique_xy:
        distance_by_xy[tuple(i)] = xy_transform.distance_from_coordinates(
            [int(i[0]), int(i[1])]
        )

    if use_grouping:
        import dask.dataframe as dd

        to_group = df  # [['_x_hnorm', '_y_hnorm']]
        grouped = to_group.groupby(["_x_hnorm", "_y_hnorm"])

        def do(group):
            first = group.iloc[0]
            coordinates = (first["_x_hnorm"], first["_y_hnorm"])
            group["hilbert_distance"] = distance_by_xy[coordinates]
            return group.reset_index(drop=True)

        df = grouped.apply(do)
    else:
        df["hilbert_distance"] = df[["_x_hnorm", "_y_hnorm"]].apply(
            lambda xy: distance_by_xy[tuple(xy)], axis=1
        )  # XXX: very slow

    df["hilbert_distance"] /= cells

    if delete_temp_features:
        df.drop(columns=["_x_hnorm", "_y_hnorm"], inplace=True)
    return df.reset_index(drop=True)


def latent_and_loss(df):
    raise DeprecationWarning()
    import keras
    preproc.data_cols
    model_path = "../alg_AE_single/reports/0408-144242 BN 1k-bx100e/"
    custom_obj = {"keras_no_norm": EMD.keras_no_norm}
    AE = keras.models.load_model(model_path + "final AE model.hdf", custom_obj)
    encoder = keras.models.load_model(
        model_path + "final encoder model.hdf", custom_obj
    )

    def get_loss_and_latent(raw_wf):
        data = AE_lib.preprocess(raw_wf)
        pred = AE.predict(data)
        latent = encoder.predict(data)
        loss = tools.loss.EMD.numpy(data, pred).flatten()
        return loss, latent

    print("Calculating AE loss and latent space…")
    loss, latent = get_loss_and_latent(df[preproc.data_cols].values)

    df["AE_loss"] = preproc.fit_transform_single(loss, MinMaxScaler())
    for i in range(latent.shape[1]):
        df["latent_" + str(i)] = preproc.fit_transform_single(
            latent[:, i], MinMaxScaler()
        )

    return df
