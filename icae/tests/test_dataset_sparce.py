from icae.tools.dataset_sparse import SparseEventDataset
import pandas.util.testing as pdt
import numpy.testing as npt
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm

tmp_path = "/tmp/"


def test_limit_event_t_size():
    cols = ["wf_AE_loss", "wf_integral"]
    limit = 850
    CHANNELS_PER_FEATURE = 5

    ds = SparseEventDataset(size=1000)
    ds.value_columns = cols.copy()
    ds.prune()
    ds.limit_event_t_size(limit)

    for i, _ in zip(ds, range(100)):
        channel, t_bins, z_bins = list(i.size())
        assert channel == len(cols) * CHANNELS_PER_FEATURE
        assert t_bins == limit + 1


def test_load():
    dtype = "float16"
    ds = SparseEventDataset(size=1000)
    ds.value_columns = ["wf_AE_loss", "wf_integral"]
    ds.prune()
    ds.save(tmp_path, dtype=dtype)

    loaded_ds = SparseEventDataset.load(tmp_path)

    for i in loaded_ds.__dict__:
        actual = ds.__dict__[i]
        loaded = loaded_ds.__dict__[i]

        if type(actual) is pd.DataFrame:
            pdt.assert_almost_equal(actual.astype(dtype), loaded)
        elif type(actual) is pd.Series:
            assert np.alltrue(loaded.values == actual.values)
        else:
            assert actual == loaded


if __name__ == "__main__":
    test_load()
    test_limit_event_t_size()

