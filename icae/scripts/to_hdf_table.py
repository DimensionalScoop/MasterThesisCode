# -*- coding: utf-8 -*-
# + {}
"""
- reads all raw data
- skips damaged raw files
- reindexes frames so each frame has a unique number
- saves everything to one compressed iterim HDF table file

Why?
- Dask only processes hdf files that were saved with the `format=table` option.
- single hdfs are easier to handle and save space
- compression allows for faster loading
"""
#%%
# %load_ext autoreload
# %autoreload 2
# %config IPCompleter.greedy=True

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from tqdm import tqdm

# import dask.dataframe as dd
from glob import glob
import h5py
from joblib import delayed, Parallel

import os
import sys

from icae.tools.config_loader import config
from icae.tools import performance

# -

in_files = config.root + config.data.raw + "*.hdf"
errors = 0
files = glob(in_files)
batches = np.array_split(files, 24)  # adjust to available RAM
# + {}
import tables


def try_read(file):
    try:
        return pd.read_hdf(file)
    except OSError:
        return None


def process_batch(batch_files, frame_counter):
    print("Reading…")
    read_errors = 0
    unique_frames_seen = 0
    tasks = [delayed(try_read)(i) for i in batch_files]
    dfs = Parallel(n_jobs=12, timeout=100)(
        tasks
    )  # change n_jobs according to you RAM / # of CPUs

    for i in range(len(dfs)):
        if dfs[i] is None:
            read_errors += 1
            continue
        df = dfs[i].reset_index()
        unique_frames_seen += df["frame"].max()
        df["frame"] += frame_counter
        frame_counter = df["frame"].max()
        dfs[i] = df.set_index(["frame", "string", "om", "starting_time"])

    df = pd.concat(dfs)
    print("Saving", config.root + config.data.retabled_single, "…")
    df.to_hdf(
        config.root + config.data.retabled_single,
        mode="a",
        append=True,
        key=config.data.hdf_key,
        format="table",
        complevel=1,
        complib="blosc:snappy",
    )

    del dfs, df
    return frame_counter, read_errors, unique_frames_seen


#%%
errors = 0
frame_counter = 0
unique_frames_seen = 0
for batch in tqdm(batches, "Converting and compressing raw data…"):
    frame_counter, new_errors, new_unique_frames = process_batch(batch, frame_counter)
    errors += new_errors
    unique_frames_seen += new_unique_frames
    print("Unique frames", unique_frames_seen)

#%%

f = h5py.File(config.root + config.data.retabled_single,'w')
f.attrs['unique_frames'] = -1 #unique_frames_seen
f.flush()
f.close()

if errors > 0:
    sys.stderr.write("%d HDF source files were unreadable." % errors)
