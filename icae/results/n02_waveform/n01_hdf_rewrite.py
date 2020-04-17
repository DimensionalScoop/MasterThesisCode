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
import tables

from icae.tools.config_loader import config
from icae.tools import performance
from icae.interactive_setup import SuppressOutput


in_files = config.root + config.data.raw
out_file = config.root + config.data.main
errors = 0
summary_statistics = True
df_summary = None
files = glob(in_files)
batches = np.array_split(files, 24)  # adjust to available RAM

if __debug__:
    print("__debug__ on: Using only small portiona of available data")
    batches = [batches[0]]

def try_read(file):
    try:
        return pd.read_hdf(file)
    except OSError:
        return None

waveform_columns = [f"t={i}" for i in range(128)]

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
        dfs[i] = df.set_index("frame") #  "string", "om", "starting_time"

    df = pd.concat(dfs)
    df["integral"] = df[waveform_columns].sum(axis=1)

    if summary_statistics:
        global df_summary
        df_summary = calc_summary_statistics(df, df_summary)

    print("Saving…")
    df.to_hdf(
        out_file,
        mode="a",
        append=True,
        key="events",
        format="table",
        complevel=1,
        complib="lzo",
        #complib="blosc:snappy",
    )

    del dfs, df
    return frame_counter, read_errors, unique_frames_seen


def calc_summary_statistics(df: pd.DataFrame, df_summary):
    print("Summarizing…")
    dfs = df_summary
    if dfs is not None:
        desc = df.describe()
        c1, m1 = dfs.loc["count"], dfs.loc["mean"]
        c2, m2 = desc.loc["count"], desc.loc["mean"]
        dfs.loc["mean"] = (m1 * c1 + m2 * c2) / (c1 + c2)
        dfs.loc["count"] += desc.loc["count"]
        dfs.loc["min"] = np.min([desc.loc["min"], dfs.loc["min"]], axis=0)
        dfs.loc["max"] = np.max([desc.loc["max"], dfs.loc["max"]], axis=0)
    else:
        dfs = df.describe().loc[["count", "mean", "min", "max"]]

    return dfs


#%%

if __name__ == "__main__":
    frame_counter = 0
    unique_frames_seen = 0
    for batch in tqdm(batches, "Converting and compressing raw data"):
        frame_counter, new_errors, new_unique_frames = process_batch(
            batch, frame_counter
        )
        errors += new_errors
        unique_frames_seen += new_unique_frames
        print("Unique frames", unique_frames_seen)

    df_summary.to_hdf(
        out_file,
        mode="a",
        append=True,
        key="summary",
        format="table",
        complevel=1,
        complib="lzo",
    )

    f = h5py.File(out_file, "r+")
    f.attrs["unique_frames"] = unique_frames_seen
    f.attrs["file_read_errors"] = errors
    f.flush()
    f.close()

    if errors > 0:
        sys.stderr.write("%d HDF source files were unreadable." % errors)
