import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob


def load_raw(
    files="../data/01-raw/*.hdf", show_progress=True, as_numpy=True, max_files=-1
):
    files = glob(files)  # [f for f in os.listdir(path) if f.endswith('.hdf')]
    if max_files != -1:
        files = files[:max_files]
    parts = []
    for f in tqdm(files, "Loading MC data from disk…", disable=not show_progress):
        parts.append(pd.read_hdf(f))

    if as_numpy:
        return np.concatenate(list(np.asarray(i.values) for i in parts), axis=0)
    else:
        return pd.concat(parts)
