import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from glob import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from tqdm import tqdm
import warnings
from box import Box
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler

from icae.tools.config_loader import config
from icae.tools.dataset_sparse import SparseEventDataset


bin_config = Box.from_yaml(filename=config.root + config.data.binning_config)


def sort_df_by_brightness(df: pd.DataFrame):
    df = df.sort_values("wf_integral", ascending=False)
    return df


class ListDataset(SparseEventDataset):
    def __init__(
        self,
        filename="default",
        size=-1,
        df_transforms=[],
        max_triggered_doms=100,
        verbose=False,
    ):
        super().__init__(filename, size, df_transforms, verbose)

        self.max_triggered_doms = max_triggered_doms
        df_transforms.insert(0, sort_df_by_brightness)

    def __getitem__(self, idx):
        frame_idx = idx
        event = self._df_loc(frame_idx).astype("float32")
        event = self._apply_transforms(event)

        tensor_shape = (
            len(self.value_columns),
            bin_config.max_xy_degeneracy,
            self.count_t_bins + 1,
            bin_config.count_z_bins + 1,
        )

        triggered_doms, measurements = event.values.shape
        if triggered_doms >= self.max_triggered_doms:
            tensor = torch.from_numpy(event.values[: self.max_triggered_doms, :])
        elif triggered_doms < self.max_triggered_doms:
            tensor = torch.zeros([self.max_triggered_doms, measurements])
            tensor[:triggered_doms, :] = torch.from_numpy(event.values)

        return tensor
