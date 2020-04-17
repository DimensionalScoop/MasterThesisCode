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


bin_config = Box.from_yaml(filename=config.root + config.data.binning_config)


class SparseEventDataset(Dataset):
    index_columns = ["xy_pc_sorting_index", "t_bin", "z_bin"]
    value_columns = [
        "wf_latent_0",
        "wf_latent_1",
        "wf_latent_2",
        "wf_AE_loss",
        "wf_integral",
    ]
    split_name = ["train", "val", "test"]

    def __init__(
        self,
        filename="default",
        size=-1,
        df_transforms=None,
        count_channels=5,
        verbose=False,
    ):
        """
        If `size=-1`, all available events are used. Otherwise, only `size`
        events are used.
        """
        if filename == "default":
            self.file = config.root + config.data.preprocessed_events
        else:
            self.file = filename

        if size == -1:
            print("Reading whole df…", end="")
            self.df: pd.DataFrame = pd.read_hdf(self.file)
            print(" Done.")
        else:
            mean_rows_per_event = 176 * 4  # (real mean + buffer)
            self.df = pd.read_hdf(self.file, stop=mean_rows_per_event * size)

        if size == -1:
            self.frames = self.df.frame.unique()
            size = len(self.frames)  # number of events in sample
        else:
            self.df = self.df[self.df.frame < size]
            self.frames = self.df.frame.unique()
        self.size = size

        self.df = self._prepare_df(self.df)
        self.df_frame = self.df.frame.astype("int")
        self.df.set_index("frame", inplace=True)
        # self.df.drop(columns=['frame']) # self.frames already contains this info

        if verbose:
            print(
                "dataset of %e MB," % (self.df.values.nbytes * 1e-6),
                "containing %e unique events." % len(self.frames),
            )
        del self.frames

        self.df_transforms = df_transforms

        assert self.size > 0

        self.count_t_bins = bin_config.count_t_bins

    def prune(self, dtype="float16"):
        """reduces memory footprint by removing columns not needed for training"""
        self.df = self.df[self.index_columns + self.value_columns]
        self.df = self.df.astype(dtype)

    def limit_event_t_size(self, max_t_bins, strategy="drop"):
        """Drops all events that need more than `max_t_bins`.
        With `strategy="clip"` all events are kept and long events are cut at `max_t_bins`"""

        previous_size = self.df.size
        self.move_to_t0()

        print("Removing many entries, this may take some time…")
        if strategy == "drop":
            grp = self.df.groupby(level=0)
            self.df = grp.filter(lambda g: g["t_bin"].max() < max_t_bins)
        elif strategy == "cut":
            self.df = self.df.drop(self.df["t_bin"] > max_t_bins)
        else:
            raise NotImplemented()

        self.count_t_bins = max_t_bins

        assert self.df["t_bin"].max() <= max_t_bins
        assert self.df["t_bin"].min() >= 0
        print(
            "limit_event_t_size dropped %.2e %% of rows"
            % (100 * (previous_size - self.df.size) / previous_size)
        )

    def move_to_t0(self):
        """moves all `t_bin`s so that the first triggering event has `t_bin=0`"""

        grp = self.df.groupby("frame")
        self.df["t_bin"] = self.df["t_bin"] - grp["t_bin"].min()
        assert np.alltrue(np.isclose(self.df.groupby("frame")["t_bin"].min(), 0))

    def save(self, path, dtype="float32"):
        """saves df always with the given dtype"""
        state = self.__dict__.copy()  # shallow-copy only to conserve memory
        state.pop("df")
        state.pop("df_frame")
        state = Box(state)
        # path = config.root + config.data.sparse_dataset_interim

        self.df.astype(dtype).to_hdf(path + "df.hdf", key="main")
        state.to_yaml(path + "state.yaml")

    @classmethod
    def load(cls, path):
        self = cls.__new__(cls)
        self.__dict__ = Box.from_yaml(filename=path + "state.yaml")
        self.df = pd.read_hdf(path + "df.hdf", key="main")
        self.df_frame = self.df.reset_index().frame.astype("int")
        return self

    def get_train_val_test(
        self, batch_size=5, num_workers=12, train_val_test_split=[0.8, 0.01, 0.19]
    ):
        """Returns three pytorch dataloaders for training, validation and testing."""

        start = [0] + list(np.floor(np.cumsum(train_val_test_split) * len(self)))
        start = [int(i) for i in start]
        indices = list(range(len(self)))
        splits = [indices[start[i] : start[i + 1]] for i in range(3)]
        dataloader_config = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            # 'collate_fn': SparseEventDataset.collate_fn,
            "pin_memory": False,
        }
        # XXX: use bigger batch_loading_size for val/test (but this could break the spacing)
        train = torch.utils.data.DataLoader(
            self,
            sampler=torch.utils.data.SubsetRandomSampler(splits[0]),
            **dataloader_config
        )
        val = torch.utils.data.DataLoader(
            self,
            sampler=torch.utils.data.SubsetRandomSampler(splits[1]),
            **dataloader_config
        )
        test = torch.utils.data.DataLoader(
            self,
            sampler=torch.utils.data.SubsetRandomSampler(splits[2]),
            **dataloader_config
        )

        return train, val, test

    def __len__(self):
        return self.size

    def _apply_transforms(self,event) -> pd.DataFrame:
        if self.df_transforms:
            if callable(self.df_transforms):
                event = self.df_transforms(event)
            else:
                for t in self.df_transforms:
                    event = t(event)
        return event

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
        # where degeneracy and column index are combined
        final_shape = (
            len(self.value_columns) * bin_config.max_xy_degeneracy,
            self.count_t_bins + 1,
            bin_config.count_z_bins + 1,
        )

        table = event[
            self.index_columns + self.value_columns
        ]  # reorder so first three columns are indices

        indices, values = np.split(table.values, [len(self.index_columns)], axis=1)
        value_indices = np.arange(len(self.value_columns))

        # to represent as a tensor with the dimensions
        # [column_index, xy_pc_sorting, t_bin, z_bin],
        # we need to reshape the indices:
        repeated_indices = np.repeat(indices, len(self.value_columns), axis=0)
        repeated_column_indices = np.tile(value_indices, len(table))
        new_indices = np.hstack(
            (repeated_column_indices.reshape([-1, 1]), repeated_indices)
        )
        new_indices = new_indices.astype("int")

        # to represent as a tensor with the dimensions
        # [channel, t_bin, z_bin],
        # we need to reshape the indices.
        # channel is here a combination of column_index and xy_pc_sorting
        channel_indices = new_indices[:, 1:].copy()
        channel_indices[:, 0] += new_indices[:, 0] * tensor_shape[0]

        sparse = torch.sparse_coo_tensor(
            channel_indices.T, values.flatten(), size=final_shape
        )
        
        return sparse.to_dense()

    @staticmethod
    def collate_fn(inputs):
        output = torch.stack(inputs)
        return output.view(-1, *inputs[0].shape)

    def _df_loc(self, frame):
        """identical to `self.df.loc[frame]`.
        It's ~100 times faster because self.frames is sorted and uses numpy.
        """
        a = np.searchsorted(self.df_frame, frame, side="left")
        b = np.searchsorted(self.df_frame, frame, side="right")
        return pd.DataFrame(self.df.values[a:b], columns=self.df.columns)

    def _prepare_df(self, df):
        df = df.astype("float32")  # reduce size, NN don't need much precision

        # some collisions are unavoidable, but those are very few
        before_drop = df.shape[0]
        df = df[~df["bin_collision_index"].astype("bool")]
        print("Dropped", before_drop - df.shape[0], "to get rid of duplicates")

        df.drop(
            columns=["xy_pc_sorting", "bin_collision", "bin_collision_index"],
            inplace=True,
        )

        df = self._scale_df(df)

        return df

    def _scale_df(self, df):
        scaler = MinMaxScaler()
        scale_me = df[self.value_columns]
        df[self.value_columns] = scaler.fit_transform(scale_me)
        return df
