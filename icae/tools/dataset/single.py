# -*- coding: utf-8 -*-
# + {}
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from icae.tools.config_loader import config


class SingleWaveformDataset(Dataset):
    data_cols = list("t=%d" % i for i in range(128))
    split_name = ["train", "val", "test"]

    def __init__(
        self,
        filename=config.root + config.data.main,
        size=-1,
        batch_loading_size=1,
        transform=None,
        train_val_test_split=[0.9, 0.09, 0.01],
        split=0,
        load_waveform_only=True,
    ):
        """
        If `size=-1`, all available events are used.
        `batch_loading_size` determines how many waveforms are represented
        by one index. For `batch_loading_size>1`, use the provided `collate_fn`
        with the torch.utils.data.DataLoader.
        Big `batch_loading_size` speed up training, but mini-batches tend to look
        the same. This can be somewhat alleviated by using `set_index_offset`.
        """
        if filename == "default":
            self.file = config.root + config.data.retabled_single
        else:
            self.file = filename
        self.store = pd.HDFStore(self.file, mode="r")
        self.table = self.store.get_storer("events")
        self.load_waveform_only = load_waveform_only
        self.column_names = None

        self.batch_loading_size = batch_loading_size
        self.transform = transform

        if size == -1:
            size = self.table.nrows
            print("dataset size %e" % size)
        self.train_val_test_split = train_val_test_split
        self.split = split
        self.split_size = train_val_test_split[split]
        self.split_start = int(np.cumsum([0] + train_val_test_split)[split] * size)
        # self.data_end = np.cumsum(train_val_test_split)[split]

        size = int(size * self.split_size)
        self.size = size // batch_loading_size - 1
        # the subtraction is due to the index_offset
        assert self.table.nrows >= self.size * batch_loading_size

        self.index_offset = 0

        assert self.size > 0
        print(
            SingleWaveformDataset.split_name[self.split],
            "\tsize of",
            size,
            "\tbatches a",
            batch_loading_size,
        )

    def __len__(self):
        return self.size

    def __read_from_disk(self, idx):
        start = idx * self.batch_loading_size + self.index_offset + self.split_start
        args = {}
        if self.load_waveform_only:
            args["columns"] = SingleWaveformDataset.data_cols

        # with pd.HDFStore(self.file, "r") as store:
        df = pd.read_hdf(
            self.store,
            key="events",
            mode="r",
            start=start,
            stop=start + self.batch_loading_size,
            **args
        )

        if self.column_names is not None:
            self.column_names = np.array(df.columns.values)
        return df

    def __getitem__(self, idx):
        df = self.__read_from_disk(idx)
        sample = {}
        if self.load_waveform_only:
            sample["data"] = df.values.astype("float32")
        else:
            sample["data"] = df[self.data_cols].values.astype("float32")
            sample["info"] = df.drop(columns=self.data_cols).to_dict("list")

        if self.transform:
            sample["data"] = self.transform(sample["data"])

        return sample

    def select_columns(self, tensor: torch.Tensor, columns):
        assert (
            self.column_names is not None
        ), "you must read at least once from disk to get the column list"

        columns = np.array(columns)
        mask_single = torch.tensor(columns == self.column_names)
        mask = mask_single.expand(tensor.shape[::-1]).t()

        return tensor[mask]

    def set_index_offset(self, offset):
        assert 0 <= offset <= self.batch_loading_size - 1
        self.index_offset = offset

    @staticmethod
    def collate_fn(inputs):
        output = torch.stack(inputs)
        return output.view(-1, inputs[0].shape[-1])


# +
class SingleWaveformPreprocessing:
    """Transforms each waveform in a way so that the smallest bin will be 0 and the largest 1.
    """

    def __init__(self):
        pass

    def __call__(self, waveform):
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform)
        shape = waveform.shape[::-1]

        min_of_each_vector = waveform.min(dim=1).values
        waveform = waveform - min_of_each_vector.expand(shape).t()  # R → R^+
        max_of_each_vector = waveform.max(dim=1).values
        waveform = waveform / max_of_each_vector.expand(shape).t()  # R^+ → [0,1]

        infs = torch.isinf(waveform)
        nans = torch.isnan(waveform)

        if infs.sum():
            warnings.warn("Infinities in input data")
        if nans.sum():
            warnings.warn("Nans in input data")

        if nans.sum() or infs.sum():
            waveform[nans] = 0  # remove nans
            waveform[infs] = 0  # remove nans
            waveform[
                nans | infs
            ] = waveform.mean()  # replace them with mean (torch has no ignore_nan_mean)

        return waveform

    # areas = waveform.sum(dim=1)
    # waveform = waveform / areas.expand(shape).t()


# -


if __name__ == "__main__":
    dataset = SingleWaveformDataset(
        -1, 128, transforms.Compose([SingleWaveformPreprocessing()])
    )
    train_loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=12,
        collate_fn=dataset.collate_fn,
    )

    #%%time
    for i in tqdm(train_loader):
        i

