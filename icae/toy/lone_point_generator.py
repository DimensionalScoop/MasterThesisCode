import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

from icae.tools import performance


class LonePoint(Dataset):
    def __init__(self, edge_length, dims, use_channels=True):
        self.edge_length = edge_length
        self.shape = [edge_length] * dims
        self.use_channels = use_channels
        if use_channels:  # add a fake channel dimension
            self.shape = [1] + self.shape
        self.size = edge_length ** dims

        # fastai compatibility?
        self.c = None

    def get_dataloader(self, batch_size=8, **kwargs):
        default_config = {
            "batch_size": batch_size,
            "num_workers": 1,  # 12,
            "pin_memory": True,
        }
        default_config.update(*kwargs)
        dataloader = torch.utils.data.DataLoader(self, shuffle=True, **default_config)
        return dataloader

    def __getitem__(self, idx):
        item = torch.zeros(self.shape)
        index = np.unravel_index(idx, shape=self.shape)
        item[index] = 1
        return_value = {
            "label": torch.tensor(idx / self.size).float(),
            "coords": torch.tensor(index).float(),
            "data": item,
        }
        if self.use_channels:  # drop useless first coordinate
            return_value["coords"] = torch.tensor(index[1:]).float()
        return_value["coords"] /= self.edge_length
        return return_value

    def __len__(self):
        return self.size

    def expected_mse(self, noise_level=0.01):
        sample_idx = 0  # arbitrary index, doesn't matter for mse
        sample = self.__getitem__(sample_idx)["data"]
        noise = torch.rand(self.shape) * noise_level
        return F.mse_loss(sample, sample + noise)

    def coords_to_data(self, coords):
        item = torch.zeros(self.shape)
        item[coords] = 1
        return item

    def to_disk(self, path, samples=1000):
        def save(index):
            d = self.__getitem__(index)
            filename = str(index) + ".png"
            utils.save_image(d["data"], path + filename)
            return [filename] + d["coords"].tolist()

        samples = np.random.randint(0, self.__len__(), size=samples)
        saved_idx = performance.parallize(save, samples)
        columns = ["filename"] + [
            "label_" + str(i) for i, l in enumerate(saved_idx[0][1:])
        ]
        pd.DataFrame(saved_idx, columns=columns).to_feather(path + "desc.feather")


class LonePointValidation(LonePoint):
    def __init__(self, edge_length, dims, size=100, use_channels=True):
        super().__init__(edge_length, dims, use_channels)

        self.validation_size = size
        self.size_factor = self.size / self.validation_size

    def __len__(self):
        return self.validation_size

    def __getitem__(self, idx):
        actual_idx = int(idx * self.size_factor)
        assert actual_idx < self.size
        return super().__getitem__(actual_idx)

    def to_disk(self, path, samples=1000):
        raise NotImplemented()


if __name__ == "__main__":
    test_config = {"edge_length": 1024, "dims": 2}

    dataloader_config = {
        "batch_size": 10,
        "num_workers": 12,
        # 'collate_fn': SparseEventDataset.collate_fn,
        "pin_memory": True,
    }

    path = "data/toy/1024/"
    data_set = LonePoint(**test_config)
    data_set.to_disk(path, 50000)
    sys.exit()
