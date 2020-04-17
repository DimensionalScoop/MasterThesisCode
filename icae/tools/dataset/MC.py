#%%
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from icae.tools.config_loader import config
import icae.toy.waveformMC as toy


class MCDataset(Dataset):
    def __init__(
        self,
        filename=None,
        df=None,
        key=None,
        transform=None,
        MC_types="all",
        use_gpu=False,
        device=None
    ):
        if df and filename:
            raise ValueError("Either `df` or `filename` must be specified")
        elif df:
            self.df=df
        elif filename:
            self.df = pd.read_hdf(filename, key)
        else:
            raise ValueError("Either `df` or `filename` must be specified")
        
        if MC_types != "all":
            self.df = self.df[self.df.MC_type==MC_types]
        self.transform = transform
        self.size = len(self.df)
        self.mc_type_values = self.df.MC_type.values.astype("float32")
        self.values = self.df.values[:,-128:].astype("float32")
        self.use_gpu = use_gpu
        if use_gpu:
            print("copying to gpu")
            self.values = torch.tensor(self.values,device=device)
            print("done")

    def __len__(self):
        return self.size

    def __getitem__(self,idx):
        return self._getitem_fast(idx)

    def _getitem_slow(self, idx):
        waveform = self.df.loc[idx, toy.Generator.wf_columns].values.astype("float32")
        waveform = waveform.reshape(1,128) # fake batch dim
        MC_type = self.df.loc[idx, 'MC_type'].astype("float32")
        if self.transform:
            waveform = self.transform(waveform)

        return {'data':waveform, 'MC_type':MC_type}
    
    def _getitem_fast(self, idx):
        waveform = self.values[idx]
        waveform = waveform.reshape(1,128) # fake batch dim
        MC_type = self.mc_type_values[idx]
        if self.transform:
            waveform = self.transform(waveform)

        return {'data':waveform, 'MC_type':MC_type}

    @staticmethod
    def collate_fn(inputs):
        output = torch.stack(inputs)
        return output.view(-1, inputs[0].shape[-1])




# # %%
# from icae.tools.config_loader import config
# filename = config.root + config.MC.filename
# dataset_train = MCDataset(
#     filename=filename, key="val/waveforms", )

# # %%
# %%timeit
# dataset_train.__getitem__(100)

# # %%
# %%timeit
# dataset_train._getitem_fast(100)


# # %%
# import numpy.testing as npt
# for i in range(100):
#     npt.assert_equal(dataset_train.__getitem__(i),dataset_train._getitem_fast(i))

# # %%
