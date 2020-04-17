#%%
import icae.interactive_setup

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision as tv
import fastai.vision as faiv
import fastai.train as fait
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import pandas as pd

import icae.toy.lone_point_models as models

#%%
path = "data/toy/1024/"
data_df = pd.read_feather(path + "desc.feather")
label_columns = list(data_df.columns[data_df.columns.str.contains("label_*")])
data = (
    faiv.ImageList.from_df(data_df, path)
    .split_by_rand_pct(0.01)
    .label_from_df(cols=label_columns, label_cls=faiv.FloatList)
    .databunch(bs=10)
)

shape = list(data.one_batch()[0].size())
model = models.SimpleClassifier(
    shape[2:],
    50,
    2,
    # 2,
    kernel=[3, 3],
    channel_progression=lambda x: x + 1,
    batch_normalization=True,
    conv_init=nn.init.zeros_,
    cut_channels=True,
)

learner = faiv.Learner(data, model)

learner.load("model")
#%%
learner.validate()

#%%
learner.recorder

#%%
