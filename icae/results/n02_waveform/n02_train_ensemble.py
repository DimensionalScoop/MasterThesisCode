#%%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tqdm import tqdm
import pickle
import uuid
from box import Box
from glob import glob
from scipy.stats.stats import pearsonr
import torch
from torch.utils.data import DataLoader

from icae.tools.torch.gym import Gym
import icae.tools.loss.EMD as EMD
import icae.interactive_setup as interactive
from icae.models.waveform.simple import ConvAE
from icae.tools.analysis import calc_auc, plot_auc, TrainingStability
from icae.tools.dataset.single import SingleWaveformDataset, SingleWaveformPreprocessing
from icae.tools.torch.gym import Gym
from icae.models.waveform.simple import ConvAE
from icae.tools.loss import EMD
import icae.interactive_setup

plt.set_plot_path(__file__)
interactive.set_saved_value_yaml_root_by_filename(__file__)
#%%
device = torch.device("cuda")

dataset_train = SingleWaveformDataset(
    load_waveform_only=False,
    transform=SingleWaveformPreprocessing(),
    batch_loading_size=32,
    split=0,
)
dataset_val = SingleWaveformDataset(
    load_waveform_only=False,
    transform=SingleWaveformPreprocessing(),
    batch_loading_size=1024,
    split=1,
)
# num_workers=0: https://github.com/facebookresearch/maskrcnn-benchmark/issues/103
train = DataLoader(dataset_train, shuffle=True, batch_size=10, num_workers=0)
val = DataLoader(dataset_val, batch_size=12, num_workers=0)
del dataset_val.table

#%%
model_config = {
    "device": device,
    "data_train": train,
    "data_val": val,
    "verbose": True,
    "max_validation_steps": 100,
}
gym_factory = lambda: Gym(
    ConvAE(config={"latent_size": 3}), **model_config, loss_func=EMD.torch_auto
)

#%%
while True:
    for batches in [ 1000000/3.2,3000000/3.2]: # 100/3.2, 1000/3.2, 10000/3.2, 10000/3.2, 100000/3.2,
        r = TrainingStability("Conv_on_ICMC_correct_batch_size_3", gym_factory, batches=int(batches))
        r.run(20)

# %%
