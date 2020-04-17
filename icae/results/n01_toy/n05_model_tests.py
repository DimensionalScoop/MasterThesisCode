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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy

from icae.tools.config_loader import config
import icae.results.n01_toy.n03_mode_training as training
from icae.tools.torch.gym import Gym
import icae.tools.loss.EMD as EMD
from icae.tools.dataset.single import SingleWaveformPreprocessing
import icae.interactive_setup as interactive
from icae.models.waveform.simple import ConvAE
from icae.tools.dataset.MC import MCDataset
from icae.tools import analysis

plt.set_plot_path(__file__)
interactive.set_saved_value_yaml_root_by_filename(__file__)

#%%
filename = config.root + config.MC.filename
device = torch.device("cuda")
dl_config = {
    "batch_size": 32 * 10,
    "num_workers": 12,
    "shuffle": True,
    # "collate_fn": MCDataset.collate_fn,
    "pin_memory": True,
}

dataset_train_wo_outliers = MCDataset(
    filename=filename,
    key="train/waveforms",
    transform=SingleWaveformPreprocessing(),
    MC_types=0,  # only train on valid events
)
dataset_val = MCDataset(
    filename=filename, key="val/waveforms", transform=SingleWaveformPreprocessing()
)
train_wo_outliers = DataLoader(dataset_train_wo_outliers, **dl_config)
val = DataLoader(dataset_val, batch_size=1024, num_workers=12)

val_classes = dataset_val.df.MC_type

#%%
while True:
    for name,batches in zip(["1e5","1e6"],[1000,10000]):
        gym_factory = lambda: Gym(
            training.best_model_class(),
            device,
            EMD.torch_auto,
            train_wo_outliers,
            val,
            verbose=False,
        )
        interactive.save_value(
            "waveforms used for training wo outliers "+name,
            batches * gym_factory().data_train.batch_size,
        )
        r = analysis.TrainingStability(
            "Ribbles_wo_outliers_"+name, gym_factory, batches=batches, auc_classes=val_classes
        )
        r.run(20)