#%%
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.utils import pin_in_object_store
from ray import tune

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time

import sklearn.metrics as metrics

import sys

from icae.tools.config_loader import config

import icae.tools.loss.EMD as EMD
from icae.tools.dataset.MC import MCDataset
from icae.tools.dataset.single import SingleWaveformPreprocessing
from icae.tools.torch.gym import Gym
from icae.tools.loss import sparse as loss_sparse
import icae.interactive_setup
import icae.results.n01_toy.n03_mode_training as training
from icae.tools.torch.gym import Gym
import icae.tools.loss.EMD as EMD
import icae.interactive_setup as interactive
from icae.models.waveform.simple import ConvAE
from icae.tools.analysis import calc_auc, plot_auc, TrainingStability

plt.set_plot_path(__file__)
interactive.set_saved_value_yaml_root_by_filename(__file__)

#%%
filename = config.root + config.MC.filename
device = torch.device("cuda")

dl_config = {
    "batch_size": 32*10,
    "num_workers": 12,
    "shuffle": True,
    # "collate_fn": MCDataset.collate_fn,
    "pin_memory": True,
}

dataset_train = MCDataset(
    filename=filename, key="train/waveforms", transform=SingleWaveformPreprocessing()
)
dataset_val = MCDataset(
    filename=filename, key="val/waveforms", transform=SingleWaveformPreprocessing()
)
train = DataLoader(dataset_train, **dl_config)
val = DataLoader(dataset_val, batch_size=1024, num_workers=12)
#%%
runs = TrainingStability("Ribbles_w_outliers_1e6",None)
runs.load()

#%%
def argmedian(array):
    return np.argsort(array)[len(array)//2]
best_run = argmedian(runs.loss_train)
best_auc = np.argmax(runs.auc)
gym = training.best_model_factory()
model = gym.model
model.load_state_dict(runs.model[best_run].to_dict())
model.eval()

#%%
loss = runs.losses_train[best_run]
x = np.linspace(1, (len(loss)+ 1) * dl_config["batch_size"], len(loss))
plt.plot(x, loss, label="training loss")
plt.xlabel("# of waveforms used for training")
plt.ylabel(f"EMD loss")
plt.xscale("log")
plt.show_and_save("loss during training")

# %%
val_losses = runs.loss_val[best_run]


names = dataset_val.df["MC_name"].values
unames = np.unique(names)
data = [val_losses[names == name] for name in unames]

name_translator = {
    "valid":"normal",
    "gaussian_reshaped":"Gaussian reshaped",
    "double pulse":"double peak"
}
label_names = [name_translator[n] for n in unames]

_, bins = np.histogram(
    val_losses, int(np.sqrt(len(val_losses))), density=True #, label="all waveforms"
)

hists = []
for d in data:
    h,_ = np.histogram(d,bins = bins,density=True)
    hists.append(h)
combined = hists[0]*0.025 + hists[1]*0.025 + hists[2]*0.95
width=(bins[1]-bins[0])
combined /= combined.sum()
combined /= width
#plt.bar((bins[:-1]+bins[1:])/2,combined,width=width,label="combined",alpha=0.2)

plt.hist(data, bins=bins, label=label_names, histtype="step", density=True)

#h,_ = np.histogram(np.concatenate([data[0],data[1]]),bins = bins,density=True)
#plt.bar((bins[:-1]+bins[1:])/2,h/19,width=width,label="combined",alpha=0.2)


plt.xlabel("EMD loss")
plt.ylabel("frequency")
plt.legend()
#plt.ylim(0,2)
plt.xlim(0,0.1)
#plt.yscale('log')
plt.show_and_save(f"loss_hist_compare")

#%%
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    taken from https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

# re-weighted to 5% outliers
weights = np.concatenate([np.repeat(0.025,len(data[0])),np.repeat(0.025,len(data[1])),np.repeat(0.95,len(data[2]))])
d = np.concatenate(data)
mean,std = weighted_avg_and_std(d,weights)

peak_index = np.argmax(combined)
peak = (bins[peak_index] + bins[peak_index+1])/2

interactive.save_value("mean val loss reweighted",mean,".1e")
interactive.save_value("std val loss reweighted",std,".1e")
interactive.save_value("peak val loss reweighted",peak,".1e")

# %%
truth = (names != "valid").astype("int")
pred = val_losses
fpr, tpr, _ = metrics.roc_curve(truth, pred)
auc = metrics.auc(fpr, tpr)

lw = 2
plt.plot(
    fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([-0.005, 1.0])
plt.ylim([0.0, 1.007])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show_and_save(f"ROC")