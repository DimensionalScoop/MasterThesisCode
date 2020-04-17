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
import pandas as pd

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
    "batch_size": 32*10,
    "num_workers": 12,
    "shuffle": True,
    # "collate_fn": MCDataset.collate_fn,
    "pin_memory": True,
}

dataset_peaks = MCDataset(
    filename=filename,
    key="only_peaks/waveforms",
    transform=SingleWaveformPreprocessing(),
)
data = DataLoader(dataset_peaks, batch_size=1024, num_workers=12)
params = pd.read_hdf(filename, "only_peaks/parameters") 

separations = []
for _, row in params.iterrows():
    if row["MC_name"] == "valid":
        sep = 0
    else:
        sep = row["params"]["separation"]
    separations.extend([sep]*row["count"])
separations = np.asarray(separations)

#%%
runs = analysis.TrainingStability("Ribbles_w_outliers_1e6",None)
runs.load()
def argmedian(array):
    return np.argsort(array)[len(array)//2]
best_loss = argmedian(runs.loss_train)
best_auc = np.argmax(runs.auc)
gym = training.best_model_factory()
model = gym.model
model.load_state_dict(runs.model[best_loss].to_dict())
model.eval()

# %%
losses = []
for d in tqdm(data):
    d = d["data"]
    d = d.to(device)
    pred = model(d)
    loss = EMD.torch_auto(pred,d,False).detach().flatten().cpu().numpy()
    losses.append(loss)
losses = np.hstack(losses)

# %%
unique_seps = np.unique(separations)
loss_valid = losses[separations==0]

aucs = []
for sep in unique_seps:
    if sep==0: continue
    loss_peak = losses[separations==sep]
    classes = [0]*len(loss_valid) + [1]*len(loss_peak)
    auc = analysis.calc_auc(np.hstack((loss_valid, loss_peak)),classes)
    aucs.append(auc)

# %%
plt.plot(unique_seps[1:],aucs)
plt.xlabel("peak separation in s")
plt.ylabel("AUC")
plt.show_and_save("peak separation vs auc")

# %%
means = []
stds = []
for sep in unique_seps:
    loss = losses[separations==sep]
    means.append(loss.mean())
    stds.append(loss.std(ddof=1))

means = np.array(means)
stds = np.array(stds)
plt.plot(unique_seps,means,label="mean loss of outlier waveforms")
plt.fill_between(unique_seps,means+stds,means-stds,alpha=0.3, label="standard deviation of loss")

m = loss_valid.mean()
s = loss_valid.std(ddof=1)
plt.plot(unique_seps,[m]*len(unique_seps),label="mean loss of normal waveforms")
plt.fill_between(unique_seps,[m+s]*len(unique_seps),[m-s]*len(unique_seps),alpha=0.3)

plt.xlabel("peak separation in s")
plt.ylabel("EMD loss")
plt.legend(loc="upper left")
plt.show_and_save("peak separation vs loss")

# %%
r,p_val = pearsonr(losses,separations)
interactive.save_value("Pearson correlation of loss to peak separation",r,".2f")
interactive.save_value("Pearson p-value of loss to peak separation",p_val,".2e")

# %%
