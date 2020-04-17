#%%
import pickle
import uuid
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import torch.optim as optim
from box import Box
from scipy.stats.stats import pearsonr
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

import icae.interactive_setup as interactive
import icae.results.n01_toy.n03_mode_training as training
import icae.tools.loss.EMD as EMD
from icae.models.waveform.simple import ConvAE
from icae.tools import analysis
from icae.tools.config_loader import config
from icae.tools.dataset.MC import MCDataset
from icae.tools.dataset.single import SingleWaveformPreprocessing
from icae.tools.torch.gym import Gym

plt.set_plot_path(__file__)
interactive.set_saved_value_yaml_root_by_filename(__file__)
training_set_size = 1e6  # TODO: read from yaml

# %%
r = analysis.TrainingStability("corrected training time vs outliers", None)
r.load()

# %%
times = np.unique(r.count_training_waveforms)
actual_times = []
latents = np.unique(r.latent_size)
time_index = [np.where(r.count_training_waveforms == t) for t in times]
latent_index = [np.where(r.latent_size == s) for s in latents]
# %%
aucs_mean = np.zeros([len(times), len(latents)]) * np.NaN
aucs_best = np.zeros_like(aucs_mean) * np.NaN
aucs_std = np.zeros_like(aucs_mean) * np.NaN
val_loss_mean = np.zeros_like(aucs_mean) * np.NaN
val_loss_mean_std = np.zeros_like(aucs_mean) * np.NaN
train_loss_mean = np.zeros_like(aucs_mean) * np.NaN
n_models = []
for i_t, time_indices in enumerate(time_index):
    for i_l, latent_indices in enumerate(latent_index):
        indices = np.intersect1d(time_indices, latent_indices)
        if len(indices) == 0:
            continue

        def argmedian(array):
            return np.argsort(array)[len(array) // 2]

        best = argmedian(r.loss_train[indices])

        actual_times.append(np.mean([len(x) for x in r.losses_train[indices]]))
        aucs_best[i_t, i_l] = r.auc[indices][best]
        aucs_mean[i_t, i_l] = np.mean(r.auc[indices])
        aucs_std[i_t, i_l] = np.std(r.auc[indices], ddof=1) / np.sqrt(
            len(r.auc[indices])
        )

        mean_val_losses = r.loss_val[indices].mean(axis=1)
        val_loss_mean[i_t, i_l] = mean_val_losses.mean()
        train_loss_mean[i_t, i_l] = r.loss_train[indices].mean()
        val_loss_mean_std[i_t, i_l] = mean_val_losses.std(ddof=1) / np.sqrt(
            len(mean_val_losses)
        )
        n_models.append(len(r.auc[indices]))

interactive.save_value("mean number of models data point", np.mean(n_models), ".1e")
assert len(np.unique(actual_times)) == len(
    np.array(times)
), "models haven't been training for as long as they should have been!"
# %%
for i, latent_dim in enumerate(latents):
    plt.errorbar(
        times, aucs_mean[:, i], yerr=aucs_std[:, i], label=f"latent dim: {latent_dim}"
    )
# TODO: insert line at 1e6 to indicate when training set is exhausted

plt.ylabel("mean AUC")
plt.xlabel("waveforms used for training")
plt.xscale("log")
plt.ylim([0.5, 0.9])
plt.legend()  # loc='lower bottom')

ax1 = plt.gca()
ax2 = ax1.twiny()
ax2.plot(times, times)
labels = [
    "${10^{" + f"{np.log10(i):.0f}" + "}}$"
    for i in ax1.get_xticks() / training_set_size
]
ax2.set_xscale("log")
ax2.set_xticklabels(labels)
ax2.set_xlabel("epochs")

plt.show_and_save("training time vs outliers mean")

#%%
for i, latent_dim in enumerate(latents):
    plt.errorbar(
        times,
        val_loss_mean[:, i],
        yerr=val_loss_mean_std[:, i],
        label=f"latent dim: {latent_dim}",
        color=f"C{i}",
    )

    # plt.plot(times,train_loss_mean[:,i],color=f"C{i}")

plt.ylabel("mean validation EMD loss")
plt.xlabel("waveforms used for training")
plt.xscale("log")
# plt.ylim([0.5,0.9])
plt.legend()  # loc='lower bottom')

ax1 = plt.gca()
ax2 = ax1.twiny()
# ax2.set_xlim([min(times),max(times)])
plt.plot(times, val_loss_mean[:, 0])
labels = [
    "${10^{" + f"{np.log10(i):.0f}" + "}}$"
    for i in ax1.get_xticks() / training_set_size
]
ax2.set_xscale("log")
ax2.set_xticklabels(labels)
ax2.set_xlabel("epochs")

plt.show_and_save("training time vs val mean")

# %%
batch_size = 320
training_times = np.array([len(i) for i in r.losses_train])
unique_training_times = np.unique(training_times)
unique_training_times_names = [
    f"$n={int(t*batch_size/10**int(np.log10(t*batch_size)))} \cdot"
    + "10^{"
    + f"{int(np.log10(t*batch_size))}"
    + "}$"
    for t in unique_training_times
]

val_losses = []
training_losses = []
for t in unique_training_times:
    filter = (training_times == t) & (r.latent_size == 3)

    def median_index(input):
        return np.argsort(input)[len(input) // 2]
    
    model_index = median_index(r.loss_val[filter].mean(axis=1))
    losses = r.loss_val[filter][model_index] #r.losses_train[filter][model_index]
    #losses = losses[int(len(losses) * 0.5) :]
    val_losses.append(losses)
    training_losses.append(r.losses_train[filter][model_index])
#%%
plt.subplot(2,1,1)
data = val_losses[3::2][:-1]
labels = unique_training_times_names[3::2][:-1]
bins = 3 * int(np.sqrt(len(data[0])))
_,edges,_ = plt.hist(data, bins=bins, label=labels, density=True, histtype="step")
plt.xlabel("EMD loss")
plt.ylabel("frequency")
# plt.xscale('log')
plt.xlim(-0.0001, 0.025)
plt.legend()

plt.subplot(2,1,2)
data = [val_losses[-4]] + val_losses[-2:]
labels = [unique_training_times_names[-4]] + unique_training_times_names[-2:]
plt.hist(data, bins=edges, label=labels, density=True, histtype="step",color=["C2","C3","C4"])
plt.xlabel("EMD loss")
plt.ylabel("frequency")
# plt.xscale('log')
plt.xlim(-0.0001, 0.025)
plt.legend()

plt.show_and_save("loss hists")
# %%
peaks = []
quantiles = []
aucs = []
times = unique_training_times*batch_size
for data,time in zip(val_losses,times):
    hist,_ = np.histogram(data,bins=edges)
    peaks.append(edges[np.argmax(hist)])
    quantiles.append(np.quantile(data,0.95))

plt.subplot(2,1,1)
plt.plot(times,peaks,label="peak")
plt.plot(times,quantiles,label="95% quantile")
plt.xscale('log')
plt.ylim(0,0.05)
#plt.yscale('log')
plt.legend()
plt.xlabel("# of training waveforms")
plt.ylabel("EMD loss")

plt.subplot(2,1,2)
plt.plot(times,peaks,label="peak")
plt.plot(times,quantiles,label="95% quantile")
plt.xscale('log')
#plt.ylim(0,0.05)
#plt.yscale('log')
plt.legend()
plt.xlabel("# of training waveforms")
plt.ylabel("EMD loss")

plt.show_and_save("loss peak and quant")

# %%

#%%
fully_trained_by = times[4]
info = []
n_models = []
for i, latent_dim in enumerate(latents):
    time_constrain = np.where(r.count_training_waveforms >= fully_trained_by)
    latent_constrain = np.where(r.latent_size == latent_dim)
    filter = np.intersect1d(time_constrain, latent_constrain)

    pearson, p_val = scipy.stats.pearsonr(
        r.count_training_waveforms[filter], r.auc[filter]
    )
    info.append([latent_dim, pearson, p_val])
    n_models.append(np.sum(filter))

info = pd.DataFrame(
    info, columns=["latent dimension $d_l$", "Pearson correlation $r$", "p-value"]
)
interactive.save_table("training_correlation_all", info, float_format="%.3f")
interactive.save_value(
    "model fully trained after n waveforms", int(fully_trained_by), ".1e"
)
interactive.save_value(
    "number of models used for correlation", np.mean(n_models), ".1e"
)
info

#%%
info = []
for i, latent_dim in enumerate(latents):
    pearson, p_val = scipy.stats.pearsonr(times[4:], aucs_mean[:, i][4:])
    plt.errorbar(
        times[4:],
        aucs_mean[:, i][4:],
        yerr=aucs_std[4:, i],
        label=f"$d_l={latent_dim}$, $r={pearson:.1f}$ (p= {p_val:.1f})",
    )
    info.append([latent_dim, f"{pearson:.3f}", p_val])

info = pd.DataFrame(
    info, columns=["Latent Dimension $d_l$", "Pearson correlation $r$", "p-value"]
)
interactive.save_table("training_correlation_mean", info)

plt.ylabel("mean AUC")
plt.xlabel("waveforms used for training")
plt.xscale("log")
# plt.ylim([0.5,0.9])
plt.legend(loc="lower right")
plt.show_and_save("training time vs outliers zoom")
# %%
for i, latent_dim in enumerate(latents):
    plt.errorbar(times, aucs_best[:, i], label=f"latent dim: {latent_dim}")

plt.ylabel("AUC of best loss")
plt.xlabel("waveforms used for training")
plt.xscale("log")
plt.ylim([0.5, 0.9])
plt.legend()
plt.show_and_save("training time vs outliers best")

# %%
aucs_mean = []
aucs_std = []
aucs_best = []
for i_t, time_indices in enumerate(time_index):
    if len(indices) == 0:
        continue
    best = np.argmin(r.loss_train[time_indices])

    aucs_best.append(r.auc[time_indices][best])
    aucs_mean.append(np.mean(r.auc[time_indices]))
    aucs_std.append(np.std(r.auc[time_indices], ddof=1) / np.sqrt(len(time_indices)))

# %%
plt.errorbar(times, aucs_mean, yerr=aucs_std, label="combined")

plt.ylabel("mean AUC")
plt.xlabel("waveforms used for training")
plt.xscale("log")
plt.ylim([0.5, 1])
plt.legend()
plt.show_and_save("training time vs outliers combined")

# %%
