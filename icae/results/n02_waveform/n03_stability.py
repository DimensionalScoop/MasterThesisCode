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
batch_size = 320
r = TrainingStability("Conv_on_ICMC_correct_batch_size_3", None)
r.load()
#%%
training_times = np.array([len(i) for i in r.losses_train])
unique_training_times = np.unique(training_times)
unique_training_times_names = [
    f"$n={int(t*batch_size/10**int(np.log10(t*batch_size)))} \cdot"
    + "10^{"
    + f"{int(np.log10(t*batch_size))}"
    + "}$"
    for t in unique_training_times
]

losses = []
for t in unique_training_times:
    losses.append(r.loss_val[training_times == t].mean(axis=1))

# plt.hist(losses,label=unique_training_times*batch_size);
# plt.legend()
# plt.clf()
#%%
def remove_nan_inf(array):
    array = array[array == array]
    array = array[array != np.inf]
    array = array[array != -np.inf]
    return array


#%%
equivalent_1e6 = unique_training_times[-2]
data = r.loss_val[training_times == equivalent_1e6].mean(axis=1)
# in some rare cases, model training fails completely
#data = remove_nan_inf(data)
#data = data[data>=0]

#high_loss = data[data > 1]
low_loss = data#[data < 1]
bins = int(np.sqrt(len(data)))

#plt.subplot(2, 1, 1)
plt.hist(low_loss, bins=bins)
plt.xlabel("EMD loss")
plt.ylabel("count")
plt.yscale("log")
#plt.xlim(0,max(low_loss)*1.01)

# plt.subplot(2, 1, 2)
# plt.hist(data[data > 0], bins=bins)
# plt.xlabel("EMD loss")
# plt.ylabel("count")
# plt.yscale("log")

plt.show_and_save("loss hist 1e6")

interactive.save_value("loss hist 1e6 number of models", len(data), ".1e")
interactive.save_value(
    "number of waveforms for loss hist 1e6 plot", equivalent_1e6 * batch_size, ".1e"
)
interactive.save_value(
    "percentage of bad models", len(data[data > 0.07]) / len(data) * 100, ".1f"
)

interactive.save_value(
    "median loss for 1e6", np.median(data), ".3f"
)

#%%
# remove failed models
#filter = (r.loss_val.mean()>0)&(r.loss_val.mean()<1)&(r.loss_val.mean()==r.loss_val.mean())
#r.loss_val = r.loss_val[filter]
#r.losses_train = r.loss_val[filter]
#r.loss_train = r.loss_train[filter]

#%%
means = []
stds = []
for t in unique_training_times:
    filter = training_times == t
    m = r.loss_val[filter].mean(axis=1)
    #m = m[(m==m)&(m<np.inf)] # remove nans, infs
    means.append(m.mean())
    stds.append(m.std(ddof=1) / np.sqrt(len(m)))

plt.errorbar(unique_training_times * batch_size, means, stds)
plt.xlabel("# of training waveforms")
plt.ylabel("EMD loss")
plt.xscale("log")
plt.show_and_save("training time vs loss")
interactive.save_value(
    "training time vs loss models per point",
    r.loss_val.shape[0] / len(unique_training_times),
    ".1e",
)

#%%
val_losses = []
training_losses = []
for t in unique_training_times:
    filter = training_times == t

    def median_index(input):
        return np.argsort(input)[len(input) // 2]

    model_index = median_index(r.loss_val[filter].mean(axis=1))
    losses = r.loss_val[filter][model_index]
    val_losses.append(losses)
    training_losses.append(r.losses_train[filter][model_index])
#%%
plt.subplot(1, 2, 1)
data = val_losses[:3]
labels = unique_training_times_names[:3]
bins = int(np.sqrt(val_losses[0].shape[0])) // 5
_, edges, _ = plt.hist(data, bins=bins, label=labels, density=True, histtype="step")
plt.xlabel("EMD loss")
plt.ylabel("frequency")
plt.xlim(0, 0.2)
plt.legend()

plt.subplot(1, 2, 2)
data = val_losses[2:]
labels = unique_training_times_names[2:]
colors = ["C2", "C3", "C4"]

_, edges, _ = plt.hist(
    data, bins=edges, label=labels, density=True, histtype="step", color=colors
)
plt.xlabel("EMD loss")
plt.ylabel("frequency")
plt.xlim(0, 0.1)
plt.legend()

plt.show_and_save("loss hists")
#%%
peaks = []
quantiles = []
aucs = []
times = unique_training_times * batch_size
for data, time in zip(val_losses, times):
    hist, _ = np.histogram(data, bins=edges)
    peaks.append(edges[np.argmax(hist)])
    quantiles.append(np.quantile(data, 0.95))

plt.subplot(2, 1, 1)
plt.plot(times, peaks, label="peak")
plt.plot(times, quantiles, label="95% quantile")
plt.xscale("log")
plt.ylim(0, 0.1)
# plt.yscale('log')
plt.legend()
plt.xlabel("# of training waveforms")
plt.ylabel("EMD loss")

plt.subplot(2, 1, 2)
plt.plot(times, peaks, label="peak")
plt.plot(times, quantiles, label="95% quantile")
plt.xscale("log")
# plt.ylim(0,0.05)
# plt.yscale('log')
plt.legend()
plt.xlabel("# of training waveforms")
plt.ylabel("EMD loss")

plt.show_and_save("loss peak and quant")

#%%
peaks = []
quantiles = []
aucs = []
times = unique_training_times * batch_size
for data, time in zip(val_losses, times):
    hist, _ = np.histogram(data, bins=edges)
    peaks.append(edges[np.argmax(hist)])
    quantiles.append(np.quantile(data, 0.8))

plt.plot(times, peaks, label="peak")
plt.plot(times, quantiles, label="80% quantile")
plt.xscale("log")
plt.legend()
# plt.yscale('log')
plt.xlabel("# of training waveforms")
plt.ylabel("EMD loss")
plt.show_and_save("loss peak and quant")

#%%
loss_of_best_1e6_equivalent = training_losses[-2]
x = np.linspace(
    1, 1 + training_times[-1] * batch_size, len(loss_of_best_1e6_equivalent)
)
plt.plot(x, loss_of_best_1e6_equivalent)
plt.xlabel("# of waveforms")
plt.ylabel("EMD loss")
plt.xscale("log")
plt.show_and_save("loss train")
#%%
hist, edges = np.histogram(
    loss_of_best_1e6_equivalent, bins=int(np.sqrt(len(loss_of_best_1e6_equivalent)))
)
peak_index = np.argmax(hist)
peak = (edges[peak_index] + edges[peak_index + 1]) / 2
mean = np.mean(loss_of_best_1e6_equivalent)
std = np.std(loss_of_best_1e6_equivalent)
interactive.save_value("mean val loss", mean, ".1e")
interactive.save_value("peak val loss", peak, ".1e")
interactive.save_value("std val loss", std, ".1e")


#%%
# count_training_waveforms = (steps + 1) *batch_loading_size*train.batch_size
# x = np.linspace(0, count_training_waveforms, len(loss))
plt.plot(x, loss)
plt.xlabel("# of waveforms used for training")
plt.ylabel(f"loss {lossname}")
plt.xscale("log")
plt.figtext(0, 0, name)
plt.show_and_save(f"{name} + training")
plt.clf()

# %%
dataset_val = SingleWaveformDataset(
    load_waveform_only=False,
    transform=SingleWaveformPreprocessing(),
    batch_loading_size=1024,
    split=1,
)
# %%
xs = []
ys = []
zs = []
for d in tqdm(dataset_val):
    data, info = d["data"], d["info"]
    xs.extend(info["x"])
    ys.extend(info["y"])
    zs.extend(info["z"])

# %%
plt.hist2d(val_losses[-1], zs[: len(val_losses[-1])], bins=80)

# %%
_, edges = np.histogram(zs[: len(val_losses[-3])], bins=80)
bin_indices = np.digitize(zs[: len(val_losses[-3])], edges)
bins = sorted(np.unique(bin_indices))
means = []
stds = []
for i in bins:
    losses_in_this_layer = val_losses[-3][bin_indices == i]
    means.append(np.quantile(losses_in_this_layer, 0.5))
    stds.append(losses_in_this_layer.std(ddof=1) / np.sqrt(len(losses_in_this_layer)))
plt.errorbar(bins, means, stds)
# %%
unique, count = np.unique(bin_indices, return_counts=True)
weights = np.zeros_like(bin_indices)
for i, c in zip(unique, count):
    weights[bin_indices == i] = c

plt.hist(bin_indices, weights=1 / weights)
# %%
plt.hist2d(val_losses[-3], bin_indices, weights=1 / weights, bins=80)


# %%

