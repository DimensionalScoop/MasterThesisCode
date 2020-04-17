# -*- coding: utf-8 -*-
"""like 02a but instead of loading everything into RAM it uses on the fly loading to reduce the memory footprint"""

import matplotlib.pyplot as plt
# +
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from icae.models import single_event as AE_models
from icae.tools import nn, status_report, AE_training, AE_single as AEs_tools, data_loader, on_the_fly_loader
from icae.tools.config_loader import config

# %load_ext autoreload
# %autoreload 2
# -
# +
# in_file = config.root + config.data.retabled_single
# import dask.dataframe as dd
# ddf = dd.read_hdf(in_file, key=config.data.hdf_key)
# unique = ddf.reset_index()['frame'].unique()
# len(unique.compute())
# OUT: 393750
# -

df = pd.read_hdf(
    config.root + config.data.retabled_single, mode="r", start=0, stop=1000
)

data_cols = list("t=%d" % i for i in range(128))
df[data_cols].iloc[10].values


class SingleEventsDataset(Dataset):
    def __init__(self, size, transform=None):
        data_cols = list("t=%d" % i for i in range(128))

        self.file = config.root + config.data.retabled_single
        self.size = size
        self.df = pd.read_hdf(self.file, mode="r", start=0, stop=size)[data_cols]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        waveform = self.df.iloc[idx, :].values
        waveform = waveform.astype("float").reshape(-1, 1)
        sample = {"waveform": waveform}
        if self.transform:
            sample = self.transform(sample)

        return sample


class SingleEventsDatasetOTF(Dataset):
    def __init__(self, size, transform=None):
        self.data_cols = list("t=%d" % i for i in range(128))

        self.file = config.root + config.data.retabled_single
        self.size = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        waveform = pd.read_hdf(self.file, mode="r", start=idx, stop=idx + 1)[
            self.data_cols
        ].values
        waveform = waveform.astype("float").reshape(-1, 1)
        sample = {"waveform": waveform}
        if self.transform:
            sample = self.transform(sample)

        return sample


class Preprocessing:
    def __init__(self):
        pass

    def __call__(self, sample):
        waveform = sample["waveform"]
        waveform = torch.from_numpy(waveform)
        waveform = waveform / waveform.mean()

        return {"waveform": waveform}


dataset = SingleEventsDataset(5000, transforms.Compose([Preprocessing()]))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=12)

# %%time
for i_batch, sample_batched in enumerate(tqdm(dataloader)):
    a, b = i_batch, sample_batched

in_file = config.root + config.data.retabled_single
data = on_the_fly_loader.DataRepresentation(in_file, batch_size=1000)

# %%timeit
df = data.validation_sample()

# %%timeit
df = data.validation_sample_chunk()

model, encoder = AE_models.optimal_NB(3)  # , loss_method = loss_method)

hist = []
tmp_hist = model.fit_generator(data.get_config(), steps_per_epoch=10)
hist.append(tmp_hist)
nn.plot_history(hist, True)

hist = []
tmp_hist = model.fit(**data.get_config(), epochs=10, verbose=1)
hist.append(tmp_hist)
nn.plot_history(hist, True)

status_report.init(model, "test: OTF loader, optimal_NB", "-")
status_report.save_plot("loss")

# data = AE_lib.preprocess(AE_lib.load_mc())
AE_training.plot_results(model, data.validation_sample(1000))
status_report.save_plot("overview-no-translation")

ß
inlier, outlier = AEs_tools.seperate_outliers(model, data)
status_report.save_plot("outlier-seperation")
status_report.save_obj(
    {"inlier indices": inlier, "outlier indices": outlier}, "inlier-outlier-indices"
)

tools.plot_data.plot_latent_space(encoder, data[:10000000], log=True)
status_report.save_plot("latent-space")


def plot_loss_waveform_overview(model, data, max_loss=50, log=False):
    pred = np.array(model.predict(data), dtype=float)
    loss = tools.loss.EMD.numpy(data, pred).flatten()

    plt.figure(figsize=[20, 20])

    min_loss = loss.min()
    steps = 10
    window = (max_loss - min_loss) / steps
    loss_intervals = np.array(
        [
            np.linspace(min_loss, max_loss - window, 10),
            np.linspace(min_loss, max_loss - window, 10) + window,
        ]
    ).T

    for col, interval in enumerate(loss_intervals):
        a, b = interval
        d = data[(loss > a) & (loss < b)]
        l = loss[(loss > a) & (loss < b)]
        for row in range(np.min([10, len(d)])):
            plt.subplot(10, 10, 1 + row * 10 + col)
            i = np.random.randint(0, len(d))
            plt.title("loss: %d" % (l[i]))
            plt.plot(d[i])
            plt.axis("off")
    plt.tight_layout()

    plt.subplot(11, 1, 11)
    plt.hist(loss, len(loss) // 100)
    plt.xlabel("loss")
    plt.ylabel("number of events")
    if log:
        plt.yscale("log")
        plt.grid()
    plt.xlim(min_loss, max_loss)


plot_loss_waveform_overview(model, data, 100)
status_report.save_plot("waveform-sampling-vs-loss")

# +
pred = np.array(model.predict(data), dtype=float)
loss = tools.loss.EMD.numpy(data, pred).flatten()
le_cut = 30

clean_data = data[loss <= le_cut]
clean_loss = loss[loss <= le_cut]
clean_latent = encoder.predict(clean_data)
clean_data.shape[0] / data.shape[0] * 100
# -

clean_model, clean_encoder = AE_models.optimal_simple(3, loss_method=loss_method)
AE_training.train(clean_model, clean_data, epochs=5, batch_size=10, verbose=1)

tools.plot_data.plot_latent_space(clean_encoder, clean_data, log=True)
status_report.save_plot("latent-space-clean-extra-train")

AE_training.plot_results(clean_model, clean_data)
status_report.save_plot("overview-superclean")

plot_loss_waveform_overview(clean_model, clean_data, 50)
status_report.save_plot("waveform-sampling-vs-loss-clean")

status_report.save_model(clean_model, "clean")

# +
pred = np.array(clean_model.predict(clean_data), dtype=float)
clean_loss = tools.loss.EMD.numpy(clean_data, pred).flatten()
le_cut = 12

ultraclean_data = clean_data[clean_loss <= le_cut]
ultraclean_loss = clean_loss[clean_loss <= le_cut]
ultraclean_latent = encoder.predict(ultraclean_data)
ultraclean_integral = ultraclean_data.sum(axis=1)
ultraclean_data.shape[0] / data.shape[0] * 100
# -

ultraclean_model, ultraclean_encoder = AE_models.optimal_simple(
    3, loss_method=loss_method
)
AE_training.train(
    ultraclean_model, ultraclean_data, epochs=10, batch_size=10, verbose=1
)

ultraclean_model.save("final model.hdf", overwrite=False)

tools.plot_data.plot_latent_space(ultraclean_encoder, ultraclean_data, log=True)
status_report.save_plot("latent-space-ultraclean-extra-train")

AE_training.plot_results(ultraclean_model, ultraclean_data)
status_report.save_plot("overview-ultraclean")

status_report.save_model(ultraclean_model, "final AE ")
status_report.save_model(ultraclean_encoder, "final encoder ")

plot_loss_waveform_overview(ultraclean_model, ultraclean_data, 30)
status_report.save_plot("waveform-sampling-vs-loss-ultraclean")

plot_loss_waveform_overview(
    ultraclean_model, AE_training.preprocess(data_loader.load_raw()), 300
)
status_report.save_plot("waveform-sampling-ultraclean-training-all-data-300")

plot_loss_waveform_overview(
    ultraclean_model, AE_training.preprocess(data_loader.load_raw()), 40
)
status_report.save_plot("waveform-sampling-ultraclean-training-all-data-40")

plot_loss_waveform_overview(
    ultraclean_model, AE_training.preprocess(data_loader.load_raw()), 1000, log=True
)
status_report.save_plot("waveform-sampling-ultraclean-training-all-data-1k-log")

# TODO: output model?
