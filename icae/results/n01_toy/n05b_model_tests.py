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
class big_runs:
    name = "1e6"
    with_outliers = analysis.TrainingStability("Ribbles_w_outliers_1e6", None)
    with_outliers.load()

    without_outliers = analysis.TrainingStability("Ribbles_wo_outliers_1e6", None)
    without_outliers.load()

    interactive.save_value("ensemble size 1e6 with outliers",len(with_outliers.runs),".1e")
    interactive.save_value("ensemble size 1e6 without outliers",len(without_outliers.runs),".1e")


class small_runs:
    name = "1e5"
    with_outliers = analysis.TrainingStability("Ribbles_w_outliers_1e5", None)
    with_outliers.load()
    
    without_outliers = analysis.TrainingStability("Ribbles_wo_outliers_1e5", None)
    without_outliers.load()

    interactive.save_value("ensemble size 1e5 with outliers",len(with_outliers.runs),".1e")
    interactive.save_value("ensemble size 1e5 without outliers",len(without_outliers.runs),".1e")


# %%
for run in [small_runs, big_runs]:
    data = [run.with_outliers.auc, run.without_outliers.auc[run.without_outliers.auc != None]]
    labels = ["with outliers", "without outliers"]
    plt.hist(data, bins=30, histtype="step", density=True, label=labels)
    plt.xlabel("AUC")
    plt.ylabel("frequency")
    plt.legend(loc="upper left")
    plt.show_and_save("training without outliers - AUC comparison "+run.name)
    KS_statistic, p_value = scipy.stats.ks_2samp(*data,mode='exact')
    interactive.save_value(
        "p-value null hypo AUC of training wo outliers is the same as training with outliers "+run.name,
        p_value,".2e"
    )
    interactive.save_value(
        "ks statistic null hypo AUC of training wo outliers is the same as training with outliers "+run.name,
        KS_statistic,".1e"
    )
    interactive.save_value("mean AUC with outliers "+run.name, f"{data[0].mean():.2f} ")
    interactive.save_value("mean AUC without outliers "+run.name, f"{data[1].mean():.2f}")

    data = [run.with_outliers.loss_val.mean(axis=1), run.without_outliers.loss_val.mean(axis=1)]
    plt.hist(data, bins=30, histtype="step", density=True, label=labels)
    plt.xlabel("EMD mean validation loss")
    plt.ylabel("frequency")
    plt.legend(loc="upper right")
    plt.show_and_save("training without outliers - validation losses comparison "+run.name)
    KS_statistic, p_value = scipy.stats.ks_2samp(*data,mode='exact')
    interactive.save_value(
        "p-value of EMD loss training without outliers is not the same as training with outliers "+run.name,
        p_value,".3f"
    )
# %%
# compare AUCs of runs
data = [big_runs.with_outliers.auc, small_runs.with_outliers.auc]
labels = [big_runs.name+" batches",small_runs.name+" batches"]
plt.hist(data, bins=30, histtype="step", density=True, label=labels)
plt.xlabel("AUC")
plt.ylabel("frequency")
plt.legend(loc="upper left")
plt.show_and_save("training batches vs AUC comparison")

data = [big_runs.with_outliers.loss_val.mean(axis=1), small_runs.with_outliers.loss_val.mean(axis=1)]
labels = [big_runs.name+" batches",small_runs.name+" batches"]
plt.hist(data, bins=30, histtype="step", density=True, label=labels)
plt.xlabel("EMD mean validation loss")
plt.ylabel("frequency")
plt.legend(loc="upper left")
plt.show_and_save("training batches vs val loss comparison")
# %%
