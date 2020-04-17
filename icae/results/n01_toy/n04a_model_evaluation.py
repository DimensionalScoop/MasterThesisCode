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

import icae.results.n01_toy.n03_mode_training as training
from icae.tools.torch.gym import Gym
import icae.tools.loss.EMD as EMD
import icae.interactive_setup as interactive
from icae.models.waveform.simple import ConvAE
from icae.tools.analysis import calc_auc, plot_auc, TrainingStability

plt.set_plot_path(__file__)
interactive.set_saved_value_yaml_root_by_filename(__file__)

#%%
# Stability Test


def gym_factory():
    m = training.best_model_factory()
    m.verbose = False
    m.max_validation_steps = -1
    return m


def auc_classes(model):
    names = model.data_val.dataset.df["MC_name"].values
    truth = (names != "valid").astype("int")
    return truth


while True:
    for name, batches in zip(["1e5", "1e6"], [1000, 10000]):
        interactive.save_value(
            "waveforms used for training "+name, batches * gym_factory().data_train.batch_size
        )
        r = TrainingStability(
            "Ribbles_w_outliers_"+name, gym_factory, auc_classes, batches=batches
        )
        r.run(20)

# %%
