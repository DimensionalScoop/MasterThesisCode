
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

batch_size = 320
r = TrainingStability("Conv_on_ICMC_3",None)
r.load()

training_times = np.array([len(i) for i in r.losses_train])
unique_training_times = np.unique(training_times)
unique_training_times_names = ["$10^{"+f"{int(np.log10(t*batch_size))}"+"}$" for t in unique_training_times]

def median_index(input):
    return np.argsort(input)[len(input)//2]

val_losses = []
for t in unique_training_times:
    filter = (training_times==t)
    
    model_index = median_index(r.loss_val[filter].mean(axis=1))
    losses = r.loss_val[filter][model_index]
    val_losses.append(losses)