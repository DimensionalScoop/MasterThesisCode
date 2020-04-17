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
runs = TrainingStability("Conv_on_ICMC_correct_batch_size_3", None)
runs.load(False)

# %%
bad_runs = []
for r in runs.runs:
    if not np.alltrue(r.loss_val==r.loss_val):
        bad_runs.append(r)
    if np.sum(r.loss_val==np.inf)>0:
        bad_runs.append(r)
    if np.sum(r.loss_val==-np.inf)>0:
        bad_runs.append(r)
    if r.loss_val.mean()==np.inf:
        bad_runs.append(r)

# %%
print(f"nans or infs: {len(bad_runs)}")

# %%
