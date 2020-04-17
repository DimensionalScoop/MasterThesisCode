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

#%%
filename = config.root + config.MC.filename
device = torch.device("cuda")
dl_config = {
    "batch_size": 32 * 10,
    "num_workers": 3,
    "shuffle": True,
    "pin_memory": True,
}

dataset_train = MCDataset(
    filename=filename, key="train/waveforms", transform=SingleWaveformPreprocessing(),
)
dataset_val = MCDataset(
    filename=filename, key="val/waveforms", transform=SingleWaveformPreprocessing()
)
train = DataLoader(dataset_train, **dl_config)
val = DataLoader(dataset_val, batch_size=1024, num_workers=3)

val_classes = dataset_val.df.MC_type

#%%
# [np.logspace(4,8,10),np.logspace(8,9,3)[1:]]
todo = np.concatenate([np.logspace(4,8,10), np.logspace(8,10,5)[1:]])/dl_config['batch_size'] #/dl_config['batch_size']
training_batches = [todo[-4]]#[7:-3] #np.logspace(9.3,11,4)/dl_config['batch_size']
latent_dims = [2,3,4,6]
samples_per_time = 1 #5
time_per_batch = 16/(1e8/dl_config['batch_size'])
print("estimated runtime in hrs:",samples_per_time*len(latent_dims)*np.sum(training_batches)*time_per_batch/60/60)

#%%
while(True):
    for count_batches in tqdm(training_batches,"Progress",leave=True):
        for dim in latent_dims:
            model_config = {"latent_size":dim}
            gym_factory = lambda: Gym(
                training.best_model_class(model_config),
                device,
                EMD.torch_auto,
                train,
                val,
                verbose=False,
            )
            r = analysis.TrainingStability(f"corrected training time vs outliers", gym_factory, batches=int(count_batches), auc_classes=val_classes,add_attributes=model_config)
            r.run(samples_per_time)
    print("-----------Done, restarting task...")
