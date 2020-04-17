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

# %%
# Load data
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
from icae.models.waveform.simple import ConvAE
import icae.models.waveform.flexible as flexible

model_config = {
    "device": device,
    "data_train": train,
    "data_val": val,
    "verbose": True,
    "max_validation_steps": -1,
}
models = [
    Gym(ConvAE(), **model_config, loss_func=EMD.torch_auto),
    Gym(ConvAE(), **model_config, loss_func=F.mse_loss),
    Gym(ConvAE(), **model_config, loss_func=loss_sparse.CustomKL(allow_broadcast=True, do_not_enforce=True)),
    Gym(ConvAE(), **model_config, loss_func=loss_sparse.CustomKL(allow_broadcast=True, enforcement_strength=2)),
    Gym(flexible.short_shallow(), **model_config, loss_func=EMD.torch_auto),
    Gym(flexible.short_deep(), **model_config, loss_func=EMD.torch_auto),
    Gym(flexible.tall_deep(), **model_config, loss_func=EMD.torch_auto),
]

best_model_class = ConvAE
best_model_factory = lambda: Gym(ConvAE(), **model_config, loss_func=EMD.torch_auto)

steps = len(train)
#%%
if __name__ == "__main__":
    raise DeprecationWarning("This file isn't used in the analysis directly anymore. It just remains for the definitions it declares on import")
    plt.set_plot_path(__file__)


    for i,model in enumerate(models):
        model.verbose = False
        try:
            lossname=model.loss_func.__name__
        except AttributeError:
            lossname=model.loss_func.__class__.__name__
        try:
            modelname = model.model.name
        except:
            modelname = model.model.__class__.__name__
        name = f"{i}, Model:{modelname}, Loss: {lossname}"
        print(f"Training {name}:")
        
        loss = model.train_batches(steps)

        x = np.linspace(0, (steps + 1) * dl_config["batch_size"], len(loss))
        plt.plot(x, loss)
        plt.xlabel("# of waveforms used for training")
        plt.ylabel(f"loss {lossname}")
        plt.xscale("log")
        plt.figtext(0, 0, name)
        plt.show_and_save(f"{name} + training")

        loss_func = lambda p, t: EMD.torch_auto(p, t, mean=False)
        val_losses = np.hstack(model.validation_loss(loss_func))  # restack batches
        names = dataset_val.df["MC_name"].values

        _, bins, _ = plt.hist(val_losses, int(np.sqrt(len(val_losses))), label="everything")
        plt.clf()

        unames = np.unique(names)
        data = [val_losses[names == name] for name in unames]
        plt.hist(data, bins=bins, label=unames, stacked=True)
        plt.xlabel("EMD loss")
        plt.ylabel("frequency")
        plt.legend()
        plt.show_and_save(f"{name} + loss_hist")

        _, bins, _ = plt.hist(
            val_losses, int(np.sqrt(len(val_losses))), label="everything", density=True
        )
        plt.hist(data, bins=bins, label=unames, histtype="step", density=True)
        plt.xlabel("EMD loss")
        plt.ylabel("frequency")
        plt.legend()
        plt.show_and_save(f"{name} + loss_hist_compare")

        try:
            truth = (names != "valid").astype("int")
            pred = val_losses
            fpr, tpr, _ = metrics.roc_curve(truth, pred)
            auc = metrics.auc(fpr, tpr)

            lw = 2
            plt.plot(
                fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.show_and_save(f"{name} + ROC")
        except ValueError as e:
            print(e)

# %%


# %%
