#%%
import torch
from torch.nn import functional as F, init as init
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import ConfigSpace as CS
import ray
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
import icae.models.waveform.flexible as flexible
from icae.tools.torch.waveform_trainable import MCTrainable

from icae.models.waveform.simple import ConvAE
import icae.models.waveform.flexible as flexible
from icae.tools.hyperparam import mappings
from icae import interactive_setup as interactive

plt.set_plot_path(__file__)
interactive.set_saved_value_yaml_root_by_filename(__file__)

CPU_WORKERS_PER_TRAIL = 1
# Load data
filename = config.root + config.MC.filename
device = torch.device("cuda")

batches_per_step = 100
max_val_steps = 10
val_batch_size = 1024
max_iterations = 100
interactive.save_value("validation waveforms",max_val_steps*val_batch_size, ".1e")
interactive.save_value("max HPO iterations",max_iterations)
dl_config = {
    "batch_size": 320,
    "num_workers": CPU_WORKERS_PER_TRAIL,
    "shuffle": True,
    # "collate_fn": MCDataset.collate_fn,
    # "pin_memory": True,
}
waveforms_per_step = batches_per_step*dl_config["batch_size"]
interactive.save_value("batch size",dl_config["batch_size"])
#%%
config_space = CS.ConfigurationSpace()


class _:
    add = config_space.add_hyperparameter
    extend = config_space.add_hyperparameters
    int = CS.UniformIntegerHyperparameter
    float = CS.UniformFloatHyperparameter
    cat = CS.CategoricalHyperparameter

    cond = config_space.add_condition
    eq = CS.EqualsCondition
    neq = CS.NotEqualsCondition
    or_ = CS.OrConjunction

    #add(cat("get_train_pin", [get_train_pin]))
    #add(cat("get_val_pin", [get_val_pin]))
    add(cat("use_gpu", [True]))

    # add(int("batch_size", lower=2, upper=250))
    add(int("latent_size", lower=1, upper=6))

    model = cat(
        "model_factory",
        choices= mappings.models.keys(),
    )
    # lr = float("lr", lower=0.001, upper=1)
    # momentum = float("momentum", lower=0.1, upper=10)
    # extend([opt, lr, momentum])
    # cond(eq(lr, opt, "SGD"))
    # cond(eq(momentum, opt, "SGD"))

    eactiv = cat(
        "encoder_activation",
        choices=mappings.activations.keys(),
    )

    dactiv = cat(
        "decoder_activation",
        choices=mappings.activations.keys(),
    )
    add(model)
    add(eactiv)
    add(dactiv)
    cond(neq(eactiv, model, "conv"))
    cond(neq(dactiv, model, "conv"))

    # add(int("latent_size", lower=1, upper=6))
    # add(
    #     cat(
    #         "encoder_size_calc",
    #         choices=[flexible.default_decoder_size_calc, flexible.random_size],
    #     )
    # )
    # add(
    #     cat(
    #         "decoder_size_calc",
    #         choices=[flexible.default_encoder_size_calc, flexible.random_size],
    #     )
    # )
    # add(int("encoder_hidden_layers", lower=0, upper=4))
    # add(int("decoder_hidden_layers", lower=0, upper=4))

    # add(
    #     cat(
    #         "init_func",
    #         choices=[
    #             init.orthogonal_,
    #             init.zeros_,
    #             init.ones_,
    #             init.kaiming_normal_,
    #             init.kaiming_uniform_,
    #             init.xavier_normal_,
    #             init.xavier_uniform_,
    #         ],
    #     )
    # )
    add(cat("verbose", choices=[False]))
    add(cat("max_validation_steps", choices=[max_val_steps]))

    opt = cat("optimizer", choices=["Adam"])  # "SGD",
    add(opt)
    # lr = float("lr", lower=0.001, upper=1)
    # momentum = float("momentum", lower=0.1, upper=10)
    # extend([opt, lr, momentum])
    # cond(eq(lr, opt, "SGD"))
    # cond(eq(momentum, opt, "SGD"))

    add(
        cat(
            "training_loss",
            choices=mappings.losses.keys()
        )
    )

    add(cat("batches_per_step", [batches_per_step]))


metric = dict(metric="auc", mode="max")
search = TuneBOHB(config_space, max_concurrent=10, **metric)
scheduling = HyperBandForBOHB(time_attr="training_iteration", max_t=max_iterations, **metric)

#%%
if __name__ == "__main__":
    print("---Loading datasets")
    dataset_train = MCDataset(
        filename=filename, key="train/waveforms", transform=SingleWaveformPreprocessing()
    )
    dataset_val = MCDataset(
        filename=filename, key="val/waveforms", transform=SingleWaveformPreprocessing()
    )
    train = DataLoader(dataset_train, **dl_config)
    val = DataLoader(
        dataset_val, batch_size=val_batch_size, num_workers=CPU_WORKERS_PER_TRAIL, shuffle=True,
    )

    total_waveforms_available = len(train)*dl_config["batch_size"]
    interactive.save_value("total training waveforms simulated",total_waveforms_available,".1e")

    ray.init(num_cpus=12, num_gpus=1, memory=25000000000, object_store_memory=8000000000)
    # ray.init(local_mode=True)

    print("---Pinning objects...")
    train = pin_in_object_store(train)
    val = pin_in_object_store(val)


    def get_train_pin():
        return train


    def get_val_pin():
        return val


    print("---Done")
    
    print("---Initialization finished")

    class MCTrainable_with_data(MCTrainable):
        def _setup(self, config):
            MCTrainable._setup(self, config)
            self.get_val_pin = get_val_pin
            self.get_train_pin = get_train_pin
    
    
    analysis = tune.run(
        MCTrainable_with_data,
        scheduler=scheduling,
        search_alg=search,
        name="final-1-test",
        num_samples=10000,
        stop={"training_iteration": 100},
        max_failures=1,
        resources_per_trial={"cpu": 1, "gpu": 1 / 10},
        # checkpoint_dir='./ray',
        checkpoint_at_end=True,
        checkpoint_freq=10,
        # queue_trials=True,
    )

# %%
