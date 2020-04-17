import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import ray
import ray.tune as tune
from ray.tune import Trainable
import os
import numpy as np
from tqdm import tqdm
from tqdm.gui import tqdm as gtqdm
from ray.tune.utils import pin_in_object_store, get_pinned_object

import icae.interactive_setup
import icae.models.event as models

# faster training
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class Gym:
    def __init__(
        self,
        model,
        device,
        loss_func,
        data_train,
        data_val,
        verbose=True,
        max_validation_steps=-1,
    ):
        self.model = model
        self.loss_func = loss_func
        self.device = device
        self.data_train = data_train
        self.data_val = data_val
        self.verbose = verbose
        if max_validation_steps == -1:
            self.max_validation_steps = len(data_val)
        else:
            self.max_validation_steps = max_validation_steps

        self.optimizer = optim.Adam(model.parameters())
        self.model.to(device)

    def train_batches(
        self, train_for_n_batches=1, callback=None, overwrite_loss_func=False
    ):
        losses = []
        self.model.train()

        if overwrite_loss_func:
            loss_func = overwrite_loss_func
        else:
            loss_func = self.loss_func

        count = tqdm(
            range(train_for_n_batches),
            desc="Training",
            disable=not self.verbose,
            leave=True,
        )

        batches_trained = 0
        while batches_trained < train_for_n_batches:
            for data, i in zip(self.data_train, count):
                batches_trained +=1
                data = data["data"].to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(data)
                loss = loss_func(pred, data)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss))
                if callback:
                    callback(i)
        return losses

    def validation_loss(self, overwrite_loss_func=False, return_val_data=False):
        self.model.eval()

        if overwrite_loss_func:
            loss_func = overwrite_loss_func
        else:
            loss_func = self.loss_func

        if self.max_validation_steps == -1:
            self.max_validation_steps = len(self.data_val)

        if return_val_data:
            return_val_data = []

        val_loss = []
        with torch.no_grad():
            count = tqdm(
                range(self.max_validation_steps),
                desc="Validation",
                disable=not self.verbose,
                leave=True,
            )
            for data, i in zip(self.data_val, count):
                data = data["data"].to(self.device)
                pred = self.model(data)
                val_loss.append(loss_func(pred, data).cpu().numpy())
                if return_val_data:
                    return_val_data.append(data.cpu().numpy())

        if return_val_data:
            return np.asarray(val_loss), np.asarry(return_val_data)
        return np.asarray(val_loss)

    def latent_space_of_val(self):
        self.model.eval()
        if self.max_validation_steps == -1:
            self.max_validation_steps = len(self.data_val)

        latent = []
        with torch.no_grad():
            count = tqdm(
                range(self.max_validation_steps),
                desc="Latent Space",
                disable=not self.verbose,
                leave=True,
            )
            for data, i in zip(self.data_val, count):
                data = data["data"].to(self.device)
                pred = self.model.encode(data)
                latent.append(pred.cpu().numpy())

        return np.asarray(latent)
