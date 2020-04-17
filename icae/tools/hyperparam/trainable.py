import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import ray
import ray.tune as tune
from ray.tune import Trainable
import os
import numpy as np
from tqdm import tqdm
from ray.tune.utils import pin_in_object_store, get_pinned_object

import icae.interactive_setup
import icae.models.event as models
from icae.tools.loss import sparse as loss_sparse
import icae.models.waveform.flexible as flexible
from icae.models.waveform.simple import ConvAE
import icae.tools.loss.EMD as EMD
from icae.tools.hyperparam import mappings

raise DeprecationWarning()

class MCTrainable(tune.Trainable):
    def _setup(self, config):
        # controls how often workers report model performance
        self.batches_per_step = config.get("batches_per_step", 1)
        self.max_validation_steps = config.get("max_validation_steps", 5)
        # removes channels from data to increase performance
        self.use_only_first_channel = config.get("use_only_first_channel", False)
        self.train_data = get_pinned_object(config.get("train_data"))
        self.val_data = get_pinned_object(config.get("val_data"))

        # GPU or CPU?
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        print("CUDA:", use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.batch_size = config.get("batch_size", 2)

        self.verbose = config.get("verbose", False)

        # Abstract: implement self.model

        optimizer = config.get("optimizer", "Adam")
        if optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.get("lr", 0.01),
                momentum=config.get("momentum", 0.9),
            )
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            raise NotImplemented()
        self.validation_loss_F = F.mse_loss
        loss_name = config.get("training_loss", "mse_loss")
        self.train_loss_F = mappings.losses[loss_name]

    def _train(self):
        training_loss = self._train_batches(self.batches_per_step)
        loss = self._validation_loss()
        return {
            "validation_loss": loss,
            "training_loss": np.mean(training_loss),
            "training_" + self.train_loss_F.__name__ + "_loss": np.mean(training_loss),
        }

    def _train_batches(self, train_for_n_batches=1):
        losses = []
        self.model.train()

        count = tqdm(
            range(train_for_n_batches), desc="Training", disable=not self.verbose
        )

        # train_data = get_pinned_object(self.train_data)
        for data, i in zip(self.train_data, count):
            if self.use_only_first_channel:
                data = data.narrow(1, 0, 1)
            if self.batch_size < data.size()[0]:
                data = data.narrow(0, 0, self.batch_size)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data)

            loss = self.train_loss_F(pred, data)  # target = data
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss))
        return losses

    def _validation_loss(self):
        self.model.eval()

        val_loss = []
        with torch.no_grad():
            count = tqdm(
                range(self.max_validation_steps),
                desc="Validation",
                disable=not self.verbose,
            )
            for data, i in zip(get_pinned_object(self.val_data), count):
                if self.use_only_first_channel:
                    data = data.narrow(1, 0, 1)
                data = data.to(self.device)
                pred = self.model(data)
                val_loss.append(float(self.validation_loss_F(pred, data)))

        return np.mean(val_loss)

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


class MultipackTrainable(MCTrainable):
    def _setup(self, config):
        self.model = models.Multipack(
            config.get("batch_normalization", False),
            config.get("kernel", [3, 3]),
            config.get("pp_kernel", [2, 2]),
            config.get("conv_layers_per_block", 2),
            config.get("conv_init", init.kaiming_normal_),
            config.get("channel_expansion", 2),
            config.get("stride", [1, 1]),
            config.get("channels"),
        )

        super()._setup(config)
