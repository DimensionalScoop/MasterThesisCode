import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import ray.tune as tune
from ray.tune import Trainable
import os

from icae.toy.lone_point_generator import LonePoint, LonePointValidation
from icae.toy.lone_point_models import NoConvBlock, SimpleClassifier


class LonePointTrainable(tune.Trainable):
    lone_point_config = dict(edge_length=512, dims=2)

    # performance configuration
    val_size = 500
    samples_per_step = 10  # controls how often workers report model performance

    def _setup(self, config):
        # GPU or CPU?
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.batch_size = config.get("batch_size", 2)

        dataset = LonePoint(**self.lone_point_config)
        val_dataset = LonePointValidation(**self.lone_point_config, size=self.val_size)
        self.train_loader = dataset.get_dataloader(self.batch_size)
        self.val_loader = val_dataset.get_dataloader(self.batch_size)

        self.model = NoConvBlock(
            dataset.shape[1::],
            config.get("conv_stop_size", 50),
            config.get("count_dense_layers", 2),
            2,
            config.get("channel_progression", "double"),
            config.get("barch_normalisation", True),
            config.get("kernel_size", [3, 3]),
            config.get("pp_kernel_size", [2, 2]),
            config.get("conv_init", init.orthogonal_),
        )

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
        self.train_loss_F = config.get("training_loss", F.mse_loss)

        self.batches_per_step = self.samples_per_step / self.batch_size

    def _validation_loss(self):
        self.model.eval()

        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                data, target = batch["data"], batch["coords"]
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                val_loss += self.validation_loss_F(output, target)

        val_loss /= len(self.val_loader.dataset)
        return float(val_loss)

    def _train_batches(self, train_for_n_batches=1):
        self.model.train()
        for batch, i in zip(self.train_loader, range(train_for_n_batches)):
            data, target = batch["data"], batch["coords"]
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data)

            loss = self.train_loss_F(pred, target)
            loss.backward()
            self.optimizer.step()

    def _train(self):
        self._train_batches(10)  # FIXME: self.batches_per_step)
        loss = self._validation_loss()
        return {"validation_loss": loss}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


