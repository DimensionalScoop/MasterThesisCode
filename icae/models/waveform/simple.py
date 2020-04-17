import torch
from torch import nn as nn
from torch.nn import functional as F, init as init


class ConvAE(nn.Module):
    """The main autoencoder used in this thesis.
    Nicknamed Ribbles."""
    def __init__(self, config={}):
        super().__init__()
        self.latent_size = config.get("latent_size",3)
        
        self.conv1 = nn.Conv1d(1, 1, kernel_size=4, stride=2)
        # self.conv2 = nn.Conv1d(1, 1, kernel_size=6, stride=1)
        self.shrink = nn.Linear(63, self.latent_size)
        self.expand1 = nn.Linear(self.latent_size, 256)
        self.expand2 = nn.Linear(256, 128)

        self._initialize_weights()

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = x.view(-1, 1, 128)
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(-1, 63)
        x = F.relu(self.shrink(x))
        return x

    def decode(self, x):
        x = F.relu(self.expand1(x))
        x = torch.sigmoid(self.expand2(x))
        return x  # F.log_softmax(x, dim=1)

    def nan_alarm(self, x):
        if torch.isnan(x).sum():
            raise ArithmeticError()

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain("relu"))
