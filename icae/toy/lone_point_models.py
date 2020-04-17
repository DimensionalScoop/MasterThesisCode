import torch
import torchvision as tv
import sys

from torch import nn as nn
from torch.nn import functional as F, init as init
import matplotlib.pyplot as plt

import numpy as np

from icae.models.event.imagenet_like import ConvBlock


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class CutChannels(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.narrow(x, 1, 0, 1)


class SimpleClassifier(nn.Module):
    @staticmethod
    def double_channel_size(in_channels):
        out_channels = in_channels * 2
        return out_channels

    def __init__(
        self,
        input_shape,
        conv_stop_size,
        count_dense_layers,
        channel_progression="double",
        batch_normalization=False,
        kernel=[3, 3],
        postprocessing_kernel=[2, 2],
        conv_layers_per_block=2,
        conv_init=init.orthogonal_,
        cut_channels=True,
        stride=1,
    ):
        super().__init__()

        assert len(input_shape) == 2

        if channel_progression == "double":
            channel_progression = SimpleClassifier.double_channel_size

        layers = []
        channels = 1
        if cut_channels:
            layers.append(CutChannels())
        pixels = np.prod(input_shape)
        while pixels > conv_stop_size:
            out_channels = channel_progression(channels)
            layers.append(
                ConvBlock(
                    channels,
                    out_channels,
                    kernel,
                    stride,
                    conv_layers_per_block,
                    conv_init,
                    pooling_size=2,
                    batch_norm=batch_normalization,
                )
            )

            input_shape = layers[-1].estimate_forward_size(
                [1, channels] + list(input_shape)
            )  # fake batch and channel dims
            input_shape = input_shape[2::]
            pixels = np.prod(input_shape)
            channels = out_channels

        layers.append(Flatten())
        in_size = pixels * channels
        dense_step_factor = np.power(in_size, 1 / count_dense_layers)
        for i in range(count_dense_layers - 1):
            out_size = int(np.ceil(in_size / dense_step_factor))
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size

        layers.append(nn.Linear(in_size, 2))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, *input):
        return self.net(*input)


class NoConvBlock(nn.Module):
    @staticmethod
    def double_channel_size(in_channels):
        out_channels = in_channels * 2
        return out_channels

    def __init__(
        self,
        input_shape,
        conv_stop_size,
        count_dense_layers,
        output_lenght=1,
        channel_progression="double",
        batch_normalization=False,
        kernel=[3, 3],
        postprocessing_kernel=[2, 2],
        conv_init=init.orthogonal_,
        cut_channels=False,
        stride=1,
    ):
        super().__init__()

        assert len(input_shape) == 2

        if channel_progression == "double":
            channel_progression = SimpleClassifier.double_channel_size
        elif channel_progression == "const":
            channel_progression = lambda x: 1
        elif callable(channel_progression):
            channel_progression = channel_progression
        else:
            raise NotImplementedError()

        layers = []
        channels = 1
        channels = 1
        if cut_channels:
            layers.append(CutChannels())
        pixels = np.prod(input_shape)
        while pixels > conv_stop_size:
            out_channels = channel_progression(channels)
            layers.append(nn.Conv2d(channels, out_channels, kernel, stride))

            x, y = input_shape[-2:]

            step = kernel[0] - 1
            x = int(np.floor(1 + (x - step - 1)))

            step = kernel[1] - 1
            y = int(np.floor(1 + (y - step - 1)))

            layers.append(nn.Sigmoid())
            if batch_normalization:
                layers.append(nn.BatchNorm2d(out_channels))

            pooling = 2
            layers.append(nn.MaxPool2d([pooling, pooling]))

            input_shape = x // 2, y // 2
            pixels = np.prod(input_shape)
            channels = out_channels

        layers.append(Flatten())
        in_size = pixels * channels
        dense_step_factor = np.power(in_size, 1 / count_dense_layers)
        for i in range(count_dense_layers - 1):
            out_size = int(np.ceil(in_size / dense_step_factor))
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size

        layers.append(nn.Linear(in_size, output_lenght))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, *input):
        return self.net(*input)

    def plot_kernels(self):
        for i in self.net:
            if type(i) == torch.nn.modules.conv.Conv2d:
                weights = i._parameters["weight"].detach().cpu().numpy()
                shape = weights.shape[-2:]
                plt.imshow(weights.reshape(shape))
                plt.show()
