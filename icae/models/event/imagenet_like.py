"""AE with a `Conv + Dense | latent space | Dense + Conv` structure"""

import torch

from torch import nn as nn
from torch.nn import functional as F, init as init


# TODO: Use nn.Sequential to improve readability
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride=[1, 1],
        count_layers=2,
        init_method=init.kaiming_normal_,
        pooling_size=2,
        batch_norm=False,
    ):
        if len(kernel) == 0 or kernel[0] <= 0 or kernel[1] <= 0:
            raise ValueError("Bad kernel values!")
        super().__init__()
        layers = []
        for i in range(count_layers):
            if i == 0:
                layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel, stride=stride)
                )
            else:
                layers.append(
                    nn.Conv2d(out_channels, out_channels, kernel, stride=stride)
                )
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
        self.convs = nn.ModuleList(layers)
        self.init_method = init_method
        self.batch_norm = batch_norm
        self.pooling_size = pooling_size

        self._initialize_weights()

    def estimate_forward_size(self, input_size):
        """Input should be [batch, channel, x, y].
        Returns [batch, channel', x', y'] after one forward pass"""
        input_size = np.array(input_size)
        for layer in self.convs:
            channel, x, y = input_size[1:]
            if type(layer) is nn.Conv2d:
                assert (
                    channel == layer.in_channels
                ), "incompatible number of channels in input"
                channel = layer.out_channels

                pad = 2 * layer.padding[0]
                step = layer.dilation[0] * (layer.kernel_size[0] - 1)
                x = np.floor(1 + (x + pad - step - 1) / layer.stride[0])

                pad = 2 * layer.padding[1]
                step = layer.dilation[1] * (layer.kernel_size[1] - 1)
                y = np.floor(1 + (y + pad - step - 1) / layer.stride[1])

                input_size[1:] = [channel, x, y]
        if self.pooling_size != 1:
            input_size[-2:] = np.floor(input_size[-2:] / self.pooling_size)

        return input_size

    def estimate_backward_size(self, output_size):
        """Input should be [batch, channel, x, y].
        Returns [batch, channel', x', y'] after one backward pass.
        XXX: rounding problems. x,y are only accurate within ±1."""
        output_size = np.array(output_size)

        if self.pooling_size != 1:
            output_size[-2:] = np.floor(output_size[-2:] * self.pooling_size)

        for layer in self.convs[::-1]:
            channel, x, y = output_size[1:]
            if type(layer) is nn.Conv2d:
                assert (
                    channel == layer.out_channels
                ), "incompatible number of channels in input"
                channel = layer.in_channels

                pad = 2 * layer.padding[0]
                step = layer.dilation[0] * (layer.kernel_size[0] - 1)
                x = (x - 1) * layer.stride[0] + 1 + step - pad

                pad = 2 * layer.padding[1]
                step = layer.dilation[1] * (layer.kernel_size[1] - 1)
                y = (y - 1) * layer.stride[1] + 1 + step - pad

                output_size[1:] = channel, x, y

        return output_size

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
            if type(layer) is not nn.BatchNorm2d:
                x = F.relu(x)  # relu after BN doesn't make much sense
        if self.pooling_size > 1:
            x = F.max_pool2d(x, self.pooling_size)
        return x

    def _initialize_weights(self):
        for layer in self.convs:
            if type(layer) is not nn.BatchNorm2d:
                try:
                    self.init_method(layer.weight, init.calculate_gain("relu"))
                except:
                    self.init_method(layer.weight)
