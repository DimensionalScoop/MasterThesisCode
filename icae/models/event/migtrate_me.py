import torch
import sys

from torch import nn as nn
from torch.nn import functional as F, init as init

import numpy as np



# HACK: channels
channels = -1


# +

# +
class Neuner(nn.Module):
    """experimental network design."""

    def __init__(self):
        super(Neuner, self).__init__()
        target_channels = 55
        self.c1 = ConvBlock(
            channels, target_channels, [4, 2], count_layers=3, batch_norm=True
        )
        self.c2 = ConvBlock(
            target_channels, target_channels, [4, 2], count_layers=3, batch_norm=True
        )
        self.c3 = ConvBlock(
            target_channels, target_channels, [4, 2], count_layers=3, batch_norm=True
        )
        self.c4 = ConvBlock(
            target_channels, target_channels, [5, 2], count_layers=3, batch_norm=True
        )
        self.c5 = ConvBlock(
            target_channels, target_channels, [5, 1], count_layers=3, batch_norm=True
        )
        self.d6 = nn.Linear(5775, 2500)
        self.d7 = nn.Linear(2500, 900)
        self.d8 = nn.Linear(900, 100)

        self.d11 = nn.Linear(100, 300)
        self.d12 = nn.Linear(300, 1000)
        self.d13 = nn.Linear(1000, 5 * 100 * 13)
        self.c14 = ConvBlock(
            5, 10, [4, 2], count_layers=3, pooling_size=1, batch_norm=True
        )
        self.c15 = ConvBlock(
            10, 15, [4, 2], count_layers=3, pooling_size=1, batch_norm=True
        )
        self.c16 = ConvBlock(
            15, 20, [4, 2], count_layers=3, pooling_size=1, batch_norm=True
        )
        self.c17 = ConvBlock(
            20, 25, [4, 2], count_layers=3, pooling_size=1, batch_norm=True
        )
        self.c18 = ConvBlock(
            25, 25, [30, 10], count_layers=3, pooling_size=1, batch_norm=True
        )

        # self._initialize_weights()

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = self.c5(self.c4(self.c3(self.c2(self.c1(x))))).view(x.shape[0], -1)
        x = F.relu(self.d6(x))
        x = F.relu(self.d7(x))
        x = F.relu(self.d8(x))
        return x

    def decode(self, y):
        y = F.relu(self.d11(y))
        y = F.relu(self.d12(y))
        y = F.relu(self.d13(y))
        y = y.view(y.shape[0], 5, 100, 13)

        y = nn.functional.interpolate(self.c14(y), scale_factor=2)
        y = nn.functional.interpolate(self.c15(y), scale_factor=2)
        y = nn.functional.interpolate(self.c16(y), scale_factor=2)
        y = nn.functional.interpolate(self.c17(y), scale_factor=2)
        y = nn.functional.interpolate(self.c18(y), scale_factor=2)

        # convolutions can behave strangly at borders
        y = y[:, :, 492:-493, 15:-16]
        return y

    # def _initialize_weights(self):
    #    init.kaiming_uniform_(self.conv1.weight, init.calculate_gain("relu"))


# +
class StoutRope(nn.Module):
    """experimental network design."""

    def __init__(self):
        super(StoutRope, self).__init__()
        target_channels = 55
        self.c1 = ConvBlock(
            channels, target_channels, [4, 2], count_layers=2, batch_norm=True
        )
        self.c2 = ConvBlock(
            target_channels, target_channels, [4, 2], count_layers=2, batch_norm=True
        )
        self.c3 = ConvBlock(
            target_channels, target_channels, [4, 2], count_layers=2, batch_norm=True
        )
        self.c4 = ConvBlock(
            target_channels, target_channels, [5, 2], count_layers=2, batch_norm=True
        )
        self.c5 = ConvBlock(
            target_channels, target_channels, [5, 1], count_layers=2, batch_norm=True
        )
        self.d6 = nn.Linear(6435, 2500)
        self.d7 = nn.Linear(2500, 900)
        self.d8 = nn.Linear(900, 100)

        self.d11 = nn.Linear(100, 300)
        self.d12 = nn.Linear(300, 1000)
        self.d13 = nn.Linear(1000, 5 * 100 * 13)
        self.c14 = ConvBlock(
            5, 10, [4, 2], count_layers=2, pooling_size=1, batch_norm=True
        )
        self.c15 = ConvBlock(
            10, 15, [4, 2], count_layers=2, pooling_size=1, batch_norm=True
        )
        self.c16 = ConvBlock(
            15, 20, [4, 2], count_layers=2, pooling_size=1, batch_norm=True
        )
        self.c17 = ConvBlock(
            20, 25, [4, 2], count_layers=2, pooling_size=1, batch_norm=True
        )
        self.c18 = ConvBlock(
            25, 25, [30, 10], count_layers=2, pooling_size=1, batch_norm=True
        )

        # self._initialize_weights()

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = self.c5(self.c4(self.c3(self.c2(self.c1(x))))).view(x.shape[0], -1)
        x = F.relu(self.d6(x))
        x = F.relu(self.d7(x))
        x = F.relu(self.d8(x))
        return x

    def decode(self, y):
        y = F.relu(self.d11(y))
        y = F.relu(self.d12(y))
        y = F.relu(self.d13(y))
        y = y.view(y.shape[0], 5, 100, 13)

        y = nn.functional.interpolate(self.c14(y), scale_factor=2)
        y = nn.functional.interpolate(self.c15(y), scale_factor=2)
        y = nn.functional.interpolate(self.c16(y), scale_factor=2)
        y = nn.functional.interpolate(self.c17(y), scale_factor=2)
        y = nn.functional.interpolate(self.c18(y), scale_factor=2)

        # convolutions can behave strangly at borders
        shape_target = np.array([1501, 151])
        shape_diff = (np.array(y.shape)[[-2, -1]] - shape_target) / 2
        y = y[
            :,
            :,
            int(np.floor(shape_diff[0])) : -int(np.ceil(shape_diff[0])),
            int(np.floor(shape_diff[1])) : -int(np.ceil(shape_diff[1])),
        ]
        return y

    # def _initialize_weights(self):
    #    init.kaiming_uniform_(self.conv1.weight, init.calculate_gain("relu"))


# +
class RedString(nn.Module):
    """experimental network design."""

    def __init__(self):
        super(RedString, self).__init__()
        target_channels = 55
        bn = False
        bn2 = False
        self.c1 = ConvBlock(channels, 6, [4, 2], count_layers=1, batch_norm=bn)
        self.c2 = ConvBlock(6, 10, [4, 2], count_layers=1, batch_norm=bn)
        self.c3 = ConvBlock(10, 12, [4, 2], count_layers=1, batch_norm=bn)
        self.c4 = ConvBlock(12, 13, [5, 2], count_layers=1, batch_norm=bn)
        self.c5 = ConvBlock(13, 15, [5, 1], count_layers=1, batch_norm=bn)
        self.c5_b = ConvBlock(15, 15, [5, 1], count_layers=1, batch_norm=bn)
        self.d7 = nn.Linear(2090, 900)
        self.d8 = nn.Linear(900, 100)

        self.d11 = nn.Linear(100, 300)
        self.d12 = nn.Linear(300, 1000)
        self.d13 = nn.Linear(1000, 5 * 100 * 13)
        self.c14 = ConvBlock(
            5, 10, [4, 2], count_layers=1, pooling_size=1, batch_norm=bn2
        )
        self.c15 = ConvBlock(
            10, 15, [4, 2], count_layers=1, pooling_size=1, batch_norm=bn2
        )
        self.c16 = ConvBlock(
            15, 20, [4, 2], count_layers=1, pooling_size=1, batch_norm=bn2
        )
        self.c17 = ConvBlock(
            20, 25, [4, 2], count_layers=1, pooling_size=1, batch_norm=bn2
        )
        self.c18 = ConvBlock(
            25, 25, [30, 10], count_layers=1, pooling_size=1, batch_norm=False
        )

        # self._initialize_weights()

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = self.c5_b(self.c5(self.c4(self.c3(self.c2(self.c1(x)))))).view(
            x.shape[0], -1
        )
        x = F.relu(self.d7(x))
        x = F.relu(self.d8(x))
        return x

    def decode(self, y):
        y = F.relu(self.d11(y))
        y = F.relu(self.d12(y))
        y = F.relu(self.d13(y))
        y = y.view(y.shape[0], 5, 100, 13)

        y = nn.functional.interpolate(self.c14(y), scale_factor=2)
        y = nn.functional.interpolate(self.c15(y), scale_factor=2)
        y = nn.functional.interpolate(self.c16(y), scale_factor=2)
        y = nn.functional.interpolate(self.c17(y), scale_factor=2)
        y = nn.functional.interpolate(self.c18(y), scale_factor=2)

        # convolutions can behave strangly at borders
        shape_target = np.array([1501, 151])
        shape_diff = (np.array(y.shape)[[-2, -1]] - shape_target) / 2
        y = y[
            :,
            :,
            int(np.floor(shape_diff[0])) : -int(np.ceil(shape_diff[0])),
            int(np.floor(shape_diff[1])) : -int(np.ceil(shape_diff[1])),
        ]
        return y

    # def _initialize_weights(self):
    #    init.kaiming_uniform_(self.conv1.weight, init.calculate_gain("relu"))


# +
class SmileShoot(nn.Module):
    """very big, multidimensional latent layer."""

    def __init__(self):
        super(SmileShoot, self).__init__()
        bn = False
        bn2 = False
        layers_per_block = 2
        self.c1 = ConvBlock(
            channels,
            channels * 2,
            [10, 5],
            count_layers=layers_per_block,
            batch_norm=bn,
        )
        self.c2 = ConvBlock(
            channels * 2,
            channels * 4,
            [10, 5],
            count_layers=layers_per_block,
            batch_norm=bn,
        )
        self.c3 = ConvBlock(
            channels * 4,
            channels * 8,
            [10, 5],
            count_layers=layers_per_block,
            batch_norm=bn,
        )

        self.c16 = ConvBlock(
            channels * 8,
            channels * 4,
            [10, 5],
            count_layers=layers_per_block,
            pooling_size=1,
            batch_norm=bn2,
        )
        self.c17 = ConvBlock(
            channels * 4,
            channels * 2,
            [10, 5],
            count_layers=layers_per_block,
            pooling_size=1,
            batch_norm=bn2,
        )
        self.c18 = ConvBlock(
            channels * 2,
            channels,
            [10, 5],
            count_layers=layers_per_block,
            pooling_size=1,
            batch_norm=bn2,
        )
        self.postprocessor = ConvBlock(
            channels,
            channels,
            [10, 5],
            count_layers=layers_per_block,
            pooling_size=1,
            batch_norm=False,
        )

        # self._initialize_weights()

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = self.c2(self.c1(x))
        return x

    def decode(self, y, print_debug=False):
        y = nn.functional.interpolate(y, scale_factor=2)
        y = nn.functional.interpolate(self.c17(y), scale_factor=2)
        y = nn.functional.interpolate(self.c18(y), scale_factor=2)
        y = self.postprocessor(y)

        # convolutions can behave strangly at borders
        shape_target = np.array([1501, 151])
        shape_now = y.shape
        shape_diff = (np.array(shape_now)[[-2, -1]] - shape_target) / 2
        y = y[
            :,
            :,
            int(np.floor(shape_diff[0])) : -int(np.ceil(shape_diff[0])),
            int(np.floor(shape_diff[1])) : -int(np.ceil(shape_diff[1])),
        ]
        if print_debug:
            print(shape_now, "→", shape_target)
        return y

    # def _initialize_weights(self):
    #    init.kaiming_uniform_(self.conv1.weight, init.calculate_gain("relu"))


# +
class SmileShootNoCut(nn.Module):
    """very big, multidimensional latent layer."""

    def __init__(self):
        super(SmileShootNoCut, self).__init__()
        bn = True
        bn2 = True
        layers_per_block = 2
        self.c1 = ConvBlock(
            channels,
            channels * 2,
            [10, 5],
            count_layers=layers_per_block,
            batch_norm=bn,
        )
        self.c2 = ConvBlock(
            channels * 2,
            channels * 4,
            [10, 5],
            count_layers=layers_per_block,
            batch_norm=bn,
        )

        self.c17 = ConvBlock(
            channels * 4,
            channels * 2,
            [10, 5],
            count_layers=layers_per_block,
            pooling_size=1,
            batch_norm=bn2,
        )
        self.c18 = ConvBlock(
            channels * 2,
            channels,
            [10, 5],
            count_layers=layers_per_block,
            pooling_size=1,
            batch_norm=bn2,
        )
        self.postprocessor = ConvBlock(
            channels,
            channels,
            [10, 5],
            count_layers=layers_per_block,
            pooling_size=1,
            batch_norm=False,
        )

        # self._initialize_weights()

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = self.c2(self.c1(x))
        return x

    def decode(self, y, print_debug=False):
        y = nn.functional.interpolate(y, scale_factor=2)
        y = nn.functional.interpolate(self.c17(y), scale_factor=2)

        # y = self.postprocessor(y)
        shape_target = [1501 + 18, 151 + 8]
        if print_debug:
            print(y.shape, "→", shape_target)

        y = nn.functional.interpolate(self.c18(y), size=shape_target)
        y = self.postprocessor(y)

        return y

    # def _initialize_weights(self):
    #    init.kaiming_uniform_(self.conv1.weight, init.calculate_gain("relu"))


class Multipack(nn.Module):
    """should do almost nothing."""

    def __init__(
        self,
        batch_normalization=False,
        kernel=[3, 3],
        postprocessing_kernel=[2, 2],
        conv_layers_per_block=2,
        conv_init=init.orthogonal_,
        channel_expansion=2,
        stride=1,
        channels=channels,
    ):
        super().__init__()

        self.ec1 = ConvBlock(
            channels,
            channels * channel_expansion,
            kernel,
            stride,
            conv_layers_per_block,
            conv_init,
            2,
            batch_normalization,
        )

        self.ec2 = ConvBlock(
            channels * channel_expansion,
            channels * channel_expansion ** 2,
            kernel,
            stride,
            conv_layers_per_block,
            conv_init,
            2,
            batch_normalization,
        )

        self.dc1 = ConvBlock(
            channels * channel_expansion ** 2,
            channels * channel_expansion,
            kernel,
            stride,
            conv_layers_per_block,
            conv_init,
            pooling_size=1,
            batch_norm=batch_normalization,
        )
        self.dc2 = ConvBlock(
            channels * channel_expansion,
            channels,
            kernel,
            stride,
            conv_layers_per_block,
            conv_init,
            pooling_size=1,
            batch_norm=batch_normalization,
        )

        if postprocessing_kernel is not None:
            self.postprocessor = ConvBlock(
                channels,
                channels,
                postprocessing_kernel,
                count_layers=3,
                pooling_size=1,
                init_method=conv_init,
                batch_norm=False,
            )

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = self.ec2(self.ec1(x))
        return x

    def decode(self, y, print_debug=False):
        y = nn.functional.interpolate(y, scale_factor=2)
        y = nn.functional.interpolate(self.dc1(y), scale_factor=2)
        y = self.dc2(y)
        y = self.postprocessor(y)

        shape_target = [1501, 151]
        y = nn.functional.interpolate(y, size=shape_target)
        y = torch.sigmoid(y)

        return y


class FastMultipack(Multipack):
    """Multipack that only uses one channel"""

    def __init__(
        self,
        batch_normalization=False,
        kernel=[3, 3],
        postprocessing_kernel=[2, 2],
        conv_layers_per_block=2,
        conv_init=init.orthogonal_,
        channel_expansion=2,
        stride=1,
        channels=channels,
    ):
        super().__init__(
            batch_normalization,
            kernel,
            postprocessing_kernel,
            conv_layers_per_block,
            conv_init,
            channel_expansion,
            stride,
            channels=1,
        )


if __name__ == "__main__":
    x = example.view(1, *shape)
    x.shape

    model = Multipack()
    model.encode(x).shape, model.decode(model.encode(x), True).shape

    # +
    target_channels = 55
    c1 = ConvBlock(channels, target_channels, [4, 2], count_layers=3)
    c2 = ConvBlock(target_channels, target_channels, [4, 2], count_layers=3)
    c3 = ConvBlock(target_channels, target_channels, [4, 2], count_layers=3)
    c4 = ConvBlock(target_channels, target_channels, [5, 2], count_layers=3)
    c5 = ConvBlock(target_channels, target_channels, [5, 1], count_layers=3)
    x = c5(c4(c3(c2(c1(x))))).view(x.shape[0], -1)

    x.shape
    # -

    d6 = nn.Linear(5775, 2500)
    d7 = nn.Linear(2500, 900)
    d8 = nn.Linear(900, 100)
    x = d8(d7(d6(x)))
    x.shape

    num_params = 2
    y = x[:, :-num_params]
    params = x[:, -num_params:]
    y.shape

    d11 = nn.Linear(98, 300)
    d12 = nn.Linear(300, 1000)
    d13 = nn.Linear(1000, 5 * 100 * 13)
    y = d13(d12(d11(y))).view(y.shape[0], 5, 100, 13)
    y.shape

    y.shape

    c14 = ConvBlock(5, 10, [4, 2], count_layers=3, pooling_size=1)
    c15 = ConvBlock(10, 15, [4, 2], count_layers=3, pooling_size=1)
    c16 = ConvBlock(15, 20, [4, 2], count_layers=3, pooling_size=1)
    c17 = ConvBlock(20, 25, [4, 2], count_layers=3, pooling_size=1)
    c18 = ConvBlock(25, 25, [30, 10], count_layers=3, pooling_size=1)

    c14(y).shape

    y = nn.functional.interpolate(c14(y), scale_factor=2)
    y = nn.functional.interpolate(c15(y), scale_factor=2)
    y = nn.functional.interpolate(c16(y), scale_factor=2)
    y = nn.functional.interpolate(c17(y), scale_factor=2)
    y = nn.functional.interpolate(c18(y), scale_factor=2)
    y.shape

    y[:, :, 492:-493, 15:-16].shape

    # +
    # How do I make this differentiable?
    # shift_x = torch.floor(params[0])
    # shift_y = torch.floor(params[1])
    # y = y.roll([shift_x,shift_y],[-1,-2])
    # -

    del dataset, example
