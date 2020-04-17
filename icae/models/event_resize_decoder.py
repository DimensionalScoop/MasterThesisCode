import torch
import sys

from torch import nn as nn
from torch.nn import functional as F, init as init

import numpy as np
import numpy.testing as npt

from icae.tools.dataset_sparse import SparseEventDataset
from icae.models.event.imagenet_like import ConvBlock

# load small dataset to get number of channels, dimensions, etc.
dataset = SparseEventDataset(size=10)

example = dataset.__getitem__(0).to_dense()
data_shape = example.shape
channels = data_shape[0]
aspect_ratio = data_shape[1] / data_shape[2]
aspect_ratio


class Postman(nn.Module):
    """scales down event without learnable parameters and
    tries to reconstruct from there.
    Every block uses as many channels as the input has."""

    def __init__(
        self,
        latent_size_factor=2,
        batch_normalization=False,
        kernel=[3, 3],
        postprocessing_kernel=[2, 2],
        conv_layers_per_block=2,
        conv_init=init.orthogonal_,
        channel_expansion=2,
        stride=1,
    ):
        super().__init__()

        self.latent_size_factor = latent_size_factor

        # self.dc1 = ConvBlock(
        #     channels,
        #     channels,
        #     kernel,
        #     stride,
        #     conv_layers_per_block,
        #     conv_init,
        #     pooling_size=1,
        #     batch_norm=batch_normalization,
        # )
        self.dc2 = ConvBlock(
            channels,
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
        else:
            self.postprocessor = None

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = nn.functional.interpolate(x, scale_factor=1 / self.latent_size_factor)
        return x

    def decode(self, y, print_debug=False):
        y = nn.functional.interpolate(
            y, scale_factor=2
        )  # BUG: magic number, should be latent_size_factor!!
        # y = nn.functional.interpolate(self.dc1(y), scale_factor=2)
        y = self.dc2(y)

        assert torch.isfinite(y).all() == 1

        # force correct shape
        shape_target = [-1, 25, 1501, 151]
        if self.postprocessor is not None:
            shape_target = self.postprocessor.estimate_backward_size(shape_target)
            # rounding errors
            # shape_target[-2:] += 1
        y = nn.functional.interpolate(y, size=tuple(shape_target[-2:]))

        if self.postprocessor is not None:
            y = self.postprocessor(y)
            npt.assert_array_equal(y.size()[1:], data_shape)

        y = y.sigmoid()
        assert y.min() >= 0
        assert y.max() <= 1
        return y
