"""Combines a single waveform AE and a list-like event AE"""

import torch

from torch import nn as nn
from torch.nn import functional as F, init as init
from icae.tools.loss.EMD import torch as EMD

import numpy as np
import numpy.testing as npt


def batch_EMD(input, target):
    batch, events, rows = input.size()
    loss = torch.empty([batch, events, 1])
    for b in range(batch):
        for e in range(events):
            loss[b, e, 0] = EMD(
                input[b, e], target[b, e], norm=True
            )  # XXX: correct shape?
    return batch_EMD


class Garnet(nn.Module):
    """Combines a single waveform and an event model into one big model that can be
    diff'ed and optimized as a whole."""

    def __init__(
        self, waveform_AE, event_AE, latent_size, loss_wf=batch_EMD, waveform_lenght=128
    ):
        super().__init__()

        self.waveform_AE = waveform_AE
        self.event_AE = event_AE
        self.waveform_lenght = waveform_lenght
        self.loss_wf = loss_wf
        self.latent_size = latent_size

    def forward(self, x, split_wf=False):
        return self.decode(self.encode(x), split_wf)

    def encode(self, x):
        waveform = x[:, :, : self.waveform_lenght]
        scalars = x[:, :, self.waveform_lenght :]

        latent_wf = self.waveform_AE.encode(waveform)
        reconstruction = self.waveform_AE.decode(latent_wf)
        loss = self.loss_wf(reconstruction, waveform)

        npt.assert_approx_equal(latent_wf.size()[:-1], scalars.size()[:-1])
        npt.assert_approx_equal(loss.size(), scalars.size()[:-1] + [1])
        assert latent_wf.size()[-1] == self.latent_size
        full = torch.stack([latent_wf, loss, scalars])

        return self.event_AE.encode(full)

    def decode(self, x, split_wf=False):
        full = self.event_AE.decode(x)
        latent_wf = full[:, :, : self.latent_size]
        scalars = full[:, :, self.latent_size :]

        reconstruction = self.waveform_AE.decode(latent_wf)
        if split_wf:
            return reconstruction, scalars
        else:
            return torch.stack([reconstruction, scalars])


class Pearl(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        encoder_progression,
        decoder_progression,
        encoder_steps,
        decoder_steps,
    ):
        """Simple dense model for a whole event.
        
        Args:
            encoder_progression: callable that maps [0,1]→[0,1]. Is called for each
            encoder layer to determine the number of neurons in that layer with an
            output of `0` meaning `input_size` and an output of `1` meaning `latent_size`.
            decoder_progression: same as `encoder_progression` but for the decoder.
            An output of `0` means `latent_size` and `1` means `input_size`.
            encoder_steps: how many encoder layers to generate
            decoder_steps: how many decoder layers to generate
        """
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder_progression = encoder_progression
        self.decoder_progression = decoder_progression
        assert encoder_steps >= 2, "_layer_generator supports only steps of 2 or higher"
        assert decoder_steps >= 2, "_layer_generator supports only steps of 2 or higher"
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps

        self._setup()
        self._initialize_weights()

    def _setup(self):
        encoder = self._layer_generator(
            self.encoder_steps,
            self.encoder_progression,
            self.input_size,
            self.latent_size,
        )
        decoder = self._layer_generator(
            self.decoder_steps,
            self.decoder_progression,
            self.latent_size,
            self.input_size,
        )

        self.encoder = nn.ModuleList(list(encoder))
        self.decoder = nn.ModuleList(list(decoder))

    def _layer_generator(self, steps, progression, start_size, end_size):
        layers = np.linspace(0, 1, steps)
        x = progression(layers)
        npt.assert_almost_equal(x.min(), 0)
        npt.assert_almost_equal(x.max(), 1)
        y = np.rint(x * (end_size - start_size) + start_size)

        for i in range(len(y)):
            if i == len(y) - 1:
                break
            next_layer_size = y[i + 1]
            this_layer_size = y[i]
            yield nn.Linear(this_layer_size, next_layer_size)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encode(x)

    def decode(self, x):
        return self.decode(x)

    def _initialize_weights(self):
        pass
        # TODO: Weight initialization
