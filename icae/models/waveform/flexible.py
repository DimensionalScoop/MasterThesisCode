import torch
from torch import nn as nn
from torch.nn import functional as F, init as init
import numpy as np



def default_encoder_size_calc(layer_index, previous_size):
    return max(previous_size // 2, 1)


def random_size(layer_index, previous_size):
    rand = max(np.random.normal(1, 0.4), 0.1)
    return int(previous_size * rand)


def default_decoder_size_calc(layer_index, previous_size):
    return min(previous_size * 2, 2000)


class FlexibleAE(nn.Module):
    def __init__(self, config):
        # avoid circular dependency
        from icae.tools.hyperparam import mappings
        super().__init__()

        self.latent_size = config.get("latent_size", 3)

        # encoder
        self.encoding_layers = nn.ModuleList()

        size = 128  # input size
        out_calc = config.get("encoder_size_calc", default_encoder_size_calc)
        count_layers = config.get("encoder_hidden_layers", 1) + 1
        for i in range(count_layers):
            if count_layers - 1 == i:  # last layer
                new_size = self.latent_size
            else:
                new_size = out_calc(i, size)
            layer = nn.Linear(size, new_size)
            self.encoding_layers.append(layer)
            size = new_size
            name = config.get("encoder_activation", "ReLU")
            self.encoding_layers.append(mappings.activations[name]())

        # decoder
        self.decoding_layers = nn.ModuleList()

        size = self.latent_size
        out_calc = config.get("decoder_size_calc", default_decoder_size_calc)
        count_layers = config.get("decoder_hidden_layers", 1) + 1
        for i in range(count_layers):
            if count_layers - 1 == i:  # last layer
                new_size = 128  # output size
            else:
                new_size = out_calc(i, size)
            layer = nn.Linear(size, new_size)
            self.decoding_layers.append(layer)
            size = new_size
            name = config.get("decoder_activation", "ReLU")
            self.decoding_layers.append(mappings.activations[name]())

        self.name = config.get("name", None)
        self._initialize_weights(config)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = x.view(-1, 128)
        for layer in self.encoding_layers:
            x = layer(x)
        return x

    def decode(self, x):
        for layer in self.decoding_layers:
            x = layer(x)
        return x

    def _initialize_weights(self, config):
        init_func = config.get("init_func", None)
        if not init_func:
            return

        for layer in self.decoding_layers:
            init_func(layer.weight)
        for layer in self.encoding_layers:
            init_func(layer.weight)


def short_shallow(config={}):
    this_config = {
        "name": "short, shallow",
        "encoder_size_calc": lambda i, _: [20][i],
        "encoder_hidden_layers": 1,
        "decoder_size_calc": lambda i, _: [20][i],
        "decoder_hidden_layers": 1,
    }
    try: config = config.get_dictionary()
    except: pass
    config.update(this_config)
    return FlexibleAE(config)

def tall_shallow(config={}):
    this_config = {
        "name": "tall, shallow",
        "encoder_size_calc": lambda i, _: [256][i],
        "encoder_hidden_layers": 1,
        "decoder_size_calc": lambda i, _: [256][i],
        "decoder_hidden_layers": 1,
    }
    try: config = config.get_dictionary()
    except: pass
    config.update(this_config)
    return FlexibleAE(config)


def short_deep(config={}):
    this_config = {
        "name": "short, deep",
        "encoder_size_calc": lambda i, _: [80, 33][i],
        "encoder_hidden_layers": 2,
        "decoder_size_calc": lambda i, _: [33, 80][i],
        "decoder_hidden_layers": 2,
    }
    try: config = config.get_dictionary()
    except: pass
    config.update(this_config)
    return FlexibleAE(config)


def tall_deep(config={}):
    this_config = {
        "name": "tall, deep",
        "encoder_size_calc": lambda i, _: [200, 100][i],
        "encoder_hidden_layers": 2,
        "decoder_size_calc": lambda i, _: [50, 200][i],
        "decoder_hidden_layers": 2,
    }
    try: config = config.get_dictionary()
    except: pass
    config.update(this_config)
    return FlexibleAE(config)