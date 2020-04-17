"""AE that use a `Conv | latent space | Conv` scheme"""

import torch
from torch import nn as nn
from torch.nn import functional as F, init as init

import numpy as np
import numpy.testing as npt