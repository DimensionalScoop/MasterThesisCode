import numpy as np
import torch

from icae.models.event_resize_decoder import Postman
from icae.tools.dataset_sparse import SparseEventDataset
from icae.tests.utils import get_example_dataset


def test_postman():
    data: torch.Tensor = get_example_dataset()
    model = Postman()
    model(data)
