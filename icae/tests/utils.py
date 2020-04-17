import numpy as np
import torch
import os

from icae.tools.dataset_sparse import SparseEventDataset

# suppress keras warnings as they break pytest
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_example_dataset(count=1) -> torch.Tensor:
    """returns example event with shape [count,25,1501,151]"""
    dataset = SparseEventDataset(size=100 + count)
    if count == 1:
        for data in dataset:
            break  # just get the first batch
        data = data.view(1, *data.size())  # simulate batch
    else:
        data = []
        for i, t in enumerate(dataset):
            if i >= count:
                break
            data.append(t)
        data = torch.stack(data)
    return data

