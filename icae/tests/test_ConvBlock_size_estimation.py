import numpy as np
import sys
import torch
import pytest

from icae.models.event.imagenet_like import ConvBlock
from icae.tools.dataset_sparse import SparseEventDataset
from icae.tests.utils import get_example_dataset


@pytest.fixture
def sample_data():
    data = get_example_dataset()
    return data


def test_conv_forward_size_estimator(sample_data):
    channels = sample_data.size()[1]
    conv = ConvBlock(channels, 100, [3, 3])

    output_size = conv(sample_data).size()
    estm_output_size = conv.estimate_forward_size(sample_data.size())
    assert np.array_equal(output_size, estm_output_size)


def test_conv_forward_size_estimator_exceptions(sample_data):
    channels = sample_data.size()[1]

    with pytest.raises(ValueError):
        conv = ConvBlock(channels, 100, [0, 0])


def test_conv_backward_size_estimator(sample_data):
    channels = 25
    conv = ConvBlock(channels, 100, [3, 3])

    output_size = conv(sample_data).size()
    estm_input_size = conv.estimate_backward_size(output_size)
    assert np.allclose(estm_input_size, sample_data.size(), atol=1)


if __name__ == "__main__":
    test_conv_backward_size_estimator(sample_data)
