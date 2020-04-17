#%%
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from icae.tests.utils import get_example_dataset
from icae.tools.dataset_sparse import SparseEventDataset
from icae.tools.plot_data import plot_pooled, cut_event_to_size

mpl.use('agg')


def test_event_cutting():
    for i in range(2, 10):
        data = torch.rand([10] * i)
        assert len(cut_event_to_size(data, 2).size()) == 2

    data = torch.rand((11, 11))
    assert torch.allclose(cut_event_to_size(data.clone(), 2), data)

    with pytest.raises(ValueError) as e_info:
        data = torch.rand([10])
        cut_event_to_size(data, 2)


def test_pooled():
    channel = 1
    data = get_example_dataset(10)
    for img_data in data:
        img = img_data[channel]
        sum_ = img.sum()
        plot_pooled(img, 10)
        assert img.sum() == sum_


def manual_plot_fake_data_test():
    shape = [1, 100, 30]
    count_examples = 5
    data = torch.rand([count_examples] + shape)

    for i in range(count_examples):
        plot_pooled(data[i])
    plt.show()


def manual_plot_MC_test():
    data = get_example_dataset(10)
    for i, event in enumerate(data):
        print(event.size())
        plot_pooled(event)


if __name__ == "__main__":
    print("Running manual plotting tests…")
    manual_plot_fake_data_test()
    manual_plot_MC_test()
    print("Done.")

#%%
