import numpy as np
import torch
import matplotlib.pyplot as plt
import pytest

from icae.tests.utils import get_example_dataset
from icae.toy.lone_point_generator import LonePoint
from icae.toy.lone_point_models import SimpleClassifier


def classifier_example():
    net = SimpleClassifier([100, 100], 10 * 10, 3)
    print(net.net)


def test_classifier():
    net = SimpleClassifier([100, 100], 10 * 10, 3)
    generator = LonePoint(100, 2)

    for i, sample in enumerate(generator):
        if i > 10:
            break
        data, coords = sample['data'], sample['coords']
        net(data.view(1, 1, *data.size()))  # fake batch, channel dim


def plot_example():
    gen = LonePoint(10, 2)
    max_iter = 20
    edge = int(np.ceil(np.sqrt(max_iter)))

    plt.figure()
    for i, data in enumerate(gen):
        if i > max_iter:
            break
        plt.subplot(edge, edge, i + 1)
        data = data.numpy()
        plt.imshow(data)
    plt.show()


if __name__ == "__main__":
    plot_example()
    classifier_example()
