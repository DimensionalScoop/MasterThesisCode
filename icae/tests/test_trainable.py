import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from tqdm import tqdm

from icae.toy.trainable import LonePointTrainable
from icae.tools.hyperparam import trainable


def test_toy_training():
    config = {
        # there should be default values for every parameter
    }
    experiment = LonePointTrainable(config)
    for i in tqdm(range(10)):
        print("Validation loss:", experiment.train())
    experiment.save(checkpoint_dir="./ray")
    print("Done")
    
def test_MC_training():
    config = {
        # there should be default values for every parameter
        "verbose":True
    }
    
    trainable.MultipackTrainable.batches_per_step = 2
    trainable.MultipackTrainable.max_validation_steps = 5
    experiment = trainable.MultipackTrainable(config)
    print("Validation loss:", experiment.train())
    experiment.save(checkpoint_dir="./ray")
    print("Done")


if __name__ == "__main__":
    test_MC_training()

