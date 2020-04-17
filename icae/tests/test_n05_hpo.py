import pytest
from tqdm import tqdm
import ray


def test_hpo_network():
    import icae.notebooks.n05a_hyperparameteroptimization as hpo

    config = hpo.config_space.sample_configuration()
    trainer = hpo.trainable.MultipackTrainable(config)
    for i in tqdm(range(10)):
        print("Loss: ", trainer.train())


if __name__ == "__main__":
    test_hpo_network()
