import ConfigSpace as CS
import ray.tune as tune
import torch.nn.functional as F
import torch.nn.init as init
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

from icae.toy.trainable import LonePointTrainable

config_space = CS.ConfigurationSpace()


class _:
    add = config_space.add_hyperparameter
    extend = config_space.add_hyperparameters
    int = CS.UniformIntegerHyperparameter
    float = CS.UniformFloatHyperparameter
    cat = CS.CategoricalHyperparameter

    cond = config_space.add_condition
    eq = CS.EqualsCondition

    add(int("batch_size", lower=1, upper=128))
    add(int("conv_stop_size", lower=9, upper=512))
    add(int("count_dense_layers", lower=1, upper=4))
    add(cat("channel_progression", choices=["double", "const", lambda x: x + 1]))
    add(cat("batch_normalisation", choices=[True, False]))
    add(cat("kernel_size", choices=[(2, 2), (3, 3), (4, 4), (5, 5)]))
    add(cat("pp_kernel_size", choices=[(1, 1), (2, 2), (3, 3), (4, 4)]))
    add(
        cat(
            "conv_init",
            choices=[
                init.orthogonal_,
                init.zeros_,
                init.ones_,
                init.kaiming_normal_,
                init.kaiming_uniform_,
            ],
        )
    )

    opt = cat("optimizer", choices=["SGD", "Adam"])
    lr = float("lr", lower=0.001, upper=1)
    momentum = float("momentum", lower=0.1, upper=10)
    extend([opt, lr, momentum])
    cond(eq(lr, opt, "SGD"))
    cond(eq(momentum, opt, "SGD"))

    add(cat("training_loss", choices=[F.mse_loss]))  # TODO: Added other losses


metric = dict(metric="validation_loss", mode="min")
scheduling = HyperBandForBOHB(
    time_attr="training_iteration", max_t=100, reduction_factor=4, **metric
)
search = TuneBOHB(config_space, **metric)

LonePointTrainable.lone_point_config['edge_length'] = 1024

# ray.init(local_mode=True) # use only one thread to make debugging easier
analysis = tune.run(
    LonePointTrainable,
    name="lonpoint-1024",
    scheduler=scheduling,
    search_alg=search,
    num_samples=1000,
    stop={"training_iteration": 100},
    # max_failures=0, # for debugging purposes
    resources_per_trial={"cpu": 1, "gpu": 0.2},
    # checkpoint_dir='./ray',
    checkpoint_at_end=True,
    checkpoint_freq=25,
)