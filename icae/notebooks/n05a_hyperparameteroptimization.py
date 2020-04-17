import sys

import ConfigSpace as CS
import ray
import torch.nn.functional as F
import torch.nn.init as init
from joblib import Memory
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.utils import pin_in_object_store
from ray import tune

from icae.tools.config_loader import config
from icae.tools.dataset_sparse import SparseEventDataset
from icae.tools.hyperparam import trainable


CPU_WORKERS_PER_TRAIL = config.machine.cpus_per_trail
cache_dir = config.data.pre_learning_cache
memory = Memory(cache_dir + "joblib", verbose=0)

# takes some time. only execute this when the respective code changes
def cache_data():
    limit = 850
    dtype = "float16"

    print("Preparing and caching data (this may take a while)…")
    ds = SparseEventDataset()
    ds.value_columns = ["wf_AE_loss", "wf_integral"]
    ds.prune(dtype)
    ds.limit_event_t_size(limit)
    ds.save(cache_dir, dtype)


# cache_data()
ds = SparseEventDataset.load(cache_dir)

data_train, data_val, data_test = ds.get_train_val_test(CPU_WORKERS_PER_TRAIL)
ray.init(num_cpus=12, num_gpus=1, memory=29000000000)
#ray.init(local_mode=True)
data_train = pin_in_object_store(data_train)
data_val = pin_in_object_store(data_val)

config_space = CS.ConfigurationSpace()

class _:
    add = config_space.add_hyperparameter
    extend = config_space.add_hyperparameters
    int = CS.UniformIntegerHyperparameter
    float = CS.UniformFloatHyperparameter
    cat = CS.CategoricalHyperparameter

    cond = config_space.add_condition
    eq = CS.EqualsCondition

    add(cat("train_data", [data_train]))
    add(cat("val_data", [data_val]))

    add(int("batch_size", lower=1, upper=6))
    # add(int("conv_stop_size", lower=9, upper=512))
    # add(int("count_dense_layers", lower=1, upper=4))
    add(cat("channel_expansion", choices=[1, 2, 3]))
    add(cat("batch_normalization", choices=[True, False]))
    add(cat("kernel", choices=[(2, 2), (3, 3), (4, 4), (5, 5)]))
    add(cat("pp_kernel", choices=[(1, 1), (2, 2), (3, 3), (4, 4)]))
    add(int("stride", lower=0, upper=1))
    add(int("conv_layers_per_block", lower=1, upper=6))
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
    add(cat("verbose", choices=[False]))
    add(cat("channels", choices=[10]))

    opt = cat("optimizer", choices=["SGD", "Adam"])
    lr = float("lr", lower=0.001, upper=1)
    momentum = float("momentum", lower=0.1, upper=10)
    extend([opt, lr, momentum])
    cond(eq(lr, opt, "SGD"))
    cond(eq(momentum, opt, "SGD"))

    add(
        cat(
            "training_loss",
            choices=[
                F.mse_loss,
                # losses.CustomKL(None),
                # losses.CustomKL("abs"),
                # losses.CustomKL(2),
                # losses.OpticalLoss(),
                # losses.BinHeights(),
                # losses.AC(),
                # losses.ACLower(),
                # losses.Motication(),
                # losses.GaussACDC(),
            ],
        )
    )

    add(cat("max_validation_steps", [100]))
    add(cat("batches_per_step", [10]))


metric = dict(metric="validation_loss", mode="min")
scheduling = HyperBandForBOHB(
    time_attr="training_iteration", max_t=500, reduction_factor=3, **metric
)
search = TuneBOHB(config_space, **metric)

if __name__ == "__main__":
    #sys.exit(10)
    # ray.init(local_mode=True)  # use only one thread to make debugging easier
    analysis = tune.run(
        trainable.MultipackTrainable,
        name="MPT_long_run",
        scheduler=scheduling,
        search_alg=search,
        num_samples=1000,
        stop={"training_iteration": 500},
        max_failures=0,  # for debugging purposes
        resources_per_trial={"cpu": CPU_WORKERS_PER_TRAIL, "gpu": 1},
        # checkpoint_dir='./ray',
        checkpoint_at_end=True,
        checkpoint_freq=10,
        queue_trials=True,
    )
