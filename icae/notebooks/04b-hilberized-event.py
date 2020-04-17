#%%
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
from icae.tools.dataset_list import ListDataset

ds = ListDataset(size=100)
#%%
ds.move_to_t0()
ds.prune('float32')
data_train, data_val, data_test = ds.get_train_val_test(12)

# %%
