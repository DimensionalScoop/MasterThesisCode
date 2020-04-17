#%%
import numpy as np
import pandas as pd
import numpy.testing as npt
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from box import Box

from icae.tools.config_loader import config
import icae.toy.waveformMC as toy
import icae.interactive_setup as interactive

interactive.set_saved_value_yaml_root_by_filename(__file__)
out_file = config.root + config.MC.filename 

#%%
distances_to_DOM = np.linspace(5, 50, num=30)  # m
angles_to_DOM = np.deg2rad(np.linspace(0, 180, num=30))
photons_per_event = stats.uniform(10, 3000)

interactive.save_value("min angle", 0)
interactive.save_value("max angle", 180)
interactive.save_value("min distance", min(distances_to_DOM))
interactive.save_value("max distance", max(distances_to_DOM))
interactive.save_value("min photons", 10)
interactive.save_value("max photons", 3000)
interactive.save_value(
    "unique parameter combinations",
    len(distances_to_DOM) * len(angles_to_DOM) * (3000 - 10),
)
# %%

# generate two datasets:
#  - Allen: big dataset with few outliers (1%) as training data
#  - Betty: small dataset with 50% outliers as validation data
#  - Conrad: dataset with only double peak to measure differentiation

