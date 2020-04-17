#%%
import ray
import ray.tune as tune
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ana = tune.Analysis('~/ray_results/MPT_long_run')

# %%
val_loss = 'validation_loss'
ana.get_best_config(val_loss, 'min')

# %%
df:pd.DataFrame = ana.dataframe().dropna(axis=0,subset=[val_loss])
interesting = [val_loss] + [i for i in df.columns if 'config' in i]
df.sort_values(val_loss).head(20)[interesting]


# %%
# %%
config_att = ["config/conv_layers_per_block", "config/channel_expansion"]
for i in config_att:
    possible_values = np.unique(df[i])
    plt.hist2d(df[val_loss],df[i], bins=[100,len(possible_values)],)
    plt.colorbar()
    plt.ylabel(i)
    plt.xlabel('validation loss')
    plt.show()

# %%
