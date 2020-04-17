#%%
import icae.interactive_setup
import ray
import ray.tune as tune
import pandas as pd

path = "~/ray_results/"
ana = tune.Analysis(path + "lonpoint-512")

# %%
df: pd.DataFrame = ana.dataframe("validation_loss")

# %%
df.sort_values("validation_loss").head(20)

# %%
