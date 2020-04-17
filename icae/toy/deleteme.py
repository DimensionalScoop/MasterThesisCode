class test:
    class_member = 10

    def test(self):
        return self.class_member


t = test()


with 10 as a:
    print(a)
print(a)

#%%
import pandas as pd
import numpy as np

#%%
df = pd.DataFrame(index=[1,1,2,3,3,4,4,4],data=[10,20,10,20,30,30,40,50],columns=['d'])
df['d']-df.groupby(level=0)['d'].max()

# %%
grp = df.groupby(level=0)
keep = grp['d'].max() > 20
grp.filter(lambda g:g['d'].max()>20)

#%%
df = pd.DataFrame(np.random.rand(1000, 1000))
print(df.values.nbytes / 1e6)

# %%
print(df.astype("float16").values.nbytes / 1e6)

# %%
from icae.tools.dataset_sparse import SparseEventDataset
from icae.notebooks.setup_04 import load_train_val_test, val_loss

data_train, data_val, data_test = load_train_val_test(batch_size=6)

#ds = SparseEventDataset()

# %%
ds.value_columns = [
    "wf_AE_loss",
    "wf_integral",
]
ds.prune()

# %%
import gc

gc.collect()

#%%
ds.limit_event_t_size(850)
# %%
for i in ds:
    print(i.size())
    break

# %%
