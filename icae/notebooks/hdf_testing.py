# +
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import delayed, Parallel
from tqdm import tqdm
import types
import h5py as h5

from icae.tools.config_loader import config
# -

input_file = config.root + config.data.preprocessed_events
output_file = config.root + 'data/03-preprocessed/all_events_split.hdf'

input = pd.HDFStore(input_file)
output = pd.HDFStore(output_file)

input.select('frame','index == 0')

input.select_column('frame','frame')

input.select('frame','frame==10')

for i in tqdm(range(100)):
    j = i+100
    df = input.select('frame','(index >= i) & (index < j)')
    
    output.put(f'events/e{i}',df)

# %%timeit
i = np.random.randint(0,20,1)[0]
output[f"events/e{i}"]

# %%timeit
i = np.random.randint(0,10000,1)
input.select('frame','index == '+str(i))



hdf.create_table_index('frame')

f = h5.File(output_file, 'w')

groups = df_small.groupby('frame')

for i,df in groups:
    i = int(i)
    if i==10: break
    f[f"events/{i}"] = df

for i in f['events'].attrs:
    print(i)

hdf = pd.HDFStore(output_file)
for i, df in groups:
    hdf.put(f'events/t{int(i)}',df)
    break
    #f['events'].create_dataset(str(int(i)),data=df)

hdf.flush()
hdf.close()

sample = f['events/1001']

sample.value

hdf.keys()

hdf.flush()
hdf.close()

hdf = pd.HDFStore("/tmp/2.hdf")
for i, df in groups:
    i = int(i)
    hdf.put(f'events/t{i}',df)
    if i == 10: break

hdf['events/t3']

hdf.close()

hdf = pd.HDFStore("/tmp/2.hdf")

hdf_f = h5.File('/tmp/2.hdf',mode='r')

hdf_f['events']['t0'].keys()

hdf_f['events']['t0']['block0_values'].value


