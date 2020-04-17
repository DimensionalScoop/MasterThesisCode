# %%
"""Prove which storage method is the fastest"""
# %%
import torch
import pandas as pd
import h5py
import numpy as np
import perfplot
from icae.tools.config_loader import config
import dask.dataframe as dd

# %%
tmp = "/home/elayn/external/test_hdf/"
sample_size = 100000
table_size = [100, 100]

def example_table(index):
    columns = [f'c{i}' for i in range(table_size[0])]
    df = pd.DataFrame(np.random.rand(*table_size),columns=columns,index=[index]*table_size[0])
    return df

data_file = config.root + config.data.retabled_single
# %%
#f = pd.HDFStore(data_file,'r')
f_raw = h5py.File(data_file, 'w')

# %%
store = f.get_storer('frame')
# %%
type(store)

# %%
ddf = dd.read_hdf(data_file,'frame')

# %%
ddf['x'].max().compute()
# %%
# def maximum(f):
chunksize = 1000000
frame = 0
col = 'x'
maximum = 0

# %%
while True:
    try:
        df = f.select('frame',f"frame>={frame} & frame < {frame+chunksize}")
        maximum = max([maximum,df[col].max()])
        frame += chunksize
    except:
        break
print(maximum)

# %%
def hdf_with_keys_table():
    f = pd.HDFStore(tmp+"with_keys_table.hdf",complib='blosc:zstd')
    for i in range(sample_size):
        df = example_table(i)
        f.put(f"event/t{i}", df, format='table')
    f.flush()
    f.close()
    
def hdf_with_keys_fixed():
    f = pd.HDFStore(tmp+"with_keys_fixed.hdf",complib='blosc:zstd')
    for i in range(sample_size):
        df = example_table(i)
        f.put(f"event/t{i}", df, format='fixed',)
    f.flush()
    f.close()

def hdf_single():
    f = pd.HDFStore(tmp+"single.hdf",complib='blosc:zstd')
    for i in range(sample_size):
        df = example_table(i)
        if i==0:
            f.put('table',df,format='table')
        else:
            f.append('table',df)
    f.create_table_index('table')
    f.flush()
    f.close()
    

def hdf_files():
    for i in range(sample_size):
        df = example_table(i)
        df.to_hdf(tmp+f"singles/f{i}.hdf","single",complib='blosc:zstd')

# %%
#hdf_with_keys_table()
hdf_with_keys_fixed()
hdf_single()
#hdf_files()
# %%
def read_with_keys_table(indices_to_read):
    f = pd.HDFStore(tmp+"with_keys_table.hdf")
    for i in indices_to_read:
        df = f[f"event/t{i}"]

def read_with_keys_fixed(indices_to_read):
    f = pd.HDFStore(tmp+"with_keys_fixed.hdf")
    for i in indices_to_read:
        df = f[f"event/t{i}"]

def read_with_single(indices_to_read):
    f = pd.HDFStore(tmp+"single.hdf")
    for i in indices_to_read:
        df = f.select('table',"index == i")

def read_files(indices_to_read):
    for i in indices_to_read:
        df = pd.read_hdf(tmp+f"singles/f{i}.hdf")
        
def read_manual_retrival_with_keys(indices_to_read):
    f = h5py.File(tmp+"with_keys_fixed_copy.hdf",'r')
    for i in indices_to_read:
        path = f"event/t{i}"
        table = f[path+'/block0_values'][:]
        columns = f[path+'/axis0'][:]
        indices = f[path +'/axis1'][:]
        df = pd.DataFrame(table,indices,columns)
    f.close()

# %%
def random_samples(lenght):
    return np.random.randint(0,sample_size,lenght)

perfplot.show(setup=random_samples,
              kernels=[read_with_keys_fixed,read_with_single, read_manual_retrival_with_keys],#[read_with_keys_table,read_with_keys_fixed,read_with_single,read_files],
              n_range=[  10000,  20000, 80000],
              equality_check=None,
            )
# %%
read_with_single([0])

# %%
f = pd.HDFStore(tmp+"single.hdf",complib='blosc:zstd')
f.keys()

# %%
f.select('table',"index=='10'")

# %%
sto = f.get_storer('table')
# %%
df = example_table()
df

# %%
%%timeit
read_files(random_samples(100))

# %%
f = h5py.File(tmp + "single.hdf",'r')

# %%
import tables

# %%
t = tables.open_file(tmp+"single.hdf")

# %%
tab = t.root.table.table

# %%
[i for i in tab.__dict__ if callable(i)]

# %%
tab['c0']

# %%
