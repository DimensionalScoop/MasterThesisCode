# -*- coding: utf-8 -*-
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

from icae.models.waveform.simple import TorchSimpleConv
from icae.processors.xy_degeneracy_solver import argsort_by_pc
from icae.tools.config_loader import config
from icae.tools.dataset import SingleWaveformPreprocessing
from icae.tools.loss.EMD import torch as loss_EMD
# -

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# event: the smallest unit this algorithm can work on
# mini-batch: a small collection of multiple events a single processor can work on
# batch: a collection of about 12 mini-batches that can be processed by all cores of a processor at the same time


input_file = config.root + config.data.retabled_single
output_file = config.root + config.data.preprocessed_events
table = pd.HDFStore(input_file, mode="r").get_storer(config.data.hdf_key)
print(
    "Opened",
    os.path.basename(input_file),
    "(size=",
    table.nrows // 1e6,
    "M rows), writing to",
    os.path.basename(output_file),
)


def rename(df):
    df.rename_axis(index={"starting_time": "t"}, inplace=True)

    return df


def trim_last_event(df):
    """`load` doesn't know where which event is. cut of the last event as it is probably incomplete."""

    previous_lenght = len(df)
    level = 0  # level='frame'
    # df is always sorted by 'frame'
    last_frame = df.last_valid_index()[level]
    df.drop(index=last_frame, level=level, inplace=True)
    assert len(df) > 0, "cut entire dataframe — are you loading big enough chunks?"
    assert len(df) <= previous_lenght
    return df


def split(df, n_parts):
    if len(df) == 0:
        return []

    # df is always sorted by 'frame'
    level = 0  # level='frame'
    first_frame = df.first_valid_index()[level]
    last_frame = df.last_valid_index()[level]
    frames = last_frame - first_frame
    if frames < n_parts:
        split = [df.loc[first_frame + i] for i in range(frames)]
    else:
        step = frames // n_parts + 1  # add one to account for round-off errors
        split = [
            df.loc[first_frame + i * step : first_frame + i * step + step]
            for i in range(frames // step + 1)
        ]

    return [i for i in split if i is not []]


def save(dfs):
    for df in dfs:
        df.reset_index(drop=True).astype("float32").to_hdf(
            output_file,
            mode="a",
            append=True,
            key=config.data.hdf_key,
            format="table",
            complevel=1,
            complib="blosc:snappy",
        )


def load(start, stop):
    if start > table.nrows:
        return None
    if stop >= table.nrows:
        stop = table.nrows

    df = pd.read_hdf(input_file, mode="r", start=start, stop=stop)
    return df


def load_and_preprocess(start):
    stop = start + batch_size
    df = load(start, stop)

    # only trim off last event when doing an incomplete read
    # otherwise the last event has been read completely
    if stop < table.nrows:
        assert len(df) == batch_size
        df = trim_last_event(df)

    df = rename(df)
    minibatches = split(df, cpu_cores*3) # don't let workers starve

    return minibatches, len(df)


# +
def find_extrema(labels):
    """use a very big batch size and just find min/max values"""
    batch_size = 2000000
    rows = table.nrows
    row_pointer = 0

    wf_columns = ["t=%d" % i for i in range(128)]
    integral_name = "integral"  # special column that is the integral

    mins = {}
    maxs = {}
    for i in labels:
        maxs[i] = -1000000
        mins[i] = 100000

    for row_pointer in tqdm(range(0, rows, batch_size)):
        df = load(row_pointer, row_pointer + batch_size).reset_index()
        if df is None:
            break  # consumed all data available

        for i in labels:
            if i is integral_name:
                df[integral_name] = df[wf_columns].values.sum(axis=1)

            maxs[i] = max(maxs[i], df[i].max())
            mins[i] = min(mins[i], df[i].min())

    return mins, maxs


def scale_to_1(data, minimum, maximum):
    bigger_than_0 = data - minimum
    smaller_than_1 = bigger_than_0 / (maximum - minimum)
    return smaller_than_1


def scale_to_1_by_name(df, names, mins, maxs):
    df[name] = scale_to_1(df[name], mins[name], maxs(name))
    return df


# -

minima, maxima = find_extrema(["x", "y", "z", "starting_time", "integral"])

print("loading single waveform model")
device = torch.device("cuda")
nn_preproc = SingleWaveformPreprocessing()
model = TorchSimpleConv()
model_name = "superlong-dict.pt"
model.load_state_dict(torch.load(config.root + "icae/models/trained/" + model_name))

# +
# binning preferences

# add a column to flag wavforms within an event
# that cannot be distiguished after binning
track_collisions = True
# many fileds are discarded after preprocessing to save memory
# setting this to true keeps them
keep_DOM_info = True

# increase z_bin count to account for Deep Core (7m spacing instead of 17m)
count_z_bins = 60 + 90  # there are 60 DOMs per string
count_t_bins = 1500  # see scripts/visualizations.py for a analysis. this should be enough to give each event an unique time bin
z_bin_edges = np.linspace(minima["z"], maxima["z"], count_z_bins)
t_bin_edges = np.linspace(
    minima["starting_time"], maxima["starting_time"], count_t_bins + 1
)
print("z bin size", z_bin_edges[1] - z_bin_edges[0], "m")
print("t bin size", t_bin_edges[1] - t_bin_edges[0], "ns")
# -

# save binning as config so dataset_sparce can read it
with open(config.root + config.data.binning_config, "w") as f:
    yaml.safe_dump(
        {
            "count_z_bins": count_z_bins,
            "count_t_bins": count_t_bins,
            "track_collisions": track_collisions,
            "z bin size in m:": float(z_bin_edges[1] - z_bin_edges[0]),
            "t bin size in ns:": float(t_bin_edges[1] - t_bin_edges[0]),
            "z_bin_edges_min": float(z_bin_edges.min()),
            "z_bin_edges_max": float(z_bin_edges.max()),
            "t_bin_edges_min": float(t_bin_edges.min()),
            "t_bin_edges_max": float(t_bin_edges.max()),
        },
        f,
    )

# +
wf_columns = ["t=%d" % i for i in range(128)]


def process_minibatch(df: pd.DataFrame):
    # do everything that can be done independend of events

    # loss and latent parameters from single waveform AE fit
    data = df.loc[:, wf_columns].values
    data = nn_preproc(data)
    pred = model.forward(data).detach()
    latent = model.encode(data).detach().numpy()
    loss_per_vector = loss_EMD(pred, data, norm=True).numpy()

    for i in range(latent.shape[1]):
        df["wf_latent_%d" % i] = latent[:, i]
    df["wf_AE_loss"] = loss_per_vector

    # integrated PE charge
    df["wf_integral"] = df.loc[:, wf_columns].values.sum(axis=1)

    df = df.reset_index(level=["frame", "t"])

    # some events contain waveform duplicates (two waveforms for the same x,y,z,t)
    # this is usually the case because there were so many PE that the waveform
    # clipped at smaller gains. The IceCube MC then includes two waveforms,
    # one with lower and one with higher gain.
    # We only need the unclipped waveform (lower gain), which can be identified
    # by taking the waveform with the highest PE count
    df.sort_values("wf_integral", inplace=True, ascending=False)
    df.drop_duplicates(subset=["frame", "x", "y", "z", "t"], keep="first", inplace=True)

    # binning
    df["z_bin"] = np.digitize(df["z"], z_bin_edges)
    df["t_bin"] = np.digitize(df["t"], t_bin_edges)

    # do per-event stuff
    grp = df.groupby("frame")
    df = grp.apply(process_event)

    # do a final ordering. use real t,z to guarantee correct order when bins are degenerated
    df.sort_values(["frame", "t", "z", "xy_pc_sorting"], inplace=True)

    if not keep_DOM_info:
        df.drop(columns=["x", "y", "z", "t"], inplace=True)

        df.reset_index(drop=True, inplace=True)
        df.drop(columns=wf_columns, inplace=True)

    # df.set_index('frame',inplace=True)

    return df


# -


def reduce_sorting(rows):
    assert rows.min() >= 0
    _, reduction = np.unique(rows.values, return_inverse=True)
    # XXX: doesn't work for negative values (unique is behaving strangely)
    # but this shouldn't matter as rows.min()>=0
    return reduction


def process_event(df):
    # x,y are going to be dropped. Ensure a constant ordering by sorting along
    # the principal component of x,y
    points = df[["x", "y"]].values
    df["xy_pc_sorting"] = argsort_by_pc(points)

    # event should now be uniquely identifiable by t,z and the xy_pc_sorting
    if track_collisions:
        df["bin_collision"] = df.duplicated(
            ["t_bin", "z_bin", "xy_pc_sorting"], keep=False
        )
    # assert (len(df) == len(np.unique(df[["t_bin","z_bin","xy_pc_sorting"]],axis=0)))

    # xy_pc_sorting contains ascending numbers from 0 to len(df)
    group = df.groupby(["z_bin", "t_bin"])["xy_pc_sorting"]
    df["xy_pc_sorting_index"] = group.transform(reduce_sorting)
    # xy_pc_sorting_index now starts counting from 0 for every new, unique z_bin, t_bin

    # event should now be uniquely identifiable by t,z and the xy_pc_sorting_index, too
    if track_collisions:
        df["bin_collision_index"] = df.duplicated(
            ["t_bin", "z_bin", "xy_pc_sorting_index"], keep=False
        )
    return df


def save_globals_to_yaml():
    d = globals().copy()
    ignore = [i for i in d.keys() if i.startswith("_")]
    ignore.append("In")
    ignore.append("Out")

    save = {}

    for i in d.keys():
        if i in ignore:
            continue
        if isinstance(d[i], types.ModuleType):
            continue
        if callable(d[i]):
            continue
        if sys.getsizeof(d[i]) // 8 > 50:
            continue

        try:
            yaml.safe_dump(d[i])
            save[i] = d[i]
        except:
            save[i] = repr(d[i])

    yaml.safe_dump(save, open(output_file + ".config", "w"))


# +
debug = False
total_collisions = 0

cpu_cores = config.machine.cpu_cores

# batch_size must be bigger than event_size*cpu_cores
batch_size = 2000 * 25 if debug else 1024 * 500
rows_processed = 0
n_jobs = 1 if debug else cpu_cores

print("starting preprocessing…")
save_globals_to_yaml()
# -

try:
    os.remove(output_file)
except FileNotFoundError:
    pass

while rows_processed != table.nrows:
    minibatches, rows_loaded = load_and_preprocess(rows_processed)
    rows_processed += rows_loaded
    if minibatches is None:
        break
    assert rows_processed <= table.nrows

    tasks = []
    for mbatch in minibatches:
        tasks.append(delayed(process_minibatch)(mbatch))
    result = Parallel(n_jobs=n_jobs,verbose=0)(tasks)

    save(result)

    print("Processed", int(rows_processed / table.nrows * 100), "% of all events")
    if track_collisions:
        total_collisions += np.sum([i.bin_collision.sum() for i in result])
        print("Collisions:", total_collisions)

    # if debug: break

result[0].head()

# +
df = pd.concat(result)

df.hist(column="xy_pc_sorting", bins=len(np.unique(df.xy_pc_sorting)))
plt.yscale("log")
plt.show()
df.hist(column="xy_pc_sorting_index", bins=len(np.unique(df.xy_pc_sorting_index)))
plt.yscale("log")
plt.show()
# -

bin_config = Box.from_yaml(filename=config.root + config.data.binning_config)
bin_config.update({"max_xy_degeneracy": 5})
bin_config.to_yaml(filename=config.root + config.data.binning_config)

sys.exit(0)

# +
# r.sort_values(["frame", "z_bin", "t_bin", "xy_pc_sorting"])
# -

for r in result:
    assert r.shape == r.drop_duplicates().shape
    break
r.sort_values(["frame", "z_bin", "t_bin", "xy_pc_sorting"])

comb = pd.concat(result)

comb.columns

comb.hist(column="xy_pc_sorting", bins=len(np.unique(comb.xy_pc_sorting)))
plt.yscale("log")

result[0]

for i in result:
    print(type(i))

result[0][result[0].bin_collision].head()

np.digitize(result[0].z, z_bin_edges)

dupl = []
for df in result:
    for i, grp in df.groupby("frame"):
        if len(grp) != len(np.unique(grp[["t_bin", "z_bin", "xy_pc_sorting"]], axis=0)):
            print("duplicate")
            dupl.append(grp)

np.sum(dupl[0].z_bin.values != np.digitize(dupl[0].z, z_bin_edges))

du = dupl[0][dupl[0].duplicated(["t_bin", "z_bin", "xy_pc_sorting"], keep=False)]

du

du.iloc[0] - du.iloc[1]

# %config IPCompleter.greedy = True

for df in dupl:
    see = df[["t_bin", "z_bin", "xy_pc_sorting"]]
    uni, count = np.unique(see, axis=0, return_counts=True)
    print(uni[count > 1])


_ = plt.hist(dupl[0].z_bin, bins=300)
# _=plt.hist(dupl[0].z,bins=300)

_ = plt.hist(dupl[0].z, bins=600)

df = load(0, 5000)
zs = df.reset_index().starting_time

dist = np.ones((len(zs), len(zs))) * 100
for i, x in enumerate(tqdm(zs)):
    for j, y in enumerate(zs):
        dist[i, j] = np.abs(x - y)


small_dists = dist.flatten().compress(dist.flatten() < 10)

_ = plt.hist(small_dists, bins=300)

dupl[0][dupl[0]["z_bin"] == 2]

len(grp)

uni[count > 1]

grp.loc[[31, 32]]

z_bin_edges[:-1] - z_bin_edges[1:]

np.digitize(grp.loc[31].z, bins=z_bin_edges), np.digitize(
    grp.loc[32].z, bins=z_bin_edges
)

grp.loc[31] - grp.loc[32]

plt.plot(grp.loc[31][wf_columns].values)
plt.plot(grp.loc[32][wf_columns])
plt.show()

for i, grp in df.groupby("frame"):
    if len(grp) != len(np.unique(grp[["t_bin", "z_bin", "xy_pc_sorting"]], axis=0)):
        print("duplicate")
        break
grp[["t_bin", "z_bin", "xy_pc_sorting"]].sort_values(
    ["t_bin", "z_bin", "xy_pc_sorting"]
)

df[["frame", "t_bin", "z_bin", "xy_pc_sorting"]].sort_values(
    ["frame", "t_bin", "z_bin", "xy_pc_sorting"]
)

for event in result:
    if len(event) == len(np.unique(event[["t_bin", "z_bin", "xy_pc_sorting"]], axis=0)):
        print("duplicate")
        break

event

result[4]

# +
event = result[0].loc[0]

# event = event.sort_values('xy_pc_sorting',ascending=False)
event  # [["x","y","z_bin","t_bin","xy_pc_sorting"]]
# -

for i in range(len(event)):
    p = event.iloc[i]
    print(p.x, p.y, p.z, p.t)
    plt.plot(p[wf_columns])
    # plt.show()
    if i == 1:
        break

len(np.unique(event[["t_bin", "z_bin", "xy_pc_sorting"]], axis=0))

len(event)

uni = np.unique(event[["t_bin", "z_bin"]], axis=0)
uni

len(uni)

uni[0] == uni[1]

# event should be uniquely identifiable by t,z and the xy_pc_sorting
assert len(event) == len(np.unique(event[["t_bin", "z_bin", "xy_pc_sorting"]], axis=1))

for i, grp in event.groupby("xy_pc_sorting"):
    # plt.plot(grp.t,grp.z)
    # plt.show();
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    plt.scatter(grp.t, range(len(grp.z)))  # ,c=event.t)
    plt.show()

# event = event.sort_values('xy_pc_sorting')
plt.plot(event.x, event.y)
# for x,y,s in zip(event.x,event.y,event.xy_pc_sorting):
#    plt.text(x,y,str(s))
# plt.scatter(event.x,event.y,c=event.xy_pc_sorting,cmap='coolwarm')
# plt.colorbar()
plt.show()

# %matplotlib inline

# %matplotlib widget

# +

fig = plt.figure()
ax = plt.axes(projection="3d")

uniques = np.unique(event.xy_pc_sorting)
pc = np.digitize(event.xy_pc_sorting, uniques)

ax.scatter3D(event.x, event.y, pc)  # ,c=event.t)
# event = event.sort_values('t')
# ax.plot3D(event.x,event.y,event.z)
# -

len(uniques)

# +
fig = plt.figure()
ax = plt.axes(projection="3d")

# uniques = np.unique(event.xy_pc_sorting)
pc = event.xy_pc_sorting  # np.digitize(event.xy_pc_sorting, uniques)

ax.scatter3D(event.x, event.y, pc)  # ,c=event.t)
# event = event.sort_values('t')
# ax.plot3D(event.x,event.y,event.z)
# -

len(np.unique(problems, axis=0))

problems


df = load(0, 1000000)
df.head()

z_bin_edges.shape

df["z_bin"] = np.digitize(df["z"], z_bin_edges)

_ = plt.hist(df.z, bins=60)

zs = df.z
zs = zs - zs.min()
zs = zs / zs.max() * 60

# plt.vlines(np.unique(zs),0,4000)
_ = plt.hist(df.z_bin, bins=60)
plt.show()

# %matplotlib widget

len(np.unique(df.z_bin))

np.digitize()

torch.cuda.is_available()

a = np.array([1, 2])

a.sum()

# +
batch_size = 10000000
rows = table.nrows
row_pointer = 0

wf_columns = ["t=%d" % i for i in range(128)]

find_minmax = ["x", "y", "z", "starting_time"]
mins = {}
maxs = {}
for i in find_minmax:
    maxs[i] = -1000000
    mins[i] = 100000

max_int = -100
min_int = 100

for row_pointer in tqdm(range(0, rows, batch_size)):
    df = load(row_pointer, row_pointer + batch_size).reset_index()
    if df is None:
        break

    for i in find_minmax:
        maxs[i] = max(maxs[i], df[i].max())
        mins[i] = min(mins[i], df[i].min())

    integral = df[wf_columns].values.sum(axis=1)
    max_int = max(max_int, integral.max())
    min_int = min(min_int, integral.min())


# +
data = df.loc[:, wf_columns].values
data = preproc(data)
pred = model.forward(data).detach()
loss_per_vector = EMD.torch(pred, data, norm=True).numpy()
latent = model.encode(data).detach().numpy()

for i in range(latent.shape[1]):
    df["wf_latent_%d" % i] = latent[:, i]
df["wf_AE_loss"] = loss_per_vector

df["wf_integral"] = df.loc[:, wf_columns].values.sum(axis=1)
# -

df.head()

_ = plt.hist(loss_per_vector, bins=100)
plt.show()

pred.shape

params.shape


df.head()

df.head()

a = params.numpy()

a

# %matplotlib widget

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(a[:, 0], a[:, 1], a[:, 2], s=2, alpha=0.01)

print("----------\nPreprocessing finished.")
