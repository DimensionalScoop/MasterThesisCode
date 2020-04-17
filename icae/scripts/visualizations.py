"""Some plots used in the interim presentation of my master thesis"""
# FIXME: where is `z_bins`? Maybe re-write file as this is only needed for presentations?

# +

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from icae.tools.config_loader import config

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# -

df = pd.read_hdf(config.root + config.data.retabled_single, start=0, stop=2000000)
print(len(np.unique(df.reset_index()["frame"])), "unique events")
print(df.values.nbytes * 1e-9, "GB of RAM used")

df.reset_index(inplace=True)
df.rename(columns={"starting_time": "t"}, inplace=True)

grp = df.groupby(by="frame")

t_min = np.min(grp.apply(lambda x: x["t"].min()))
maxs = grp.apply(lambda x: x["t"].max())

t_max = np.quantile(maxs, 0.992)
n_time_bins = (t_max - t_min) // 20
print("time bins needed:", n_time_bins)

bins_measured = np.mean(grp["x"].count()) * 128
bins_measured

print("% of bins with measurements", 100 * bins_measured / (n_time_bins * 10 * 10 * 60))

plt.figure(figsize=(10, 5))
event_extend = (grp["t"].max() - grp["t"].min()) // 20
plt.yscale("log")
plt.hist(event_extend, bins=100, density=True)
plt.ylabel("probability")
plt.xlabel("time bins needed")
plt.tight_layout()
plt.savefig("../plots/needed_time_bins.pdf")

print("bins_saved:", n_time_bins - event_extend.max())

for i, frame in grp:
    if 200 * 20 < frame.t.max() - frame.t.min() < 800 * 20:
        pe = frame.PE_count
        z = np.digitize(frame.z, z_bins)
        t = np.digitize(frame.t, t_bins)
        plt.scatter(t, z, c=pe, s=pe * 2000)
        plt.xlabel("time t (%d ns per bin)" % t_step)
        plt.ylabel("depth z (%d m per bin)" % z_step)
        plt.ylim(0, n_z_bins)
        plt.xlim(0, n_time_bins)
        plt.colorbar()
        plt.savefig("../plots/event_view_%d.pdf" % i)
        plt.show()
    if i > 100:
        break

df.head()

data_columns = ["t=%d" % i for i in range(128)]


def integrate_wf(data: pd.DataFrame):
    waveforms = data[data_columns]
    integral = np.sum(waveforms, axis=1)
    return integral


df["PE_count"] = integrate_wf(df)
df["PE_count"] = df["PE_count"] - df["PE_count"].min()
df["PE_count"] = df["PE_count"] / df["PE_count"].max()
grp = df.groupby("frame")

# +
t_step = (t_max - t_min) / n_time_bins
t_bins = np.arange(t_min, t_max, t_step)
assert np.allclose(n_time_bins, len(t_bins), atol=2)

n_z_bins = 60
z_min, z_max = df.z.min(), df.z.max()
z_step = (z_max - z_min) / n_z_bins
z_bins = np.arange(z_min, z_max, z_step)
assert np.allclose(n_z_bins, len(z_bins), atol=2)
# -

for i, frame in grp:
    pe = frame.PE_count
    z = np.digitize(frame.z, z_bins)
    t = np.digitize(frame.t, t_bins)
    plt.scatter(t, z, c=pe, s=pe * 2000)
    plt.xlabel("time t (%d ns per bin)" % t_step)
    plt.ylabel("depth z (%d m per bin)" % z_step)
    plt.ylim(0, n_z_bins)
    plt.xlim(0, n_time_bins)
    # plt.legend()
    # plt.show();
    break
    if i == 1000:
        break
plt.show()

# +
cum = np.zeros((len(t_bins) + 1, len(z_bins) + 1))
for i, frame in grp:
    pe = frame.PE_count
    z = np.digitize(frame.z, z_bins)
    t = np.digitize(frame.t, t_bins)
    for j in range(len(pe)):
        cum[t, z] += pe

    # if i==100: break
# -

plt.figure(figsize=(10, 5))
plt.imshow(cum.T, aspect="auto", cmap="inferno", norm=matplotlib.colors.LogNorm())
cbar = plt.colorbar(orientation="horizontal", label="PE counts (a.u.)")
plt.xlabel("time t (%d ns per bin)" % t_step)
plt.ylabel("depth z (%d m per bin)" % z_step)
plt.tight_layout()
plt.savefig("../plots/cum-event.pdf")

z

# +
plt.figure(figsize=(10, 5))
z = np.digitize(df.z, z_bins)

ts = []
for i, frame in grp:
    t = np.digitize(frame.t, t_bins)
    ts.append(t - t.min())
t = np.concatenate(ts)

_ = plt.hist2d(
    t,
    z,
    density=True,
    bins=(len(t_bins), len(z_bins)),
    norm=matplotlib.colors.LogNorm(),
)
cbar = plt.colorbar(orientation="horizontal", label="propability of non-zero bin")
plt.xlabel("time t (%d ns per bin)" % t_step)
plt.ylabel("depth z (%d m per bin)" % z_step)
# plt.savefig("../plots/cum-event-prop.pdf")
# -

len(t_bins)

ts = []
for i, frame in grp:
    t = np.digitize(frame.t, t_bins)
    ts.append(t - t.min())
t = np.concatenate(ts)
_ = plt.hist(t, density=True, bins=len(t_bins))  # ,norm=matplotlib.colors.LogNorm())
# cbar = plt.colorbar(orientation = 'horizontal',label="propability of non-zero bin")
plt.xlabel("time t (%d ns per bin)" % t_step)
plt.ylabel("propability of non-zero bin")
plt.yscale("log")

plt.figure(figsize=(10, 5))
z = np.digitize(df.z, z_bins)
t = np.digitize(df.t, t_bins)
_ = plt.hist(t, density=True, bins=len(t_bins))  # ,norm=matplotlib.colors.LogNorm())
# cbar = plt.colorbar(orientation = 'horizontal',label="propability of non-zero bin")
plt.xlabel("time t (%d ns per bin)" % t_step)
plt.ylabel("propability of non-zero bin")
plt.tight_layout()
plt.yscale("log")
# plt.savefig("../plots/cum-event-prop.pdf")

plt.figure(figsize=(10, 5))
z = np.digitize(df.z, z_bins)
t = np.digitize(df.t, t_bins)
_ = plt.hist(z, density=True, bins=len(z_bins))  # ,norm=matplotlib.colors.LogNorm())
# cbar = plt.colorbar(orientation = 'horizontal',label="propability of non-zero bin")
plt.xlabel("depth z (%d m per bin)" % z_step)
plt.ylabel("propability of non-zero bin")
plt.yscale("log")
plt.tight_layout()
plt.savefig("../plots/z-bin-hist.pdf")

z = np.digitize(df.z, z_bins)
t = np.digitize(df.t, t_bins)
bin_content, _ = np.histogram(z, bins=len(z_bins))
print("empty z bins", np.sum(bin_content == 0))

bin_content, _ = np.histogram(t, bins=len(t_bins))
print("empty t bins", np.sum(bin_content == 0))

bin_content

df.hist(column="PE_count", bins=100)
plt.yscale("log")

