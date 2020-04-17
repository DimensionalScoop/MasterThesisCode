#%%
from ray import tune
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from icae.tools.config_loader import config
import icae.interactive_setup as interactive
import icae.results.n01_toy.n02_hpo as hpo
from icae.models.waveform.simple import ConvAE
from icae.tools.analysis import calc_auc, plot_auc, TrainingStability
import icae.tools.loss.EMD as EMD
from icae.tools.torch.gym import Gym
from icae.tools.dataset.MC import MCDataset
from icae.tools.hyperparam import mappings
from icae.tools.dataset.single import SingleWaveformPreprocessing

plt.set_plot_path(__file__)
interactive.set_saved_value_yaml_root_by_filename(__file__)


#%%
ana = tune.Analysis("~/ray_results/final-1/")
# %%
dfs = ana.trial_dataframes
interactive.save_value("number of configurations", len(dfs), ".1e")
#%%
plot_training_overview = False
if plot_training_overview:
    ax = None  # This plots everything on the same plot
    for d in tqdm(dfs.values()):
        d.auc.plot(ax=ax, legend=False)
    plt.show_and_save("training overview")

#%%
aucs = []
times = []
for d in tqdm(dfs.values()):
    aucs.extend(d.auc)
    times.extend(range(len(d.auc)))
times = np.asarray(times)  # +1)*hpo.waveforms_per_step
plt.hist(times, bins=100)
plt.xlabel("HPO steps")
plt.ylabel("models still training")
plt.show_and_save("HPO models vs time")
interactive.save_value(
    "waveforms used for training per HPO step", hpo.waveforms_per_step, ".1e"
)
interactive.save_value(
    "total number of waveforms trained on",
    np.sum(times) * hpo.waveforms_per_step,
    ".2e",
)
#%%
h, *_ = plt.hist2d(times, aucs, bins=30)  # , norm=mpl.colors.LogNorm())
cmap = interactive.colormap_special_zero(values=h)
plt.hist2d(times, aucs, bins=30, **cmap.get_plot_config())
plt.colorbar(label="model count")
plt.xlabel("HPO steps")
plt.ylabel("AUC")
plt.show_and_save("HPO auc vs time")

# %%
aucs = []
for d in tqdm(dfs.values()):
    aucs.append(np.mean(d.auc[-1:]))
aucs = np.array(aucs)
plt.hist(aucs[aucs > 0.7])
plt.xlabel("AUC")
plt.ylabel("number of modles")
plt.show_and_save("AUC distribution after HPO")

# %%
df = ana.dataframe().sort_values("auc", ascending=False)

# %%
models = df["config/model_factory"].unique()
best_props = []
for m in models:
    model_sel = df[df["config/model_factory"] == m]
    latents = sorted(model_sel["config/latent_size"].unique())
    v = []
    for i in latents:
        v.append(model_sel[model_sel["config/latent_size"] == i].iloc[0])
    best_props.append(v)


#%%
info = []
markers = ["v", "s", "p", "D", "8", "h", "X", "o"]
colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
for i, m in enumerate(best_props):
    for j, l in enumerate(m):
        label = None
        if j == 0:
            label = models[i]
        plt.plot(
            [i], [l.auc], marker=markers[j], linestyle="", label=label, color=colors[i]
        )

        encoder_afunc = l["config/encoder_activation"]
        decoder_afunc = l["config/decoder_activation"]
        training_loss = mappings.losses_to_humanreadable[l["config/training_loss"]]
        if encoder_afunc != encoder_afunc:  # is NaN
            # the conv model only has one activation function that's not in the hpo, so hard code it here
            encoder_afunc = "ReLU"
            decoder_afunc = "ReLU, Sigmoid"

        info.append(
            [
                models[i],
                l.auc,
                l["config/latent_size"],
                encoder_afunc,
                decoder_afunc,
                training_loss,
            ]
        )
        if l.auc > 0.7:
            plt.annotate(latents[j], [i + j * 0.05, l.auc])
info = pd.DataFrame(
    info,
    columns=[
        "architecture",
        "AUC",
        "$d_l$",
        "encoder activation",
        "decoder activation",
        "loss",
    ],
)
plt.ylim([0.7, 0.9])
plt.legend()
plt.show_and_save("all best models with all latents")
info

#%%
best = df[df.auc > 0.8]
plt.hist(best["config/latent_size"], bins=np.arange(0.5, 7.5))
plt.yscale("log")
plt.ylabel("model count")
plt.xlabel("latent dimensions $l_d$")
plt.show_and_save("high-AUC models l_d histogram")

#%%
#TODO: Instead of only giving nominal values, also give stds
best = df[df.auc > 0.8]
interactive.save_value("models with high AUC", len(best))
results = []
names = []
for name, group in best.groupby("config/model_factory"):
    auc = group.groupby("config/latent_size")["auc"].mean()
    auc_of_subgroup = group.groupby("config/latent_size")["auc"]
    std = auc_of_subgroup.std(ddof=1) / np.sqrt(auc_of_subgroup.count())
    for i, l in enumerate(auc):
        if l == "-":  # nan
            auc.iloc[i] = "-"
        else:
            auc.iloc[i] = f"{l:.2f}" # ±{std.iloc[i]:.3f}"
    results.append(auc)
    names.append(name)
ld_by_model = pd.DataFrame(results, index=names) # was previously .reshape([-1, len(auc)])
ld_by_model.columns.name = ""
ld_by_model = ld_by_model.fillna("-")

interactive.save_table(
    "latent vs model mean AUC for best AUC",
    ld_by_model,
    column_format="rSSSSSS",
    index=True,
)
ld_by_model

#%%
# do the same for std
results = []
for name, group in best.groupby("config/model_factory"):
    auc_of_subgroup = group.groupby("config/latent_size")["auc"]
    results.append(auc_of_subgroup.std(ddof=1) / np.sqrt(auc_of_subgroup.count()))
ld_std_by_model = pd.DataFrame(results, index=names)
plt.hist(ld_std_by_model.values.flatten())
interactive.save_value(
    "mean std of mean of AUC table", np.nanmean(ld_std_by_model.values), ".3f"
)

#%%
best_worst_arch_AUC_diff = float(ld_by_model.max().max()) - float(ld_by_model.replace("-",99).min().min())
interactive.save_value(
    "AUC difference by best-worst architecture", best_worst_arch_AUC_diff, ".2f"
)

# %%
best_10 = info.nlargest(10, "AUC")
interactive.save_table(
    "best 10 models", best_10, float_format="%.3f", column_format="rSSlll"
)

convs = info["architecture"] == "conv"
only_ld = lambda l: info["$d_l$"] == l
best_conv_models = []
for i in range(5):
    best_conv_models.append(info[convs & only_ld(i + 1)].nlargest(1, "AUC"))
interactive.save_value(
    "AUC difference best conv ld 1-3",
    float(best_conv_models[0].AUC.values - best_conv_models[2].AUC.values),
    ".3f",
)
interactive.save_value(
    "AUC difference best conv ld 2-3",
    float(best_conv_models[1].AUC.values - best_conv_models[2].AUC.values),
    ".3f",
)
interactive.save_value(
    "AUC difference best conv ld 3-4",
    float(best_conv_models[2].AUC.values - best_conv_models[3].AUC.values),
    ".3f",
)
interactive.save_value(
    "AUC difference best conv ld 3-5",
    float(best_conv_models[2].AUC.values - best_conv_models[4].AUC.values),
    ".3f",
)

worst_7 = info.nsmallest(7, "AUC")
interactive.save_table(
    "worst 7 models", worst_7, float_format="%.3f", column_format="rSSlll"
)

#%%
plt.hist2d(info["$d_l$"], info["AUC"], bins=[np.arange(0.5, 7.5), 6])
plt.xlabel("$d_l$")
plt.ylabel("AUC")
plt.colorbar(label="model count")
plt.show_and_save("hist best models latent vs AUC best only")

#%%
cutoff = 0.6
best = df[df.auc >= cutoff]
models_shown = len(best) / len(df)
interactive.save_value(
    "hist latent vs AUC included percentage", 100 * models_shown, ".1f"
)
ybins = 12
h, *_ = plt.hist2d(
    best["config/latent_size"], best["auc"], bins=[np.arange(0.5, 7.5), ybins]
)
cmap = interactive.colormap_special_zero(values=h)
_, xedge, yedge, _ = plt.hist2d(
    best["config/latent_size"],
    best["auc"],
    bins=[np.arange(0.5, 7.5), ybins],
    **cmap.get_plot_config(),
)
yrange = yedge.max() - yedge.min()
yticks = np.linspace(
    min(yedge) + yrange / ybins / 2, max(yedge) - +yrange / ybins / 2, ybins
)
plt.yticks(yticks, labels=[f"{i:.2}" for i in yticks])
plt.xlabel("latent dimensions $d_l$")
plt.ylabel("AUC")
plt.colorbar(label="model count")
plt.show_and_save("hist best models latent vs AUC all")

interactive.save_value("median AUC", best["auc"].median(), ".2f")

#%%
best = df[df.auc > 0.8]
models = best["config/model_factory"].unique()
plt.hist(aucs[aucs > 0.7])
plt.xlabel("AUC")
plt.ylabel("number of modles")
plt.show_and_save("AUC distribution after HPO")

#%%
best = df[df.auc > 0.6]
l_d = sorted(best["config/latent_size"].unique())
filter = [
    (best["config/latent_size"] == i) & (best["config/model_factory"] != "conv")
    for i in l_d
]
mean_performance_by_dl = np.array(
    [[best[f]["auc"].mean(), best[f]["auc"].std(ddof=1)] for f in filter]
)
plt.plot(l_d, mean_performance_by_dl[:, 0], ".")
# plt.errorbar(l_d,mean_performance_by_dl[:,0],mean_performance_by_dl[:,1])
plt.xlabel("latent dimensions $l_d$")
plt.ylabel("mean AUC")
plt.show_and_save("AUC distribution by l_d")
table = pd.DataFrame(
    np.array([l_d, mean_performance_by_dl[:, 0]]).T, columns=["$d_l$", "mean AUC"]
)
table

#%%
best = df  # [df.auc > 0.8]
models = best["config/model_factory"]
only_shown = len(best) / len(df)
interactive.save_value(
    "values included from AUC vs architecture in percent", f"{100*only_shown:.2f}"
)
unique, index = np.unique(models, return_inverse=True)

h, *_ = plt.hist2d(
    best.auc,
    index,
    bins=[np.arange(0.8, 0.90, 0.01) + 0.005, np.arange(len(unique) + 1) - 0.5],
)
cmap = interactive.colormap_special_zero(values=h)
plt.hist2d(
    best.auc,
    index,
    bins=[np.arange(0.8, 0.90, 0.01) + 0.005, np.arange(len(unique) + 1) - 0.5],
    **cmap.get_plot_config(),
)
plt.yticks(np.arange(len(unique)), unique)
plt.xlabel("AUC")
plt.colorbar(label="model count")
plt.show_and_save("AUC vs model name")


#%%
cutoff = 0.6
best = df[(df.training_loss < 0.4) & (df.training_loss > 0)]  # df[df.auc >= cutoff]
ybins = 15

params = [
    best["config/latent_size"],
    best["training_loss"],
    [np.arange(0.5, 7.5), ybins],
]

# h, *_ = plt.hist2d(*params)
# cmap = interactive.colormap_special_zero(values=h)
_, xedge, yedge, _ = plt.hist2d(*params,)  # **cmap.get_plot_config())
yrange = yedge.max() - yedge.min()
yticks = np.linspace(
    min(yedge) + yrange / ybins / 2, max(yedge) - +yrange / ybins / 2, ybins
)
plt.yticks(yticks, labels=[f"{i:.2}" for i in yticks])
plt.xlabel("latent dimensions $d_l$")
plt.ylabel(
    "EMD loss"
)  # all models in this loss range have EMD as their training loss function
plt.colorbar(label="model count")
plt.show_and_save("hist best models latent vs train loss all")

#%%
best = df[(df.val_loss < 0.4) & (df.val_loss > 0)]
not_shown = 1 - len(best) / len(df)
interactive.save_value(
    "values excluded from latent vs loss hist in percent", f"{100*not_shown:.2f}"
)

count_ybins = 10
# ybins = np.linspace(0,0.8,count_ybins)

params = [
    best["config/latent_size"],
    best["val_loss"],
    [np.arange(0.5, 7.5), count_ybins],
]

h, *_ = plt.hist2d(*params)
cmap = interactive.colormap_special_zero(values=h)
_, xedge, yedge, _ = plt.hist2d(*params, **cmap.get_plot_config())
yrange = yedge.max() - yedge.min()
yticks = np.linspace(
    min(yedge) + yrange / count_ybins / 2,
    max(yedge) - +yrange / count_ybins / 2,
    count_ybins,
)
plt.yticks(yticks, labels=[f"{i:.2}" for i in yticks])
plt.xlabel("latent dimensions $d_l$")
plt.ylabel(
    "EMD loss"
)  # all models in this loss range have EMD as their training loss function
plt.colorbar(label="model count")
plt.show_and_save("hist best models latent vs val loss all")

#%%
losses_by_dl = []
for d_l in best["config/latent_size"].unique():
    only_this_dl = best["config/latent_size"] == d_l
    losses_by_dl.append(best[only_this_dl]["val_loss"])
plt.hist(losses_by_dl, density=True, histtype="step")

# %%
# plot the n largest AUCs per model
n_largest = 2
models = info["architecture"].unique()
for i, m in enumerate(models):
    info_model = info[info["architecture"] == m]
    best_models = info_model.nlargest(n_largest, "AUC")
    best_aucs = best_models["AUC"]
    best_latents = best_models["$d_l$"]

    plt.plot(
        best_latents,
        best_aucs,
        marker=markers[i],
        linestyle="",
        label=m,
        color=colors[i],
        markersize=10,
    )


# plt.ylim([0.7,0.9])
plt.legend()
plt.xlabel("latent dimensions $d_l$")
plt.ylabel("AUC")
xint = range(1, 5 + 1)
plt.xticks(xint)
plt.show_and_save("two best models per type")

# %%
# plot best and worst AUC
for i, m in enumerate(models):
    info_model = info[info["architecture"] == m]
    best_models = info_model.nlargest(1, "AUC")
    best_aucs = best_models["AUC"]
    best_latents = best_models["$d_l$"]

    plt.plot(
        best_latents,
        best_aucs,
        marker=markers[i],
        linestyle="",
        label=m,
        color=colors[i],
        markersize=10,
    )

    info_model = info[info["architecture"] == m]
    best_models = info_model.nsmallest(1, "AUC")
    best_aucs = best_models["AUC"]
    best_latents = best_models["$d_l$"]

    plt.plot(
        best_latents,
        best_aucs,
        marker=markers[i],
        linestyle="",
        color=colors[i],
        markersize=10,
    )


# plt.ylim([0.7,0.9])
plt.legend()
plt.xlabel("latent dimensions $d_l$")
plt.ylabel("AUC")
xint = range(1, 4 + 1)
plt.xticks(xint)
plt.show_and_save("best and worst model per type")


# #%%
# filename = config.root + config.MC.filename
# device = torch.device("cuda")
# dl_config = {
#     "batch_size": 32 * 10,
#     "num_workers": 12,
#     "shuffle": True,
#     # "collate_fn": MCDataset.collate_fn,
#     "pin_memory": True,
# }

# dataset_train = MCDataset(
#     filename=filename, key="train/waveforms", transform=SingleWaveformPreprocessing()
# )
# dataset_val = MCDataset(
#     filename=filename, key="val/waveforms", transform=SingleWaveformPreprocessing()
# )

# train = DataLoader(dataset_train, **dl_config)
# val = DataLoader(dataset_val, batch_size=1024, num_workers=12)
# #%%
# # look at the fluke model with d_l = 1 and AUC>0.8
# fluke_model_df = df[df["config/latent_size"] == 1].nlargest(1, "auc")
# fluke_model_path = fluke_model_df["logdir"].values[0]
# model_file = fluke_model_path + "/checkpoint_33/model_.pth"

# model = ConvAE(config={"latent_size": 1})
# model.load_state_dict(torch.load(model_file))
# gym = Gym(model, device, EMD.torch_auto, train, val)
# loss = np.hstack(gym.validation_loss(lambda p, t: EMD.torch_auto(p, t, False)))
# plt.hist(loss)
# #%%
# latent = gym.latent_space_of_val()
# latent = np.hstack([l.reshape(-1) for l in latent])
# plt.hist(latent)
# plt.show()
# #%%
# # only look at the "second half" of the floating point number
# low_latent = [float(f"{l:.16f}"[11:]) for l in latent]
# plt.hist(low_latent)

# # %%


# %%


# %%

