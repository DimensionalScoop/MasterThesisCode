#%%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tqdm import tqdm
import pickle
import uuid
from box import Box
from glob import glob
from scipy.stats.stats import pearsonr
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy

from icae.tools.config_loader import config
import icae.results.n01_toy.n03_mode_training as training
from icae.tools.torch.gym import Gym
import icae.tools.loss.EMD as EMD
from icae.tools.dataset.single import SingleWaveformPreprocessing, SingleWaveformDataset
import icae.interactive_setup as interactive
from icae.models.waveform.simple import ConvAE
from icae.tools.dataset.MC import MCDataset
from icae.tools import analysis


plt.set_plot_path(__file__)
interactive.set_saved_value_yaml_root_by_filename(__file__)
device = torch.device("cuda")
dl_config = {
    "batch_size": 32 * 10,
    "num_workers": 12,
    "shuffle": True,
    # "collate_fn": MCDataset.collate_fn,
    "pin_memory": True,
}

filename = config.root + config.MC.filename
dataset_val_toy = MCDataset(
    filename=filename, key="val/waveforms", transform=SingleWaveformPreprocessing()
)
val_toy = DataLoader(dataset_val_toy, batch_size=1024, num_workers=12)


dataset_val_MC = SingleWaveformDataset(
    load_waveform_only=False,
    transform=SingleWaveformPreprocessing(),
    batch_loading_size=1024,
    split=1,
)
val_MC = DataLoader(dataset_val_MC, batch_size=12, num_workers=1)

#%%
normal_toy_runs = analysis.TrainingStability("Ribbles_w_outliers_1e6", None)
normal_toy_runs.load()

#%%
best_loss = np.argmin(normal_toy_runs.loss_train)
best_model = normal_toy_runs.model[best_loss]
model_toy = ConvAE()
model_toy.load_state_dict(best_model.to_dict())
model_toy.eval()

interactive.save_value(
    "auc of best toy model on toy val", normal_toy_runs.auc[best_loss]
)

#%%
model_MC = ConvAE({"latent_size":6})
model_MC.load_state_dict(
    torch.load(config.root + "icae/models/trained/1e+06 samples 1.3e-02 loss latent_space 6 IceMC.pt")
)
model_MC.eval()

# %%
def compare(model1,model2,validation,max_iter=-1):
    for m in [model1,model2]:
        g = Gym(m, device, lambda i, t: EMD.torch_auto(i, t, False), None, validation,max_validation_steps=max_iter)
        yield np.hstack(g.validation_loss())

loss_MC, loss_toy = compare(model_MC,model_toy,val_toy)
# %%
plt.hist([loss_MC,loss_toy],bins=int(np.sqrt(len(loss_MC))),density=True,histtype="step", label=["IceCube MC","Toy MC"])
plt.xlabel("EMD loss")
plt.ylabel("frequency")
plt.legend()
plt.show_and_save("toy vs MC on val toy")

# %%
analysis.plot_auc(loss_MC,dataset_val_toy.df.MC_type!=0)
plt.show_and_save("MC model on toy val")
interactive.save_value("AUC of MC model on toy val",analysis.calc_auc(loss_MC,dataset_val_toy.df.MC_type!=0),".2f")

#%%
max_iter = 20
loss_MC, loss_toy = compare(model_MC,model_toy,val_MC,max_iter=max_iter)
# %%
plt.hist([loss_MC,loss_toy],bins=int(np.sqrt(len(loss_MC))),density=True,histtype="step", label=["IceCube MC","Toy MC"])
plt.xlabel("EMD loss")
plt.ylabel("frequency")
plt.xlim([0,0.15])
plt.legend()
plt.show_and_save("toy vs MC on val MC")

#%%
# get validation data
val_info_and_data = []
val_data = []
integrals = []
zs = []
for d,_ in tqdm(zip(val_MC, range(max_iter))):
    val_data.append(d['data'].reshape(-1,128).numpy())
    integ = np.array([i.numpy() for i in d['info']['integral']])
    integrals.append(integ.flatten())
    z = np.array([i.numpy() for i in d['info']['z']])
    zs.append(z.flatten())

val_data = np.concatenate(val_data)
integrals = np.concatenate(integrals)
integrals -= integrals.min()
zs = np.concatenate(zs)
#%%
#val_info = np.concatenate([d['info'].numpy() for d in val_info_and_data])
#val_data = np.concatenate([d['data'].reshape(-1,128).numpy() for d in val_info_and_data])
# %%
plt.hist(loss_MC,bins=int(np.sqrt(len(loss_MC))),density=True,histtype="step", label="IceCube MC")
plt.xlabel("EMD loss")
plt.ylabel("frequency")
plt.xlim([0,0.06])
#plt.yscale("log")
plt.legend()
plt.grid()
plt.show_and_save("MC on val MC")

#%%
log_int = np.log(integrals+integrals.mean()/100)
std = np.log(log_int).std()
mean = np.log(log_int).mean()
rang = [mean-std, mean+std]
#integrals_in = log_int[(log_int>rang[0])&(log_int<rang[1])]

integral_bins = np.linspace(-20.45,-20.2)
loss_bins = np.linspace(0,0.04)

plt.hist2d(loss_MC,log_int,bins=[loss_bins,integral_bins]);#, bins=[[0,0.075],rang]);
#%%
first_peak = np.where((0.008>loss_MC))[0]
second_peak = np.where((0.01<loss_MC) & (loss_MC<0.02))[0]
high_loss = np.where((loss_MC>0.15))[0]
len(high_loss)

#%%
for i,_ in zip(high_loss,range(50)):
    waveform = val_data[i]
    integral = log_int[i]
    plt.plot(waveform)
    plt.annotate(f"{integral:.2e}",(0,0))
    plt.show()

#%%
for i,_ in zip(high_loss,range(50)):
    waveform = val_data[i+50]
    t = torch.tensor(waveform.reshape(1,1,128)).to(device)
    reconstruction = model_MC(t).cpu().detach().numpy().flatten()
    reconstruction /= reconstruction.max()
    plt.plot(waveform,label="MC data")
    plt.plot(reconstruction,label="reconstruction")
    plt.legend()
    plt.show()


# %%
