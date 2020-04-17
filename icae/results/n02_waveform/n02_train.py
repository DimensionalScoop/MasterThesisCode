#%%
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from icae.tools.dataset.single import SingleWaveformDataset, SingleWaveformPreprocessing
from icae.tools.torch.gym import Gym
from icae.models.waveform.simple import ConvAE
from icae.tools.loss import EMD
import icae.interactive_setup

plt.set_plot_path(__file__)


device = torch.device("cuda")

dataset_train = SingleWaveformDataset(
    load_waveform_only=False,
    transform=SingleWaveformPreprocessing(),
    batch_loading_size=19,
    split=0,
)
dataset_val = SingleWaveformDataset(
    load_waveform_only=False,
    transform=SingleWaveformPreprocessing(),
    batch_loading_size=1024,
    split=1,
)
train = DataLoader(dataset_train, shuffle=True, batch_size=10, num_workers=1)
val = DataLoader(dataset_val, batch_size=12, num_workers=1)

del dataset_val.table

#%%
model_config = {
    "device": device,
    "data_train": train,
    "data_val": val,
    "verbose": True,
    "max_validation_steps": 30,
}
model = Gym(ConvAE(config={"latent_size":6}), **model_config, loss_func=EMD.torch_auto)

# %%
steps = 30000
loss = model.train_batches(steps)
#%%
loss_func = lambda p, t: EMD.torch_auto(p, t, mean=False)
val_losses = np.hstack(model.validation_loss(loss_func))  # restack batches

# %%
#model.verbose = False
try:
    lossname=model.loss_func.__name__
except AttributeError:
    lossname=model.loss_func.__class__.__name__
try:
    modelname = model.model.name
except:
    modelname = model.model.__class__.__name__
name = f"on IceCube MC latent space 6: {modelname}, Loss: {lossname}"
print(f"Training {name}:")

count_training_waveforms = (steps + 1) *dataset_train.batch_loading_size*train.batch_size
x = np.linspace(0, count_training_waveforms, len(loss))
plt.plot(x, loss)
plt.xlabel("# of waveforms used for training")
plt.ylabel(f"loss {lossname}")
plt.xscale("log")
plt.figtext(0, 0, name)
plt.show_and_save(f"{name} + training")
plt.clf()
#%%
bins = 200 # int(np.sqrt(len(val_losses)))
_, bins, _ = plt.hist(val_losses, bins, density=True)
plt.xlabel("EMD loss")
plt.ylabel("frequency")
#plt.yscale('log')
plt.show_and_save(f"{name} + hist")
plt.clf()

# %%
from icae.tools.config_loader import config
torch.save(model.model.state_dict(), config.root + f"icae/models/trained/{count_training_waveforms:.0e} samples {val_losses.mean():.1e} loss latent_space 6 IceMC.pt")
# %%
