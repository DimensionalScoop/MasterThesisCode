#%%
from torch.utils.data import DataLoader


from icae.tools.dataset.single import SingleWaveformDataset, SingleWaveformPreprocessing
from icae.tools.torch.gym import Gym
from icae.models.waveform.simple import ConvAE

# %%

raise NotImplementedError()

dataset = SingleWaveformDataset(load_waveform_only=False,transform=SingleWaveformPreprocessing(),batch_loading_size=64)
train = DataLoader(dataset, shuffle=True, batch_size=1,num_workers=1)
# TODO: check performanc


#%%
%%timeit
for i,j in zip(train,range(1000)):
    pass
# %%


# %%
