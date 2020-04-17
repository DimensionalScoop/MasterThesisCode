#%%
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time

import sys

import icae.tools.loss as losses
import icae.tools.loss.EMD as losses_EMD


from icae.tools.loss.EMD import torch as EMD
import icae.tools.dataset as datasets
from icae.tools.config_loader import config

import icae.models.waveform.simple as models

#%%

# FIXME: deprecated references to (losses, losses_EMD, models)

device = torch.device("cuda")

dataset_config = {
    "size": -1,  # -1 : take all events
    "batch_loading_size": 2 ** 14,  # 128,
    "transform": transforms.Compose([datasets.SingleWaveformPreprocessing()]),
}
dataset_train = datasets.SingleWaveformDataset(**dataset_config, split=0)
dataset_val = datasets.SingleWaveformDataset(**dataset_config, split=1)
dataset_test = datasets.SingleWaveformDataset(**dataset_config, split=2)
dataset_all = datasets.SingleWaveformDataset(
    **dataset_config, split=0, train_val_test_split=[1, 0, 0]
)

mini_batch_size = dataset_config["batch_loading_size"] * 8  # 128*8
dataloader_config = {
    "batch_size": mini_batch_size // dataset_config["batch_loading_size"],
    "num_workers": 12,
    "collate_fn": datasets.SingleWaveformDataset.collate_fn,
    "pin_memory": True,
}

# XXX: use bigger batch_loading_size for val/test (but this could break the spacing)
train = datasets.DataLoader(dataset_train, shuffle=True, **dataloader_config)
all = datasets.DataLoader(dataset_all, shuffle=False, **dataloader_config)
val = datasets.DataLoader(dataset_val, **dataloader_config)
test = datasets.DataLoader(dataset_test, **dataloader_config)


#%%

def val_loss():
    """calculates loss for each validation sample"""
    total_loss = 0
    failed_batches = 0
    for batch in tqdm(val, desc="Calculating loss on validation"):
        batch = batch.to(device)
        batch_loss = float(F.mse_loss(model(batch), batch))
        total_loss += batch_loss
    return total_loss / (len(val) - failed_batches)


model = models.single_event.TorchSimpleConv().to(device)
optimizer = optim.Adam(model.parameters())


def train_network():
    global current_epoch
    current_epoch = 0
    global batch_running_counter
    batch_running_counter = 0
    global loss_history
    loss_history = {"validation": {"x": [], "y": []}, "training": {"x": [], "y": []}}

    count_epochs = 100
    last_val_time = 0
    val_save_interval = 30  # seconds
    for epoch in range(1, count_epochs + 1):
        current_epoch += 1
        model.train()

        for batch_idx, data in enumerate(
            tqdm(
                train,
                desc="Epoch %d" % (current_epoch),
                postfix="CUDA RAM %d MB" % (torch.cuda.memory_allocated() * 1e-6),
            )
        ):

            batch_running_counter += 1
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, data)
            loss.backward()
            optimizer.step()

            # save losses
            loss_history["training"]["x"].append(batch_running_counter)
            loss_history["training"]["y"].append(float(loss))  # don't save gradients

            if time.time() - last_val_time > val_save_interval:
                loss_history["validation"]["x"].append(batch_running_counter)
                loss_history["validation"]["y"].append(val_loss())
                last_val_time = time.time()

            tools.nn.live_plot(loss_history)


train_network()

model_name = "???"
torch.save(model.state_dict(), "../models/%s-dict.pt" % model_name)
torch.save(model, "../models/%s.pt" % model_name)

tools.nn.live_plot(loss_history)

v = loss_history["validation"]
x, y = v["x"], v["y"]

plt.plot(x, y)
plt.yscale("log")
plt.xscale("log")

for batch in val:
    break

model.load_state_dict(torch.load("../models/superlong-dict.pt"))


# +
losses = []
for batch in tqdm(val):
    batch = batch.to(device).detach()
    pred = model(batch).detach()
    loss_per_vector = tools.loss.EMD.torch(pred, batch, norm=True)
    losses.append(loss_per_vector.cpu().numpy())
losses = np.concatenate(losses)

bins = int(np.sqrt(len(losses)))
_ = plt.hist(losses, bins=bins, log=True)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("EMD loss")
plt.ylabel("frequency")
plt.savefig(config.root + "plots/single-losses.png")
plt.show()
# -

for batch, _ in zip(val, range(5)):
    batch = batch.to(device)
    pred = model(batch)
    pred = pred.detach().cpu().numpy()
    truth = batch.cpu().numpy()

    for i in range(10):
        plt.plot(truth[i], label="truth")
        plt.plot(pred[i], label="pred")
        plt.xlabel("time (20ns bins)")
        plt.ylabel("PE count (a.u.)")
        plt.legend()
        plt.savefig(config.root + "plots/wv-example%d_%d.pdf" % (_, i))
        plt.show()

cut = 1.2
bins = int(np.sqrt(len(losses)))
_ = plt.hist(losses, bins=bins)
plt.xlabel("EMD loss")
plt.ylabel("frequency")
plt.xlim(0.5e0, 2e1)
plt.vlines(cut, 0, 2.5e4)
plt.show()


# +
# save a new file only containing low-loss waveforms


def save_tensor(t: torch.Tensor):
    df = pd.DataFrame(t.numpy())
    df.to_hdf(
        config.root + config.data.single_cleaned_lv1,
        mode="a",
        append=True,
        key=config.data.hdf_key,
        format="table",
        complevel=1,
        complib="blosc:snappy",
    )


total_size = 0
saved_size = 0
for batch in tqdm(all):
    batch = batch.to(device).detach()
    pred = model(batch).detach()
    loss_per_vector = tools.loss.EMD.torch(pred, batch, norm=True)
    low_loss_waveforms = batch[loss_per_vector < cut]

    total_size += batch.size()[0]
    saved_size += float(loss_per_vector.sum())

    save_tensor(low_loss_waveforms.cpu())

print("saved %e" % (saved_size / total_size))
# + {}
# TODO: Check clean data and clean again
raise NotImplementedError()

dataset_cleaned = datasets.SingleEventsDataset(
    filename=config.root + config.data.single_cleaned_lv1,
    **dataset_config,
    split=0,
    train_val_test_split=[1, 0, 0]
)

mini_batch_size = dataset_config["batch_loading_size"] * 8  # 128*8
dataloader_config = {
    "batch_size": mini_batch_size // dataset_config["batch_loading_size"],
    "num_workers": 12,
    "collate_fn": datasets.SingleEventsDataset.collate_fn,
    "pin_memory": True,
}

# XXX: use bigger batch_loading_size for val/test (but this could break the spacing)
train = datasets.DataLoader(dataset_train, shuffle=True, **dataloader_config)
# -


loss_per_vector.shape



# %%

# %%
