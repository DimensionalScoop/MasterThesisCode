# XXX: Broken references; won't fix unless needed. Easy to fix.

# %%
try:
    from IPython import get_ipython

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    # get_ipython().run_line_magic('matplotlib', 'inline')
except AttributeError:
    print("No ipython for you.")
# %%
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import time
import kornia
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import icae.tools.plot_data
import icae.models.event_resize_decoder as models
import icae.models.event as big_models
import icae.tools.nn
import icae.notebooks.setup_04 as setup
import icae.tools.torch_preproc as preproc

# %%
device = setup.cuda()
# num_workers=0 for debugging purposes
train, val, test = setup.load_train_val_test(batch_size=5, num_workers=0)

# %%
import icae.tools.loss.sparse as losses

loss_function = losses.CustomKL(2)


def train_network(continue_training=False, failover=False, do_validation=True):
    global current_epoch
    if not continue_training:
        current_epoch = 0
    global batch_running_counter
    if not continue_training:
        batch_running_counter = 0
    global loss_history
    if not continue_training:
        loss_history = {
            "training": {"x": [], "y": []},
            "training mse": {"x": [], "y": []},
            "training optical": {"x": [], "y": []},
        }  # 'validation':{'x':[],'y':[]},

    count_epochs = 100
    last_val_time = 0
    val_save_interval = 60 * 1  # seconds
    for epoch in range(1, count_epochs + 1):
        current_epoch += 1
        model.train()
        try:
            for batch_idx, data in enumerate(
                tqdm(train, desc="Epoch %d" % (current_epoch))
            ):

                batch_running_counter += 1
                data = data.to(device)
                optimizer.zero_grad()
                pred = model(data)

                loss_std = F.mse_loss(pred, data)
                loss = loss_function(pred, data)
                loss.backward()
                optimizer.step()

                # save losses
                loss_history["training"]["x"].append(batch_running_counter)
                loss_history["training"]["y"].append(
                    float(loss)
                )  # don't save gradients
                loss_history["training mse"]["x"].append(batch_running_counter)
                loss_history["training mse"]["y"].append(
                    float(loss_std)
                )  # don't save gradients

                if do_validation and time.time() - last_val_time > val_save_interval:
                    loss_history["validation"]["x"].append(batch_running_counter)
                    loss_history["validation"]["y"].append(val_loss())
                    last_val_time = time.time()

                if batch_idx % 5 == 0:
                    tools.nn.live_plot(loss_history)
        except RuntimeError:
            print(sys.exc_info())
            if failover:
                continue
            else:
                raise
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break


# %%
torch.autograd.set_detect_anomaly(True)  # Debug
model = models.Postman(conv_init=nn.init.ones_).to(device)
optimizer = optim.Adam(model.parameters())
train_network(continue_training=False, failover=False, do_validation=False)

# %%
loss_function = losses.CustomKL(enforcement_strength=2)
train_network(continue_training=True, failover=False, do_validation=False)

# %%
loss_ac.pooling_size = 1

# %%
for test_sample in test:
    break
test_sample_gpu = test_sample.to(device)
pred = model(test_sample_gpu).detach().cpu()

# %%
pred[0, 0].sum()

# %%
test_sample[0, 0].max()

# %%

# %%
for test_sample in test:
    # event = event.view(1,*event.size())
    # print(event.size())
    scale = 10
    tools.plot_data.plot_pooled(test_sample[0, 0], scale)
    test_sample_gpu = test_sample.to(device)
    pred = model(test_sample_gpu).detach().cpu()
    tools.plot_data.plot_pooled(pred[0, 0], scale)
    plt.show()

    true_hist = preproc.scale_to_01(test_sample[0]).numpy().flatten()
    pred_hist = preproc.scale_to_01(pred[0]).numpy().flatten()
    m = plt.hist([true_hist, pred_hist], bins=10, label=["true", "pred"])
    plt.legend()
    plt.yscale("log")
    plt.show()

# %%
m

# %%
bin_edges = np.arange(0, 1, 0.1)
bin_heights, bin_edges = np.histogram([true_hist, pred_hist])
# bin_heights = np.log(bin_heights)
# plt.plot(bin_edges[:-1],bin_heights)
# %%
for data in train:
    for event in data:
        print("Truth")
        tools.nn.plot_event(event, True)
        plt.show()
        print("Prediction")
        event_ = event.view(1, *event.size())  # fake batch dimension
        pred = model(event_.to(device)).detach().cpu()[0]
        tools.nn.plot_event(pred, True)
        break
    break

# %%
print(torch.sum(pred[0] != 0))
plt.figure(figsize=(10, 2))
plt.spy(pred[0].transpose(1, 0))

# %%


# %%
tools.plot_data.plot_pooled(plt.imshow(pred[0]), 1, True)

# %%
