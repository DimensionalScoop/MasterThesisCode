#%%
import os

from IPython import get_ipython

from icae.notebooks.setup_04 import cuda, load_train_val_test, val_loss

try:
    os.chdir(os.path.join(os.getcwd(), "notebooks"))
    print(os.getcwd())
except:
    pass

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
get_ipython().run_line_magic("matplotlib", "inline")


import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import time
import kornia
import sys

from icae.tools.nn import plot_event
from icae.tools.plot_data import plot_pooled
from icae.tools.torch_preproc import clip_scale


#%%
device = cuda()
train, val, test = load_train_val_test()

#%%
import icae.tools.loss.sparse as losses

loss_ac = losses.AC(2)


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
            "validation": {"x": [], "y": []},
            "training": {"x": [], "y": []},
            "training mse": {"x": [], "y": []},
        }

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
                data /= data.max()
                optimizer.zero_grad()
                pred = model(data)

                loss_std = F.mse_loss(pred, data)
                loss = loss_ac(pred, data)
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

                tools.nn.live_plot(loss_history)
        except RuntimeError:
            print(sys.exc_info())
            if failover:
                continue
            else:
                raise


#%%
model = icae.models.Multipack().to(device)
optimizer = optim.Adam(model.parameters())
train_network(continue_training=False, failover=False, do_validation=False)

# import gc
# del model, optimizer, loss_history
# torch.cuda.empty_cache()
# gc.collect()

#%%
for data in train:
    for event in data:
        print("Truth")
        plot_event(event)
        plt.show()
        break
        print("Prediction")
        event_ = event.view(1, *event.size())  # fake batch dimension
        pred = model(event_.to(device)).view(*event.size()).detach().cpu()
        plot_event(pred)
        plt.show()
        break
    break

#%%

#%%

ev = event[0].transpose(-1, -2)

gauss_blur = kornia.filters.GaussianBlur((9, 9), (1, 1))
ev = gauss_blur(ev.view(1, 1, *ev.size()))

ev = ev.view(ev.size()[2::])
plt.figure(figsize=(20, 4))
plt.imshow(ev, norm=LogNorm())
plt.colorbar()

#%%
event_ = event.view(1, *event.size())  # fake batch dimension
pred = model(event_.to(device)).detach().cpu()[0]
gauss_blur = kornia.filters.GaussianBlur((21, 21), (1, 1))
ev = gauss_blur(event_)

plot_pooled(ev[0][0], 5)
plt.title("blurred MC truth used for loss")

plot_pooled(clip_scale(event_, False)[0][0], 5, True)
plt.title("MC truth")

plot_pooled(clip_scale(F.sigmoid(pred[0]), True), 5, True)
plt.title("prediction")

#%%
# model_name = "???"
# torch.save(model.state_dict(),"../models/%s-dict.pt"%model_name)
# torch.save(model,"../models/%s.pt"%model_name)

