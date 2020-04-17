#%%
import icae.interactive_setup
import datetime
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
# %%
import torch.nn as nn
import torch.optim as optim

import icae.notebooks.setup_04 as setup
import icae.toy.lone_point_models as models
from icae.toy.lone_point_generator import LonePoint
import icae.tools.nn

device = setup.cuda()
#%%

test_config = {"edge_length": 1024, "dims": 2}

dataloader_config = {
    "batch_size": 10,
    "num_workers": 12,
    # 'collate_fn': SparseEventDataset.collate_fn,
    "pin_memory": True,
}
dataset = LonePoint(**test_config)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, **dataloader_config)

#%%
import icae.tools.loss.sparse as losses


class CustomKLOnCoords(losses.CustomKL):
    def __call__(self, input, target):
        batch_size = input.size()[0]
        output_shape = [batch_size] + dataset.shape
        assert len(output_shape) == 4

        in_ = []
        for i in input:
            in_.append(dataset.coords_to_data(i))
        input = torch.stack(in_).view(output_shape)

        tar_ = []
        for i in target:
            tar_.append(dataset.coords_to_data(i))
        input = torch.stack(tar_).view(output_shape)

        return super().__call__(input, target)


loss_function = nn.MSELoss()  # losses.CustomKL(2)
reference_losses = {"mse": nn.MSELoss(), "mae": nn.L1Loss()}


def visualize_sample(sample):
    data = sample["data"].to(device)
    pred = model(data).detach().cpu()
    truth = sample["coords"]
    a = pred[:100]
    b = truth[:100]
    plt.scatter(a[:, 0], a[:, 1], marker="x", label="pred")
    plt.scatter(b[:, 0], b[:, 1], marker="+", label="truth")
    plt.legend()


def train_network(continue_training=False, failover=False, do_validation=True):
    global current_epoch
    if not continue_training:
        current_epoch = 0
    global batch_running_counter
    if not continue_training:
        batch_running_counter = 0
    global loss_history
    if not continue_training:
        loss_history = {"training": {"x": [], "y": []}}

    expected_mse = dataset.expected_mse(0.01)

    count_epochs = 10000
    timer = datetime.datetime.now()
    starting_time = datetime.datetime.now()
    while (datetime.datetime.now() - starting_time).seconds < 60:
        current_epoch += 1
        model.train()
        try:
            for batch_idx, sample in enumerate(
                # tqdm(dataloader, desc="Epoch %d" % (current_epoch))):
                dataloader
            ):
                batch_running_counter += 1

                data = sample["data"].to(device)
                label = sample["coords"].to(device)
                optimizer.zero_grad()
                pred = model(data)

                now = datetime.datetime.now()
                now = (now - starting_time).seconds
                for l in reference_losses:
                    # don't save gradient
                    loss = float(reference_losses[l](pred, label))
                    if l not in loss_history:
                        loss_history[l] = {"x": [], "y": []}
                    loss_history[l]["x"].append(now)
                    loss_history[l]["y"].append(loss)

                loss = loss_function(pred, label)
                loss_history["training"]["x"].append(now)
                loss_history["training"]["y"].append(float(loss))
                loss.backward()
                optimizer.step()

                if timer + datetime.timedelta(seconds=10) < datetime.datetime.now():
                    timer = datetime.datetime.now()
                    # loss_history['target (0.01 noise level)']['x'].append(
                    #     now)
                    # loss_history['target (0.01 noise level)']['y'].append(
                    #     expected_mse)  # don't save gradients

                    icae.tools.nn.live_plot(loss_history)
                    visualize_sample(sample)

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
model = models.SimpleClassifier(
    dataset.shape[1:],
    50,
    2,
    # 2,
    kernel=[3, 3],
    channel_progression=lambda x: x + 1,
    batch_normalization=True,
    conv_init=nn.init.zeros_,
)
model = model.to(device)
optimizer = optim.Adam(model.parameters())
train_network(continue_training=False, failover=False, do_validation=False)
#%%
print(model)
for sample in dataloader:
    data = sample["data"].to(device)
    pred = model(data).detach().cpu()
    truth = sample["coords"]

    df = pd.DataFrame(
        torch.cat((truth, pred), 1).numpy(),
        columns=["label_a", "label_b", "pred_a", "pred_b"],
    )
    df = df[["label_a", "pred_a", "pred_b", "label_b"]]
    break
df
#%%


#%%
