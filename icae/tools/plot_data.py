import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

# import icae.tools.data_loader
from icae.tools.nn import rescale


def plot_single_ic_event(frame):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    frame["integral"] = frame.drop(columns=["x", "y", "z"]).sum(axis=1)

    plot = frame.reset_index().loc[:, ["starting_time", "integral", "x", "y", "z"]]
    plot = plot.sort_values("starting_time")
    plot["starting_time"] = list(range(len(plot["starting_time"])))

    # norm = plt.Normalize()
    # c = plt.cm.jet(norm(plot.starting_time.values))
    s = np.power(plot.integral * 1e16, 1 / 3)
    cm = plt.cm.get_cmap("jet")

    sca = ax.scatter(
        plot.x, plot.y, plot.z, s=s, c=plot.starting_time, marker=".", cmap=cm
    )  # , alpha=0.5)
    plt.colorbar(sca, label="starting time in ns")
    plt.xlabel("$x$ in m")
    plt.ylabel("$y$ in m")
    ax.set_zlabel("$z$ in m")

    return fig, ax


def plot_single_ic_event_lower_res(x, y, z, loss, time):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    cm = plt.cm.get_cmap("jet")

    sca = ax.scatter(x, y, z, s=loss, c=time, marker=".", cmap=cm)  # , alpha=0.5)
    plt.colorbar(sca, label="loss")
    plt.xlabel("$x$ in a.u.")
    plt.ylabel("$y$ in a.u.")
    ax.set_zlabel("$z$ in a.u.")

    return fig, ax


def plot_layer_output(model, output_layer_index, X, number_of_plots=3):
    raise DeprecationWarning()
    import keras
    
    m_conv = keras.Sequential(model.layers[: output_layer_index + 1])
    Y = m_conv.predict(X)
    for original, conved in zip(X[:number_of_plots], Y[:number_of_plots]):
        plt.plot(original)
        x = np.linspace(0, len(original), num=len(conved))
        plt.plot(x, conved, "C3")
        plt.title("Output of layer", model.layers[output_layer_index])
        plt.xlabel("neuron index")
        plt.ylabel("activation")
        plt.show()


def plot_latent_space(model_encoder, X, log=False):
    plt.figure(figsize=(7, 7))
    latent = model_encoder.predict(X)
    shape = latent.shape[1]

    if log:
        import matplotlib

        norm = matplotlib.colors.LogNorm()
    else:
        norm = None

    if shape == 2:
        plt.hexbin(latent[:, 1], latent[:, 0], norm=norm)
        plt.ylabel("a")
        plt.xlabel("b")
    elif shape == 3:
        plt.subplot(221)
        plt.hexbin(latent[:, 1], latent[:, 0], norm=norm)
        plt.ylabel("a")
        plt.xlabel("b")
        plt.subplot(222)
        plt.hexbin(latent[:, 2], latent[:, 0], norm=norm)
        plt.ylabel("a")
        plt.xlabel("c")
        plt.subplot(223)
        plt.hexbin(latent[:, 1], latent[:, 2], norm=norm)
        plt.ylabel("c")
        plt.xlabel("b")
    elif shape == 4:
        plt.subplot(221)
        plt.hexbin(latent[:, 1], latent[:, 0], norm=norm)
        plt.ylabel("a")
        plt.xlabel("b")
        plt.subplot(222)
        plt.hexbin(latent[:, 2], latent[:, 0], norm=norm)
        plt.ylabel("a")
        plt.xlabel("c")
        plt.subplot(223)
        plt.hexbin(latent[:, 1], latent[:, 3], norm=norm)
        plt.ylabel("d")
        plt.xlabel("b")
        plt.subplot(224)
        plt.hexbin(latent[:, 2], latent[:, 3], norm=norm)
        plt.ylabel("d")
        plt.xlabel("c")
    else:
        for i in range(shape):
            if i == 0:
                continue
            plt.subplot(shape, 1, i)
            plt.hexbin(latent[:, i - 1], latent[:, i], norm=norm)
    plt.tight_layout()


def cut_event_to_size(data, required_dim_count=2):
    shape = data.size()
    if len(shape) == required_dim_count:
        return data
    elif required_dim_count < len(shape):
        diff = len(shape) - required_dim_count
        for i in range(diff):
            data = data.select(0, 0)
        return data
    else:
        raise ValueError(
            "Data of shape "
            + str(shape)
            + " could not be reduced to"
            + " "
            + str(required_dim_count)
            + " dimensions."
        )


def plot_pooled(event, pooling_size=1, clamp_range=True, scale=False):
    # ev = cut_event_to_size(event, 2)
    ev = event
    if scale:
        ev = rescale(ev)
    if pooling_size != 1:
        ev = F.max_pool2d(ev.view(1, *ev.size()), pooling_size)
    ev = ev.view(*ev.size()[1::]).transpose(-1, -2)

    args = {}
    if clamp_range:
        args["vmin"] = 0
        args["vmax"] = 1

    ev = ev.numpy()
    plt.figure(figsize=(10, 2))
    plt.imshow(ev, cmap="coolwarm", **args)
    plt.colorbar()
