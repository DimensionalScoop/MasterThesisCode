from IPython.display import SVG
import numpy as np
from collections.abc import Iterable
from IPython.display import clear_output
from matplotlib import pyplot as plt


def live_plot(data_dict, figsize=(14, 5), title=""):
    """Plots a number of graphs into the same figure without blocking execution"""
    clear_output(wait=False)

    plt.figure(figsize=figsize)
    # plt.subplot(121)
    for label in data_dict:
        try:
            data = data_dict[label]
            x = data["x"]
            y = data["y"]
            args = data.copy()
            del args["x"]
            del args["y"]
            plt.plot(x, y, **args, label=label, alpha=0.5)
        except:
            pass
    # plt.subplot(122)
    # reset color cycle
    plt.gca().set_prop_cycle(None)
    for label in data_dict:
        try:
            data = data_dict[label]
            x = data["x"]
            y = data["y"]
            args = data.copy()
            del args["x"]
            del args["y"]
            x_i = []
            y_i = []
            y_err_i = []
            step = int(np.ceil(len(x) / 15))
            for i in range(step, len(x), step):
                x_i.append(np.mean(x[i - step : i]))
                y_i.append(np.mean(y[i - step : i]))
                y_err_i.append(np.std(y[i - step : i]))
            plt.errorbar(x_i, y_i, yerr=y_err_i, **args, label="<" + label + ">")
        except:
            pass
    plt.title(title)
    plt.grid(True)
    plt.xlabel("batch")
    plt.yscale("log")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot(model):
    raise DeprecationWarning()
    from keras.utils.vis_utils import model_to_dot
    import keras.models as models
    return SVG(
        model_to_dot(model, show_shapes=True, rankdir="LR").create(
            prog="dot", format="svg"
        )
    )


def plot_history(network_history, ylog=False):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    if isinstance(network_history, Iterable):  # there are multiple training sessions
        loss = np.concatenate([a.history["loss"] for a in network_history])
        try:
            val_loss = np.concatenate([a.history["val_loss"] for a in network_history])
        except KeyError:
            val_loss = []

        training_session_lengths = np.array(
            [len(a.history["loss"]) for a in network_history]
        )
        training_session_end_times = np.zeros(len(training_session_lengths))
        for i in range(len(training_session_lengths)):
            training_session_end_times[i] = training_session_lengths[i]
            if i > 0:
                training_session_end_times[i] += training_session_end_times[i - 1]
    else:
        loss = network_history.history["loss"]
        try:
            val_loss = network_history.history["val_loss"]
        except KeyError:
            val_loss = []
        training_session_end_times = None

    print("Min val loss:", min(val_loss))
    plt.plot(loss)
    plt.plot(val_loss)
    if training_session_end_times is not None:
        print(training_session_end_times)
        plt.vlines(
            training_session_end_times - 1,
            min(val_loss),
            max(val_loss),
            label="Next training session",
            colors="r",
        )
    plt.legend(["Training", "Validation"])
    if ylog:
        plt.yscale("log")


def load_layers(name, freeze=False):
    """
    Loads a hdf5 keras model.
    :param name:
    :param freeze: Make loaded layers non-trainable.
    :return:
    """
    raise DeprecationWarning()
    from keras.utils.vis_utils import model_to_dot
    import keras.models as models
    
    label_to_shape = models.load_model(
        "models/" + name + ".hdf5", custom_objects={"EMD": EMD.keras_no_norm}
    )
    decoding_layers = label_to_shape.layers

    if freeze:
        for l in decoding_layers:
            l.trainable = False

    return decoding_layers


def plot_event(event, skip_degeneracies=False):
    assert len(event.size()) == 3

    plt.figure(figsize=(13 * 2, 1.9 * 2))
    plt.subplot(5, 5, 1)
    for c, channel in enumerate(event):
        if skip_degeneracies and c % 5 != 0:
            continue
        plt.subplot(5, 5, c + 1)
        if channel.abs().sum() != 0:
            da = channel.transpose(0, 1).numpy()
            non_zero = np.where(da != 0)
            plt.scatter(
                *non_zero, s=np.clip((da[non_zero] + 1) * 2, 2, 10)
            )  # ,s=(c[c!=0]))
            plt.xlim(0, da.shape[0])
            plt.ylim(0, da.shape[1])
            # plt.imshow(channel.transpose(0,1))
            # plt.spy(channel.transpose(0,1))
        else:
            plt.imshow(channel.transpose(0, 1))
        plt.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,
            left=False,
            right=False,  # ticks along the top edge are off
            labeltop=False,
            labelleft=False,
            labelright=False,
            labelbottom=False,
        )  # labels along the bottom edge are off
    plt.tight_layout()


def rescale(data):
    return (data - data.min()) / (data.max() - data.min())

