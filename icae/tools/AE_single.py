"""Methods for training models on single waveforms"""
import matplotlib.pyplot as plt
import numpy as np

import icae.tools
import icae.tools.loss
import icae.tools.loss.EMD


# FIXME: Where is this file needed? (plots, separation for single waveforms).

def seperate_outliers(model, data, plot=True):
    """
    returns (indices of inliers, indices of outliers)
    """
    x = data

    pred = np.array(model.predict(x), dtype=float)
    loss = icae.tools.loss.EMD.numpy(x, pred).flatten()
    loss_sorter = np.argsort(loss)[::-1]
    loss = loss[loss_sorter]

    # make a 'natural' cut
    x_lin = range(len(loss))
    poly = np.polyfit(x_lin, loss, 1)
    cut = np.polyval(poly, 0)

    if plot:
        plt.figure(figsize=(15.5, 8.9))
        plt.subplot(1, 2, 1)
        plt.plot(loss, label="loss on data")
        plt.plot(np.polyval(poly, x_lin), label="lin fit")

        plt.fill_between(
            x_lin, cut, max(loss), alpha=0.2, color="C2", label="reject"
        )  # hlines(cut,0,len(loss))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(loss, bins=1000)
        plt.vlines(
            [cut, max(loss)], 10, len(loss) / 1000, alpha=1, color="C2", label="reject"
        )
        plt.xlabel("loss in EMD")
        plt.ylabel("number of events")
        plt.yscale("log")
        plt.xscale("log")

    inlier, outlier = loss_sorter[loss < cut], loss_sorter[loss > cut]
    print("Outliers", 100 * len(outlier) / len(loss))
    return inlier, outlier


def plot_results_subtraction(model, X_test):
    x = X_test

    pred = np.array(model.predict(x), dtype=float)
    loss = icae.tools.loss.EMD.numpy(x, pred).flatten()
    performance = (
        "PE-Loss: "
        + str(min(loss))
        + " – "
        + str(max(loss))
        + ", mean "
        + str(np.mean(loss))
    )
    print(performance)

    sorted_loss = np.argsort(loss)[::-1]
    plt.plot(loss[sorted_loss], label="loss")
    plt.xlabel("event")
    plt.ylabel("loss")
    # plt.yscale('log')
    plt.show()

    plt.plot(loss[sorted_loss], label="loss")
    pred = np.array(model.predict(x - pred), dtype=float)
    loss = icae.tools.loss.EMD.numpy(x, pred).flatten()
    performance = (
        "PE-Loss: "
        + str(min(loss))
        + " – "
        + str(max(loss))
        + ", mean "
        + str(np.mean(loss))
    )
    print(performance)

    plt.plot(loss[sorted_loss], ",", label="loss after subtraction and re-fit")
    # plt.yscale('log')
    plt.legend()
    plt.show()

    sorted_loss = np.argsort(loss)[::-1]
    plt.plot(loss[sorted_loss], label="loss")
    plt.show()

    random = np.random.randint(0, len(x), 3)
    best = np.argmin(loss)
    worst = np.argmax(loss)

    choices = np.append([worst, best], random)
    ps = pred[choices]
    ts = x[choices]
    loss_choices = loss[choices]

    titles = ["worst", "best", "random1", "random2", "random3"]

    for p, t, title, lo in zip(ps, ts, titles, loss_choices):
        plt.plot(p.reshape((128)), "C1", label="prediction")
        plt.bar(range(128), t.reshape((128)), alpha=0.5, label="truth")

        plt.legend()
        plt.title(title + " EMD:" + str(int(lo)))
        plt.show()
