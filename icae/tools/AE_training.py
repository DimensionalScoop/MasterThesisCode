"""
Provides methods used by almost all AEs.
Load MC data, preprocess it, train models,
plot result graphs.
"""

import sys

import icae.tools.loss
import icae.tools.loss.EMD

sys.path.append("..")

from icae.tools import EMD
from icae.tools import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def preprocess(wf):
    from sklearn.preprocessing import MinMaxScaler

    mm = MinMaxScaler()
    return mm.fit_transform(wf.T).T


def translational_symmetry(wf):
    return np.roll(wf, np.random.randint(0, 128, size=len(wf)), axis=1)


def split_train_val(wf):
    events = wf
    labels = range(len(wf))

    X_train, X_test, y_train, y_test = train_test_split(
        events, labels, test_size=0.33, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print(events.shape[0], "\tevents,", events.shape)
    print(X_train.shape[0], "\tfor training and")
    print(X_val.shape[0], "\tfor validation and")
    print(X_test.shape[0], "\tfor testing and")
    return X_train, X_val, X_test


def split_val(wf, test_size=0.05):
    events = wf
    labels = range(len(wf))

    X_train, X_val, y_train, y_val = train_test_split(
        events, labels, test_size=test_size, random_state=42
    )

    print(events.shape[0], "events,", events.shape)
    print(X_train.shape[0], "for training and")
    print(X_val.shape[0], "for validation")
    return X_train, X_val


def train(
    model, X_train, X_val=None, verbose=0, test_size=0.05, batch_size=10000, epochs=40
):
    hist = []
    if X_val is None:
        print("Splitting validation…")
        X_train, X_val = split_val(X_train, test_size=test_size)
    data = {"x": X_train, "y": X_train, "validation_data": (X_val, X_val)}

    print("Training…")
    history = model.fit(
        batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True, **data
    )
    hist.append(history)
    nn.plot_history(hist, True)
    return history


def plot_results(model, X_test):
    x = X_test

    pred = np.array(model.predict(x), dtype=float)
    loss = tools.loss.EMD.numpy(x, pred).flatten()
    performance = (
        "PE-Loss: "
        + str(min(loss))
        + " – "
        + str(max(loss))
        + ", mean "
        + str(np.mean(loss))
    )
    print(performance)

    plt.figure(figsize=[15.5, 8.9])
    plt.subplot(231)
    plt.plot(np.sort(loss)[::-1])
    plt.xlabel("event")
    plt.ylabel("loss")

    plt.subplot(234)
    plt.hist(loss, len(loss) // 1000)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel("loss")
    plt.ylabel("number ov events")

    random = np.random.randint(0, len(x), 3)
    best = np.argmin(loss)
    worst = np.argmax(loss)

    choices = np.append([worst, best], random)
    ps = pred[choices]
    ts = x[choices]
    loss_choices = loss[choices]
    subplot = [232, 233, 235, 236]

    titles = ["worst", "best", "random1", "random2", "random3"]

    for p, t, title, lo, sub in zip(ps, ts, titles, loss_choices, subplot):
        plt.subplot(sub)
        plt.plot(p.reshape((128)), "C1", label="prediction")
        plt.bar(range(128), t.reshape((128)), alpha=0.5, label="truth")

        plt.legend()
        plt.title(title + " EMD:" + str(int(lo)))
    plt.tight_layout()
