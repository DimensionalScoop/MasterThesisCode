from torch import nn

import icae.models.event as models
from icae.tools.loss import sparse as loss_sparse
import icae.models.waveform.flexible as flexible
from icae.models.waveform.simple import ConvAE
import icae.tools.loss.EMD as EMD


activations = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "ELU": nn.ELU,
    "Hardshrink": nn.Hardshrink,
    "SELU": nn.SELU,
    "Sigmoid": nn.Sigmoid,
}

models = {
    "tall, shallow": flexible.tall_shallow,
    "short, shallow": flexible.short_shallow,
    "short, deep": flexible.short_deep,
    "tall, deep": flexible.tall_deep,
    "conv": ConvAE,
}

def CustomKL_1(p,t):
    customKL_1 = loss_sparse.CustomKL(do_not_enforce=True, allow_broadcast=True, sigmoidtize=True)
    return customKL_1(p,t)

def CustomKL_2(p,t):
    customKL_2 = loss_sparse.CustomKL(2, allow_broadcast=True, sigmoidtize=True)
    return customKL_2(p,t)

losses = {
    "CustomKL_1":CustomKL_1,
    "CustomKL_2":CustomKL_2,
    "torch_auto":EMD.torch_auto,
    "mse_loss":nn.MSELoss(),
    "l1_loss":nn.L1Loss(),
    "Cosine_similarity":loss_sparse.Cosine_similarity,
}
losses_to_humanreadable = {
    "CustomKL_1":"auto-normed KLD",
    "CustomKL_2":"enforced KL",
    "torch_auto":"EMD",
    "mse_loss":"MSE",
    "l1_loss":"MAE",
    "Cosine_similarity":"Cosine Similarity",
}

if __name__ == "__main__":
    for i in models:
        print(i)
        print(models[i]())
        print("\n")