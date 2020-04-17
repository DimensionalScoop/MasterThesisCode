"""Loss that works with sparse (~1:1e5) targets."""
#import kornia
import numpy as np
import numpy.testing as npt
import torch
from torch.nn import functional as F


def Cosine_similarity(x1, x2):
    """Cosine similarity between two batches of vectors."""
    return torch.sum(x1 * x2) / (
        torch.norm(x1) * torch.norm(x2))

class NormedLoss:
    def __init__(self):
        pass

    def preprocess_input(self, input):
        #input = F.sigmoid(input)
        return input

    def preprocess_target(self, target):
        return target

    def loss(self, input, target):
        raise NotImplementedError()

    def __call__(self, input, target):
        input = self.preprocess_input(input)
        target = self.preprocess_target(target)
        return self.loss(input, target)


class CustomKL:
    """calculates KL on tensors with elements ∈ [0,1].
    uses target.sum() to norm target and input. I.e. the sum over the
    input tensor can actually be bigger than 1.
    enforcement_strength penalizes input tensors with sums bigger than 1."""

    def __init__(self, enforcement_strength=None, do_not_enforce=False, allow_broadcast=False, sigmoidtize = True):
        self.enforcement_strength = enforcement_strength
        self.do_not_enforce = do_not_enforce
        self.allow_broadcast = allow_broadcast
        self.sigmoidtize = sigmoidtize

    def channel_loss(self, input: torch.Tensor, target: torch.Tensor):
        """Calculates loss on just one channel."""
        npt.assert_equal(input.size(), target.size())
        assert input.min() >= 0
        assert target.min() >= 0
        assert input.max() <= 1
        assert target.max() <= 1

        # to avoid problems with log(0), bias everything slightly
        delta = 1e-9
        input = input + delta
        target = target + delta

        norm = target.sum()
        p = target / norm
        q = input / norm
        if self.do_not_enforce:
            q = input / input.sum()
        # Formula:
        # kl_div = (p*(p.log() - q.log())).sum()
        # but 0*log(0) = nan, which is not what we want.
        # kl_div_i = p*p.log() - p*q.log() = a + b
        # p_mask = (p==0)
        # p_log = p.log()
        # p_log[p_mask] = 0
        # a = p*p_log
        # q_mask = (q==0)

        kl_div = torch.sum(p * (p.log() - q.log()))
        assert bool(torch.isfinite(kl_div))


        if self.enforcement_strength is None:
            penalty = 0
        elif self.enforcement_strength is 'abs':
            # how much does the input diverge from a normed probability?
            norm_div = q.sum() - 1
            penalty = norm_div.abs()  # torch.exp(norm_div.abs()) - 1
        elif self.enforcement_strength>0:
            # how much does the input diverge from a normed probability?
            norm_div = q.sum() - 1
            penalty = torch.exp(norm_div.abs()) -1
            # only penalize sums bigger than 1 as KL already penalizes sums
            # lower than 1. Additionally, we'd rather underestimate the sum.
            # if norm_div > 0:
            #    if norm_div > 10:
            #        penalty = 10**self.enforcement_strength + norm_div
            #    else:
            #        penalty = norm_div.pow(self.enforcement_strength)
            # else:d
            #    penalty = 0

        return kl_div + penalty

    def __call__(self, input, target):
        # XXX: this is a slow-ish implementation
        if self.allow_broadcast:
            input = input.reshape_as(target)
        else:
            npt.assert_equal(input.size(), target.size())
        assert len(input.size()) == 3  # [batch, channel, time]
        batches, channels, time_bins = input.size()

        if self.sigmoidtize:
           input = torch.sigmoid(input)

        total_loss = 0
        for batch_i in range(batches):
            for channel_i in range(channels):
                total_loss += self.channel_loss(
                    input[batch_i, channel_i], target[batch_i, channel_i]
                )
        total_loss /= batches * channels

        return total_loss


class OpticalLoss(NormedLoss):
    def __init__(self, pooling_size=10):
        self.pooling_size = pooling_size
        super().__init__()

    def preprocess_input(self, input):
        input = super().preprocess_input(input)
        return F.max_pool2d(input, self.pooling_size)

    def preprocess_target(self, target):
        target = super().preprocess_target(target)
        return F.max_pool2d(target, self.pooling_size)

    def loss(self, input, target):
        return F.mse_loss(input, target)


class BinHeights(OpticalLoss):
    def __init__(self, bins=10, pooling_size=10):
        self.bins = bins
        self.bin_edges = np.linspace(0, 1, num=self.bins)
        super().__init__(pooling_size=pooling_size)

    def loss(self, input: torch.Tensor, target):
        # input = torch.log(torch.histc(input,self.bins,0,1).float())
        # target = torch.log(torch.histc(target,self.bins,0,1).float())
        input = input.sum()
        target = target.sum()
        return F.mse_loss(input, target)


class AC(NormedLoss):
    def __init__(self, gamma=2):
        self.gamma = gamma

    def loss(self, input, target):
        per_element = F.mse_loss(input, target, reduction="none") * (
            1 + target * 10
        ).pow(self.gamma)
        return per_element.sum() / np.prod(per_element.size())


class ACLower(AC):
    def __init__(self, gamma=2, threshold=0.01, subtract_target=False):
        super().__init__(gamma=gamma)
        self.threshold = threshold
        self.subtract_target = subtract_target

    def loss(self, input, target):
        activated_pixels = torch.sum(input > self.threshold)
        if self.subtract_target:
            activated_pixels -= torch.sum(target > self.threshold)
            activated_pixels = activated_pixels.abs()
        return super().loss(input, target) + activated_pixels


class PoolAC(OpticalLoss):
    def __init__(self, pooling_size=10, gamma=2, decay=0):
        super().__init__(pooling_size=pooling_size)
        self.gamma = gamma
        self.decay = decay

    def reduce_pool_size(self):
        self.pooling_size = max(1, self.pooling_size - 1)

    def loss(self, input, target):
        if self.decay > 0:
            print(self.gamma)
            self.gamma = max(self.gamma * (1 - self.decay), 1)
        per_element = F.mse_loss(input, target, reduction="none") * (
            1 + target * 10
        ).pow(self.gamma)
        return per_element.sum() / np.prod(per_element.size())


class Motication(NormedLoss):
    """mse, but increases the importance of non-zero elements by the
        sparsity."""

    def __init__(self, threshold=0.1, sparsity=1e-5):
        self.threshold = threshold
        self.sparsity = sparsity

    def loss(self, input, target):
        per_element = F.mse_loss(input, target, reduction="none")
        per_element[target > self.threshold] = (
            per_element[target > self.threshold] / self.sparsity
        )
        per_element[(target < self.threshold) & (input > self.threshold)] = (
            per_element[(target < self.threshold) & (input > self.threshold)]
            / self.sparsity
        )
        return per_element.sum() / np.prod(per_element.size())


class GaussACDC(AC):
    def __init__(self, gamma=2, threshold=0.1, kernel_size=(3, 3), sigma=(3, 3)):
        self.threshold = threshold
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.gamma = gamma
        self.gauss_blur = kornia.filters.GaussianBlur(kernel_size, sigma)

    def loss(self, input, target):
        target_blurred = self.gauss_blur(target)
        mse_normal = F.mse_loss(input, target, reduction="none")
        mse_blurred = F.mse_loss(input, target_blurred, reduction="none")

        per_element = mse_blurred * (1 + target * 10).pow(self.gamma)
        per_element[(target < self.threshold) & (input < self.threshold)] = mse_normal

        return per_element.sum() / np.prod(per_element.size())


# this should be true for any channel:
# assert target[0][0].sum() == 1

# kl input should be log probabilities
# pred = pred.log()
# kl target should be probabilities
# target = target

# loss_kl = F.kl_div(pred, target)

