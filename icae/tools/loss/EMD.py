import numpy as np


def numpy(yTrue, yPred):
    r = yTrue - yPred
    r = np.cumsum(r, axis=1)
    r = np.abs(r)
    r = np.sum(r, axis=1)
    return r


def numpy_normed(yTrue, yPred):
    """Converts `yTrue` and `yPred` to PDFs by scaling them so their integral is 1."""
    t = np.reshape(yTrue, (yTrue.shape[0], 128))
    p = np.reshape(yPred, (yPred.shape[0], 128))

    t_norm = np.sum(t, axis=1)
    p_norm = np.sum(p, axis=1)

    t /= t_norm[:, None]
    p /= p_norm[:, None]
    return numpy(t, p)


def norm_area_torch(input):
    shape = list(input.shape[:-1])
    areas = input.sum(dim=-1).reshape(shape + [1])
    return input / areas.expand_as(input)


def torch(input, target, norm=False, mean_over_batch=False):
    if norm:
        input = norm_area_torch(input)
        target = norm_area_torch(target)

    r = target - input
    r = r.cumsum(1)
    r = r.abs()
    r = r.sum(dim=1)

    # the maximum EMD value should be 1
    # the farthest distance between two bins is
    # n_bins - 1
    if norm:
        r = r / (input.shape[-1] - 1)

    if mean_over_batch:
        return r.mean()
    return r


def torch_auto(input, target, mean=True):
    bins = input.shape[-1]
    input = norm_area_torch(input).reshape(-1,bins)
    target = norm_area_torch(target).reshape(-1,bins)

    r = target - input
    r = r.cumsum(1)
    r = r.abs()
    r = r.sum(dim=1)

    # the maximum EMD value should be 1
    # the farthest distance between two bins is
    # n_bins - 1
    r = r / (input.shape[-1] - 1)

    if mean:
        return r.mean()
    else:
        return r