import torch


class Euclid:
    def __call__(self, input, target):
        assert input.size()[-1] >= 2
        dist = torch.pow(target - input, 2).sum(-1).sqrt()
        return dist.mean()  # over channels, batches

if __name__ == "__main__":
    k = Euclid()
    t = torch.rand([10, 1, 2])
    p = torch.rand([10, 1, 2])

