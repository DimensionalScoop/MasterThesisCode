import torch


def clip_scale(batch, items_normalized):
    """modifies input in-place."""
    for i, batch_element in enumerate(batch):
        for j, channel in enumerate(batch_element):
            if channel.sum() == 0:
                continue
            if items_normalized:
                batch[i, j] = channel / channel.max()
            else:
                batch[i, j] = (channel - channel.min()) / (
                    channel.max() - channel.min()
                )
    return batch


def scale_to_01(batch):
    output = torch.zeros_like(batch)
    for i, batch_element in enumerate(batch):
        for j, channel in enumerate(batch_element):
            if channel.abs().sum() == 0:
                continue
            output[i, j] = (channel - channel.min()) / (channel.max() - channel.min())
    assert output.max() <= 1
    assert output.min() >= 0
    return output


def channels_to_pdf(input: torch.Tensor):
    size = len(input.size())
    if size == 3:
        input = input.view(1, *input.size())
    else:
        assert size == 4

    returnValue = torch.empty_like(input)
    for i, event in enumerate(input):
        for j, channel in enumerate(event):
            if __debug__:
                assert channel.min() >= 0
                assert channel.max() <= 1
            returnValue[i, j] = channel / channel.sum()

    if size == 3:
        returnValue = returnValue.view(*input.size()[1::])
    return returnValue

