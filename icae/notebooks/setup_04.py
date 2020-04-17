import torch
import torch.utils
from torch.nn import functional as F
from tqdm import tqdm


def cuda():
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    return device





def val_loss(val, model):
    """calculates loss for each validation sample"""
    total_loss = 0
    failed_batches = 0
    for batch in tqdm(val, desc="Calculating loss on validation"):
        batch = batch.to(device)
        batch_loss = float(F.mse_loss(model(batch), batch))
        total_loss += batch_loss
    return total_loss / (len(val) - failed_batches)

