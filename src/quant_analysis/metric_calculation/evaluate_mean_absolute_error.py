import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_mae(model: nn.Module, dataloader: DataLoader):

    model.eval()
    device = next(model.parameters()).device
    num_samples = len(dataloader.dataset)
    absolute_error = 0.0
    loss_function = nn.L1Loss(reduction="sum")

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)
            pred = model(X)
            absolute_error += loss_function(pred, y).item()

    return absolute_error / num_samples
