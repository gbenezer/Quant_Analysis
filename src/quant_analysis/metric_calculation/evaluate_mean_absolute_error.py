import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_mae(
    model: nn.Module, dataloader: DataLoader, input_dtype: torch.dtype = torch.float32
):

    model.eval()
    device = next(model.parameters()).device
    num_samples = len(dataloader.dataset)
    absolute_error = 0.0
    loss_function = nn.L1Loss(reduction="sum")

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device=device, dtype=input_dtype)
            pred = model(X)
            y = batch[1].to(device=device, dtype=pred.dtype)
            absolute_error += loss_function(pred, y).item()

    return absolute_error / num_samples
