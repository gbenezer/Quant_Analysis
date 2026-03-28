import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_mae(
    model: nn.Module, dataloader: DataLoader, input_dtype: torch.dtype = torch.float32
):
    """
    Computers the Mean Absolute Error (MAE) of a model over a dataset.

    This function evaluates a trained PyTorch model on data provided by a DataLoader
    and returns the average absolute difference between the model's predictions
    and the true targets.

    Params:
        model (nn.Module): The trained PyTorch model to evaluate.
        dataloader (DataLoader): An iterable DataLoader providing batches of (input, target) pairs.
        input_dtype (torch.dtype, optional): The data type to which input tensors are cast before being passed to
            the model. Default is torch.float32.
    Returns:
        float: MAE across all samples in the dataset.
    """

    # Put model in eval mode.
    model.eval()

    # Get the device the model is on (CPU or GPU).
    device = next(model.parameters()).device

    num_samples = len(dataloader.dataset)
    absolute_error = 0.0

    # L1 loss = MAE, sum to accumulate total error.
    loss_function = nn.L1Loss(reduction="sum")

    # Disable gradient tracking for speed.
    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to correct device.
            X = batch[0].to(device=device, dtype=input_dtype)
            # Get predictions.
            pred = model(X)
            # Move targets to same device.
            y = batch[1].to(device=device, dtype=pred.dtype)
            # Add total absolute error for this device.
            absolute_error += loss_function(pred, y).item()

    # Return MAE across all samples.
    return absolute_error / num_samples
