# function to load a PyTorch model from a state_dict

from pathlib import Path
from typing import Callable, Dict, Union

import torch
import torch.nn as nn


def load_model_from_pth(
    state_dict_pth_path: Union[str, Path],
    neural_network: nn.Module,
    map_location: Union[Callable, str, torch.device, Dict] = "cpu",
    load_device: Union[str, torch.device] = "cpu",
    train: bool = False,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """_summary_

    Args:
        state_dict_pth_path (Union[str, Path]): The
        neural_network (nn.Module): the initialized model class to load the state dictionary parameters into
        map_location (Union[Callable, str, torch.device, Dict], optional): _description_. Defaults to "cpu".
        load_device (Union[str, torch.device], optional): _description_. Defaults to "cpu".
        train (bool, optional): _description_. Defaults to False.
        dtype (torch.dtype, optional): _description_. Defaults to False.

    Raises:
        FileNotFoundError: If the given path to the state dictionary is not a file

    Returns:
        nn.Module: returns the initial nn.Module neural_network with the state dictionary loaded
    """
    if isinstance(state_dict_pth_path, str):
        path = Path(state_dict_pth_path)
    else:
        path = state_dict_pth_path

    if not path.is_file():
        raise FileNotFoundError(f"State dict not found: {path}")

    state_dict = torch.load(f=path, weights_only=True, map_location=map_location)

    neural_network.load_state_dict(state_dict=state_dict, strict=True)

    neural_network.to(device=load_device)

    if dtype is not None:
        neural_network.to(dtype=dtype)

    if train:
        neural_network.train()

    else:
        neural_network.eval()

    return neural_network


# smoke test with given directory structure
if __name__ == "__main__":
    import copy

    from src.quant_analysis.evaluation_model_construction.simple_mlp import \
        SimpleMLP

    model_class_instance = SimpleMLP()

    print("Initial Random State Dictionary")

    for param_tensor in model_class_instance.state_dict():
        print(f"Parameter Tensor: {param_tensor}")
        print(f"Value:\n{model_class_instance.state_dict()[param_tensor]}")

    model = load_model_from_pth(
        state_dict_pth_path=(
            Path.cwd() / "models" / "state_dicts" / "base_model_FP32.pth"
        ),
        neural_network=copy.deepcopy(model_class_instance),
    )

    print("Trained State Dictionary")

    for param_tensor in model.state_dict():
        print(f"Parameter Tensor: {param_tensor}")
        print(f"Value:\n{model.state_dict()[param_tensor]}")
