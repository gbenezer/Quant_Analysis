import copy
import inspect
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_linear_bn_eval
from torch.utils.data import DataLoader
from torchao.core.config import AOBaseConfig
from torchao.quantization import quantize_

from src.quant_analysis.model_architecture import SimpleMLP


# helper function to evaluate whether or not a static configuration supports
# the new interface
def supports_step(config_cls) -> bool:
    """helper function to assess if a static PTQ configuration supports the 'step'
    interface

    Args:
        config_cls (AOBaseConfig): the PTQ config to evaluate

    Returns:
        bool: whether or not the PTQ configuration supports the 'step' interface
    """
    return "step" in inspect.signature(config_cls).parameters


# helper function to enable batch norm fusion, which enables better quantization
def fuse_mlp_bn(model: SimpleMLP) -> SimpleMLP:
    """A helper function to fuse all the batch normalization layers into the linear
    layers to help with PTQ compatibility

    Args:
        model (SimpleMLP): the model with batch normalization layers to fuse

    Returns:
        SimpleMLP: a copy of the model that has no more batch normalization layers
    """
    # make sure the original model is not modified
    new_model = copy.deepcopy(model)
    new_model.eval()

    # extract the linear stack portion of the new model
    # and extract the layers as a list
    seq = new_model.linear_stack
    modules = list(seq.children())

    # while there are still layers to iterate through
    i = 0
    while i < len(modules) - 1:
        
        # if a linear layer is encountered, followed by a 1D batch norm layer
        # fuse them into one linear layer and then replace the batch norm layer
        # with the identity operation
        if isinstance(modules[i], nn.Linear) and isinstance(
            modules[i + 1], nn.BatchNorm1d
        ):
            modules[i] = fuse_linear_bn_eval(modules[i], modules[i + 1])
            modules[i + 1] = nn.Identity()
        i += 1

    # put the list back together into a Sequential module within the new model
    new_model.linear_stack = nn.Sequential(*modules)
    return new_model


def quantize_ptq(
    base_model: Union[nn.Module, SimpleMLP],
    ao_config: type[AOBaseConfig],
    is_static: bool = False,
    quantize_device: str | torch.device = "cpu",
    data: Optional[DataLoader] = None,
    **kwargs,
):
    """Take a base nn.Module (or SimpleMLP) and return a quantized copy by applying the AOBaseConfig
    PTQ method specified

    Args:
        base_model (Union[nn.Module, SimpleMLP]): The base float32 precision model
        ao_config (type[AOBaseConfig]): The TorchAO PTQ configuration
        is_static (bool, optional): Specification whether the PTQ requires static calibration.
            Defaults to False.
        quantize_device (str | torch.device, optional): The device to place the quantized model on. Defaults to "cpu".
        data (Optional[DataLoader], optional): The data to use for static quantization.
            Must be specified if is_static == True. Defaults to None.

    Returns:
        nn.Module: the quantized PyTorch network
    """
    
    # create a copy of the model and move it to the proper device
    model = copy.deepcopy(base_model).to(device=quantize_device).eval()
    
    # if the model is a SimpleMLP, fuse the batch norm layers into the linear layers
    if isinstance(model, SimpleMLP):
        model = fuse_mlp_bn(model)

    # TorchAO is fragile, so this whole thing is wrapped in a try-except block
    try:
        
        # if the static PTQ configuration supports the 'step' interface
        # PTQ calibration observers are automatically inserted and calibration
        # only requires passing data through
        if is_static:
            if supports_step(ao_config):
                quantize_(model=model, config=ao_config(step="prepare"))

                with torch.no_grad():
                    if data is not None:
                        for batch in data:
                            x = batch[0] if isinstance(batch, (tuple, list)) else batch
                            x = x.to(quantize_device)
                            model(x)

                quantize_(model=model, config=ao_config(step="convert", **kwargs))

            else:
                # If 'step' is not supported, the PTQ will be technically applied,
                # but this is effectively a no-op
                quantize_(model=model, config=ao_config(**kwargs))

        else:
            # this applies either a dynamic calibration PTQ or a weight-only PTQ
            # configuration
            quantize_(model=model, config=ao_config(**kwargs))

        return model

    except (AssertionError, RuntimeError) as e:
        # if the operation fails, emit the AssertionError or RuntimeError message
        # to allow for debugging
        print(f"Skipping {ao_config.__name__}: {e}")
        return None
