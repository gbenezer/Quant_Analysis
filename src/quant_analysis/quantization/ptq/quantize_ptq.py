import copy
import inspect
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_linear_bn_eval
from torch.utils.data import DataLoader
from torchao.quantization import quantize_

from src.quant_analysis.model_architecture import SimpleMLP


# helper function
def supports_step(config_cls):
    return "step" in inspect.signature(config_cls).parameters


# helper function to enable batch norm fusion, which enables better quantization
def fuse_mlp_bn(model: SimpleMLP):

    new_model = copy.deepcopy(model)
    new_model.eval()

    seq = new_model.linear_stack

    i = 0
    while i < len(seq) - 1:
        if isinstance(seq[i], torch.nn.Linear) and isinstance(
            seq[i + 1], torch.nn.BatchNorm1d
        ):
            fused = fuse_linear_bn_eval(seq[i], seq[i + 1])
            seq[i] = fused
            seq[i + 1] = torch.nn.Identity()
        i += 1

    return new_model


def quantize_ptq(
    base_model: Union[nn.Module, SimpleMLP],
    ao_config: Any,
    is_static: bool = False,
    quantize_device: str | torch.device = "cpu",
    data: Optional[DataLoader] = None,
    **kwargs,
):

    model = copy.deepcopy(base_model).to(device=quantize_device).eval()
    if isinstance(model, SimpleMLP):
        model = fuse_mlp_bn(model)

    try:
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
                # configs without observer flow
                quantize_(model=model, config=ao_config(**kwargs))

        else:
            quantize_(model=model, config=ao_config(**kwargs))

        return model

    except AssertionError as e:
        print(f"Skipping {ao_config.__name__}: {e}")
        return None
