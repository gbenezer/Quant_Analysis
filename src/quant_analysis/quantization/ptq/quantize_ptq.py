import copy
import inspect
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_linear_bn_eval
from torch.utils.data import DataLoader
from torchao.core.config import AOBaseConfig
from torchao.quantization import Int4Tensor, Int4WeightOnlyConfig, quantize_

from src.quant_analysis.model_architecture import SimpleMLP


# helper function to evaluate whether or not a static configuration supports
# the new interface
def supports_step(config_cls):
    return "step" in inspect.signature(config_cls).parameters


# helper function to enable batch norm fusion, which enables better quantization
def fuse_mlp_bn(model: SimpleMLP) -> SimpleMLP:
    new_model = copy.deepcopy(model)
    new_model.eval()

    seq = new_model.linear_stack
    modules = list(seq.children())

    i = 0
    while i < len(modules) - 1:
        if isinstance(modules[i], nn.Linear) and isinstance(
            modules[i + 1], nn.BatchNorm1d
        ):
            modules[i] = fuse_linear_bn_eval(modules[i], modules[i + 1])
            modules[i + 1] = nn.Identity()
        i += 1

    new_model.linear_stack = nn.Sequential(*modules)
    return new_model


# helper function to evaluate which layers of an Int4 quantized model are actually quantized
# to prevent silent failure
def check_int4_quantization(model: nn.Module) -> Dict[str, Dict[str, Any]]:

    results = {}

    for name, module in model.named_modules():
        entry = {
            "module_type": type(module).__name__,
            "has_weight": False,
            "is_int4_quantized": False,
        }

        if hasattr(module, "weight"):
            entry["has_weight"] = True

            try:
                weight = module.weight

                if isinstance(weight, Int4Tensor):
                    entry["is_int4_quantized"] = True

            except Exception:
                pass

        results[name] = entry

    return results


def quantize_ptq(
    base_model: Union[nn.Module, SimpleMLP],
    ao_config: type[AOBaseConfig],
    is_static: bool = False,
    quantize_device: str | torch.device = "cpu",
    data: Optional[DataLoader] = None,
    **kwargs,
):

    model = copy.deepcopy(base_model).to(device=quantize_device).eval()
    if isinstance(model, SimpleMLP):
        model = fuse_mlp_bn(model)

    if issubclass(ao_config, Int4WeightOnlyConfig) and torch.device(
        device=quantize_device
    ) == torch.device("cuda"):
        print("For Int4WeightOnlyConfig, model must be contiguous on GPU and also BF16")
        print("Converting to BF16 prior to quantization")
        model = model.to(dtype=torch.bfloat16)

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

    except (AssertionError, RuntimeError) as e:
        print(f"Skipping {ao_config.__name__}: {e}")
        return None
