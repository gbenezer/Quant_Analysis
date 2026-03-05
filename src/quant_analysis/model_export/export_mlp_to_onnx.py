import re
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn

from src.quant_analysis.model_architecture import (
    SimpleMLP,
    SimpleMLPConfig,
    SuperconductorLightning,
)
from src.quant_analysis.model_loading.load_mlp_from_pth import load_mlp_from_pth


def export_mlp_to_onnx(
    file_path: Path,
    file_type: Literal["state_dict", "checkpoint"],
    onnx_path: Path,
    model_export_dtype: torch.dtype = torch.float32,
    map_location: Optional[Union[Callable, str, torch.device, Dict]] = "cpu",
):

    suffix_map = {
        "state_dict": re.compile(r".*\.pth$"),
        "checkpoint": re.compile(r".*\.ckpt$"),
    }

    file_suffix_regex = suffix_map[file_type]

    if file_suffix_regex.match(str(file_path)) is None:
        raise ValueError(
            f"The specified file type {file_type} does not match the path suffix"
        )

    def _export(model: nn.Module, input_dim: int, onnx_path: Path, dtype: torch.dtype):
        torch.set_grad_enabled(False)
        model.eval()

        input_sample = torch.randn(1, input_dim, dtype=dtype)

        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            (input_sample,),
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes=({0: "batch"},),
            opset_version=18,
        )

    if file_type == "checkpoint":
        lightning_model = SuperconductorLightning.load_from_checkpoint(
            checkpoint_path=file_path, map_location=map_location, weights_only=False
        ).eval()

        model = lightning_model.model
        model = model.to(dtype=model_export_dtype).eval()

        _export(
            model=model,
            input_dim=model.config.input_dim,
            onnx_path=onnx_path,
            dtype=model_export_dtype,
        )

    elif file_type == "state_dict":
        neural_network = load_mlp_from_pth(path=file_path)

        input_dimensions = neural_network.config.input_dim

        _export(
            model=neural_network,
            input_dim=input_dimensions,
            onnx_path=onnx_path,
            dtype=model_export_dtype,
        )

    else:
        raise ValueError(f"Invalid file type {file_type}")


if __name__ == "__main__":
    MODEL_PATH = Path.cwd() / "models"
    export_mlp_to_onnx(
        file_path=(MODEL_PATH / "checkpoints" / "base_model_FP32.ckpt"),
        file_type="checkpoint",
        onnx_path=(MODEL_PATH / "onnx" / "base_model_FP32.onnx"),
        model_export_dtype=torch.float32,
    )
