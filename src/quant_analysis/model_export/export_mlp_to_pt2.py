import re
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import torch
from torch.export import export

from src.quant_analysis.evaluation_model_construction import (
    SimpleMLP, SuperconductorLightning)


def export_mlp_to_pt2(
    file_path: Path,
    file_type: Literal["state_dict", "checkpoint"],
    pt2_path: Path,
    model_export_dtype: torch.dtype = torch.float32,
    map_location: Optional[Union[Callable, str, torch.device, Dict]] = "cpu",
    input_dim: int = 81,
    output_dim: int = 1,
    neurons: List[int] = [324, 162, 81],
    specified_activation: Literal["relu", "leaky_relu", "elu", "gelu", "celu"] = "relu",
    batch_norm: bool = True,
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

    if file_type == "checkpoint":
        lightning_model = SuperconductorLightning.load_from_checkpoint(
            checkpoint_path=file_path, map_location=map_location
        ).eval()

        model = lightning_model.model
        model = model.to(dtype=model_export_dtype).eval()

        input_sample = torch.randn(1, model.input_dim, dtype=model_export_dtype)
        
        torch.set_grad_enabled(False)
        model.eval()

        pt2_path.parent.mkdir(parents=True, exist_ok=True)
        exported_program = export(model, (input_sample,))
        torch.export.save(exported_program, pt2_path)

    elif file_type == "state_dict":
        neural_network = (
            SimpleMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                neurons=neurons,
                specified_activation=specified_activation,
                batch_norm=batch_norm,
            )
            .to(device="cpu")
            .eval()
        )

        input_dimensions = neural_network.input_dim

        state_dict = torch.load(
            f=file_path, weights_only=True, map_location=map_location
        )

        neural_network.load_state_dict(state_dict=state_dict, strict=True)
        
        torch.set_grad_enabled(False)
        neural_network.eval()

        input_sample = torch.randn(1, input_dimensions, dtype=model_export_dtype)

        pt2_path.parent.mkdir(parents=True, exist_ok=True)
        exported_program = export(neural_network, (input_sample,))
        torch.export.save(exported_program, pt2_path)

    else:
        raise ValueError(f"Invalid file type {file_type}")


if __name__ == "__main__":
    MODEL_PATH = Path.cwd() / "models"
    export_mlp_to_pt2(
        file_path=(MODEL_PATH / "checkpoints" / "base_model_FP32.ckpt"),
        file_type="checkpoint",
        pt2_path=(MODEL_PATH / "pt2" / "base_model_FP32.pt2"),
        model_export_dtype=torch.float32,
    )
