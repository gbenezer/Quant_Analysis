# function to load a PyTorch model from a state_dict
from pathlib import Path

import torch

from src.quant_analysis.model_architecture import SimpleMLP, SimpleMLPConfig


def load_mlp_from_pth(path: Path, device="cpu") -> SimpleMLP:
    checkpoint = torch.load(path, map_location=device)

    config = SimpleMLPConfig(**checkpoint["config"])

    model = SimpleMLP(config)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    return model


# smoke test with given directory structure
if __name__ == "__main__":
    test_model = load_mlp_from_pth(
        path=(Path.cwd() / "models" / "state_dicts" / "base_model_FP32.pth")
    )

    print("Trained State Dictionary")

    for param_tensor in test_model.state_dict():
        print(f"Parameter Tensor: {param_tensor}")
        print(f"Value:\n{test_model.state_dict()[param_tensor]}")
