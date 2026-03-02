# file for model architecture config dataclass definitions and helpers
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

# dataclass definition section


@dataclass
class SimpleMLPConfig:
    input_dim: int
    output_dim: int
    neurons_per_layer: list[int]
    activation: str
    use_batch_norm: bool


# helper function section


# generic model config saving
def save_model_config(config, path: Path) -> None:
    if not is_dataclass(config):
        raise ValueError("The config must be a valid dataclass")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=4)


def load_model_config(path: Path, config_cls):
    with open(path) as f:
        data = json.load(f)

    if not is_dataclass(config_cls):
        raise ValueError("config_cls must be a dataclass")

    return config_cls(**data)
