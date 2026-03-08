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


@dataclass
class TransformerRegressorConfig:
    input_dim: int              # Raw feature dimension
    d_model: int                # Transformer hidden dimension
    n_heads_encoder: int        # Attention heads per encoder layer
    n_heads_decoder: int        # Attention heads per decoder layer
    n_layers_encoder: int       # Encoder layers
    n_layers_decoder: int       # Decoder layers
    # d_ff: int                 # Feedforward dimension (typically 4 * d_model; will be hard-coded as such in the code)
    dropout: float              # Dropout rate
    output_dim: int             # Regression output (1 for scalar)
    pooling: str                # "cls", "mean", or "last"
    max_seq_len: int = 1        # For tabular: 1; for time-series: sequence length
    use_positional_encoding: bool = False  # True for time-series

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
