from typing import Any, List, Union
from warnings import warn

import torch
import torch.nn as nn

from src.quant_analysis.model_architecture.model_configs import SimpleMLPConfig


class SimpleMLP(nn.Module):
    """A class defining a simple feedforward neural network

    Args:
        nn (nn.Module): the PyTorch nn.Module neural network superclass
    """
    def __init__(
        self,
        config: SimpleMLPConfig,
    ):
        """Constructor for the simple feedforward network given a SimpleMLPConfig
        dataclass containing the necessary arguments

        Args:
            config (SimpleMLPConfig): The config dataclass containing layer widths,
                activation function, and whether or not to use 1D batch normalization between layers

        Raises:
            ValueError: If the number of layers is zero, raises a ValueError
            ValueError: If the width of a single layer is not positive, raises a ValueError
            ValueError: If the activation function specified is not in the set ["relu", "leaky_relu", "elu", "gelu", "celu"],
                raises a ValueError
        """
        super().__init__()

        self.config = config

        # validate input
        if len(self.config.neurons_per_layer) <= 0:
            raise ValueError("There has to be a positive number of layers")
        
        for width in self.config.neurons_per_layer:
            if width <= 0:
                raise ValueError("Each layer has to have a positive width")

        if self.config.activation not in ["relu", "leaky_relu", "elu", "gelu", "celu"]:
            raise ValueError(
                "The activation type selected is not supported by this class"
            )

        # getting the number of hidden layers
        number_hidden_layers = len(self.config.neurons_per_layer)

        # creating new activation objects for each layer independently
        # code adaptation recommended by ChatGPT for downstream compatibility
        # with ONNX, torch.compile, and torch.fx graph tracing
        activation_map = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "celu": nn.CELU,
        }

        self.activation_cls: Union[type[nn.Module], None] = activation_map[
            self.config.activation
        ]

        if self.activation_cls is None:
            warn(
                message=f"Activation type {self.config.activation} is not valid. Defaulting to ReLU"
            )
            self.activation_cls = nn.ReLU

        # construct the linear layer sequence adaptively
        layers: List[Any] = [
            nn.Linear(
                in_features=self.config.input_dim,
                out_features=self.config.neurons_per_layer[0],
            )
        ]

        for layer in range(1, number_hidden_layers):
            # get the input and output number of features using the width list
            layer_input_features = self.config.neurons_per_layer[(layer - 1)]
            layer_output_features = self.config.neurons_per_layer[layer]

            # if batch norm is specified, add a 1D batch normalization layer
            if self.config.use_batch_norm:
                norm_layer = nn.BatchNorm1d(num_features=layer_input_features)
                layers.append(norm_layer)

            # add the activation layer
            layers.append(self.activation_cls())

            # add the linear layer
            current_linear_layer = nn.Linear(
                in_features=layer_input_features, out_features=layer_output_features
            )
            layers.append(current_linear_layer)

        # add the last layers
        # regressing to temperature
        if self.config.use_batch_norm:
            layers.append(
                nn.BatchNorm1d(num_features=self.config.neurons_per_layer[-1])
            )
        layers.append(self.activation_cls())
        layers.append(
            nn.Linear(
                in_features=self.config.neurons_per_layer[-1],
                out_features=self.config.output_dim,
            )
        )

        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Defines the forward pass of the neural network

        Args:
            x (torch.Tensor): the input sample data

        Returns:
            torch.Tensor: the output activations
        """
        return self.linear_stack(x).squeeze()


# small output smoke test to evaluate factory functionality
if __name__ == "__main__":
    config_1 = SimpleMLPConfig(
        input_dim=81,
        output_dim=1,
        neurons_per_layer=[324, 162, 81],
        activation="relu",
        use_batch_norm=True,
    )

    config_2 = SimpleMLPConfig(
        input_dim=81,
        output_dim=1,
        neurons_per_layer=[324, 162, 81],
        activation="relu",
        use_batch_norm=False,
    )

    config_3 = SimpleMLPConfig(
        input_dim=81,
        output_dim=1,
        neurons_per_layer=[324, 162, 81],
        activation="gelu",
        use_batch_norm=True,
    )

    config_4 = SimpleMLPConfig(
        input_dim=81,
        output_dim=1,
        neurons_per_layer=[1000, 500, 250, 100],
        activation="relu",
        use_batch_norm=True,
    )

    test_model = SimpleMLP(config_1)
    print(test_model)
    test_model = SimpleMLP(config_2)
    print(test_model)
    test_model = SimpleMLP(config_3)
    print(test_model)
    test_model = SimpleMLP(config_4)
    print(test_model)
