from typing import List, Literal
from warnings import warn

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim=81,
        output_dim=1,
        neurons: List[int] = [324, 162, 81],
        specified_activation: Literal[
            "relu", "leaky_relu", "elu", "gelu", "celu"
        ] = "relu",
        batch_norm: bool = True,
    ):
        """_summary_

        Args:
            neurons (List[int], optional): _description_. Defaults to [324, 162, 81].
            specified_activation (Literal[ 'relu', 'leaky_relu', 'elu', 'gelu', 'celu'], optional): _description_. Defaults to 'relu'.
            batch_norm (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        super().__init__()

        # validate input
        if len(neurons) <= 0:
            raise ValueError("The number of layer widths has to be positive")

        if specified_activation not in ["relu", "leaky_relu", "elu", "gelu", "celu"]:
            raise ValueError(
                "The activation type selected is not supported by this class"
            )

        # getting the number of hidden layers
        number_hidden_layers = len(neurons)

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

        self.activation_cls = activation_map[specified_activation]

        if self.activation_cls is None:
            warn(
                message=f"Activation type {specified_activation} is not valid. Defaulting to ReLU"
            )
            self.activation_cls = nn.ReLU

        # construct the linear layer sequence adaptively
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=neurons[0])
        )

        for layer in range(1, number_hidden_layers):
            # get the input and output number of features using the width list
            layer_input_features = neurons[(layer - 1)]
            layer_output_features = neurons[layer]

            # if batch norm is specified, add a 1D batch normalization layer
            if batch_norm:
                norm_layer = nn.BatchNorm1d(num_features=layer_input_features)
                self.linear_stack.append(norm_layer)

            # add the activation layer
            self.linear_stack.append(self.activation_cls())

            # add the linear layer
            current_linear_layer = nn.Linear(
                in_features=layer_input_features, out_features=layer_output_features
            )
            self.linear_stack.append(current_linear_layer)

        # add the last layers
        # regressing to temperature
        if batch_norm:
            self.linear_stack.append(nn.BatchNorm1d(num_features=neurons[-1]))
        self.linear_stack.append(self.activation_cls())
        self.linear_stack.append(
            nn.Linear(in_features=neurons[-1], out_features=output_dim)
        )

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        return self.linear_stack(x).squeeze()


# small output smoke test to evaluate factory functionality
if __name__ == "__main__":
    test_model = SimpleMLP()
    print(test_model)
    test_model = SimpleMLP(batch_norm=False)
    print(test_model)
    test_model = SimpleMLP(specified_activation="gelu")
    print(test_model)
    test_model = SimpleMLP(neurons=[1000, 500, 250, 100])
    print(test_model)
