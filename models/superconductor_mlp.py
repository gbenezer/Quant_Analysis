from typing import List, Literal
from warnings import warn

import torch
import torch.nn as nn


class SuperconductorMLP(nn.Module):
    def __init__(
        self,
        neurons: List[int] = [324, 162, 81],
        specified_activation: Literal[
            "relu", "leaky_relu", "elu", "gelu", "celu"
        ] = "relu",
        batch_norm: bool = True,
        model_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()

        # validate input
        if len(neurons) <= 0:
            raise ValueError("The number of layer widths has to be positive")

        if specified_activation not in ["relu", "leaky_relu", "elu", "gelu", "celu"]:
            raise ValueError(
                "The activation type selected is not supported by this class"
            )

        # specify that the activation should be a PyTorch Module
        # for type hinting
        self.activation: nn.Module

        # getting the number of hidden layers
        number_hidden_layers = len(neurons)

        # construct the network sequence
        match specified_activation:
            case "relu":
                self.activation = nn.ReLU()
            case "leaky_relu":
                # uses default negative slope
                self.activation = nn.LeakyReLU()
            case "elu":
                # uses default alpha
                self.activation = nn.ELU()
            case "gelu":
                self.activation = nn.GELU()
            case "celu":
                # uses default alpha
                self.activation = nn.CELU()
            case _:
                warn(
                    message="For some reason, the activation selection code fell through. Defaulting to ReLU.",
                    category=RuntimeError,
                )
                self.activation = nn.ReLU()

        # construct the linear layer sequence adaptively
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=81, out_features=neurons[0], dtype=model_dtype)
        )

        for layer in range(1, number_hidden_layers):
            # get the input and output number of features using the width list
            layer_input_features = neurons[(layer - 1)]
            layer_output_features = neurons[layer]

            # if batch norm is specified, add a 1D batch normalization layer
            if batch_norm:
                norm_layer = nn.BatchNorm1d(
                    num_features=layer_input_features, dtype=model_dtype
                )
                self.linear_stack.append(norm_layer)

            # add the activation layer
            self.linear_stack.append(self.activation)

            # add the linear layer
            current_linear_layer = nn.Linear(
                in_features=layer_input_features,
                out_features=layer_output_features,
                dtype=model_dtype,
            )
            self.linear_stack.append(current_linear_layer)

        # add the last layers
        # regressing to temperature
        if batch_norm:
            self.linear_stack.append(
                nn.BatchNorm1d(num_features=neurons[-1], dtype=model_dtype)
            )
        self.linear_stack.append(self.activation)
        self.linear_stack.append(
            nn.Linear(in_features=neurons[-1], out_features=1, dtype=model_dtype)
        )

        # store the intended dtype for the model
        self.model_dtype = model_dtype

    def forward(self, x):
        return self.linear_stack(x)


# small output smoke test to evaluate factory functionality
if __name__ == "__main__":
    test_model = SuperconductorMLP()
    print(test_model)
    test_model = SuperconductorMLP(batch_norm=False)
    print(test_model)
    test_model = SuperconductorMLP(specified_activation="gelu")
    print(test_model)
    test_model = SuperconductorMLP(neurons=[1000, 500, 250, 100])
    print(test_model)
