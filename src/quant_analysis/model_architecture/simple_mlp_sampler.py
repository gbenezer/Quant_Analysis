# uses Sobol sampling for number of neurons and discrete number of
import random
from typing import List, Optional, Tuple

import pandas as pd
import torch
from .model_configs import SimpleMLPConfig
from torch.quasirandom import SobolEngine

from .model_configs import SimpleMLPConfig


# helper function to go from [0, 1] -> [low, high]
def _scale_to_discrete(x: torch.Tensor, low: int, high: int) -> torch.Tensor:

    # get the appropriate real number between the high and low values
    real_numbers = x * (high - low + 1)

    # round the number down
    raw_integers = real_numbers.floor()

    # clamp the number so that there are no edge cases
    clamped_integers = raw_integers.clamp(low, high)

    # return a torch.int64 tensor object
    return clamped_integers.long()


# helper function to generate a set of neural network layer width samples
def _sobol_sample_layer_widths(
    number_samples: int,
    layer_bounds: List[Tuple[int, int]],
    random_seed: Optional[float] = None,
):

    # get the number of neural network layers
    number_layers = len(layer_bounds)

    # create the Sobol sampler
    sampler = SobolEngine(dimension=number_layers, scramble=True, seed=random_seed)

    # create the sample itself
    sample = sampler.draw(n=number_samples)

    # create a list of 1D column tensors of integers
    # between the appropriate bounds using a comprehension and helper function
    tensor_list = [
        _scale_to_discrete(sample[:, column], low_value, high_value)
        for column, (low_value, high_value) in enumerate(layer_bounds)
    ]

    # return the tensors stacked in the column dimension
    return torch.stack(tensors=tensor_list, dim=1)


def generate_mlp_sample_dataframe(
    number_samples: int,
    layer_bounds: List[Tuple[int, int]],
    activations_considered: List[str] = ["relu", "leaky_relu", "elu", "gelu", "celu"],
    test_batch_norm: bool = True,
    random_seed: Optional[float] = None,
):

    # evaluate whether enough samples have been requested to test each set of discrete
    # variables once
    if test_batch_norm:
        if number_samples < 2 * len(activations_considered):
            raise ValueError(
                f"Need to create at least {2 * len(activations_considered)} samples to test each discrete level once. Samples requested: {number_samples}."
            )
    else:
        if number_samples < len(activations_considered):
            raise ValueError(
                f"Need to create at least {len(activations_considered)} samples to test each discrete level once. Samples requested: {number_samples}."
            )

    # create a random number generator
    rng = random.Random(random_seed)

    # get the number of layers
    number_hidden_layers: int = len(layer_bounds)

    # create combinations if necessary
    if test_batch_norm:
        discrete_categories = [
            (activation, bn)
            for activation in activations_considered
            for bn in (True, False)
        ]
    else:
        discrete_categories = activations_considered.copy()

    # get the number of discrete categories and the remainder
    n_categories = len(discrete_categories)
    n_remainder = number_samples - n_categories

    # make sure that each category is considered at least once
    coverage = discrete_categories.copy()
    rng.shuffle(coverage)

    # create the full set of discrete samples
    if n_remainder > 0:
        remainder = rng.choices(discrete_categories, k=n_remainder)
        all_categories = coverage + remainder
    else:
        all_categories = coverage

    # create the set of widths
    width_sample = _sobol_sample_layer_widths(
        number_samples=number_samples,
        layer_bounds=layer_bounds,
        random_seed=random_seed,
    )

    # create the DataFrame data
    columns = {
        f"hidden_layer_{i + 1}_neurons": widths.tolist()
        for i, widths in enumerate(width_sample.unbind(dim=1))
    }
    columns["total_hidden_neurons"] = [
        int(torch.sum(width_sample[i, :])) for i in range(width_sample.shape[0])
    ]
    columns["neurons_per_layer"] = [
        widths.tolist() for widths in width_sample.unbind(dim=0)
    ]

    if test_batch_norm:
        columns["activation"] = [category[0] for category in all_categories]
        columns["use_batch_norm"] = [category[1] for category in all_categories]
    else:
        columns["activation"] = [category for category in all_categories]

    return pd.DataFrame(data=columns)


def generate_mlp_config_list_from_dataframe(
    df: pd.DataFrame, input_dim: int, output_dim: int
):

    # go from dataframe to list of row dictionaries
    config_arguments = df.to_dict("records")

    if "use_batch_norm" in df.columns:
        output_config_list = [
            SimpleMLPConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                neurons_per_layer=config["neurons_per_layer"],
                activation=config["activation"],
                use_batch_norm=config["use_batch_norm"],
            )
            for config in config_arguments
        ]
    else:
        output_config_list = [
            SimpleMLPConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                neurons_per_layer=config["neurons_per_layer"],
                activation=config["activation"],
                use_batch_norm=False,
            )
            for config in config_arguments
        ]

    return output_config_list


if __name__ == "__main__":
    example_layer_bound_list = [(4, 12), (8, 24), (35, 155)]
    test_dataframe = generate_mlp_sample_dataframe(
        number_samples=32, layer_bounds=example_layer_bound_list, random_seed=32
    )
    test_configs = generate_mlp_config_list_from_dataframe(
        df=test_dataframe, input_dim=81, output_dim=1
    )
    print(test_configs)
