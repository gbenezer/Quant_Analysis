import pytest
import pandas as pd
import torch

from src.quant_analysis.model_architecture.simple_mlp_sampler import (
    _scale_to_discrete,
    _sobol_sample_layer_widths,
    generate_mlp_sample_dataframe,
    generate_mlp_config_list_from_dataframe,
)
from src.quant_analysis.model_architecture.model_configs import SimpleMLPConfig


# ---------------------------------------------------------------------------
# _scale_to_discrete
# ---------------------------------------------------------------------------

class TestScaleToDiscrete:
    def test_zero_maps_to_low(self):
        x = torch.tensor([0.0])
        result = _scale_to_discrete(x, low=2, high=8)
        assert result.item() == 2

    def test_one_maps_to_high(self):
        # x=1.0 overflows without clamp; clamp ensures result == high
        x = torch.tensor([1.0])
        result = _scale_to_discrete(x, low=2, high=8)
        assert result.item() == 8

    def test_mid_value_in_range(self):
        x = torch.tensor([0.5])
        result = _scale_to_discrete(x, low=0, high=9)
        assert 0 <= result.item() <= 9

    def test_output_dtype_is_int64(self):
        x = torch.rand(10)
        result = _scale_to_discrete(x, low=1, high=5)
        assert result.dtype == torch.int64

    def test_all_values_within_bounds(self):
        x = torch.rand(1000)
        result = _scale_to_discrete(x, low=4, high=12)
        assert result.min().item() >= 4
        assert result.max().item() <= 12

    def test_same_low_and_high_always_returns_that_value(self):
        x = torch.rand(20)
        result = _scale_to_discrete(x, low=7, high=7)
        assert (result == 7).all()


# ---------------------------------------------------------------------------
# _sobol_sample_layer_widths
# ---------------------------------------------------------------------------

class TestSobolSampleLayerWidths:
    def test_output_shape(self):
        bounds = [(4, 16), (8, 32), (2, 8)]
        result = _sobol_sample_layer_widths(number_samples=10, layer_bounds=bounds)
        assert result.shape == (10, 3)

    def test_output_dtype_is_int64(self):
        bounds = [(4, 16)]
        result = _sobol_sample_layer_widths(number_samples=5, layer_bounds=bounds)
        assert result.dtype == torch.int64

    def test_values_within_bounds(self):
        bounds = [(4, 16), (8, 32)]
        result = _sobol_sample_layer_widths(number_samples=50, layer_bounds=bounds)
        assert result[:, 0].min().item() >= 4
        assert result[:, 0].max().item() <= 16
        assert result[:, 1].min().item() >= 8
        assert result[:, 1].max().item() <= 32

    def test_reproducible_with_seed(self):
        bounds = [(4, 16), (8, 32)]
        r1 = _sobol_sample_layer_widths(10, bounds, random_seed=42)
        r2 = _sobol_sample_layer_widths(10, bounds, random_seed=42)
        assert torch.equal(r1, r2)

    def test_single_layer(self):
        result = _sobol_sample_layer_widths(5, [(1, 10)])
        assert result.shape == (5, 1)


# ---------------------------------------------------------------------------
# generate_mlp_sample_dataframe
# ---------------------------------------------------------------------------

LAYER_BOUNDS = [(4, 16), (8, 32)]
DEFAULT_ACTIVATIONS = ["relu", "leaky_relu", "elu", "gelu", "celu"]


class TestGenerateMlpSampleDataframe:
    def test_returns_dataframe(self):
        df = generate_mlp_sample_dataframe(10, LAYER_BOUNDS)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_number_samples(self):
        df = generate_mlp_sample_dataframe(12, LAYER_BOUNDS)
        assert len(df) == 12

    def test_expected_columns_with_batch_norm(self):
        df = generate_mlp_sample_dataframe(10, LAYER_BOUNDS, test_batch_norm=True)
        assert "hidden_layer_1_neurons" in df.columns
        assert "hidden_layer_2_neurons" in df.columns
        assert "total_hidden_neurons" in df.columns
        assert "neurons_per_layer" in df.columns
        assert "activation" in df.columns
        assert "use_batch_norm" in df.columns

    def test_expected_columns_without_batch_norm(self):
        df = generate_mlp_sample_dataframe(5, LAYER_BOUNDS, test_batch_norm=False)
        assert "activation" in df.columns
        assert "use_batch_norm" not in df.columns

    def test_all_activations_covered(self):
        activations = ["relu", "elu"]
        df = generate_mlp_sample_dataframe(
            4, LAYER_BOUNDS, activations_considered=activations, test_batch_norm=False
        )
        assert set(activations).issubset(set(df["activation"].tolist()))

    def test_all_activations_and_bn_combos_covered(self):
        activations = ["relu", "elu"]
        df = generate_mlp_sample_dataframe(
            4, LAYER_BOUNDS, activations_considered=activations, test_batch_norm=True
        )
        combos = set(zip(df["activation"], df["use_batch_norm"]))
        for act in activations:
            assert (act, True) in combos
            assert (act, False) in combos

    def test_neuron_widths_within_bounds(self):
        df = generate_mlp_sample_dataframe(20, LAYER_BOUNDS)
        assert df["hidden_layer_1_neurons"].between(4, 16).all()
        assert df["hidden_layer_2_neurons"].between(8, 32).all()

    def test_total_hidden_neurons_is_row_sum(self):
        df = generate_mlp_sample_dataframe(10, LAYER_BOUNDS)
        expected = df["hidden_layer_1_neurons"] + df["hidden_layer_2_neurons"]
        assert (df["total_hidden_neurons"] == expected).all()

    def test_neurons_per_layer_matches_individual_columns(self):
        df = generate_mlp_sample_dataframe(10, LAYER_BOUNDS)
        for _, row in df.iterrows():
            assert row["neurons_per_layer"] == [
                row["hidden_layer_1_neurons"],
                row["hidden_layer_2_neurons"],
            ]

    def test_raises_too_few_samples_with_batch_norm(self):
        # need >= 2 * len(activations) = 10 samples
        with pytest.raises(ValueError, match="at least"):
            generate_mlp_sample_dataframe(9, LAYER_BOUNDS, test_batch_norm=True)

    def test_raises_too_few_samples_without_batch_norm(self):
        # need >= len(activations) = 5 samples
        with pytest.raises(ValueError, match="at least"):
            generate_mlp_sample_dataframe(4, LAYER_BOUNDS, test_batch_norm=False)

    def test_reproducible_with_seed(self):
        df1 = generate_mlp_sample_dataframe(10, LAYER_BOUNDS, random_seed=7)
        df2 = generate_mlp_sample_dataframe(10, LAYER_BOUNDS, random_seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_custom_activations_only_appear_in_output(self):
        activations = ["relu", "gelu"]
        df = generate_mlp_sample_dataframe(
            4, LAYER_BOUNDS, activations_considered=activations, test_batch_norm=False
        )
        assert set(df["activation"].unique()).issubset(set(activations))


# ---------------------------------------------------------------------------
# generate_mlp_config_list_from_dataframe
# ---------------------------------------------------------------------------

class TestGenerateMlpConfigListFromDataframe:
    def _sample_df(self, n=10, test_batch_norm=True):
        return generate_mlp_sample_dataframe(
            n, LAYER_BOUNDS, test_batch_norm=test_batch_norm, random_seed=0
        )

    def test_returns_list(self):
        df = self._sample_df()
        configs = generate_mlp_config_list_from_dataframe(df, input_dim=8, output_dim=1)
        assert isinstance(configs, list)

    def test_list_length_matches_dataframe(self):
        df = self._sample_df(n=12)
        configs = generate_mlp_config_list_from_dataframe(df, input_dim=8, output_dim=1)
        assert len(configs) == 12

    def test_all_items_are_simple_mlp_config(self):
        df = self._sample_df()
        configs = generate_mlp_config_list_from_dataframe(df, input_dim=8, output_dim=1)
        assert all(isinstance(c, SimpleMLPConfig) for c in configs)

    def test_input_and_output_dim_propagated(self):
        df = self._sample_df()
        configs = generate_mlp_config_list_from_dataframe(df, input_dim=16, output_dim=3)
        for c in configs:
            assert c.input_dim == 16
            assert c.output_dim == 3

    def test_neurons_per_layer_matches_dataframe(self):
        df = self._sample_df()
        configs = generate_mlp_config_list_from_dataframe(df, input_dim=8, output_dim=1)
        for config, (_, row) in zip(configs, df.iterrows()):
            assert config.neurons_per_layer == row["neurons_per_layer"]

    def test_activation_matches_dataframe(self):
        df = self._sample_df()
        configs = generate_mlp_config_list_from_dataframe(df, input_dim=8, output_dim=1)
        for config, (_, row) in zip(configs, df.iterrows()):
            assert config.activation == row["activation"]

    def test_use_batch_norm_matches_dataframe(self):
        df = self._sample_df(test_batch_norm=True)
        configs = generate_mlp_config_list_from_dataframe(df, input_dim=8, output_dim=1)
        for config, (_, row) in zip(configs, df.iterrows()):
            assert config.use_batch_norm == row["use_batch_norm"]

    def test_defaults_batch_norm_false_when_column_absent(self):
        df = self._sample_df(test_batch_norm=False)
        configs = generate_mlp_config_list_from_dataframe(df, input_dim=8, output_dim=1)
        assert all(c.use_batch_norm is False for c in configs)
