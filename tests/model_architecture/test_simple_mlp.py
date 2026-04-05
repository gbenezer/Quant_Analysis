import pytest
import torch
import torch.nn as nn

from src.quant_analysis.model_architecture.model_configs import SimpleMLPConfig
from src.quant_analysis.model_architecture.simple_mlp import SimpleMLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides) -> SimpleMLPConfig:
    defaults = dict(
        input_dim=8,
        output_dim=1,
        neurons_per_layer=[16, 8],
        activation="relu",
        use_batch_norm=False,
    )
    defaults.update(overrides)
    return SimpleMLPConfig(**defaults)


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_minimal_single_hidden_layer(self):
        config = make_config(neurons_per_layer=[4])
        model = SimpleMLP(config)
        assert isinstance(model, nn.Module)

    def test_multiple_hidden_layers(self):
        config = make_config(neurons_per_layer=[32, 16, 8])
        model = SimpleMLP(config)
        assert isinstance(model, nn.Module)

    def test_raises_on_empty_layer_list(self):
        config = make_config(neurons_per_layer=[])
        with pytest.raises(ValueError, match="positive number of layers"):
            SimpleMLP(config)

    def test_raises_on_zero_width_layer(self):
        config = make_config(neurons_per_layer=[16, 0, 8])
        with pytest.raises(ValueError, match="positive width"):
            SimpleMLP(config)

    def test_raises_on_negative_width_layer(self):
        config = make_config(neurons_per_layer=[-1])
        with pytest.raises(ValueError, match="positive width"):
            SimpleMLP(config)

    def test_raises_on_unsupported_activation(self):
        config = make_config(activation="sigmoid")
        with pytest.raises(ValueError, match="not supported"):
            SimpleMLP(config)


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class TestActivations:
    @pytest.mark.parametrize("act", ["relu", "leaky_relu", "elu", "gelu", "celu"])
    def test_all_supported_activations_construct(self, act):
        config = make_config(activation=act)
        model = SimpleMLP(config)
        assert isinstance(model, nn.Module)

    @pytest.mark.parametrize("act, cls", [
        ("relu", nn.ReLU),
        ("leaky_relu", nn.LeakyReLU),
        ("elu", nn.ELU),
        ("gelu", nn.GELU),
        ("celu", nn.CELU),
    ])
    def test_correct_activation_class_stored(self, act, cls):
        config = make_config(activation=act)
        model = SimpleMLP(config)
        assert model.activation_cls is cls


# ---------------------------------------------------------------------------
# Batch normalisation
# ---------------------------------------------------------------------------

class TestBatchNorm:
    def test_batch_norm_layers_present_when_enabled(self):
        config = make_config(neurons_per_layer=[16, 8], use_batch_norm=True)
        model = SimpleMLP(config)
        bn_layers = [m for m in model.linear_stack if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) > 0

    def test_no_batch_norm_layers_when_disabled(self):
        config = make_config(neurons_per_layer=[16, 8], use_batch_norm=False)
        model = SimpleMLP(config)
        bn_layers = [m for m in model.linear_stack if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == 0

    def test_batch_norm_count_matches_layer_count(self):
        # one BN per hidden layer boundary plus the final one = len(neurons_per_layer)
        neurons = [32, 16, 8]
        config = make_config(neurons_per_layer=neurons, use_batch_norm=True)
        model = SimpleMLP(config)
        bn_layers = [m for m in model.linear_stack if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == len(neurons)


# ---------------------------------------------------------------------------
# Layer topology
# ---------------------------------------------------------------------------

class TestLayerTopology:
    def test_linear_layer_count(self):
        # total linear layers = len(neurons_per_layer) + 1  (hidden layers + output)
        neurons = [16, 8, 4]
        config = make_config(neurons_per_layer=neurons)
        model = SimpleMLP(config)
        linear_layers = [m for m in model.linear_stack if isinstance(m, nn.Linear)]
        assert len(linear_layers) == len(neurons) + 1

    def test_first_linear_input_dim(self):
        config = make_config(input_dim=10, neurons_per_layer=[20, 10])
        model = SimpleMLP(config)
        first_linear = next(m for m in model.linear_stack if isinstance(m, nn.Linear))
        assert first_linear.in_features == 10

    def test_last_linear_output_dim(self):
        config = make_config(output_dim=3, neurons_per_layer=[16, 8])
        model = SimpleMLP(config)
        linear_layers = [m for m in model.linear_stack if isinstance(m, nn.Linear)]
        assert linear_layers[-1].out_features == 3

    def test_hidden_layer_widths(self):
        neurons = [32, 16, 8]
        config = make_config(neurons_per_layer=neurons)
        model = SimpleMLP(config)
        linear_layers = [m for m in model.linear_stack if isinstance(m, nn.Linear)]
        # layers[0] input->neurons[0], layers[1] neurons[0]->neurons[1], etc.
        for i, width in enumerate(neurons):
            assert linear_layers[i].out_features == width


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape_single_output_batched(self):
        config = make_config(input_dim=8, output_dim=1, neurons_per_layer=[16, 8])
        model = SimpleMLP(config)
        model.eval()
        x = torch.randn(32, 8)
        out = model(x)
        # squeeze removes the trailing dim-1, giving shape (batch,)
        assert out.shape == (32,)

    def test_output_shape_multi_output_batched(self):
        config = make_config(input_dim=8, output_dim=4, neurons_per_layer=[16, 8])
        model = SimpleMLP(config)
        model.eval()
        x = torch.randn(16, 8)
        out = model(x)
        assert out.shape == (16, 4)

    def test_output_shape_single_sample(self):
        config = make_config(input_dim=8, output_dim=1, neurons_per_layer=[16, 8])
        model = SimpleMLP(config)
        model.eval()
        x = torch.randn(1, 8)
        out = model(x)
        # squeeze on (1,1) -> scalar tensor
        assert out.ndim == 0

    def test_output_is_float_tensor(self):
        config = make_config(input_dim=4, output_dim=1, neurons_per_layer=[8])
        model = SimpleMLP(config)
        model.eval()
        x = torch.randn(5, 4)
        out = model(x)
        assert out.dtype == torch.float32

    def test_forward_with_batch_norm_train_mode(self):
        config = make_config(input_dim=8, output_dim=1, neurons_per_layer=[16, 8], use_batch_norm=True)
        model = SimpleMLP(config)
        model.train()
        # batch norm requires batch size > 1
        x = torch.randn(4, 8)
        out = model(x)
        assert out.shape == (4,)

    def test_gradients_flow(self):
        config = make_config(input_dim=8, output_dim=1, neurons_per_layer=[16, 8])
        model = SimpleMLP(config)
        x = torch.randn(4, 8)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_deterministic_output_in_eval_mode(self):
        config = make_config(input_dim=8, output_dim=1, neurons_per_layer=[16, 8])
        model = SimpleMLP(config)
        model.eval()
        x = torch.randn(4, 8)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Config storage
# ---------------------------------------------------------------------------

class TestConfigStorage:
    def test_config_is_stored(self):
        config = make_config()
        model = SimpleMLP(config)
        assert model.config is config

    def test_config_fields_accessible(self):
        config = make_config(input_dim=5, output_dim=2, neurons_per_layer=[10])
        model = SimpleMLP(config)
        assert model.config.input_dim == 5
        assert model.config.output_dim == 2
        assert model.config.neurons_per_layer == [10]
