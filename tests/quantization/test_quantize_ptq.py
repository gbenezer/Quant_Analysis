from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.quant_analysis.model_architecture.model_configs import SimpleMLPConfig
from src.quant_analysis.model_architecture.simple_mlp import SimpleMLP
from src.quant_analysis.quantization.ptq.quantize_ptq import (
    fuse_mlp_bn,
    quantize_ptq,
    supports_step,
)


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


def make_model(**overrides) -> SimpleMLP:
    return SimpleMLP(make_config(**overrides))


def make_dataloader(input_dim=8, n_samples=16, batch_size=4) -> DataLoader:
    x = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


# ---------------------------------------------------------------------------
# TestSupportsStep
# ---------------------------------------------------------------------------

class TestSupportsStep:
    def test_returns_true_when_step_in_signature(self):
        class ConfigWithStep:
            def __init__(self, step: str = "prepare", **kwargs):
                pass

        assert supports_step(ConfigWithStep) is True

    def test_returns_false_when_step_not_in_signature(self):
        class ConfigWithoutStep:
            def __init__(self, bits: int = 8, **kwargs):
                pass

        assert supports_step(ConfigWithoutStep) is False

    def test_returns_false_for_no_parameters(self):
        class EmptyConfig:
            def __init__(self):
                pass

        assert supports_step(EmptyConfig) is False

    def test_returns_true_for_step_among_multiple_parameters(self):
        class MultiParamConfig:
            def __init__(self, mode: str = "default", step: str = "prepare", bits: int = 8):
                pass

        assert supports_step(MultiParamConfig) is True


# ---------------------------------------------------------------------------
# TestFuseMlpBn
# ---------------------------------------------------------------------------

class TestFuseMlpBn:
    def test_returns_new_model_instance(self):
        model = make_model(use_batch_norm=True)
        assert fuse_mlp_bn(model) is not model

    def test_does_not_modify_original_model(self):
        model = make_model(use_batch_norm=True)
        bn_count_before = sum(1 for m in model.linear_stack if isinstance(m, nn.BatchNorm1d))
        fuse_mlp_bn(model)
        bn_count_after = sum(1 for m in model.linear_stack if isinstance(m, nn.BatchNorm1d))
        assert bn_count_before == bn_count_after

    def test_no_batch_norm_layers_after_fusion(self):
        model = make_model(use_batch_norm=True)
        fused = fuse_mlp_bn(model)
        bn_layers = [m for m in fused.linear_stack if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == 0

    def test_bn_layers_replaced_with_identity(self):
        model = make_model(use_batch_norm=True)
        fused = fuse_mlp_bn(model)
        identity_layers = [m for m in fused.linear_stack if isinstance(m, nn.Identity)]
        assert len(identity_layers) > 0

    def test_fused_model_is_in_eval_mode(self):
        model = make_model(use_batch_norm=True)
        fused = fuse_mlp_bn(model)
        assert not fused.training

    def test_returns_simple_mlp_instance(self):
        model = make_model(use_batch_norm=True)
        assert isinstance(fuse_mlp_bn(model), SimpleMLP)

    def test_output_shape_preserved_after_fusion(self):
        model = make_model(use_batch_norm=True)
        model.eval()
        fused = fuse_mlp_bn(model)
        x = torch.randn(4, 8)
        with torch.no_grad():
            assert fused(x).shape == model(x).shape

    def test_no_op_for_model_without_batch_norm(self):
        model = make_model(use_batch_norm=False)
        fused = fuse_mlp_bn(model)
        assert isinstance(fused, SimpleMLP)
        assert sum(1 for m in fused.linear_stack if isinstance(m, nn.BatchNorm1d)) == 0

    def test_linear_layer_count_preserved(self):
        model = make_model(use_batch_norm=True, neurons_per_layer=[16, 8])
        fused = fuse_mlp_bn(model)
        original_linears = [m for m in model.linear_stack if isinstance(m, nn.Linear)]
        fused_linears = [m for m in fused.linear_stack if isinstance(m, nn.Linear)]
        assert len(fused_linears) == len(original_linears)


# ---------------------------------------------------------------------------
# TestQuantizePtq
# ---------------------------------------------------------------------------

_QUANTIZE_ = "src.quant_analysis.quantization.ptq.quantize_ptq.quantize_"
_SUPPORTS_STEP = "src.quant_analysis.quantization.ptq.quantize_ptq.supports_step"
_FUSE_MLP_BN = "src.quant_analysis.quantization.ptq.quantize_ptq.fuse_mlp_bn"


def _mock_config(name="MockConfig") -> MagicMock:
    cfg = MagicMock()
    cfg.__name__ = name
    return cfg


class TestQuantizePtq:
    def test_returns_nn_module_on_success(self):
        model = make_model()
        with patch(_QUANTIZE_):
            result = quantize_ptq(model, _mock_config(), is_static=False)
        assert isinstance(result, nn.Module)

    def test_does_not_modify_base_model(self):
        model = make_model()
        params_before = {k: v.clone() for k, v in model.named_parameters()}
        with patch(_QUANTIZE_):
            quantize_ptq(model, _mock_config(), is_static=False)
        for k, v in model.named_parameters():
            assert torch.allclose(params_before[k], v), f"Parameter {k} was modified"

    def test_returned_model_is_in_eval_mode(self):
        model = make_model()
        with patch(_QUANTIZE_):
            result = quantize_ptq(model, _mock_config(), is_static=False)
        assert not result.training

    def test_model_placed_on_cpu(self):
        model = make_model()
        with patch(_QUANTIZE_):
            result = quantize_ptq(model, _mock_config(), quantize_device="cpu")
        for param in result.parameters():
            assert param.device.type == "cpu"

    def test_returns_none_on_assertion_error(self):
        model = make_model()
        with patch(_QUANTIZE_, side_effect=AssertionError("quantization failed")):
            assert quantize_ptq(model, _mock_config(), is_static=False) is None

    def test_returns_none_on_runtime_error(self):
        model = make_model()
        with patch(_QUANTIZE_, side_effect=RuntimeError("runtime failure")):
            assert quantize_ptq(model, _mock_config(), is_static=False) is None

    def test_prints_config_name_and_error_on_failure(self, capsys):
        model = make_model()
        with patch(_QUANTIZE_, side_effect=AssertionError("specific error")):
            quantize_ptq(model, _mock_config(name="FailingConfig"), is_static=False)
        out = capsys.readouterr().out
        assert "FailingConfig" in out
        assert "specific error" in out

    def test_fuses_bn_for_simple_mlp(self):
        model = make_model(use_batch_norm=True)
        with patch(_QUANTIZE_), patch(_FUSE_MLP_BN, wraps=fuse_mlp_bn) as mock_fuse:
            quantize_ptq(model, _mock_config(), is_static=False)
        mock_fuse.assert_called_once()

    def test_skips_bn_fusion_for_generic_module(self):
        plain = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        with patch(_QUANTIZE_), patch(_FUSE_MLP_BN) as mock_fuse:
            quantize_ptq(plain, _mock_config(), is_static=False)
        mock_fuse.assert_not_called()

    def test_dynamic_calls_quantize_once(self):
        model = make_model()
        with patch(_QUANTIZE_) as mock_q:
            quantize_ptq(model, _mock_config(), is_static=False)
        mock_q.assert_called_once()

    def test_kwargs_forwarded_to_config(self):
        model = make_model()
        received = {}

        def capturing_config(**kwargs):
            received.update(kwargs)
            return MagicMock()

        capturing_config.__name__ = "CapturingConfig"
        with patch(_QUANTIZE_):
            quantize_ptq(model, capturing_config, is_static=False, bits=8, mode="symmetric")

        assert received.get("bits") == 8
        assert received.get("mode") == "symmetric"

    def test_static_with_step_calls_quantize_twice(self):
        model = make_model()
        data = make_dataloader()
        with patch(_SUPPORTS_STEP, return_value=True), patch(_QUANTIZE_) as mock_q:
            quantize_ptq(model, _mock_config(), is_static=True, data=data)
        assert mock_q.call_count == 2

    def test_static_without_step_calls_quantize_once(self):
        model = make_model()
        data = make_dataloader()
        with patch(_SUPPORTS_STEP, return_value=False), patch(_QUANTIZE_) as mock_q:
            quantize_ptq(model, _mock_config(), is_static=True, data=data)
        mock_q.assert_called_once()

    def test_static_with_step_calls_prepare_then_convert(self):
        model = make_model()
        data = make_dataloader()
        config = MagicMock()
        config.__name__ = "StepConfig"

        with patch(_SUPPORTS_STEP, return_value=True), patch(_QUANTIZE_):
            quantize_ptq(model, config, is_static=True, data=data)

        call_kwargs = [c.kwargs for c in config.call_args_list]
        steps = [kw.get("step") for kw in call_kwargs]
        assert "prepare" in steps
        assert "convert" in steps
        assert steps.index("prepare") < steps.index("convert")

    def test_static_kwargs_forwarded_to_convert_step(self):
        model = make_model()
        data = make_dataloader()
        config = MagicMock()
        config.__name__ = "StepConfig"

        with patch(_SUPPORTS_STEP, return_value=True), patch(_QUANTIZE_):
            quantize_ptq(model, config, is_static=True, data=data, bits=4)

        convert_call = next(
            c for c in config.call_args_list if c.kwargs.get("step") == "convert"
        )
        assert convert_call.kwargs.get("bits") == 4

    def test_static_with_none_data_does_not_raise(self):
        model = make_model()

        def config_factory(step=None, **kwargs):
            return MagicMock()

        config_factory.__name__ = "StepConfig"
        with patch(_SUPPORTS_STEP, return_value=True), patch(_QUANTIZE_):
            result = quantize_ptq(model, config_factory, is_static=True, data=None)
        assert result is not None

    def test_static_tuple_batch_uses_first_element(self):
        """DataLoader yielding (x, y) tuples should pass only x to the model."""
        model = make_model()
        data = make_dataloader()  # yields (x, y) tuples
        forward_inputs = []
        original_forward = model.__class__.forward

        def capturing_forward(self, x):
            forward_inputs.append(x)
            return original_forward(self, x)

        def config_factory(step=None, **kwargs):
            return MagicMock()

        config_factory.__name__ = "StepConfig"

        with patch(_SUPPORTS_STEP, return_value=True), \
             patch(_QUANTIZE_), \
             patch.object(SimpleMLP, "forward", capturing_forward):
            quantize_ptq(model, config_factory, is_static=True, data=data)

        # Each captured input should have shape (batch_size, input_dim), not include labels
        for inp in forward_inputs:
            assert inp.shape[-1] == 8
