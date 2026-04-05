from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.quant_analysis.model_architecture.model_configs import SimpleMLPConfig
from src.quant_analysis.model_architecture.simple_mlp import SimpleMLP
from src.quant_analysis.quantization.ptq.ptq_config_metadata import ConfigAndMetadataPTQ
from src.quant_analysis.quantization.ptq.run_ptq import build_quantized_models, run_ptq

# ---------------------------------------------------------------------------
# CNN / RNN helpers
# ---------------------------------------------------------------------------

_SEQ_LEN = 8


class TinyCNNRegressor(nn.Module):
    """Conv1d + Linear regression model. Accepts (batch, _SEQ_LEN) input."""

    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(8 * _SEQ_LEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)       # (B, seq_len) → (B, 1, seq_len)
        x = self.conv_block(x)   # (B, 8, seq_len)
        return self.head(x.flatten(1)).squeeze(1)


class TinyGRURegressor(nn.Module):
    """GRU + Linear regression model. Accepts (batch, _SEQ_LEN) input."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)       # (B, seq_len) → (B, seq_len, 1)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(1)


class TinyLSTMRegressor(nn.Module):
    """LSTM + Linear regression model. Accepts (batch, _SEQ_LEN) input."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


class TinyTransformerRegressor(nn.Module):
    """Small encoder-only transformer regression model.

    Accepts a flat (batch, N_TOKENS * IN_FEATURES) input. IN_FEATURES=16 ensures
    all nn.Linear layers meet TorchAO's minimum in_features requirement.
    All evaluators in run_ptq are mocked in these tests so the model is never
    actually called with the shared _SEQ_LEN=8 dataloader data.
    """

    N_TOKENS = 2
    IN_FEATURES = 16
    D_MODEL = 16
    NHEAD = 2

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(self.IN_FEATURES, self.D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL,
            nhead=self.NHEAD,
            dim_feedforward=32,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(self.D_MODEL, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.N_TOKENS, self.IN_FEATURES)
        x = self.input_proj(x)      # (B, N_TOKENS, D_MODEL)
        x = self.encoder(x)         # (B, N_TOKENS, D_MODEL)
        x = x.mean(dim=1)           # global average pooling: (B, D_MODEL)
        return self.head(x).squeeze(1)  # (B,)


def make_seq_dataloader(n_samples: int = 32, batch_size: int = 8) -> DataLoader:
    """Dataloader yielding (batch, _SEQ_LEN) tensors, compatible with CNN, RNN, and Transformer architectures."""
    x = torch.randn(n_samples, _SEQ_LEN)
    y = torch.randn(n_samples)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MOD = "src.quant_analysis.quantization.ptq.run_ptq"
_LATENCY_TUPLE = (1000.0, 0.010, 0.020, 0.030)
_RELATIVE_TUPLE = (0.5, 0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mlp_config(**overrides) -> SimpleMLPConfig:
    defaults = dict(
        input_dim=4,
        output_dim=1,
        neurons_per_layer=[8, 4],
        activation="relu",
        use_batch_norm=False,
    )
    defaults.update(overrides)
    return SimpleMLPConfig(**defaults)


def make_mlp(**overrides) -> SimpleMLP:
    return SimpleMLP(make_mlp_config(**overrides))


def make_dataloader(input_dim=4, n_samples=32, batch_size=8) -> DataLoader:
    x = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def make_fake_metadata(bits=8, dynamic=False, weight_only=False) -> ConfigAndMetadataPTQ:
    return ConfigAndMetadataPTQ(
        ao_config=MagicMock(),
        precision="int8",
        bits_per_weight=bits,
        dynamic_calibration=dynamic,
        weight_only=weight_only,
        cuda_compute_capacity_compatibility=8.6,
        ao_config_kwargs={},
    )


def make_fake_configs(names=("cfg_a",), bits=8) -> dict:
    return {name: make_fake_metadata(bits=bits) for name in names}


class _MinimalPatches:
    """
    Context manager that patches all heavy operations in run_ptq,
    allowing tests to focus on a single behavior.
    """

    def __init__(self, model_dict: dict, weight_only: bool = False):
        self._model_dict = model_dict
        self._weight_only = weight_only
        self._stack = ExitStack()

    def __enter__(self):
        self._stack.__enter__()
        ec = self._stack.enter_context
        ec(patch(f"{_MOD}.build_quantized_models", return_value=self._model_dict))
        ec(patch(f"{_MOD}.evaluate_mae", return_value=1.0))
        ec(patch(f"{_MOD}.evaluate_pytorch_latency_and_estimate_size", return_value=_LATENCY_TUPLE))
        ec(patch(f"{_MOD}.assess_relative_performance", return_value=_RELATIVE_TUPLE))
        if self._weight_only:
            ec(patch(f"{_MOD}.evaluate_onnx_latency_and_size", return_value=_LATENCY_TUPLE))
            ec(patch(f"{_MOD}.evaluate_pt2_latency_and_size", return_value=_LATENCY_TUPLE))
        return self

    def __exit__(self, *args):
        return self._stack.__exit__(*args)


def _one_config_dict() -> dict:
    return {"cfg1": (nn.Linear(4, 1), make_fake_metadata())}


# ---------------------------------------------------------------------------
# TestBuildQuantizedModels
# ---------------------------------------------------------------------------

_QUANTIZE_PTQ = f"{_MOD}.quantize_ptq"


class TestBuildQuantizedModels:
    def test_returns_dict(self):
        with patch(_QUANTIZE_PTQ, return_value=nn.Linear(4, 1)):
            result = build_quantized_models(make_mlp(), make_fake_configs(), make_dataloader())
        assert isinstance(result, dict)

    def test_empty_configs_returns_empty_dict(self):
        result = build_quantized_models(make_mlp(), {}, make_dataloader())
        assert result == {}

    def test_successful_config_included(self):
        with patch(_QUANTIZE_PTQ, return_value=nn.Linear(4, 1)):
            result = build_quantized_models(make_mlp(), make_fake_configs(["cfg_a"]), make_dataloader())
        assert "cfg_a" in result

    def test_result_value_is_model_metadata_tuple(self):
        quant_model = nn.Linear(4, 1)
        configs = make_fake_configs(["cfg_a"])
        with patch(_QUANTIZE_PTQ, return_value=quant_model):
            result = build_quantized_models(make_mlp(), configs, make_dataloader())
        model_out, meta_out = result["cfg_a"]
        assert isinstance(model_out, nn.Module)
        assert meta_out is configs["cfg_a"]

    def test_config_returning_none_excluded(self):
        with patch(_QUANTIZE_PTQ, return_value=None):
            result = build_quantized_models(make_mlp(), make_fake_configs(["cfg_a"]), make_dataloader())
        assert "cfg_a" not in result

    def test_config_raising_exception_excluded(self):
        with patch(_QUANTIZE_PTQ, side_effect=RuntimeError("boom")):
            result = build_quantized_models(make_mlp(), make_fake_configs(["cfg_a"]), make_dataloader())
        assert "cfg_a" not in result

    def test_exception_during_config_prints_name(self, capsys):
        with patch(_QUANTIZE_PTQ, side_effect=RuntimeError("err")):
            build_quantized_models(make_mlp(), make_fake_configs(["my_cfg"]), make_dataloader())
        assert "my_cfg" in capsys.readouterr().out

    def test_multiple_configs_all_successful(self):
        with patch(_QUANTIZE_PTQ, return_value=nn.Linear(4, 1)):
            result = build_quantized_models(
                make_mlp(), make_fake_configs(["a", "b", "c"]), make_dataloader()
            )
        assert set(result.keys()) == {"a", "b", "c"}

    def test_partial_failure_only_failed_excluded(self):
        configs = make_fake_configs(["good", "bad"])
        good_model = nn.Linear(4, 1)

        def side_effect(base_model, ao_config, **kwargs):
            return good_model if ao_config is configs["good"]["ao_config"] else None

        with patch(_QUANTIZE_PTQ, side_effect=side_effect):
            result = build_quantized_models(make_mlp(), configs, make_dataloader())

        assert "good" in result
        assert "bad" not in result


# ---------------------------------------------------------------------------
# TestRunPtqReturnType
# ---------------------------------------------------------------------------


class TestRunPtqReturnType:
    def test_returns_dict(self):
        model, loader = make_mlp(), make_dataloader()
        with _MinimalPatches(_one_config_dict()):
            result = run_ptq(model, loader, runs=2, warmup=1)
        assert isinstance(result, dict)

    def test_empty_model_dict_returns_empty_dict(self):
        model, loader = make_mlp(), make_dataloader()
        with _MinimalPatches({}):
            result = run_ptq(model, loader, runs=2, warmup=1)
        assert result == {}

    def test_result_keys_match_config_names(self):
        model, loader = make_mlp(), make_dataloader()
        with _MinimalPatches(_one_config_dict()):
            result = run_ptq(model, loader, runs=2, warmup=1)
        assert "cfg1" in result


# ---------------------------------------------------------------------------
# TestRunPtqOutputKeys
# ---------------------------------------------------------------------------


class TestRunPtqOutputKeys:
    @pytest.fixture(scope="class")
    def result(self):
        model, loader = make_mlp(), make_dataloader()
        with _MinimalPatches(_one_config_dict()):
            return run_ptq(model, loader, runs=2, warmup=1)

    def test_pytorch_result_present(self, result):
        assert "pytorch_result" in result["cfg1"]

    def test_config_field_present(self, result):
        assert "config" in result["cfg1"]

    def test_pytorch_result_has_mae_keys(self, result):
        pytorch = result["cfg1"]["pytorch_result"]
        assert "quantized_MAE" in pytorch
        assert "relative_MAE" in pytorch

    def test_pytorch_result_has_absolute_latency_keys(self, result):
        pytorch = result["cfg1"]["pytorch_result"]
        for key in ("quantized_model_size", "quantized_median_latency",
                    "quantized_p95_latency", "quantized_p99_latency"):
            assert key in pytorch

    def test_pytorch_result_has_relative_latency_keys(self, result):
        pytorch = result["cfg1"]["pytorch_result"]
        for key in ("relative_model_size", "relative_median_latency",
                    "relative_p95_latency", "relative_p99_latency"):
            assert key in pytorch

    def test_weight_only_false_no_onnx_result(self):
        model, loader = make_mlp(), make_dataloader()
        with _MinimalPatches(_one_config_dict(), weight_only=False):
            result = run_ptq(model, loader, weight_only=False, runs=2, warmup=1)
        assert "onnx_result" not in result["cfg1"]

    def test_weight_only_false_no_pt2_result(self):
        model, loader = make_mlp(), make_dataloader()
        with _MinimalPatches(_one_config_dict(), weight_only=False):
            result = run_ptq(model, loader, weight_only=False, runs=2, warmup=1)
        assert "pt2_result" not in result["cfg1"]

    def test_weight_only_true_has_onnx_result(self):
        model, loader = make_mlp(), make_dataloader()
        with _MinimalPatches(_one_config_dict(), weight_only=True):
            result = run_ptq(model, loader, weight_only=True, runs=2, warmup=1)
        assert "onnx_result" in result["cfg1"]

    def test_weight_only_true_has_pt2_result(self):
        model, loader = make_mlp(), make_dataloader()
        with _MinimalPatches(_one_config_dict(), weight_only=True):
            result = run_ptq(model, loader, weight_only=True, runs=2, warmup=1)
        assert "pt2_result" in result["cfg1"]


# ---------------------------------------------------------------------------
# TestRunPtqConfigSelection
# ---------------------------------------------------------------------------


class TestRunPtqConfigSelection:
    def test_weight_only_false_uses_full_config_metadata(self):
        model, loader = make_mlp(), make_dataloader()
        sentinel = {"cfg_x": make_fake_metadata()}

        with patch(f"{_MOD}.PTQ_QUANT_CONFIG_METADATA", sentinel), \
             patch(f"{_MOD}.build_quantized_models", return_value={}) as mock_build, \
             patch(f"{_MOD}.evaluate_mae", return_value=1.0), \
             patch(f"{_MOD}.evaluate_pytorch_latency_and_estimate_size", return_value=_LATENCY_TUPLE):
            run_ptq(model, loader, weight_only=False, runs=2, warmup=1)

        assert mock_build.call_args.kwargs["configs"] is sentinel

    def test_weight_only_true_uses_weight_only_config_metadata(self):
        model, loader = make_mlp(), make_dataloader()
        sentinel = {"cfg_w": make_fake_metadata()}

        with patch(f"{_MOD}.PTQ_WEIGHT_ONLY_CONFIG_METADATA", sentinel), \
             patch(f"{_MOD}.build_quantized_models", return_value={}) as mock_build, \
             patch(f"{_MOD}.evaluate_mae", return_value=1.0), \
             patch(f"{_MOD}.evaluate_pytorch_latency_and_estimate_size", return_value=_LATENCY_TUPLE), \
             patch(f"{_MOD}.evaluate_onnx_latency_and_size", return_value=_LATENCY_TUPLE), \
             patch(f"{_MOD}.evaluate_pt2_latency_and_size", return_value=_LATENCY_TUPLE):
            run_ptq(model, loader, weight_only=True, runs=2, warmup=1)

        assert mock_build.call_args.kwargs["configs"] is sentinel


# ---------------------------------------------------------------------------
# TestRunPtqBaselineEvaluation
# ---------------------------------------------------------------------------


class TestRunPtqBaselineEvaluation:
    def test_baseline_mae_failure_raises_runtime_error(self):
        model, loader = make_mlp(), make_dataloader()
        with patch(f"{_MOD}.build_quantized_models", return_value={}), \
             patch(f"{_MOD}.evaluate_mae", side_effect=RuntimeError("MAE boom")):
            with pytest.raises(RuntimeError):
                run_ptq(model, loader, runs=2, warmup=1)

    def test_weight_only_false_does_not_call_onnx_evaluator(self):
        model, loader = make_mlp(), make_dataloader()
        with patch(f"{_MOD}.build_quantized_models", return_value={}), \
             patch(f"{_MOD}.evaluate_mae", return_value=1.0), \
             patch(f"{_MOD}.evaluate_pytorch_latency_and_estimate_size", return_value=_LATENCY_TUPLE), \
             patch(f"{_MOD}.evaluate_onnx_latency_and_size") as mock_onnx, \
             patch(f"{_MOD}.evaluate_pt2_latency_and_size") as mock_pt2:
            run_ptq(model, loader, weight_only=False, runs=2, warmup=1)

        mock_onnx.assert_not_called()
        mock_pt2.assert_not_called()

    def test_weight_only_true_evaluates_onnx_and_pt2_for_baseline(self):
        model, loader = make_mlp(), make_dataloader()
        with patch(f"{_MOD}.build_quantized_models", return_value={}), \
             patch(f"{_MOD}.evaluate_mae", return_value=1.0), \
             patch(f"{_MOD}.evaluate_pytorch_latency_and_estimate_size", return_value=_LATENCY_TUPLE), \
             patch(f"{_MOD}.evaluate_onnx_latency_and_size", return_value=_LATENCY_TUPLE) as mock_onnx, \
             patch(f"{_MOD}.evaluate_pt2_latency_and_size", return_value=_LATENCY_TUPLE) as mock_pt2:
            run_ptq(model, loader, weight_only=True, runs=2, warmup=1)

        mock_onnx.assert_called()
        mock_pt2.assert_called()


# ---------------------------------------------------------------------------
# TestRunPtqQuantizedEvaluation
# ---------------------------------------------------------------------------


class TestRunPtqQuantizedEvaluation:
    def test_relative_mae_computed_as_ratio(self):
        model, loader = make_mlp(), make_dataloader()
        # baseline MAE = 2.0, quantized MAE = 1.0 → relative = 0.5
        mae_vals = iter([2.0, 1.0])

        with patch(f"{_MOD}.build_quantized_models", return_value=_one_config_dict()), \
             patch(f"{_MOD}.evaluate_mae", side_effect=lambda *a, **kw: next(mae_vals)), \
             patch(f"{_MOD}.evaluate_pytorch_latency_and_estimate_size", return_value=_LATENCY_TUPLE), \
             patch(f"{_MOD}.assess_relative_performance", return_value=_RELATIVE_TUPLE):
            result = run_ptq(model, loader, runs=2, warmup=1)

        assert result["cfg1"]["pytorch_result"]["relative_MAE"] == pytest.approx(0.5)

    def test_config_with_quantized_mae_failure_is_skipped(self):
        model, loader = make_mlp(), make_dataloader()
        call_count = {"n": 0}

        def mae_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise RuntimeError("quantized MAE failed")
            return 1.0

        with patch(f"{_MOD}.build_quantized_models", return_value=_one_config_dict()), \
             patch(f"{_MOD}.evaluate_mae", side_effect=mae_side_effect), \
             patch(f"{_MOD}.evaluate_pytorch_latency_and_estimate_size", return_value=_LATENCY_TUPLE), \
             patch(f"{_MOD}.assess_relative_performance", return_value=_RELATIVE_TUPLE):
            result = run_ptq(model, loader, runs=2, warmup=1)

        assert "cfg1" not in result

    def test_absolute_pytorch_metrics_stored_correctly(self):
        model, loader = make_mlp(), make_dataloader()
        latency = (999.0, 0.011, 0.021, 0.031)

        with patch(f"{_MOD}.build_quantized_models", return_value=_one_config_dict()), \
             patch(f"{_MOD}.evaluate_mae", return_value=1.0), \
             patch(f"{_MOD}.evaluate_pytorch_latency_and_estimate_size", return_value=latency), \
             patch(f"{_MOD}.assess_relative_performance", return_value=_RELATIVE_TUPLE):
            result = run_ptq(model, loader, runs=2, warmup=1)

        pytorch = result["cfg1"]["pytorch_result"]
        assert pytorch["quantized_model_size"] == 999.0
        assert pytorch["quantized_median_latency"] == pytest.approx(0.011)
        assert pytorch["quantized_p95_latency"] == pytest.approx(0.021)
        assert pytorch["quantized_p99_latency"] == pytest.approx(0.031)

    def test_relative_pytorch_metrics_come_from_assess_relative_performance(self):
        model, loader = make_mlp(), make_dataloader()
        rel = (0.25, 0.30, 0.35, 0.40)

        with patch(f"{_MOD}.build_quantized_models", return_value=_one_config_dict()), \
             patch(f"{_MOD}.evaluate_mae", return_value=1.0), \
             patch(f"{_MOD}.evaluate_pytorch_latency_and_estimate_size", return_value=_LATENCY_TUPLE), \
             patch(f"{_MOD}.assess_relative_performance", return_value=rel):
            result = run_ptq(model, loader, runs=2, warmup=1)

        pytorch = result["cfg1"]["pytorch_result"]
        assert pytorch["relative_model_size"] == pytest.approx(0.25)
        assert pytorch["relative_median_latency"] == pytest.approx(0.30)
        assert pytorch["relative_p95_latency"] == pytest.approx(0.35)
        assert pytorch["relative_p99_latency"] == pytest.approx(0.40)

    def test_multiple_configs_all_in_output(self):
        model, loader = make_mlp(), make_dataloader()
        model_dict = {
            "cfg_a": (nn.Linear(4, 1), make_fake_metadata()),
            "cfg_b": (nn.Linear(4, 1), make_fake_metadata()),
        }

        with patch(f"{_MOD}.build_quantized_models", return_value=model_dict), \
             patch(f"{_MOD}.evaluate_mae", return_value=1.0), \
             patch(f"{_MOD}.evaluate_pytorch_latency_and_estimate_size", return_value=_LATENCY_TUPLE), \
             patch(f"{_MOD}.assess_relative_performance", return_value=_RELATIVE_TUPLE):
            result = run_ptq(model, loader, runs=2, warmup=1)

        assert "cfg_a" in result
        assert "cfg_b" in result

    def test_config_stored_in_result(self):
        model, loader = make_mlp(), make_dataloader()
        meta = make_fake_metadata(bits=16)
        model_dict = {"cfg1": (nn.Linear(4, 1), meta)}

        with patch(f"{_MOD}.build_quantized_models", return_value=model_dict), \
             patch(f"{_MOD}.evaluate_mae", return_value=1.0), \
             patch(f"{_MOD}.evaluate_pytorch_latency_and_estimate_size", return_value=_LATENCY_TUPLE), \
             patch(f"{_MOD}.assess_relative_performance", return_value=_RELATIVE_TUPLE):
            result = run_ptq(model, loader, runs=2, warmup=1)

        assert result["cfg1"]["config"] is meta


# ---------------------------------------------------------------------------
# TestRunPtqModelFusion
# ---------------------------------------------------------------------------


class TestRunPtqModelFusion:
    def test_simple_mlp_triggers_fuse_mlp_bn(self):
        model = make_mlp(use_batch_norm=True)
        loader = make_dataloader()
        with _MinimalPatches(_one_config_dict()), \
             patch(f"{_MOD}.fuse_mlp_bn", side_effect=lambda m: m) as mock_fuse:
            run_ptq(model, loader, runs=2, warmup=1)
        mock_fuse.assert_called_once()

    def test_generic_module_does_not_trigger_fuse_mlp_bn(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        loader = make_dataloader()
        with _MinimalPatches(_one_config_dict()), \
             patch(f"{_MOD}.fuse_mlp_bn") as mock_fuse:
            run_ptq(model, loader, runs=2, warmup=1)
        mock_fuse.assert_not_called()

    @pytest.mark.parametrize(
        "model_cls",
        [TinyCNNRegressor, TinyGRURegressor, TinyLSTMRegressor, TinyTransformerRegressor],
        ids=["cnn", "gru", "lstm", "transformer"],
    )
    def test_cnn_and_rnn_architectures_do_not_trigger_fuse_mlp_bn(self, model_cls):
        model = model_cls()
        loader = make_seq_dataloader()
        with _MinimalPatches(_one_config_dict()), \
             patch(f"{_MOD}.fuse_mlp_bn") as mock_fuse:
            run_ptq(model, loader, runs=2, warmup=1)
        mock_fuse.assert_not_called()


# ---------------------------------------------------------------------------
# TestRunPtqWithNonSimpleMLP  — CNN, GRU, LSTM compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls",
    [TinyCNNRegressor, TinyGRURegressor, TinyLSTMRegressor, TinyTransformerRegressor],
    ids=["cnn", "gru", "lstm", "transformer"],
)
class TestRunPtqWithNonSimpleMLP:
    """Verify run_ptq produces the correct output structure and honours the
    same contracts for CNN and RNN base models as it does for SimpleMLP."""

    def test_returns_dict(self, model_cls):
        model, loader = model_cls(), make_seq_dataloader()
        with _MinimalPatches(_one_config_dict()):
            result = run_ptq(model, loader, runs=2, warmup=1)
        assert isinstance(result, dict)

    def test_result_contains_config_name(self, model_cls):
        model, loader = model_cls(), make_seq_dataloader()
        with _MinimalPatches(_one_config_dict()):
            result = run_ptq(model, loader, runs=2, warmup=1)
        assert "cfg1" in result

    def test_pytorch_result_keys_present(self, model_cls):
        model, loader = model_cls(), make_seq_dataloader()
        with _MinimalPatches(_one_config_dict()):
            result = run_ptq(model, loader, runs=2, warmup=1)
        pytorch = result["cfg1"]["pytorch_result"]
        for key in (
            "quantized_MAE", "relative_MAE",
            "quantized_model_size",
            "quantized_median_latency", "quantized_p95_latency", "quantized_p99_latency",
            "relative_model_size",
            "relative_median_latency", "relative_p95_latency", "relative_p99_latency",
        ):
            assert key in pytorch

    def test_weight_only_true_includes_onnx_and_pt2_results(self, model_cls):
        model, loader = model_cls(), make_seq_dataloader()
        with _MinimalPatches(_one_config_dict(), weight_only=True):
            result = run_ptq(model, loader, weight_only=True, runs=2, warmup=1)
        assert "onnx_result" in result["cfg1"]
        assert "pt2_result" in result["cfg1"]

    def test_baseline_mae_failure_raises_runtime_error(self, model_cls):
        model, loader = model_cls(), make_seq_dataloader()
        with patch(f"{_MOD}.build_quantized_models", return_value={}), \
             patch(f"{_MOD}.evaluate_mae", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError):
                run_ptq(model, loader, runs=2, warmup=1)

    def test_does_not_mutate_base_model_parameters(self, model_cls):
        model, loader = model_cls(), make_seq_dataloader()
        params_before = {k: v.clone() for k, v in model.named_parameters()}
        with _MinimalPatches(_one_config_dict()):
            run_ptq(model, loader, runs=2, warmup=1)
        for k, v in model.named_parameters():
            assert torch.allclose(params_before[k], v), f"Parameter {k} was mutated"
