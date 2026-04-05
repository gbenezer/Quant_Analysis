import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from src.quant_analysis.metric_calculation.evaluate_size_and_latency import (
    assess_relative_performance,
    estimate_quantized_size,
    evaluate_onnx_latency_and_size,
    evaluate_pt2_latency_and_size,
    evaluate_pytorch_latency_and_estimate_size,
    measure_latency_onnx,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal two-layer MLP for fast integration tests."""

    def __init__(self, in_features: int = 4, out_features: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def make_sample_input(batch: int = 1, features: int = 4) -> torch.Tensor:
    return torch.randn(batch, features)


def make_mock_onnx_session(output_shape=(1, 2)):
    """Return a mock ort.InferenceSession that immediately returns zeros."""
    session = MagicMock()
    session.run.return_value = [np.zeros(output_shape, dtype=np.float32)]
    input_info = MagicMock()
    input_info.name = "input"
    session.get_inputs.return_value = [input_info]
    return session


# ---------------------------------------------------------------------------
# estimate_quantized_size
# ---------------------------------------------------------------------------


class TestEstimateQuantizedSize:
    def test_returns_float(self):
        model = TinyModel()
        result = estimate_quantized_size(model, bits_per_weight=8)
        assert isinstance(result, float)

    def test_8bit_equals_one_byte_per_param(self):
        model = TinyModel(in_features=4, out_features=2)
        total_params = sum(p.numel() for p in model.parameters())
        assert estimate_quantized_size(model, bits_per_weight=8) == pytest.approx(
            total_params * 1.0
        )

    def test_32bit_equals_four_bytes_per_param(self):
        model = TinyModel()
        total_params = sum(p.numel() for p in model.parameters())
        assert estimate_quantized_size(model, bits_per_weight=32) == pytest.approx(
            total_params * 4.0
        )

    def test_proportional_to_bits(self):
        model = TinyModel()
        size_8 = estimate_quantized_size(model, bits_per_weight=8)
        size_16 = estimate_quantized_size(model, bits_per_weight=16)
        assert size_16 == pytest.approx(size_8 * 2.0)

    def test_4bit_is_half_of_8bit(self):
        model = TinyModel()
        size_8 = estimate_quantized_size(model, bits_per_weight=8)
        size_4 = estimate_quantized_size(model, bits_per_weight=4)
        assert size_4 == pytest.approx(size_8 / 2.0)

    def test_larger_model_yields_larger_size(self):
        small = TinyModel(in_features=4, out_features=2)
        large = TinyModel(in_features=64, out_features=32)
        assert estimate_quantized_size(large, 8) > estimate_quantized_size(small, 8)

    def test_size_is_positive(self):
        model = TinyModel()
        assert estimate_quantized_size(model, bits_per_weight=8) > 0


# ---------------------------------------------------------------------------
# assess_relative_performance
# ---------------------------------------------------------------------------


class TestAssessRelativePerformance:
    def test_returns_four_tuple(self):
        perf = (100.0, 0.01, 0.02, 0.03)
        result = assess_relative_performance(perf, perf)
        assert len(result) == 4

    def test_identical_models_yield_all_ones(self):
        perf = (200.0, 0.005, 0.008, 0.010)
        result = assess_relative_performance(perf, perf)
        assert all(r == pytest.approx(1.0) for r in result)

    def test_half_size_and_latency_yield_half_ratios(self):
        base = (200.0, 0.010, 0.020, 0.030)
        quant = (100.0, 0.005, 0.010, 0.015)
        result = assess_relative_performance(quant, base)
        assert all(r == pytest.approx(0.5) for r in result)

    def test_relative_size_ratio_is_correct(self):
        base = (400, 1.0, 1.0, 1.0)
        quant = (100, 1.0, 1.0, 1.0)
        rel_size, *_ = assess_relative_performance(quant, base)
        assert rel_size == pytest.approx(0.25)

    def test_relative_latency_ratios_are_correct(self):
        base = (100, 0.020, 0.040, 0.060)
        quant = (100, 0.010, 0.020, 0.030)
        _, rel_med, rel_p95, rel_p99 = assess_relative_performance(quant, base)
        assert rel_med == pytest.approx(0.5)
        assert rel_p95 == pytest.approx(0.5)
        assert rel_p99 == pytest.approx(0.5)

    def test_accepts_int_sizes(self):
        base = (800, 0.01, 0.02, 0.03)
        quant = (200, 0.005, 0.01, 0.015)
        result = assess_relative_performance(quant, base)
        assert result[0] == pytest.approx(0.25)

    def test_worse_model_yields_ratio_greater_than_one(self):
        base = (100.0, 0.005, 0.010, 0.015)
        quant = (200.0, 0.010, 0.020, 0.030)
        result = assess_relative_performance(quant, base)
        assert all(r == pytest.approx(2.0) for r in result)


# ---------------------------------------------------------------------------
# measure_latency_onnx
# ---------------------------------------------------------------------------


class TestMeasureLatencyOnnx:
    def _run(self, runs=10, warmup=3):
        session = make_mock_onnx_session()
        x = np.zeros((1, 4), dtype=np.float32)
        return measure_latency_onnx(session, "input", x, runs=runs, warmup=warmup)

    def test_returns_three_tuple(self):
        result = self._run()
        assert len(result) == 3

    def test_all_values_are_floats(self):
        median, p95, p99 = self._run()
        assert isinstance(median, float)
        assert isinstance(p95, float)
        assert isinstance(p99, float)

    def test_latency_ordering(self):
        median, p95, p99 = self._run(runs=50)
        assert median <= p95 <= p99

    def test_all_values_non_negative(self):
        median, p95, p99 = self._run()
        assert median >= 0.0
        assert p95 >= 0.0
        assert p99 >= 0.0

    def test_warmup_calls_are_made(self):
        session = make_mock_onnx_session()
        x = np.zeros((1, 4), dtype=np.float32)
        warmup = 5
        runs = 10
        measure_latency_onnx(session, "input", x, runs=runs, warmup=warmup)
        assert session.run.call_count == warmup + runs

    def test_single_run(self):
        session = make_mock_onnx_session()
        x = np.zeros((1, 4), dtype=np.float32)
        median, p95, p99 = measure_latency_onnx(session, "input", x, runs=1, warmup=0)
        assert median == p95 == p99


# ---------------------------------------------------------------------------
# evaluate_onnx_latency_and_size  (integration)
# ---------------------------------------------------------------------------


class TestEvaluateOnnxLatencyAndSize:
    @pytest.fixture(scope="class")
    def result(self):
        model = TinyModel(in_features=4, out_features=2)
        sample = make_sample_input(batch=1, features=4)
        return evaluate_onnx_latency_and_size(
            model, sample, device="cpu", runs=5, warmup=2
        )

    def test_returns_four_tuple(self, result):
        assert len(result) == 4

    def test_size_is_positive_integer(self, result):
        size, *_ = result
        assert isinstance(size, int)
        assert size > 0

    def test_latency_values_are_floats(self, result):
        _, median, p95, p99 = result
        assert isinstance(median, float)
        assert isinstance(p95, float)
        assert isinstance(p99, float)

    def test_latency_ordering(self, result):
        _, median, p95, p99 = result
        assert median <= p95 <= p99

    def test_all_latencies_positive(self, result):
        _, median, p95, p99 = result
        assert median > 0.0
        assert p95 > 0.0
        assert p99 > 0.0

    def test_does_not_mutate_original_model(self):
        model = TinyModel()
        original_weights = {
            k: v.clone() for k, v in model.state_dict().items()
        }
        sample = make_sample_input()
        evaluate_onnx_latency_and_size(model, sample, device="cpu", runs=3, warmup=1)
        for k, v in model.state_dict().items():
            assert torch.equal(v, original_weights[k]), f"Parameter {k} was mutated"


# ---------------------------------------------------------------------------
# evaluate_pt2_latency_and_size  (integration)
# ---------------------------------------------------------------------------


class TestEvaluatePt2LatencyAndSize:
    @pytest.fixture(scope="class")
    def result(self):
        model = TinyModel(in_features=4, out_features=2)
        sample = make_sample_input(batch=1, features=4)
        return evaluate_pt2_latency_and_size(
            model, sample, device="cpu", runs=5, warmup=2
        )

    def test_returns_four_tuple(self, result):
        assert len(result) == 4

    def test_size_is_positive_integer(self, result):
        size, *_ = result
        assert isinstance(size, int)
        assert size > 0

    def test_latency_values_are_floats(self, result):
        _, median, p95, p99 = result
        assert isinstance(median, float)
        assert isinstance(p95, float)
        assert isinstance(p99, float)

    def test_latency_ordering(self, result):
        _, median, p95, p99 = result
        assert median <= p95 <= p99

    def test_all_latencies_positive(self, result):
        _, median, p95, p99 = result
        assert median > 0.0


# ---------------------------------------------------------------------------
# evaluate_pytorch_latency_and_estimate_size
# ---------------------------------------------------------------------------


class TestEvaluatePytorchLatencyAndEstimateSize:
    @pytest.fixture(scope="class")
    def model_and_result(self):
        model = TinyModel(in_features=4, out_features=2)
        sample = make_sample_input(batch=1, features=4)
        result = evaluate_pytorch_latency_and_estimate_size(
            model, sample, device="cpu", bits_per_weight=8, runs=10, warmup=3
        )
        return model, result

    def test_returns_four_tuple(self, model_and_result):
        _, result = model_and_result
        assert len(result) == 4

    def test_size_matches_estimate_quantized_size(self, model_and_result):
        model, result = model_and_result
        expected = estimate_quantized_size(model, bits_per_weight=8)
        size, *_ = result
        assert size == pytest.approx(expected)

    def test_latency_values_are_floats(self, model_and_result):
        _, result = model_and_result
        _, median, p95, p99 = result
        assert isinstance(median, float)
        assert isinstance(p95, float)
        assert isinstance(p99, float)

    def test_latency_ordering(self, model_and_result):
        _, result = model_and_result
        _, median, p95, p99 = result
        assert median <= p95 <= p99

    def test_all_latencies_positive(self, model_and_result):
        _, result = model_and_result
        _, median, p95, p99 = result
        assert median > 0.0

    def test_size_scales_with_bits(self):
        model = TinyModel()
        sample = make_sample_input()
        size_8, *_ = evaluate_pytorch_latency_and_estimate_size(
            model, sample, "cpu", bits_per_weight=8, runs=5, warmup=1
        )
        size_16, *_ = evaluate_pytorch_latency_and_estimate_size(
            model, sample, "cpu", bits_per_weight=16, runs=5, warmup=1
        )
        assert size_16 == pytest.approx(size_8 * 2.0)

    def test_does_not_mutate_original_model(self):
        model = TinyModel()
        original_weights = {k: v.clone() for k, v in model.state_dict().items()}
        sample = make_sample_input()
        evaluate_pytorch_latency_and_estimate_size(
            model, sample, "cpu", bits_per_weight=8, runs=5, warmup=1
        )
        for k, v in model.state_dict().items():
            assert torch.equal(v, original_weights[k]), f"Parameter {k} was mutated"

    def test_original_training_mode_unchanged(self):
        # The function deepcopies the model, so the original's training state is preserved.
        model = TinyModel()
        model.train()
        sample = make_sample_input()
        evaluate_pytorch_latency_and_estimate_size(
            model, sample, "cpu", bits_per_weight=8, runs=5, warmup=1
        )
        assert model.training
