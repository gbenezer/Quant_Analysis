import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.quant_analysis.metric_calculation.evaluate_mean_absolute_error import evaluate_mae


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ConstantModel(nn.Module):
    """Always predicts a fixed constant, regardless of input."""

    def __init__(self, constant: float):
        super().__init__()
        # A single bias-only linear layer whose weight is frozen at 0 and bias at `constant`.
        self.layer = nn.Linear(1, 1, bias=True)
        with torch.no_grad():
            self.layer.weight.fill_(0.0)
            self.layer.bias.fill_(constant)

    def forward(self, x):
        # Return a scalar prediction per sample.
        return self.layer(x[:, :1]).squeeze(1)


class IdentityModel(nn.Module):
    """Returns the first input feature as its prediction."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.layer.weight.fill_(1.0)

    def forward(self, x):
        return self.layer(x[:, :1]).squeeze(1)


def make_dataloader(inputs: torch.Tensor, targets: torch.Tensor, batch_size: int = 16) -> DataLoader:
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Return type and basic contract
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_float(self):
        inputs = torch.zeros(10, 1)
        targets = torch.zeros(10)
        loader = make_dataloader(inputs, targets)
        model = ConstantModel(0.0)
        result = evaluate_mae(model, loader)
        assert isinstance(result, float)

    def test_mae_is_non_negative(self):
        inputs = torch.randn(20, 4)
        targets = torch.randn(20)
        loader = make_dataloader(inputs, targets)
        model = ConstantModel(0.0)
        assert evaluate_mae(model, loader) >= 0.0


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_perfect_predictions_yield_zero_mae(self):
        # Model predicts exactly the target for every sample.
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        inputs = targets.unsqueeze(1)          # shape (4, 1)
        loader = make_dataloader(inputs, targets, batch_size=4)
        model = IdentityModel()
        assert evaluate_mae(model, loader) == pytest.approx(0.0, abs=1e-6)

    def test_constant_offset_error(self):
        # Model always predicts 0; targets are all 1 → MAE should be exactly 1.
        n = 20
        inputs = torch.zeros(n, 1)
        targets = torch.ones(n)
        loader = make_dataloader(inputs, targets, batch_size=n)
        model = ConstantModel(0.0)
        assert evaluate_mae(model, loader) == pytest.approx(1.0)

    def test_known_mae_value(self):
        # Targets: [0, 1, 2, 3], model always predicts 1.5 → MAE = (1.5+0.5+0.5+1.5)/4 = 1.0
        targets = torch.tensor([0.0, 1.0, 2.0, 3.0])
        inputs = torch.zeros(4, 1)
        loader = make_dataloader(inputs, targets, batch_size=4)
        model = ConstantModel(1.5)
        assert evaluate_mae(model, loader) == pytest.approx(1.0)

    def test_mae_consistent_across_batch_sizes(self):
        # The MAE should be the same regardless of how samples are batched.
        n = 40
        targets = torch.arange(n, dtype=torch.float32)
        inputs = torch.zeros(n, 1)
        model = ConstantModel(5.0)

        mae_batch1 = evaluate_mae(model, make_dataloader(inputs, targets, batch_size=1))
        mae_batch8 = evaluate_mae(model, make_dataloader(inputs, targets, batch_size=8))
        mae_batch40 = evaluate_mae(model, make_dataloader(inputs, targets, batch_size=40))

        assert mae_batch1 == pytest.approx(mae_batch8)
        assert mae_batch8 == pytest.approx(mae_batch40)

    def test_single_sample_dataset(self):
        # Edge case: dataset with only one sample.
        inputs = torch.tensor([[0.0]])
        targets = torch.tensor([3.0])
        loader = make_dataloader(inputs, targets, batch_size=1)
        model = ConstantModel(0.0)
        assert evaluate_mae(model, loader) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

class TestModelState:
    def test_model_is_set_to_eval_mode(self):
        inputs = torch.zeros(10, 1)
        targets = torch.zeros(10)
        loader = make_dataloader(inputs, targets)
        model = ConstantModel(0.0)
        model.train()
        evaluate_mae(model, loader)
        assert not model.training

    def test_no_gradients_accumulated(self):
        inputs = torch.zeros(10, 1)
        targets = torch.zeros(10)
        loader = make_dataloader(inputs, targets)
        model = ConstantModel(0.0)
        evaluate_mae(model, loader)
        for param in model.parameters():
            assert param.grad is None


# ---------------------------------------------------------------------------
# Input dtype casting
# ---------------------------------------------------------------------------

class TestInputDtype:
    def test_default_dtype_is_float32(self):
        # Supply float64 inputs; function should cast them to float32 before forward pass.
        inputs = torch.zeros(10, 1, dtype=torch.float64)
        targets = torch.zeros(10, dtype=torch.float64)
        loader = make_dataloader(inputs, targets)
        model = ConstantModel(0.0)
        # Should not raise a dtype mismatch error.
        result = evaluate_mae(model, loader)
        assert isinstance(result, float)

    def test_explicit_float64_dtype(self):
        inputs = torch.zeros(10, 1, dtype=torch.float64)
        targets = torch.zeros(10, dtype=torch.float64)
        loader = make_dataloader(inputs, targets)

        # Use a float64 model to avoid dtype mismatch.
        model = ConstantModel(0.0).double()
        result = evaluate_mae(model, loader, input_dtype=torch.float64)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_float16_inputs_cast_correctly(self):
        # Supply float16 inputs and request float16 casting.
        inputs = torch.zeros(10, 1, dtype=torch.float16)
        targets = torch.zeros(10, dtype=torch.float16)
        loader = make_dataloader(inputs, targets)
        model = ConstantModel(0.0).half()
        result = evaluate_mae(model, loader, input_dtype=torch.float16)
        assert result == pytest.approx(0.0, abs=1e-3)
