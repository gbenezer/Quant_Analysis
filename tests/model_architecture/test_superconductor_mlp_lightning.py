from unittest.mock import MagicMock, patch

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.quant_analysis.model_architecture.model_configs import SimpleMLPConfig
from src.quant_analysis.model_architecture.simple_mlp import SimpleMLP
from src.quant_analysis.model_architecture.superconductor_mlp_lightning import (
    SuperconductorLightning,
    construct_mlp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8


def make_config(**overrides) -> SimpleMLPConfig:
    defaults = dict(
        input_dim=INPUT_DIM,
        output_dim=1,
        neurons_per_layer=[16, 8],
        activation="relu",
        use_batch_norm=False,
    )
    defaults.update(overrides)
    return SimpleMLPConfig(**defaults)


def make_batch(batch_size: int = 4, input_dim: int = INPUT_DIM):
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size)
    return x, y


def make_loader(n: int = 16, input_dim: int = INPUT_DIM, batch_size: int = 4):
    x = torch.randn(n, input_dim)
    y = torch.randn(n)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_is_lightning_module(self):
        module = SuperconductorLightning(make_config())
        assert isinstance(module, L.LightningModule)

    def test_stores_config(self):
        config = make_config()
        module = SuperconductorLightning(config)
        assert module.config is config

    def test_stores_learning_rate(self):
        module = SuperconductorLightning(make_config(), learning_rate=5e-4)
        assert module.lr == 5e-4

    def test_default_learning_rate(self):
        module = SuperconductorLightning(make_config())
        assert module.lr == 1e-3

    def test_inner_model_is_simple_mlp(self):
        module = SuperconductorLightning(make_config())
        assert isinstance(module.model, SimpleMLP)

    def test_inner_model_config_matches(self):
        config = make_config(neurons_per_layer=[32, 16])
        module = SuperconductorLightning(config)
        assert module.model.config is config


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape_batched(self):
        module = SuperconductorLightning(make_config(input_dim=8, output_dim=1))
        module.eval()
        x = torch.randn(6, 8)
        with torch.no_grad():
            out = module(x)
        assert out.shape == (6,)

    def test_output_is_float32(self):
        module = SuperconductorLightning(make_config())
        module.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            out = module(x)
        assert out.dtype == torch.float32

    def test_forward_delegates_to_inner_model(self):
        config = make_config()
        module = SuperconductorLightning(config)
        module.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            out_lightning = module(x)
            out_model = module.model(x)
        assert torch.allclose(out_lightning, out_model)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

class TestTrainingStep:
    def test_returns_scalar_tensor(self):
        module = SuperconductorLightning(make_config())
        module.log = MagicMock()
        loss = module.training_step(make_batch())
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_loss_is_non_negative(self):
        module = SuperconductorLightning(make_config())
        module.log = MagicMock()
        loss = module.training_step(make_batch())
        assert loss.item() >= 0.0

    def test_logs_train_loss(self):
        module = SuperconductorLightning(make_config())
        module.log = MagicMock()
        module.training_step(make_batch())
        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert "train_loss" in logged_keys

    def test_gradients_computable(self):
        module = SuperconductorLightning(make_config())
        module.log = MagicMock()
        loss = module.training_step(make_batch())
        loss.backward()
        for name, param in module.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

class TestValidationStep:
    def test_returns_none(self):
        module = SuperconductorLightning(make_config())
        module.log = MagicMock()
        result = module.validation_step(make_batch())
        assert result is None

    def test_logs_valid_loss(self):
        module = SuperconductorLightning(make_config())
        module.log = MagicMock()
        module.validation_step(make_batch())
        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert "valid_loss" in logged_keys


# ---------------------------------------------------------------------------
# Test step
# ---------------------------------------------------------------------------

class TestTestStep:
    def test_returns_none(self):
        module = SuperconductorLightning(make_config())
        module.log = MagicMock()
        result = module.test_step(make_batch())
        assert result is None

    def test_logs_test_loss(self):
        module = SuperconductorLightning(make_config())
        module.log = MagicMock()
        module.test_step(make_batch())
        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert "test_loss" in logged_keys


# ---------------------------------------------------------------------------
# Loss consistency across steps
# ---------------------------------------------------------------------------

class TestLossConsistency:
    """Training, validation, and test steps all use L1 loss on the same batch;
    their scalar values should be equal."""

    def test_same_loss_value_across_steps(self):
        module = SuperconductorLightning(make_config())
        module.log = MagicMock()
        module.eval()
        batch = make_batch()
        with torch.no_grad():
            train_loss = module.training_step(batch)
            val_loss_logged = module.log.call_args_list[-1]  # last log call

        module.log.reset_mock()
        with torch.no_grad():
            module.validation_step(batch)
        val_loss_value = module.log.call_args_list[0].args[1]

        assert torch.isclose(train_loss.detach(), torch.tensor(val_loss_value))


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------

class TestConfigureOptimizers:
    def test_returns_adam_optimizer(self):
        module = SuperconductorLightning(make_config())
        optimizer = module.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)

    def test_optimizer_learning_rate(self):
        lr = 2e-4
        module = SuperconductorLightning(make_config(), learning_rate=lr)
        optimizer = module.configure_optimizers()
        assert optimizer.param_groups[0]["lr"] == lr

    def test_optimizer_has_module_parameters(self):
        module = SuperconductorLightning(make_config())
        optimizer = module.configure_optimizers()
        param_ids = {id(p) for p in module.parameters()}
        opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert opt_param_ids == param_ids


# ---------------------------------------------------------------------------
# construct_mlp
# ---------------------------------------------------------------------------

MODULE_PATH = "src.quant_analysis.model_architecture.superconductor_mlp_lightning"


def _make_fake_data_fn(input_dim=INPUT_DIM):
    """Returns a mock replacement for get_superconductivity_data."""
    loader = make_loader(input_dim=input_dim)
    return MagicMock(return_value=(None, None, None, None, loader, loader, loader))


class TestConstructMlp:
    @patch(f"{MODULE_PATH}.seed_everything")
    @patch(f"{MODULE_PATH}.Trainer")
    @patch(f"{MODULE_PATH}.get_superconductivity_data")
    def test_returns_simple_mlp(self, mock_data, mock_trainer_cls, mock_seed):
        mock_data.side_effect = _make_fake_data_fn().side_effect
        loader = make_loader()
        mock_data.return_value = (None, None, None, None, loader, loader, loader)
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        config = make_config()
        result = construct_mlp(config, max_epochs=1)

        assert isinstance(result, SimpleMLP)

    @patch(f"{MODULE_PATH}.seed_everything")
    @patch(f"{MODULE_PATH}.Trainer")
    @patch(f"{MODULE_PATH}.get_superconductivity_data")
    def test_returned_model_is_deepcopy(self, mock_data, mock_trainer_cls, mock_seed):
        loader = make_loader()
        mock_data.return_value = (None, None, None, None, loader, loader, loader)
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        config = make_config()
        result = construct_mlp(config, max_epochs=1)

        # deepcopy means it is not the same object as the inner lightning model
        assert result is not mock_trainer_cls.return_value

    @patch(f"{MODULE_PATH}.seed_everything")
    @patch(f"{MODULE_PATH}.Trainer")
    @patch(f"{MODULE_PATH}.get_superconductivity_data")
    def test_seed_everything_called_with_seed(self, mock_data, mock_trainer_cls, mock_seed):
        loader = make_loader()
        mock_data.return_value = (None, None, None, None, loader, loader, loader)
        mock_trainer_cls.return_value = MagicMock()

        construct_mlp(make_config(), seed=99, max_epochs=1)

        mock_seed.assert_called_once_with(99)

    @patch(f"{MODULE_PATH}.seed_everything")
    @patch(f"{MODULE_PATH}.Trainer")
    @patch(f"{MODULE_PATH}.get_superconductivity_data")
    def test_trainer_fit_and_test_called(self, mock_data, mock_trainer_cls, mock_seed):
        loader = make_loader()
        mock_data.return_value = (None, None, None, None, loader, loader, loader)
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        construct_mlp(make_config(), max_epochs=1)

        mock_trainer.fit.assert_called_once()
        mock_trainer.test.assert_called_once()

    @patch(f"{MODULE_PATH}.seed_everything")
    @patch(f"{MODULE_PATH}.Trainer")
    @patch(f"{MODULE_PATH}.get_superconductivity_data")
    def test_data_loader_called_with_correct_fractions(self, mock_data, mock_trainer_cls, mock_seed):
        loader = make_loader()
        mock_data.return_value = (None, None, None, None, loader, loader, loader)
        mock_trainer_cls.return_value = MagicMock()

        construct_mlp(make_config(), test_fraction=0.15, batch_n=32, n_workers=2, max_epochs=1)

        _, kwargs = mock_data.call_args
        assert kwargs["test_fraction"] == 0.15
        assert kwargs["batch_n"] == 32
        assert kwargs["n_workers"] == 2

    @patch(f"{MODULE_PATH}.save_model_config")
    @patch(f"{MODULE_PATH}.seed_everything")
    @patch(f"{MODULE_PATH}.Trainer")
    @patch(f"{MODULE_PATH}.get_superconductivity_data")
    def test_no_files_saved_when_save_output_false(
        self, mock_data, mock_trainer_cls, mock_seed, mock_save_config
    ):
        loader = make_loader()
        mock_data.return_value = (None, None, None, None, loader, loader, loader)
        mock_trainer_cls.return_value = MagicMock()

        with patch("torch.save") as mock_torch_save:
            construct_mlp(make_config(), save_output=False, max_epochs=1)
            mock_torch_save.assert_not_called()

        mock_save_config.assert_not_called()
