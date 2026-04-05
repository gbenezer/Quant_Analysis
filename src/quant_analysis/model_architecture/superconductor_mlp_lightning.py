import copy
from dataclasses import asdict
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from data import get_superconductivity_data
from src.quant_analysis.model_architecture.model_configs import (
    SimpleMLPConfig,
    save_model_config,
)
from src.quant_analysis.model_architecture.simple_mlp import SimpleMLP


class SuperconductorLightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for specification of training, validation, testing, and optimizer configuration
    of SimpleMLP neural networks trained on superconductor critical temperature prediction data.
    """

    def __init__(
        self,
        config: SimpleMLPConfig,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = SimpleMLP(config=config)
        self.lr = learning_rate

    def forward(self, x: torch.Tensor):
        """Forward pass through the neural network

        Args:
            x (torch.Tensor): a set of superconductor material features

        Returns:
            torch.Tensor: predicted critical temperatures
        """
        return self.model(x)

    def training_step(self, batch: torch.Tensor):
        """The training step for neural network backpropagation

        Args:
            batch (torch.Tensor): a set of superconductor material features

        Returns:
            torch.Tensor: the mean absolute error in predicting critical temperatures
        """
        # get the data for the mini-batch
        inputs, target = batch

        # evaluate
        output = self.model(inputs)
        loss = F.l1_loss(output, target)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch: torch.Tensor):
        """The validation step for neural network training

        Args:
            batch (torch.Tensor): a set of superconductor material features
        """
        # get the data for the mini-batch
        inputs, target = batch

        # evaluate
        output = self.model(inputs)
        loss = F.l1_loss(output, target)

        self.log("valid_loss", loss)

    def test_step(self, batch: torch.Tensor):
        """The testing step for neural network evaluation

        Args:
            batch (torch.Tensor): a set of superconductor material features
        """
        # get the data for the mini-batch
        inputs, target = batch

        # evaluate
        output = self.model(inputs)
        loss = F.l1_loss(output, target)

        self.log("test_loss", loss)

    def configure_optimizers(
        self,
    ):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def construct_mlp(
    config: SimpleMLPConfig,
    learning_rate: float = 1e-3,
    max_epochs: int = 25,
    name: str = "placeholder_name",
    logging_directory: Path = (Path.cwd() / "models" / "logs"),
    checkpoint_directory: Path = (Path.cwd() / "models" / "checkpoints"),
    config_directory: Path = (Path.cwd() / "models" / "configs"),
    state_dict_directory: Path = (Path.cwd() / "models" / "state_dicts"),
    test_fraction: float = 0.2,
    seed: int | None = 42,
    n_workers: int = 4,
    batch_n: int = 64,
    save_output: bool = False,
):
    """A convenience function to construct a SimpleMLP (feedforward neural network),
    train it on superconductivity critical temperature prediction using PyTorch Lightning,
    and output logs, a checkpoint file, a JSON file storing the SimpleMLPConfig parameters,
    along with a state dictionary.

    Args:
        config (SimpleMLPConfig): the SimpleMLPConfig dataclass specifying the feedforward network architecture
        learning_rate (float, optional): the learning rate to set. Defaults to 1e-3.
        max_epochs (int, optional): the number of training epochs. 
            max refers to a prior iteration where early stopping was implemented. Defaults to 25.
        name (str, optional): model name to save all files to. Defaults to "placeholder_name".
        logging_directory (Path, optional): directory to save PyTorch Lightning logfiles to. 
            Defaults to (Path.cwd() / "models" / "logs").
        checkpoint_directory (Path, optional): directory to save PyTorch Lightning checkpoint file to.
            Defaults to (Path.cwd() / "models" / "checkpoints").
        config_directory (Path, optional): directory to save SimpleMLPConfig parameters to (in a JSON file).
            Defaults to (Path.cwd() / "models" / "configs").
        state_dict_directory (Path, optional): directory to save PyTorch state dictionary
            (including SimpleMLPConfig parameters) to. Defaults to (Path.cwd() / "models" / "state_dicts").
        test_fraction (float, optional): Fraction of superconductivity data to use as test partition. Defaults to 0.2.
        seed (int | None, optional): random seed. Defaults to 42.
        n_workers (int, optional): number of workers/threads to allow dataloader to use. Defaults to 4.
        batch_n (int, optional): number of sample points per batch. Defaults to 64.
        save_output (bool, optional): whether or not to save any external files. Defaults to False.

    Returns:
        SimpleMLP: the trained SimpleMLP feedforward PyTorch nn.Module neural network
    """
    seed_everything(seed)

    # get the datasets
    _, _, _, _, train_loader, valid_loader, test_loader = get_superconductivity_data(
        test_fraction=test_fraction,
        random_seed=seed,
        n_workers=n_workers,
        batch_n=batch_n,
        validation_set=True,
    )

    mlp = SuperconductorLightning(config=config, learning_rate=learning_rate)
    mlp.compile()

    if save_output:
        trainer = Trainer(
            logger=CSVLogger((logging_directory / name), name=(name + "_csv_log")),
            callbacks=[
                ModelCheckpoint(checkpoint_directory, filename=name),
            ],
            max_epochs=max_epochs,
        )
    else:
        trainer = Trainer(logger=False, max_epochs=max_epochs)

    trainer.fit(
        model=mlp,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    trainer.test(model=mlp, dataloaders=test_loader)

    mlp.model.eval()

    if save_output:
        config_dict = asdict(mlp.config)
        state_dict = mlp.model.state_dict()

        torch.save(
            {"config": config_dict, "state_dict": state_dict},
            (state_dict_directory / f"{name}.pth"),
        )

        save_model_config(config=mlp.config, path=(config_directory / f"{name}.json"))

    return copy.deepcopy(mlp.model)


if __name__ == "__main__":
    # train the base model and export to state dict and config
    NUMBER_EPOCHS = 25
    base_config = SimpleMLPConfig(
        input_dim=81,
        output_dim=1,
        neurons_per_layer=[512, 256, 128],
        activation="relu",
        use_batch_norm=True,
    )
    output_mlp = construct_mlp(
        config=base_config, name="base_model_FP32_smoke_test", max_epochs=NUMBER_EPOCHS
    )
    print(output_mlp)
