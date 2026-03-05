from dataclasses import asdict
from pathlib import Path
from typing import List, Literal

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

NUMBER_EPOCHS = 25


class SuperconductorLightning(L.LightningModule):
    """
    _summary_
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
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        return self.model(x)

    def training_step(self, batch: torch.Tensor):
        """_summary_

        Args:
            batch (torch.Tensor): _description_

        Returns:
            _type_: _description_
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
        """_summary_

        Args:
            batch (torch.Tensor): _description_
        """
        # get the data for the mini-batch
        inputs, target = batch

        # evaluate
        output = self.model(inputs)
        loss = F.l1_loss(output, target)

        self.log("valid_loss", loss)

    def test_step(self, batch: torch.Tensor):
        """_summary_

        Args:
            batch (torch.Tensor): _description_
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
    max_epochs: int = 1000,
    name: str = "placeholder_name",
    logging_directory: Path = (Path.cwd() / "models" / "logs"),
    checkpoint_directory: Path = (Path.cwd() / "models" / "checkpoints"),
    config_directory: Path = (Path.cwd() / "models" / "configs"),
    state_dict_directory: Path = (Path.cwd() / "models" / "state_dicts"),
    test_fraction: float = 0.2,
    seed: int = 42,
    n_workers: int = 4,
    batch_n: int = 64,
):

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

    trainer = Trainer(
        logger=CSVLogger((logging_directory / name), name=(name + "_csv_log")),
        callbacks=[
            ModelCheckpoint(checkpoint_directory, filename=name),
        ],
        max_epochs=max_epochs,
    )

    trainer.fit(
        model=mlp,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    trainer.test(model=mlp, dataloaders=test_loader)
    config_dict = asdict(mlp.config)
    state_dict = mlp.model.state_dict()

    torch.save(
        {"config": config_dict, "state_dict": state_dict},
        (state_dict_directory / f"{name}.pth"),
    )

    save_model_config(config=mlp.config, path=(config_directory / f"{name}.json"))


if __name__ == "__main__":
    # train the base models and save them both to checkpoint files and ONNX files
    base_config = SimpleMLPConfig(
        input_dim=81,
        output_dim=1,
        neurons_per_layer=[324, 162, 81],
        activation="relu",
        use_batch_norm=True,
    )
    construct_mlp(config=base_config, name="base_model_FP32", max_epochs=NUMBER_EPOCHS)
