from pathlib import Path
from typing import List, Literal

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from data import get_superconductivity_data
from src.quant_analysis.model_architecture import SimpleMLP

NUMBER_EPOCHS = 25


class SuperconductorLightning(L.LightningModule):
    """
    _summary_
    """

    def __init__(
        self,
        neurons: List[int] = [324, 162, 81],
        specified_activation: Literal[
            "relu", "leaky_relu", "elu", "gelu", "celu"
        ] = "relu",
        batch_norm: bool = True,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleMLP(
            neurons=neurons,
            specified_activation=specified_activation,
            batch_norm=batch_norm,
        )
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
    neurons: List[int] = [324, 162, 81],
    specified_activation: Literal["relu", "leaky_relu", "elu", "gelu", "celu"] = "relu",
    batch_norm: bool = True,
    learning_rate: float = 1e-3,
    max_epochs: int = 1000,
    name: str = "placeholder_name",
    logging_directory: Path = (Path.cwd() / "models" / "logs"),
    checkpoint_directory: Path = (Path.cwd() / "models" / "checkpoints"),
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

    mlp = SuperconductorLightning(
        neurons=neurons,
        specified_activation=specified_activation,
        batch_norm=batch_norm,
        learning_rate=learning_rate,
    )
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

    torch.save(mlp.model.state_dict(), (state_dict_directory / f"{name}.pth"))


if __name__ == "__main__":
    # train the base models and save them both to checkpoint files and ONNX files
    construct_mlp(name="base_model_FP32", max_epochs=NUMBER_EPOCHS)
