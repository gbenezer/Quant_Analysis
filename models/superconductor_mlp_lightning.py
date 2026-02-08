import os
import pstats
from pathlib import Path
from typing import List, Literal

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from data import get_superconductivity_data
from models import SuperconductorMLP

MODEL_PATH = Path(os.getcwd()) / "models"


class SuperconductorLightning(L.LightningModule):
    def __init__(
        self,
        neurons: List[int] = [324, 162, 81],
        specified_activation: Literal[
            "relu", "leaky_relu", "elu", "gelu", "celu"
        ] = "relu",
        batch_norm: bool = True,
        learning_rate: float = 1e-3,
        model_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.model = SuperconductorMLP(
            neurons=neurons,
            specified_activation=specified_activation,
            batch_norm=batch_norm,
            model_dtype=model_dtype,
        )
        self.lr = learning_rate

    def training_step(self, batch: torch.Tensor):

        # get the data for the mini-batch
        inputs, target = batch

        # if the input is not of the same dtype as the model,
        # explicitly cast the input to the model precision
        if inputs.dtype != self.model.model_dtype:
            inputs = inputs.to(dtype=self.model.model_dtype)

        # evaluate in model precision
        model_output = self.model(inputs).squeeze()

        # cast the model output back to the target dtype for loss calculation
        target_dtype = target.dtype
        output = model_output.to(dtype=target_dtype)
        loss = F.l1_loss(output, target)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch: torch.Tensor):

        # get the data for the mini-batch
        inputs, target = batch

        # if the input is not of the same dtype as the model,
        # explicitly cast the input to the model precision
        if inputs.dtype != self.model.model_dtype:
            inputs = inputs.to(dtype=self.model.model_dtype)

        # evaluate in model precision
        model_output = self.model(inputs).squeeze()

        # cast the model output back to the target dtype for loss calculation
        target_dtype = target.dtype
        output = model_output.to(dtype=target_dtype)
        loss = F.l1_loss(output, target)

        self.log("valid_loss", loss)

    def test_step(self, batch: torch.Tensor):
        # get the data for the mini-batch
        inputs, target = batch

        # if the input is not of the same dtype as the model,
        # explicitly cast the input to the model precision
        if inputs.dtype != self.model.model_dtype:
            inputs = inputs.to(dtype=self.model.model_dtype)

        # evaluate in model precision
        model_output = self.model(inputs).squeeze()

        # cast the model output back to the target dtype for loss calculation
        target_dtype = target.dtype
        output = model_output.to(dtype=target_dtype)
        loss = F.l1_loss(output, target)

        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def construct_mlp(
    neurons: List[int] = [324, 162, 81],
    specified_activation: Literal["relu", "leaky_relu", "elu", "gelu", "celu"] = "relu",
    batch_norm: bool = True,
    learning_rate: float = 1e-3,
    model_dtype: torch.dtype = torch.float64,
    name: str = "placeholder_name",
    logging_directory: Path = (Path(os.getcwd()) / "models" / "logs"),
    checkpoint_directory: Path = (Path(os.getcwd()) / "models" / "checkpoints"),
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
        model_dtype=model_dtype,
    )

    trainer = Trainer(
        logger=CSVLogger((logging_directory / name), name=(name + "_csv_log")),
        callbacks=[
            EarlyStopping(
                monitor="valid_loss",
                mode="min",
                check_on_train_epoch_end=False,
                patience=5,
            ),
            ModelCheckpoint(checkpoint_directory, filename=name),
        ],
    )

    trainer.fit(
        model=mlp,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    trainer.test(model=mlp, dataloaders=test_loader)
