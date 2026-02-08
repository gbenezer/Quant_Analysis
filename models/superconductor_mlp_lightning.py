from pathlib import Path
import os
import torch
from typing import List, Literal
import torch.nn.functional as F
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

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
    ):
        super().__init__()
        self.model = SuperconductorMLP(
            neurons=neurons,
            specified_activation=specified_activation,
            batch_norm=batch_norm,
        )
        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        model_output = self.model(inputs).squeeze()
        loss = F.l1_loss(model_output, target=target)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        model_output = self.model(inputs).squeeze()
        loss = F.l1_loss(model_output, target=target)
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        model_output = self.model(inputs).squeeze()
        loss = F.l1_loss(model_output, target=target)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


# get the datasets
_, _, _, _, train_loader, valid_loader, test_loader = get_superconductivity_data(
    test_fraction=0.2,
    random_seed=42,
    n_workers=4,
    batch_n=64,
    validation_set=True,
)

# validate the architecture
# test_module = SuperconductorMLP()
# print(test_module)
# train_features, train_labels = next(iter(train_loader))
# print(train_features.shape)
# print(train_features)

# train the base model
seed_everything(seed=42, workers=True)
superconductor_lightning_mlp = SuperconductorLightning()
trainer = Trainer(
    logger=CSVLogger(MODEL_PATH, name="logs"),
    callbacks=[
        EarlyStopping(
            monitor="valid_loss", mode="min", check_on_train_epoch_end=False, patience=5
        ),
        ModelCheckpoint((MODEL_PATH / "checkpoints"), filename="base_model_checkpoint"),
    ],
)
trainer.fit(
    model=superconductor_lightning_mlp,
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader,
)
trainer.test(model=superconductor_lightning_mlp, dataloaders=test_loader)
