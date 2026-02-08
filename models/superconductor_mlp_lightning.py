import os
from pathlib import Path
from typing import List, Literal, Optional

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from torch.export import export

from data import get_superconductivity_data
from models import SuperconductorMLP

MODEL_PATH = Path(os.getcwd()) / "models"
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
        self.model = SuperconductorMLP(
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

        # evaluate in input precision
        model_output = self.model.to(dtype=inputs.dtype)(inputs)

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
        """_summary_

        Args:
            batch (torch.Tensor): _description_
        """
        # get the data for the mini-batch
        inputs, target = batch

        # evaluate in input precision
        model_output = self.model.to(dtype=inputs.dtype)(inputs)

        # cast the model output back to the target dtype for loss calculation
        target_dtype = target.dtype
        output = model_output.to(dtype=target_dtype)
        loss = F.l1_loss(output, target)

        self.log("valid_loss", loss)

    def test_step(self, batch: torch.Tensor):
        """_summary_

        Args:
            batch (torch.Tensor): _description_
        """
        # get the data for the mini-batch
        inputs, target = batch

        # evaluate in input precision
        model_output = self.model.to(dtype=inputs.dtype)(inputs)

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
    max_epochs: int = 1000,
    name: str = "placeholder_name",
    logging_directory: Path = (Path(os.getcwd()) / "models" / "logs"),
    checkpoint_directory: Path = (Path(os.getcwd()) / "models" / "checkpoints"),
    test_fraction: float = 0.2,
    seed: int = 42,
    n_workers: int = 4,
    batch_n: int = 64,
    precision: Optional[_PRECISION_INPUT] = None
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
        precision=precision
    )

    trainer.fit(
        model=mlp,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    trainer.test(model=mlp, dataloaders=test_loader)


def export_mlp_to_onnx(
    checkpoint_path: Path = (
        Path(os.getcwd()) / "models" / "checkpoints" / "base_model_FP32.ckpt"
    ),
    onnx_path: Path = (Path(os.getcwd()) / "models" / "onnx" / "base_model_FP32.onnx"),
    model_export_dtype: torch.dtype = torch.float32,
):
    model = (
        SuperconductorLightning.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cpu",
        )
        .eval()
        .to(dtype=model_export_dtype)
    )
    input_sample = torch.rand((1, 81), dtype=model_export_dtype)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.to_onnx(file_path=onnx_path, input_sample=input_sample)


def export_mlp_to_pt2(
    checkpoint_path: Path = (
        Path(os.getcwd()) / "models" / "checkpoints" / "base_model_FP32.ckpt"
    ),
    export_path: Path = (Path(os.getcwd()) / "models" / "pt2" / "base_model_FP32.pt2"),
    model_export_dtype: torch.dtype = torch.float32,
):
    model = (
        SuperconductorLightning.load_from_checkpoint(
            checkpoint_path=checkpoint_path, map_location="cpu"
        )
        .eval()
        .to(dtype=model_export_dtype)
    )
    input_sample = torch.rand((1, 81), dtype=model_export_dtype)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    exported_program = export(model, (input_sample,))
    torch.export.save(exported_program, export_path)


if __name__ == "__main__":
    # train the 4 base models and save them both to checkpoint files and ONNX files
    construct_mlp(name="base_model_FP32", max_epochs=NUMBER_EPOCHS)
    construct_mlp(
        name="base_model_FP32_no_norm", max_epochs=NUMBER_EPOCHS, batch_norm=False
    )

    # name, model dtype, and batch_norm
    elements = [
        ("base_model_FP32", torch.float32),
        ("base_model_FP32_no_norm", torch.float32),
    ]

    # export to onnx
    for model_name, model_dtype in elements:
        export_mlp_to_onnx(
            checkpoint_path=(MODEL_PATH / "checkpoints" / f"{model_name}.ckpt"),
            onnx_path=(MODEL_PATH / "onnx" / f"{model_name}.onnx"),
            model_export_dtype=model_dtype,
        )

        export_mlp_to_pt2(
            checkpoint_path=(MODEL_PATH / "checkpoints" / f"{model_name}.ckpt"),
            export_path=(MODEL_PATH / "pt2" / f"{model_name}.pt2"),
            model_export_dtype=model_dtype,
        )
