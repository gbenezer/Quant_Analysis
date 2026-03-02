import os
from pathlib import Path
from typing import List, Literal

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.export import export

from data import get_superconductivity_data
from models import SimpleMLP

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
    logging_directory: Path = (Path(os.getcwd()) / "models" / "logs"),
    checkpoint_directory: Path = (Path(os.getcwd()) / "models" / "checkpoints"),
    state_dict_directory: Path = (Path(os.getcwd()) / "models" / "state_dicts"),
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


def export_mlp_to_onnx(
    checkpoint_path: Path = (
        Path(os.getcwd()) / "models" / "checkpoints" / "base_model_FP32.ckpt"
    ),
    onnx_path: Path = (Path(os.getcwd()) / "models" / "onnx" / "base_model_FP32.onnx"),
    model_export_dtype: torch.dtype = torch.float32,
):
    """_summary_

    Args:
        checkpoint_path (Path, optional): _description_.
            Defaults to ( Path(os.getcwd()) / "models" / "checkpoints" / "base_model_FP32.ckpt" ).
        onnx_path (Path, optional): _description_.
            Defaults to (Path(os.getcwd()) / "models" / "onnx" / "base_model_FP32.onnx").
        model_export_dtype (torch.dtype, optional): _description_. Defaults to torch.float32.
    """

    lightning_model = SuperconductorLightning.load_from_checkpoint(
        checkpoint_path=checkpoint_path, map_location="cpu"
    ).eval()

    model = lightning_model.model
    model = model.to(dtype=model_export_dtype).eval()
    
    input_sample = torch.randn(1, model.input_dim, dtype=model_export_dtype)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (input_sample,),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"input": {0: "batch"}},
        opset_version=17,
    )


def export_mlp_to_pt2(
    checkpoint_path: Path = (
        Path(os.getcwd()) / "models" / "checkpoints" / "base_model_FP32.ckpt"
    ),
    export_path: Path = (Path(os.getcwd()) / "models" / "pt2" / "base_model_FP32.pt2"),
    model_export_dtype: torch.dtype = torch.float32,
):
    """_summary_

    Args:
        checkpoint_path (Path, optional): _description_.
            Defaults to ( Path(os.getcwd()) / "models" / "checkpoints" / "base_model_FP32.ckpt" ).
        export_path (Path, optional): _description_.
            Defaults to (Path(os.getcwd()) / "models" / "pt2" / "base_model_FP32.pt2").
        model_export_dtype (torch.dtype, optional): _description_. Defaults to torch.float32.
    """
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
    # train the base models and save them both to checkpoint files and ONNX files
    construct_mlp(name="base_model_FP32", max_epochs=NUMBER_EPOCHS)

    # export
    export_mlp_to_onnx(
        checkpoint_path=(MODEL_PATH / "checkpoints" / "base_model_FP32.ckpt"),
        onnx_path=(MODEL_PATH / "onnx" / "base_model_FP32.onnx"),
        model_export_dtype=torch.float32,
    )

    export_mlp_to_pt2(
        checkpoint_path=(MODEL_PATH / "checkpoints" / "base_model_FP32.ckpt"),
        export_path=(MODEL_PATH / "pt2" / "base_model_FP32.pt2"),
        model_export_dtype=torch.float32,
    )
