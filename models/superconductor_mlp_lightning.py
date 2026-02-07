import torch
from typing import List, Literal
import torch.nn.functional as F
import lightning as L

from data import get_superconductivity_data
from models import SuperconductorMLP

class SuperconductorLightning(L.LightningModule):
    
    def __init__(
        self,
        neurons: List[int] = [324, 162, 81],
        specified_activation: Literal["relu", "leaky_relu", "elu", "gelu", "celu"] = "relu",
        batch_norm: bool = True,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.model = SuperconductorMLP(
            neurons=neurons,
            specified_activation=specified_activation,
            batch_norm=batch_norm
        )
        self.lr = learning_rate
    
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        model_output = self.model(inputs)
        loss = F.l1_loss(model_output, target=target)
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, target = batch
        model_output = self.model(inputs, target)
        loss = F.l1_loss(model_output, target=target)
        self.log("test_loss", loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)
        return optimizer
    
