import math

import torch
import torch.nn as nn


class ContinuousWavenumberEncoding(nn.Module):
    def __init__(self, d_model, wn_min, wn_max):
        super().__init__()
        self.d_model = d_model
        self.wn_min = wn_min
        self.wn_max = wn_max
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, wavenumbers):
        # wavenumbers: (batch, seq_len) or (seq_len,)
        # normalize to [0, 1]
        wn_norm = (wavenumbers - self.wn_min) / (self.wn_max - self.wn_min)

        pe = torch.zeros(*wavenumbers.shape, self.d_model, device=wavenumbers.device)
        pe[..., 0::2] = torch.sin(wn_norm.unsqueeze(-1) * self.div_term)
        pe[..., 1::2] = torch.cos(wn_norm.unsqueeze(-1) * self.div_term)
        return pe


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        model_dim: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        # Precompute positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim)
        )

        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but moves with model to device)
        if batch_first:
            pe = pe.unsqueeze(0)  # (1, max_len, model_dim)
        else:
            pe = pe.unsqueeze(1)  # (max_len, 1, model_dim)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            x = x + self.pe[:, : x.size(1)]
        else:
            x = x + self.pe[: x.size(0)]

        return self.dropout(x)
