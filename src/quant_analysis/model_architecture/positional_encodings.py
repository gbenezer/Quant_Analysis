import math

import torch
import torch.nn as nn


class ContinuousWavenumberEncoding(nn.Module):
    def __init__(self, d_model, wn_min, wn_max):
        """
        Initialize continuous sinusoidal encoding based on wavenumber values.

        This module extends standard positional encoding to continuous scalar
        inputs by mapping normalized values to sinusoidal embeddings.

        Args:
            d_model (int): Dimensionality of the output encoding.

            wn_min (float): Minimum wavenumber used for normalization.

            wn_max (float): Maximum wavenumber used for normalization.
        """
        super().__init__()
        self.d_model = d_model
        self.wn_min = wn_min
        self.wn_max = wn_max
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, wavenumbers):
        """
        Compute sinusoidal encodings for continuous wavenumber inputs.

        Args:
            wavenumbers (torch.Tensor) - Input tensor of shape: (batch, seq_len) or (seq_len,)

        Returns:
            torch.Tensor - Encoded representation of shape: (batch, seq_len, d_model) or (seq_len, d_model)
        """
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
        """
        Initialize standard sinusoidal positional encoding.
        This module generates fixed positional encodings.

        Args:
            model_dim (int) - Dimensionality of the model.

            max_len (int, optional) - Maximum sequence length supported. Defaults to 5000.

            dropout (float, optional) - Dropout probability applied after adding positional encoding.

            batch_first (bool, optional) - 
                If True, expects input shape (batch, seq_len, model_dim).
                If False, expects (seq_len, batch, model_dim).
        """
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
        """
        Add positional encoding to input embeddings.

        Args:
            x (torch.Tensor) - Input tensor of shape: (batch, seq_len, model_dim) if batch_first=True 
            or (seq_len, batch, model_dim) otherwise

        Returns:
            torch.Tensor - Tensor of the same shape as input, with positional encoding added.
        """
        if self.batch_first:
            x = x + self.pe[:, : x.size(1)]
        else:
            x = x + self.pe[: x.size(0)]

        return self.dropout(x)
