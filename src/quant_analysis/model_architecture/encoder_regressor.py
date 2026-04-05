import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from src.quant_analysis.model_architecture.model_configs import EncoderRegressorConfig


class EncoderRegressor(nn.Module):
    def __init__(self, config: EncoderRegressorConfig):
        """Initialize the encoder regressor.

        Params:
            config: Configuration dataclass specifying model architecture.

        Raises:
            ValueError: If model_dim is not divisible by n_heads_encoder,
                       or if an unsupported pooling/activation is specified.
        """
        super().__init__()
        self.config = config

        # Validate configuration
        self._validate_config()

        # Input projection: raw features -> model dimension
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)

        # CLS token for classification-style pooling
        if config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.model_dim))

        # Positional encoding
        if config.use_positional_encoding:
            # +1 for potential CLS token
            max_len = (
                config.max_seq_len + 1
                if config.pooling == "cls"
                else config.max_seq_len
            )
            self.pos_encoder = SinusoidalPositionalEncoding(
                model_dim=config.model_dim,
                max_len=max_len,
                dropout=config.dropout,
                batch_first=config.batch_first,
            )
        else:
            self.pos_encoder = None

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.n_heads_encoder,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=config.batch_first,
            norm_first=config.norm_first,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.n_layers_encoder,
            norm=nn.LayerNorm(config.model_dim),
            enable_nested_tensor=False,  # More predictable behavior
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(config.model_dim, config.feedforward_dim),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _validate_config(self) -> None:
        """
        Validate configuration parameters for consistency and correctness.

        Raises:
            ValueError - If any configuration parameter is invalid.
        """
        if self.config.model_dim % self.config.n_heads_encoder != 0:
            raise ValueError(
                f"model_dim ({self.config.model_dim}) must be divisible by "
                f"n_heads_encoder ({self.config.n_heads_encoder})"
            )

        if self.config.pooling not in ["cls", "mean", "last"]:
            raise ValueError(
                f"pooling must be 'cls', 'mean', or 'last', got '{self.config.pooling}'"
            )

        if self.config.activation not in ["relu", "gelu"]:
            raise ValueError(
                f"activation must be 'relu' or 'gelu', got '{self.config.activation}'"
            )

    def _init_weights(self) -> None:
        """
        Initialize model parameters.

        Additional behavior:
            - For deep encoders (n_layers_encoder > 6), residual projections
            are scaled to improve training stability.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale residual connections for deeper networks
        if self.config.n_layers_encoder > 6:
            self._scale_residual_weights()

        # Small initialization for final regression layer
        final_layer = self.head[-1]
        nn.init.normal_(final_layer.weight, std=0.01)
        nn.init.zeros_(final_layer.bias)

        # Initialize CLS token if present
        if hasattr(self, "cls_token"):
            nn.init.normal_(self.cls_token, std=0.02)

    def _scale_residual_weights(self) -> None:
        """
        Scale residual projection weights for deep transformer networks.

        This helps stabilize training when using many encoder layers.
        """
        scale = (2 * self.config.n_layers_encoder) ** -0.5

        for name, param in self.named_parameters():
            # Target attention output and FFN output projections
            if any(key in name for key in ["out_proj.weight", "linear2.weight"]):
                param.data *= scale

    def _pool_sequence(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool sequence representation to a single vector.

        Params:
            x: Encoded sequence of shape (batch, seq_len, model_dim) if batch_first,
               else (seq_len, batch, model_dim).
            padding_mask: Optional boolean mask where True indicates padded positions.
                         Shape: (batch, seq_len).

        Returns:
            Pooled representation of shape (batch, model_dim).
        """
        if not self.config.batch_first:
            x = x.transpose(0, 1)  # -> (batch, seq_len, model_dim)

        if self.config.pooling == "cls":
            # CLS token is always at position 0
            return x[:, 0]

        elif self.config.pooling == "last":
            if padding_mask is not None:
                # Find last non-padded position for each sequence
                lengths = (~padding_mask).sum(dim=1) - 1  # (batch,)
                batch_indices = torch.arange(x.size(0), device=x.device)
                return x[batch_indices, lengths]
            else:
                return x[:, -1]

        elif self.config.pooling == "mean":
            if padding_mask is not None:
                # Masked mean: exclude padded positions
                mask = ~padding_mask.unsqueeze(-1)  # (batch, seq_len, 1)
                x_masked = x * mask
                return x_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                return x.mean(dim=1)

        else:
            raise ValueError(f"Unknown pooling method: {self.config.pooling}")

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the encoder regressor.

        Params:
            x: Input tensor of shape (batch, seq_len, input_dim) if batch_first,
               else (seq_len, batch, input_dim).
            src_key_padding_mask: Optional boolean mask where True indicates
                                  padded positions to ignore. Shape: (batch, seq_len).

        Returns:
            Regression output of shape (batch, output_dim).
        """
        # Get batch size based on tensor layout
        if self.config.batch_first:
            batch_size = x.size(0)
        else:
            batch_size = x.size(1)

        # Project input features to model dimension
        x = self.input_projection(x)

        # Prepend CLS token if using CLS pooling
        if self.config.pooling == "cls":
            if self.config.batch_first:
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
            else:
                cls_tokens = self.cls_token.transpose(0, 1).expand(-1, batch_size, -1)
                x = torch.cat([cls_tokens, x], dim=0)

            # Extend padding mask for CLS token (never masked)
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                src_key_padding_mask = torch.cat(
                    [cls_mask, src_key_padding_mask], dim=1
                )

        # Add positional encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        # Pass through transformer encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Pool sequence to single vector
        x = self._pool_sequence(x, padding_mask=src_key_padding_mask)

        # Regression head
        return self.head(x)


# Smoke test
if __name__ == "__main__":
    # Test configuration for time-series forecasting
    config_ts = EncoderRegressorConfig(
        input_dim=1,
        model_dim=128,
        n_heads_encoder=4,
        n_layers_encoder=4,
        feedforward_dim=512,
        dropout=0.1,
        output_dim=24,
        pooling="last",
        activation="gelu",
        max_seq_len=48,
        use_positional_encoding=True,
        batch_first=True,
        norm_first=True,
    )

    model_ts = EncoderRegressor(config_ts)
    print("Time-series config:")
    print(model_ts)
    print(f"\nTotal parameters: {sum(p.numel() for p in model_ts.parameters()):,}")

    # Test forward pass
    x_ts = torch.randn(32, 48, 1)  # (batch, seq_len, input_dim)
    y_ts = model_ts(x_ts)
    print(f"Input shape: {x_ts.shape}")
    print(f"Output shape: {y_ts.shape}")

    # Test configuration for tabular regression (like superconductor)
    config_tab = EncoderRegressorConfig(
        input_dim=1,
        model_dim=64,
        n_heads_encoder=4,
        n_layers_encoder=2,
        feedforward_dim=256,
        dropout=0.1,
        output_dim=1,
        pooling="mean",
        activation="gelu",
        max_seq_len=81,  # 81 features as sequence
        use_positional_encoding=False,
        batch_first=True,
        norm_first=True,
    )

    model_tab = EncoderRegressor(config_tab)
    print("\n" + "=" * 60)
    print("Tabular config:")
    print(model_tab)
    print(f"\nTotal parameters: {sum(p.numel() for p in model_tab.parameters()):,}")

    # Test forward pass - treating each feature as a token
    x_tab = torch.randn(32, 81, 1)  # (batch, n_features, 1)
    y_tab = model_tab(x_tab)
    print(f"Input shape: {x_tab.shape}")
    print(f"Output shape: {y_tab.shape}")

    # Test with CLS pooling
    config_cls = EncoderRegressorConfig(
        input_dim=8,
        model_dim=128,
        n_heads_encoder=8,
        n_layers_encoder=3,
        feedforward_dim=512,
        dropout=0.1,
        output_dim=1,
        pooling="cls",
        activation="gelu",
        max_seq_len=100,
        use_positional_encoding=True,
        batch_first=True,
        norm_first=True,
    )

    model_cls = EncoderRegressor(config_cls)
    print("\n" + "=" * 60)
    print("CLS pooling config:")
    print(f"Total parameters: {sum(p.numel() for p in model_cls.parameters()):,}")

    x_cls = torch.randn(16, 100, 8)
    y_cls = model_cls(x_cls)
    print(f"Input shape: {x_cls.shape}")
    print(f"Output shape: {y_cls.shape}")

    # Test with padding mask
    print("\n" + "=" * 60)
    print("Testing with padding mask:")
    padding_mask = torch.zeros(16, 100, dtype=torch.bool)
    padding_mask[:, 80:] = True  # Last 20 positions are padded
    y_masked = model_cls(x_cls, src_key_padding_mask=padding_mask)
    print(f"Output shape with masking: {y_masked.shape}")

    print("\nAll tests passed!")
