# =====================================================
# IMPORT LIBRARIES
# =====================================================
import math
import torch
import torch.nn as nn


# =====================================================
# POSITIONAL ENCODING
# Adds position information to sequence tokens
# =====================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x = [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T]


# =====================================================
# TRANSFORMER MODEL
# =====================================================
class SleepTransformer(nn.Module):
    def __init__(
        self,
        in_channels=4,         # dataset channels
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=5
    ):
        super().__init__()

        # Input projection: [B, C, T] -> [B, D, T']
        self.input_proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=7,
            stride=2,
            padding=3
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Stack encoder layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: [B, 4, 3000]

        x = self.input_proj(x)      # [B, 64, T']
        x = x.permute(0, 2, 1)      # [B, T', 64]

        x = self.pos_encoder(x)
        x = self.transformer(x)

        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)           # [B, 64]

        out = self.classifier(x)   # [B, 5]

        return out