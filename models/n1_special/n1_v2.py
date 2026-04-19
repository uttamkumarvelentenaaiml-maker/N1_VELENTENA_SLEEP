# =====================================================
# FILE: models/n1_special/n1_v2.py
# N1 SPECIAL V2 (ALL-IN-ONE)
# Residual Multi-Scale CNN + BiLSTM + Multihead Attention
# =====================================================

import torch
import torch.nn as nn


# =====================================================
# RESIDUAL BLOCK
# =====================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),

            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        out = out + x
        return self.relu(out)


# =====================================================
# MAIN MODEL
# =====================================================
class N1V2Net(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # ---------------------------------------------
        # MULTI-SCALE INPUT BRANCHES
        # ---------------------------------------------
        self.b3 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.b7 = nn.Conv1d(4, 32, kernel_size=7, padding=3)
        self.b15 = nn.Conv1d(4, 32, kernel_size=15, padding=7)

        self.bn = nn.BatchNorm1d(96)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        # ---------------------------------------------
        # FEATURE PROJECTION
        # ---------------------------------------------
        self.proj = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # ---------------------------------------------
        # RESIDUAL REFINEMENT
        # ---------------------------------------------
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)

        # ---------------------------------------------
        # BiLSTM
        # ---------------------------------------------
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # ---------------------------------------------
        # MULTIHEAD ATTENTION
        # ---------------------------------------------
        self.mha = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        # ---------------------------------------------
        # ATTENTION SCORE POOLING
        # ---------------------------------------------
        self.score = nn.Linear(128, 1)

        # ---------------------------------------------
        # CLASSIFIER
        # ---------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x = [B, 4, 3000]

        # Multi-scale branches
        x1 = self.b3(x)
        x2 = self.b7(x)
        x3 = self.b15(x)

        x = torch.cat([x1, x2, x3], dim=1)   # [B,96,T]
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        # Projection
        x = self.proj(x)                     # [B,128,T]

        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)

        # To sequence format
        x = x.permute(0, 2, 1)              # [B,T,128]

        # BiLSTM
        x, _ = self.lstm(x)

        # Multihead attention
        x, _ = self.mha(x, x, x)

        # Score pooling
        w = self.score(x)
        w = torch.softmax(w, dim=1)

        x = (x * w).sum(dim=1)              # [B,128]

        # Output
        out = self.fc(x)
        return out