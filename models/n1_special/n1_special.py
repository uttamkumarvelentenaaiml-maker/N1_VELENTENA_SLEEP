# =====================================================
# FILE: models/n1_special/n1_special.py
# STRONG N1-SPECIAL MODEL FOR RESEARCH
# Multi-Scale CNN + BiLSTM + Attention
# =====================================================

import torch
import torch.nn as nn


class StrongN1Net(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # -------------------------------------------------
        # MULTI-SCALE CNN BRANCHES
        # Input: [B, 4, 3000]
        # -------------------------------------------------
        self.branch3 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.branch7 = nn.Conv1d(4, 32, kernel_size=7, padding=3)
        self.branch15 = nn.Conv1d(4, 32, kernel_size=15, padding=7)

        self.bn = nn.BatchNorm1d(96)
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(2)

        # -------------------------------------------------
        # FEATURE REFINEMENT
        # -------------------------------------------------
        self.conv_refine = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # -------------------------------------------------
        # BiLSTM
        # -------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # -------------------------------------------------
        # ATTENTION
        # -------------------------------------------------
        self.attn = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # -------------------------------------------------
        # CLASSIFIER
        # -------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x = [B, 4, 3000]

        # Multi-scale features
        b1 = self.branch3(x)
        b2 = self.branch7(x)
        b3 = self.branch15(x)

        x = torch.cat([b1, b2, b3], dim=1)   # [B, 96, T]
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        # Refine features
        x = self.conv_refine(x)              # [B, 128, T]

        # Prepare for LSTM
        x = x.permute(0, 2, 1)              # [B, T, 128]

        # BiLSTM
        x, _ = self.lstm(x)                 # [B, T, 128]

        # Attention weights
        w = self.attn(x)                    # [B, T, 1]
        w = torch.softmax(w, dim=1)

        # Weighted context vector
        x = (x * w).sum(dim=1)              # [B, 128]

        # Final classification
        out = self.fc(x)

        return out