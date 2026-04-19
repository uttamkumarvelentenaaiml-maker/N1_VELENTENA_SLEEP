# =====================================================
# IMPORT LIBRARIES
# =====================================================
import torch
import torch.nn as nn


# =====================================================
# ATTENTION MODEL
# CNN + BiLSTM + Attention
# =====================================================
class SleepAttentionNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # -------------------------------------------------
        # CNN FEATURE EXTRACTOR
        # Input: [B, 4, 3000]
        # -------------------------------------------------
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # -------------------------------------------------
        # BiLSTM
        # -------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # -------------------------------------------------
        # ATTENTION LAYER
        # -------------------------------------------------
        self.attn = nn.Linear(128, 1)

        # -------------------------------------------------
        # CLASSIFIER
        # -------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x = [B, 4, 3000]

        # CNN
        x = self.cnn(x)             # [B, 64, T]

        # LSTM expects [B, T, F]
        x = x.permute(0, 2, 1)

        # BiLSTM
        x, _ = self.lstm(x)        # [B, T, 128]

        # Attention scores
        w = self.attn(x)           # [B, T, 1]
        w = torch.softmax(w, dim=1)

        # Weighted sum
        x = (x * w).sum(dim=1)     # [B, 128]

        # Classifier
        out = self.fc(x)

        return out