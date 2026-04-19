import torch
import torch.nn as nn

class SleepCNNBiLSTM(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Input shape:
        # (batch, channels, time)

        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T)

        x = self.cnn(x)          # (B, 128, T')

        x = x.permute(0, 2, 1)  # (B, T', 128)

        x, _ = self.lstm(x)     # (B, T', 256)

        x = x[:, -1, :]         # last timestep

        x = self.fc(x)          # (B, 5)

        return x