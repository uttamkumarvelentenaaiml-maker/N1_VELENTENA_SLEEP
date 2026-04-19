# =====================================================
# FILE: train/n1_special_train/n1_v2_train.py
# N1 SPECIAL V2 TRAINING (1 EPOCH)
# Saves to results/n1_special/v2/
# =====================================================

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.n1_special.n1_v2 import N1V2Net

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "data/processed_sleep_edf"
RESULT_DIR = "results/n1_special/v2"

BATCH_SIZE = 32
EPOCHS = 1
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULT_DIR, exist_ok=True)

print("Using Device:", DEVICE)

# =====================================================
# DATASET
# =====================================================
class SleepDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(os.path.join(DATA_PATH, csv_file))
        self.samples = []

        for sid in df["subject_id"]:
            x_path = os.path.join(DATA_PATH, f"subject_{sid:03d}_X.npy")
            y_path = os.path.join(DATA_PATH, f"subject_{sid:03d}_y.npy")

            y = np.load(y_path)

            for i in range(len(y)):
                self.samples.append((x_path, y_path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_path, y_path, i = self.samples[idx]

        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")

        x = X[i].astype(np.float32)
        label = int(y[i])

        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-8
        x = (x - mean) / std

        return torch.tensor(x), torch.tensor(label)

# =====================================================
# LOAD DATA
# =====================================================
train_ds = SleepDataset("train_split.csv")
val_ds   = SleepDataset("val_split.csv")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Train Samples:", len(train_ds))
print("Val Samples  :", len(val_ds))

# =====================================================
# MODEL
# =====================================================
model = N1V2Net().to(DEVICE)

weights = torch.tensor([1.0, 3.5, 1.0, 1.2, 1.2], device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

# =====================================================
# TRAIN LOOP
# =====================================================
for epoch in range(EPOCHS):

    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(train_loader):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            out = model(x)
            pred = out.argmax(dim=1)

            val_correct += (pred == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")
    print(f"Val Acc    : {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            os.path.join(RESULT_DIR, "best_n1_v2.pth")
        )

        print("Best model saved!")

print("\nTraining Complete.")