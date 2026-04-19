import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay
)

from models.baseline.baseline_cnn_bilstm import SleepCNNBiLSTM

# ==================================
# CONFIG
# ==================================
DATA_PATH = "data/processed_sleep_edf"
RESULT_DIR = "results/baseline"
MODEL_PATH = os.path.join(RESULT_DIR, "best_baseline.pth")

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

os.makedirs(RESULT_DIR, exist_ok=True)

print("Using Device:", DEVICE)

# ==================================
# DATASET
# ==================================
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

# ==================================
# LOAD TEST DATA
# ==================================
test_ds = SleepDataset("test_split.csv")

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Test Samples:", len(test_ds))

# ==================================
# LOAD MODEL
# ==================================
model = SleepCNNBiLSTM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==================================
# EVALUATE
# ==================================
all_preds = []
all_true = []

print("\nRunning Evaluation...\n")

with torch.no_grad():
    for x, y in tqdm(test_loader):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        out = model(x)
        pred = out.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_true.extend(y.cpu().numpy())

# ==================================
# METRICS
# ==================================
report_txt = classification_report(
    all_true,
    all_preds,
    target_names=LABEL_NAMES,
    digits=4
)

report_dict = classification_report(
    all_true,
    all_preds,
    target_names=LABEL_NAMES,
    digits=4,
    output_dict=True
)

cm = confusion_matrix(all_true, all_preds)

acc = accuracy_score(all_true, all_preds)
macro_f1 = f1_score(all_true, all_preds, average="macro")
weighted_f1 = f1_score(all_true, all_preds, average="weighted")

n1_precision = report_dict["N1"]["precision"]
n1_recall = report_dict["N1"]["recall"]
n1_f1 = report_dict["N1"]["f1-score"]

# ==================================
# SAVE FILES
# ==================================

# Full classification report
with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w") as f:
    f.write(report_txt)

# Summary metrics
summary = pd.DataFrame([{
    "model": "baseline",
    "accuracy": round(acc, 4),
    "macro_f1": round(macro_f1, 4),
    "weighted_f1": round(weighted_f1, 4),
    "n1_precision": round(n1_precision, 4),
    "n1_recall": round(n1_recall, 4),
    "n1_f1": round(n1_f1, 4)
}])

summary.to_csv(
    os.path.join(RESULT_DIR, "summary_metrics.csv"),
    index=False
)

# Per-class metrics
rows = []
for label in LABEL_NAMES:
    rows.append({
        "class": label,
        "precision": round(report_dict[label]["precision"], 4),
        "recall": round(report_dict[label]["recall"], 4),
        "f1_score": round(report_dict[label]["f1-score"], 4),
        "support": int(report_dict[label]["support"])
    })

pd.DataFrame(rows).to_csv(
    os.path.join(RESULT_DIR, "per_class_metrics.csv"),
    index=False
)

# Confusion matrix CSV
cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
cm_df.to_csv(os.path.join(RESULT_DIR, "confusion_matrix.csv"))

# Confusion matrix image
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Baseline Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# N1 specific report
with open(os.path.join(RESULT_DIR, "n1_focus_report.txt"), "w") as f:
    f.write("N1 (Most Important Class)\n")
    f.write("=========================\n\n")
    f.write(f"Precision : {n1_precision*100:.2f}%\n")
    f.write(f"Recall    : {n1_recall*100:.2f}%\n")
    f.write(f"F1-score  : {n1_f1*100:.2f}%\n\n")

    f.write("Core Research Insight\n")
    f.write("---------------------\n")
    f.write("Baseline model performs strongly overall,\n")
    f.write("but N1 detection remains challenging.\n")
    f.write("This class should be the main improvement target.\n")

# Human readable summary
with open(os.path.join(RESULT_DIR, "results_readme.txt"), "w") as f:
    f.write("BASELINE MODEL RESULTS\n")
    f.write("======================\n\n")
    f.write(f"Accuracy     : {acc*100:.2f}%\n")
    f.write(f"Macro F1     : {macro_f1*100:.2f}%\n")
    f.write(f"Weighted F1  : {weighted_f1*100:.2f}%\n\n")

    f.write("N1 PERFORMANCE\n")
    f.write("----------------\n")
    f.write(f"Precision     : {n1_precision*100:.2f}%\n")
    f.write(f"Recall        : {n1_recall*100:.2f}%\n")
    f.write(f"F1-score      : {n1_f1*100:.2f}%\n")

# ==================================
# TERMINAL OUTPUT
# ==================================
print("\n===== CLASSIFICATION REPORT =====\n")
print(report_txt)

print("\nSaved Files in results/baseline/")
print("- classification_report.txt")
print("- summary_metrics.csv")
print("- per_class_metrics.csv")
print("- confusion_matrix.csv")
print("- confusion_matrix.png")
print("- n1_focus_report.txt")
print("- results_readme.txt")