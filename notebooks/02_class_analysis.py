import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed_sleep_edf"

LABEL_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

# =========================
# FIND ALL LABEL FILES
# =========================
y_files = sorted(glob.glob(os.path.join(DATA_PATH, "*_y.npy")))

all_labels = []

# =========================
# LOAD ALL LABELS
# =========================
for file in y_files:
    y = np.load(file)
    all_labels.extend(y.tolist())

all_labels = np.array(all_labels)

# =========================
# COUNT CLASSES
# =========================
counts = Counter(all_labels)
total = len(all_labels)

rows = []
print("\n===== CLASS DISTRIBUTION =====\n")

for cls_id in sorted(LABEL_NAMES.keys()):
    count = counts.get(cls_id, 0)
    pct = 100 * count / total
    rows.append([cls_id, LABEL_NAMES[cls_id], count, round(pct, 2)])

    print(f"{LABEL_NAMES[cls_id]:<5} : {count:>8} samples ({pct:.2f}%)")

# =========================
# SAVE TABLE
# =========================
df = pd.DataFrame(rows, columns=["id", "stage", "count", "percent"])
df.to_csv(os.path.join(DATA_PATH, "class_distribution.csv"), index=False)

# =========================
# PLOT
# =========================
stages = df["stage"]
values = df["count"]

plt.figure(figsize=(10,6))
bars = plt.bar(stages, values)

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h, f"{int(h)}",
             ha="center", va="bottom", fontsize=10)

plt.title("Sleep Stage Distribution")
plt.xlabel("Class")
plt.ylabel("Samples")
plt.tight_layout()
plt.show()

print("\nSaved: class_distribution.csv")