import os
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed_sleep_edf"
SEED = 42

# =========================
# LOAD METADATA
# =========================
meta = pd.read_csv(os.path.join(DATA_PATH, "metadata.csv"))

# unique subjects
subjects = meta["subject_id"].unique()

# shuffle reproducibly
rng = np.random.RandomState(SEED)
rng.shuffle(subjects)

n = len(subjects)

# split ratios
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

train_subjects = subjects[:train_end]
val_subjects   = subjects[train_end:val_end]
test_subjects  = subjects[val_end:]

# build dataframes
train_df = meta[meta["subject_id"].isin(train_subjects)]
val_df   = meta[meta["subject_id"].isin(val_subjects)]
test_df  = meta[meta["subject_id"].isin(test_subjects)]

# save
train_df.to_csv(os.path.join(DATA_PATH, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(DATA_PATH, "val_split.csv"), index=False)
test_df.to_csv(os.path.join(DATA_PATH, "test_split.csv"), index=False)

# print summary
print("===== SUBJECT-WISE SPLIT =====")
print("Total Subjects :", n)
print("Train Subjects :", len(train_subjects))
print("Val Subjects   :", len(val_subjects))
print("Test Subjects  :", len(test_subjects))
print()
print("Saved Files:")
print("train_split.csv")
print("val_split.csv")
print("test_split.csv")