import os
import glob
import gc
import numpy as np
import pandas as pd
import mne
import warnings

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
DATA_PATH = "data/sleep-edfx/sleep-cassette"
SAVE_PATH = "data/processed_sleep_edf"
os.makedirs(SAVE_PATH, exist_ok=True)

CHANNELS = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal",
    "EMG submental"
]

LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4
}

# =========================
# FILES
# =========================
psg_files = sorted(glob.glob(os.path.join(DATA_PATH, "*PSG.edf")))
hyp_files = sorted(glob.glob(os.path.join(DATA_PATH, "*Hypnogram.edf")))

meta_rows = []

# =========================
# LOOP SUBJECTS
# =========================
for idx, (psg_file, hyp_file) in enumerate(zip(psg_files, hyp_files)):
    name = os.path.basename(psg_file)
    print(f"\n[{idx+1}/{len(psg_files)}] {name}")

    try:
        raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)

        available = [ch for ch in CHANNELS if ch in raw.ch_names]
        if len(available) == 0:
            print("Skipped: no channels")
            continue

        raw.pick(available)

        annotations = mne.read_annotations(hyp_file)
        raw.set_annotations(annotations)

        present_labels = set(annotations.description)
        current_event_id = {
            k: v for k, v in LABEL_MAP.items()
            if k in present_labels
        }

        if len(current_event_id) == 0:
            print("Skipped: no labels")
            continue

        events, _ = mne.events_from_annotations(
            raw,
            event_id=current_event_id,
            chunk_duration=30.0,
            verbose=False
        )

        if len(events) == 0:
            print("Skipped: no events")
            continue

        epochs = mne.Epochs(
            raw,
            events,
            event_id=current_event_id,
            tmin=0,
            tmax=30 - 1/raw.info["sfreq"],
            baseline=None,
            preload=True,
            verbose=False
        )

        X = epochs.get_data().astype(np.float32)
        y = epochs.events[:, -1].astype(np.int64)

        # Save subject files
        x_name = f"subject_{idx:03d}_X.npy"
        y_name = f"subject_{idx:03d}_y.npy"

        np.save(os.path.join(SAVE_PATH, x_name), X)
        np.save(os.path.join(SAVE_PATH, y_name), y)

        meta_rows.append({
            "subject_id": idx,
            "file_name": name,
            "epochs": len(y),
            "channels": X.shape[1]
        })

        print("Saved:", x_name, "| Samples:", len(y))

        # Free memory
        del raw, epochs, X, y
        gc.collect()

    except Exception as e:
        print("Skipped:", e)

# =========================
# SAVE METADATA
# =========================
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(os.path.join(SAVE_PATH, "metadata.csv"), index=False)

print("\n========================")
print("V3 PROCESS COMPLETE")
print("========================")
print("Subjects Saved:", len(meta_rows))
print("Saved Path:", SAVE_PATH)
print("Metadata: metadata.csv")