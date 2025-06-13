# data/load.py

import os
import pandas as pd
import numpy as np
import pickle
from config import CONFIG
from utils.context import precompute_context_features

def load_raw_dataframe(root_dir="trajectory_data"):
    """Loads and merges all NGSIM trajectory CSVs under root_dir."""
    all_data = []
    print(f"[INFO] Walking directory: {root_dir}")

    for dirpath, _, filenames in os.walk(root_dir):
        print(f"[DEBUG] Checking: {dirpath}")
        for file in filenames:
            print(f"[DEBUG] Found file: {file}")
            if file.endswith(".csv") and "trajector" in file.lower():
                full_path = os.path.join(dirpath, file)
                print(f"[INFO] Loading CSV: {full_path}")
                df = pd.read_csv(full_path)
                df["source_tag"] = os.path.relpath(full_path, root_dir).replace("\\", "_").replace("/", "_").replace(".csv", "")
                all_data.append(df)

    if not all_data:
        raise FileNotFoundError(f"[ERROR] No CSV files found in: {root_dir}")

    print(f"[INFO] Loaded {len(all_data)} CSV file(s) from {root_dir}")
    return pd.concat(all_data, ignore_index=True)

def load_context_cache():
    cache_path = CONFIG["CONTEXT_CACHE_PATH"]
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print(f"[INFO] Loaded cached context features from: {cache_path}")
            context_df = pickle.load(f)
    else:
        print("[INFO] Context cache not found. Computing from scratch...")
        df = load_raw_dataframe()
        frame_lookup = {f: fdf for f, fdf in df.groupby("Frame_ID")}
        context_df = precompute_context_features(df, frame_lookup)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(context_df, f)
        print(f"[INFO] Saved context cache to: {cache_path}")

    context_cache = {
        (int(row["Vehicle_ID"]), int(row["Frame_ID"])): row.drop(["Vehicle_ID", "Frame_ID"]).values
        for _, row in context_df.iterrows()
    }
    return context_cache

def load_sequences(path, return_meta=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Sequence file not found: {path}")
    print(f"[INFO] Loading sequences from: {path}")
    data = np.load(path, allow_pickle=True)

    # Check for legacy format
    if "seq" in data and "tgt" in data:
        sequences = data["seq"]
        targets = data["tgt"]
    elif "seq_path" in data and "tgt_path" in data:
        seq_path = data["seq_path"].item()
        tgt_path = data["tgt_path"].item()
        print(f"[INFO] Using memory-mapped arrays:\n  - seq: {seq_path}\n  - tgt: {tgt_path}")
        sequences = np.load(seq_path, mmap_mode="r")
        targets = np.load(tgt_path, mmap_mode="r")
    else:
        raise ValueError("[ERROR] Unrecognized sequence file format.")

    if return_meta:
        meta_vids = data["meta_vids"]
        meta_fids = data["meta_fids"]
        meta_info = list(zip(meta_vids, meta_fids))
        return sequences, targets, meta_info

    return sequences, targets
