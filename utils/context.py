from utils.surrounding_features import get_surrounding_features
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler

def precompute_context_features(df_raw, frame_lookup):
    context_cache = {}

    total = sum(len(frame_df) for frame_df in frame_lookup.values())
    print(f"[INFO] Precomputing context features for {total} vehicle-frames...")

    context_records = []

    with tqdm(total=total, desc="Building context cache") as pbar:
        for _, frame_df in frame_lookup.items():
            for _, row in frame_df.iterrows():
                key = (int(row['Vehicle_ID']), int(row['Frame_ID']))
                context = get_surrounding_features(frame_df, row)
                context_records.append({
                    "Vehicle_ID": key[0],
                    "Frame_ID": key[1],
                    **{f"context_{i}": val for i, val in enumerate(context)}
                })
                pbar.update(1)

    context_df = pd.DataFrame(context_records)

    # Standardise context features
    context_cols = [col for col in context_df.columns if col.startswith("context_")]
    context_df[context_cols] = StandardScaler().fit_transform(context_df[context_cols])

    print("[DEBUG] Example context feature rows:")
    print(context_df.head(10).T)

    return context_df