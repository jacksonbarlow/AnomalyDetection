# data/preprocess.py

import numpy as np
import pandas as pd
import pywt
import joblib
from sklearn.preprocessing import StandardScaler
from config import CONFIG
from utils.visualisation import (
    plot_feature_smoothing,
    plot_velocity_reconstruction,
    plot_clipping_effect,
    plot_standardisation_check
)
import os

def load_all_trajectory_data(root_dir="trajectory_data"):
    all_data = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".csv") and "trajectories" in file.lower():
                full_path = os.path.join(dirpath, file)
                df = pd.read_csv(full_path)
                
                # Add a source identifier
                df["source_tag"] = os.path.relpath(full_path, root_dir).replace("/", "_").replace(".csv", "")
                all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

### Helper Functions ###
def apply_cwt(series, wavelet='db4', level=1):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def apply_sema(series, alpha=0.2):
    return series.ewm(alpha=alpha, adjust=False).mean()

def compute_velocity_acceleration(df):
    df = df.sort_values(by=["Vehicle_ID", "Frame_ID"])
    dt = 0.1
    df["v_Vel"] = df.groupby("Vehicle_ID")["Local_Y"].diff() / dt
    df["v_Acc"] = df.groupby("Vehicle_ID")["v_Vel"].diff() / dt
    df["v_Vel"] = df["v_Vel"].clip(lower=0)
    df = df.dropna(subset=["v_Vel", "v_Acc"]).reset_index(drop=True)

    assert not df[["v_Vel", "v_Acc"]].isnull().any().any(), "NaNs present in velocity or acceleration"
    return df

def get_static_features(row):
    # v_Class: 1 = Motorcycle, 2 = Car, 3 = Truck
    class_id = int(row.get("v_Class", 2))  # Default to car

    if class_id == 1:
        vehicle_type = 1  # motorcycle
    elif class_id == 3:
        vehicle_type = 2  # truck
    else:
        vehicle_type = 0  # car

    vehicle_length = row.get("v_Length", 0.0)
    vehicle_width = row.get("v_Width", 0.0)

    return [vehicle_type, vehicle_length, vehicle_width]

def compute_headways(df):
    df = df.sort_values(by=["Vehicle_ID", "Frame_ID"]).copy()
    df["Space_Headway"] = np.nan
    df["Time_Headway"] = np.nan

    grouped = df.groupby("Frame_ID")
    for _, frame in grouped:
        frame = frame.sort_values(by="Local_Y", ascending=False)
        for i in range(len(frame) - 1):
            ego_idx = frame.index[i + 1]
            leader_idx = frame.index[i]

            space = frame.loc[leader_idx, "Local_Y"] - frame.loc[ego_idx, "Local_Y"]
            vel = frame.loc[ego_idx, "v_Vel"]

            df.at[ego_idx, "Space_Headway"] = max(space, 0.0)
            df.at[ego_idx, "Time_Headway"] = space / vel if vel > 0 else np.nan

    return df

def select_diagnostic_vehicle(df,
                               min_valid_headway_fraction=0.3,
                               max_time_headway=20.0,
                               min_velocity_std=2.0,
                               min_acceleration_std=1.0,
                               min_trajectory_length=200):
    candidate_scores = []
    for vehicle_id, group in df.groupby("Vehicle_ID"):
        if len(group) < min_trajectory_length:
            continue

        valid_headway = (
            (group["Time_Headway"] > 0) & (group["Time_Headway"] < max_time_headway) &
            (group["Space_Headway"] > 0)
        )
        headway_fraction = valid_headway.mean()
        if headway_fraction < min_valid_headway_fraction:
            continue

        v_std = group["v_Vel"].std()
        a_std = group["v_Acc"].std()
        if v_std < min_velocity_std or a_std < min_acceleration_std:
            continue

        score = headway_fraction * (v_std + a_std)
        candidate_scores.append((vehicle_id, score))

    if not candidate_scores:
        raise ValueError("No vehicle meets all diagnostic criteria for headway + dynamics.")

    return max(candidate_scores, key=lambda x: x[1])[0]

### Main Function ###
def compute_headways(df):
    df = df.sort_values(by=["Vehicle_ID", "Frame_ID"]).copy()
    df["Space_Headway"] = np.nan
    df["Time_Headway"] = np.nan

    grouped = df.groupby("Frame_ID")
    for _, frame in grouped:
        frame = frame.sort_values(by="Local_Y", ascending=False)
        for i in range(len(frame) - 1):
            ego_idx = frame.index[i + 1]
            leader_idx = frame.index[i]

            space = frame.loc[leader_idx, "Local_Y"] - frame.loc[ego_idx, "Local_Y"]
            vel = frame.loc[ego_idx, "v_Vel"]

            df.at[ego_idx, "Space_Headway"] = max(space, 0.0)
            df.at[ego_idx, "Time_Headway"] = space / vel if vel > 0 else np.nan

    return df


def preprocess_and_clip(df, clip_percentile=(1, 99), sequence_cache_path=None):
    save_dir = CONFIG["PREPROCESSING_PLOTS_DIR"]
    os.makedirs(save_dir, exist_ok=True)

    df = df.copy().sort_values(by=["Vehicle_ID", "Frame_ID"])
    print("[INFO] Selecting diagnostic vehicle for plotting...")
    vehicle_id = select_diagnostic_vehicle(df)
    vehicle_df = df[df["Vehicle_ID"] == vehicle_id][["Frame_ID", "Local_Y"]].copy()

    # Backup raw Local_Y before any smoothing
    df_original = df[["Vehicle_ID", "Frame_ID", "Local_Y", "v_Vel", "v_Acc"]].copy()
    df_original.rename(columns={"v_Vel": "v_Vel_NGSIM", "v_Acc": "v_Acc_NGSIM"}, inplace=True)

    # === Smooth Local_Y before computing velocity ===
    try:
        df["Local_Y"] = apply_cwt(df["Local_Y"].values)
    except Exception:
        pass
    df["Local_Y"] = apply_sema(df["Local_Y"])

    # === Compute v_Vel and v_Acc ===
    df = compute_velocity_acceleration(df)

    # === Clip to physically realistic ranges ===
    df = df[(df["v_Vel"] >= 0) & (df["v_Vel"] <= 70)]
    df = df[(df["v_Acc"] >= -10) & (df["v_Acc"] <= 10)]

    # === Compute headways ===
    df = compute_headways(df)


    df = df[(df["Time_Headway"] > 0) & (df["Time_Headway"] < 100)]
    df = df[(df["Space_Headway"] > 0) & (df["Space_Headway"] < 500)]

     # === Backup raw before smoothing ===
    raw_reconstructed = df[["v_Vel", "v_Acc", "Space_Headway", "Time_Headway"]].copy(deep=True)
    if "Space_Headway" in raw_reconstructed.columns:
        raw_reconstructed["Space_Headway"] = df["Space_Headway"]
    if "Time_Headway" in raw_reconstructed.columns:
        raw_reconstructed["Time_Headway"] = df["Time_Headway"]


    # === Plot velocity and acceleration reconstruction ===
    print("[INFO] Plotting velocity & acceleration reconstruction...")
    vehicle_post = df[df["Vehicle_ID"] == vehicle_id]
    vehicle_orig = df_original[df_original["Vehicle_ID"] == vehicle_id]
    diagnostic_df = vehicle_post.merge(
        vehicle_orig[["Vehicle_ID", "Frame_ID", "v_Vel_NGSIM", "v_Acc_NGSIM", "Local_Y"]],
        on=["Vehicle_ID", "Frame_ID"], how="left"
    )
    diagnostic_df.rename(columns={"Local_Y_y": "Local_Y"}, inplace=True)
    diagnostic_df.drop(columns=["Local_Y_x", "Local_Y_y"], inplace=True, errors="ignore")
    print("[DEBUG] Diagnostic columns:", diagnostic_df.columns.tolist())


    plot_velocity_reconstruction(
        df=diagnostic_df,
        vehicle_id=vehicle_id,
        save_dir=save_dir
    )


    # === Smooth all features ===
    for col in CONFIG["FEATURES"]:
        series = df[col]
        if series.nunique() <= 1:
            continue  # Skip degenerate features

        try:
            df[col] = apply_cwt(series.values)
        except Exception:
            pass
        df[col] = apply_sema(df[col])


    smoothed = df[CONFIG["FEATURES"]].copy()


    print("[INFO] Plotting smoothed vs. raw features...")
    plot_feature_smoothing(
        raw_features=raw_reconstructed.loc[vehicle_post.index],
        smoothed_features=smoothed.loc[vehicle_post.index],
        vehicle_id=vehicle_id,
        save_path=os.path.join(save_dir, f"smoothing_plot_vehicle_{vehicle_id}.png")
    )

    # === Log transform headways ===
    df["Time_Headway"] = np.log1p(df["Time_Headway"])
    df["Space_Headway"] = np.log1p(df["Space_Headway"])

    # === Clipping outliers ===
    for col in CONFIG["FEATURES"]:
        raw_col = smoothed[col].copy()
        lower, upper = np.percentile(raw_col, clip_percentile)
        clipped_col = np.clip(raw_col, lower, upper)

        print(f"[INFO] Plotting clipping effect for {col}...")
        plot_clipping_effect(
            raw_col=raw_col,
            clipped_col=clipped_col,
            feature_name=col,
            save_dir=save_dir
        )
        df[col] = clipped_col

    # === Final cleanup and scaling ===
    df_cleaned = df.dropna(subset=CONFIG["FEATURES"]).reset_index(drop=True)
    features_only = df_cleaned[CONFIG["FEATURES"]]

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(features_only)
    df_scaled = pd.DataFrame(scaled_array, columns=CONFIG["FEATURES"])
    df_scaled["Vehicle_ID"] = df_cleaned["Vehicle_ID"].values
    df_scaled["Frame_ID"] = df_cleaned["Frame_ID"].values

    print("[INFO] Plotting standardisation checks...")
    for col in CONFIG["FEATURES"]:
        plot_standardisation_check(
            scaled_df=df_scaled,
            feature_name=col,
            save_dir=save_dir
        )

    if sequence_cache_path:
        os.makedirs(os.path.dirname(sequence_cache_path), exist_ok=True)
        df_scaled.to_parquet(sequence_cache_path, index=False)
        print(f"[INFO] Scaled features saved to {sequence_cache_path}")

    joblib.dump(scaler, CONFIG["SCALER_PATH"])
    print(f"[INFO] Scaler saved to {CONFIG['SCALER_PATH']}")

    return df_scaled, scaler


def sequences_to_deltas(sequences):
    return sequences[:, 1:, :] - sequences[:, :-1, :]
