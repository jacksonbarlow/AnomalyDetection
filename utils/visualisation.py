# utils/visualisation.py

### Library Imports and Setup ###
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from config import CONFIG
import pandas as pd

### Main Function ###
def plot_top_anomalies(indices, sequences, feature_names, n_samples=3):
    for i in indices[:n_samples]:
        seq = sequences[i, :, :4]  # Only ego features
        for f, feat in enumerate(feature_names):
            plt.figure(figsize=(6, 3))
            plt.plot(seq[:, f], label="True")
            plt.title(f"Anomaly {i} - Feature: {feat}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def plot_reconstruction_comparison(true_seq, recon_seq, feature_names):
    for f, feat in enumerate(feature_names):
        plt.figure(figsize=(6, 3))
        plt.plot(true_seq[:, f], label="True")
        plt.plot(recon_seq[:, f], linestyle="--", label="Reconstructed")
        plt.title(f"Reconstruction - Feature: {feat}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_reconstruction_errors(error_matrix, feature_names):
    for i, feat in enumerate(feature_names):
        plt.figure(figsize=(6, 3))
        plt.hist(error_matrix[:, i], bins=50, color="skyblue", edgecolor="black")
        plt.title(f"Reconstruction Error: {feat}")
        plt.xlabel("MSE")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def save_reconstruction_plot(true_seq, recon_seq, feature_names, epoch, save_path="reconstruction_samples"):
    import os
    for f, feat in enumerate(feature_names):
        plt.figure(figsize=(6, 3))
        plt.plot(true_seq[:, f], label="True")
        plt.plot(recon_seq[:, f], linestyle="--", label="Reconstructed")
        plt.title(f"Epoch {epoch} - Feature: {feat}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = os.path.join(save_path, f"epoch_{epoch}_{feat}.png")
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure full directory exists
        plt.savefig(filename)
        plt.close()


import matplotlib.pyplot as plt
import os

def plot_velocity_reconstruction(df, vehicle_id, save_dir):
    """
    Plots Local_Y and compares NGSIM vs. computed velocity and acceleration with distinct colors.
    Assumes df contains:
    - Frame_ID, Local_Y
    - v_Vel, v_Vel_NGSIM (optional)
    - v_Acc, v_Acc_NGSIM (optional)
    """

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Position trace
    axs[0].plot(df["Frame_ID"], df["Local_Y"], color="black", label="Raw Local_Y")
    axs[0].set_ylabel("Local_Y")
    axs[0].legend()
    axs[0].grid(True)

    # Velocity
    axs[1].plot(df["Frame_ID"], df["v_Vel"], label="Computed v_Vel", color="blue", linewidth=2)
    if "v_Vel_NGSIM" in df.columns:
        axs[1].plot(df["Frame_ID"], df["v_Vel_NGSIM"], label="Original v_Vel (NGSIM)", color="gray", linestyle="--")
    axs[1].set_ylabel("Velocity")
    axs[1].legend()
    axs[1].grid(True)

    # Acceleration
    axs[2].plot(df["Frame_ID"], df["v_Acc"], label="Computed v_Acc", color="orange", linewidth=2)
    if "v_Acc_NGSIM" in df.columns:
        axs[2].plot(df["Frame_ID"], df["v_Acc_NGSIM"], label="Original v_Acc (NGSIM)", color="gray", linestyle="--")
    axs[2].set_ylabel("Acceleration")
    axs[2].set_xlabel("Frame Index")
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle(f"Velocity and Acceleration Reconstruction vs. Original (Vehicle {vehicle_id})")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"velocity_reconstruction_vehicle_{vehicle_id}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_smoothing(raw_features, smoothed_features, vehicle_id=None, save_path=None, inset_range=(100, 150)):
    import matplotlib.pyplot as plt

    features = ['v_Vel', 'v_Acc', 'Space_Headway', 'Time_Headway']
    time = list(range(len(raw_features[features[0]])))

    fig, axs = plt.subplots(len(features), 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Raw vs Smoothed Features" + (f" for Vehicle {vehicle_id}" if vehicle_id else ""), fontsize=16)

    for i, feature in enumerate(features):
        raw = raw_features[feature]
        smoothed = smoothed_features[feature]

        # === DEBUG print ===
        print(f"\n[DEBUG] Feature: {feature}")
        print(f"Raw min/max: {raw.min():.4f}, {raw.max():.4f}")
        print(f"Smoothed min/max: {smoothed.min():.4f}, {smoothed.max():.4f}")
        print(f"Raw first few: {raw.iloc[:5].values}")
        print(f"Smoothed first few: {smoothed.iloc[:5].values}")

        axs[i].plot(time, raw, label='Raw', alpha=0.6, color='tab:blue')
        axs[i].plot(time, smoothed, label='Smoothed', linewidth=2, color='tab:orange')
        axs[i].set_ylabel(feature)
        axs[i].legend()
        axs[i].grid(True)

        # Add zoomed-in inset for the first 3 features only
        if i < 3:
            ax_inset = axs[i].inset_axes([0.65, 0.45, 0.3, 0.4])
            ax_inset.plot(
                time[inset_range[0]:inset_range[1]],
                raw.iloc[inset_range[0]:inset_range[1]],
                alpha=0.6, color='tab:blue'
            )
            ax_inset.plot(
                time[inset_range[0]:inset_range[1]],
                smoothed.iloc[inset_range[0]:inset_range[1]],
                linewidth=2, color='tab:orange'
            )
            ax_inset.set_title("Zoom: frames {}–{}".format(*inset_range), fontsize=8)
            ax_inset.tick_params(labelsize=6)
            axs[i].indicate_inset_zoom(ax_inset, edgecolor="gray")

    axs[-1].set_xlabel('Frame')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()


def plot_clipping_effect(raw_col, clipped_col, feature_name, save_dir):
    """
    Plots histogram of raw vs clipped values, with 1st and 99th percentile markers.
    Applies log y-axis for long-tailed features and optionally caps axis for clarity.
    """
    plt.figure(figsize=(10, 5))
    
    # Compute clipping thresholds
    p1, p99 = np.percentile(raw_col, [1, 99])

    # Plot histograms
    plt.hist(raw_col, bins=100, alpha=0.5, label="Raw", color="gray")
    plt.hist(clipped_col, bins=100, alpha=0.7, label="Clipped", color="blue")

    # Percentile lines
    plt.axvline(p1, color='red', linestyle='--', label="1st percentile")
    plt.axvline(p99, color='red', linestyle='--', label="99th percentile")

    # Labels
    plt.title(f"Clipping Effect on {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    # Log scale if long-tailed or high skew
    if feature_name in ["Time_Headway", "Space_Headway", "v_Acc"]:
        plt.yscale("log")
        plt.ylim(bottom=1)  # Avoid log(0)

    # Optional cap for linear features
    elif feature_name == "v_Vel":
        plt.ylim(top=100000)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"clipping_effect_{feature_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_standardisation_check(scaled_df, feature_name, save_dir):
    data = scaled_df[feature_name]
    sns.histplot(data, bins=100, stat="density", label="Standardised Feature", kde=True, color='skyblue')

    # Overlay standard normal curve
    x = np.linspace(-4, 4, 200)
    plt.plot(x, norm.pdf(x), label="N(0,1)", color="red", linestyle="--")

    plt.title(f"Standardisation Check: {feature_name}")
    plt.xlabel("Z-score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/standardisation_check_{feature_name}.png", dpi=300)
    plt.close()

def plot_delta_sequence_heatmap(delta_sequence, sample_id, feature_names, save_dir):
    plt.figure(figsize=(12, 6))
    sns.heatmap(delta_sequence.T, cmap="coolwarm", xticklabels=20, yticklabels=feature_names)
    plt.title(f"Delta Sequence Heatmap (Sample {sample_id})")
    plt.xlabel("Time Step")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/delta_sequence_heatmap_sample_{sample_id}.png", dpi=300)
    plt.close()

def plot_raw_vs_preprocessed_features(df_raw, df_scaled, vehicle_ids, feature_names, save_path, scaler):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    num_features = len(feature_names)
    fig, axs = plt.subplots(num_features, len(vehicle_ids), figsize=(14, 3 * num_features), sharex="col")

    if len(vehicle_ids) == 1:
        axs = axs.reshape(-1, 1)

    log_transformed_features = ["Space_Headway", "Time_Headway"]

    for col_idx, vehicle_id in enumerate(vehicle_ids):
        raw = df_raw[df_raw["Vehicle_ID"] == vehicle_id]
        scaled = df_scaled[df_scaled["Vehicle_ID"] == vehicle_id]

        # Align by Frame_ID and Vehicle_ID
        merged = pd.merge(
            raw[["Frame_ID", "Vehicle_ID"] + feature_names],
            scaled[["Frame_ID", "Vehicle_ID"] + feature_names],
            on=["Vehicle_ID", "Frame_ID"],
            suffixes=("_raw", "_scaled")
        )

        if merged.empty:
            print(f"[WARNING] No aligned data found for vehicle {vehicle_id}. Skipping.")
            continue

        # Inverse transform scaled features (already in log-space for headways)
        scaled_values = merged[[f + "_scaled" for f in feature_names]].values
        processed_inv = pd.DataFrame(scaler.inverse_transform(scaled_values), columns=feature_names)

        frame_ids = merged["Frame_ID"]

        print(f"[DEBUG] Vehicle {vehicle_id} — merged shape: {merged.shape}")
        print(f"[DEBUG] Frame_IDs: {frame_ids.values[:5]}")

        for i, feature in enumerate(feature_names):
            raw_vals = merged[f"{feature}_raw"].values
            inv_vals = processed_inv[feature].values

            if feature in log_transformed_features:
                print(f"[DEBUG] Log-transforming raw {feature} for comparison")
                raw_vals = np.log1p(raw_vals)

            print(f"[DEBUG] Raw {feature}[:5]: {raw_vals[:5]}")
            print(f"[DEBUG] Inv {feature}[:5]: {inv_vals[:5]}")

            axs[i, col_idx].plot(frame_ids, raw_vals, label="Raw", alpha=0.8)
            axs[i, col_idx].plot(frame_ids, inv_vals, label="Processed (Inverse)", alpha=0.8)
            label_title = f"{feature} (log)" if feature in log_transformed_features else feature
            axs[i, col_idx].set_title(f"{label_title} — Vehicle {vehicle_id}")
            axs[i, col_idx].legend()
            axs[i, col_idx].grid(True)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[INFO] Saved raw vs preprocessed comparison to {save_path}")


def plot_sequence_vs_trajectory(df, sequences, feature_names, metadata, save_path, scaler, num_sequences=3):
    """
    Overlay the first T values of v_Vel and v_Acc from sequences onto the original trajectory
    using inverse-transformed values (i.e., real-world scale).

    Args:
        df: pandas DataFrame of full preprocessed data with Vehicle_ID, Frame_ID, features (unscaled)
        sequences: [N, T, F] numpy array of extracted sequences (scaled)
        feature_names: list of feature names (must include first 4 ego features)
        metadata: list of dicts with 'Vehicle_ID' and 'Start_Frame' per sequence
        save_path: path to save the plot
        scaler: StandardScaler fitted on the original data
        num_sequences: how many different vehicles to show
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    ego_feature_names = feature_names[:4]
    v_idx = ego_feature_names.index("v_Vel")
    a_idx = ego_feature_names.index("v_Acc")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Pick one sequence per unique vehicle
    seen_vehicles = set()
    selected = []

    for i, meta in enumerate(metadata):
        if meta["Vehicle_ID"] not in seen_vehicles:
            seen_vehicles.add(meta["Vehicle_ID"])
            selected.append((i, meta))
        if len(selected) >= num_sequences:
            break

    actual_n = len(selected)
    fig, axs = plt.subplots(actual_n, 2, figsize=(12, 4 * actual_n))

    if actual_n == 1:
        axs = [axs]

    for row, (seq_idx, meta) in enumerate(selected):
        vehicle_id = meta["Vehicle_ID"]
        start_frame = meta["Start_Frame"]

        traj = df[df["Vehicle_ID"] == vehicle_id].sort_values("Frame_ID")
        seq = sequences[seq_idx]  # shape: [T, F]
        T = seq.shape[0]
        frames = list(range(start_frame, start_frame + T))

        # Extract ego features
        ego_seq = seq[:, :4]
        T = ego_seq.shape[0]

        # Create zero-filled array with same shape as scaler input
        dummy_full = np.zeros((T, len(feature_names)))
        dummy_full[:, :4] = ego_seq  # fill ego features only

        # Inverse-transform full array, then take ego part
        inv_full = scaler.inverse_transform(dummy_full)
        inv_ego_seq = inv_full[:, :4]


        # Plot v_Vel
        axs[row][0].plot(traj["Frame_ID"], traj["v_Vel"], label="Trajectory v_Vel", color="lightgray")
        axs[row][0].plot(frames, inv_ego_seq[:, v_idx], label="Sequence v_Vel (inv)", color="blue", linewidth=2)
        axs[row][0].set_title(f"Vehicle {vehicle_id} — v_Vel")
        axs[row][0].legend()
        axs[row][0].grid(True)

        # Plot v_Acc
        axs[row][1].plot(traj["Frame_ID"], traj["v_Acc"], label="Trajectory v_Acc", color="lightgray")
        axs[row][1].plot(frames, inv_ego_seq[:, a_idx], label="Sequence v_Acc (inv)", color="orange", linewidth=2)
        axs[row][1].set_title(f"Vehicle {vehicle_id} — v_Acc")
        axs[row][1].legend()
        axs[row][1].grid(True)

    plt.suptitle("Sequence vs. Original Trajectory (Inverse Transformed)", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_context_during_lane_change(ego_df, context_df, save_path):
    """
    Plots context variables for an ego vehicle around a lane change.

    Args:
        ego_df: DataFrame with ego vehicle's own info (must include Frame_ID, Lane_ID, Time_Headway, Space_Headway)
        context_df: DataFrame with context info over time (e.g., relative position/lane of front/adjacent vehicles)
        save_path: where to save the resulting plot
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(ego_df["Frame_ID"], ego_df["Space_Headway"], label="Space Headway")
    axs[0].set_ylabel("Space Headway (m)")
    axs[0].legend()

    axs[1].plot(ego_df["Frame_ID"], ego_df["Time_Headway"], label="Time Headway", color='orange')
    axs[1].set_ylabel("Time Headway (s)")
    axs[1].legend()

    # Relative lateral offsets for 4 neighbours
    axs[2].plot(context_df["Frame_ID"], context_df["LF_Lateral"], label="LF Offset", color="blue")
    axs[2].plot(context_df["Frame_ID"], context_df["LR_Lateral"], label="LR Offset", color="orange")
    axs[2].plot(context_df["Frame_ID"], context_df["RF_Lateral"], label="RF Offset", color="green")
    axs[2].plot(context_df["Frame_ID"], context_df["RR_Lateral"], label="RR Offset", color="red")
    axs[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[2].set_ylabel("Lateral Offset (m)")
    axs[2].legend()

    axs[3].plot(ego_df["Frame_ID"], ego_df["Lane_ID"], label="Lane ID", color='purple')
    axs[3].set_ylabel("Lane ID")
    axs[3].set_xlabel("Frame")
    axs[3].legend()

    plt.suptitle("Context Validation During Lane Change")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_lane_change_neighbour_dynamics(ego_df, context_df, save_path):
    """
    Plots how neighbour lateral positions shift during an ego vehicle's lane change.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot relative lateral position of neighbours
    for nb in ["LF", "LR", "RF", "RR"]:
        axs[0].plot(
            context_df["Frame_ID"],
            context_df[f"{nb}_Lateral"],
            label=f"{nb} Lateral Offset"
        )
    axs[0].set_ylabel("Lateral Offset (m)")
    axs[0].axhline(0, linestyle="--", color="gray", alpha=0.5)
    axs[0].legend()
    axs[0].set_title("Neighbour Lateral Offsets During Lane Change")

    # Plot space headway of neighbours (or relative longitudinal offset if headway is unavailable)
    for nb in ["LF", "LR", "RF", "RR"]:
        headway_key = f"{nb}_Headway"
        if headway_key in context_df.columns:
            axs[1].plot(
                context_df["Frame_ID"],
                context_df[headway_key],
                label=f"{nb} Headway"
            )
    axs[1].set_ylabel("Space Headway (m)")
    axs[1].legend()
    axs[1].set_title("Neighbour Headways During Lane Change")
    axs[1].set_xlabel("Frame ID")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_sequence_heatmap(sequence, feature_names, save_path, title="Heatmap of Sequence [T x Features]"):
    """
    Plots a heatmap of a single sequence [T x F], showing feature values across timesteps.

    Args:
        sequence: numpy array of shape [T, F]
        feature_names: list of F feature names
        save_path: where to save the heatmap
        title: optional plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Transpose for plotting: rows = features, cols = timesteps
    data = sequence.T

    plt.figure(figsize=(12, 6))
    sns.heatmap(data, cmap="viridis", cbar_kws={'label': 'Feature Value'}, xticklabels=5, yticklabels=feature_names)
    plt.xlabel("Timestep")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_reconstruction_score_distribution(scores, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    thresholds = {
        "Top 1%": np.percentile(scores, 99),
        "Top 5%": np.percentile(scores, 95),
        "Top 10%": np.percentile(scores, 90),
    }

    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=100, alpha=0.7, color="skyblue", edgecolor="black")
    for label, val in thresholds.items():
        plt.axvline(val, linestyle="--", label=f"{label}: {val:.4f}")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Count")
    plt.title("Reconstruction Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_mahalanobis_score_distribution(maha_scores, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(maha_scores, kde=True, bins=100, color="orange")
    plt.title("Mahalanobis Score Distribution")
    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_hybrid_score_comparison(recon_scores, maha_scores, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(recon_scores, maha_scores, alpha=0.5, edgecolor='k', s=20)
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Mahalanobis Distance")
    plt.title("Hybrid Score Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_top_n_reconstructions(X, X_hat, scores, feature_names, metadata, top_n, save_dir):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(save_dir, exist_ok=True)
    sorted_indices = np.argsort(scores)[::-1]

    seen = set()
    top_indices = []
    min_frame_gap = 100  # adjust as needed

    for idx in sorted_indices:
        vid, frame_ids = metadata[idx]
        start_frame = frame_ids[0]

        # Check against previously seen (vehicle_id, start_frame)
        is_duplicate = any(
            vid == prev_vid and abs(start_frame - prev_start) < min_frame_gap
            for prev_vid, prev_start in seen
        )

        if not is_duplicate:
            seen.add((vid, start_frame))
            top_indices.append(idx)

        if len(top_indices) == top_n:
            break

    for i, idx in enumerate(top_indices):
        x = X[idx]
        x_hat = X_hat[idx]
        meta = metadata[idx]
        vehicle_id, frame_ids = meta[0], meta[1]

        fig, ax1 = plt.subplots(figsize=(10, 4))

        # Plot velocity on left axis
        ax1.plot(x[:, 0], label=f"{feature_names[0]} (original)", linestyle='--', color='tab:blue')
        ax1.plot(x_hat[:, 0], label=f"{feature_names[0]} (recon)", color='tab:blue')
        ax1.set_ylabel(feature_names[0], color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Twin axis for acceleration
        ax2 = ax1.twinx()
        ax2.plot(x[:, 1], label=f"{feature_names[1]} (original)", linestyle='--', color='tab:red')
        ax2.plot(x_hat[:, 1], label=f"{feature_names[1]} (recon)", color='tab:red')
        ax2.set_ylabel(feature_names[1], color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Title and labels
        ax1.set_title(f"{vehicle_id}, Start Frame: {frame_ids[0]}")
        ax1.set_xlabel("Timestep")

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, f"top_{i+1}_anomaly.png"))
        plt.close()

def plot_score_vs_lane_change(scores, lane_change_mask, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.DataFrame({
        "Score": scores,
        "Type": ["Lane Change" if x else "No Lane Change" for x in lane_change_mask]
    })

    plt.figure(figsize=(6, 5))
    ax = sns.violinplot(x="Type", y="Score", data=df, cut=0)

    # Set y-axis to log scale
    ax.set_yscale("log")
    ax.set_title("Score Distribution: Lane Change vs. Non-Lane Change (Log Scale)")
    ax.set_ylabel("Score (log scale)")
    ax.set_xlabel("")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_lane_change_frequency(lane_change_mask_path, sequence_metadata_path, save_path):
    # === Load lane change mask ===
    lane_change_mask = np.load(lane_change_mask_path)  # shape: (num_sequences,) — values: 0 or 1

    # === Load metadata to group by vehicle ===
    # Assumes you have something like a .npz or .pkl with vehicle IDs per sequence
    metadata = np.load(sequence_metadata_path, allow_pickle=True)
    vehicle_ids = metadata["vehicle_ids"]  # shape: (num_sequences,) — vehicle ID per sequence

    # === Count lane changes per vehicle ===
    from collections import defaultdict
    lane_change_counts = defaultdict(int)

    for vehicle_id, is_lane_change in zip(vehicle_ids, lane_change_mask):
        lane_change_counts[vehicle_id] += int(is_lane_change)

    # === Convert to list of counts ===
    counts = list(lane_change_counts.values())

    # === Plot histogram ===
    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=range(max(counts)+2), edgecolor='black', align='left')
    plt.xlabel("Number of Lane Changes per Vehicle")
    plt.ylabel("Number of Vehicles")
    plt.title("Histogram of Lane Change Frequency per Vehicle")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved lane change histogram to {save_path}")

def plot_lane_change_timeline(meta, lane_change_mask, save_path):
    """
    Plots a timeline of lane changes across Frame_IDs.
    Each vehicle is assigned a random y-offset to avoid overlapping marks.
    
    Args:
        meta (List[dict]): metadata for each sequence, must contain "Vehicle_ID" and "Start_Frame"
        lane_change_mask (np.ndarray): Boolean array aligned with meta
        save_path (str): Path to save the figure
    """
    assert len(meta) == len(lane_change_mask), "meta and mask length mismatch"
    
    vehicle_ids = [m["Vehicle_ID"] for m in meta]
    start_frames = [m["Start_Frame"] for m in meta]
    
    # Assign each vehicle a consistent y-offset
    unique_vehicles = sorted(set(vehicle_ids))
    vehicle_to_y = {vid: i for i, vid in enumerate(unique_vehicles)}
    
    # Plot
    plt.figure(figsize=(12, 6))
    for i, (vid, frame, is_lc) in enumerate(zip(vehicle_ids, start_frames, lane_change_mask)):
        y = vehicle_to_y[vid]
        color = "red" if is_lc else "lightgray"
        alpha = 1.0 if is_lc else 0.3
        marker = "o" if is_lc else "x"
        plt.scatter(frame, y, color=color, alpha=alpha, s=10, marker=marker)
    
    plt.yticks([])  # Optional: hide y-axis labels
    plt.xlabel("Frame ID")
    plt.ylabel("Vehicle Offset")
    plt.title("Timeline of Lane Changes Across Frames")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved lane change timeline plot to {save_path}")

def plot_lane_change_proportion(lane_change_mask, save_path, kind="bar"):
    """
    Plots the proportion of lane-change-aligned vs. non-aligned sequences.
    
    Args:
        lane_change_mask (np.ndarray): Boolean array where True = near lane change
        save_path (str): Where to save the figure
        kind (str): "bar" or "pie"
    """
    aligned_count = lane_change_mask.sum()
    non_aligned_count = len(lane_change_mask) - aligned_count

    labels = ["Lane-Change Aligned", "Non-Aligned"]
    counts = [aligned_count, non_aligned_count]

    plt.figure(figsize=(6, 4))

    if kind == "bar":
        plt.bar(labels, counts, color=["red", "gray"])
        plt.ylabel("Number of Sequences")
        plt.title("Distribution of Lane-Change-Aligned Sequences")
        for i, v in enumerate(counts):
            plt.text(i, v + 0.01 * max(counts), str(v), ha='center', va='bottom')
    elif kind == "pie":
        plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=["red", "gray"])
        plt.title("Proportion of Lane-Change-Aligned Sequences")
    else:
        raise ValueError("Invalid kind. Use 'bar' or 'pie'.")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved lane change proportion plot to {save_path}")