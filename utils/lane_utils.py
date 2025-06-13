# utils/lane_utils.py

### Library Imports and Setup ###
import numpy as np
import pandas as pd

### Main Function ###
def detect_lane_changes(df):
    """
    Detects frames where vehicles change lanes.
    Returns a dictionary mapping vehicle_id -> set of frame_ids where lane change occurred.
    """
    lane_changes = {}
    for vid, group in df.groupby("Vehicle_ID"):
        group_sorted = group.sort_values("Frame_ID")
        lane_ids = group_sorted["Lane_ID"].values
        frame_ids = group_sorted["Frame_ID"].values

        lane_diff = np.diff(lane_ids)
        change_indices = np.where(lane_diff != 0)[0]
        change_frames = set(frame_ids[change_indices + 1])  # +1: the frame after change
        lane_changes[vid] = change_frames

    return lane_changes

def label_sequences_by_lane_change(meta_info, lane_changes, window_radius=5):
    """
    Labels each sequence in meta_info with 1 if any frame in the sequence is within +/- window_radius of a lane change.
    meta_info: list of (vehicle_id, [frame_ids])
    Returns: binary label array (1 = lane change nearby)
    """
    labels = []
    for vid, frame_seq in meta_info:
        change_frames = lane_changes.get(vid, set())
        label = any(
            any(abs(fid - chg) <= window_radius for chg in change_frames) for fid in frame_seq
        )
        labels.append(1 if label else 0)
    return np.array(labels)

def find_lane_change_vehicle(df):
    grouped = df.groupby("Vehicle_ID")["Lane_ID"]
    for vid, lane_series in grouped:
        if lane_series.nunique() > 1:
            return vid

    raise ValueError("No lane-changing vehicle found.")

def find_vehicle_with_full_context(context_cache, min_valid_frames=10):
    """
    Return the Vehicle_ID for the first vehicle with full context (all 4 neighbours present across all timesteps).
    """
    for vehicle_id, context_array in context_cache.items():
        # Skip malformed entries
        if not isinstance(context_array, np.ndarray) or context_array.ndim != 2:
            continue
        if context_array.shape[1] < 8:
            continue  # Not enough context columns

        neighbour_ids = context_array[:, :8:2]  # LF, LR, RF, RR IDs
        if (neighbour_ids != -1).all():
            return vehicle_id
    raise ValueError("No vehicle found with complete context across all timesteps.")

def find_vehicle_with_most_valid_context(context_cache, required_neighbours=4, min_valid_ratio=0.8):
    """
    Find a vehicle with the highest number of valid context neighbour entries (out of 4) 
    across all timesteps, subject to a minimum valid ratio.
    """
    best_vehicle = None
    best_score = 0

    for vehicle_id, context_array in context_cache.items():
        if not isinstance(context_array, np.ndarray) or context_array.ndim != 2 or context_array.shape[1] < 8:
            continue

        neighbour_ids = context_array[:, :8:2]  # LF, LR, RF, RR
        valid_counts = (neighbour_ids != -1).sum(axis=1)  # valid neighbours per timestep
        valid_ratio = (valid_counts >= required_neighbours).mean()

        if valid_ratio >= min_valid_ratio and valid_ratio > best_score:
            best_vehicle = vehicle_id
            best_score = valid_ratio

    if best_vehicle is None:
        raise ValueError("No vehicle found with sufficient context coverage.")
    return best_vehicle

def get_context_dataframe(ego_df, context_cache):
    context_rows = []
    valid_rows = 0

    for _, row in ego_df.iterrows():
        key = (int(row["Vehicle_ID"]), int(row["Frame_ID"]))
        if key in context_cache:
            context_rows.append(np.concatenate(([row["Vehicle_ID"], row["Frame_ID"]], context_cache[key])))
            valid_rows += 1

    if not context_rows:
        raise ValueError(f"No context data found for vehicle {int(ego_df['Vehicle_ID'].iloc[0])}")

    # Human-readable context feature names (6 neighbors × 4 features each)
    context_cols = [
        "LF_Longitudinal", "LF_Velocity", "LF_Acceleration", "LF_Lateral",
        "LR_Longitudinal", "LR_Velocity", "LR_Acceleration", "LR_Lateral",
        "RF_Longitudinal", "RF_Velocity", "RF_Acceleration", "RF_Lateral",
        "RR_Longitudinal", "RR_Velocity", "RR_Acceleration", "RR_Lateral",
        "F_Longitudinal",  "F_Velocity",  "F_Acceleration",  "F_Lateral",
        "R_Longitudinal",  "R_Velocity",  "R_Acceleration",  "R_Lateral"
    ]

    columns = ["Vehicle_ID", "Frame_ID"] + context_cols
    context_df = pd.DataFrame(context_rows, columns=columns)

    return context_df



def find_best_vehicle_by_context_coverage(context_cache):
    """
    Select the vehicle with the most valid context entries across all timesteps,
    regardless of neighbour position or consistency.
    """
    best_vehicle = None
    max_valid_entries = 0

    for vehicle_id, context_array in context_cache.items():
        if not isinstance(context_array, np.ndarray) or context_array.ndim != 2 or context_array.shape[1] < 8:
            continue

        # Get LF, LR, RF, RR neighbour IDs: columns 0,2,4,6
        neighbour_ids = context_array[:, :8:2]
        valid_count = np.sum(neighbour_ids != -1)

        if valid_count > max_valid_entries:
            best_vehicle = vehicle_id
            max_valid_entries = valid_count

    if best_vehicle is None:
        raise ValueError("No vehicle found with any valid context data.")

    return best_vehicle

# utils/lane_utils.py

def find_vehicle_with_valid_context(df, context_cache, min_valid_timesteps=10):
    """
    Returns the ID of a vehicle with at least `min_valid_timesteps` non-zero context entries.
    """
    from collections import defaultdict
    counts = defaultdict(int)

    for (vid, _), context in context_cache.items():
        if np.any(context != 0):
            counts[vid] += 1

    # Filter vehicles with sufficient valid context
    valid_vids = [vid for vid, count in counts.items() if count >= min_valid_timesteps]

    if not valid_vids:
        raise ValueError("No vehicle found with any valid context data.")

    # Optionally: pick the one with the most valid timesteps
    return max(valid_vids, key=lambda vid: counts[vid])

def get_lane_change_mask(meta):
    """
    Returns a binary mask of same length as meta,
    where each element is 1 if the sequence corresponds to a lane change.
    """
    mask = []
    for entry in meta:
        # e.g., meta contains dicts with {'Vehicle_ID': ..., 'Frame_ID': ...}
        vehicle_id = entry.get("Vehicle_ID")
        frame_id = entry.get("Frame_ID")
        # Use your existing lane change detector
        is_lane_change = check_if_lane_change(vehicle_id, frame_id)
        mask.append(int(is_lane_change))
    return np.array(mask)

# Assumes lane_changes is computed once globally and reused
lane_changes_dict = None  # Global cache to avoid recomputation

def check_if_lane_change(vehicle_id, frame_id):
    global lane_changes_dict
    if lane_changes_dict is None:
        raise ValueError("lane_changes_dict not initialised. Call set_lane_changes() first.")
    frames = lane_changes_dict.get(vehicle_id, set())
    return frame_id in frames

def set_lane_changes(df):
    """
    Call this once with the full dataframe to populate the global lane_changes_dict.
    """
    global lane_changes_dict
    lane_changes_dict = detect_lane_changes(df)
