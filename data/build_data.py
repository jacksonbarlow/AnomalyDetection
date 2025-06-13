# data/build_data.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import numpy as np
import os
import uuid
import time
from config import CONFIG
from data.load import load_raw_dataframe, load_context_cache
from data.preprocess import preprocess_and_clip
from utils.visualisation import (
    plot_context_during_lane_change,
    plot_sequence_vs_trajectory,
    plot_raw_vs_preprocessed_features,
    plot_sequence_heatmap,
    plot_lane_change_neighbour_dynamics
)
from utils.lane_utils import find_vehicle_with_valid_context, get_context_dataframe
from data.preprocess import get_static_features

def create_sequences_with_context(df_scaled, df_raw, frame_lookup, context_cache, window_size=20, step_size=1):
    FEATURES = CONFIG["FEATURES"]
    TEMP_SAVE_DIR = CONFIG.get("SEQUENCE_SAVE_DIR", "temp_sequences")
    os.makedirs(TEMP_SAVE_DIR, exist_ok=True)
    compress_chunks=True

    df_combined = df_raw[[
        "Vehicle_ID", "Frame_ID", "Lane_ID", "Local_X", "Local_Y",
        "v_Class", "v_Length", "v_Width", "source_tag"
    ]].copy().reset_index(drop=True)
    df_scaled = df_scaled.reset_index(drop=True)
    df_combined[FEATURES] = df_scaled[FEATURES]
    df_combined["Global_Vehicle_ID"] = (
        df_combined["source_tag"].astype(str) + "_" + df_combined["Vehicle_ID"].astype(str)
    )

    vehicle_ids = df_combined["Global_Vehicle_ID"].unique()
    print(f"[INFO] Total vehicles: {len(vehicle_ids)}")

    # Filter vehicles upfront
    valid_ids = [
        vid for vid in vehicle_ids
        if len(df_combined[df_combined["Global_Vehicle_ID"] == vid]) >= window_size + 1
    ]
    print(f"[INFO] Vehicles passing basic length check: {len(valid_ids)}")

    sequence_chunks, target_chunks, meta_info = [], [], []
    chunk_size = 1000  # reduced for lower memory footprint
    chunk_idx = 0
    total_seq_count = 0
    t_start = time.time()

    for vehicle_id in tqdm(valid_ids, desc="Building sequences"):
        group = df_combined[df_combined["Global_Vehicle_ID"] == vehicle_id].sort_values("Frame_ID")
        data = group[FEATURES + ["Frame_ID"]].values
        full_group = group.reset_index(drop=True)
        static_features = get_static_features(full_group.iloc[0])

        for i in range(0, len(data) - window_size, step_size):
            seq_input, frame_ids_seq = [], []
            for j in range(window_size):
                ego_features = data[i + j, :4]
                ego_row = full_group.iloc[i + j]
                frame_id = int(ego_row["Frame_ID"])
                key = (int(ego_row["Vehicle_ID"]), frame_id)
                context_features = context_cache.get(key, np.zeros(24))
                full_vector = np.concatenate([ego_features, context_features, static_features])
                seq_input.append(full_vector)
                frame_ids_seq.append(frame_id)

            target = data[i + window_size, :4]
            sequence_chunks.append(seq_input)
            target_chunks.append(target)
            meta_info.append((vehicle_id, frame_ids_seq))
            total_seq_count += 1

            # Periodic flush
            if len(sequence_chunks) >= chunk_size:
                seq_arr = np.array(sequence_chunks, dtype=np.float32)
                tgt_arr = np.array(target_chunks, dtype=np.float32)
                save_path = os.path.join(TEMP_SAVE_DIR, f"chunk_{chunk_idx}")

                if compress_chunks:
                    np.savez_compressed(save_path, sequences=seq_arr, targets=tgt_arr)
                else:
                    np.save(save_path + "_sequences.npy", seq_arr)
                    np.save(save_path + "_targets.npy", tgt_arr)

                sequence_chunks.clear()
                target_chunks.clear()
                chunk_idx += 1
                del seq_arr, tgt_arr
                gc.collect()

            #if total_seq_count % 1000 == 0:
                #elapsed = time.time() - t_start
                #print(f"[DEBUG] {total_seq_count} sequences generated — {elapsed:.1f}s elapsed")

    # Final flush
    if sequence_chunks:
        seq_arr = np.array(sequence_chunks, dtype=np.float32)
        tgt_arr = np.array(target_chunks, dtype=np.float32)
        save_path = os.path.join(TEMP_SAVE_DIR, f"chunk_{chunk_idx}")
        if compress_chunks:
            np.savez_compressed(save_path, sequences=seq_arr, targets=tgt_arr)
        else:
            np.save(save_path + "_sequences.npy", seq_arr)
            np.save(save_path + "_targets.npy", tgt_arr)

        del seq_arr, tgt_arr
        gc.collect()

    print(f"[INFO] Finished. Total sequences: {total_seq_count}")
    print(f"[INFO] Chunks saved under: {TEMP_SAVE_DIR}")

    
    return TEMP_SAVE_DIR, meta_info

def build_and_save_sequences(force_rebuild=False):
    import random
    import glob
    from numpy.lib.format import open_memmap

    df = load_raw_dataframe()

    df_scaled, scaler = preprocess_and_clip(df, sequence_cache_path=CONFIG["SCALED_FEATURES_PATH"])

    frame_lookup = {f: fdf for f, fdf in df.groupby("Frame_ID")}
    context_cache = load_context_cache()

    # Context plots
    try:
        vehicle_id = find_vehicle_with_valid_context(df, context_cache, min_valid_timesteps=10)
        ego_df = df[df["Vehicle_ID"] == vehicle_id]
        context_df = get_context_dataframe(ego_df, context_cache)

        plot_context_during_lane_change(
            ego_df, context_df,
            save_path=os.path.join(CONFIG["CONTEXT_PLOTS_DIR"], "context_during_lane_change.png")
        )
        plot_lane_change_neighbour_dynamics(
            ego_df, context_df,
            save_path=os.path.join(CONFIG["CONTEXT_PLOTS_DIR"], "lane_change_neighbour_dynamics.png")
        )
    except ValueError as e:
        print(f"[WARNING] {e} — skipping context plots")

    seq_path = CONFIG["SEQUENCE_FILE"]
    if os.path.exists(seq_path) and not force_rebuild:
        print(f"[INFO] Loading cached sequences from {seq_path}")
        with np.load(seq_path, allow_pickle=True) as data:
            seq_path_memmap = data["seq_path"].item()
            tgt_path_memmap = data["tgt_path"].item()
            meta_vids = data["meta_vids"]
            meta_fids = data["meta_fids"]

        sequences = np.load(seq_path_memmap, mmap_mode="r")
        targets = np.load(tgt_path_memmap, mmap_mode="r")

    else:
        temp_dir, meta_info = create_sequences_with_context(
            df_scaled, df, frame_lookup, context_cache, window_size=CONFIG["WINDOW_SIZE"]
        )

        # Gather temp chunks from disk
        sequence_files = sorted(glob.glob(os.path.join(temp_dir, "sequences_*.npy")))
        target_files = sorted(glob.glob(os.path.join(temp_dir, "targets_*.npy")))

        if not sequence_files:
            raise RuntimeError("No sequence files found for final assembly")

        num_seqs = sum(np.load(f).shape[0] for f in sequence_files)
        seq_shape = (num_seqs, CONFIG["WINDOW_SIZE"], np.load(sequence_files[0]).shape[2])

        print(f"[INFO] Writing {num_seqs} sequences to disk using memmap...")
        seq_memmap_path = seq_path.replace(".npz", "_seq_memmap.npy")
        tgt_memmap_path = seq_path.replace(".npz", "_tgt_memmap.npy")

        seq_memmap = open_memmap(seq_memmap_path, dtype='float32', mode='w+', shape=seq_shape)
        tgt_memmap = open_memmap(tgt_memmap_path, dtype='float32', mode='w+', shape=(num_seqs, 4))

        offset = 0
        for sfile, tfile in tqdm(zip(sequence_files, target_files), total=len(sequence_files), desc="Assembling memmap"):
            s_chunk = np.load(sfile)
            t_chunk = np.load(tfile)
            n = s_chunk.shape[0]
            seq_memmap[offset:offset + n] = s_chunk
            tgt_memmap[offset:offset + n] = t_chunk
            offset += n

        meta_vids = np.array([vid for vid, _ in meta_info])
        meta_fids = np.array([fids for _, fids in meta_info], dtype=object)
        np.savez(seq_path, seq_path=seq_memmap_path, tgt_path=tgt_memmap_path,
                 meta_vids=meta_vids, meta_fids=meta_fids)

        sequences = seq_memmap
        targets = tgt_memmap

        print(f"[INFO] Memory-mapped sequences saved to: {seq_path}")

    sequence_metadata = [
        {"Vehicle_ID": vid, "Start_Frame": int(start[0])}
        for vid, start in zip(meta_vids, meta_fids)
    ]

    print("[INFO] Starting Plots")

    plot_sequence_vs_trajectory(
        df=df,
        sequences=sequences,
        feature_names=CONFIG["FEATURES"],
        metadata=sequence_metadata,
        save_path=os.path.join(CONFIG["PREPROCESSING_PLOTS_DIR"], "sequence_vs_trajectory_temporal_check.png"),
        scaler=scaler,
        num_sequences=3
    )

    diffs = df["v_Acc"].reset_index(drop=True) - df_scaled["v_Acc"]
    abs_diffs = abs(diffs)
    vehicle_scores = abs_diffs.groupby(df_scaled["Vehicle_ID"]).mean()
    median_score = vehicle_scores.median()

    # Sort vehicles by score
    sorted_scores = vehicle_scores.sort_values()

    # Find vehicle just below median
    least_changed = sorted_scores[sorted_scores < median_score].iloc[6:].index[0]  # max of below-median

    # Find vehicle just above median
    most_changed = sorted_scores[sorted_scores > median_score].iloc[7:].index[0]  # min of above-median

    print(f"[INFO] Median score: {median_score:.4f}")
    print(f"[INFO] Most changed (above median): {most_changed} — score: {vehicle_scores[most_changed]:.4f}")
    print(f"[INFO] Least changed (below median): {least_changed} — score: {vehicle_scores[least_changed]:.4f}")

    comparison_df = df_scaled.copy()

    # Align raw df to df_scaled using the index
    raw_aligned = df.loc[df_scaled.index, CONFIG["FEATURES"]]

    for col in CONFIG["FEATURES"]:
        comparison_df[col + "_raw"] = raw_aligned[col].values


    print("[DEBUG] Scaler was trained on:", scaler.feature_names_in_)
    print("[DEBUG] CONFIG[FEATURES]:", CONFIG["FEATURES"])

    plot_raw_vs_preprocessed_features(
        df_raw=df,
        df_scaled=df_scaled,
        vehicle_ids=[least_changed, most_changed],
        feature_names=CONFIG["FEATURES"],
        save_path=os.path.join(CONFIG["PREPROCESSING_PLOTS_DIR"], "raw_vs_preprocessed_features.png"),
        scaler=scaler
    )


    ego_features = CONFIG["FEATURES"][:4]
    context_feature_names = [f"Context_{i}" for i in range(24)]
    static_feature_names = ["Vehicle_Type", "v_Length", "v_Width"]
    feature_names = ego_features + context_feature_names + static_feature_names


    plot_sequence_heatmap(
        sequence=sequences[0],
        feature_names=feature_names,
        save_path=os.path.join(CONFIG["PREPROCESSING_PLOTS_DIR"], "sequence_feature_heatmap.png"),
        title="Heatmap of First Sequence [T x Features]"
    )

    lane_change_path = CONFIG["LANE_CHANGE_MASK_PATH"]

    if os.path.exists(lane_change_path):
        lane_change_mask = np.load(lane_change_path)
        print(f"[INFO] Loaded existing lane change mask from {lane_change_path}")
    else:
        lane_change_mask = None
        print("[INFO] Lane change mask not found. Will be generated if needed.")

        lane_change_mask = []
        for i in range(len(meta_vids)):
            vid = meta_vids[i]
            frame_ids = meta_fids[i]
            veh_df = df[df["Vehicle_ID"] == int(str(vid).split("_")[-1])]
            lane_ids = veh_df.set_index("Frame_ID").loc[frame_ids]["Lane_ID"].values
            changed = not np.all(lane_ids == lane_ids[0])
            lane_change_mask.append(changed)

        lane_change_mask = np.array(lane_change_mask)
        np.save(CONFIG["LANE_CHANGE_MASK_PATH"], lane_change_mask)
        print(f"[INFO] Saved lane change mask to {CONFIG['LANE_CHANGE_MASK_PATH']}")
