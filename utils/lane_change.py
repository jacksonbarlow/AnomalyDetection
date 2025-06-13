# utils/lane_change.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import CONFIG
from data.load import load_sequences, load_raw_dataframe
from utils.lane_utils import detect_lane_changes, label_sequences_by_lane_change
from scipy.stats import ttest_ind
import os

def run_lane_change_analysis():
    print("[INFO] Running lane change analysis...")

    # Load anomaly scores and metadata
    sequences, _, meta_info = load_sequences(CONFIG["SEQUENCE_FILE"], return_meta=True)
    scores = np.load("data/anomaly_scores.npy")

    # Load raw data and detect lane changes
    df = load_raw_dataframe()
    lane_changes = detect_lane_changes(df)

    # Label sequences
    labels = label_sequences_by_lane_change(meta_info, lane_changes, window_radius=5)
    assert len(scores) == len(labels), "Mismatch between scores and lane change labels"

    # Compute statistics
    lane_scores = scores[labels == 1]
    non_lane_scores = scores[labels == 0]
    t_stat, p_val = ttest_ind(lane_scores, non_lane_scores, equal_var=False)
    print(f"[INFO] T-test result: t = {t_stat:.2f}, p = {p_val:.4f}")

    # Plot violin plot
    df_plot = pd.DataFrame({
        "AnomalyScore": np.concatenate([lane_scores, non_lane_scores]),
        "LaneChange": ["Lane Change"] * len(lane_scores) + ["No Lane Change"] * len(non_lane_scores)
    })

    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df_plot, x="LaneChange", y="AnomalyScore", inner="box")
    plt.title("Anomaly Scores vs Lane Change Events")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(CONFIG["EVALUATION_PLOTS_DIR"], "lane_change_score_comparison.png")
    os.makedirs(CONFIG["EVALUATION_PLOTS_DIR"], exist_ok=True)
    plt.savefig(save_path)
    print(f"[INFO] Saved lane change analysis plot to {save_path}")