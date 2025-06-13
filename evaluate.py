### evaluate.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import joblib
from models.autoencoder import LSTMAutoencoder
from data.load import load_sequences
from utils.metrics import get_flat_recon_errors, get_mahalanobis_scores
from config import CONFIG
from models.traditional import TraditionalAnomalyDetector
from utils.visualisation import (
    plot_reconstruction_score_distribution,
    plot_mahalanobis_score_distribution,
    plot_hybrid_score_comparison,
    plot_top_n_reconstructions,
    plot_score_vs_lane_change,
    plot_lane_change_frequency,
    plot_lane_change_timeline,
    plot_lane_change_proportion
)
from utils.lane_utils import set_lane_changes

def evaluate_model(model_type="autoencoder"):
    print(f"[INFO] Evaluating model: {model_type}")
    if model_type != "autoencoder":
        raise NotImplementedError("Only 'autoencoder' evaluation is supported for now.")

    # === Load and preprocess data ===
    sequences, targets, meta = load_sequences(CONFIG["SEQUENCE_FILE"], return_meta=True)
    print(f"[DEBUG] Loaded sequences shape: {sequences.shape}")
    print(f"[DEBUG] Sample sequence values (scaled): {sequences[0, :5, :4]}")

    subset_fraction = 0.1
    total = len(sequences)
    subset_size = int(total * subset_fraction)
    print(f"[INFO] Subsampling {subset_size}/{total} sequences...")
    np.random.seed(42)
    subset_indices = np.random.choice(total, subset_size, replace=False)

    sequences = sequences[subset_indices]
    targets = targets[subset_indices]
    meta = [meta[i] for i in subset_indices]

    # === Compute deltas (match training input) ===
    deltas = sequences[:, 1:, :] - sequences[:, :-1, :]
    nan_mask = ~np.isnan(deltas).any(axis=(1, 2)) & ~np.isinf(deltas).any(axis=(1, 2))
    deltas = deltas[nan_mask]
    sequences = sequences[nan_mask]
    targets = targets[nan_mask]
    meta = [m for i, m in enumerate(meta) if nan_mask[i]]
    print(f"[DEBUG] Delta sequences shape: {deltas.shape}")

    # === Convert to tensor ===
    sequence_tensor = torch.tensor(deltas, dtype=torch.float32)
    loader = DataLoader(TensorDataset(sequence_tensor), batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    # === Load model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(
        input_dim=deltas.shape[2],
        hidden_dim=CONFIG["HIDDEN_DIM"],
        latent_dim=CONFIG["LATENT_DIM"],
        num_layers=CONFIG["NUM_LAYERS"]
    ).to(device)

    checkpoint = torch.load(CONFIG["MODEL_PATH"], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("[INFO] Model loaded for evaluation.")

    # === Run model ===
    print("[INFO] Computing reconstructions...")
    with torch.no_grad():
        recon_all = []
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon, _ = model(batch_x)
            recon_all.append(recon.cpu().numpy())

    recon_all = np.concatenate(recon_all, axis=0)  # [N, T-1, F]
    print("[INFO] Reconstructions complete.")

    # === Compare reconstructed deltas to true deltas (ego features only) ===
    true_ego_deltas = deltas[:, :, :4]
    recon_ego_deltas = recon_all[:, :, :4]
    errors = true_ego_deltas - recon_ego_deltas
    mse_per_timestep = (errors ** 2).mean(axis=2)
    reconstruction_scores = mse_per_timestep.mean(axis=1)
    print("[INFO] Reconstruction scoring complete.")
    np.save("data/reconstruction_scores.npy", reconstruction_scores)

    # === Mahalanobis ===
    flat_errors = errors.reshape(errors.shape[0], -1)
    flat_errors = np.nan_to_num(flat_errors, nan=0.0, posinf=0.0, neginf=0.0)
    mean_vec = flat_errors.mean(axis=0)
    cov = np.cov(flat_errors, rowvar=False)
    cov_inv = np.linalg.pinv(cov, hermitian=True)
    mahalanobis_scores = get_mahalanobis_scores(flat_errors, mean_vec, cov_inv)
    np.save("data/mahalanobis_scores.npy", mahalanobis_scores)

    # === Traditional Anomaly Models ===
    iso = TraditionalAnomalyDetector(method="iforest")
    iso.fit(flat_errors)
    iso_scores = iso.score(flat_errors)
    np.save("data/isolation_forest_scores.npy", iso_scores)

    lof = TraditionalAnomalyDetector(method="lof")
    try:
        lof.fit(flat_errors)
        lof_scores = lof.score(flat_errors)
    except ValueError:
        print("[WARNING] LOF failed due to NaNs. Skipping.")
        lof_scores = np.zeros_like(iso_scores)
    np.save("data/lof_scores.npy", lof_scores)

    # === Optional: Inverse transform and reconstruct absolute ego features ===
    true_ego_abs = np.cumsum(true_ego_deltas, axis=1)
    recon_ego_abs = np.cumsum(recon_ego_deltas, axis=1)

    # Prepend the first absolute value
    start_vals = sequences[:, :1, :4]
    true_ego_abs = np.concatenate([start_vals, start_vals + true_ego_abs], axis=1)
    recon_ego_abs = np.concatenate([start_vals, start_vals + recon_ego_abs], axis=1)

    if CONFIG.get("SCALER_PATH") and os.path.exists(CONFIG["SCALER_PATH"]):
        scaler = joblib.load(CONFIG["SCALER_PATH"])
        for i in range(4):
            mean = scaler.mean_[i]
            scale = scaler.scale_[i]
            true_ego_abs[:, :, i] = true_ego_abs[:, :, i] * scale + mean
            recon_ego_abs[:, :, i] = recon_ego_abs[:, :, i] * scale + mean

    print("[DEBUG] true_ego_abs shape:", true_ego_abs.shape)
    print("[DEBUG] recon_ego_abs shape:", recon_ego_abs.shape)

    # === Lane change labels ===
    lane_change_mask = np.load(CONFIG["LANE_CHANGE_MASK_PATH"])
    lane_change_mask = lane_change_mask[:len(true_ego_abs)]
    assert len(lane_change_mask) == len(reconstruction_scores), "Mismatch in lane change mask length"

    # === Plots ===
    plot_reconstruction_score_distribution(
        reconstruction_scores,
        save_path=os.path.join(CONFIG["EVALUATION_PLOTS_DIR"], "reconstruction_score_distribution.png")
    )

    plot_mahalanobis_score_distribution(
        mahalanobis_scores,
        save_path=os.path.join(CONFIG["EVALUATION_PLOTS_DIR"], "mahalanobis_score_distribution.png")
    )

    plot_hybrid_score_comparison(
        reconstruction_scores,
        mahalanobis_scores,
        save_path=os.path.join(CONFIG["EVALUATION_PLOTS_DIR"], "hybrid_score_comparison.png")
    )

    plot_top_n_reconstructions(
        X=true_ego_abs,
        X_hat=recon_ego_abs,
        scores=reconstruction_scores,
        feature_names=CONFIG["FEATURES"][:4],
        metadata=meta,
        top_n=5,
        save_dir=os.path.join(CONFIG["EVALUATION_PLOTS_DIR"], "top_n_recon")
    )

    plot_score_vs_lane_change(
        reconstruction_scores,
        lane_change_mask,
        save_path=os.path.join(CONFIG["EVALUATION_PLOTS_DIR"], "score_vs_lane_change.png")
    )

    # === Lane Change Frequency Plot ===
    vehicle_ids = np.array([m["Vehicle_ID"] for m in meta])
    plot_lane_change_frequency(
        lane_change_mask=lane_change_mask,
        vehicle_ids=vehicle_ids,
        save_path=os.path.join(CONFIG["EVALUATION_PLOTS_DIR"], "lane_change_frequency_histogram.png")
    )

    plot_lane_change_timeline(
        meta=meta,
        lane_change_mask=lane_change_mask,
        save_path=os.path.join(CONFIG["EVALUATION_PLOTS_DIR"], "lane_change_timeline.png")
    )

    plot_lane_change_proportion(
        lane_change_mask,
        save_path=os.path.join(CONFIG["EVALUATION_PLOTS_DIR"], "lane_change_proportion.png"),
        kind="bar"  # or "pie"
    )

    return {
        "reconstruction": reconstruction_scores,
        "mahalanobis": mahalanobis_scores,
        "isolation_forest": iso_scores,
        "lof": lof_scores,
        "lane_change_mask": lane_change_mask,
    }
