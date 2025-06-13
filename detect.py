### detect.py

import os
import numpy as np
import torch
from config import CONFIG
from utils.visualisation import plot_reconstruction_comparison
from data.load import load_sequences
from models.autoencoder import LSTMAutoencoder


def normalise_scores(scores, method='zscore'):
    if method == 'zscore':
        return (scores - np.mean(scores)) / (np.std(scores) + 1e-6)
    elif method == 'minmax':
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-6)
    else:
        raise ValueError("Unsupported normalisation method")


def detect_anomalies(model_type="autoencoder"):
    print(f"[INFO] Detecting anomalies using model: {model_type}")
    if model_type != "autoencoder":
        raise NotImplementedError("Only 'autoencoder' detection is supported for now.")

    ### Load precomputed scores ###
    score_dir = "data"
    recon = np.load(os.path.join(score_dir, "reconstruction_scores.npy"))
    mahal = np.load(os.path.join(score_dir, "mahalanobis_scores.npy"))

    try:
        iso = np.load(os.path.join(score_dir, "isolation_forest_scores.npy"))
        lof = np.load(os.path.join(score_dir, "lof_scores.npy"))
        use_traditional = True
    except FileNotFoundError:
        print("[INFO] Traditional model scores not found; skipping.")
        use_traditional = False

    ### Normalise ###
    recon_norm = normalise_scores(recon)
    mahal_norm = normalise_scores(mahal)
    if use_traditional:
        iso_norm = normalise_scores(iso)
        lof_norm = normalise_scores(lof)

    ### Combine Scores ###
    if use_traditional:
        w = CONFIG["HYBRID_WEIGHTS"]
        hybrid_score = (
            w["reconstruction"] * recon_norm +
            w["mahalanobis"] * mahal_norm +
            w["isolation_forest"] * iso_norm +
            w["lof"] * lof_norm
        )
    else:
        hybrid_score = (recon_norm + mahal_norm) / 2

    np.save(os.path.join(score_dir, "hybrid_scores.npy"), hybrid_score)

    ### Identify Top Anomalies ###
    sequences, _, _ = load_sequences(CONFIG["SEQUENCE_FILE"], return_meta=True)
    scores = hybrid_score
    top_k = 5
    worst_indices = np.argsort(scores)[-top_k:]

    print("[INFO] Top anomaly indices:", worst_indices)
    for i in worst_indices:
        print(f"Sequence {i} | Score: {scores[i]:.4f}")

    ### Load Model ###
    input_dim = sequences.shape[2]
    model = LSTMAutoencoder(
        input_dim=input_dim,
        latent_dim=CONFIG["LATENT_DIM"],
        hidden_dim=CONFIG["HIDDEN_DIM"],
        num_layers=CONFIG["NUM_LAYERS"]
    ).to(CONFIG["DEVICE"])

    checkpoint = torch.load(CONFIG["MODEL_PATH"], map_location=CONFIG["DEVICE"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ### Plot Reconstructions for Top Anomalies ###
    for idx in worst_indices:
        original = sequences[idx:idx+1]  # shape: (1, seq_len, features)
        original_tensor = torch.tensor(original, dtype=torch.float32).to(CONFIG["DEVICE"])

        with torch.no_grad():
            reconstruction = model(original_tensor)[0].cpu().numpy()[0]

        plot_reconstruction_comparison(
            true_seq=original[0, :, :4],
            recon_seq=reconstruction[:, :4],
            feature_names=CONFIG["FEATURES"][:4]
        )

    print("[INFO] Anomaly plots generated.")

    ### Plot Reconstructions for Least Anomalous ###
    bottom_k = 5
    best_indices = np.argsort(scores)[:bottom_k]

    print("[INFO] Low anomaly score indices:", best_indices)
    for i in best_indices:
        print(f"Sequence {i} | Score: {scores[i]:.4f}")

    for idx in best_indices:
        original = sequences[idx:idx+1]
        original_tensor = torch.tensor(original, dtype=torch.float32).to(CONFIG["DEVICE"])
        with torch.no_grad():
            reconstruction = model(original_tensor)[0].cpu().numpy()[0]

        plot_reconstruction_comparison(
            true_seq=original[0, :, :4],
            recon_seq=reconstruction[:, :4],
            feature_names=CONFIG["FEATURES"][:4]
        )
