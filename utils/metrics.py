# utils/metrics.py

import torch
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from config import CONFIG
import os
import joblib

def get_flat_recon_errors(model, dataloader, return_3d=False, cache_path=None):
    """
    Computes reconstruction errors for ego features.
    If return_3d=True, returns shape (N, T, 4), else returns flattened (N, T*4).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached reconstruction errors from {cache_path}")
        return np.load(cache_path)

    model.eval()
    all_errors = []
    device = next(model.parameters()).device

    print("[INFO] Computing reconstruction errors...")
    with torch.no_grad():
        for (batch_x,) in tqdm(dataloader, desc="Processing batches"):
            batch_x = batch_x.to(device)

            if batch_x.size(-1) != 31:
                raise ValueError(
                    f"Expected input with 31 features, got {batch_x.size(-1)}. "
                    "Ensure input includes both ego and context."
                )

            recon, _ = model(batch_x)
            true_ego = batch_x[:, :, :4]
            recon_ego = recon[:, :, :4]
            err = recon_ego - true_ego

            if return_3d:
                all_errors.append(err.cpu().numpy())
            else:
                flat_err = err.cpu().numpy().reshape(err.size(0), -1)
                all_errors.append(flat_err)

    errors = np.concatenate(all_errors, axis=0)
    print(f"[INFO] Reconstruction errors shape: {errors.shape}")

    if cache_path:
        np.save(cache_path, errors)
        print(f"[INFO] Saved reconstruction errors to {cache_path}")

    return errors



def compute_mahalanobis_stats(errors, save_path=None):
    """
    Computes and optionally saves Mahalanobis mean and covariance.
    """
    print("[INFO] Computing mean and covariance of errors for Mahalanobis scoring...")
    mean_vec = np.mean(errors, axis=0)
    cov_matrix = np.cov(errors, rowvar=False)

    # Handle singular matrix
    cov_inv = np.linalg.pinv(cov_matrix)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({"mean": mean_vec, "cov_inv": cov_inv}, save_path)
        print(f"[INFO] Saved Mahalanobis stats to {save_path}")

    return mean_vec, cov_inv


def load_mahalanobis_stats(path):
    """
    Loads Mahalanobis statistics from a saved file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Mahalanobis stats not found at {path}")
    data = joblib.load(path)
    return data["mean"], data["cov_inv"]


def get_mahalanobis_scores(errors, mean_vec, cov_inv, save_path=None):
    """
    Computes Mahalanobis distance for each error vector.
    Optionally saves scores to file.
    """
    print("[INFO] Computing Mahalanobis scores...")
    scores = np.array([distance.mahalanobis(e, mean_vec, cov_inv) for e in errors])

    print(f"[INFO] Mahalanobis scoring complete. Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

    if save_path:
        np.save(save_path, scores)
        print(f"[INFO] Saved Mahalanobis scores to {save_path}")

    return scores
