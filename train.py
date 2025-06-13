### train.py

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG
from data.load import load_sequences
from models.autoencoder import LSTMAutoencoder
from utils.training import train_autoencoder, get_autoencoder_model, get_dataloaders
from tqdm import tqdm


def train_model(model_type="autoencoder", load=False, window=20, latent_dim=64, epochs=50):
    import time
    import gc

    print(f"[INFO] Training model: {model_type}")
    t0 = time.time()

    # Load input sequences and targets
    sequences, targets = load_sequences(CONFIG["SEQUENCE_FILE"])
    print(f"[TIMER] After load: {time.time() - t0:.2f}s")

    # === Optional Subsampling ===
    subset_fraction = CONFIG.get("SUBSET_FRACTION", 0.1)  # Default to 10%
    if 0 < subset_fraction < 1.0:
        num_total = len(sequences)
        num_keep = int(num_total * subset_fraction)
        print(f"[INFO] Subsampling to {num_keep}/{num_total} sequences ({subset_fraction*100:.0f}%)")
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(num_total, num_keep, replace=False)
        sequences = sequences[indices]
        targets = targets[indices]
    print(f"[TIMER] After subsampling: {time.time() - t0:.2f}s")

    # Compute delta sequences (frame-to-frame differences)
    sequences_deltas = sequences[:, 1:, :] - sequences[:, :-1, :]
    print(f"[TIMER] After delta computation: {time.time() - t0:.2f}s")

    # Drop sequences with any NaNs or Infs
    nan_mask = ~np.isnan(sequences_deltas).any(axis=(1, 2)) & ~np.isinf(sequences_deltas).any(axis=(1, 2))
    sequences_deltas = sequences_deltas[nan_mask]
    targets = targets[nan_mask]

    print(f"[INFO] Filtered to {len(sequences_deltas)} clean sequences (no NaNs or Infs)")
    gc.collect()
    print(f"[TIMER] After filtering: {time.time() - t0:.2f}s")

    # Split into training and validation sets
    seq_train, seq_val, tgt_train, tgt_val = train_test_split(
        sequences_deltas, targets, test_size=0.2, random_state=42
    )
    print(f"[TIMER] After train-test split: {time.time() - t0:.2f}s")

    # Create data loaders
    train_loader, val_loader = get_dataloaders(seq_train, seq_val, tgt_train, tgt_val)
    print(f"[TIMER] After DataLoader setup: {time.time() - t0:.2f}s")

    # Build model
    model = get_autoencoder_model(
        input_dim=sequences_deltas.shape[2],
        hidden_dim=CONFIG["HIDDEN_DIM"],
        latent_dim=latent_dim,
        num_layers=CONFIG["NUM_LAYERS"],
        load=load
    )
    print(f"[TIMER] After model init: {time.time() - t0:.2f}s")

    # Debug: Sequence stats
    print("[DEBUG] NaNs in original sequences:", np.isnan(sequences).any())
    print("[DEBUG] NaNs in delta sequences:", np.isnan(sequences_deltas).any())
    print("[DEBUG] Infs in delta sequences:", np.isinf(sequences_deltas).any())
    print("[DEBUG] Delta sequence stats:")
    print("  min:", np.min(sequences_deltas))
    print("  max:", np.max(sequences_deltas))
    print("  mean:", np.mean(sequences_deltas))
    print("===============================")

    # Train the model
    train_autoencoder(model, train_loader, val_loader=val_loader, num_epochs=epochs)

