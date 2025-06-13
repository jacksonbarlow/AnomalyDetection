import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import CONFIG
from models.autoencoder import LSTMAutoencoder
from tqdm import tqdm
import numpy as np
from utils.visualisation import save_reconstruction_plot

def train_autoencoder(model, train_loader, val_loader=None, num_epochs=10, lr=1e-3):
    device = next(model.parameters()).device
    optimiser = optim.Adam(model.parameters(), lr=lr)
    checkpoint_dir = CONFIG["CHECKPOINT_DIR"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        weights = None
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for i, (batch_x, _) in enumerate(pbar):
            batch_x = batch_x.to(device)

            if weights is None:
                std = batch_x[:, :, :4].reshape(-1, 4).std(dim=0)
                weights = 1.0 / (std + 1e-6)
                weights = weights.view(1, 1, -1).to(device)

            optimiser.zero_grad()
            recon, _ = model(batch_x)

            diff = recon[:, :, :4] - batch_x[:, :, :4]
            weighted_diff = diff * weights
            loss = nn.functional.smooth_l1_loss(weighted_diff, torch.zeros_like(weighted_diff))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            total_loss += loss.item()
            pbar.set_postfix(train_loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)

        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_x, _ in val_loader:
                    val_x = val_x.to(device)
                    val_recon, _ = model(val_x)
                    val_diff = val_recon[:, :, :4] - val_x[:, :, :4]
                    val_weighted = val_diff * weights
                    val_loss += nn.functional.smooth_l1_loss(val_weighted, torch.zeros_like(val_weighted)).item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            path = os.path.join(checkpoint_dir, f"autoencoder_epoch{epoch+1}.pt")
            torch.save({'model_state_dict': model.state_dict()}, path)
            print(f"[INFO] Saved checkpoint: {path}")

        # Save recon plot every N epochs
        if val_loader and (epoch + 1) % 10 == 0:
            model.eval()
            val_batch, _ = next(iter(val_loader))
            rand_idx = torch.randint(0, val_batch.size(0), (1,)).item()
            input_seq = val_batch[rand_idx].unsqueeze(0).to(device)
            output_seq = model(input_seq)[0]  # tensor with grad
            recon_seq = output_seq.detach().cpu().numpy()[0, :, :4]
            true_seq = input_seq[0, :, :4].cpu().numpy()
            save_path = os.path.join(CONFIG["EVALUATION_PLOTS_DIR"], f"reconstruction_epoch{epoch+1}.png")
            save_reconstruction_plot(true_seq, recon_seq[:, :4], CONFIG["FEATURES"][:4], save_path)

    # Final checkpoint on completion or interrupt
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, f"autoencoder_final.pt"))
    print(f"[INFO] Final model saved to {checkpoint_dir}/autoencoder_final.pt")

def get_dataloaders(seq_train, seq_val, tgt_train, tgt_val, batch_size=64):
    train_dataset = TensorDataset(
        torch.tensor(seq_train, dtype=torch.float32),
        torch.tensor(tgt_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(seq_val, dtype=torch.float32),
        torch.tensor(tgt_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_autoencoder_model(input_dim, hidden_dim, latent_dim, num_layers, load=False):
    model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, num_layers).to(CONFIG["DEVICE"])
    path = CONFIG["MODEL_PATH"]

    if load and os.path.exists(path):
        checkpoint = torch.load(path, map_location=CONFIG["DEVICE"])
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[INFO] Loaded model from {path}")
    elif load:
        print(f"[WARNING] No checkpoint found at {path}; training from scratch.")

    return model
