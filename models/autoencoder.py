# models/autoencoder.py

### Library Imports and Setup ###
import torch
import torch.nn as nn
from config import CONFIG

### Main Function ###
# models/autoencoder.py
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Explicit dimensions
        self.ego_dim = CONFIG["EGO_DIM"]
        self.context_dim = CONFIG["CONTEXT_DIM"]
        self.static_dim = CONFIG["STATIC_DIM"]

        # Context gating
        self.context_gate = nn.Sequential(
            nn.Linear(self.context_dim, self.context_dim),
            nn.Sigmoid()
        )

        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, bidirectional=True, dropout=0.3)
        self.latent = nn.Linear(hidden_dim * 2, latent_dim)

        # Context summary for conditional decoding
        self.context_summary = nn.Sequential(
            nn.Linear(self.context_dim, 16),
            nn.ReLU()
        )

        # Hidden state init
        self.expand = nn.Linear(CONFIG["LATENT_DIM"] + 16, CONFIG["HIDDEN_DIM"])

        # Decoder input projection (ego_dim → hidden_dim)
        self.decoder_input_proj = nn.Linear(self.ego_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=0.3)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.bias_correction = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # === Split input ===
        ego = x[:, :, :self.ego_dim]
        context = x[:, :, self.ego_dim:self.ego_dim + self.context_dim]
        static = x[:, :, self.ego_dim + self.context_dim:]  # optional

        # === Context gating ===
        gate = self.context_gate(context)
        gated_context = context * gate

        # === Concatenate everything ===
        x_gated = torch.cat([ego, gated_context, static], dim=-1)

        # === Encode ===
        _, (hn, _) = self.encoder(x_gated)
        h_forward = hn[-2, :, :]
        h_backward = hn[-1, :, :]
        h_combined = torch.cat((h_forward, h_backward), dim=1)  # [B, 2H]
        z = self.latent(h_combined)  # [B, latent_dim]

        # === Context summary ===
        context_summary = self.context_summary(gated_context.mean(dim=1))  # [B, 16]

        # === Decoder init ===
        combined_latent = torch.cat([z, context_summary], dim=1)  # [B, latent + 16]

        h_0_expanded = self.expand(combined_latent)                # [B, hidden_dim]
        h_0 = h_0_expanded.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, B, hidden_dim]
        c_0 = torch.zeros_like(h_0)

        # === Decode ===
        decoder_inputs_proj = self.decoder_input_proj(ego)  # [B, T, hidden_dim]
        decoded, _ = self.decoder(decoder_inputs_proj, (h_0, c_0))
        out = self.output_layer(decoded) + self.bias_correction.view(1, 1, -1)  # [B, T, input_dim]

        return out, z
