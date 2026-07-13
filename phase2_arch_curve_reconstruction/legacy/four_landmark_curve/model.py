import torch
import torch.nn as nn

from dataset import LANDMARK_COORD_DIM, NUM_LANDMARKS, SEQ_LEN


class MaskedArchRegressor(nn.Module):
    """Predict four ordered landmarks for every tooth position."""

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        num_layers=6,
        feedforward_dim=512,
    ):
        super().__init__()
        # Coordinates for four landmarks plus one tooth-level missing flag.
        self.input_proj = nn.Linear(LANDMARK_COORD_DIM + 1, embed_dim)
        self.pos_embed = nn.Embedding(SEQ_LEN, embed_dim)
        self.jaw_embed = nn.Embedding(2, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, LANDMARK_COORD_DIM),
        )

    def forward(self, features):
        batch_size = features.size(0)
        coords_and_mask = features[:, :, : LANDMARK_COORD_DIM + 1]
        jaw_type = features[:, :, LANDMARK_COORD_DIM + 1].long()
        positions = torch.arange(SEQ_LEN, device=features.device)
        positions = positions.unsqueeze(0).expand(batch_size, SEQ_LEN)

        embeddings = (
            self.input_proj(coords_and_mask)
            + self.pos_embed(positions)
            + self.jaw_embed(jaw_type)
        )
        encoded = self.transformer(embeddings)
        predictions = self.output_proj(encoded)
        return predictions.reshape(batch_size, SEQ_LEN, NUM_LANDMARKS, 3)
