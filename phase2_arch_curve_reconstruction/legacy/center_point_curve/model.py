import torch
import torch.nn as nn
from dataset import SEQ_LEN


class MaskedArchRegressor(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        # 重新变回 4 维 (X, Y, Z, is_missing) 提取坐标
        self.input_proj = nn.Linear(4, embed_dim)
        self.pos_embed = nn.Embedding(SEQ_LEN, embed_dim)

        # 【救命神技】：为上下颌建立专属的类别 Embedding！
        self.jaw_embed = nn.Embedding(2, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3)
        )

    def forward(self, x):
        B = x.size(0)

        # 将传入的 5 维特征拆解
        coords_and_mask = x[:, :, :4]  # 前 4 列: 坐标与掩码 (B, 16, 4)
        jaw_type = x[:, :, 4].long()  # 第 5 列: 上下颌标识 (B, 16)

        positions = torch.arange(SEQ_LEN, device=x.device).unsqueeze(0).expand(B, SEQ_LEN)

        # 三位一体完美融合：坐标特征 + 牙位排序特征 + 上下颌专属形态特征
        x_emb = self.input_proj(coords_and_mask) + self.pos_embed(positions) + self.jaw_embed(jaw_type)

        out_emb = self.transformer(x_emb)
        preds = self.output_proj(out_emb)
        return preds