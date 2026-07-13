import torch
import torch.nn as nn


class TransitionDown(nn.Module):
    """随机下采样，通过 1x1 卷积升维特征"""

    def __init__(self, in_c: int, out_c: int, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.mlp = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        B, _, N = xyz.shape
        if N <= self.stride:
            idx = torch.arange(N, device=xyz.device)
        else:
            k = max(1, N // self.stride)
            idx = torch.randperm(N, device=xyz.device)[:k]

        xyz_out = xyz[:, :, idx]
        feat_out = feat[:, :, idx]
        feat_out = self.mlp(feat_out)
        return xyz_out, feat_out


class PointTransformerLayer(nn.Module):
    """Multi-Head Self-Attention (MHSA) 层"""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xyz, feat):
        B, C, N = feat.shape
        x = feat.permute(0, 2, 1)  # (B, N, C)
        shortcut = x

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def reshape_heads(t):
            return t.view(B, N, self.heads, C // self.heads).transpose(1, 2)

        q, k, v = map(reshape_heads, (q, k, v))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)

        out = self.relu(self.norm(out + shortcut))
        return xyz, out.permute(0, 2, 1)


class LandmarkPointTransformer(nn.Module):
    """单齿朝向坐标回归网络 (6D 输入 + FDI 先验)"""

    def __init__(self, num_landmarks: int = 5, embed_dim: int = 96):
        super().__init__()

        # 1. 第一层输入通道由 3 变为 6 (XYZ + Normal)
        self.sa1 = TransitionDown(6, embed_dim, stride=4)
        self.sa2 = TransitionDown(embed_dim, embed_dim * 2, stride=4)
        self.sa3 = TransitionDown(embed_dim * 2, embed_dim * 4, stride=4)

        self.pt1 = PointTransformerLayer(embed_dim * 4, heads=4)
        self.pt2 = PointTransformerLayer(embed_dim * 4, heads=4)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 2. 牙位 Embedding
        self.tooth_emb = nn.Embedding(50, 64)

        # 3. 融合后的回归头: 384(点云) + 64(牙位) = 448
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * 4 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_landmarks * 3)
        )
        self.num_landmarks = num_landmarks

    def forward(self, pts: torch.Tensor, tooth_id: torch.Tensor):
        # pts: (B, N, 6)

        # 提取前 3 维作为空间参考 (B, 3, N)
        xyz = pts[:, :, :3].permute(0, 2, 1)

        # 提取全部 6 维作为初始特征 (B, 6, N)
        feat = pts.permute(0, 2, 1)

        xyz, feat = self.sa1(xyz, feat)
        xyz, feat = self.sa2(xyz, feat)
        xyz, feat = self.sa3(xyz, feat)
        xyz, feat = self.pt1(xyz, feat)
        xyz, feat = self.pt2(xyz, feat)

        pc_feat = self.global_pool(feat).squeeze(-1)  # (B, 384)
        t_feat = self.tooth_emb(tooth_id)  # (B, 64)

        combined_feat = torch.cat([pc_feat, t_feat], dim=1)  # (B, 448)
        out = self.fc(combined_feat)

        return out.view(-1, self.num_landmarks, 3)