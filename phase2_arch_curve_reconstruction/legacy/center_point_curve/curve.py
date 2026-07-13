import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import KDTree
import random

# ==========================================
# 1. 牙位与序列映射配置 (以上颌 Upper 为例)
# 上颌 FDI 编号: 右侧 18-11, 左侧 21-28
# 我们将其映射为长度为 16 的固定序列索引 (0-15)
# ==========================================
UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_TO_IDX = {fdi: idx for idx, fdi in enumerate(UPPER_FDI)}
SEQ_LEN = len(UPPER_FDI)
TARGET_LANDMARKS = ["Mesial", "Distal", "OuterPoint", "InnerPoint"]  # 近中, 远中, 颊侧, 舌侧


# ==========================================
# 2. 数据提取与预处理 (离线执行一次即可)
# ==========================================
def extract_arch_centers(obj_path, label_path, kpt_path):
    """
    基于你之前的 KDTree 逻辑，从单个牙弓中提取每颗牙的 4 个关键点，并计算几何中心。
    返回: shape (16, 3) 的坐标数组, 以及一个表明该位置是否有牙齿的 valid_mask (16,)
    """
    import trimesh
    mesh = trimesh.load(obj_path, process=False)

    with open(label_path, 'r') as f:
        labels = np.array(json.load(f)['labels'])
    with open(kpt_path, 'r') as f:
        landmarks = json.load(f).get('objects', [])

    kdtree = KDTree(mesh.vertices)

    # 将地标按牙位分组
    tooth_lms = {}
    for lm in landmarks:
        coord = np.array(lm['coord'])
        _, idx = kdtree.query(coord)
        tooth_id = labels[idx]
        if tooth_id in UPPER_FDI and lm['class'] in TARGET_LANDMARKS:
            if tooth_id not in tooth_lms:
                tooth_lms[tooth_id] = []
            tooth_lms[tooth_id].append(coord)

    # 初始化序列 (16, 3) 和有效掩码
    arch_centers = np.zeros((SEQ_LEN, 3), dtype=np.float32)
    valid_mask = np.zeros(SEQ_LEN, dtype=bool)

    # 计算几何中心
    for tooth_id, coords in tooth_lms.items():
        # 只要能提取到关键点（不论是几个，理想是4个），就求平均作为几何中心
        if len(coords) > 0:
            center = np.mean(coords, axis=0)
            seq_idx = FDI_TO_IDX[tooth_id]
            arch_centers[seq_idx] = center
            valid_mask[seq_idx] = True

    return arch_centers, valid_mask


# ==========================================
# 3. 动态掩码数据集 (Dynamic Masking Dataset)
# ==========================================
class DentalArchDataset(Dataset):
    def __init__(self, data_list, is_train=True):
        """
        data_list: 预处理好的数据列表，格式为 [(centers, valid_mask), ...]
        """
        self.data_list = data_list
        self.is_train = is_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        centers, valid_mask = self.data_list[idx]
        centers = centers.copy()

        # 记录哪些牙齿被我们人为“拔掉”了（用于计算Loss）
        dropped_mask = np.zeros(SEQ_LEN, dtype=bool)

        if self.is_train:
            # 找到当前牙弓中真正存在的牙齿索引
            existing_indices = np.where(valid_mask)[0]

            # 随机决定拔掉 1 到 3 颗牙
            num_to_drop = random.randint(1, min(3, len(existing_indices) - 1))
            drop_indices = random.sample(list(existing_indices), num_to_drop)

            for d_idx in drop_indices:
                dropped_mask[d_idx] = True
                centers[d_idx] = [0.0, 0.0, 0.0]  # 坐标归零，模拟缺失

        # 构建输入特征: (X, Y, Z, is_missing_flag) -> (16, 4)
        # is_missing_flag: 1 表示这个位置当前没有有效坐标（天然缺失或被人为拔掉）
        is_missing = (~valid_mask | dropped_mask).astype(np.float32)
        features = np.concatenate([centers, is_missing[:, None]], axis=-1)

        # 需要返回 Ground Truth 以计算 Loss
        gt_centers = self.data_list[idx][0].copy()

        return {
            "features": torch.tensor(features, dtype=torch.float32),  # (16, 4)
            "gt_centers": torch.tensor(gt_centers, dtype=torch.float32),  # (16, 3)
            "dropped_mask": torch.tensor(dropped_mask, dtype=torch.bool),  # (16,)
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool)  # (16,)
        }


# ==========================================
# 4. 轻量化 Transformer 模型
# ==========================================
class MaskedArchRegressor(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        # 输入维度是 4 (x, y, z, is_missing)
        self.input_proj = nn.Linear(4, embed_dim)

        # 牙位序列编码 (16 个固定位置)
        self.pos_embed = nn.Embedding(SEQ_LEN, embed_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出回归头 (预测 X, Y, Z)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3)
        )

    def forward(self, x):
        # x shape: (B, 16, 4)
        B = x.size(0)

        # 获取位置向量
        positions = torch.arange(SEQ_LEN, device=x.device).unsqueeze(0).expand(B, SEQ_LEN)
        pos_emb = self.pos_embed(positions)

        # 投影并注入位置信息
        x_emb = self.input_proj(x) + pos_emb

        # Transformer 交互上下文
        out_emb = self.transformer(x_emb)

        # 回归三维坐标
        preds = self.output_proj(out_emb)  # (B, 16, 3)
        return preds


# ==========================================
# 5. 训练与测试主流程
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 模拟数据加载 (你需要替换为实际的文件夹遍历逻辑)
    # ---------------------------------------------------------
    print("Extracting centers from raw data...")
    # TODO: 这里写一个循环遍历你的 120 个 Upper 牙弓文件夹
    # 伪代码：
    # all_arch_data = []
    # for folder in upper_folders:
    #     centers, mask = extract_arch_centers(obj, label, kpt)
    #     all_arch_data.append((centers, mask))

    # ---------------------------------------------------------
    # 真实数据加载逻辑 (替换之前的模拟数据生成部分)
    # ---------------------------------------------------------
    print("开始从本地路径提取真实的牙弓关键点数据...")

    # 你的真实根目录路径 (注意字符串前面的 r，防止转义)
    SEG_ROOT = r"F:\NDCS_3DS_data\segmentation_data_for_single_teeth\train\upper"
    KPT_ROOT = r"F:\NDCS_3DS_data\3DTeethLand_landmarks_train\upper"

    all_arch_data = []

    # 确保路径存在
    if not os.path.exists(SEG_ROOT) or not os.path.exists(KPT_ROOT):
        print("🚨 错误: 找不到指定的数据文件夹，请检查路径！")
        return

    # 遍历 upper 目录下的所有病人文件夹 (如 '01A6GW4A')
    for patient_id in os.listdir(SEG_ROOT):
        patient_seg_dir = os.path.join(SEG_ROOT, patient_id)
        patient_kpt_dir = os.path.join(KPT_ROOT, patient_id)

        # 只处理文件夹
        if not os.path.isdir(patient_seg_dir):
            continue

        # 严格拼接三个文件的绝对路径
        obj_path = os.path.join(patient_seg_dir, f"{patient_id}_upper.obj")
        label_path = os.path.join(patient_seg_dir, f"{patient_id}_upper.json")
        kpt_path = os.path.join(patient_kpt_dir, f"{patient_id}_upper__kpt.json")

        # 检查文件是否齐全
        if os.path.exists(obj_path) and os.path.exists(label_path) and os.path.exists(kpt_path):
            try:
                # 调用我们在第二步写的特征提取函数
                centers, valid_mask = extract_arch_centers(obj_path, label_path, kpt_path)

                # 过滤掉完全提取不到关键点的异常牙弓
                if valid_mask.sum() > 0:
                    all_arch_data.append((centers, valid_mask))
                else:
                    print(f"⚠️ 警告: {patient_id} 没有提取到任何有效目标关键点，已跳过。")
            except Exception as e:
                print(f"❌ 错误: 处理 {patient_id} 时发生异常 ({e})")
        else:
            print(f"⚠️ 警告: {patient_id} 的文件不全，已跳过。")

    print(f"✅ 数据加载完成！成功读取 {len(all_arch_data)} 个上颌牙弓。")

    if len(all_arch_data) == 0:
        print("🚨 没有读取到有效数据，程序退出。")
        return

    # ---------------------------------------------------------
    # 划分训练集和测试集 (80% / 20%)
    # ---------------------------------------------------------
    random.shuffle(all_arch_data)
    split_idx = int(len(all_arch_data) * 0.8)
    train_data = all_arch_data[:split_idx]
    test_data = all_arch_data[split_idx:]

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    train_loader = DataLoader(DentalArchDataset(train_data, is_train=True), batch_size=16, shuffle=True)
    test_loader = DataLoader(DentalArchDataset(test_data, is_train=False), batch_size=16, shuffle=False)

    # ---------------------------------------------------------
    # 初始化模型、优化器和损失函数
    # ---------------------------------------------------------
    model = MaskedArchRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction='none')  # 使用 none 以便应用 mask

    epochs = 100

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            gt_centers = batch["gt_centers"].to(device)
            dropped_mask = batch["dropped_mask"].to(device)  # 只计算被人为拔掉的牙齿的Loss

            optimizer.zero_grad()
            preds = model(features)

            # 计算 MSE 损失
            loss_matrix = criterion(preds, gt_centers)  # (B, 16, 3)
            loss_per_tooth = loss_matrix.mean(dim=-1)  # (B, 16)

            # [核心] 掩码损失：只统计人为 drop 掉的位置
            valid_loss = loss_per_tooth[dropped_mask]

            if len(valid_loss) > 0:
                loss = valid_loss.mean()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

        # ---------------------------------------------------------
        # 测试 / 验证流程
        # 注意：测试集通常不需要 mask，或者采用固定的 mask 策略来评估
        # 这里为了验证泛化能力，我们对测试集手动 drop 每副牙齿的第 3 和 10 颗牙进行评估
        # ---------------------------------------------------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0.0
            test_batches = 0

            with torch.no_grad():
                for batch in test_loader:
                    # 获取真实数据
                    test_centers = batch["gt_centers"].clone()
                    test_valid_mask = batch["valid_mask"]

                    # 强行掩码测试集中的几个位置 (模拟缺失)
                    test_dropped_mask = torch.zeros_like(test_valid_mask)
                    # 假设我们固定测试预测索引为 4 和 12 的牙齿
                    test_dropped_mask[:, [4, 12]] = True
                    test_dropped_mask = test_dropped_mask & test_valid_mask  # 确保原本这颗牙是存在的

                    test_centers[test_dropped_mask] = 0.0

                    is_missing = (~test_valid_mask | test_dropped_mask).float()
                    test_features = torch.cat([test_centers, is_missing.unsqueeze(-1)], dim=-1).to(device)

                    preds = model(test_features)

                    t_loss_matrix = criterion(preds, batch["gt_centers"].to(device)).mean(dim=-1)
                    t_valid_loss = t_loss_matrix[test_dropped_mask.to(device)]

                    if len(t_valid_loss) > 0:
                        test_loss += t_valid_loss.mean().item()
                        test_batches += 1

            avg_test_loss = test_loss / test_batches if test_batches > 0 else 0.0
            print(
                f"Epoch {epoch + 1:03d} | Train Masked Loss: {total_train_loss / len(train_loader):.4f} | Test Loss (Idx 4,12): {avg_test_loss:.4f}")


if __name__ == "__main__":
    main()