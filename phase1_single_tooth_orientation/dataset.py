import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

STANDARD_CLASSES = ["Mesial", "Distal", "InnerPoint", "OuterPoint", "FacialPoint"]


class SingleToothPCDataset(Dataset):
    """
    加载 6D 点云数据 (XYZ + Normals)，并提取 FDI 牙位编号
    """

    def __init__(self, data_root: str, augment: bool = True):
        super().__init__()
        self.augment = augment

        # 匹配带有法向量特征的新数据
        self.pc_files = glob.glob(os.path.join(data_root, "**", "*_normals.npy"), recursive=True)

        if len(self.pc_files) == 0:
            print(f"⚠️ 警告: 在 {data_root} 中没有找到任何 *_normals.npy 文件！请检查路径。")

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, idx):
        pc_path = self.pc_files[idx]

        # 定位 JSON 路径 (将 _pc4096_normals.npy 替换为 .json)
        json_path = pc_path.rsplit("_pc", 1)[0] + ".json"

        # ---------------- 1. 提取 FDI 牙位编号 ----------------
        try:
            filename = os.path.basename(pc_path)
            tooth_id_str = filename.split("tooth_")[1].split("_")[0]
            tooth_id = int(tooth_id_str)
        except Exception:
            tooth_id = 0  # 默认未知

        # ---------------- 2. 加载 6D 点云与标签 ----------------
        # data_6d 形状: (N, 6)，前3列是 xyz，后3列是 nx, ny, nz
        data_6d = np.load(pc_path).astype(np.float32)
        points = data_6d[:, :3]
        normals = data_6d[:, 3:]

        with open(json_path, 'r') as f:
            lm_data = json.load(f)
        lm_dict = {item['class']: np.array(item['coord'], dtype=np.float32) for item in lm_data}

        landmarks = np.zeros((5, 3), dtype=np.float32)
        lm_mask = np.zeros(5, dtype=np.float32)

        for i, cls_name in enumerate(STANDARD_CLASSES):
            if cls_name in lm_dict:
                landmarks[i] = lm_dict[cls_name]
                lm_mask[i] = 1.0

        # ---------------- 3. 数据增强 (3D 旋转) ----------------
        if self.augment:
            rot_matrix = R.random().as_matrix().astype(np.float32)

            # 核心：坐标和法向量必须经历完全一样的旋转
            points = points @ rot_matrix.T
            normals = normals @ rot_matrix.T

            valid_idx = np.where(lm_mask == 1.0)[0]
            if len(valid_idx) > 0:
                landmarks[valid_idx] = landmarks[valid_idx] @ rot_matrix.T

        # 重新拼接成 (N, 6)
        features_6d = np.concatenate([points, normals], axis=1)

        return {
            "points": torch.from_numpy(features_6d),
            "landmarks": torch.from_numpy(landmarks),
            "lm_mask": torch.from_numpy(lm_mask),
            "tooth_id": torch.tensor(tooth_id, dtype=torch.long),
            "sid": os.path.basename(pc_path)
        }


def make_loader(data_root: str, batch_size: int, shuffle: bool, num_workers: int = 4,
                augment: bool = True) -> DataLoader:
    ds = SingleToothPCDataset(data_root=data_root, augment=augment)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )


def mean_euclidean_error(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    diff = torch.norm(pred - gt, dim=-1)
    valid_sum = mask.sum()
    if valid_sum == 0:
        return 0.0
    return ((diff * mask).sum() / valid_sum).item()