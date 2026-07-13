import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
from scipy.spatial import KDTree
import random

# ==========================================
# 统一序列配置 (支持上颌和下颌映射到同一个 16 长度的序列)
# ==========================================
UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

# 统一映射字典 (无论上下颌，都映射到 0-15 的位置)
FDI_TO_IDX = {fdi: idx for idx, fdi in enumerate(UPPER_FDI)}
FDI_TO_IDX.update({fdi: idx for idx, fdi in enumerate(LOWER_FDI)})

SEQ_LEN = 16
TARGET_LANDMARKS = ["Mesial", "Distal", "OuterPoint", "InnerPoint"]


def extract_arch_centers(obj_path, label_path, kpt_path, jaw_type):
    """提取关键点并计算几何中心"""
    mesh = trimesh.load(obj_path, process=False)
    with open(label_path, 'r') as f:
        labels = np.array(json.load(f)['labels'])
    with open(kpt_path, 'r') as f:
        landmarks = json.load(f).get('objects', [])

    kdtree = KDTree(mesh.vertices)
    tooth_lms = {}
    valid_fdi_set = UPPER_FDI if jaw_type == 0 else LOWER_FDI

    for lm in landmarks:
        coord = np.array(lm['coord'])
        _, idx = kdtree.query(coord)
        tooth_id = labels[idx]
        if tooth_id in valid_fdi_set and lm['class'] in TARGET_LANDMARKS:
            if tooth_id not in tooth_lms:
                tooth_lms[tooth_id] = []
            tooth_lms[tooth_id].append(coord)

    arch_centers = np.zeros((SEQ_LEN, 3), dtype=np.float32)
    valid_mask = np.zeros(SEQ_LEN, dtype=bool)

    for tooth_id, coords in tooth_lms.items():
        if len(coords) > 0:
            center = np.mean(coords, axis=0)
            seq_idx = FDI_TO_IDX[tooth_id]
            arch_centers[seq_idx] = center
            valid_mask[seq_idx] = True

    return arch_centers, valid_mask


def load_real_data(seg_root_base, kpt_root_base):
    """同时加载 upper 和 lower 文件夹的数据"""
    all_arch_data = []
    patient_ids = []

    # 0 代表 Upper, 1 代表 Lower
    jaw_types = [('upper', 0), ('lower', 1)]

    for jaw_name, jaw_val in jaw_types:
        seg_dir = os.path.join(seg_root_base, jaw_name)
        kpt_dir = os.path.join(kpt_root_base, jaw_name)

        if not os.path.exists(seg_dir): continue

        for patient_id in os.listdir(seg_dir):
            patient_seg_dir = os.path.join(seg_dir, patient_id)
            patient_kpt_dir = os.path.join(kpt_dir, patient_id)

            if not os.path.isdir(patient_seg_dir): continue

            obj_path = os.path.join(patient_seg_dir, f"{patient_id}_{jaw_name}.obj")
            label_path = os.path.join(patient_seg_dir, f"{patient_id}_{jaw_name}.json")
            kpt_path = os.path.join(patient_kpt_dir, f"{patient_id}_{jaw_name}__kpt.json")

            if os.path.exists(obj_path) and os.path.exists(label_path) and os.path.exists(kpt_path):
                centers, valid_mask = extract_arch_centers(obj_path, label_path, kpt_path, jaw_val)
                if valid_mask.sum() > 0:
                    # 连同 jaw_val 一起保存
                    all_arch_data.append((centers, valid_mask, jaw_val))
                    # 区分上下颌的文件名，防止同病人重名
                    patient_ids.append(f"{patient_id}_{jaw_name}")

    return all_arch_data, patient_ids


class DentalArchDataset(Dataset):
    def __init__(self, data_list, is_train=True):
        self.data_list = data_list
        self.is_train = is_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        centers, valid_mask, jaw_val = self.data_list[idx]
        centers = centers.copy()

        # 3D 数据增强
        if self.is_train:
            translation = np.random.uniform(-2.0, 2.0, size=(1, 3))
            centers += translation
            angle_rad = np.radians(np.random.uniform(-5.0, 5.0))
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            centers = np.dot(centers, rot_z.T)

        dropped_mask = np.zeros(SEQ_LEN, dtype=bool)
        if self.is_train:
            existing_indices = np.where(valid_mask)[0]
            if len(existing_indices) > 2:
                num_to_drop = random.randint(1, min(3, len(existing_indices) - 2))
                drop_indices = random.sample(list(existing_indices), num_to_drop)
                for d_idx in drop_indices:
                    dropped_mask[d_idx] = True
                    centers[d_idx] = [0.0, 0.0, 0.0]

        is_missing = (~valid_mask | dropped_mask).astype(np.float32)

        # 构建 5 维输入特征: (X, Y, Z, is_missing, jaw_type)
        jaw_feature = np.full((SEQ_LEN, 1), jaw_val, dtype=np.float32)
        features = np.concatenate([centers, is_missing[:, None], jaw_feature], axis=-1)

        gt_centers = self.data_list[idx][0].copy()
        if self.is_train:
            gt_centers += translation
            gt_centers = np.dot(gt_centers, rot_z.T)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "gt_centers": torch.tensor(gt_centers, dtype=torch.float32),
            "dropped_mask": torch.tensor(dropped_mask, dtype=torch.bool),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool)
        }