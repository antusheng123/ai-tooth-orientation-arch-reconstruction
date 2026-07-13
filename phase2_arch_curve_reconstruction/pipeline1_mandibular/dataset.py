import json
import os
import random

import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree
from torch.utils.data import Dataset


UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
FDI_TO_IDX = {fdi: idx for idx, fdi in enumerate(UPPER_FDI)}
FDI_TO_IDX.update({fdi: idx for idx, fdi in enumerate(LOWER_FDI)})

SEQ_LEN = 16
LANDMARK_NAMES = ("Point0", "Point1")
LANDMARK_TO_IDX = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}
NUM_LANDMARKS = len(LANDMARK_NAMES)
LANDMARK_COORD_DIM = NUM_LANDMARKS * 3

SOURCE_LANDMARK_NAMES = ("Mesial", "Distal", "OuterPoint", "Cusp")
ANTERIOR_FDI = {13, 12, 11, 21, 22, 23, 43, 42, 41, 31, 32, 33}
POSTERIOR_FDI = {
    18,
    17,
    16,
    15,
    14,
    24,
    25,
    26,
    27,
    28,
    48,
    47,
    46,
    45,
    44,
    34,
    35,
    36,
    37,
    38,
}

BUCCAL_CUSP_MIN_SCORE = 0.0
BUCCAL_CUSP_MIN_GAP_RATIO = 0.15
SYNTHETIC_CUSP_MIN_OFFSET_RATIO = 0.25


def _select_reliable_buccal_cusps(cusp_points, outer_point, tooth_vertices):
    """
    Select two cusps on the buccal side of a posterior tooth.

    OuterPoint defines the buccal direction from the tooth mesh center. Cusps
    are ranked by projection onto that direction, not by Euclidean distance to
    OuterPoint. If the second cusp is not confidently buccal, return None so
    the tooth is treated as incomplete rather than mislabeled.
    """
    cusp_points = np.asarray(cusp_points, dtype=np.float32)
    tooth_vertices = np.asarray(tooth_vertices, dtype=np.float32)
    if cusp_points.shape[0] < NUM_LANDMARKS or tooth_vertices.size == 0:
        return None

    tooth_center = tooth_vertices.mean(axis=0)
    buccal_axis = np.asarray(outer_point, dtype=np.float32) - tooth_center
    axis_norm = float(np.linalg.norm(buccal_axis))
    if not np.isfinite(axis_norm) or axis_norm < 1e-6:
        return None
    buccal_axis = buccal_axis / axis_norm

    scores = (cusp_points - tooth_center) @ buccal_axis
    if not np.isfinite(scores).all():
        return None

    ranked_indices = np.argsort(-scores, kind="stable")
    top_indices = ranked_indices[:NUM_LANDMARKS]
    top_scores = scores[top_indices]

    if top_scores[-1] <= BUCCAL_CUSP_MIN_SCORE:
        return None

    if len(ranked_indices) > NUM_LANDMARKS:
        score_range = float(scores[ranked_indices[0]] - scores[ranked_indices[-1]])
        required_gap = BUCCAL_CUSP_MIN_GAP_RATIO * max(score_range, 1e-6)
        actual_gap = float(scores[ranked_indices[1]] - scores[ranked_indices[2]])
        if actual_gap < required_gap:
            return None

    return cusp_points[top_indices]


def _select_most_buccal_cusp(cusp_points, outer_point, tooth_vertices):
    """Return the single Cusp point most aligned with the buccal direction."""
    cusp_points = np.asarray(cusp_points, dtype=np.float32)
    tooth_vertices = np.asarray(tooth_vertices, dtype=np.float32)
    if cusp_points.shape[0] == 0 or tooth_vertices.size == 0:
        return None

    tooth_center = tooth_vertices.mean(axis=0)
    buccal_axis = np.asarray(outer_point, dtype=np.float32) - tooth_center
    axis_norm = float(np.linalg.norm(buccal_axis))
    if not np.isfinite(axis_norm) or axis_norm < 1e-6:
        return None
    buccal_axis = buccal_axis / axis_norm

    scores = (cusp_points - tooth_center) @ buccal_axis
    if not np.isfinite(scores).all():
        return None

    best_index = int(np.argmax(scores))
    if scores[best_index] <= BUCCAL_CUSP_MIN_SCORE:
        return None
    return cusp_points[best_index]


def _estimate_lateral_axis(mesial_point, distal_point):
    """Use the Mesial-Distal connection as the mirror plane normal."""
    lateral_axis = np.asarray(distal_point, dtype=np.float32) - np.asarray(
        mesial_point,
        dtype=np.float32,
    )
    axis_norm = float(np.linalg.norm(lateral_axis))
    if not np.isfinite(axis_norm) or axis_norm < 1e-6:
        return None
    return lateral_axis.astype(np.float32) / axis_norm


def _mirror_single_buccal_cusp(
    cusp_point,
    outer_point,
    tooth_vertices,
    mesial_point,
    distal_point,
):
    """
    Synthesize a second buccal cusp by mirroring one annotated cusp.

    The mirror plane passes through the tooth mesh center. Its normal is the
    Mesial-Distal connection. The generated point keeps the original cusp's
    non-lateral component and flips only the Mesial-Distal component.
    """
    tooth_vertices = np.asarray(tooth_vertices, dtype=np.float32)
    if tooth_vertices.size == 0:
        return None

    cusp_point = np.asarray(cusp_point, dtype=np.float32)
    tooth_center = tooth_vertices.mean(axis=0)
    buccal_axis = np.asarray(outer_point, dtype=np.float32) - tooth_center
    buccal_norm = float(np.linalg.norm(buccal_axis))
    if not np.isfinite(buccal_norm) or buccal_norm < 1e-6:
        return None
    buccal_axis = buccal_axis / buccal_norm

    cusp_vector = cusp_point - tooth_center
    buccal_score = float(np.dot(cusp_vector, buccal_axis))
    if not np.isfinite(buccal_score) or buccal_score <= BUCCAL_CUSP_MIN_SCORE:
        return None

    lateral_axis = _estimate_lateral_axis(mesial_point, distal_point)
    if lateral_axis is None:
        return None

    vertex_offsets = (tooth_vertices - tooth_center) @ lateral_axis
    typical_offset = float(np.percentile(np.abs(vertex_offsets), 75))
    lateral_offset = float(np.dot(cusp_vector, lateral_axis))
    min_offset = SYNTHETIC_CUSP_MIN_OFFSET_RATIO * max(typical_offset, 1e-6)
    if abs(lateral_offset) < min_offset:
        lateral_offset = min_offset if lateral_offset >= 0.0 else -min_offset

    residual = cusp_vector - np.dot(cusp_vector, lateral_axis) * lateral_axis
    mirrored = tooth_center + residual - lateral_offset * lateral_axis
    if not np.isfinite(mirrored).all():
        return None

    return np.stack([cusp_point, mirrored.astype(np.float32)], axis=0)


def extract_arch_landmarks(obj_path, label_path, kpt_path, jaw_type):
    """
    Return ordered Pipeline 1 tooth landmarks with shape (16, 2, 3).

    Anterior teeth use Mesial and Distal. Posterior teeth prefer two reliable
    buccal Cusp points. If only one Cusp is judged to be on the buccal side,
    a second buccal point is synthesized by mirroring that point with the
    Mesial-Distal mirror plane. Teeth that still cannot produce two points remain
    zero and are excluded from supervision.
    """
    mesh = trimesh.load(obj_path, process=False)
    with open(label_path, "r", encoding="utf-8") as file:
        labels = np.asarray(json.load(file)["labels"])
    with open(kpt_path, "r", encoding="utf-8") as file:
        landmark_objects = json.load(file).get("objects", [])

    if len(labels) != len(mesh.vertices):
        raise ValueError(
            f"Vertex/label count mismatch: {len(mesh.vertices)} vertices, "
            f"{len(labels)} labels"
        )

    ordered_fdi = UPPER_FDI if jaw_type == 0 else LOWER_FDI
    valid_fdi = set(ordered_fdi)
    kdtree = KDTree(mesh.vertices)
    grouped = {}
    tooth_vertices_by_fdi = {
        tooth_id: mesh.vertices[labels == tooth_id] for tooth_id in valid_fdi
    }

    for landmark in landmark_objects:
        class_name = landmark.get("class")
        coord = landmark.get("coord")
        if class_name not in SOURCE_LANDMARK_NAMES or coord is None:
            continue

        coord = np.asarray(coord, dtype=np.float32)
        if coord.shape != (3,) or not np.isfinite(coord).all():
            continue

        _, vertex_idx = kdtree.query(coord)
        tooth_id = int(labels[vertex_idx])
        if tooth_id not in valid_fdi:
            continue

        tooth_classes = grouped.setdefault(tooth_id, {})
        tooth_classes.setdefault(class_name, []).append(coord)

    landmarks = np.zeros((SEQ_LEN, NUM_LANDMARKS, 3), dtype=np.float32)
    landmark_valid_mask = np.zeros((SEQ_LEN, NUM_LANDMARKS), dtype=bool)

    for tooth_id in ordered_fdi:
        class_points = grouped.get(tooth_id, {})
        tooth_idx = FDI_TO_IDX[tooth_id]
        if tooth_id in ANTERIOR_FDI:
            if "Mesial" not in class_points or "Distal" not in class_points:
                continue
            landmarks[tooth_idx, 0] = np.mean(class_points["Mesial"], axis=0)
            landmarks[tooth_idx, 1] = np.mean(class_points["Distal"], axis=0)
            landmark_valid_mask[tooth_idx] = True
        elif tooth_id in POSTERIOR_FDI:
            cusp_points = class_points.get("Cusp", [])
            if "OuterPoint" not in class_points or len(cusp_points) == 0:
                continue
            outer_point = np.mean(class_points["OuterPoint"], axis=0)
            tooth_vertices = tooth_vertices_by_fdi.get(tooth_id, np.empty((0, 3)))
            selected_cusps = None
            if len(cusp_points) >= NUM_LANDMARKS:
                selected_cusps = _select_reliable_buccal_cusps(
                    cusp_points,
                    outer_point,
                    tooth_vertices,
                )
            if selected_cusps is None:
                most_buccal_cusp = _select_most_buccal_cusp(
                    cusp_points,
                    outer_point,
                    tooth_vertices,
                )
                if most_buccal_cusp is None:
                    continue
                if "Mesial" not in class_points or "Distal" not in class_points:
                    continue
                mesial_point = np.mean(class_points["Mesial"], axis=0)
                distal_point = np.mean(class_points["Distal"], axis=0)
                selected_cusps = _mirror_single_buccal_cusp(
                    most_buccal_cusp,
                    outer_point,
                    tooth_vertices,
                    mesial_point,
                    distal_point,
                )
            if selected_cusps is None:
                continue
            landmarks[tooth_idx] = selected_cusps
            landmark_valid_mask[tooth_idx] = True

    tooth_valid_mask = landmark_valid_mask.all(axis=1)
    landmarks[~tooth_valid_mask] = 0.0
    landmark_valid_mask[~tooth_valid_mask] = False
    return landmarks, tooth_valid_mask, landmark_valid_mask


def compute_normalization(landmarks, landmark_valid_mask):
    """Compute one translation and one isotropic scale for an entire arch."""
    valid_points = landmarks[landmark_valid_mask]
    if len(valid_points) == 0:
        raise ValueError("Cannot normalize an arch without valid landmarks")

    origin = valid_points.mean(axis=0).astype(np.float32)
    centered = valid_points - origin
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=-1))))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError(f"Invalid normalization scale: {scale}")
    return origin, np.float32(scale)


def normalize_landmarks(landmarks, landmark_valid_mask, origin=None, scale=None):
    """Normalize valid points while keeping all invalid entries exactly zero."""
    if origin is None or scale is None:
        origin, scale = compute_normalization(landmarks, landmark_valid_mask)

    normalized = np.zeros_like(landmarks, dtype=np.float32)
    normalized[landmark_valid_mask] = (
        landmarks[landmark_valid_mask] - np.asarray(origin, dtype=np.float32)
    ) / np.float32(scale)
    return normalized, np.asarray(origin, dtype=np.float32), np.float32(scale)


def denormalize_landmarks(landmarks, origin, scale, landmark_valid_mask=None):
    """Map normalized landmarks back to world coordinates."""
    restored = landmarks.astype(np.float32, copy=True) * np.float32(scale)
    restored += np.asarray(origin, dtype=np.float32)
    if landmark_valid_mask is not None:
        restored[~landmark_valid_mask] = 0.0
    return restored


def landmarks_to_centers(landmarks, tooth_valid_mask=None):
    """Derive one geometric midpoint per tooth from its two ordered points."""
    centers = landmarks.mean(axis=-2).astype(np.float32)
    if tooth_valid_mask is not None:
        centers = centers.copy()
        centers[~tooth_valid_mask] = 0.0
    return centers


def _rotation_matrix_xyz(angles_rad):
    x_angle, y_angle, z_angle = angles_rad
    cx, sx = np.cos(x_angle), np.sin(x_angle)
    cy, sy = np.cos(y_angle), np.sin(y_angle)
    cz, sz = np.cos(z_angle), np.sin(z_angle)

    rotate_x = np.array(
        [[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32
    )
    rotate_y = np.array(
        [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32
    )
    rotate_z = np.array(
        [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32
    )
    return rotate_z @ rotate_y @ rotate_x


def augment_landmarks(landmarks, landmark_valid_mask):
    """Apply one coherent transform to all valid points in a normalized arch."""
    augmented = landmarks.copy()
    angles = np.radians(
        [
            np.random.uniform(-3.0, 3.0),
            np.random.uniform(-3.0, 3.0),
            np.random.uniform(-8.0, 8.0),
        ]
    )
    rotation = _rotation_matrix_xyz(angles)
    scale_jitter = np.float32(np.random.uniform(0.95, 1.05))

    valid_points = augmented[landmark_valid_mask]
    valid_points = (valid_points @ rotation.T) * scale_jitter
    noise = np.random.normal(0.0, 0.005, size=valid_points.shape).astype(np.float32)
    augmented[landmark_valid_mask] = valid_points + noise
    augmented[~landmark_valid_mask] = 0.0
    return augmented


def sample_consecutive_valid_teeth(
    tooth_valid_mask,
    min_drop=3,
    max_drop=6,
    min_remaining_teeth=4,
    rng=None,
):
    """Sample a consecutive window over currently valid teeth."""
    rng = rng or random
    valid_indices = np.flatnonzero(tooth_valid_mask)
    max_allowed_drop = min(max_drop, len(valid_indices) - min_remaining_teeth)
    if max_allowed_drop < min_drop:
        return np.zeros(SEQ_LEN, dtype=bool)

    drop_count = rng.randint(min_drop, max_allowed_drop)
    start_offset = rng.randint(0, len(valid_indices) - drop_count)
    drop_indices = valid_indices[start_offset : start_offset + drop_count]
    dropped_mask = np.zeros(SEQ_LEN, dtype=bool)
    dropped_mask[drop_indices] = True
    return dropped_mask


def load_real_data(seg_root_base, kpt_root_base):
    """Load upper and lower arches using the shared 16-position sequence."""
    all_arch_data = []
    patient_ids = []

    for jaw_name, jaw_value in (("upper", 0), ("lower", 1)):
        seg_dir = os.path.join(seg_root_base, jaw_name)
        kpt_dir = os.path.join(kpt_root_base, jaw_name)
        if not os.path.isdir(seg_dir) or not os.path.isdir(kpt_dir):
            continue

        for patient_id in sorted(os.listdir(seg_dir)):
            patient_seg_dir = os.path.join(seg_dir, patient_id)
            patient_kpt_dir = os.path.join(kpt_dir, patient_id)
            if not os.path.isdir(patient_seg_dir):
                continue

            obj_path = os.path.join(patient_seg_dir, f"{patient_id}_{jaw_name}.obj")
            label_path = os.path.join(patient_seg_dir, f"{patient_id}_{jaw_name}.json")
            kpt_path = os.path.join(
                patient_kpt_dir, f"{patient_id}_{jaw_name}__kpt.json"
            )
            if not all(os.path.exists(path) for path in (obj_path, label_path, kpt_path)):
                continue

            landmarks, tooth_mask, landmark_mask = extract_arch_landmarks(
                obj_path, label_path, kpt_path, jaw_value
            )
            if tooth_mask.sum() >= 7:
                all_arch_data.append(
                    (landmarks, tooth_mask, landmark_mask, jaw_value)
                )
                patient_ids.append(f"{patient_id}_{jaw_name}")

    return all_arch_data, patient_ids


class DentalArchDataset(Dataset):
    def __init__(self, data_list, is_train=True, min_remaining_teeth=4):
        self.data_list = data_list
        self.is_train = is_train
        self.min_remaining_teeth = min_remaining_teeth

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        raw_landmarks, tooth_valid_mask, landmark_valid_mask, jaw_value = self.data_list[idx]
        landmarks, origin, scale = normalize_landmarks(
            raw_landmarks, landmark_valid_mask
        )
        if self.is_train:
            landmarks = augment_landmarks(landmarks, landmark_valid_mask)

        gt_landmarks = landmarks.copy()
        if self.is_train:
            dropped_mask = sample_consecutive_valid_teeth(
                tooth_valid_mask,
                min_drop=3,
                max_drop=6,
                min_remaining_teeth=self.min_remaining_teeth,
            )
        else:
            dropped_mask = np.zeros(SEQ_LEN, dtype=bool)

        input_landmarks = landmarks.copy()
        input_landmarks[dropped_mask] = 0.0
        input_landmarks[~tooth_valid_mask] = 0.0

        tooth_missing = (~tooth_valid_mask | dropped_mask).astype(np.float32)
        flattened = input_landmarks.reshape(SEQ_LEN, LANDMARK_COORD_DIM)
        jaw_feature = np.full((SEQ_LEN, 1), jaw_value, dtype=np.float32)
        features = np.concatenate(
            [flattened, tooth_missing[:, None], jaw_feature], axis=-1
        )

        return {
            "features": torch.from_numpy(features),
            "gt_landmarks": torch.from_numpy(gt_landmarks),
            "dropped_mask": torch.from_numpy(dropped_mask),
            "tooth_valid_mask": torch.from_numpy(tooth_valid_mask.copy()),
            "landmark_valid_mask": torch.from_numpy(landmark_valid_mask.copy()),
            "normalization_origin": torch.from_numpy(origin),
            "normalization_scale": torch.tensor(scale, dtype=torch.float32),
        }
