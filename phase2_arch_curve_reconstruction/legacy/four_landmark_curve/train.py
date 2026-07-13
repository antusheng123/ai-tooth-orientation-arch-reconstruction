import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    LANDMARK_COORD_DIM,
    DentalArchDataset,
    load_real_data,
)
from model import MaskedArchRegressor


SEG_ROOT_BASE = r"F:\NDCS_3DS_data\segmentation_data_for_single_teeth\train"
KPT_ROOT_BASE = r"F:\NDCS_3DS_data\3DTeethLand_landmarks_train"
MODEL_PATH = "best_landmark_model.pth"

POINT_LOSS_WEIGHT = 1.0
CENTER_LOSS_WEIGHT = 0.25
GEOMETRY_LOSS_WEIGHT = 0.10


def split_by_patient(data, patient_ids, train_fraction=0.8, seed=42):
    """Keep upper/lower arches from the same patient in the same split."""
    grouped = {}
    for sample, sample_id in zip(data, patient_ids):
        patient_id = sample_id.rsplit("_", 1)[0]
        grouped.setdefault(patient_id, []).append(sample)

    patient_keys = list(grouped)
    random.Random(seed).shuffle(patient_keys)
    split_idx = int(len(patient_keys) * train_fraction)
    train_keys = set(patient_keys[:split_idx])

    train_data = []
    test_data = []
    for patient_id, samples in grouped.items():
        target = train_data if patient_id in train_keys else test_data
        target.extend(samples)
    return train_data, test_data


def masked_landmark_loss(predictions, targets, tooth_mask):
    """Combine point, derived-center, and intra-tooth geometry losses."""
    pred_teeth = predictions[tooth_mask]
    target_teeth = targets[tooth_mask]
    if pred_teeth.numel() == 0:
        return None, {}

    point_loss = F.smooth_l1_loss(pred_teeth, target_teeth)
    center_loss = F.smooth_l1_loss(
        pred_teeth.mean(dim=1), target_teeth.mean(dim=1)
    )

    pair_indices = torch.triu_indices(4, 4, offset=1, device=predictions.device)
    pred_distances = torch.linalg.vector_norm(
        pred_teeth[:, pair_indices[0]] - pred_teeth[:, pair_indices[1]], dim=-1
    )
    target_distances = torch.linalg.vector_norm(
        target_teeth[:, pair_indices[0]] - target_teeth[:, pair_indices[1]], dim=-1
    )
    geometry_loss = F.smooth_l1_loss(pred_distances, target_distances)

    total = (
        POINT_LOSS_WEIGHT * point_loss
        + CENTER_LOSS_WEIGHT * center_loss
        + GEOMETRY_LOSS_WEIGHT * geometry_loss
    )
    parts = {
        "point": point_loss.detach(),
        "center": center_loss.detach(),
        "geometry": geometry_loss.detach(),
    }
    return total, parts


def make_masked_validation_batch(batch, device, batch_index, seed=1042):
    targets = batch["gt_landmarks"].to(device)
    tooth_valid_mask = batch["tooth_valid_mask"]
    dropped_mask = torch.zeros_like(tooth_valid_mask)
    rng = random.Random(seed + batch_index)

    for row in range(tooth_valid_mask.shape[0]):
        valid_indices = torch.where(tooth_valid_mask[row])[0]
        if len(valid_indices) > 4:
            max_drop = min(5, len(valid_indices) - 4)
            drop_count = rng.randint(1, max_drop)
            selected = rng.sample(valid_indices.tolist(), drop_count)
            dropped_mask[row, selected] = True

    input_landmarks = targets.clone()
    input_landmarks[dropped_mask.to(device)] = 0.0
    input_landmarks[~tooth_valid_mask.to(device)] = 0.0
    tooth_missing = (~tooth_valid_mask | dropped_mask).float().to(device)
    jaw_type = batch["features"][:, :, -1:].to(device)
    flattened = input_landmarks.reshape(
        input_landmarks.shape[0], input_landmarks.shape[1], LANDMARK_COORD_DIM
    )
    features = torch.cat([flattened, tooth_missing.unsqueeze(-1), jaw_type], dim=-1)
    return features, targets, dropped_mask.to(device)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading four-landmark arch data...")
    all_data, patient_ids = load_real_data(SEG_ROOT_BASE, KPT_ROOT_BASE)
    if not all_data:
        print("No valid arches found. Check dataset paths and landmark annotations.")
        return

    train_data, test_data = split_by_patient(all_data, patient_ids)
    print(f"Loaded {len(all_data)} arches: {len(train_data)} train, {len(test_data)} test")
    if not train_data or not test_data:
        print("The patient-level split produced an empty subset.")
        return

    train_loader = DataLoader(
        DentalArchDataset(train_data, is_train=True),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        DentalArchDataset(test_data, is_train=False),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = MaskedArchRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=12
    )

    epochs = 1500
    best_test_loss = float("inf")
    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["gt_landmarks"].to(device)
            dropped_mask = batch["dropped_mask"].to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss, _ = masked_landmark_loss(predictions, targets, dropped_mask)
            if loss is None:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            train_steps += 1

        if (epoch + 1) % 5 != 0:
            continue

        model.eval()
        test_loss_sum = 0.0
        test_steps = 0
        point_error_mm_sum = 0.0

        with torch.no_grad():
            for batch_index, batch in enumerate(test_loader):
                features, targets, dropped_mask = make_masked_validation_batch(
                    batch, device, batch_index
                )
                predictions = model(features)
                loss, _ = masked_landmark_loss(predictions, targets, dropped_mask)
                if loss is None:
                    continue

                selected_errors = torch.linalg.vector_norm(
                    predictions[dropped_mask] - targets[dropped_mask], dim=-1
                )
                selected_scales = batch["normalization_scale"].to(device)
                selected_scales = selected_scales[:, None].expand_as(
                    batch["tooth_valid_mask"]
                ).to(device)[dropped_mask]
                point_error_mm = (
                    selected_errors * selected_scales[:, None]
                ).mean()

                test_loss_sum += loss.item()
                point_error_mm_sum += point_error_mm.item()
                test_steps += 1

        average_train = train_loss_sum / max(train_steps, 1)
        average_test = test_loss_sum / max(test_steps, 1)
        average_point_mm = point_error_mm_sum / max(test_steps, 1)
        scheduler.step(average_test)

        status = ""
        if test_steps > 0 and average_test < best_test_loss:
            best_test_loss = average_test
            torch.save(model.state_dict(), MODEL_PATH)
            status = "<-- saved new best"

        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:04d} | LR {learning_rate:.1e} | "
            f"Train {average_train:.5f} | Test {average_test:.5f} | "
            f"Landmark error {average_point_mm:.3f} mm {status}"
        )


if __name__ == "__main__":
    main()
