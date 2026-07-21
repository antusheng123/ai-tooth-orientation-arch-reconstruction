"""Run Phase 2 arch completion on P05-style clinical landmark JSON files.

The original Phase 2 apps expect separate segmentation labels and landmark
files from the training dataset. P05 provides already grouped clinical
landmarks, so this module adapts those landmarks into the same 16-token model
input without changing the research pipelines.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
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
SEQ_LEN = 16
NUM_LANDMARKS = 2
LANDMARK_COORD_DIM = NUM_LANDMARKS * 3
CLASS_ALIASES = {
    "Inner": "InnerPoint",
    "Outer": "OuterPoint",
    "Cusps": "Cusp",
}
P05_LOWER_FDI_BY_KEY = {
    1: 41,
    2: 45,
    7: 42,
    9: 43,
    10: 44,
    5: 31,
    8: 32,
    3: 33,
    6: 34,
    4: 36,
}
P05_UPPER_FDI_BY_KEY = {
    1: 21,
    2: 13,
    3: 22,
    6: 23,
    7: 24,
    4: 26,
    5: 27,
}
R05_LOWER_FDI_BY_KEY = {
    1: 41,
    5: 31,
    7: 42,
    12: 43,
    2: 45,
    9: 46,
    10: 47,
    13: 48,
    11: 32,
    3: 33,
    6: 34,
    4: 36,
    8: 37,
}
R05_UPPER_FDI_BY_KEY = {
    1: 21,
    5: 26,
    3: 11,
    4: 14,
    2: 16,
}


def parse_fdi_map(value: str | None) -> dict[int, int]:
    """Parse a comma-separated clinical-key to FDI mapping."""
    if not value:
        return {}

    mapping = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected KEY=FDI mapping item, got {item!r}")
        key_text, fdi_text = item.split("=", 1)
        mapping[int(key_text.strip())] = int(fdi_text.strip())
    return mapping


def parse_observed_fdi(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def get_builtin_fdi_map(
    jaw_name: str,
    case_name: str,
    landmarks_path: str | Path,
) -> dict[int, int]:
    """Return verified built-in clinical key mappings when available."""
    path_name = Path(landmarks_path).name.lower()
    if (
        jaw_name == "lower"
        and case_name.upper() == "P05"
        and path_name == "p05_lower_landmarks.json"
    ):
        return P05_LOWER_FDI_BY_KEY.copy()
    if (
        jaw_name == "upper"
        and case_name.upper() == "P05"
        and path_name == "p05_upper_landmarks.json"
    ):
        return P05_UPPER_FDI_BY_KEY.copy()
    if (
        jaw_name == "lower"
        and case_name.upper() == "R05"
        and path_name == "r05_lower_landmarks.json"
    ):
        return R05_LOWER_FDI_BY_KEY.copy()
    if (
        jaw_name == "upper"
        and case_name.upper() == "R05"
        and path_name == "r05_upper_landmarks.json"
    ):
        return R05_UPPER_FDI_BY_KEY.copy()
    return {}


def load_clinical_landmark_groups(path: str | Path) -> dict[int, dict[str, list[np.ndarray]]]:
    """Load P05-style landmarks grouped by clinical tooth key."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    groups: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for obj in data.get("objects", []):
        key = str(obj.get("key", "")).split("_", 1)[0]
        class_name = obj.get("class")
        coord = obj.get("coord")
        if not key or class_name is None or coord is None:
            continue
        coord_array = np.asarray(coord, dtype=np.float32)
        if coord_array.shape != (3,) or not np.isfinite(coord_array).all():
            continue
        groups[int(key)][CLASS_ALIASES.get(class_name, class_name)].append(coord_array)

    return {key: dict(classes) for key, classes in groups.items()}


def infer_fdi_map_from_key_order(
    groups: dict[int, dict[str, list[np.ndarray]]],
    fdi_order: list[int],
    observed_fdi: list[int] | None = None,
) -> dict[int, int]:
    """Map sorted clinical keys to an explicit FDI list or the start of jaw order."""
    sorted_keys = sorted(groups)
    if observed_fdi:
        if len(observed_fdi) != len(sorted_keys):
            raise ValueError(
                f"--observed-fdi contains {len(observed_fdi)} entries, "
                f"but landmark JSON contains {len(sorted_keys)} tooth keys"
            )
        unknown = sorted(set(observed_fdi) - set(fdi_order))
        if unknown:
            raise ValueError(f"Observed FDI values are not in this jaw order: {unknown}")
        return dict(zip(sorted_keys, observed_fdi))

    return dict(zip(sorted_keys, fdi_order[: len(sorted_keys)]))


def _mean_point(class_points: dict[str, list[np.ndarray]], class_name: str) -> np.ndarray | None:
    points = class_points.get(class_name)
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=np.float32), axis=0)


def _select_two_cusps(class_points: dict[str, list[np.ndarray]]) -> np.ndarray | None:
    cusps = class_points.get("Cusp")
    if not cusps:
        return None
    cusps_array = np.asarray(cusps, dtype=np.float32)
    if len(cusps_array) >= 2:
        outer = _mean_point(class_points, "OuterPoint")
        if outer is not None:
            distances = np.linalg.norm(cusps_array - outer, axis=1)
            return cusps_array[np.argsort(distances, kind="stable")[:2]]
        return cusps_array[:2]

    cusp = cusps_array[0]
    mesial = _mean_point(class_points, "Mesial")
    distal = _mean_point(class_points, "Distal")
    if mesial is None or distal is None:
        return None
    offset = (distal - mesial) * 0.25
    return np.stack([cusp - offset, cusp + offset], axis=0).astype(np.float32)


def build_landmark_sequence(
    groups: dict[int, dict[str, list[np.ndarray]]],
    fdi_by_key: dict[int, int],
    fdi_order: list[int],
    pipeline: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert clinical landmark groups into the 16-position Phase 2 tensor."""
    fdi_to_idx = {fdi: idx for idx, fdi in enumerate(fdi_order)}
    landmarks = np.zeros((SEQ_LEN, NUM_LANDMARKS, 3), dtype=np.float32)
    landmark_mask = np.zeros((SEQ_LEN, NUM_LANDMARKS), dtype=bool)

    for key, class_points in groups.items():
        fdi = fdi_by_key.get(key)
        if fdi not in fdi_to_idx:
            continue
        idx = fdi_to_idx[fdi]

        mesial = _mean_point(class_points, "Mesial")
        distal = _mean_point(class_points, "Distal")
        if mesial is None or distal is None:
            continue

        if pipeline == "pipeline2":
            if fdi in ANTERIOR_FDI:
                inner = _mean_point(class_points, "InnerPoint")
                if inner is None:
                    continue
                landmarks[idx, 0] = (inner + mesial) / 2.0
                landmarks[idx, 1] = (inner + distal) / 2.0
            else:
                landmarks[idx, 0] = mesial
                landmarks[idx, 1] = distal
        elif pipeline == "pipeline1":
            if fdi in ANTERIOR_FDI:
                landmarks[idx, 0] = mesial
                landmarks[idx, 1] = distal
            else:
                cusps = _select_two_cusps(class_points)
                if cusps is None:
                    continue
                landmarks[idx] = cusps
        else:
            raise ValueError(f"Unsupported pipeline: {pipeline}")

        landmark_mask[idx] = True

    tooth_mask = landmark_mask.all(axis=1)
    landmarks[~tooth_mask] = 0.0
    landmark_mask[~tooth_mask] = False
    return landmarks, tooth_mask, landmark_mask


def normalize_landmarks(
    landmarks: np.ndarray,
    landmark_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    valid_points = landmarks[landmark_mask]
    if len(valid_points) == 0:
        raise ValueError("No valid landmarks were available for normalization")

    origin = valid_points.mean(axis=0).astype(np.float32)
    centered = valid_points - origin
    scale = np.float32(np.sqrt(np.mean(np.sum(centered * centered, axis=-1))))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError(f"Invalid normalization scale: {scale}")

    normalized = np.zeros_like(landmarks, dtype=np.float32)
    normalized[landmark_mask] = (landmarks[landmark_mask] - origin) / scale
    return normalized, origin, scale


def denormalize_landmarks(
    landmarks: np.ndarray,
    origin: np.ndarray,
    scale: np.float32,
) -> np.ndarray:
    return landmarks.astype(np.float32) * scale + origin


def build_features(
    normalized_landmarks: np.ndarray,
    tooth_mask: np.ndarray,
    jaw_value: int,
) -> np.ndarray:
    tooth_missing = (~tooth_mask).astype(np.float32)
    jaw_feature = np.full((SEQ_LEN, 1), jaw_value, dtype=np.float32)
    return np.concatenate(
        [
            normalized_landmarks.reshape(SEQ_LEN, LANDMARK_COORD_DIM),
            tooth_missing[:, None],
            jaw_feature,
        ],
        axis=-1,
    )


def load_pipeline_model(pipeline_dir: Path, weights_path: Path, device_name: str):
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required to run model inference") from error

    sys.path.insert(0, str(pipeline_dir))
    try:
        if "model" in sys.modules:
            del sys.modules["model"]
        model_module = importlib.import_module("model")
        model = model_module.MaskedArchRegressor()
    finally:
        sys.path.pop(0)

    device = torch.device(device_name)
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, torch, device


def run_inference(
    landmarks: np.ndarray,
    tooth_mask: np.ndarray,
    landmark_mask: np.ndarray,
    jaw_value: int,
    pipeline_dir: Path,
    weights_path: Path,
    device_name: str,
) -> np.ndarray:
    normalized, origin, scale = normalize_landmarks(landmarks, landmark_mask)
    normalized[~tooth_mask] = 0.0
    features = build_features(normalized, tooth_mask, jaw_value)
    model, torch, device = load_pipeline_model(pipeline_dir, weights_path, device_name)

    with torch.no_grad():
        feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        predicted_normalized = model(feature_tensor)[0].cpu().numpy()

    return denormalize_landmarks(predicted_normalized, origin, scale)


def collect_curve_anchor_points(
    landmarks: np.ndarray,
    tooth_mask: np.ndarray,
    fdi_order: list[int],
    pipeline: str,
) -> np.ndarray:
    anchors = []
    previous = None
    for idx, is_valid in enumerate(tooth_mask):
        if not is_valid:
            continue
        fdi = fdi_order[idx]
        points = landmarks[idx]
        if pipeline == "pipeline2" or fdi in ANTERIOR_FDI:
            order = (1, 0) if fdi // 10 in (1, 4) else (0, 1)
        elif previous is not None:
            order = min(
                ((0, 1), (1, 0)),
                key=lambda candidate: np.linalg.norm(points[candidate[0]] - previous),
            )
        else:
            order = (0, 1)
        ordered = points[list(order)]
        anchors.extend(ordered)
        previous = ordered[-1]
    return np.asarray(anchors, dtype=np.float32)


def generate_curve(anchor_points: np.ndarray, pipeline_dir: Path) -> np.ndarray:
    sys.path.insert(0, str(pipeline_dir))
    try:
        if "postprocess" in sys.modules:
            del sys.modules["postprocess"]
        postprocess = importlib.import_module("postprocess")
        return postprocess.generate_bspline_curve(anchor_points, num_eval_points=500)
    finally:
        sys.path.pop(0)


def build_visualization_figure(
    fdi_order: list[int],
    observed_mask: np.ndarray,
    final_landmarks: np.ndarray,
    curve_points: np.ndarray,
    title: str,
):
    import plotly.graph_objects as go

    figure = go.Figure()
    if len(curve_points):
        figure.add_trace(
            go.Scatter3d(
                x=curve_points[:, 0],
                y=curve_points[:, 1],
                z=curve_points[:, 2],
                mode="lines",
                line=dict(color="#0F766E", width=8),
                name="Predicted arch curve",
                hovertemplate="Predicted arch curve<extra></extra>",
            )
        )

    for state, mask, color, symbol in (
        ("Observed landmarks", observed_mask, "#2563EB", "circle"),
        ("Predicted landmarks", ~observed_mask, "#DC2626", "diamond"),
    ):
        points = final_landmarks[mask].reshape(-1, 3)
        if not len(points):
            continue
        labels = [
            f"FDI {fdi} P{point_idx}"
            for fdi, is_selected in zip(fdi_order, mask)
            if is_selected
            for point_idx in range(NUM_LANDMARKS)
        ]
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers+text",
                marker=dict(size=5, color=color, symbol=symbol),
                text=labels,
                textposition="top center",
                name=state,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    figure.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=48),
        height=820,
        legend=dict(orientation="h", y=0.02, x=0.0),
    )
    return figure


def write_outputs(
    output_dir: Path,
    case_name: str,
    jaw_name: str,
    pipeline: str,
    fdi_order: list[int],
    observed_mask: np.ndarray,
    final_landmarks: np.ndarray,
    predicted_landmarks: np.ndarray,
    curve_points: np.ndarray,
    fdi_by_key: dict[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{case_name}_{jaw_name}_{pipeline}"

    rows = []
    for idx, fdi in enumerate(fdi_order):
        state = "observed" if observed_mask[idx] else "predicted"
        source = final_landmarks[idx]
        if not observed_mask[idx] and not np.isfinite(predicted_landmarks[idx]).all():
            continue
        for point_idx, coord in enumerate(source):
            rows.append(
                {
                    "fdi": fdi,
                    "point": point_idx,
                    "state": state,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": float(coord[2]),
                }
            )

    csv_path = output_dir / f"{prefix}_landmarks.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["fdi", "point", "state", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "case": case_name,
        "jaw": jaw_name,
        "pipeline": pipeline,
        "fdi_by_clinical_key": {str(key): fdi for key, fdi in sorted(fdi_by_key.items())},
        "landmarks": rows,
        "curve_points": curve_points.astype(float).tolist(),
    }
    json_path = output_dir / f"{prefix}_result.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure = build_visualization_figure(
        fdi_order,
        observed_mask,
        final_landmarks,
        curve_points,
        title=f"{case_name} {jaw_name} {pipeline}",
    )
    html_path = output_dir / f"{prefix}_visualization.html"
    figure.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)


def default_pipeline_dir(repo_root: Path, pipeline: str) -> Path:
    folder = "pipeline1_mandibular" if pipeline == "pipeline1" else "pipeline2_maxillary"
    return repo_root / "phase2_arch_curve_reconstruction" / folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt P05-style clinical landmarks to Phase 2 arch completion. "
            "For clinically meaningful output, pass --fdi-map or --observed-fdi."
        )
    )
    parser.add_argument("--jaw", choices=["upper", "lower"], required=True)
    parser.add_argument("--pipeline", choices=["pipeline1", "pipeline2"], required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--case-name", default="P05")
    parser.add_argument("--fdi-map", help="Comma list such as 1=48,2=47,3=46")
    parser.add_argument(
        "--observed-fdi",
        help="Comma list mapped to sorted clinical keys, such as 48,47,46",
    )
    parser.add_argument("--pipeline-dir", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "P05")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    fdi_order = UPPER_FDI if args.jaw == "upper" else LOWER_FDI
    jaw_value = 0 if args.jaw == "upper" else 1
    pipeline_dir = args.pipeline_dir or default_pipeline_dir(repo_root, args.pipeline)
    weights_path = args.weights or pipeline_dir / "weights" / "best_landmark_model.pth"

    groups = load_clinical_landmark_groups(args.landmarks)
    fdi_by_key = parse_fdi_map(args.fdi_map)
    if not fdi_by_key:
        fdi_by_key = get_builtin_fdi_map(args.jaw, args.case_name, args.landmarks)
        if fdi_by_key:
            print(
                f"Using built-in verified {args.case_name} {args.jaw} key-to-FDI mapping.",
                file=sys.stderr,
            )
        else:
            fdi_by_key = infer_fdi_map_from_key_order(
                groups,
                fdi_order,
                parse_observed_fdi(args.observed_fdi),
            )
            print(
                "Warning: using sorted clinical key order for FDI mapping. "
                "Pass --fdi-map for verified clinical tooth identities.",
                file=sys.stderr,
            )

    landmarks, tooth_mask, landmark_mask = build_landmark_sequence(
        groups,
        fdi_by_key,
        fdi_order,
        args.pipeline,
    )
    valid_count = int(tooth_mask.sum())
    if valid_count < 4:
        raise RuntimeError(f"Only {valid_count} valid teeth were built; at least 4 are needed")

    predicted = run_inference(
        landmarks,
        tooth_mask,
        landmark_mask,
        jaw_value,
        pipeline_dir,
        weights_path,
        args.device,
    )
    final_landmarks = landmarks.copy()
    final_landmarks[~tooth_mask] = predicted[~tooth_mask]
    final_mask = np.ones(SEQ_LEN, dtype=bool)
    anchors = collect_curve_anchor_points(final_landmarks, final_mask, fdi_order, args.pipeline)
    curve = generate_curve(anchors, pipeline_dir)

    write_outputs(
        args.output_dir,
        args.case_name,
        args.jaw,
        args.pipeline,
        fdi_order,
        tooth_mask,
        final_landmarks,
        predicted,
        curve,
        fdi_by_key,
    )

    print(f"Valid observed teeth: {valid_count}/16")
    print(f"Predicted missing teeth: {SEQ_LEN - valid_count}/16")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
