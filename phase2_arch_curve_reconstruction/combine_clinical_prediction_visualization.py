"""Combine clinical landmark keys and model prediction output in one HTML view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .clinical_p05_inference import LOWER_FDI, UPPER_FDI, load_clinical_landmark_groups
    from .visualize_clinical_landmarks import (
        build_combined_prediction_figure,
        load_obj_mesh,
    )
except ImportError:
    from clinical_p05_inference import LOWER_FDI, UPPER_FDI, load_clinical_landmark_groups
    from visualize_clinical_landmarks import build_combined_prediction_figure, load_obj_mesh


def load_prediction_result(path: str | Path, fdi_order: list[int]):
    with open(path, "r", encoding="utf-8") as file:
        result = json.load(file)

    fdi_to_idx = {fdi: idx for idx, fdi in enumerate(fdi_order)}
    final_landmarks = np.zeros((len(fdi_order), 2, 3), dtype=np.float32)
    observed_mask = np.zeros(len(fdi_order), dtype=bool)

    for row in result.get("landmarks", []):
        fdi = int(row["fdi"])
        point_idx = int(row["point"])
        if fdi not in fdi_to_idx or point_idx not in (0, 1):
            continue
        idx = fdi_to_idx[fdi]
        final_landmarks[idx, point_idx] = [
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
        ]
        if row.get("state") == "observed":
            observed_mask[idx] = True

    curve_points = np.asarray(result.get("curve_points", []), dtype=np.float32)
    if curve_points.size == 0:
        curve_points = np.empty((0, 3), dtype=np.float32)
    return final_landmarks, observed_mask, curve_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay clinical key landmarks, OBJ mesh, model points, and arch curve."
    )
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--jaw", choices=["upper", "lower"], required=True)
    parser.add_argument("--title", default="Combined clinical and model visualization")
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fdi_order = UPPER_FDI if args.jaw == "upper" else LOWER_FDI
    vertices, faces = load_obj_mesh(args.obj)
    groups = load_clinical_landmark_groups(args.landmarks)
    final_landmarks, observed_mask, curve_points = load_prediction_result(
        args.result_json,
        fdi_order,
    )
    figure = build_combined_prediction_figure(
        vertices=vertices,
        faces=faces,
        groups=groups,
        fdi_order=fdi_order,
        observed_mask=observed_mask,
        final_landmarks=final_landmarks,
        curve_points=curve_points,
        title=args.title,
    )

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(args.output_html), include_plotlyjs="cdn", full_html=True)
    print(f"HTML: {args.output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
