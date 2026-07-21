"""Visualize P05-style clinical landmarks on their OBJ arch mesh."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .clinical_p05_inference import load_clinical_landmark_groups
except ImportError:
    from clinical_p05_inference import load_clinical_landmark_groups


KEY_COLORS = [
    "#2563EB",
    "#DC2626",
    "#16A34A",
    "#9333EA",
    "#EA580C",
    "#0891B2",
    "#BE123C",
    "#4D7C0F",
    "#7C3AED",
    "#B45309",
    "#0F766E",
    "#C026D3",
    "#1D4ED8",
    "#B91C1C",
    "#15803D",
    "#6D28D9",
]


def parse_obj_mesh_lines(lines) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            indices = []
            for token in line.split()[1:]:
                vertex_token = token.split("/", 1)[0]
                if vertex_token:
                    indices.append(int(vertex_token) - 1)
            if len(indices) >= 3:
                base = indices[0]
                for offset in range(1, len(indices) - 1):
                    faces.append([base, indices[offset], indices[offset + 1]])

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )


def load_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as file:
        return parse_obj_mesh_lines(file)


def flatten_group_points(class_points: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, list[str]]:
    points = []
    labels = []
    for class_name in sorted(class_points):
        for point_index, point in enumerate(class_points[class_name], start=1):
            points.append(np.asarray(point, dtype=np.float32))
            labels.append(f"{class_name} {point_index}")
    if not points:
        return np.empty((0, 3), dtype=np.float32), []
    return np.asarray(points, dtype=np.float32), labels


def build_key_summary(groups: dict[int, dict[str, list[np.ndarray]]]) -> list[dict[str, object]]:
    rows = []
    for key in sorted(groups):
        class_points = groups[key]
        points, _ = flatten_group_points(class_points)
        class_counts = Counter(
            {
                class_name: len(points_for_class)
                for class_name, points_for_class in class_points.items()
            }
        )
        centroid = points.mean(axis=0) if len(points) else np.zeros(3, dtype=np.float32)
        rows.append(
            {
                "key": key,
                "landmark_count": int(len(points)),
                "classes": ", ".join(
                    f"{class_name}:{class_counts[class_name]}"
                    for class_name in sorted(class_counts)
                ),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
            }
        )
    return rows


def build_landmark_mesh_figure(
    vertices: np.ndarray,
    faces: np.ndarray,
    groups: dict[int, dict[str, list[np.ndarray]]],
    title: str,
):
    import plotly.graph_objects as go

    figure = go.Figure()
    if len(vertices) and len(faces):
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#D1D5DB",
                opacity=0.22,
                name="OBJ mesh",
                hoverinfo="skip",
            )
        )

    centroids = []
    centroid_labels = []
    for color_index, key in enumerate(sorted(groups)):
        class_points = groups[key]
        points, point_labels = flatten_group_points(class_points)
        if not len(points):
            continue
        color = KEY_COLORS[color_index % len(KEY_COLORS)]
        labels = [f"key {key}<br>{label}" for label in point_labels]
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=6, color=color),
                text=labels,
                name=f"key {key} landmarks",
                hovertemplate="%{text}<br>x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>",
            )
        )
        centroids.append(points.mean(axis=0))
        centroid_labels.append(f"key {key}")

    if centroids:
        centroid_points = np.asarray(centroids, dtype=np.float32)
        figure.add_trace(
            go.Scatter3d(
                x=centroid_points[:, 0],
                y=centroid_points[:, 1],
                z=centroid_points[:, 2],
                mode="markers+text",
                marker=dict(size=9, color="#111827", symbol="diamond"),
                text=centroid_labels,
                textposition="top center",
                name="key labels",
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
        height=860,
        legend=dict(itemsizing="constant"),
    )
    return figure


def add_mesh_and_key_traces(figure, vertices, faces, groups) -> None:
    import plotly.graph_objects as go

    if len(vertices) and len(faces):
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#D1D5DB",
                opacity=0.18,
                name="OBJ mesh",
                hoverinfo="skip",
            )
        )

    centroids = []
    centroid_labels = []
    for color_index, key in enumerate(sorted(groups)):
        class_points = groups[key]
        points, point_labels = flatten_group_points(class_points)
        if not len(points):
            continue
        color = KEY_COLORS[color_index % len(KEY_COLORS)]
        labels = [f"key {key}<br>{label}" for label in point_labels]
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.62),
                text=labels,
                name=f"key {key} landmarks",
                hovertemplate="%{text}<br>x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>",
            )
        )
        centroids.append(points.mean(axis=0))
        centroid_labels.append(f"key {key}")

    if centroids:
        centroid_points = np.asarray(centroids, dtype=np.float32)
        figure.add_trace(
            go.Scatter3d(
                x=centroid_points[:, 0],
                y=centroid_points[:, 1],
                z=centroid_points[:, 2],
                mode="markers+text",
                marker=dict(size=7, color="#111827", symbol="diamond"),
                text=centroid_labels,
                textposition="top center",
                name="key labels",
                hovertemplate="%{text}<extra></extra>",
            )
        )


def add_prediction_traces(
    figure,
    fdi_order: list[int],
    observed_mask: np.ndarray,
    final_landmarks: np.ndarray,
    curve_points: np.ndarray,
) -> None:
    import plotly.graph_objects as go

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
        ("Observed model landmarks", observed_mask, "#2563EB", "circle"),
        ("Predicted model landmarks", ~observed_mask, "#DC2626", "diamond"),
    ):
        points = final_landmarks[mask].reshape(-1, 3)
        if not len(points):
            continue
        labels = [
            f"FDI {fdi} P{point_idx}"
            for fdi, is_selected in zip(fdi_order, mask)
            if is_selected
            for point_idx in range(2)
        ]
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers+text",
                marker=dict(size=6, color=color, symbol=symbol),
                text=labels,
                textposition="top center",
                name=state,
                hovertemplate="%{text}<br>x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>",
            )
        )


def build_combined_prediction_figure(
    vertices: np.ndarray,
    faces: np.ndarray,
    groups: dict[int, dict[str, list[np.ndarray]]],
    fdi_order: list[int],
    observed_mask: np.ndarray,
    final_landmarks: np.ndarray,
    curve_points: np.ndarray,
    title: str,
):
    import plotly.graph_objects as go

    figure = go.Figure()
    add_mesh_and_key_traces(figure, vertices, faces, groups)
    add_prediction_traces(figure, fdi_order, observed_mask, final_landmarks, curve_points)
    figure.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=48),
        height=900,
        legend=dict(itemsizing="constant"),
    )
    return figure


def write_summary_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "key",
        "landmark_count",
        "classes",
        "centroid_x",
        "centroid_y",
        "centroid_z",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an HTML view of P05 clinical landmark keys on an OBJ mesh."
    )
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--title", default="P05 lower clinical landmark keys")
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("outputs") / "P05" / "P05_lower_key_landmarks.html",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs") / "P05" / "P05_lower_key_summary.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vertices, faces = load_obj_mesh(args.obj)
    groups = load_clinical_landmark_groups(args.landmarks)
    figure = build_landmark_mesh_figure(vertices, faces, groups, args.title)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(args.output_html), include_plotlyjs="cdn", full_html=True)
    write_summary_csv(args.output_csv, build_key_summary(groups))

    print(f"OBJ vertices: {len(vertices)}")
    print(f"OBJ faces: {len(faces)}")
    print(f"Clinical tooth keys: {', '.join(str(key) for key in sorted(groups))}")
    print(f"HTML: {args.output_html}")
    print(f"CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
