import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh
from scipy.spatial import KDTree


BUCCAL_CUSP_MIN_SCORE = 0.0
SYNTHETIC_CUSP_MIN_OFFSET_RATIO = 0.25


def load_tooth_landmarks(obj_path, label_path, kpt_path, fdi):
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

    tooth_vertices = np.asarray(mesh.vertices[labels == fdi], dtype=np.float32)
    if len(tooth_vertices) == 0:
        raise ValueError(f"No mesh vertices found for FDI {fdi}")

    kdtree = KDTree(mesh.vertices)
    grouped = {"Mesial": [], "Distal": [], "OuterPoint": [], "Cusp": []}
    for landmark in landmark_objects:
        class_name = landmark.get("class")
        coord = landmark.get("coord")
        if class_name not in grouped or coord is None:
            continue

        coord = np.asarray(coord, dtype=np.float32)
        if coord.shape != (3,) or not np.isfinite(coord).all():
            continue

        _, vertex_idx = kdtree.query(coord)
        if int(labels[vertex_idx]) == fdi:
            grouped[class_name].append(coord)

    if not grouped["OuterPoint"]:
        raise ValueError(f"No OuterPoint assigned to FDI {fdi}")
    if not grouped["Cusp"]:
        raise ValueError(f"No Cusp points assigned to FDI {fdi}")
    if not grouped["Mesial"]:
        raise ValueError(f"No Mesial point assigned to FDI {fdi}")
    if not grouped["Distal"]:
        raise ValueError(f"No Distal point assigned to FDI {fdi}")

    return (
        tooth_vertices,
        np.asarray(grouped["Mesial"]),
        np.asarray(grouped["Distal"]),
        np.asarray(grouped["OuterPoint"]),
        np.asarray(grouped["Cusp"]),
    )


def normalize_vector(vector):
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    return vector / norm


def estimate_lateral_axis(mesial_point, distal_point):
    return normalize_vector(distal_point - mesial_point)


def analyze_geometry(tooth_vertices, mesial_points, distal_points, outer_points, cusp_points):
    tooth_center = tooth_vertices.mean(axis=0)
    mesial_point = mesial_points.mean(axis=0)
    distal_point = distal_points.mean(axis=0)
    outer_point = outer_points.mean(axis=0)
    buccal_axis = normalize_vector(outer_point - tooth_center)
    if buccal_axis is None:
        raise ValueError("Cannot estimate buccal axis from tooth center to OuterPoint")

    lateral_axis = estimate_lateral_axis(mesial_point, distal_point)
    if lateral_axis is None:
        raise ValueError("Cannot estimate lateral axis from Mesial-Distal points")

    cusp_scores = (cusp_points - tooth_center) @ buccal_axis
    best_idx = int(np.argmax(cusp_scores))
    most_buccal_cusp = cusp_points[best_idx]
    if cusp_scores[best_idx] <= BUCCAL_CUSP_MIN_SCORE:
        mirrored_cusp = None
    else:
        cusp_vector = most_buccal_cusp - tooth_center
        vertex_offsets = (tooth_vertices - tooth_center) @ lateral_axis
        typical_offset = float(np.percentile(np.abs(vertex_offsets), 75))
        lateral_offset = float(np.dot(cusp_vector, lateral_axis))
        min_offset = SYNTHETIC_CUSP_MIN_OFFSET_RATIO * max(typical_offset, 1e-6)
        adjusted_lateral_offset = lateral_offset
        if abs(adjusted_lateral_offset) < min_offset:
            adjusted_lateral_offset = (
                min_offset if adjusted_lateral_offset >= 0.0 else -min_offset
            )
        residual = cusp_vector - lateral_offset * lateral_axis
        mirrored_cusp = (
            tooth_center + residual - adjusted_lateral_offset * lateral_axis
        ).astype(np.float32)

    return {
        "tooth_center": tooth_center,
        "mesial_point": mesial_point,
        "distal_point": distal_point,
        "outer_point": outer_point,
        "buccal_axis": buccal_axis,
        "lateral_axis": lateral_axis,
        "cusp_scores": cusp_scores,
        "most_buccal_cusp": most_buccal_cusp,
        "most_buccal_index": best_idx,
        "mirrored_cusp": mirrored_cusp,
    }


def sampled_vertices(vertices, max_points=4000):
    if len(vertices) <= max_points:
        return vertices
    indices = np.linspace(0, len(vertices) - 1, max_points).astype(int)
    return vertices[indices]


def add_point(fig, point, name, color, size=8, symbol="circle"):
    fig.add_trace(
        go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode="markers+text",
            marker=dict(size=size, color=color, symbol=symbol),
            text=[name],
            textposition="top center",
            name=name,
        )
    )


def add_vector(fig, start, vector, name, color, scale):
    end = start + vector * scale
    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines+text",
            line=dict(color=color, width=8),
            text=["", name],
            textposition="top center",
            name=name,
        )
    )
    fig.add_trace(
        go.Cone(
            x=[end[0]],
            y=[end[1]],
            z=[end[2]],
            u=[vector[0]],
            v=[vector[1]],
            w=[vector[2]],
            sizemode="absolute",
            sizeref=scale * 0.12,
            anchor="tip",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f"{name} arrow",
        )
    )


def add_mirror_plane(fig, center, lateral_axis, buccal_axis, vertices):
    height_axis = np.cross(lateral_axis, buccal_axis)
    height_axis = normalize_vector(height_axis)
    if height_axis is None:
        return

    centered = vertices - center
    buccal_extent = np.percentile(np.abs(centered @ buccal_axis), 90)
    height_extent = np.percentile(np.abs(centered @ height_axis), 90)
    buccal_extent = max(float(buccal_extent), 1.0)
    height_extent = max(float(height_extent), 1.0)

    u = np.linspace(-buccal_extent, buccal_extent, 2)
    v = np.linspace(-height_extent, height_extent, 2)
    uu, vv = np.meshgrid(u, v)
    plane = center + uu[..., None] * buccal_axis + vv[..., None] * height_axis

    fig.add_trace(
        go.Surface(
            x=plane[:, :, 0],
            y=plane[:, :, 1],
            z=plane[:, :, 2],
            opacity=0.28,
            colorscale=[[0, "#FACC15"], [1, "#FACC15"]],
            showscale=False,
            name="mirror plane",
            hovertemplate="Mirror plane<br>through tooth center<br>normal = Mesial-Distal axis<extra></extra>",
        )
    )


def build_figure(tooth_vertices, mesial_points, distal_points, outer_points, cusp_points, analysis, fdi):
    fig = go.Figure()
    vertices = sampled_vertices(tooth_vertices)
    center = analysis["tooth_center"]
    buccal_axis = analysis["buccal_axis"]
    lateral_axis = analysis["lateral_axis"]
    axis_scale = max(float(np.linalg.norm(tooth_vertices.ptp(axis=0))) * 0.45, 1.0)

    fig.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker=dict(size=2, color="#94A3B8", opacity=0.45),
            name=f"FDI {fdi} tooth mesh vertices",
        )
    )

    add_mirror_plane(fig, center, lateral_axis, buccal_axis, tooth_vertices)
    add_point(fig, center, "tooth center", "#111827", size=9)
    add_point(fig, analysis["mesial_point"], "mean Mesial", "#2563EB", size=8)
    add_point(fig, analysis["distal_point"], "mean Distal", "#7C2D12", size=8)
    add_point(fig, analysis["outer_point"], "mean OuterPoint", "#0F766E", size=9)

    for idx, point in enumerate(mesial_points):
        add_point(fig, point, f"Mesial {idx}", "#60A5FA", size=5)

    for idx, point in enumerate(distal_points):
        add_point(fig, point, f"Distal {idx}", "#FB923C", size=5)

    for idx, point in enumerate(outer_points):
        add_point(fig, point, f"OuterPoint {idx}", "#14B8A6", size=5)

    scores = analysis["cusp_scores"]
    best_idx = analysis["most_buccal_index"]
    for idx, point in enumerate(cusp_points):
        color = "#DC2626" if idx == best_idx else "#7C3AED"
        name = f"Cusp {idx} score={scores[idx]:.3f}"
        add_point(fig, point, name, color, size=8 if idx == best_idx else 6)

    mirrored = analysis["mirrored_cusp"]
    if mirrored is not None:
        add_point(fig, mirrored, "synthetic mirrored buccal cusp", "#F97316", size=9, symbol="diamond")
        source = analysis["most_buccal_cusp"]
        fig.add_trace(
            go.Scatter3d(
                x=[source[0], mirrored[0]],
                y=[source[1], mirrored[1]],
                z=[source[2], mirrored[2]],
                mode="lines",
                line=dict(color="#F97316", width=5, dash="dash"),
                name="mirror pair",
            )
        )

    add_vector(fig, center, buccal_axis, "buccal axis: center -> OuterPoint", "#0EA5E9", axis_scale)
    add_vector(fig, center, lateral_axis, "Mesial-Distal axis: mirror normal", "#EAB308", axis_scale)

    fig.update_layout(
        title=(
            f"FDI {fdi} cusp mirror geometry<br>"
            "Plane: through tooth center, normal = Mesial-Distal axis. "
            "Cusp scores are projections onto buccal axis."
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=72),
        legend=dict(itemsizing="constant"),
    )
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize posterior cusp buccal selection and mirror geometry."
    )
    parser.add_argument("--obj", required=True, help="Path to patient jaw OBJ file")
    parser.add_argument("--label", required=True, help="Path to segmentation label JSON")
    parser.add_argument("--kpt", required=True, help="Path to keypoint JSON")
    parser.add_argument("--fdi", required=True, type=int, help="Posterior tooth FDI id")
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Defaults to cusp_mirror_FDI_<fdi>.html",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output or f"cusp_mirror_FDI_{args.fdi}.html")

    tooth_vertices, mesial_points, distal_points, outer_points, cusp_points = load_tooth_landmarks(
        args.obj, args.label, args.kpt, args.fdi
    )
    analysis = analyze_geometry(
        tooth_vertices,
        mesial_points,
        distal_points,
        outer_points,
        cusp_points,
    )
    fig = build_figure(
        tooth_vertices,
        mesial_points,
        distal_points,
        outer_points,
        cusp_points,
        analysis,
        args.fdi,
    )
    fig.write_html(output, include_plotlyjs="cdn")

    print(f"Wrote {output.resolve()}")
    print(f"Tooth vertices: {len(tooth_vertices)}")
    print(f"Mesial count: {len(mesial_points)}")
    print(f"Distal count: {len(distal_points)}")
    print(f"OuterPoint count: {len(outer_points)}")
    print(f"Cusp count: {len(cusp_points)}")
    print(f"Most buccal cusp index: {analysis['most_buccal_index']}")
    print("Cusp buccal scores:", np.round(analysis["cusp_scores"], 4).tolist())
    print("Tooth center:", np.round(analysis["tooth_center"], 4).tolist())
    print("Buccal axis:", np.round(analysis["buccal_axis"], 4).tolist())
    print("Mesial-Distal axis:", np.round(analysis["lateral_axis"], 4).tolist())
    if analysis["mirrored_cusp"] is None:
        print("Synthetic cusp: not generated because no cusp is in the buccal half-space")
    else:
        print("Synthetic cusp:", np.round(analysis["mirrored_cusp"], 4).tolist())


if __name__ == "__main__":
    main()
