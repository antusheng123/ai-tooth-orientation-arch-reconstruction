import os
import glob
import json
import numpy as np
import torch
import trimesh
import streamlit as st
import plotly.graph_objects as go
from torch.amp import autocast
import re

# Import the model you just trained
from point_transformer_model import LandmarkPointTransformer

# ==========================================
# 1. Configuration & Global Constants
# ==========================================
st.set_page_config(page_title="🦷 Dental Surface Colorizer", layout="wide")

# Path to your model weights
MODEL_PATH = "best_single_tooth_model_6d.pt"
# Root directory for test-set raw models (.off or .obj)
DATA_ROOT = r"F:\NDCS_3DS_data\Single_Teeth_Y\test"
METRIC_THRESHOLDS_MM = (1.0, 2.0, 3.0)
STANDARD_CLASSES = ["Mesial", "Distal", "InnerPoint", "OuterPoint", "FacialPoint"]

# Class mapping and color definitions
CLASS_INFO = {
    0: {"name": "Mesial", "color": "rgb(255, 99, 132)", "rgb": [255, 99, 132]},  # Red
    1: {"name": "Distal", "color": "rgb(54, 162, 235)", "rgb": [54, 162, 235]},  # Blue
    2: {"name": "Lingual/Inner", "color": "rgb(75, 192, 192)", "rgb": [75, 192, 192]},  # Green
    3: {"name": "Buccal/Outer", "color": "rgb(255, 206, 86)", "rgb": [255, 206, 86]},  # Yellow
    4: {"name": "Occlusal/Facial", "color": "rgb(153, 102, 255)", "rgb": [153, 102, 255]}  # Purple
}
DEFAULT_TOOTH_COLOR = "rgb(245, 245, 235)"  # Bone White


# ==========================================
# 2. Cached Loading & Inference Logic
# ==========================================
@st.cache_resource(show_spinner="Loading Model...")
def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LandmarkPointTransformer(num_landmarks=5).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    else:
        st.error(f"❌ Model weights file not found: {model_path}")
        return None, device


@st.cache_data(show_spinner="Processing 3D Mesh...")
def process_mesh_and_predict(mesh_path, _model, device):
    """Load Mesh -> Extract 6D features -> Model Inference -> Return points and mesh data"""
    try:
        # 1. Load 3D Mesh
        mesh = trimesh.load(mesh_path, process=False)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        # 2. Sample 4096 points and calculate normals (6D input)
        points, face_indices = trimesh.sample.sample_surface(mesh, 4096)
        normals = mesh.face_normals[face_indices]
        data_6d = np.concatenate([points, normals], axis=1).astype(np.float32)

        # 3. Extract FDI Tooth ID
        try:
            filename = os.path.basename(mesh_path)
            # Regex to automatically grab the numbers after 'tooth_'
            match = re.search(r'tooth_(\d+)', filename)
            if match:
                tooth_id = int(match.group(1))
            else:
                tooth_id = 0
        except:
            tooth_id = 0

        # 4. Convert to Tensor and Inference
        pts_tensor = torch.from_numpy(data_6d).unsqueeze(0).to(device)  # (1, 4096, 6)
        tid_tensor = torch.tensor([tooth_id], dtype=torch.long).to(device)

        with torch.no_grad():
            with autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
                pred = _model(pts_tensor, tid_tensor)  # (1, 5, 3)

        pred_landmarks = pred[0].float().cpu().numpy()

        return vertices, faces, pred_landmarks, tooth_id
    except Exception as e:
        st.error(f"Failed to parse mesh: {e}")
        return None, None, None, None


@st.cache_data(show_spinner=False)
def load_ground_truth_landmarks(mesh_path):
    """Load the same-name JSON landmark file if it exists."""
    json_path = os.path.splitext(mesh_path)[0] + ".json"
    if not os.path.exists(json_path):
        return None, None, json_path

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.warning(f"Ground-truth JSON exists but cannot be read: {e}")
        return None, None, json_path

    lm_dict = {
        item.get("class"): np.asarray(item.get("coord"), dtype=np.float32)
        for item in data
        if item.get("class") in STANDARD_CLASSES and item.get("coord") is not None
    }

    gt = np.zeros((len(STANDARD_CLASSES), 3), dtype=np.float32)
    mask = np.zeros(len(STANDARD_CLASSES), dtype=bool)
    for idx, class_name in enumerate(STANDARD_CLASSES):
        if class_name in lm_dict:
            gt[idx] = lm_dict[class_name]
            mask[idx] = True

    return gt, mask, json_path


def compute_landmark_metrics(pred_landmarks, gt_landmarks, gt_mask, thresholds=METRIC_THRESHOLDS_MM):
    """Compute one-to-one same-class landmark localization metrics."""
    valid_idx = np.where(gt_mask)[0]
    if len(valid_idx) == 0:
        return None

    errors = np.linalg.norm(pred_landmarks[valid_idx] - gt_landmarks[valid_idx], axis=1)
    metrics = {
        "valid_count": int(len(valid_idx)),
        "mee": float(np.mean(errors)),
        "per_landmark": []
    }

    for local_i, class_idx in enumerate(valid_idx):
        metrics["per_landmark"].append({
            "Surface": CLASS_INFO[class_idx]["name"],
            "Error (mm)": float(errors[local_i]),
            "Pred X": float(pred_landmarks[class_idx, 0]),
            "Pred Y": float(pred_landmarks[class_idx, 1]),
            "Pred Z": float(pred_landmarks[class_idx, 2]),
            "GT X": float(gt_landmarks[class_idx, 0]),
            "GT Y": float(gt_landmarks[class_idx, 1]),
            "GT Z": float(gt_landmarks[class_idx, 2]),
        })

    for threshold in thresholds:
        tp = int(np.sum(errors <= threshold))
        total = int(len(errors))

        # One prediction is produced for each anatomical class, so this is the
        # threshold hit rate over valid same-class landmarks.
        metrics[f"success_rate@{threshold:g}mm"] = tp / total
        metrics[f"hits@{threshold:g}mm"] = tp

    return metrics


def summarize_errors(errors, thresholds=METRIC_THRESHOLDS_MM):
    if not errors:
        return None

    errors_np = np.asarray(errors, dtype=np.float32)
    summary = {
        "Samples": int(len(errors_np)),
        "MEE (mm)": float(np.mean(errors_np)),
        "Median Error (mm)": float(np.median(errors_np)),
        "Max Error (mm)": float(np.max(errors_np)),
    }

    for threshold in thresholds:
        hits = int(np.sum(errors_np <= threshold))
        rate = hits / len(errors_np)
        summary[f"Success Rate@{threshold:g}mm"] = rate
        summary[f"Hits@{threshold:g}mm"] = f"{hits}/{len(errors_np)}"

    return summary


def format_batch_summary_tables(summary):
    """Format dataset-level metrics into semantically grouped tables."""
    coverage_table = [{
        "Total Test Teeth": summary.get("Total Teeth Found", "-"),
        "Evaluated Teeth": summary.get("Evaluated Teeth", "-"),
        "Skipped Teeth": summary.get("Skipped Teeth", "-"),
        "Valid Landmark Samples": summary.get("Samples", "-"),
    }]

    error_table = [
        {"Statistic": "Mean Euclidean Error", "Value (mm)": summary["MEE (mm)"]},
        {"Statistic": "Median Error", "Value (mm)": summary["Median Error (mm)"]},
        {"Statistic": "Max Error", "Value (mm)": summary["Max Error (mm)"]},
    ]

    threshold_table = []
    for threshold in METRIC_THRESHOLDS_MM:
        rate_key = f"Success Rate@{threshold:g}mm"
        hits_key = f"Hits@{threshold:g}mm"
        if rate_key in summary and hits_key in summary:
            threshold_table.append({
                "Threshold (mm)": threshold,
                "Success Rate (%)": summary[rate_key] * 100.0,
                "Hits / Total": summary[hits_key],
            })

    return coverage_table, error_table, threshold_table


def evaluate_mesh_files(mesh_files, model, device):
    all_errors = []
    used_teeth = 0
    skipped_teeth = 0

    eval_files = mesh_files
    progress = st.progress(0.0, text="Running batch evaluation...")

    for idx, mesh_path in enumerate(eval_files):
        gt_landmarks, gt_mask, _ = load_ground_truth_landmarks(mesh_path)
        if gt_landmarks is None or gt_mask is None or not np.any(gt_mask):
            skipped_teeth += 1
            progress.progress((idx + 1) / len(eval_files), text="Running batch evaluation...")
            continue

        _, _, pred_landmarks, _ = process_mesh_and_predict(mesh_path, model, device)
        if pred_landmarks is None:
            skipped_teeth += 1
            progress.progress((idx + 1) / len(eval_files), text="Running batch evaluation...")
            continue

        valid_idx = np.where(gt_mask)[0]
        errors = np.linalg.norm(pred_landmarks[valid_idx] - gt_landmarks[valid_idx], axis=1)
        all_errors.extend(errors.tolist())
        used_teeth += 1
        progress.progress((idx + 1) / len(eval_files), text="Running batch evaluation...")

    progress.empty()
    summary = summarize_errors(all_errors)
    if summary is None:
        return None

    summary["Evaluated Teeth"] = used_teeth
    summary["Skipped Teeth"] = skipped_teeth
    summary["Total Teeth Found"] = len(eval_files)
    return summary


# ==========================================
# 3. Streamlit UI Construction
# ==========================================
st.title("🦷 Automated Dental Surface Colorization System")
st.markdown("High-precision landmark mapping based on Point Transformer (6D + FDI)")

# --- Sidebar: Control Panel ---
with st.sidebar:
    st.header("⚙️ Control Panel")

    model_path = st.text_input("Model weights path", value=MODEL_PATH)
    data_root = st.text_input("Data root", value=DATA_ROOT)

    # Scan file selection
    mesh_files = glob.glob(os.path.join(data_root, "**", "*.off"), recursive=True) + \
                 glob.glob(os.path.join(data_root, "**", "*.obj"), recursive=True)

    if not mesh_files:
        st.warning(f"No 3D model files found in {data_root}.")
        st.stop()

    selected_file = st.selectbox("📂 Select a tooth for analysis", mesh_files, format_func=lambda x: os.path.basename(x))

    st.divider()
    st.subheader("🎨 Color Display Settings")

    # Color radius slider (Core function: controls color diffusion range)
    color_radius = st.slider(
        "Color Spread Radius (mm)",
        min_value=2.0, max_value=12.0, value=6.0, step=0.5,
        help="The distance from the predicted landmark within which vertices will be colored."
    )

    st.write("✅ **Select surfaces to render**")

    # UI State: Record which classes are checked
    selected_classes = []
    for idx, info in CLASS_INFO.items():
        # Default: check the first 4 surfaces (internship requirement), uncheck the 5th (Occlusal)
        default_state = True if idx < 4 else False
        if st.checkbox(info["name"], value=default_state):
            selected_classes.append(idx)

    st.divider()
    show_landmarks = st.checkbox("📍 Show AI Predicted Landmarks", value=True)

    st.divider()
    st.subheader("📊 Batch Evaluation")
    st.caption(f"Batch metrics will evaluate all {len(mesh_files)} teeth found under the test data root.")
    run_batch_eval = st.button("Run batch metrics")

model, device = load_model(model_path)

# --- Main Interface: Rendering and Display ---
if model and run_batch_eval:
    st.subheader("Dataset-Level Metrics")
    batch_summary = evaluate_mesh_files(mesh_files, model, device)
    if batch_summary is None:
        st.warning("No evaluable teeth found. Each mesh needs a same-name JSON with ground-truth landmarks.")
    else:
        coverage_table, error_table, threshold_table = format_batch_summary_tables(batch_summary)

        st.write("**Dataset Coverage**")
        st.dataframe(coverage_table, hide_index=True, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Error Statistics**")
            st.dataframe(
                error_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Value (mm)": st.column_config.NumberColumn(format="%.4f"),
                },
            )

        with col_b:
            st.write("**Threshold Performance**")
            st.dataframe(
                threshold_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Threshold (mm)": st.column_config.NumberColumn(format="%.1f"),
                    "Success Rate (%)": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        st.caption("Success rate is the percentage of valid same-class landmarks whose prediction falls within the distance threshold.")

if model and selected_file:
    # Run inference logic
    vertices, faces, pred_landmarks, tooth_id = process_mesh_and_predict(selected_file, model, device)

    if vertices is not None:
        gt_landmarks, gt_mask, gt_json_path = load_ground_truth_landmarks(selected_file)
        landmark_metrics = None
        if gt_landmarks is not None and gt_mask is not None:
            landmark_metrics = compute_landmark_metrics(pred_landmarks, gt_landmarks, gt_mask)

        col1, col2 = st.columns([1, 3])

        with col1:
            st.info(f"**File:** {os.path.basename(selected_file)}")
            st.success(f"**FDI Tooth ID:** {tooth_id if tooth_id != 0 else 'Unknown'}")
            st.metric("Vertex Count", len(vertices))
            st.metric("Face Count", len(faces))

            st.divider()
            st.subheader("📊 Landmark Metrics")
            if landmark_metrics is None:
                st.caption("No same-name ground-truth JSON found, so only prediction visualization is available.")
                st.caption(f"Expected: {gt_json_path}")
            else:
                st.metric("MEE", f"{landmark_metrics['mee']:.3f} mm")
                st.caption(f"Valid GT landmarks: {landmark_metrics['valid_count']}")

                for threshold in METRIC_THRESHOLDS_MM:
                    key = f"{threshold:g}mm"
                    success_rate = landmark_metrics[f"success_rate@{key}"]
                    hits = landmark_metrics[f"hits@{key}"]
                    st.write(f"**@ {threshold:g} mm**")
                    st.metric(
                        "Success Rate",
                        f"{success_rate:.1%}",
                        help=f"{hits}/{landmark_metrics['valid_count']} valid landmarks within {threshold:g} mm"
                    )

            st.write("💡 **Legend**")
            for idx in selected_classes:
                st.markdown(
                    f"<span style='color: {CLASS_INFO[idx]['color']}; font-size: 20px;'>●</span> {CLASS_INFO[idx]['name']}",
                    unsafe_allow_html=True)

        with col2:
            # ==========================================
            # Core Algorithm: Distance-based Mesh Vertex Coloring (Voronoi)
            # ==========================================
            # Initialize all vertex colors to the default tooth color
            vertex_colors = np.full((len(vertices), 3), [245, 245, 235], dtype=np.uint8)
            # Track current shortest distance (initialize with max radius)
            min_dists = np.full(len(vertices), color_radius)

            for class_idx in selected_classes:
                lm_coord = pred_landmarks[class_idx]
                # Calculate distance from all vertices to this landmark (broadcasting)
                dists = np.linalg.norm(vertices - lm_coord, axis=1)

                # Mask: vertices within threshold and closer than other landmarks
                mask = dists < min_dists

                # Update colors and shortest distances
                vertex_colors[mask] = CLASS_INFO[class_idx]["rgb"]
                min_dists[mask] = dists[mask]

            # Convert to Plotly-compatible RGB strings
            vertex_colors_str = [f'rgb({c[0]},{c[1]},{c[2]})' for c in vertex_colors]

            # ==========================================
            # 3D Chart Rendering
            # ==========================================
            fig = go.Figure()

            # Add Colored 3D Mesh
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                vertexcolor=vertex_colors_str,  # Inject colors
                opacity=1.0,
                name="Tooth Mesh",
                hoverinfo="skip"
            ))

            # Optional: Display AI predicted landmark points
            if show_landmarks:
                for class_idx in selected_classes:
                    lm_coord = pred_landmarks[class_idx]
                    fig.add_trace(go.Scatter3d(
                        x=[lm_coord[0]], y=[lm_coord[1]], z=[lm_coord[2]],
                        mode='markers+text',
                        marker=dict(size=8, color=CLASS_INFO[class_idx]["color"], symbol='diamond',
                                    line=dict(color='black', width=2)),
                        name=CLASS_INFO[class_idx]["name"],
                        text=[CLASS_INFO[class_idx]["name"].split(" ")[0]],
                        textposition="top center",
                        hovertemplate=f"{CLASS_INFO[class_idx]['name']}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>"
                    ))

            fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=700,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            if landmark_metrics is not None:
                with st.expander("Per-landmark error table", expanded=False):
                    st.dataframe(
                        landmark_metrics["per_landmark"],
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Error (mm)": st.column_config.NumberColumn(format="%.3f"),
                            "Pred X": st.column_config.NumberColumn(format="%.3f"),
                            "Pred Y": st.column_config.NumberColumn(format="%.3f"),
                            "Pred Z": st.column_config.NumberColumn(format="%.3f"),
                            "GT X": st.column_config.NumberColumn(format="%.3f"),
                            "GT Y": st.column_config.NumberColumn(format="%.3f"),
                            "GT Z": st.column_config.NumberColumn(format="%.3f"),
                        }
                    )

