import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from dataset import (
    ANTERIOR_FDI,
    LANDMARK_COORD_DIM,
    LANDMARK_NAMES,
    LOWER_FDI,
    NUM_LANDMARKS,
    POSTERIOR_FDI,
    SEQ_LEN,
    UPPER_FDI,
    denormalize_landmarks,
    extract_arch_landmarks,
    landmarks_to_centers,
    normalize_landmarks,
    sample_consecutive_valid_teeth,
)
from model import MaskedArchRegressor
from postprocess import generate_bspline_curve


SEG_ROOT = r"F:\NDCS_3DS_data\segmentation_data_for_single_teeth\train"
KPT_ROOT = r"F:\NDCS_3DS_data\3DTeethLand_landmarks_train"
MODEL_PATH = "best_landmark_model.pth"

LANDMARK_COLORS = {
    "Anterior Point0": "#0057FF",
    "Anterior Point1": "#00A651",
    "Posterior Point0": "#FF7A00",
    "Posterior Point1": "#9B00FF",
}

STATE_COLORS = {
    "Observed": "#6B7280",
    "Predicted": "#DC2626",
    "Ground truth": "#111827",
}


st.set_page_config(page_title="Dental Arch Landmark Completion", layout="wide")
st.title("Dental Arch Pipeline 1 Completion")
st.caption(
    "Consecutive-mask Transformer prediction with point-anchor B-spline fitting"
)


@st.cache_resource(show_spinner="Loading landmark prediction model...")
def load_model_weights():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedArchRegressor().to(device)
    if not os.path.exists(MODEL_PATH):
        return None, device

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as error:
        st.error(f"Model weights are incompatible with the Pipeline 1 model: {error}")
        return None, device
    model.eval()
    return model, device


@st.cache_data(show_spinner=None)
def get_single_patient_data(option_string):
    patient_id, jaw_name = option_string.rsplit("_", 1)
    jaw_value = 0 if jaw_name == "upper" else 1
    patient_seg_dir = os.path.join(SEG_ROOT, jaw_name, patient_id)
    patient_kpt_dir = os.path.join(KPT_ROOT, jaw_name, patient_id)

    obj_path = os.path.join(patient_seg_dir, f"{patient_id}_{jaw_name}.obj")
    label_path = os.path.join(patient_seg_dir, f"{patient_id}_{jaw_name}.json")
    kpt_path = os.path.join(
        patient_kpt_dir, f"{patient_id}_{jaw_name}__kpt.json"
    )
    if not all(os.path.exists(path) for path in (obj_path, label_path, kpt_path)):
        return None, None, None, "Cannot find the OBJ, label, and landmark files"

    try:
        landmarks, tooth_mask, landmark_mask = extract_arch_landmarks(
            obj_path, label_path, kpt_path, jaw_value
        )
        return landmarks, tooth_mask, landmark_mask, None
    except Exception as error:
        return None, None, None, str(error)


def find_patient_options():
    options = []
    for jaw_name in ("upper", "lower"):
        seg_jaw_dir = os.path.join(SEG_ROOT, jaw_name)
        kpt_jaw_dir = os.path.join(KPT_ROOT, jaw_name)
        if not os.path.isdir(seg_jaw_dir) or not os.path.isdir(kpt_jaw_dir):
            continue

        for patient_id in os.listdir(seg_jaw_dir):
            seg_patient = os.path.join(seg_jaw_dir, patient_id)
            kpt_patient = os.path.join(kpt_jaw_dir, patient_id)
            required = (
                os.path.join(seg_patient, f"{patient_id}_{jaw_name}.obj"),
                os.path.join(seg_patient, f"{patient_id}_{jaw_name}.json"),
                os.path.join(kpt_patient, f"{patient_id}_{jaw_name}__kpt.json"),
            )
            if all(os.path.exists(path) for path in required):
                options.append(f"{patient_id}_{jaw_name}")
    return sorted(options)


def tooth_region(fdi):
    if fdi in ANTERIOR_FDI:
        return "Anterior"
    if fdi in POSTERIOR_FDI:
        return "Posterior"
    return "Unknown"


def point_display_name(fdi, point_idx):
    if fdi in ANTERIOR_FDI:
        return "Mesial" if point_idx == 0 else "Distal"
    if fdi in POSTERIOR_FDI:
        return f"Buccal cusp {point_idx + 1}"
    return LANDMARK_NAMES[point_idx]


def point_color_key(fdi, point_idx):
    return f"{tooth_region(fdi)} Point{point_idx}"


def anterior_curve_point_order(fdi):
    """Order Mesial/Distal so adjacent anterior contact sides stay adjacent."""
    quadrant = fdi // 10
    if quadrant in (1, 4):
        return (1, 0)  # distal -> mesial while traversing right side to midline
    return (0, 1)  # mesial -> distal while traversing left side from midline


def collect_curve_anchor_points(landmarks, tooth_valid_mask, fdi_order):
    """
    Build ordered Pipeline 1 anchors for curve fitting.

    Anterior teeth have known proximal semantics. Posterior teeth only expose
    two selected buccal cusps, so their within-tooth order is chosen to keep
    the anchor path locally continuous.
    """
    anchors = []
    previous_point = None
    valid_indices = [idx for idx in range(SEQ_LEN) if tooth_valid_mask[idx]]

    for position, tooth_idx in enumerate(valid_indices):
        fdi = fdi_order[tooth_idx]
        tooth_points = landmarks[tooth_idx]

        if fdi in ANTERIOR_FDI:
            candidate_orders = [anterior_curve_point_order(fdi)]
        else:
            candidate_orders = [(0, 1), (1, 0)]

        if previous_point is None and len(valid_indices) > position + 1:
            next_points = landmarks[valid_indices[position + 1]]
            order = min(
                candidate_orders,
                key=lambda candidate: np.min(
                    np.linalg.norm(next_points - tooth_points[candidate[-1]], axis=1)
                ),
            )
        elif previous_point is not None:
            order = min(
                candidate_orders,
                key=lambda candidate: np.linalg.norm(
                    tooth_points[candidate[0]] - previous_point
                ),
            )
        else:
            order = candidate_orders[0]

        ordered_points = tooth_points[list(order)]
        anchors.extend(ordered_points)
        previous_point = ordered_points[-1]

    return np.asarray(anchors, dtype=np.float32)


model, device = load_model_weights()
patient_options = find_patient_options()

if model is None:
    st.error(f"Model weights not found: {MODEL_PATH}. Train the new model first.")
    st.stop()
if not patient_options:
    st.error("No complete patient file sets were found under the configured paths.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    selected_patient = st.selectbox("Patient arch", patient_options)

with st.spinner(f"Reading {selected_patient}..."):
    raw_landmarks, tooth_valid_mask, landmark_valid_mask, error = (
        get_single_patient_data(selected_patient)
    )

if error is not None:
    st.error(error)
    st.stop()
if tooth_valid_mask.sum() < 4:
    st.error(
        "This arch has fewer than four teeth with complete landmark annotations."
    )
    st.stop()

_, jaw_name = selected_patient.rsplit("_", 1)
current_fdi = UPPER_FDI if jaw_name == "upper" else LOWER_FDI
jaw_value = 0 if jaw_name == "upper" else 1

# Establish the patient-specific reference before applying any simulated mask.
ground_truth_centers = landmarks_to_centers(raw_landmarks, tooth_valid_mask)
curve_anchor_points = collect_curve_anchor_points(
    raw_landmarks, tooth_valid_mask, current_fdi
)
curve_points = generate_bspline_curve(
    curve_anchor_points, num_eval_points=500
)

with st.sidebar:
    incomplete_count = SEQ_LEN - int(tooth_valid_mask.sum())
    st.caption(
        f"{int(tooth_valid_mask.sum())} complete teeth; "
        f"{incomplete_count} missing or incompletely annotated positions"
    )
    valid_count = int(tooth_valid_mask.sum())
    max_drop = min(6, valid_count - 4)
    can_mask = max_drop >= 3
    enable_mask = st.checkbox("Enable consecutive mask", value=True)
    if enable_mask and can_mask:
        mask_count = st.slider(
            "Consecutive missing teeth",
            min_value=3,
            max_value=max_drop,
            value=min(3, max_drop),
            help=(
                "A random consecutive window over valid FDI positions is masked, "
                "while at least four observed teeth are retained."
            ),
        )
        mask_state_key = f"pipeline1_mask_{selected_patient}_{mask_count}"
        if (
            st.button("Randomize mask")
            or st.session_state.get("mask_state_key") != mask_state_key
        ):
            st.session_state["mask_state_key"] = mask_state_key
            st.session_state["pipeline1_dropped_mask"] = (
                sample_consecutive_valid_teeth(
                    tooth_valid_mask,
                    min_drop=mask_count,
                    max_drop=mask_count,
                    min_remaining_teeth=4,
                )
            )
        dropped_mask = st.session_state.get("pipeline1_dropped_mask")
        if dropped_mask is None:
            dropped_mask = sample_consecutive_valid_teeth(
                tooth_valid_mask,
                min_drop=mask_count,
                max_drop=mask_count,
                min_remaining_teeth=4,
            )
            st.session_state["pipeline1_dropped_mask"] = dropped_mask
        masked_fdi = [
            current_fdi[idx] for idx in range(SEQ_LEN) if dropped_mask[idx]
        ]
        st.write("Masked FDI:", ", ".join(str(fdi) for fdi in masked_fdi))
    else:
        if enable_mask:
            st.warning("At least seven valid teeth are required to mask 3 and keep 4.")
        dropped_mask = np.zeros(SEQ_LEN, dtype=bool)

normalized, normalization_origin, normalization_scale = normalize_landmarks(
    raw_landmarks, landmark_valid_mask
)

input_landmarks = normalized.copy()
input_landmarks[dropped_mask] = 0.0
input_landmarks[~tooth_valid_mask] = 0.0
tooth_missing = (~tooth_valid_mask | dropped_mask).astype(np.float32)
jaw_feature = np.full((SEQ_LEN, 1), jaw_value, dtype=np.float32)
features = np.concatenate(
    [
        input_landmarks.reshape(SEQ_LEN, LANDMARK_COORD_DIM),
        tooth_missing[:, None],
        jaw_feature,
    ],
    axis=-1,
)

with torch.no_grad():
    feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    predicted_normalized = model(feature_tensor)[0].cpu().numpy()

prediction_mask = np.repeat(dropped_mask[:, None], NUM_LANDMARKS, axis=1)
predicted_world = denormalize_landmarks(
    predicted_normalized,
    normalization_origin,
    normalization_scale,
    prediction_mask,
)

final_landmarks = raw_landmarks.copy()
final_landmarks[dropped_mask] = predicted_world[dropped_mask]

display_centers = landmarks_to_centers(final_landmarks, tooth_valid_mask)
observed_mask = tooth_valid_mask & ~dropped_mask
observed_centers = display_centers[observed_mask]
predicted_centers = display_centers[dropped_mask]
masked_ground_truth_centers = ground_truth_centers[dropped_mask]
center_errors = np.linalg.norm(
    predicted_centers - masked_ground_truth_centers, axis=1
)
landmark_errors = np.linalg.norm(
    predicted_world[dropped_mask] - raw_landmarks[dropped_mask], axis=-1
).mean(axis=1)

predicted_fdi = [current_fdi[idx] for idx in range(SEQ_LEN) if dropped_mask[idx]]
masked_indices = np.flatnonzero(dropped_mask)

with st.sidebar:
    st.divider()
    st.header("View")
    show_curve = st.checkbox("Fixed reference curve", value=True)
    show_hidden_truth = st.checkbox("Hidden ground truth", value=True)
    show_midpoints = st.checkbox("Midpoints", value=False)
    show_error_lines = st.checkbox("Midpoint error lines", value=False)

summary_col, plot_col = st.columns([1.15, 3.85])
with summary_col:
    st.metric("Observed teeth", int(observed_mask.sum()))
    st.metric("Masked run", int(dropped_mask.sum()))
    st.metric("Curve anchors", len(curve_anchor_points))

    st.subheader("Pipeline 1 Points")
    st.markdown(
        "- **Anterior teeth:** Point0 = Mesial, Point1 = Distal\n"
        "- **Posterior teeth:** Point0/Point1 = two buccal cusps nearest OuterPoint\n"
        "- **Curve:** fixed from all ground-truth Pipeline 1 points before masking"
    )

    if predicted_fdi:
        st.subheader("Masked Teeth")
        for idx in masked_indices:
            fdi = current_fdi[idx]
            st.write(f"FDI {fdi} · {tooth_region(fdi)}")

    if len(center_errors):
        with st.expander("Prediction errors", expanded=False):
            error_rows = []
            for fdi, center_error, landmark_error in zip(
                predicted_fdi, center_errors, landmark_errors
            ):
                error_rows.append(
                    {
                        "FDI": fdi,
                        "region": tooth_region(fdi),
                        "mean point error mm": round(float(landmark_error), 3),
                        "midpoint error mm": round(float(center_error), 3),
                    }
                )
            st.dataframe(error_rows, hide_index=True, use_container_width=True)

with plot_col:
    figure = go.Figure()

    if show_curve and len(curve_points) >= 2:
        figure.add_trace(
            go.Scatter3d(
                x=curve_points[:, 0],
                y=curve_points[:, 1],
                z=curve_points[:, 2],
                mode="lines",
                line=dict(color="#0F766E", width=7),
                name="Fixed ground-truth B-spline",
                hovertemplate="Fixed curve<extra></extra>",
            )
        )

    for point_idx in range(NUM_LANDMARKS):
        for region_name in ("Anterior", "Posterior"):
            source_indices = [
                idx
                for idx in range(SEQ_LEN)
                if observed_mask[idx] and tooth_region(current_fdi[idx]) == region_name
            ]
            if not source_indices:
                continue
            points = raw_landmarks[source_indices, point_idx]
            labels = [
                f"FDI {current_fdi[idx]}<br>{point_display_name(current_fdi[idx], point_idx)}"
                for idx in source_indices
            ]
            color = LANDMARK_COLORS[f"{region_name} Point{point_idx}"]
            figure.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(size=6, color=color, opacity=0.82),
                    text=labels,
                    name=f"{region_name} {point_display_name(current_fdi[source_indices[0]], point_idx)} observed",
                    hovertemplate="%{text}<br>Observed<extra></extra>",
                )
            )

    if show_hidden_truth:
        for point_idx in range(NUM_LANDMARKS):
            points = raw_landmarks[dropped_mask, point_idx]
            if not len(points):
                continue
            labels = [
                f"FDI {current_fdi[idx]}<br>{point_display_name(current_fdi[idx], point_idx)}"
                for idx in masked_indices
            ]
            figure.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=7,
                        color=STATE_COLORS["Ground truth"],
                        symbol="circle-open",
                        line=dict(width=3),
                    ),
                    text=labels,
                    name=f"Hidden ground truth Point{point_idx}",
                    hovertemplate="%{text}<br>Hidden ground truth<extra></extra>",
                )
            )

    for point_idx in range(NUM_LANDMARKS):
        points = predicted_world[dropped_mask, point_idx]
        if not len(points):
            continue
        labels = [
            f"FDI {current_fdi[idx]}<br>{point_display_name(current_fdi[idx], point_idx)}"
            for idx in masked_indices
        ]
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=STATE_COLORS["Predicted"],
                    symbol="diamond",
                    line=dict(width=2, color="white"),
                ),
                text=[f"{current_fdi[idx]} P{point_idx}" for idx in masked_indices],
                customdata=labels,
                textposition="top center",
                name=f"Predicted Point{point_idx}",
                hovertemplate="%{customdata}<br>Predicted<extra></extra>",
            )
        )

    if show_midpoints:
        if len(observed_centers):
            figure.add_trace(
                go.Scatter3d(
                    x=observed_centers[:, 0],
                    y=observed_centers[:, 1],
                    z=observed_centers[:, 2],
                    mode="markers",
                    marker=dict(size=4, color=STATE_COLORS["Observed"], opacity=0.55),
                    name="Observed midpoints",
                    hovertemplate="Observed midpoint<extra></extra>",
                )
            )
        if len(predicted_centers):
            figure.add_trace(
                go.Scatter3d(
                    x=predicted_centers[:, 0],
                    y=predicted_centers[:, 1],
                    z=predicted_centers[:, 2],
                    mode="markers",
                    marker=dict(size=7, color=STATE_COLORS["Predicted"], symbol="x"),
                    name="Predicted midpoints",
                    hovertemplate="Predicted midpoint<extra></extra>",
                )
            )

    if show_error_lines and len(predicted_centers):
        for predicted_center, ground_truth_center, fdi in zip(
            predicted_centers, masked_ground_truth_centers, predicted_fdi
        ):
            figure.add_trace(
                go.Scatter3d(
                    x=[predicted_center[0], ground_truth_center[0]],
                    y=[predicted_center[1], ground_truth_center[1]],
                    z=[predicted_center[2], ground_truth_center[2]],
                    mode="lines",
                    line=dict(color=STATE_COLORS["Predicted"], width=4, dash="dot"),
                    showlegend=False,
                    hovertemplate=f"FDI {fdi} midpoint error<extra></extra>",
                )
            )

    figure.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=760,
        legend=dict(itemsizing="constant", orientation="h", y=0.02, x=0.0),
    )
    st.plotly_chart(figure, use_container_width=True)
