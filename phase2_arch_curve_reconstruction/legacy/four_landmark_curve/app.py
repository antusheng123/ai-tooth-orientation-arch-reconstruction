import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from dataset import (
    LANDMARK_COORD_DIM,
    LANDMARK_NAMES,
    LOWER_FDI,
    NUM_LANDMARKS,
    SEQ_LEN,
    UPPER_FDI,
    denormalize_landmarks,
    extract_arch_landmarks,
    landmarks_to_centers,
    normalize_landmarks,
)
from model import MaskedArchRegressor
from postprocess import generate_bspline_curve


SEG_ROOT = r"F:\NDCS_3DS_data\segmentation_data_for_single_teeth\train"
KPT_ROOT = r"F:\NDCS_3DS_data\3DTeethLand_landmarks_train"
MODEL_PATH = "best_landmark_model.pth"

LANDMARK_COLORS = {
    "Mesial": "#2E86DE",
    "Distal": "#00A86B",
    "OuterPoint": "#F39C12",
    "InnerPoint": "#8E44AD",
}


st.set_page_config(page_title="Dental Arch Landmark Completion", layout="wide")
st.title("Dental Arch Four-Landmark Completion")
st.caption(
    "Masked Transformer prediction with tooth-center parametric B-spline fitting"
)


@st.cache_resource(show_spinner="Loading landmark prediction model...")
def load_model_weights():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedArchRegressor().to(device)
    if not os.path.exists(MODEL_PATH):
        return None, device

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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


def compute_center_error_sensitivity(
    model,
    features,
    target_idx,
    target_center_normalized,
    normalization_scale,
    observed_mask,
    device,
):
    """
    Measure local center-error sensitivity to each observed input landmark.

    The gradient of the selected target's center error in millimeters is taken
    with respect to each landmark's normalized XYZ coordinates. The XYZ
    gradient norm gives one non-negative sensitivity score per landmark.
    """
    feature_tensor = (
        torch.from_numpy(features).unsqueeze(0).to(device).requires_grad_(True)
    )
    predictions = model(feature_tensor)[0]
    predicted_center = predictions[target_idx].mean(dim=0)
    target_center = torch.as_tensor(
        target_center_normalized,
        dtype=predicted_center.dtype,
        device=device,
    )
    center_error_mm = (
        torch.linalg.vector_norm(predicted_center - target_center)
        * float(normalization_scale)
    )
    feature_gradients = torch.autograd.grad(
        center_error_mm, feature_tensor, retain_graph=False
    )[0]
    coordinate_gradients = feature_gradients[
        0, :, :LANDMARK_COORD_DIM
    ].reshape(SEQ_LEN, NUM_LANDMARKS, 3)
    sensitivities = torch.linalg.vector_norm(
        coordinate_gradients, dim=-1
    ).detach().cpu().numpy()
    sensitivities[~observed_mask] = 0.0
    return float(center_error_mm.detach().cpu()), sensitivities


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
curve_points = generate_bspline_curve(
    ground_truth_centers[tooth_valid_mask], num_eval_points=500
)

with st.sidebar:
    incomplete_count = SEQ_LEN - int(tooth_valid_mask.sum())
    st.caption(
        f"{int(tooth_valid_mask.sum())} complete teeth; "
        f"{incomplete_count} missing or incompletely annotated positions"
    )
    existing_fdi = [
        current_fdi[idx] for idx in range(SEQ_LEN) if tooth_valid_mask[idx]
    ]
    teeth_to_drop = st.multiselect(
        "Simulate missing teeth (FDI)",
        existing_fdi,
        max_selections=max(0, len(existing_fdi) - 4),
        help=(
            "At least four observed teeth are retained, matching the model's "
            "training setup and keeping the reference curve well defined."
        ),
    )

normalized, normalization_origin, normalization_scale = normalize_landmarks(
    raw_landmarks, landmark_valid_mask
)
dropped_mask = np.zeros(SEQ_LEN, dtype=bool)
for fdi in teeth_to_drop:
    dropped_mask[current_fdi.index(fdi)] = True

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

summary_col, plot_col = st.columns([1, 4])
with summary_col:
    st.metric("Observed teeth", int(observed_mask.sum()))
    st.metric("Predicted teeth", int(dropped_mask.sum()))
    st.write(
        "The reference curve is fixed from all available ground-truth tooth "
        "centers before any UI masking."
    )
    predicted_fdi = [
        current_fdi[idx] for idx in range(SEQ_LEN) if dropped_mask[idx]
    ]
    if len(center_errors):
        st.write("Prediction error against ground truth:")
        for fdi, center_error, landmark_error in zip(
            predicted_fdi, center_errors, landmark_errors
        ):
            st.metric(
                f"FDI {fdi}",
                f"{center_error:.3f} mm center",
                f"{landmark_error:.3f} mm mean landmark error",
                delta_color="off",
            )

with plot_col:
    figure = go.Figure()

    for landmark_idx, landmark_name in enumerate(LANDMARK_NAMES):
        observed_points = final_landmarks[observed_mask, landmark_idx]
        if len(observed_points):
            figure.add_trace(
                go.Scatter3d(
                    x=observed_points[:, 0],
                    y=observed_points[:, 1],
                    z=observed_points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5, color=LANDMARK_COLORS[landmark_name], opacity=0.75
                    ),
                    name=f"{landmark_name} observed",
                )
            )

        predicted_points = final_landmarks[dropped_mask, landmark_idx]
        if len(predicted_points):
            figure.add_trace(
                go.Scatter3d(
                    x=predicted_points[:, 0],
                    y=predicted_points[:, 1],
                    z=predicted_points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=LANDMARK_COLORS[landmark_name],
                        symbol="diamond",
                        line=dict(width=2, color="white"),
                    ),
                    name=f"{landmark_name} predicted",
                )
            )

    observed_labels = [
        str(current_fdi[idx])
        for idx in range(SEQ_LEN)
        if observed_mask[idx]
    ]
    if len(observed_centers):
        figure.add_trace(
            go.Scatter3d(
                x=observed_centers[:, 0],
                y=observed_centers[:, 1],
                z=observed_centers[:, 2],
                mode="markers+text",
                marker=dict(size=5, color="#202020"),
                text=observed_labels,
                textposition="top center",
                name="Observed centers",
            )
        )

    predicted_labels = [
        f"{current_fdi[idx]} predicted"
        for idx in range(SEQ_LEN)
        if dropped_mask[idx]
    ]
    if len(predicted_centers):
        figure.add_trace(
            go.Scatter3d(
                x=predicted_centers[:, 0],
                y=predicted_centers[:, 1],
                z=predicted_centers[:, 2],
                mode="markers+text",
                marker=dict(size=7, color="#E74C3C", symbol="diamond"),
                text=predicted_labels,
                textposition="top center",
                name="Predicted centers",
            )
        )

        for predicted_center, ground_truth_center in zip(
            predicted_centers, masked_ground_truth_centers
        ):
            figure.add_trace(
                go.Scatter3d(
                    x=[predicted_center[0], ground_truth_center[0]],
                    y=[predicted_center[1], ground_truth_center[1]],
                    z=[predicted_center[2], ground_truth_center[2]],
                    mode="lines",
                    line=dict(color="#E74C3C", width=4, dash="dot"),
                    showlegend=False,
                    hovertemplate="Prediction to ground truth<extra></extra>",
                )
            )

    if len(curve_points) >= 2:
        figure.add_trace(
            go.Scatter3d(
                x=curve_points[:, 0],
                y=curve_points[:, 1],
                z=curve_points[:, 2],
                mode="lines",
                line=dict(color="#17A589", width=7),
                name="Fixed ground-truth center B-spline",
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
        legend=dict(itemsizing="constant"),
    )
    st.plotly_chart(figure, use_container_width=True)

if dropped_mask.any():
    st.divider()
    st.subheader("Landmark contribution to center offset")
    st.caption(
        "The predicted center is the mean of its four landmarks. Each bar is "
        "that landmark's signed contribution along the final center-offset "
        "direction. The four contributions sum exactly to the center error."
    )

    attribution_fdi = st.selectbox(
        "Predicted tooth to explain (FDI)",
        predicted_fdi,
        key="attribution_target_fdi",
    )
    attribution_idx = current_fdi.index(attribution_fdi)
    landmark_error_vectors = (
        predicted_world[attribution_idx] - raw_landmarks[attribution_idx]
    )
    center_offset_vector = landmark_error_vectors.mean(axis=0)
    center_offset_mm = float(np.linalg.norm(center_offset_vector))
    if center_offset_mm > 1e-8:
        offset_direction = center_offset_vector / center_offset_mm
        signed_contributions_mm = (
            landmark_error_vectors @ offset_direction
        ) / NUM_LANDMARKS
        signed_contribution_share = (
            signed_contributions_mm / center_offset_mm * 100.0
        )
    else:
        signed_contributions_mm = np.zeros(NUM_LANDMARKS, dtype=np.float32)
        signed_contribution_share = np.zeros(NUM_LANDMARKS, dtype=np.float32)
    individual_landmark_errors = np.linalg.norm(
        landmark_error_vectors, axis=1
    )

    contribution_figure = go.Figure(
        go.Bar(
            x=list(LANDMARK_NAMES),
            y=signed_contributions_mm,
            marker_color=[
                LANDMARK_COLORS[name] for name in LANDMARK_NAMES
            ],
            customdata=np.column_stack(
                [signed_contribution_share, individual_landmark_errors]
            ),
            hovertemplate=(
                "%{x}<br>Signed center contribution: %{y:.3f} mm"
                "<br>Signed share: %{customdata[0]:.1f}%"
                "<br>Landmark error magnitude: %{customdata[1]:.3f} mm"
                "<extra></extra>"
            ),
        )
    )
    contribution_figure.add_hline(y=0.0, line_color="#555555")
    contribution_figure.update_layout(
        title=f"FDI {attribution_fdi}: exact center-offset decomposition",
        xaxis_title="Predicted landmark",
        yaxis_title="Signed contribution to center offset (mm)",
        margin=dict(l=20, r=20, b=20, t=60),
        height=380,
    )

    attribution_summary, contribution_plot = st.columns([1, 3])
    with attribution_summary:
        st.metric("Center offset", f"{center_offset_mm:.3f} mm")
        if center_offset_mm > 1e-8:
            largest_type_idx = int(np.argmax(signed_contributions_mm))
            st.metric(
                "Largest positive contributor",
                LANDMARK_NAMES[largest_type_idx],
                (
                    f"{signed_contributions_mm[largest_type_idx]:.3f} mm "
                    f"({signed_contribution_share[largest_type_idx]:.1f}%)"
                ),
                delta_color="off",
            )
            st.write(
                "Positive values push the center along its final error "
                "direction; negative values partially cancel that offset."
            )
        else:
            st.info("The predicted and ground-truth centers coincide.")
    with contribution_plot:
        st.plotly_chart(contribution_figure, use_container_width=True)

    with st.expander("Advanced: observed-input gradient sensitivity"):
        st.caption(
            "This separate view asks which observed source landmarks the "
            "model is locally most sensitive to. It is not an exact causal "
            "decomposition."
        )
        target_center_normalized = normalized[attribution_idx].mean(axis=0)
        attributed_error_mm, landmark_sensitivities = (
            compute_center_error_sensitivity(
                model,
                features,
                attribution_idx,
                target_center_normalized,
                normalization_scale,
                observed_mask,
                device,
            )
        )

        type_sensitivity = landmark_sensitivities.sum(axis=0)
        total_sensitivity = float(type_sensitivity.sum())
        if total_sensitivity > 0.0:
            type_share = type_sensitivity / total_sensitivity * 100.0
        else:
            type_share = np.zeros_like(type_sensitivity)

        type_figure = go.Figure(
            go.Bar(
                x=list(LANDMARK_NAMES),
                y=type_share,
                marker_color=[
                    LANDMARK_COLORS[name] for name in LANDMARK_NAMES
                ],
                customdata=type_sensitivity,
                hovertemplate=(
                    "%{x}<br>Sensitivity share: %{y:.2f}%"
                    "<br>Gradient norm sum: %{customdata:.4f}<extra></extra>"
                ),
            )
        )
        type_figure.update_layout(
            title=f"FDI {attribution_fdi}: sensitivity by landmark type",
            xaxis_title="Observed landmark type",
            yaxis_title="Local sensitivity share (%)",
            yaxis_range=[0, max(100.0, float(type_share.max()) * 1.1)],
            margin=dict(l=20, r=20, b=20, t=60),
            height=380,
        )

        observed_indices = np.flatnonzero(observed_mask)
        heatmap_values = landmark_sensitivities[observed_indices].T
        heatmap_figure = go.Figure(
            go.Heatmap(
                z=heatmap_values,
                x=[str(current_fdi[idx]) for idx in observed_indices],
                y=list(LANDMARK_NAMES),
                colorscale="YlOrRd",
                colorbar=dict(title="Gradient norm"),
                hovertemplate=(
                    "Source FDI %{x}<br>%{y}<br>"
                    "Gradient norm: %{z:.4f}<extra></extra>"
                ),
            )
        )
        heatmap_figure.update_layout(
            title="Sensitivity by source tooth and landmark",
            xaxis_title="Observed source tooth (FDI)",
            yaxis_title="Landmark type",
            margin=dict(l=20, r=20, b=20, t=60),
            height=380,
        )

        sensitivity_summary, sensitivity_type_plot, sensitivity_heatmap = (
            st.columns([1, 2, 3])
        )
        with sensitivity_summary:
            st.metric(
                "Gradient target error", f"{attributed_error_mm:.3f} mm"
            )
            if total_sensitivity > 0.0:
                largest_type_idx = int(np.argmax(type_share))
                st.metric(
                    "Most sensitive input type",
                    LANDMARK_NAMES[largest_type_idx],
                    f"{type_share[largest_type_idx]:.1f}% share",
                    delta_color="off",
                )
                source_tooth_idx, source_landmark_idx = np.unravel_index(
                    np.argmax(landmark_sensitivities),
                    landmark_sensitivities.shape,
                )
                st.write(
                    "Strongest individual input: "
                    f"FDI {current_fdi[source_tooth_idx]} "
                    f"{LANDMARK_NAMES[source_landmark_idx]}"
                )
            else:
                st.info("The local gradient is zero for this prediction.")
        with sensitivity_type_plot:
            st.plotly_chart(type_figure, use_container_width=True)
        with sensitivity_heatmap:
            st.plotly_chart(heatmap_figure, use_container_width=True)
