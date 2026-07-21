from pathlib import Path

import numpy as np

from phase2_arch_curve_reconstruction.clinical_p05_inference import (
    LOWER_FDI,
    _select_two_cusps,
    build_visualization_figure,
    build_landmark_sequence,
    get_builtin_fdi_map,
    load_clinical_landmark_groups,
    parse_fdi_map,
)

SAMPLE_ROOT = Path("clinical_samples") / "P05_R05" / "samples"


def test_parse_fdi_map_accepts_key_value_pairs():
    assert parse_fdi_map("1=48,2=47,10=38") == {1: 48, 2: 47, 10: 38}


def test_get_builtin_fdi_map_returns_verified_p05_lower_mapping():
    assert get_builtin_fdi_map(
        jaw_name="lower",
        case_name="P05",
        landmarks_path=SAMPLE_ROOT / "P05" / "P05_lower_landmarks.json",
    ) == {
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


def test_get_builtin_fdi_map_returns_verified_p05_upper_mapping():
    assert get_builtin_fdi_map(
        jaw_name="upper",
        case_name="P05",
        landmarks_path=SAMPLE_ROOT / "P05" / "P05_upper_landmarks.json",
    ) == {
        1: 21,
        2: 13,
        3: 22,
        6: 23,
        7: 24,
        4: 26,
        5: 27,
    }


def test_get_builtin_fdi_map_returns_verified_r05_lower_mapping():
    assert get_builtin_fdi_map(
        jaw_name="lower",
        case_name="R05",
        landmarks_path=SAMPLE_ROOT / "R05" / "R05_lower_landmarks.json",
    ) == {
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


def test_get_builtin_fdi_map_returns_verified_r05_upper_mapping():
    assert get_builtin_fdi_map(
        jaw_name="upper",
        case_name="R05",
        landmarks_path=SAMPLE_ROOT / "R05" / "R05_upper_landmarks.json",
    ) == {
        1: 21,
        5: 26,
        3: 11,
        4: 14,
        2: 16,
    }


def test_load_clinical_landmark_groups_accepts_p05_aliases():
    groups = load_clinical_landmark_groups(SAMPLE_ROOT / "P05" / "P05_lower_landmarks.json")

    assert "InnerPoint" in groups[1]
    assert "OuterPoint" in groups[1]
    assert "Cusp" in groups[2]


def test_select_two_cusps_uses_two_nearest_to_outerpoint():
    class_points = {
        "OuterPoint": [np.array([0, 0, 0], dtype=np.float32)],
        "Mesial": [np.array([20, 0, 0], dtype=np.float32)],
        "Distal": [np.array([20, 1, 0], dtype=np.float32)],
        "Cusp": [
            np.array([20, 0, 0], dtype=np.float32),
            np.array([1, 0, 0], dtype=np.float32),
            np.array([2, 0, 0], dtype=np.float32),
        ],
    }

    cusps = _select_two_cusps(class_points)

    np.testing.assert_allclose(cusps, [[1, 0, 0], [2, 0, 0]])


def test_build_pipeline2_sequence_from_clinical_landmarks():
    groups = {
        1: {
            "Mesial": [np.array([0, 0, 0], dtype=np.float32)],
            "Distal": [np.array([2, 0, 0], dtype=np.float32)],
            "InnerPoint": [np.array([1, 2, 0], dtype=np.float32)],
        },
        2: {
            "Mesial": [np.array([10, 0, 0], dtype=np.float32)],
            "Distal": [np.array([12, 0, 0], dtype=np.float32)],
            "InnerPoint": [np.array([11, 2, 0], dtype=np.float32)],
        },
    }
    landmarks, tooth_mask, landmark_mask = build_landmark_sequence(
        groups,
        fdi_by_key={1: 43, 2: 42},
        fdi_order=LOWER_FDI,
        pipeline="pipeline2",
    )

    idx_43 = LOWER_FDI.index(43)
    idx_42 = LOWER_FDI.index(42)
    assert tooth_mask.sum() == 2
    assert landmark_mask[idx_43].all()
    assert landmark_mask[idx_42].all()
    np.testing.assert_allclose(landmarks[idx_43, 0], [0.5, 1.0, 0.0])
    np.testing.assert_allclose(landmarks[idx_43, 1], [1.5, 1.0, 0.0])


def test_build_visualization_figure_contains_curve_observed_and_predicted_traces():
    landmarks = np.zeros((16, 2, 3), dtype=np.float32)
    landmarks[0] = [[0, 0, 0], [1, 0, 0]]
    landmarks[1] = [[2, 0, 0], [3, 0, 0]]
    observed_mask = np.zeros(16, dtype=bool)
    observed_mask[0] = True
    curve_points = np.asarray([[0, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=np.float32)

    figure = build_visualization_figure(
        fdi_order=LOWER_FDI,
        observed_mask=observed_mask,
        final_landmarks=landmarks,
        curve_points=curve_points,
        title="P05 lower pipeline1",
    )

    names = {trace.name for trace in figure.data}
    assert "Predicted arch curve" in names
    assert "Observed landmarks" in names
    assert "Predicted landmarks" in names
