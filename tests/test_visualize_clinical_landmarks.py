import numpy as np

from phase2_arch_curve_reconstruction.visualize_clinical_landmarks import (
    build_combined_prediction_figure,
    build_key_summary,
    build_landmark_mesh_figure,
    parse_obj_mesh_lines,
)


def test_parse_obj_mesh_lines_reads_vertices_and_faces():
    vertices, faces = parse_obj_mesh_lines(
        [
            "v 0 0 0 1 1 1\n",
            "v 1 0 0 1 1 1\n",
            "v 0 1 0 1 1 1\n",
            "f 1 2 3\n",
        ]
    )

    np.testing.assert_allclose(vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(faces, [[0, 1, 2]])


def test_build_key_summary_counts_landmarks_and_centroid():
    groups = {
        1: {
            "Mesial": [np.array([0, 0, 0], dtype=np.float32)],
            "Distal": [np.array([2, 0, 0], dtype=np.float32)],
        }
    }

    rows = build_key_summary(groups)

    assert rows == [
        {
            "key": 1,
            "landmark_count": 2,
            "classes": "Distal:1, Mesial:1",
            "centroid_x": 1.0,
            "centroid_y": 0.0,
            "centroid_z": 0.0,
        }
    ]


def test_build_landmark_mesh_figure_labels_each_key():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    groups = {
        1: {
            "Mesial": [np.array([0, 0, 0], dtype=np.float32)],
            "Distal": [np.array([1, 0, 0], dtype=np.float32)],
        },
        2: {
            "Mesial": [np.array([0, 1, 0], dtype=np.float32)],
            "Distal": [np.array([1, 1, 0], dtype=np.float32)],
        },
    }

    figure = build_landmark_mesh_figure(vertices, faces, groups, title="lower")

    names = {trace.name for trace in figure.data}
    assert "OBJ mesh" in names
    assert "key 1 landmarks" in names
    assert "key 2 landmarks" in names
    assert "key labels" in names


def test_build_combined_prediction_figure_overlays_mesh_keys_predictions_and_curve():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    groups = {
        1: {
            "Mesial": [np.array([0, 0, 0], dtype=np.float32)],
            "Distal": [np.array([1, 0, 0], dtype=np.float32)],
        }
    }
    fdi_order = [48, 47]
    final_landmarks = np.asarray(
        [
            [[0, 0, 0], [1, 0, 0]],
            [[2, 0, 0], [3, 0, 0]],
        ],
        dtype=np.float32,
    )
    observed_mask = np.asarray([True, False])
    curve_points = np.asarray([[0, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=np.float32)

    figure = build_combined_prediction_figure(
        vertices=vertices,
        faces=faces,
        groups=groups,
        fdi_order=fdi_order,
        observed_mask=observed_mask,
        final_landmarks=final_landmarks,
        curve_points=curve_points,
        title="R05 lower combined",
    )

    names = {trace.name for trace in figure.data}
    assert "OBJ mesh" in names
    assert "key 1 landmarks" in names
    assert "Observed model landmarks" in names
    assert "Predicted model landmarks" in names
    assert "Predicted arch curve" in names
