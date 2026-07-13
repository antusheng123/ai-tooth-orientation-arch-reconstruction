import numpy as np
from scipy.interpolate import make_interp_spline


def generate_bspline_curve(anchor_points, num_eval_points=200):
    """
    Fit a parametric B-spline through ordered 3D anchor points.

    Chord-length parameterization avoids assuming that the dental arch can be
    represented as y=f(x), and therefore supports arbitrary scan orientation.
    """
    anchor_points = np.asarray(anchor_points, dtype=np.float32)
    if anchor_points.ndim != 2 or anchor_points.shape[1] != 3:
        raise ValueError(
            f"Expected anchor points with shape (N, 3), got {anchor_points.shape}"
        )

    finite_mask = np.isfinite(anchor_points).all(axis=1)
    anchor_points = anchor_points[finite_mask]
    if len(anchor_points) < 2:
        return anchor_points

    segment_lengths = np.linalg.norm(np.diff(anchor_points, axis=0), axis=1)
    keep_mask = np.concatenate([[True], segment_lengths > 1e-6])
    anchor_points = anchor_points[keep_mask]
    if len(anchor_points) < 2:
        return anchor_points

    chord_lengths = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(anchor_points, axis=0), axis=1))]
    )
    parameters = chord_lengths / chord_lengths[-1]
    degree = min(3, len(anchor_points) - 1)
    spline = make_interp_spline(parameters, anchor_points, k=degree, axis=0)
    evaluation_parameters = np.linspace(0.0, 1.0, num_eval_points)
    return spline(evaluation_parameters).astype(np.float32)
