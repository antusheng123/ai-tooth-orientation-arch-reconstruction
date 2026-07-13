import numpy as np
from scipy.interpolate import make_interp_spline


def generate_bspline_curve(centers, num_eval_points=200):
    """
    Fit a parametric B-spline through ordered tooth centers.

    Chord-length parameterization avoids assuming that the dental arch can be
    represented as y=f(x), and therefore supports arbitrary scan orientation.
    """
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"Expected centers with shape (N, 3), got {centers.shape}")

    finite_mask = np.isfinite(centers).all(axis=1)
    centers = centers[finite_mask]
    if len(centers) < 2:
        return centers

    segment_lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    keep_mask = np.concatenate([[True], segment_lengths > 1e-6])
    centers = centers[keep_mask]
    if len(centers) < 2:
        return centers

    chord_lengths = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))]
    )
    parameters = chord_lengths / chord_lengths[-1]
    degree = min(3, len(centers) - 1)
    spline = make_interp_spline(parameters, centers, k=degree, axis=0)
    evaluation_parameters = np.linspace(0.0, 1.0, num_eval_points)
    return spline(evaluation_parameters).astype(np.float32)
