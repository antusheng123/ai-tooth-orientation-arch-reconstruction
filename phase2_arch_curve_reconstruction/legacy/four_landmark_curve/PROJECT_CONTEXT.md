# Project Context

Last updated: 2026-06-12

Read this file first in future sessions. Then open only the source files relevant
to the requested change.

## Purpose

This project predicts four 3D landmarks for simulated missing teeth in one
upper or lower dental arch. A Streamlit app masks selected known teeth, runs a
Transformer completion model, and visualizes observed landmarks, predicted
landmarks, tooth centers, and a reference dental-arch B-spline.

## Files

- `app.py`: Streamlit inference, masking, metrics, and Plotly visualization.
- `dataset.py`: FDI ordering, landmark extraction, normalization, augmentation,
  data loading, and `DentalArchDataset`.
- `model.py`: `MaskedArchRegressor`, a Transformer encoder.
- `train.py`: patient-level split, masking, losses, validation, and training.
- `postprocess.py`: parametric 3D B-spline generation.
- `best_landmark_model.pth`: current trained model weights.
- `requirements.txt`: NumPy, SciPy, PyTorch, Trimesh, Plotly, Streamlit.

## Core Shapes And Ordering

- One arch has `SEQ_LEN = 16` tooth positions.
- Upper FDI order:
  `18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28`.
- Lower FDI order:
  `48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38`.
- Each tooth has four ordered landmarks:
  `Mesial`, `Distal`, `OuterPoint`, `InnerPoint`.
- Landmark tensor shape: `(16, 4, 3)`.
- One tooth center is the mean of its four landmarks.
- Flattened coordinate dimension per tooth: `12`.
- Model feature dimension per tooth: `14`:
  12 coordinates + 1 missing flag + 1 jaw flag.
- Model output shape: `(batch, 16, 4, 3)`.
- Jaw flag: upper `0`, lower `1`.

## Data Extraction

`extract_arch_landmarks()` loads:

- an OBJ mesh,
- per-vertex tooth labels from JSON,
- landmark objects from keypoint JSON.

Each landmark coordinate is assigned to the nearest mesh vertex with a KDTree,
then mapped to that vertex's FDI tooth label. Repeated annotations of the same
class are averaged. A tooth is valid only if all four landmark classes exist.
Invalid or incomplete teeth remain zero.

Important masks:

- `landmark_valid_mask`: shape `(16, 4)`.
- `tooth_valid_mask`: shape `(16,)`, true only for complete teeth.
- `dropped_mask`: valid teeth intentionally hidden for prediction.
- Model missing flag: `~tooth_valid_mask | dropped_mask`.

## Normalization

An entire arch uses one translation and one isotropic scale:

- origin = mean of all valid landmark coordinates,
- scale = RMS distance of valid coordinates from the origin.

Only valid landmarks are normalized. Missing entries stay exactly zero.
Predictions are converted back to world coordinates using the same origin and
scale. Coordinates and displayed errors are assumed to be in millimeters.

## Model

`MaskedArchRegressor` uses:

- linear projection of coordinates plus missing flag,
- learned 16-position embedding,
- learned upper/lower jaw embedding,
- 6 Transformer encoder layers by default,
- embedding size 128, 4 heads, feed-forward size 512,
- output projection to 12 coordinates per tooth.

The model predicts all positions, but training loss and UI evaluation select
only intentionally dropped teeth.

## Training

Data is split by patient so upper and lower arches from the same patient stay
in the same subset. Training samples randomly drop 1 to at most 5 valid teeth,
while retaining at least 4 known teeth.

Loss on dropped teeth:

- point Smooth L1, weight `1.0`,
- tooth-center Smooth L1, weight `0.25`,
- six pairwise intra-tooth distance Smooth L1, weight `0.10`.

Optimizer is AdamW with learning rate `1e-4` and weight decay `1e-4`.
Training is configured for 1500 epochs; validation runs every 5 epochs.
The best validation checkpoint is saved as `best_landmark_model.pth`.

## Streamlit Flow

1. Find complete OBJ/label/keypoint file sets under the configured data roots.
2. Load one patient arch and its ground-truth landmarks.
3. Before any simulated masking, compute every valid ground-truth tooth center.
4. Build the fixed reference B-spline from all valid ground-truth centers.
5. Let the user mask valid FDI teeth, retaining at least 4 observed teeth.
6. Zero masked inputs, set missing flags, and run model inference.
7. Replace masked positions only in a display copy with model predictions.
8. Show observed and predicted landmarks/centers and the fixed curve.
9. Evaluate masked predictions against their hidden ground truth.

## Fixed-Curve Business Rule

This rule is intentional and must not be changed:

- The reference B-spline is established from all initially available
  ground-truth tooth centers before Streamlit masking.
- Selecting or deselecting simulated missing teeth must not alter the curve.
- Predicted centers must never become B-spline anchors.
- The curve must not be recomputed from only the currently visible teeth.

In `app.py`, this is implemented before `teeth_to_drop`/`dropped_mask`:

```python
ground_truth_centers = landmarks_to_centers(raw_landmarks, tooth_valid_mask)
curve_points = generate_bspline_curve(
    ground_truth_centers[tooth_valid_mask], num_eval_points=500
)
```

## UI Evaluation Semantics

For every intentionally masked tooth:

- center error = Euclidean distance between predicted and ground-truth center,
- mean landmark error = mean of the four 3D Euclidean landmark errors.

The UI reports both in millimeters. Red dotted lines connect each predicted
center to its ground-truth center. Do not report prediction-to-curve distance
unless a future request explicitly reintroduces it.

## Landmark Attribution

When at least one tooth is intentionally masked, the UI first provides an
exact decomposition of the selected predicted tooth's center offset:

- each landmark error vector is predicted landmark minus ground truth,
- the center offset is the mean of the four landmark error vectors,
- each signed contribution is one quarter of that landmark error projected
  onto the final center-offset direction,
- the four signed contributions in millimeters sum exactly to the center
  error,
- positive contributions push along the final error direction and negative
  contributions partially cancel it.

An advanced expander also explains the selected prediction using local
gradient sensitivity:

- target scalar = selected predicted center error against ground truth in mm,
- input variables = normalized XYZ coordinates of observed landmarks,
- one landmark score = L2 norm of its three coordinate gradients,
- landmark-type share = that type's scores summed over observed teeth and
  normalized across the four landmark types,
- a heatmap retains the source-tooth and landmark-type breakdown.

This is a local model sensitivity measure, not a causal decomposition of the
error. Single-landmark zero ablation is intentionally avoided because the
model has only a tooth-level missing flag and was not trained on partially
missing teeth.

## B-Spline

`generate_bspline_curve()` accepts ordered centers with shape `(N, 3)`.
It removes non-finite and consecutive duplicate centers, uses chord-length
parameterization, and creates a parametric interpolation spline with degree
`min(3, N - 1)`. It supports arbitrary scan orientation and is not `y=f(x)`.

Because it is interpolating, it passes through its supplied anchors. Correct
anchor selection is therefore critical.

## Data Paths And Commands

Current hard-coded roots in `app.py` and `train.py`:

```text
F:\NDCS_3DS_data\segmentation_data_for_single_teeth\train
F:\NDCS_3DS_data\3DTeethLand_landmarks_train
```

Typical commands:

```powershell
streamlit run app.py
python train.py
```

## Verification Notes

- Source syntax can be checked without writing bytecode using `ast.parse`.
- `python -m py_compile` previously failed because Windows could not replace an
  existing `__pycache__` file; that was not a Python syntax failure.
- `git` was not available on PATH in the observed environment.
- No dedicated automated test suite currently exists.

## Future Reading Guide

- UI, masking, fixed curve, or metrics: read `app.py` and `postprocess.py`.
- Input shapes, FDI mapping, or normalization: read `dataset.py`.
- Architecture changes: read `model.py`, then matching feature code in
  `dataset.py`, `train.py`, and `app.py`.
- Loss, sampling, validation, or checkpoint changes: read `train.py` and
  `dataset.py`.
