# Project Context

Last updated: 2026-06-20

Read this file first in future sessions. `Second pipeline.md` describes the
requested Pipeline 2 migration; the source code now follows that pipeline.

## Purpose

This project predicts two 3D Pipeline 2 points for simulated consecutive
missing teeth in one upper or lower dental arch. A Streamlit app masks a
consecutive run of valid teeth, runs a Transformer completion model, and
visualizes observed points, predicted points, tooth midpoints, and a fixed
reference dental-arch B-spline.

## Files

- `app.py`: Streamlit inference, consecutive masking, optional metrics, and
  Plotly visualization.
- `dataset.py`: FDI ordering, anterior/posterior classification, two-point
  derivation/extraction, normalization, augmentation, data loading, and
  `DentalArchDataset`.
- `model.py`: `MaskedArchRegressor`, a Transformer encoder.
- `train.py`: patient-level split, consecutive masking, losses, validation,
  and training.
- `postprocess.py`: parametric 3D B-spline generation from ordered anchor
  points.
- `best_landmark_model.pth`: model weights path. Existing checkpoints trained
  on another point definition are incompatible with Pipeline 2 and must be
  retrained.
- `requirements.txt`: NumPy, SciPy, PyTorch, Trimesh, Plotly, Streamlit.

## Core Shapes And Ordering

- One arch has `SEQ_LEN = 16` tooth positions.
- Upper FDI order:
  `18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28`.
- Lower FDI order:
  `48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38`.
- Pipeline 2 tensor shape per arch: `(16, 2, 3)`.
- Point names are generic: `Point0`, `Point1`.
- Flattened coordinate dimension per tooth: `6`.
- Model feature dimension per tooth: `8`:
  6 coordinates + 1 missing flag + 1 jaw flag.
- Model output shape: `(batch, 16, 2, 3)`.
- Jaw flag: upper `0`, lower `1`.

## Two-Point Derivation

`extract_arch_landmarks()` loads:

- an OBJ mesh,
- per-vertex tooth labels from JSON,
- landmark objects from keypoint JSON.

Each source landmark coordinate is assigned to the nearest mesh vertex with a
KDTree, then mapped to that vertex's FDI tooth label.

Tooth classes:

- Anterior teeth:
  upper `13,12,11,21,22,23`; lower `43,42,41,31,32,33`.
- Posterior teeth:
  upper `18,17,16,15,14,24,25,26,27,28`;
  lower `48,47,46,45,44,34,35,36,37,38`.

Point construction:

- Anterior teeth:
  - `Point0` = `(InnerPoint + Mesial) / 2`
  - `Point1` = `(InnerPoint + Distal) / 2`
- Posterior teeth:
  - `Point0` = mean `Mesial`
  - `Point1` = mean `Distal`

A tooth is valid only if the required two points can be constructed. Invalid
or incomplete teeth remain zero.

Important masks:

- `landmark_valid_mask`: shape `(16, 2)`.
- `tooth_valid_mask`: shape `(16,)`, true only for complete Pipeline 2 teeth.
- `dropped_mask`: valid teeth intentionally hidden for prediction.
- Model missing flag: `~tooth_valid_mask | dropped_mask`.

## Normalization

An entire arch uses one translation and one isotropic scale:

- origin = mean of all valid point coordinates,
- scale = RMS distance of valid coordinates from the origin.

Only valid points are normalized. Missing entries stay exactly zero.
Predictions are converted back to world coordinates using the same origin and
scale. Coordinates and displayed errors are assumed to be in millimeters.

## Model

`MaskedArchRegressor` uses:

- linear projection of two-point coordinates plus missing flag,
- learned 16-position embedding,
- learned upper/lower jaw embedding,
- 6 Transformer encoder layers by default,
- embedding size 128, 4 heads, feed-forward size 512,
- output projection to 6 coordinates per tooth.

The model predicts all positions, but training loss and UI evaluation select
only intentionally dropped teeth.

## Training

Data is split by patient so upper and lower arches from the same patient stay
in the same subset. Training only loads arches with at least 7 valid Pipeline
2 teeth. Training samples randomly mask a consecutive window over currently
valid teeth:

- dropped count `K` is sampled from 3 to 6,
- at least 4 valid teeth remain observed,
- if an arch cannot satisfy this constraint, no teeth are dropped for that
  sample and it contributes no masked loss.

Loss on dropped teeth:

- point Smooth L1, weight `1.0`,
- two-point midpoint Smooth L1, weight `0.25`,
- intra-tooth `Point0`-`Point1` distance Smooth L1, weight `0.10`.

Optimizer is AdamW with learning rate `1e-4` and weight decay `1e-4`.
Training is configured for 1500 epochs; validation runs every 5 epochs.
The best validation checkpoint is saved as `best_landmark_model.pth`.

## Streamlit Flow

1. Find complete OBJ/label/keypoint file sets under the configured data roots.
2. Load one patient arch and its ground-truth Pipeline 2 points.
3. Before any simulated masking, collect all valid ground-truth points in FDI
   order as `(2N, 3)` anchors.
4. Build the fixed reference B-spline from those anchors.
5. Randomly mask a consecutive run of valid FDI positions, retaining at least
   4 observed teeth.
6. Zero masked inputs, set missing flags, and run model inference.
7. Replace masked positions only in a display copy with model predictions.
8. Show observed and predicted points/midpoints and the fixed curve.
9. Evaluate masked predictions against their hidden ground truth.

## Fixed-Curve Business Rule

This rule is intentional and must not be changed:

- The reference B-spline is established from all initially available
  ground-truth Pipeline 2 points before Streamlit masking.
- Selecting or randomizing simulated missing teeth must not alter the curve.
- Predicted points or midpoints must never become B-spline anchors.
- The curve must not be recomputed from only the currently visible teeth.

In `app.py`, this is implemented before inference masking affects display:

```python
curve_anchor_points = collect_curve_anchor_points(
    raw_landmarks, tooth_valid_mask, current_fdi
)
curve_points = generate_bspline_curve(curve_anchor_points, num_eval_points=500)
```

## UI Evaluation Semantics

For every intentionally masked tooth:

- midpoint error = Euclidean distance between predicted and ground-truth
  two-point midpoints,
- mean point error = mean of the two 3D Euclidean point errors.

The UI reports both in millimeters. Red dotted lines connect each predicted
midpoint to its ground-truth midpoint. Do not report prediction-to-curve
distance unless a future request explicitly reintroduces it.

## B-Spline

`generate_bspline_curve()` accepts ordered anchor points with shape `(N, 3)`.
It removes non-finite and consecutive duplicate anchors, uses chord-length
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
- `python -m py_compile` may fail in this environment if Windows cannot
  replace an existing `__pycache__` file; that is not necessarily a syntax
  failure.
- No dedicated automated test suite currently exists.
