# Task Description: Dental Landmark Completion & Curve Fitting (Pipeline 1)

## 1. Project Overview & Context
You are modifying an existing PyTorch + Streamlit project that predicts 3D dental landmarks for missing teeth. 
- The original project uses `SEQ_LEN = 16` teeth per arch, and 4 landmarks per tooth (`Mesial`, `Distal`, `OuterPoint`, `InnerPoint`), resulting in a `(16, 4, 3)` tensor shape.
- We are completely replacing the training and curve-fitting strategy. In Pipeline 1, each tooth position will now track exactly **2 specific points** instead of 4, changing the target tensor shape to `(16, 2, 3)`.

## 2. Core Modifications Required

### A. Consecutive Masking Strategy (In `train.py` & `app.py`)
- **Old Strategy:** Randomly drop 1 to 5 discrete valid teeth.
- **New Strategy:** Implement **Consecutive Masking**. For each training sample/inference step, randomly select a starting valid tooth position and mask $K$ **consecutive** valid teeth, where $K \ge 3$ (e.g., randomly choose $K$ between 3 and 6). 
- **Constraint:** Ensure that at least 4 valid teeth remain observed (unmasked) in the arch.

### B. Tooth Classification & 2-Point Feature Extraction (In `dataset.py`)
Classify teeth using FDI notation:
- **Anterior Teeth (前牙):** Upper `[13, 12, 11, 21, 22, 23]`, Lower `[43, 42, 41, 31, 32, 33]`.
- **Posterior Teeth (后牙):** Upper `[18, 17, 16, 15, 14, 24, 25, 26, 27, 28]`, Lower `[48, 47, 46, 45, 44, 34, 35, 36, 37, 38]`.

For the target `(16, 2, 3)` landmark tensor, extract the 2 points per tooth position as follows:
1. **For Anterior Teeth:** - Point 0: `Mesial` point (🔴)
   - Point 1: `Distal` point (🟢)
2. **For Posterior Teeth:** - Point 0 & Point 1: The **two Cusp points (🔵) located on the Buccal (outer/cheek) side**.
   - *Implementation Note:* You must update `extract_arch_landmarks()` to parse Cusp points from the keypoint JSON. If multiple cusps exist, select the two closest to the `OuterPoint` (or designated as buccal cusps in the dataset) to represent the cheek-side chewing peaks.

### C. Shape Changes & Model Adaptation (In `model.py`, `dataset.py`, `train.py`)
- The new landmark tensor shape is `(batch, 16, 2, 3)`.
- Flattened coordinate dimension per tooth becomes `6` (2 points * 3 coordinates).
- Model input feature dimension per tooth becomes `8` (6 coordinates + 1 missing flag + 1 jaw flag).
- Update `MaskedArchRegressor` linear projections to match input dimension 8 and output dimension 6 per tooth.
- Update Normalization/Denormalization to operate on the `(16, 2, 3)` structure.

### D. B-Spline Curve Anchors (In `postprocess.py` & `app.py`)
- **Old Strategy:** B-spline generated using 3D tooth centers.
- **New Strategy:** The B-spline curve must now be fitted using the individual extracted landmarks directly.
- For all valid, unmasked ground-truth teeth (ordered by FDI sequence), collect their 2 points sequentially. For $N$ valid teeth, this yields an ordered sequence of $2N$ coordinates with shape `(2N, 3)`. Feed this `(2N, 3)` tensor as anchors into `generate_bspline_curve()`.
- **Maintain Business Rule:** The reference curve must be established *before* Streamlit masking using initially available ground-truth landmarks and must remain fixed.

### E. Loss Function & UI Changes (In `train.py` & `app.py`)
- Compute Smooth L1 loss on the `(16, 2, 3)` points for the consecutively dropped teeth. Adjust intra-tooth distance loss to monitor the single distance between Point 0 and Point 1.
- Update the Streamlit UI to display the new points, evaluate metrics (mean landmark error) based on these 2 points per masked tooth, and visualize the new B-spline arch curve.

## 3. Execution Plan
1. Refactor `dataset.py` to parse buccal cusps and output `(16, 2, 3)` shapes.
2. Modify `model.py` dimensions to support the new 8-in/6-out feature setup.
3. Update `train.py` with consecutive masking logic ($K \ge 3$) and adjust the loss calculation. Run training.
4. Update `postprocess.py` and `app.py` to update the B-spline anchor stream and UI metrics visualization.