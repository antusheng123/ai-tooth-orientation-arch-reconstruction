# P05/R05 Clinical Samples

This folder collects the standalone P05 and R05 clinical samples used to test
the Phase 2 arch-completion pipelines.

## Contents

```text
samples/
  P05/
    P05_lower.obj
    P05_lower_landmarks.json
    P05_upper.obj
    P05_upper_landmarks.json
  R05/
    R05_lower.obj
    R05_lower_landmarks.json
    R05_upper.obj
    R05_upper_landmarks.json

landmark_selection_summary.md
```

Generated demo outputs are stored separately under:

```text
assets/demo_outputs/P05_R05/
```

That output folder includes the lightweight CSV/JSON prediction artifacts and
the combined OBJ + clinical landmark + model prediction HTML visualizations:

```text
P05/P05_lower_combined_pipeline1_visualization.html
P05/P05_upper_combined_pipeline2_visualization.html
R05/R05_lower_combined_pipeline1_visualization.html
R05/R05_upper_combined_pipeline2_visualization.html
```

## Clinical Adapter

The adapter script is:

```text
phase2_arch_curve_reconstruction/clinical_p05_inference.py
```

It converts P05/R05 clinical landmark JSON files into the 16-token Phase 2 model
input. These standalone samples do not include the original training-dataset
per-vertex tooth label JSON files, so the adapter uses verified clinical key to
FDI mappings documented in `landmark_selection_summary.md`.

## Reproduce Outputs

From the repository root, run the lower-jaw P05 mandibular pipeline:

```powershell
python phase2_arch_curve_reconstruction/clinical_p05_inference.py `
  --case-name P05 `
  --jaw lower `
  --pipeline pipeline1 `
  --landmarks clinical_samples/P05_R05/samples/P05/P05_lower_landmarks.json `
  --output-dir outputs/P05
```

Run the upper-jaw P05 maxillary pipeline:

```powershell
python phase2_arch_curve_reconstruction/clinical_p05_inference.py `
  --case-name P05 `
  --jaw upper `
  --pipeline pipeline2 `
  --landmarks clinical_samples/P05_R05/samples/P05/P05_upper_landmarks.json `
  --output-dir outputs/P05
```

Use `R05` and the corresponding `clinical_samples/P05_R05/samples/R05/...`
landmark paths to reproduce the R05 outputs.

To overlay the OBJ mesh, clinical keys, predicted model landmarks, and predicted
arch curve in one HTML file, run:

```powershell
python phase2_arch_curve_reconstruction/combine_clinical_prediction_visualization.py `
  --obj clinical_samples/P05_R05/samples/P05/P05_upper.obj `
  --landmarks clinical_samples/P05_R05/samples/P05/P05_upper_landmarks.json `
  --result-json outputs/P05/P05_upper_pipeline2_result.json `
  --jaw upper `
  --title "P05 upper combined pipeline2" `
  --output-html outputs/P05/P05_upper_combined_pipeline2_visualization.html
```
