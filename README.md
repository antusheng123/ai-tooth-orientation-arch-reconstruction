# AI-Based Tooth Orientation Prediction and Dental Arch Curve Reconstruction

This repository contains research code and trained weights for a two-phase digital dentistry project developed around automated CAD support for removable partial dentures and dental arch reconstruction.

## Overview

The project has two main components:

- `phase1_single_tooth_orientation/`: single-tooth landmark and orientation prediction from 6D point clouds.
- `phase2_arch_curve_reconstruction/`: dental arch landmark completion and 3D B-Spline curve reconstruction under missing-tooth scenarios.

The implementation is organized as a reproducible research repository. Trained weights are included because they are small enough for normal Git storage.

## Repository Structure

```text
phase1_single_tooth_orientation/
  data_preparation/      Scripts for segmenting full-arch scans into single-tooth samples
  results/               Phase 1 result figure
  weights/               Trained Phase 1 model weights
  dataset.py             PyTorch dataset for 6D point-cloud samples
  point_transformer_model.py
  pre_sample_pc.py       Mesh-to-6D-point-cloud preprocessing
  train_point_transformer.py
  visualizer.py          Streamlit + Plotly inference and evaluation UI

phase2_arch_curve_reconstruction/
  pipeline1_mandibular/  Final mandibular-oriented pipeline
  pipeline2_maxillary/   Final maxillary-oriented pipeline
  legacy/                Earlier curve reconstruction methods

docs/
  final_report.pdf
  phase1_summary.md
  phase2_summary.md
  reports_archive/

assets/
  screenshots/
  demo_outputs/
```

## Datasets

Dataset archives are not committed to Git. Download or place them outside the repository, then update the hard-coded paths or refactor them into command-line arguments before running experiments.

Required raw datasets:

```text
segmentation_data_for_single_teeth/train/
3DTeethLand_landmarks_train/
```

Optional preprocessed dataset for directly reproducing Phase 1 training:

```text
Single_Teeth_PC_6D/
```

The current scripts were originally run with paths under:

```text
F:\NDCS_3DS_data\
```

## Weights

Included weights:

```text
phase1_single_tooth_orientation/weights/best_single_tooth_model_6d.pt
phase2_arch_curve_reconstruction/pipeline1_mandibular/weights/best_landmark_model.pth
phase2_arch_curve_reconstruction/pipeline2_maxillary/weights/best_landmark_model.pth
phase2_arch_curve_reconstruction/legacy/center_point_curve/best_curve_model.pth
phase2_arch_curve_reconstruction/legacy/four_landmark_curve/best_landmark_model.pth
phase2_arch_curve_reconstruction/legacy/pipeline1_original/best_landmark_model.pth
phase2_arch_curve_reconstruction/legacy/pipeline1_posterior_old/best_landmark_model.pth
```

## Results

Phase 1 predicts Mesial, Distal, Inner/Lingual, Outer/Buccal, and Facial landmarks for individual teeth. The final report records a Mean Euclidean Error of 1.2426 mm and Success Rate@2mm of 87.55%.

Phase 2 reconstructs dental arch curves from a fixed 16-token FDI sequence using masked Transformer models and 3D B-Spline interpolation. The final report records 1.093 mm MEE for the maxillary trajectory and 1.272 mm MEE for the mandibular trajectory.

## Notes

- `pipeline1_mandibular/` and `pipeline2_maxillary/` are the final Phase 2 pipelines.
- `legacy/pipeline1_posterior_old/` is retained for reproducibility, but its posterior mirroring implementation is not the final method.
- Original raw datasets and generated large archives should be distributed externally.
