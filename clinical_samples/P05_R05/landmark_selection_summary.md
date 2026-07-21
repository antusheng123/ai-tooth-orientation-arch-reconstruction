# R05/P05 Landmark Selection Summary

## Context

This repository has two Phase 2 arch-completion pipelines:

- `pipeline1_mandibular`: each tooth is represented by two points. Anterior teeth use mesial/distal points; posterior teeth use buccal cusp points.
- `pipeline2_maxillary`: each tooth is represented by two points. Posterior teeth use mesial/distal points; anterior teeth use two derived inner-proximal midpoints.

For the standalone clinical samples `R05` and `P05`, the code path is:

```text
phase2_arch_curve_reconstruction/clinical_p05_inference.py
```

These samples contain OBJ meshes and clinical landmark JSON files, but they do not use the original training-dataset per-vertex tooth label JSON. Therefore, the clinical adapter does not assign landmarks to teeth through mesh labels/KDTree. Instead, it uses verified clinical `key` to FDI mappings and then converts each clinical tooth group into the same 16-tooth tensor format expected by the Phase 2 models.

## FDI Sequence

The model uses a fixed 16-position FDI order.

Upper:

```text
[18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
```

Lower:

```text
[48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
```

Anterior teeth:

```text
13, 12, 11, 21, 22, 23, 43, 42, 41, 31, 32, 33
```

All other teeth in the 16-position jaw sequence are treated as posterior teeth.

## Clinical Key To FDI Mapping

The built-in verified mappings are defined in `clinical_p05_inference.py`.

### P05 Lower

```text
1 -> 41
2 -> 45
7 -> 42
9 -> 43
10 -> 44
5 -> 31
8 -> 32
3 -> 33
6 -> 34
4 -> 36
```

### P05 Upper

```text
1 -> 21
2 -> 13
3 -> 22
6 -> 23
7 -> 24
4 -> 26
5 -> 27
```

### R05 Lower

```text
1 -> 41
5 -> 31
7 -> 42
12 -> 43
2 -> 45
9 -> 46
10 -> 47
13 -> 48
11 -> 32
3 -> 33
6 -> 34
4 -> 36
8 -> 37
```

### R05 Upper

```text
1 -> 21
5 -> 26
3 -> 11
4 -> 14
2 -> 16
```

## JSON Grouping Rule

The clinical JSON loader reads each item in `objects` and groups points by clinical tooth key and landmark class.

The key is parsed from the part before the first underscore:

```text
"7_Mesial_0" -> key 7
```

Class aliases are normalized:

```text
Inner  -> InnerPoint
Outer  -> OuterPoint
Cusps  -> Cusp
```

Only finite 3D coordinates with shape `(3,)` are kept.

## Point Selection Rule Per Tooth

Each valid tooth must produce exactly two 3D points. If a tooth cannot produce both points, it is treated as missing/invalid in the observed input and its entry is zeroed.

### Pipeline 1

For anterior teeth:

```text
Point0 = mean(Mesial)
Point1 = mean(Distal)
```

For posterior teeth:

```text
Point0, Point1 = selected Cusp points
```

Clinical R05/P05 adapter cusp selection:

- If there are at least two `Cusp` points and an `OuterPoint`, select the two cusp points with smallest Euclidean distance to the mean `OuterPoint`.
- If there are at least two `Cusp` points but no `OuterPoint`, use the first two cusp points.
- If there is exactly one `Cusp`, and both `Mesial` and `Distal` exist, synthesize two points around the cusp:

```text
offset = (Distal - Mesial) * 0.25
Point0 = Cusp - offset
Point1 = Cusp + offset
```

- If no valid two-point result can be built, the tooth is excluded from the observed mask.

### Pipeline 2

For anterior teeth:

```text
Point0 = (mean(InnerPoint) + mean(Mesial)) / 2
Point1 = (mean(InnerPoint) + mean(Distal)) / 2
```

For posterior teeth:

```text
Point0 = mean(Mesial)
Point1 = mean(Distal)
```

If the required source classes are missing, the tooth is excluded from the observed mask.

## Actual Valid Teeth For R05/P05

These were computed by running the existing `build_landmark_sequence()` logic on the four clinical JSON files.

### P05 Lower

Both pipelines produce 10 observed teeth:

```text
45, 44, 43, 42, 41, 31, 32, 33, 34, 36
```

### P05 Upper

Both pipelines produce 7 observed teeth:

```text
13, 21, 22, 23, 24, 26, 27
```

### R05 Lower

Pipeline 1 produces 12 observed teeth:

```text
48, 47, 45, 43, 42, 41, 31, 32, 33, 34, 36, 37
```

Pipeline 2 produces 13 observed teeth:

```text
48, 47, 46, 45, 43, 42, 41, 31, 32, 33, 34, 36, 37
```

The difference is FDI `46`: it has enough `Mesial`/`Distal` information for Pipeline 2, but it lacks usable `Cusp` information for Pipeline 1 posterior-tooth selection.

### R05 Upper

Both pipelines produce 5 observed teeth:

```text
16, 14, 11, 21, 26
```

## Important Distinction From Training Dataset Logic

The original training dataset loaders use OBJ mesh vertices plus a per-vertex label JSON. For each landmark coordinate, they build a KDTree over mesh vertices, find the nearest vertex, and use that vertex's label as the FDI tooth ID.

That training-data logic appears in:

```text
phase2_arch_curve_reconstruction/pipeline1_mandibular/dataset.py
phase2_arch_curve_reconstruction/pipeline2_maxillary/dataset.py
```

For `R05` and `P05`, the active clinical adapter does not use KDTree tooth assignment because the standalone clinical folders contain only:

```text
R05_upper.obj
R05_upper_landmarks.json
R05_lower.obj
R05_lower_landmarks.json
P05_upper.obj
P05_upper_landmarks.json
P05_lower.obj
P05_lower_landmarks.json
```

There are no corresponding per-vertex tooth label JSON files in the sample folders. The clinical adapter therefore relies on the built-in key-to-FDI mappings listed above.

## Source Files Checked

Main clinical adapter:

```text
phase2_arch_curve_reconstruction/clinical_p05_inference.py
```

Tests that confirm the mappings and selection behavior:

```text
tests/test_clinical_p05_inference.py
```

Original training-dataset extraction logic:

```text
phase2_arch_curve_reconstruction/pipeline1_mandibular/dataset.py
phase2_arch_curve_reconstruction/pipeline2_maxillary/dataset.py
```

