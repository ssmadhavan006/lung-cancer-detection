# Lung Cancer Detection

This project implements a reproducible lung CT slice classification pipeline for
the IQ-OTH/NCCD dataset, plus a Streamlit dashboard for interactive inference.
The current implementation is strongest on three-class classification:

- `benign`
- `malignant`
- `normal`

The dashboard has been validated in local use and correctly classifies uploaded
images across the three target categories.

## Current status

- Best trained model: `efficientnet_b0`
- Training mode: pretrained transfer learning
- Device used: CUDA GPU
- Best benchmark run: `outputs/baseline_pretrained_8ep/`
- Dashboard behavior: auto-loads the strongest available completed run under `outputs/`

## Dataset findings

The local dataset folder contains:

- `1097` JPG images across three class folders
- `1` text file with dataset-level notes
- No nested per-case folders
- No reliable patient or case identifiers in filenames
- No useful case mapping in image metadata

Observed local class counts:

| Class | Images |
| --- | ---: |
| Benign | 120 |
| Malignant | 561 |
| Normal | 416 |
| Total | 1097 |

Important note:

- The included text file says the original dataset contains `1190` images from `110` cases.
- The local exported JPG folder available for training currently contains `1097` images.
- Because the export is flat and case IDs are not recoverable, current evaluation is slice-level, not patient-level.

## What we built

- Dataset inspection and corruption checks
- Image enhancement with CLAHE, denoising, normalization, and contrast sharpening
- Training augmentation with rotation, flips, brightness/contrast, scaling, and elastic deformation
- Comparative benchmarking across `resnet50`, `efficientnet_b0`, and `densenet121`
- Grad-CAM explainability artifact generation
- Grad-CAM-derived weak localization, bounding boxes, and pixel measurements
- Benchmark reports in JSON, CSV, and PDF
- Streamlit dashboard for single-image inference, model metrics, and downloadable reports
- Split audit reporting to document validation limitations

## Steps taken

1. Inspected the dataset structure and verified the class distribution.
2. Built a modular training project with `data/`, `preprocessing/`, `models/`, `evaluation/`, `reports/`, and `ui/`.
3. Implemented dataset analysis, preprocessing previews, augmentation, and training utilities.
4. Trained an initial from-scratch baseline benchmark.
5. Added headless plotting, local artifact caches, and more robust reporting.
6. Switched to pretrained transfer learning and reran the benchmark.
7. Added explainability artifact generation for the winning run.
8. Added split-audit reporting and group-aware split support for future case-aware datasets.
9. Updated the app to auto-discover the strongest completed model run.
10. Added Grad-CAM localization, estimated measurements, risk scoring, and report downloads to the dashboard.

## Benchmark summary

### Before improvement

Run: `outputs/baseline_benchmark_4ep/`

| Model | Epoch | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | ROC-AUC (weighted OVR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| densenet121 | 1 | 0.6205 | 0.6556 | 0.6205 | 0.5418 | 0.9066 |
| resnet50 | 3 | 0.5120 | 0.2622 | 0.5120 | 0.3468 | 0.8059 |
| efficientnet_b0 | 2 | 0.1747 | 0.9042 | 0.1747 | 0.1422 | 0.6501 |

Best scratch baseline:

- Model: `densenet121`
- Accuracy: `62.05%`
- Weighted F1: `54.18%`
- Weighted ROC-AUC: `90.66%`

### After improvement

Run: `outputs/baseline_pretrained_8ep/`

| Model | Epoch | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | ROC-AUC (weighted OVR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| efficientnet_b0 | 7 | 0.9759 | 0.9759 | 0.9759 | 0.9759 | 0.9963 |
| densenet121 | 8 | 0.9578 | 0.9615 | 0.9578 | 0.9589 | 0.9945 |
| resnet50 | 4 | 0.9337 | 0.9342 | 0.9337 | 0.9337 | 0.9894 |

Best improved model:

- Model: `efficientnet_b0`
- Accuracy: `97.59%`
- Weighted F1: `97.59%`
- Weighted ROC-AUC: `99.63%`

## Before vs after

Best baseline vs best improved model:

| Metric | Scratch baseline | Pretrained improved | Absolute gain |
| --- | ---: | ---: | ---: |
| Accuracy | 0.6205 | 0.9759 | +0.3554 |
| Weighted F1 | 0.5418 | 0.9759 | +0.4341 |
| Weighted ROC-AUC | 0.9066 | 0.9963 | +0.0897 |

Main reason for the jump:

- Pretrained transfer learning was the single biggest improvement.
- `efficientnet_b0` benefited the most from pretrained initialization.
- The final dashboard uses this improved checkpoint automatically.

## Explainability artifacts

The best pretrained run generated Grad-CAM overlays in:

- `outputs/baseline_pretrained_8ep/explainability/efficientnet_b0/`

These artifacts show:

- original CT slice
- heatmap
- overlay
- predicted class confidence

Grad-CAM is used as an explainability and weak-localization tool. It highlights
regions that influenced the classifier, but it is not equivalent to a radiologist
tumor annotation or supervised segmentation mask.

## Dashboard outputs

For each uploaded CT slice, the Streamlit app now provides:

- Prediction result: `Lung Cancer Detected`, `Abnormal Lung Finding Detected (Benign)`, or `Lung Cancer Not Detected`
- Model class: `malignant`, `benign`, or `normal`
- Confidence score
- Per-class probabilities
- Grad-CAM heatmap overlay
- Estimated bounding box around the strongest activation region
- Thresholded Grad-CAM pseudo-mask
- Estimated suspicious-region area in pixels
- Estimated maximum and minimum diameter in pixels
- Estimated image-region location such as `Upper Right Lung Region`
- Risk score from `0` to `100`
- Risk category: `Low`, `Moderate`, or `High`
- Stage-related indicators such as activation burden and malignancy probability
- Downloadable JSON and PDF report
- Direct model benchmark metrics in the web app

The dashboard uses recommended internal localization defaults. It reports
pixel-based measurements for the current JPG dataset and does not display
unavailable physical measurements.

## Measurement limitations

The measurement layer is useful for a research prototype and demo, but it is not
a substitute for radiologist annotations or DICOM-based measurement.

- Grad-CAM is weak localization, not supervised tumor segmentation.
- Estimated area and diameter come from a thresholded heatmap pseudo-mask.
- True physical measurements require pixel spacing from the source DICOM.
- Cancer staging cannot be inferred from a single JPG slice.
- TNM staging requires tumor extent, nodal involvement, metastasis status, and clinical context.

## Validation note

The pipeline now writes a split audit file at:

- `outputs/<run_name>/analysis/split_audit.json`

Current audit finding from the smoke validation run:

- requested split mode: `auto`
- used split mode: `slice`
- recoverable case IDs: `0`
- warning: `Auto split selected slice-level validation because reliable case identifiers were not recoverable from the current dataset layout.`

This means the current benchmark is strong for slice-level classification, but it
should not yet be presented as patient-level validation.

## Quick start

```powershell
python -m pip install -r requirements.txt
python train.py --dataset-dir dataset --epochs 8 --models resnet50 efficientnet_b0 densenet121 --split-mode auto --pretrained
streamlit run app.py
```

## Outputs

Training artifacts are saved in `outputs/`:

- `analysis/`: dataset reports, preprocessing previews, and split audits
- `checkpoints/`: saved model weights
- `metrics/`: confusion matrices, ROC curves, and learning history
- `explainability/`: Grad-CAM images
- `reports/`: benchmark leaderboard and summary reports

Most important produced artifacts:

- Best benchmark JSON: `outputs/baseline_pretrained_8ep/reports/best_model.json`
- Benchmark table: `outputs/baseline_pretrained_8ep/reports/benchmark_results.json`
- PDF summary: `outputs/baseline_pretrained_8ep/reports/best_model_summary.pdf`
- Best checkpoint: `outputs/baseline_pretrained_8ep/checkpoints/efficientnet_b0_best.pt`
- Dashboard reports: generated on demand as JSON and PDF from uploaded scans

## Next improvement

The best next research improvement is case-aware evaluation. If future dataset
organization includes per-case folders or a case mapping file, the training
pipeline is already prepared to support group-aware splitting.
