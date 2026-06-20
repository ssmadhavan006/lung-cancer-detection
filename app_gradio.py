"""
Lung Cancer Detection — Gradio Application
Research-oriented CT slice classifier for the IQ-OTH/NCCD dataset.
Migrated from Streamlit to Gradio for Hugging Face Spaces deployment.

All ML pipeline code is reused from existing modules (models/, preprocessing/,
explainability/, measurements/, reports/). Only the UI layer has been replaced.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from explainability.gradcam import GradCAM
from measurements import (
    compute_risk_score,
    make_overlay_and_bbox,
    measure_gradcam_region,
)
from models.classification import create_model
from preprocessing.transforms import build_eval_transforms

# ──────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────
DEFAULT_HEATMAP_THRESHOLD = 0.55
DEFAULT_PIXEL_SPACING_MM = None
OUTPUTS_DIR = Path("outputs")

# ──────────────────────────────────────────────────────
# Helper functions (ported from app.py, not modified)
# ──────────────────────────────────────────────────────

def discover_best_model_report(outputs_dir: Path) -> tuple[Path, dict]:
    candidates: list[tuple[float, float, Path, dict]] = []
    for report_path in outputs_dir.glob("**/reports/best_model.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoint_path = Path(report.get("checkpoint_path", ""))
        if not checkpoint_path.exists():
            continue
        f1_score = float(report.get("f1_weighted", -1.0))
        accuracy = float(report.get("accuracy", -1.0))
        candidates.append((f1_score, accuracy, report_path, report))

    if not candidates:
        raise FileNotFoundError(
            "No valid trained model reports were found under outputs/."
        )
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])), reverse=True)
    _, _, report_path, report = candidates[0]
    return report_path, report


def load_checkpoint(checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = create_model(
        checkpoint["model_name"], len(class_names), use_pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_names, device


def load_benchmark_results(best_model_path: Path) -> list[dict]:
    benchmark_path = best_model_path.with_name("benchmark_results.json")
    if not benchmark_path.exists():
        return []
    try:
        return json.loads(benchmark_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def diagnosis_label(predicted_class: str) -> str:
    if predicted_class == "malignant":
        return "Lung Cancer Detected"
    if predicted_class == "benign":
        return "Abnormal Lung Finding Detected (Benign)"
    return "Lung Cancer Not Detected"


def tumor_burden_indicator(area_fraction: float) -> str:
    if area_fraction <= 0:
        return "No focal activation burden"
    if area_fraction < 0.015:
        return "Small activation burden"
    if area_fraction < 0.05:
        return "Moderate activation burden"
    return "Large activation burden"


def build_report(
    uploaded_name: str,
    predicted_class: str,
    confidence: float,
    probabilities: dict[str, float],
    measurements: dict,
    risk_score: int,
    risk_category: str,
) -> dict:
    report = {
        "input_image": uploaded_name,
        "prediction": diagnosis_label(predicted_class),
        "model_class": predicted_class,
        "confidence_percent": round(confidence * 100, 2),
        "class_probabilities_percent": {
            k: round(v * 100, 2) for k, v in probabilities.items()
        },
        "tumor_location_estimate": measurements["location_label"],
        "bounding_box_xyxy": measurements["bbox_xyxy"],
        "estimated_area_pixels": measurements["area_px"],
        "estimated_area_fraction": round(measurements["area_fraction"], 5),
        "estimated_max_diameter_pixels": round(measurements["max_diameter_px"], 2),
        "estimated_min_diameter_pixels": round(measurements["min_diameter_px"], 2),
        "risk_score": risk_score,
        "risk_category": risk_category,
        "stage_related_indicators": {
            "activation_burden": tumor_burden_indicator(
                measurements["area_fraction"]
            ),
            "malignancy_probability_percent": round(
                probabilities.get("malignant", 0.0) * 100, 2
            ),
            "staging_note": "TNM cancer stage cannot be inferred from this JPG slice alone.",
        },
        "clinical_limitations": [
            "Grad-CAM localization is weakly supervised and approximate.",
            "Pixel measurements are not equivalent to clinical measurements.",
            "Millimeter estimates require a manually supplied pixel spacing value.",
            "Cancer staging requires clinical metadata, nodal status, metastasis "
            "assessment, and calibrated imaging measurements.",
        ],
    }
    if measurements["estimated_area_mm2"] is not None:
        report["estimated_area_mm2"] = round(measurements["estimated_area_mm2"], 2)
    if measurements["estimated_max_diameter_mm"] is not None:
        report["estimated_max_diameter_mm"] = round(
            measurements["estimated_max_diameter_mm"], 2
        )
    return report


def report_to_pdf_bytes(report: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = tempfile.SpooledTemporaryFile()
    # Write to a real file so we can seek / read back, but BytesIO is simpler:
    bio = __import__("io").BytesIO()
    pdf = canvas.Canvas(bio, pagesize=A4)
    _, height = A4
    y = height - 48
    pdf.setTitle("Lung Cancer Detection Report")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(48, y, "Lung Cancer Detection Report")
    y -= 34
    pdf.setFont("Helvetica", 10)

    lines = [
        ("Input Image", report["input_image"]),
        ("Prediction", report["prediction"]),
        ("Model Class", report["model_class"]),
        ("Confidence", f"{report['confidence_percent']:.2f}%"),
        ("Tumor Location Estimate", report["tumor_location_estimate"]),
        ("Bounding Box", str(report["bounding_box_xyxy"])),
        ("Estimated Area", f"{report['estimated_area_pixels']} px"),
        ("Estimated Diameter", f"{report['estimated_max_diameter_pixels']:.2f} px"),
        ("Risk Score", f"{report['risk_score']} / 100"),
        ("Risk Category", report["risk_category"]),
        ("Activation Burden", report["stage_related_indicators"]["activation_burden"]),
        (
            "Malignancy Probability",
            f"{report['stage_related_indicators']['malignancy_probability_percent']:.2f}%",
        ),
    ]
    if "estimated_area_mm2" in report:
        lines.insert(8, ("Estimated Area mm2", f"{report['estimated_area_mm2']:.2f}"))
    if "estimated_max_diameter_mm" in report:
        lines.insert(
            9, ("Estimated Diameter mm", f"{report['estimated_max_diameter_mm']:.2f}")
        )
    for label, value in lines:
        pdf.drawString(48, y, f"{label}: {value}")
        y -= 18

    y -= 8
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(48, y, "Limitations")
    y -= 18
    pdf.setFont("Helvetica", 9)
    for limitation in report["clinical_limitations"]:
        pdf.drawString(58, y, f"- {limitation}")
        y -= 15

    pdf.save()
    bio.seek(0)
    return bio.read()


# ──────────────────────────────────────────────────────
# Startup: load model
# ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not OUTPUTS_DIR.exists():
    raise RuntimeError("No outputs/ directory found. Run `python train.py` first.")

try:
    best_model_path, best_model_info = discover_best_model_report(OUTPUTS_DIR)
except FileNotFoundError as e:
    raise RuntimeError(str(e))

checkpoint_path = Path(best_model_info["checkpoint_path"])
model, class_names, device = load_checkpoint(checkpoint_path)
transform = build_eval_transforms(image_size=224)
benchmark_results = load_benchmark_results(best_model_path)

MODEL_NAME = best_model_info["model_name"]
MODEL_F1 = best_model_info["f1_weighted"]
MODEL_ACC = best_model_info["accuracy"]
MODEL_AUC = best_model_info.get("roc_auc_ovr_weighted", "N/A")

# ──────────────────────────────────────────────────────
# Inference function
# ──────────────────────────────────────────────────────
def analyze_image(image_np: np.ndarray | None):
    if image_np is None:
        return [gr.update()] * 20

    image_pil = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")
    image_rgb = np.array(image_pil)
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(probabilities.argmax())
    predicted_class = class_names[pred_idx]
    confidence = float(probabilities[pred_idx])
    prob_map = {c: float(p) for c, p in zip(class_names, probabilities)}

    malignant_idx = (
        class_names.index("malignant") if "malignant" in class_names else pred_idx
    )
    cam_idx = pred_idx if predicted_class != "normal" else malignant_idx

    grad_cam = GradCAM(model)
    try:
        with torch.enable_grad():
            heatmap = grad_cam.generate(tensor, cam_idx)
    finally:
        grad_cam.remove_hooks()

    measurements, region_mask = measure_gradcam_region(
        heatmap=heatmap,
        image_shape=image_rgb.shape[:2],
        threshold=DEFAULT_HEATMAP_THRESHOLD,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
    )
    overlay = make_overlay_and_bbox(image_rgb, heatmap, measurements)
    risk_score, risk_category = compute_risk_score(
        predicted_class, prob_map, measurements
    )
    report = build_report(
        uploaded_name="ct_slice.jpg",
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=prob_map,
        measurements=measurements.to_dict(),
        risk_score=risk_score,
        risk_category=risk_category,
    )

    report_json_str = json.dumps(report, indent=2)
    report_pdf_bytes = report_to_pdf_bytes(report)

    # ── Probabilities as text bars ──
    prob_lines = []
    for cls_name, score in prob_map.items():
        bar_len = max(1, int(score * 18))
        bar = "▓" * bar_len + "░" * (18 - bar_len)
        prob_lines.append(f"**{cls_name}:** {score:.2%}  `{bar}`")
    prob_text = "\n\n".join(prob_lines)

    # ── Measurements ──
    loc = measurements.location_label
    area_px = measurements.area_px
    max_d = measurements.max_diameter_px
    min_d = measurements.min_diameter_px
    burden = tumor_burden_indicator(measurements.area_fraction)
    malign_pct = prob_map.get("malignant", 0.0)
    mm_a = measurements.estimated_area_mm2
    mm_d = measurements.estimated_max_diameter_mm

    # ── Report ──
    json_path = Path(tempfile.mkdtemp()) / "lung_cancer_report.json"
    pdf_path = json_path.with_suffix(".pdf")
    json_path.write_text(report_json_str, encoding="utf-8")
    pdf_path.write_bytes(report_pdf_bytes)

    return [
        image_rgb,                                       # 0  original display
        {"label": diagnosis_label(predicted_class),       # 1  prediction label
         "confidences": {c: float(p) for c, p in prob_map.items()}},
        f"**{confidence:.2%}**",                         # 2  confidence
        f"**{risk_score}** / 100",                       # 3  risk score
        f"**{risk_category}**",                          # 4  risk category
        prob_text,                                        # 5  probabilities
        overlay,                                          # 6  grad-cam overlay
        region_mask,                                      # 7  pseudo-mask
        f"**{loc}**",                                    # 8  location
        f"**{area_px:,}** px",                           # 9  area
        f"**{max_d:.1f}** px",                           # 10 max diameter
        f"**{min_d:.1f}** px",                           # 11 min diameter
        f"**{burden}**",                                 # 12 activation burden
        f"**{malign_pct:.2%}**",                         # 13 malignancy prob
        f"**{mm_a:.2f} mm²**" if mm_a else "Not available",  # 14 area mm
        f"**{mm_d:.2f} mm**" if mm_d else "Not available",   # 15 diam mm
        report,                                            # 16 report JSON
        gr.File(value=str(json_path),                     # 17 json download
                label="Download JSON Report", visible=True),
        gr.File(value=str(pdf_path),                      # 18 pdf download
                label="Download PDF Report", visible=True),
    ]


# ──────────────────────────────────────────────────────
# Build Gradio UI
# ──────────────────────────────────────────────────────

_CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
footer { display: none !important; }
h1 { font-weight: 600 !important; color: #1a1a2e; }
.prose p { font-size: 0.95rem; line-height: 1.6; }
.disclaimer { font-size: 0.85rem; color: #666;
  border-left: 4px solid #4A90D9; padding: 10px 16px;
  margin: 12px 0; background: #f8f9fa; border-radius: 4px; }
.metric-row { text-align: center; padding: 8px 0; }
"""


def build_app():
    with gr.Blocks(
        title="Lung Cancer Detection",
    ) as demo:

        # ── Header ──
        gr.Markdown("# Lung Cancer Detection Dashboard")
        gr.Markdown(
            "Research-oriented CT slice classifier for the IQ-OTH/NCCD dataset. "
            "Built with EfficientNet-B0 via transfer learning."
        )
        gr.HTML(
            '<div class="disclaimer">'
            "<strong>Research Prototype</strong> — Not for clinical use. "
            "This system is a demonstration tool and is not FDA-cleared or HIPAA-compliant. "
            "Always consult a qualified medical professional for diagnosis."
            "</div>"
        )
        with gr.Row():
            gr.Markdown(f"**Model:** `{MODEL_NAME}`")
            gr.Markdown(f"**Accuracy:** `{MODEL_ACC:.4f}`")
            gr.Markdown(f"**Weighted F1:** `{MODEL_F1:.4f}`")
            gr.Markdown(f"**ROC-AUC:** `{MODEL_AUC}`")

        gr.Markdown("---")

        # ── Tabs ──
        with gr.Tabs():
            # ═══════════ Tab 1: Image Analysis ═══════════
            with gr.Tab("📋 Image Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload CT Slice (JPG / PNG)",
                            height=320,
                        )
                        analyze_btn = gr.Button(
                            "Analyze Slice", variant="primary", size="lg"
                        )
                    with gr.Column(scale=1):
                        original_display = gr.Image(
                            label="Original Image", height=320
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Prediction")
                        prediction_label = gr.Label(label="Diagnosis", num_top_classes=3)
                        confidence_display = gr.Markdown("**Confidence:** —")
                    with gr.Column(scale=1):
                        gr.Markdown("### Risk Assessment")
                        risk_score_display = gr.Markdown("**Risk Score:** —")
                        risk_category_display = gr.Markdown("**Risk Category:** —")

                gr.Markdown("### Class Probabilities")
                prob_markdown = gr.Markdown(
                    "Upload an image and click **Analyze Slice** to see results."
                )

            # ═══════════ Tab 2: Explainability ═══════════
            with gr.Tab("🔬 Explainability"):
                gr.Markdown("### Grad-CAM Visualization")
                gr.Markdown(
                    "Grad-CAM highlights the regions that most influenced the model's decision. "
                    "The overlay includes an estimated bounding box around the strongest "
                    "activation region with a center marker."
                )
                with gr.Row():
                    with gr.Column():
                        overlay_display = gr.Image(
                            label="Grad-CAM Overlay", height=400
                        )
                    with gr.Column():
                        mask_display = gr.Image(
                            label="Thresholded Pseudo-Mask", height=400
                        )
                gr.Markdown(
                    "The pseudo-mask is obtained by adaptive thresholding of the Grad-CAM heatmap "
                    "followed by morphological cleanup. **Note:** This is weak localization, "
                    "not supervised segmentation."
                )

            # ═══════════ Tab 3: Clinical Measurements ═══════════
            with gr.Tab("📐 Measurements"):
                gr.Markdown("### Tumor Measurements")
                gr.Markdown(
                    "Pixel-based measurements derived from the thresholded Grad-CAM "
                    "activation region. Physical millimeter values require DICOM pixel spacing "
                    "metadata, which is not available in this JPG dataset."
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Anatomical Location")
                        location_display = gr.Markdown("—")
                    with gr.Column():
                        gr.Markdown("#### Pixel Area")
                        area_display = gr.Markdown("—")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Maximum Diameter")
                        max_diam_display = gr.Markdown("—")
                    with gr.Column():
                        gr.Markdown("#### Minimum Diameter")
                        min_diam_display = gr.Markdown("—")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Activation Burden")
                        burden_display = gr.Markdown("—")
                    with gr.Column():
                        gr.Markdown("#### Malignancy Probability")
                        malignancy_display = gr.Markdown("—")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Estimated Area (mm²)")
                        area_mm_display = gr.Markdown("Not available (requires DICOM)")
                    with gr.Column():
                        gr.Markdown("#### Estimated Diameter (mm)")
                        diam_mm_display = gr.Markdown("Not available (requires DICOM)")

            # ═══════════ Tab 4: Reports ═══════════
            with gr.Tab("📄 Reports"):
                gr.Markdown("### Clinical Report")
                gr.Markdown(
                    "Download the analysis as JSON or PDF. The report includes the prediction, "
                    "confidence, tumor measurements, risk assessment, and clinical limitations."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        download_json = gr.File(
                            label="Download JSON Report", visible=False
                        )
                        download_pdf = gr.File(
                            label="Download PDF Report", visible=False
                        )
                        gr.Markdown(
                            "Run an analysis first to generate downloadable reports."
                        )
                    with gr.Column(scale=2):
                        report_preview = gr.JSON(label="Report Preview")

            # ═══════════ Tab 5: Benchmark ═══════════
            with gr.Tab("📊 Benchmark"):
                gr.Markdown("### Model Benchmark Results")
                gr.Markdown(
                    "Comparative evaluation of three CNN architectures on the IQ-OTH/NCCD "
                    "dataset using pretrained transfer learning. "
                    "Train/val/test: 70/15/15 stratified split."
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"**Best Model**  \n`{MODEL_NAME}`")
                    with gr.Column():
                        gr.Markdown(f"**Accuracy**  \n`{MODEL_ACC:.4f}`")
                    with gr.Column():
                        gr.Markdown(f"**Weighted F1**  \n`{MODEL_F1:.4f}`")
                    with gr.Column():
                        gr.Markdown(f"**ROC-AUC**  \n`{MODEL_AUC}`")

                gr.Markdown("#### Architecture Comparison")

                bench_rows = []
                if benchmark_results:
                    for item in benchmark_results:
                        auc = item.get("roc_auc_ovr_weighted")
                        bench_rows.append(
                            {
                                "Model": item["model_name"],
                                "Epoch": item["best_epoch"],
                                "Accuracy": f"{item['accuracy']:.4f}",
                                "Precision": f"{item['precision_weighted']:.4f}",
                                "Recall": f"{item['recall_weighted']:.4f}",
                                "F1": f"{item['f1_weighted']:.4f}",
                                "ROC-AUC": f"{auc:.4f}" if auc else "N/A",
                            }
                        )

                gr.Dataframe(
                    value=bench_rows,
                    headers=[
                        "Model", "Epoch", "Accuracy", "Precision",
                        "Recall", "F1", "ROC-AUC"
                    ],
                    label="Benchmark Results",
                    interactive=False,
                )

                gr.Markdown(
                    "> **Key finding:** Transfer learning from ImageNet weights improved "
                    "accuracy from 62.05% to 97.59% (+35.5 pp). EfficientNet-B0 benefited "
                    "the most from pretrained initialization."
                )

        # ── Wire up the analyze button ──
        outputs = [
            original_display,       # 0
            prediction_label,       # 1
            confidence_display,     # 2
            risk_score_display,     # 3
            risk_category_display,  # 4
            prob_markdown,          # 5
            overlay_display,        # 6
            mask_display,           # 7
            location_display,       # 8
            area_display,           # 9
            max_diam_display,       # 10
            min_diam_display,       # 11
            burden_display,         # 12
            malignancy_display,     # 13
            area_mm_display,        # 14
            diam_mm_display,        # 15
            report_preview,         # 16
            download_json,          # 17
            download_pdf,           # 18
        ]

        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input],
            outputs=outputs,
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        css=_CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="gray"),
        server_name="0.0.0.0",
        show_error=True,
    )
