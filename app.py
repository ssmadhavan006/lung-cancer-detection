from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import streamlit as st
import torch
import torch.nn.functional as F

from explainability.gradcam import GradCAM
from measurements import compute_risk_score, make_overlay_and_bbox, measure_gradcam_region
from models.classification import create_model
from preprocessing.transforms import build_eval_transforms


DEFAULT_HEATMAP_THRESHOLD = 0.55
DEFAULT_PIXEL_SPACING_MM = None


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
        raise FileNotFoundError("No valid trained model reports were found under outputs/.")

    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])), reverse=True)
    _, _, report_path, report = candidates[0]
    return report_path, report


def load_checkpoint(checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = create_model(checkpoint["model_name"], len(class_names), use_pretrained=False).to(device)
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


def benchmark_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for item in results:
        rows.append(
            {
                "Model": item["model_name"],
                "Best Epoch": item["best_epoch"],
                "Accuracy": item["accuracy"],
                "Precision": item["precision_weighted"],
                "Recall": item["recall_weighted"],
                "F1": item["f1_weighted"],
                "ROC-AUC": item["roc_auc_ovr_weighted"],
            }
        )
    return pd.DataFrame(rows)


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
            class_name: round(score * 100, 2) for class_name, score in probabilities.items()
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
            "activation_burden": tumor_burden_indicator(measurements["area_fraction"]),
            "malignancy_probability_percent": round(probabilities.get("malignant", 0.0) * 100, 2),
            "staging_note": "TNM cancer stage cannot be inferred from this JPG slice alone.",
        },
        "clinical_limitations": [
            "Grad-CAM localization is weakly supervised and approximate.",
            "Pixel measurements are not equivalent to clinical measurements.",
            "Millimeter estimates require a manually supplied pixel spacing value.",
            "Cancer staging requires clinical metadata, nodal status, metastasis assessment, and calibrated imaging measurements.",
        ],
    }
    if measurements["estimated_area_mm2"] is not None:
        report["estimated_area_mm2"] = round(measurements["estimated_area_mm2"], 2)
    if measurements["estimated_max_diameter_mm"] is not None:
        report["estimated_max_diameter_mm"] = round(measurements["estimated_max_diameter_mm"], 2)
    return report


def report_to_pdf_bytes(report: dict) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
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
        ("Malignancy Probability", f"{report['stage_related_indicators']['malignancy_probability_percent']:.2f}%"),
    ]
    if "estimated_area_mm2" in report:
        lines.insert(8, ("Estimated Area mm2", f"{report['estimated_area_mm2']:.2f}"))
    if "estimated_max_diameter_mm" in report:
        lines.insert(9, ("Estimated Diameter mm", f"{report['estimated_max_diameter_mm']:.2f}"))
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
    buffer.seek(0)
    return buffer.read()


st.set_page_config(page_title="Lung Cancer Detection", layout="wide")
st.title("Lung Cancer Detection Dashboard")
st.caption("Research-oriented CT slice classifier for the IQ-OTH/NCCD dataset.")

outputs_dir = Path("outputs")
if not outputs_dir.exists():
    st.warning("Run `python train.py` first to generate a trained checkpoint.")
    st.stop()

try:
    best_model_path, best_model_info = discover_best_model_report(outputs_dir)
except FileNotFoundError:
    st.warning("Run `python train.py` first to generate a trained checkpoint.")
    st.stop()

checkpoint_path = Path(best_model_info["checkpoint_path"])
model, class_names, device = load_checkpoint(checkpoint_path)
transform = build_eval_transforms(image_size=224)
benchmark_results = load_benchmark_results(best_model_path)

st.info(
    "Loaded best available run: "
    f"`{best_model_info['model_name']}` from `{best_model_path.parent.parent.name}` "
    f"(F1 `{best_model_info['f1_weighted']:.4f}`, accuracy `{best_model_info['accuracy']:.4f}`)."
)

uploaded_file = st.file_uploader("Upload CT slice image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    prediction_index = int(probabilities.argmax())
    predicted_class = class_names[prediction_index]
    confidence = float(probabilities[prediction_index])
    probability_map = {
        class_name: float(score) for class_name, score in zip(class_names, probabilities)
    }
    malignant_index = class_names.index("malignant") if "malignant" in class_names else prediction_index
    cam_index = prediction_index if predicted_class != "normal" else malignant_index

    grad_cam = GradCAM(model)
    try:
        with torch.enable_grad():
            heatmap = grad_cam.generate(tensor, cam_index)
    finally:
        grad_cam.remove_hooks()

    measurements, region_mask = measure_gradcam_region(
        heatmap=heatmap,
        image_shape=image_np.shape[:2],
        threshold=DEFAULT_HEATMAP_THRESHOLD,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
    )
    overlay = make_overlay_and_bbox(image_np, heatmap, measurements)
    risk_score, risk_category = compute_risk_score(predicted_class, probability_map, measurements)
    report = build_report(
        uploaded_name=uploaded_file.name,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probability_map,
        measurements=measurements.to_dict(),
        risk_score=risk_score,
        risk_category=risk_category,
    )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        view_tab, heatmap_tab, mask_tab = st.tabs(["Original", "Heatmap", "Pseudo-mask"])
        with view_tab:
            st.image(image, caption="Uploaded CT slice", use_container_width=True)
        with heatmap_tab:
            st.image(overlay, caption="Grad-CAM overlay with estimated bounding box", use_container_width=True)
        with mask_tab:
            st.image(region_mask, caption="Thresholded Grad-CAM pseudo-mask", use_container_width=True)
    with col2:
        st.subheader("Prediction")
        st.metric("Result", diagnosis_label(predicted_class))
        st.metric("Confidence", f"{confidence:.2%}")
        st.metric("Risk Score", f"{risk_score}/100", risk_category)
        st.write(f"Model class: `{predicted_class}`")
        st.subheader("Class Probabilities")
        for class_name, score in probability_map.items():
            st.progress(float(score), text=f"{class_name}: {score:.2%}")

    st.subheader("Tumor Localization and Measurements")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Location", measurements.location_label)
    metric_cols[1].metric("Area", f"{measurements.area_px} px")
    metric_cols[2].metric("Max Diameter", f"{measurements.max_diameter_px:.1f} px")
    metric_cols[3].metric("Activation Burden", tumor_burden_indicator(measurements.area_fraction))

    if measurements.estimated_area_mm2 is not None and measurements.estimated_max_diameter_mm is not None:
        mm_cols = st.columns(3)
        mm_cols[0].metric("Estimated Area mm2", f"{measurements.estimated_area_mm2:.2f} mm2")
        mm_cols[1].metric("Estimated Diameter mm", f"{measurements.estimated_max_diameter_mm:.2f} mm")
        mm_cols[2].metric("Malignancy Probability", f"{probability_map.get('malignant', 0.0):.2%}")
    else:
        st.metric("Malignancy Probability", f"{probability_map.get('malignant', 0.0):.2%}")

    st.caption(
        "Grad-CAM highlights influential image regions, not ground-truth tumor boundaries. "
        "Pixel measurements are approximate because this JPG dataset does not provide DICOM spacing or masks."
    )

    st.subheader("Report Summary")
    summary_rows = {
        "Prediction": report["prediction"],
        "Confidence": f"{report['confidence_percent']:.2f}%",
        "Location Estimate": report["tumor_location_estimate"],
        "Estimated Area": f"{report['estimated_area_pixels']} px",
        "Estimated Max Diameter": f"{report['estimated_max_diameter_pixels']:.2f} px",
        "Risk": f"{report['risk_category']} ({report['risk_score']}/100)",
    }
    st.table(pd.DataFrame(summary_rows.items(), columns=["Field", "Value"]))
    report_json = json.dumps(report, indent=2).encode("utf-8")
    report_pdf = report_to_pdf_bytes(report)
    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        label="Download JSON report",
        data=report_json,
        file_name=f"{Path(uploaded_file.name).stem}_lung_report.json",
        mime="application/json",
    )
    download_col2.download_button(
        label="Download PDF report",
        data=report_pdf,
        file_name=f"{Path(uploaded_file.name).stem}_lung_report.pdf",
        mime="application/pdf",
    )

if benchmark_results:
    st.subheader("Model Benchmark")
    best_result = benchmark_results[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Best Model", best_result["model_name"])
    metric_cols[1].metric("Accuracy", f"{best_result['accuracy']:.2%}")
    metric_cols[2].metric("Weighted F1", f"{best_result['f1_weighted']:.2%}")
    metric_cols[3].metric("ROC-AUC", f"{best_result['roc_auc_ovr_weighted']:.2%}")
    st.dataframe(
        benchmark_dataframe(benchmark_results),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "Precision": st.column_config.NumberColumn(format="%.4f"),
            "Recall": st.column_config.NumberColumn(format="%.4f"),
            "F1": st.column_config.NumberColumn(format="%.4f"),
            "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
        },
    )
