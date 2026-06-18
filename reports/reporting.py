from __future__ import annotations

from pathlib import Path
import csv
import json

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def save_benchmark_report(results: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    json_path = output_dir / "benchmark_results.json"

    fieldnames = [
        "model_name",
        "best_epoch",
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc_ovr_weighted",
        "checkpoint_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return csv_path, json_path


def save_clinical_summary_pdf(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    y = height - 50
    pdf.setTitle("Lung Cancer Detection Summary")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Patient Scan Analysis")
    y -= 40
    pdf.setFont("Helvetica", 11)

    for key, value in summary.items():
        pdf.drawString(50, y, f"{key}: {value}")
        y -= 22
        if y < 70:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 11)

    pdf.save()
