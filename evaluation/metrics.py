from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _to_serializable(value):
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def compute_metrics(
    labels: list[int],
    predictions: list[int],
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_weighted": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }
    try:
        one_hot = label_binarize(labels, classes=list(range(len(class_names))))
        metrics["roc_auc_ovr_weighted"] = float(
            roc_auc_score(one_hot, probabilities, multi_class="ovr", average="weighted")
        )
    except ValueError:
        metrics["roc_auc_ovr_weighted"] = None
    return metrics


def save_confusion_matrix(
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    figure, axis = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names).plot(
        cmap="Blues",
        ax=axis,
        colorbar=False,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_roc_curve(
    labels: list[int],
    probabilities: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    one_hot = label_binarize(labels, classes=list(range(len(class_names))))
    figure, axis = plt.subplots(figsize=(7, 6))
    for class_index, class_name in enumerate(class_names):
        if one_hot[:, class_index].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(one_hot[:, class_index], probabilities[:, class_index])
        axis.plot(fpr, tpr, label=class_name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("One-vs-Rest ROC")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(_to_serializable(metrics), indent=2), encoding="utf-8")
