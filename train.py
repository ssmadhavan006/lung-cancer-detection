from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TORCH_HOME", str((Path("outputs") / ".torch").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from config import TrainingConfig
from data.dataset import create_dataloaders, create_split_records, set_seed
from evaluation.metrics import compute_metrics, save_confusion_matrix, save_metrics_json, save_roc_curve
from explainability.gradcam import create_explainability_artifacts
from models.classification import SUPPORTED_MODELS, create_model
from preprocessing.transforms import build_eval_transforms, build_train_transforms
from preprocessing.visualization import save_preprocessing_preview
from reports.reporting import save_benchmark_report, save_clinical_summary_pdf


def serialize_for_json(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {key: serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_for_json(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lung cancer classification models.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--models", nargs="+", default=["resnet50", "efficientnet_b0", "densenet121"])
    parser.add_argument("--split-mode", choices=["auto", "slice", "group"], default="auto")
    parser.add_argument("--group-regex", type=str, default=None)
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "analysis": output_dir / "analysis",
        "checkpoints": output_dir / "checkpoints",
        "metrics": output_dir / "metrics",
        "reports": output_dir / "reports",
        "explainability": output_dir / "explainability",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_history(history: dict[str, list[float]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_f1"], label="train")
    axes[1].plot(history["val_f1"], label="val")
    axes[1].set_title("Weighted F1")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
    scaler: GradScaler,
    train: bool,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    context = torch.enable_grad if train else torch.no_grad
    with context():
        for images, labels, _ in tqdm(dataloader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            labels_all.extend(labels.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())

    mean_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(
        labels_all,
        predictions_all,
        probabilities=np.eye(len(set(labels_all)))[predictions_all],
        class_names=[str(index) for index in sorted(set(labels_all))],
    )
    return mean_loss, metrics["f1_weighted"]


def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    class_names: list[str],
) -> tuple[float, dict, list[int], list[int], np.ndarray]:
    model.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    probabilities_all: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            probabilities = F.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            labels_all.extend(labels.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())
            probabilities_all.extend(probabilities.detach().cpu().numpy())

    probability_array = np.vstack(probabilities_all)
    metrics = compute_metrics(labels_all, predictions_all, probability_array, class_names)
    mean_loss = total_loss / len(dataloader.dataset)
    return mean_loss, metrics, labels_all, predictions_all, probability_array


def train_single_model(
    model_name: str,
    config: TrainingConfig,
    split_records,
    output_paths: dict[str, Path],
    device: torch.device,
) -> dict:
    train_transform = build_train_transforms(config.image_size)
    eval_transform = build_eval_transforms(config.image_size)
    loaders, loss_weights = create_dataloaders(
        split_records=split_records,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = create_model(model_name, len(split_records.class_names), config.use_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    checkpoint_path = output_paths["checkpoints"] / f"{model_name}_best.pt"
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_f1 = run_epoch(model, loaders["train"], optimizer, criterion, device, scaler, train=True)
        val_loss, val_metrics, _, _, _ = evaluate(model, loaders["val"], criterion, device, split_records.class_names)
        val_f1 = val_metrics["f1_weighted"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "class_names": split_records.class_names,
                    "config": serialize_for_json(asdict(config)),
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    test_loss, test_metrics, labels, predictions, probabilities = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        split_records.class_names,
    )

    metrics_dir = output_paths["metrics"] / model_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_history(history, metrics_dir / "history.png")
    save_metrics_json(test_metrics, metrics_dir / "metrics.json")
    save_confusion_matrix(labels, predictions, split_records.class_names, metrics_dir / "confusion_matrix.png")
    save_roc_curve(labels, probabilities, split_records.class_names, metrics_dir / "roc_curve.png")

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "accuracy": round(test_metrics["accuracy"], 4),
        "precision_weighted": round(test_metrics["precision_weighted"], 4),
        "recall_weighted": round(test_metrics["recall_weighted"], 4),
        "f1_weighted": round(test_metrics["f1_weighted"], 4),
        "roc_auc_ovr_weighted": (
            round(test_metrics["roc_auc_ovr_weighted"], 4)
            if test_metrics["roc_auc_ovr_weighted"] is not None
            else None
        ),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }


def load_best_model(best_result: dict, class_names: list[str], device: torch.device) -> nn.Module:
    model = create_model(best_result["model_name"], len(class_names), use_pretrained=False).to(device)
    state = torch.load(best_result["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    invalid_models = [model_name for model_name in args.models if model_name not in SUPPORTED_MODELS]
    if invalid_models:
        raise ValueError(f"Unsupported model names: {invalid_models}. Supported: {SUPPORTED_MODELS}")

    config = TrainingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        use_pretrained=args.pretrained,
        models=args.models,
        split_mode=args.split_mode,
        group_regex=args.group_regex,
    )
    set_seed(config.seed)
    output_paths = ensure_dirs(config.output_dir)
    split_records = create_split_records(
        dataset_dir=config.dataset_dir,
        analysis_dir=output_paths["analysis"],
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
        split_mode=config.split_mode,
        group_regex=config.group_regex,
    )
    save_preprocessing_preview(split_records.train, output_paths["analysis"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_metadata = {
        "started_at_unix": time.time(),
        "device": str(device),
        "config": serialize_for_json(asdict(config)),
        "class_names": split_records.class_names,
        "split_sizes": {
            "train": len(split_records.train),
            "val": len(split_records.val),
            "test": len(split_records.test),
        },
        "split_summary": split_records.split_summary,
    }
    (config.output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    results = []
    for model_name in config.models:
        results.append(
            train_single_model(
                model_name=model_name,
                config=config,
                split_records=split_records,
                output_paths=output_paths,
                device=device,
            )
        )

    results.sort(key=lambda item: item["f1_weighted"], reverse=True)
    best_result = results[0]
    save_benchmark_report(results, output_paths["reports"])
    (output_paths["reports"] / "best_model.json").write_text(json.dumps(best_result, indent=2), encoding="utf-8")

    train_transform = build_train_transforms(config.image_size)
    eval_transform = build_eval_transforms(config.image_size)
    loaders, _ = create_dataloaders(
        split_records=split_records,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    best_model = load_best_model(best_result, split_records.class_names, device)
    explainability_note = None
    try:
        create_explainability_artifacts(
            model=best_model,
            dataloader=loaders["test"],
            device=device,
            class_names=split_records.class_names,
            output_dir=output_paths["explainability"] / best_result["model_name"],
        )
    except Exception as error:
        explainability_note = f"Explainability artifact generation skipped: {error}"
        (output_paths["reports"] / "explainability_warning.txt").write_text(
            explainability_note,
            encoding="utf-8",
        )

    best_summary = {
        "Prediction": f"Best benchmark model: {best_result['model_name']}",
        "Classification Confidence": f"Weighted F1 = {best_result['f1_weighted']:.4f}",
        "Accuracy": f"{best_result['accuracy']:.4f}",
        "Malignancy Probability": "Derived per-image during inference from softmax scores.",
        "Tumor Area": "Not available from the current slice-level dataset without segmentation masks.",
        "Tumor Location": "Weakly supervised heatmaps are available in explainability outputs.",
        "Risk Category": "Use clinical judgement; this pipeline is research-only.",
    }
    if explainability_note is not None:
        best_summary["Explainability"] = explainability_note
    save_clinical_summary_pdf(best_summary, output_paths["reports"] / "best_model_summary.pdf")

    print(json.dumps({"best_model": best_result, "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
