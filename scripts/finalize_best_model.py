from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import TrainingConfig
from data.dataset import create_dataloaders, create_split_records
from explainability.gradcam import create_explainability_artifacts
from preprocessing.transforms import build_eval_transforms, build_train_transforms
from reports.reporting import save_clinical_summary_pdf
from train import ensure_dirs, load_best_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize a completed benchmark run.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = ensure_dirs(args.output_dir)
    results_path = output_paths["reports"] / "benchmark_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing benchmark results: {results_path}")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    best_result = results[0]
    (output_paths["reports"] / "best_model.json").write_text(
        json.dumps(best_result, indent=2),
        encoding="utf-8",
    )

    config = TrainingConfig(output_dir=args.output_dir)
    split_records = create_split_records(
        dataset_dir=args.dataset_dir,
        analysis_dir=output_paths["analysis"],
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    loaders, _ = create_dataloaders(
        split_records=split_records,
        train_transform=build_train_transforms(config.image_size),
        eval_transform=build_eval_transforms(config.image_size),
        batch_size=config.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_best_model(best_result, split_records.class_names, device)

    warning = None
    try:
        create_explainability_artifacts(
            model=model,
            dataloader=loaders["test"],
            device=device,
            class_names=split_records.class_names,
            output_dir=output_paths["explainability"] / best_result["model_name"],
        )
    except Exception as error:
        warning = f"Explainability artifact generation skipped: {error}"
        (output_paths["reports"] / "explainability_warning.txt").write_text(
            warning,
            encoding="utf-8",
        )

    summary = {
        "Prediction": f"Best benchmark model: {best_result['model_name']}",
        "Classification Confidence": f"Weighted F1 = {best_result['f1_weighted']:.4f}",
        "Accuracy": f"{best_result['accuracy']:.4f}",
        "Malignancy Probability": "Derived per-image during inference from softmax scores.",
        "Tumor Area": "Not available from the current slice-level dataset without segmentation masks.",
        "Tumor Location": "Weakly supervised heatmaps are available in explainability outputs.",
        "Risk Category": "Use clinical judgement; this pipeline is research-only.",
    }
    if warning is not None:
        summary["Explainability"] = warning

    save_clinical_summary_pdf(summary, output_paths["reports"] / "best_model_summary.pdf")
    print(json.dumps({"best_model": best_result, "warning": warning}, indent=2))


if __name__ == "__main__":
    main()
