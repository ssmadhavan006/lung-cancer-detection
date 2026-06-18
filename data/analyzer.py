from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re

import numpy as np
from PIL import Image, UnidentifiedImageError


CLASS_NAME_MAP = {
    "bengin cases": "benign",
    "benign cases": "benign",
    "malignant cases": "malignant",
    "normal cases": "normal",
}


def normalize_class_name(folder_name: str) -> str:
    key = folder_name.strip().lower()
    return CLASS_NAME_MAP.get(key, key.replace(" cases", "").replace(" case", ""))


def infer_case_metadata(
    image_path: Path,
    dataset_dir: Path,
    class_name: str,
    group_regex: str | None = None,
) -> dict:
    relative_path = image_path.resolve().relative_to(dataset_dir.resolve())
    class_root = Path(relative_path.parts[0])
    relative_within_class = Path(*relative_path.parts[1:]) if len(relative_path.parts) > 1 else Path(image_path.name)

    if len(relative_within_class.parts) >= 2:
        case_id = str(relative_within_class.parent).replace("\\", "/")
        return {
            "case_id": case_id,
            "case_source": "folder",
            "case_reliable": True,
        }

    if group_regex:
        match = re.search(group_regex, image_path.stem)
        if match:
            case_fragment = match.group(1) if match.groups() else match.group(0)
            return {
                "case_id": f"{class_name}:{case_fragment}",
                "case_source": "filename_regex",
                "case_reliable": True,
            }

    return {
        "case_id": None,
        "case_source": "unavailable",
        "case_reliable": False,
    }


def analyze_dataset(dataset_dir: Path, output_dir: Path, group_regex: str | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    corrupted_files: list[str] = []
    resolutions: list[tuple[int, int]] = []
    file_extensions = Counter()
    class_counts = Counter()
    case_sources = Counter()
    case_ids = set()

    class_dirs = sorted([path for path in dataset_dir.iterdir() if path.is_dir()])
    for class_dir in class_dirs:
        class_name = normalize_class_name(class_dir.name)
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue
            file_extensions[image_path.suffix.lower()] += 1
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                    mode = image.mode
                resolutions.append((width, height))
                class_counts[class_name] += 1
                case_meta = infer_case_metadata(image_path, dataset_dir, class_name, group_regex=group_regex)
                case_sources[case_meta["case_source"]] += 1
                if case_meta["case_id"] is not None:
                    case_ids.add(case_meta["case_id"])
                records.append(
                    {
                        "path": str(image_path.resolve()),
                        "class_name": class_name,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "extension": image_path.suffix.lower(),
                        "relative_path": str(image_path.resolve().relative_to(dataset_dir.resolve())).replace("\\", "/"),
                        "filename": image_path.name,
                        "case_id": case_meta["case_id"],
                        "case_source": case_meta["case_source"],
                        "case_reliable": case_meta["case_reliable"],
                    }
                )
            except (UnidentifiedImageError, OSError, ValueError):
                corrupted_files.append(str(image_path.resolve()))

    widths = np.array([item[0] for item in resolutions], dtype=np.float32)
    heights = np.array([item[1] for item in resolutions], dtype=np.float32)
    imbalance_ratio = (
        float(max(class_counts.values()) / min(class_counts.values()))
        if class_counts
        else 0.0
    )
    report = {
        "dataset_dir": str(dataset_dir.resolve()),
        "total_images": len(records),
        "class_counts": dict(class_counts),
        "class_distribution": {
            key: round((value / len(records)) * 100, 2) if records else 0.0
            for key, value in class_counts.items()
        },
        "file_extensions": dict(file_extensions),
        "resolution_stats": {
            "min_width": int(widths.min()) if len(widths) else 0,
            "max_width": int(widths.max()) if len(widths) else 0,
            "mean_width": float(widths.mean()) if len(widths) else 0.0,
            "min_height": int(heights.min()) if len(heights) else 0,
            "max_height": int(heights.max()) if len(heights) else 0,
            "mean_height": float(heights.mean()) if len(heights) else 0.0,
        },
        "imbalance_ratio": round(imbalance_ratio, 3),
        "corrupted_files": corrupted_files,
        "case_grouping": {
            "recoverable_case_ids": len(case_ids),
            "source_counts": dict(case_sources),
            "reliable_case_fraction": round(
                sum(1 for record in records if record["case_reliable"]) / len(records),
                4,
            )
            if records
            else 0.0,
            "group_regex": group_regex,
        },
        "notes": [
            "This dataset is slice-level JPG data derived from CT studies.",
            "Patient-level grouping is only reliable when case folders or metadata are available.",
            "Localization and segmentation phases need extra annotations for supervised training.",
        ],
        "records": records,
    }

    report_path = output_dir / "dataset_analysis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_path = output_dir / "dataset_summary.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("class_name,count,percentage\n")
        for class_name, count in sorted(class_counts.items()):
            percentage = report["class_distribution"][class_name]
            handle.write(f"{class_name},{count},{percentage}\n")

    return report
