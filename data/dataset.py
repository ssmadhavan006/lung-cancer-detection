from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import random
import json

import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import DEFAULT_CLASS_NAMES
from data.analyzer import analyze_dataset


@dataclass(slots=True)
class SplitRecords:
    train: list[dict]
    val: list[dict]
    test: list[dict]
    class_names: list[str]
    split_summary: dict


class LungCancerDataset(Dataset):
    def __init__(self, records: list[dict], class_names: list[str], transform: Callable | None):
        self.records = records
        self.class_names = class_names
        self.class_to_idx = {name: index for index, name in enumerate(class_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record = self.records[index]
        image_path = Path(record["path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_np = np.array(image)

        if self.transform is not None:
            image_np = self.transform(image=image_np)["image"]

        label = self.class_to_idx[record["class_name"]]
        return image_np, label, str(image_path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records(
    dataset_dir: Path,
    analysis_dir: Path,
    group_regex: str | None = None,
) -> tuple[list[dict], list[str], dict]:
    report = analyze_dataset(dataset_dir, analysis_dir, group_regex=group_regex)
    records = report["records"]
    class_names = [name for name in DEFAULT_CLASS_NAMES if name in report["class_counts"]]
    return records, class_names, report


def _build_split_summary(
    split_mode_requested: str,
    split_mode_used: str,
    report: dict,
    split_records: dict[str, list[dict]],
    warning: str | None,
) -> dict:
    overlap = {
        "train_val": sorted(
            set(record["path"] for record in split_records["train"])
            & set(record["path"] for record in split_records["val"])
        ),
        "train_test": sorted(
            set(record["path"] for record in split_records["train"])
            & set(record["path"] for record in split_records["test"])
        ),
        "val_test": sorted(
            set(record["path"] for record in split_records["val"])
            & set(record["path"] for record in split_records["test"])
        ),
    }
    group_overlap = None
    if split_mode_used == "group":
        group_overlap = {
            "train_val": sorted(
                {record["case_id"] for record in split_records["train"] if record["case_id"]}
                & {record["case_id"] for record in split_records["val"] if record["case_id"]}
            ),
            "train_test": sorted(
                {record["case_id"] for record in split_records["train"] if record["case_id"]}
                & {record["case_id"] for record in split_records["test"] if record["case_id"]}
            ),
            "val_test": sorted(
                {record["case_id"] for record in split_records["val"] if record["case_id"]}
                & {record["case_id"] for record in split_records["test"] if record["case_id"]}
            ),
        }

    return {
        "requested_split_mode": split_mode_requested,
        "used_split_mode": split_mode_used,
        "warning": warning,
        "case_grouping": report.get("case_grouping", {}),
        "split_sizes": {name: len(items) for name, items in split_records.items()},
        "class_counts": {
            name: {
                class_name: sum(1 for record in items if record["class_name"] == class_name)
                for class_name in report["class_counts"].keys()
            }
            for name, items in split_records.items()
        },
        "path_overlap": overlap,
        "case_overlap": group_overlap,
        "validation_note": (
            "Patient-level leakage is mitigated only when `used_split_mode` is `group` "
            "and recoverable case IDs are available."
        ),
    }


def _write_split_summary(analysis_dir: Path, summary: dict) -> None:
    (analysis_dir / "split_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _slice_split(records: list[dict], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    labels = [record["class_name"] for record in records]
    train_records, temp_records = train_test_split(
        records,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = [record["class_name"] for record in temp_records]
    val_size = val_ratio / (1.0 - train_ratio)
    val_records, test_records = train_test_split(
        temp_records,
        test_size=1.0 - val_size,
        random_state=seed,
        stratify=temp_labels,
    )
    return train_records, val_records, test_records


def _group_split(records: list[dict], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    groups = np.array([record["case_id"] for record in records], dtype=object)
    labels = np.array([record["class_name"] for record in records], dtype=object)
    indices = np.arange(len(records))

    train_gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(train_gss.split(indices, labels, groups))

    remaining_ratio = 1.0 - train_ratio
    val_share_of_temp = val_ratio / remaining_ratio
    temp_groups = groups[temp_idx]
    temp_labels = labels[temp_idx]
    val_gss = GroupShuffleSplit(n_splits=1, train_size=val_share_of_temp, random_state=seed)
    val_sub_idx, test_sub_idx = next(val_gss.split(temp_idx, temp_labels, temp_groups))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]
    return (
        [records[index] for index in train_idx.tolist()],
        [records[index] for index in val_idx.tolist()],
        [records[index] for index in test_idx.tolist()],
    )


def create_split_records(
    dataset_dir: Path,
    analysis_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    split_mode: str = "auto",
    group_regex: str | None = None,
) -> SplitRecords:
    set_seed(seed)
    records, class_names, report = load_records(dataset_dir, analysis_dir, group_regex=group_regex)
    reliable_records = [record for record in records if record["case_reliable"]]
    has_grouping = len(reliable_records) == len(records) and len({record["case_id"] for record in reliable_records}) >= 3

    warning = None
    if split_mode == "group":
        if not has_grouping:
            warning = (
                "Group-aware split was requested, but reliable case identifiers were not recoverable "
                "from the current dataset layout. Falling back to slice-level split."
            )
            split_mode_used = "slice"
        else:
            split_mode_used = "group"
    elif split_mode == "auto":
        split_mode_used = "group" if has_grouping else "slice"
        if split_mode_used == "slice":
            warning = (
                "Auto split selected slice-level validation because reliable case identifiers "
                "were not recoverable from the current dataset layout."
            )
    else:
        split_mode_used = "slice"

    if split_mode_used == "group":
        train_records, val_records, test_records = _group_split(records, train_ratio, val_ratio, seed)
    else:
        train_records, val_records, test_records = _slice_split(records, train_ratio, val_ratio, seed)

    split_summary = _build_split_summary(
        split_mode_requested=split_mode,
        split_mode_used=split_mode_used,
        report=report,
        split_records={
            "train": train_records,
            "val": val_records,
            "test": test_records,
        },
        warning=warning,
    )
    _write_split_summary(analysis_dir, split_summary)
    return SplitRecords(
        train=train_records,
        val=val_records,
        test=test_records,
        class_names=class_names,
        split_summary=split_summary,
    )


def create_dataloaders(
    split_records: SplitRecords,
    train_transform: Callable,
    eval_transform: Callable,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, DataLoader], torch.Tensor]:
    train_dataset = LungCancerDataset(split_records.train, split_records.class_names, train_transform)
    val_dataset = LungCancerDataset(split_records.val, split_records.class_names, eval_transform)
    test_dataset = LungCancerDataset(split_records.test, split_records.class_names, eval_transform)

    train_labels = [train_dataset.class_to_idx[record["class_name"]] for record in split_records.train]
    class_counts = np.bincount(train_labels, minlength=len(split_records.class_names))
    class_weights = 1.0 / np.clip(class_counts, a_min=1, a_max=None)
    sample_weights = np.array([class_weights[label] for label in train_labels], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    loss_weights = torch.tensor(class_weights, dtype=torch.float32)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    loaders = {
        "train": DataLoader(train_dataset, sampler=sampler, **loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    }
    return loaders, loss_weights
