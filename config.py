from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Literal


@dataclass(slots=True)
class TrainingConfig:
    dataset_dir: Path = Path("dataset")
    output_dir: Path = Path("outputs")
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 3
    seed: int = 42
    num_workers: int = min(8, os.cpu_count() or 1)
    use_pretrained: bool = False
    models: list[str] = field(
        default_factory=lambda: [
            "resnet50",
            "efficientnet_b0",
            "densenet121",
        ]
    )
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    split_mode: Literal["auto", "slice", "group"] = "auto"
    group_regex: str | None = None


DEFAULT_CLASS_NAMES = ["benign", "malignant", "normal"]
