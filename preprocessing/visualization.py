from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from preprocessing.transforms import enhance_image


def save_preprocessing_preview(records: list[dict], output_dir: Path, sample_count: int = 6) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_records = records[:sample_count]
    figure, axes = plt.subplots(len(preview_records), 2, figsize=(8, 3 * len(preview_records)))
    if len(preview_records) == 1:
        axes = np.array([axes])

    for row_index, record in enumerate(preview_records):
        with Image.open(record["path"]) as image:
            original = image.convert("RGB")
            original_np = np.array(original)
        enhanced = enhance_image(original_np)

        axes[row_index, 0].imshow(original_np, cmap="gray")
        axes[row_index, 0].set_title(f"Original: {record['class_name']}")
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(enhanced, cmap="gray")
        axes[row_index, 1].set_title("Enhanced")
        axes[row_index, 1].axis("off")

    figure.tight_layout()
    output_path = output_dir / "preprocessing_preview.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path
