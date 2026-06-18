from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass(slots=True)
class LocalizationMeasurements:
    detected: bool
    bbox_xyxy: tuple[int, int, int, int] | None
    center_xy: tuple[int, int] | None
    location_label: str
    area_px: int
    area_fraction: float
    max_diameter_px: float
    min_diameter_px: float
    estimated_area_mm2: float | None
    estimated_max_diameter_mm: float | None
    heatmap_peak: float
    heatmap_mean_in_region: float

    def to_dict(self) -> dict:
        return asdict(self)


def _location_from_center(center_x: int, center_y: int, width: int, height: int) -> str:
    vertical = "Upper" if center_y < height / 3 else "Lower" if center_y > (2 * height) / 3 else "Middle"
    horizontal = "Left" if center_x < width / 3 else "Right" if center_x > (2 * width) / 3 else "Central"
    if horizontal == "Central":
        return f"{vertical} Central Lung Region"
    return f"{vertical} {horizontal} Lung Region"


def measure_gradcam_region(
    heatmap: np.ndarray,
    image_shape: tuple[int, int],
    threshold: float = 0.55,
    pixel_spacing_mm: float | None = None,
) -> tuple[LocalizationMeasurements, np.ndarray]:
    height, width = image_shape
    resized_heatmap = cv2.resize(heatmap.astype(np.float32), (width, height))
    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

    adaptive_threshold = max(float(threshold), float(np.percentile(resized_heatmap, 88)))
    mask = (resized_heatmap >= adaptive_threshold).astype(np.uint8) * 255
    kernel_size = max(3, int(round(min(width, height) * 0.015)))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        measurements = LocalizationMeasurements(
            detected=False,
            bbox_xyxy=None,
            center_xy=None,
            location_label="No focal activation region",
            area_px=0,
            area_fraction=0.0,
            max_diameter_px=0.0,
            min_diameter_px=0.0,
            estimated_area_mm2=None,
            estimated_max_diameter_mm=None,
            heatmap_peak=float(resized_heatmap.max()),
            heatmap_mean_in_region=0.0,
        )
        return measurements, mask

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, box_width, box_height = cv2.boundingRect(largest_contour)
    x2 = x + box_width
    y2 = y + box_height
    area_px = int(cv2.contourArea(largest_contour))
    area_fraction = float(area_px / max(width * height, 1))
    center_x = int(x + box_width / 2)
    center_y = int(y + box_height / 2)
    max_diameter_px = float(max(box_width, box_height))
    min_diameter_px = float(min(box_width, box_height))

    region_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(region_mask, [largest_contour], -1, 255, thickness=-1)
    region_values = resized_heatmap[region_mask > 0]
    heatmap_mean = float(region_values.mean()) if len(region_values) else 0.0

    estimated_area_mm2 = None
    estimated_max_diameter_mm = None
    if pixel_spacing_mm is not None and pixel_spacing_mm > 0:
        estimated_area_mm2 = float(area_px * (pixel_spacing_mm**2))
        estimated_max_diameter_mm = float(max_diameter_px * pixel_spacing_mm)

    measurements = LocalizationMeasurements(
        detected=True,
        bbox_xyxy=(int(x), int(y), int(x2), int(y2)),
        center_xy=(center_x, center_y),
        location_label=_location_from_center(center_x, center_y, width, height),
        area_px=area_px,
        area_fraction=area_fraction,
        max_diameter_px=max_diameter_px,
        min_diameter_px=min_diameter_px,
        estimated_area_mm2=estimated_area_mm2,
        estimated_max_diameter_mm=estimated_max_diameter_mm,
        heatmap_peak=float(resized_heatmap.max()),
        heatmap_mean_in_region=heatmap_mean,
    )
    return measurements, region_mask


def make_overlay_and_bbox(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    measurements: LocalizationMeasurements,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    resized_heatmap = cv2.resize(heatmap.astype(np.float32), (width, height))
    heatmap_uint8 = (np.clip(resized_heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(image_rgb[:, :, ::-1], 0.58, heatmap_color, 0.42, 0)

    if measurements.bbox_xyxy is not None:
        x1, y1, x2, y2 = measurements.bbox_xyxy
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if measurements.center_xy is not None:
            cv2.circle(overlay_bgr, measurements.center_xy, 4, (0, 255, 255), -1)

    return overlay_bgr[:, :, ::-1]


def compute_risk_score(
    predicted_class: str,
    probabilities: dict[str, float],
    measurements: LocalizationMeasurements,
) -> tuple[int, str]:
    malignant_probability = probabilities.get("malignant", 0.0)
    benign_probability = probabilities.get("benign", 0.0)
    activation_component = min(measurements.area_fraction * 250.0, 1.0) if measurements.detected else 0.0

    if predicted_class == "normal":
        score = 100.0 * (0.75 * malignant_probability + 0.15 * benign_probability + 0.10 * activation_component)
    elif predicted_class == "benign":
        score = 100.0 * (0.45 * benign_probability + 0.35 * malignant_probability + 0.20 * activation_component)
    else:
        score = 100.0 * (0.75 * malignant_probability + 0.20 * activation_component + 0.05 * benign_probability)

    score_int = int(round(np.clip(score, 0, 100)))
    category = "Low" if score_int < 35 else "Moderate" if score_int < 70 else "High"
    return score_int, category
