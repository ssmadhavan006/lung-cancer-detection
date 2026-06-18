from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def enhance_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
        rgb_input = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb_input = image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    sharpened = cv2.addWeighted(normalized, 1.25, cv2.GaussianBlur(normalized, (0, 0), 3), -0.25, 0)
    rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    if rgb_input.shape == rgb.shape:
        return rgb
    return cv2.resize(rgb, (rgb_input.shape[1], rgb_input.shape[0]))


class MedicalPreprocess(A.ImageOnlyTransform):
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return enhance_image(img)


def build_train_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            MedicalPreprocess(p=1.0),
            A.Resize(image_size, image_size),
            A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                scale=(0.9, 1.1),
                rotate=(-5, 5),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.ElasticTransform(alpha=20, sigma=6, border_mode=cv2.BORDER_REFLECT_101, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            MedicalPreprocess(p=1.0),
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
