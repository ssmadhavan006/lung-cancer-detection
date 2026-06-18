from __future__ import annotations

import warnings

import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = [
    "resnet50",
    "efficientnet_b0",
    "densenet121",
    "efficientnet_v2_s",
    "convnext_tiny",
    "vit_b_16",
]


WEIGHT_MAP = {
    "resnet50": models.ResNet50_Weights.DEFAULT,
    "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
    "densenet121": models.DenseNet121_Weights.DEFAULT,
    "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.DEFAULT,
    "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
    "vit_b_16": models.ViT_B_16_Weights.DEFAULT,
}


def _disable_inplace_relu(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model


def _safe_weights(model_name: str, use_pretrained: bool):
    if not use_pretrained:
        return None
    try:
        return WEIGHT_MAP[model_name]
    except Exception as error:
        warnings.warn(f"Falling back to random initialization for {model_name}: {error}")
        return None


def _build_model(model_name: str, weights):
    if model_name == "resnet50":
        return models.resnet50(weights=weights)

    if model_name == "efficientnet_b0":
        return models.efficientnet_b0(weights=weights)

    if model_name == "densenet121":
        return models.densenet121(weights=weights)

    if model_name == "efficientnet_v2_s":
        return models.efficientnet_v2_s(weights=weights)

    if model_name == "convnext_tiny":
        return models.convnext_tiny(weights=weights)

    if model_name == "vit_b_16":
        return models.vit_b_16(weights=weights)

    raise ValueError(f"Unsupported model: {model_name}")


def _replace_head(model_name: str, model: nn.Module, num_classes: int) -> nn.Module:
    if model_name == "resnet50":
        model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, num_classes))
        return model

    if model_name == "efficientnet_b0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if model_name == "efficientnet_v2_s":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "convnext_tiny":
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if model_name == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def create_model(model_name: str, num_classes: int, use_pretrained: bool) -> nn.Module:
    weights = _safe_weights(model_name, use_pretrained)
    try:
        model = _build_model(model_name, weights)
    except Exception as error:
        if use_pretrained:
            warnings.warn(
                f"Pretrained weights unavailable for {model_name}. Falling back to random initialization: {error}"
            )
            model = _build_model(model_name, None)
        else:
            raise
    model = _disable_inplace_relu(model)
    return _replace_head(model_name, model, num_classes)
