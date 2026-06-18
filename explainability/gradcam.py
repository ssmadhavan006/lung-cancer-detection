from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _resolve_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        features = model.features
        if isinstance(features, torch.nn.Sequential):
            return features[-1]
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    raise ValueError("Could not infer a Grad-CAM target layer for this model.")


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or _resolve_target_layer(model)
        self.activations = None
        self.gradients = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, inputs: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        outputs = self.model(inputs)
        if class_index is None:
            class_index = int(outputs.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        score = outputs[:, class_index].sum()
        score.backward(retain_graph=True)

        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted = pooled_gradients * self.activations
        heatmap = weighted.sum(dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (inputs.shape[-1], inputs.shape[-2]))
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap


def _tensor_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = np.clip((image * std) + mean, 0, 1)
    return (image * 255).astype(np.uint8)


def create_explainability_artifacts(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    class_names: list[str],
    output_dir: Path,
    limit: int = 6,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grad_cam = GradCAM(model)
    model.eval()
    count = 0
    with torch.enable_grad():
        for images, _, paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            for index in range(images.size(0)):
                if count >= limit:
                    grad_cam.remove_hooks()
                    return
                image = images[index : index + 1]
                heatmap = grad_cam.generate(image, int(predictions[index].item()))
                original = _tensor_to_uint8(images[index])
                heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(original[:, :, ::-1], 0.55, heatmap_color, 0.45, 0)

                figure, axes = plt.subplots(1, 3, figsize=(10, 4))
                axes[0].imshow(original)
                axes[0].set_title("Original")
                axes[1].imshow(heatmap, cmap="jet")
                axes[1].set_title("Grad-CAM")
                axes[2].imshow(overlay[:, :, ::-1])
                label = class_names[int(predictions[index].item())]
                score = float(probabilities[index, predictions[index]].item())
                axes[2].set_title(f"{label}: {score:.2%}")
                for axis in axes:
                    axis.axis("off")
                figure.tight_layout()
                image_name = Path(paths[index]).stem
                figure.savefig(output_dir / f"{count:02d}_{image_name}_gradcam.png", dpi=200, bbox_inches="tight")
                plt.close(figure)
                count += 1
    grad_cam.remove_hooks()
