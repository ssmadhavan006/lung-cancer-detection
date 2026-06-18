from __future__ import annotations

import torch


def integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_index: int,
    baseline: torch.Tensor | None = None,
    steps: int = 32,
) -> torch.Tensor:
    baseline = torch.zeros_like(inputs) if baseline is None else baseline
    scaled_inputs = [baseline + (float(step) / steps) * (inputs - baseline) for step in range(steps + 1)]
    gradients = []

    for scaled in scaled_inputs:
        scaled = scaled.clone().detach().requires_grad_(True)
        outputs = model(scaled)
        score = outputs[:, target_index].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        gradients.append(scaled.grad.detach())

    avg_gradients = torch.stack(gradients[:-1]).mean(dim=0)
    attributions = (inputs - baseline) * avg_gradients
    return attributions
