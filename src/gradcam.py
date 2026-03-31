from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, target_class].sum()
        score.backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    image = image.convert("RGB").resize((224, 224))
    image_np = np.array(image).astype(np.float32)

    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((224, 224))
    heatmap_np = np.array(heatmap_img).astype(np.float32)

    colored = np.zeros((224, 224, 3), dtype=np.float32)
    colored[:, :, 0] = heatmap_np
    colored[:, :, 1] = 0
    colored[:, :, 2] = 0

    blended = image_np * (1 - alpha) + colored * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)