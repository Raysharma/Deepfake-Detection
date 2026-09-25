import os
import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.zero_grad()

        output = self.model(input_tensor)

        score = output[:, 0].sum()
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze(0).cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (
                cam.max() - cam.min()
            )

        return cam


def save_gradcam(
    model,
    input_tensor,
    original_spectrogram,
    output_path
):
    target_layer = model.features[-1]

    gradcam = GradCAM(model, target_layer)

    cam = gradcam.generate(input_tensor)

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    plt.figure(figsize=(10, 4))

    plt.imshow(
        original_spectrogram,
        aspect="auto",
        origin="lower"
    )

    plt.imshow(
        cam,
        aspect="auto",
        origin="lower",
        alpha=0.45
    )

    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=150
    )

    plt.close()

    return output_path