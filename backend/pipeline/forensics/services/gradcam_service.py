import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from backend.pipeline.forensics.utils.file_utils import get_artifact_paths
from backend.pipeline.forensics.utils.image_utils import (
    prepare_model_tensor,
    pil_to_cv2,
    save_image_safely
)
from backend.pipeline.forensics.services.model_service import (
    get_or_load_model,
    get_target_convolutional_layer,
    CLASSES
)

class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    for deep convolutional neural networks.
    Hooks into activations and backpropagated gradients of the specified target layer.
    """
    def __init__(self, model: Any, target_layer: Any):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle = None
        self.backward_handle = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple where element 0 is the gradient w.r.t layer output
            self.gradients = grad_out[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        if self.forward_handle:
            self.forward_handle.remove()
        if self.backward_handle:
            self.backward_handle.remove()

    def generate(self, input_tensor: "torch.Tensor", class_idx: int) -> np.ndarray:
        """
        Executes forward and backward passes to compute spatial activation weights.
        Returns a normalized 2D heatmap in range [0.0, 1.0].
        """
        self.model.eval()
        # Enable gradient tracking for backpropagation
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        logits = self.model(input_tensor)
        
        # Target score for specified class
        score = logits[:, class_idx]
        
        # Zero gradients and backpropagate
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations from target layer.")

        # Compute importance weights alpha_k via Global Average Pooling of gradients
        # Shape: (batch_size, channels, H, W) -> weights: (batch_size, channels, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Linear combination of weighted activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU to focus on features having positive contribution to target class
        cam = F.relu(cam)
        
        # Normalize to [0.0, 1.0]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

def generate_gradcam_analysis(
    pil_image: Image.Image,
    prefix: str,
    target_class_name: str = "deepfake",
    layer_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates genuine Grad-CAM visual heatmaps using the active deep learning model.
    Produces:
    1. Raw JET colormap heatmap resized to original image dimensions.
    2. Blended overlay visualization (60% original image + 40% heatmap).
    3. Persists both artifacts to outputs/forensic/.
    
    IMPORTANT:
    Grad-CAM explains spatial regions contributing to the neural network classification.
    It is an interpretability method, NOT an independent deepfake detector.
    """
    model = get_or_load_model()
    if model is None or not HAS_TORCH:
        return {
            "available": False,
            "target_class": target_class_name,
            "target_layer": None,
            "heatmap_url": None,
            "overlay_url": None,
            "description": "Grad-CAM could not be generated.",
            "error": "PyTorch deepfake model is unavailable."
        }

    target_layer_module, resolved_layer_name = get_target_convolutional_layer(model, layer_name)
    if target_layer_module is None:
        return {
            "available": False,
            "target_class": target_class_name,
            "target_layer": None,
            "heatmap_url": None,
            "overlay_url": None,
            "description": "Grad-CAM could not be generated.",
            "error": f"Target convolutional layer could not be located ({resolved_layer_name})."
        }

    gradcam_engine = None
    try:
        class_idx = 1 if target_class_name == "deepfake" else 0
        input_tensor = prepare_model_tensor(pil_image)

        gradcam_engine = GradCAM(model, target_layer_module)
        cam_2d = gradcam_engine.generate(input_tensor, class_idx)

        # Original dimensions
        orig_w, orig_h = pil_image.size
        # Resize activation map to original image resolution using bicubic interpolation
        cam_resized = cv2.resize(cam_2d, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)
        cam_uint8 = (cam_resized * 255.0).astype(np.uint8)

        # Generate JET color heatmap
        heatmap_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        # Convert original image to OpenCV BGR for overlay blending
        orig_bgr = pil_to_cv2(pil_image)
        overlay = cv2.addWeighted(orig_bgr, 0.60, heatmap_color, 0.40, 0)

        # Persist raw heatmap
        raw_abs_path, raw_url_path = get_artifact_paths(prefix, "gradcam", ext=".jpg")
        save_image_safely(heatmap_color, raw_abs_path, quality=95)

        # Persist overlay
        overlay_abs_path, overlay_url_path = get_artifact_paths(prefix, "gradcam_overlay", ext=".jpg")
        save_image_safely(overlay, overlay_abs_path, quality=95)

        return {
            "available": True,
            "target_class": target_class_name,
            "target_layer": resolved_layer_name,
            "heatmap_url": raw_url_path,
            "overlay_url": overlay_url_path,
            "description": (
                f"Grad-CAM visual explanation generated for class '{target_class_name}' "
                f"via layer '{resolved_layer_name}'. Highlights spatial regions that contributed to model classification."
            ),
            "error": None
        }

    except Exception as e:
        return {
            "available": False,
            "target_class": target_class_name,
            "target_layer": resolved_layer_name,
            "heatmap_url": None,
            "overlay_url": None,
            "description": "Grad-CAM generation encountered an unexpected error.",
            "error": f"Grad-CAM error: {str(e)}"
        }
    finally:
        if gradcam_engine is not None:
            gradcam_engine.remove_hooks()
