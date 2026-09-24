import pytest
import numpy as np
from PIL import Image

from backend.forensics.services.gradcam_service import generate_gradcam_analysis
from backend.forensics.services.model_service import HAS_TORCH

@pytest.fixture
def sample_test_image():
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for Grad-CAM test")
def test_gradcam_valid_layer(sample_test_image):
    result = generate_gradcam_analysis(sample_test_image, prefix="test_gc_valid", target_class_name="deepfake", layer_name="layer4")
    assert result["available"] is True
    assert result["error"] is None
    assert result["heatmap_url"] is not None
    assert result["overlay_url"] is not None
    assert result["target_class"] == "deepfake"

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for Grad-CAM test")
def test_gradcam_invalid_layer(sample_test_image):
    result = generate_gradcam_analysis(sample_test_image, prefix="test_gc_bad", layer_name="completely_invalid_layer_xyz")
    # Must report error cleanly without crashing or faking output
    if not result["available"]:
        assert result["error"] is not None
        assert "target convolutional layer" in result["error"].lower() or "could not" in result["error"].lower()
