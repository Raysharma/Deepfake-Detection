import pytest
import numpy as np
from PIL import Image

from backend.forensics.services.model_service import (
    predict_deepfake,
    get_or_load_model,
    get_target_convolutional_layer,
    HAS_TORCH
)

@pytest.fixture
def sample_face_image():
    # Simple RGB face-like image
    arr = np.full((224, 224, 3), 180, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for model test")
def test_model_loading_and_inference(sample_face_image):
    model = get_or_load_model()
    assert model is not None

    pred = predict_deepfake(sample_face_image)
    assert pred["success"] is True
    assert pred["label"] in ["real", "deepfake"]
    assert 0.0 <= pred["confidence"] <= 1.0
    assert "real" in pred["class_probabilities"]
    assert "deepfake" in pred["class_probabilities"]

    # Sum of probabilities should be approx 1.0
    total_prob = pred["class_probabilities"]["real"] + pred["class_probabilities"]["deepfake"]
    assert pytest.approx(total_prob, 0.01) == 1.0

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for model test")
def test_target_layer_discovery():
    model = get_or_load_model()
    layer, name = get_target_convolutional_layer(model, "layer4")
    assert layer is not None
    assert "layer4" in name

    # Test invalid layer name
    bad_layer, bad_name = get_target_convolutional_layer(model, "nonexistent_submodule_layer")
    # Should fall back or fail gracefully
    assert bad_layer is not None or "Could not find" in str(bad_name)
