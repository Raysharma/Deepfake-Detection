import pytest
import numpy as np
from PIL import Image

from backend.pipeline.forensics.services.fft_service import generate_fft_analysis

@pytest.fixture
def sample_fft_image():
    # 128x128 image with sinusoidal frequency patterns
    x = np.linspace(0, 10 * np.pi, 128)
    y = np.linspace(0, 10 * np.pi, 128)
    xx, yy = np.meshgrid(x, y)
    pattern = (np.sin(xx) + np.cos(yy) + 2.0) * 60.0
    arr = np.clip(pattern, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L").convert("RGB")

def test_fft_valid_image(sample_fft_image):
    result = generate_fft_analysis(sample_fft_image, prefix="test_fft_valid")
    assert result["available"] is True
    assert result["error"] is None
    assert result["mean_magnitude"] > 0.0
    assert result["max_magnitude"] > 0.0
    assert result["std_magnitude"] >= 0.0
    assert result["high_frequency_ratio"] > 0.0
    assert result["image_url"] is not None

def test_fft_tiny_image():
    tiny_img = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), "RGB")
    result = generate_fft_analysis(tiny_img, prefix="test_fft_tiny")
    assert result["available"] is False
    assert "too small" in result["error"].lower()
