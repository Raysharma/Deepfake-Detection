import pytest
import numpy as np
from PIL import Image
import os
import io

from backend.pipeline.forensics.services.ela_service import generate_ela_analysis

@pytest.fixture
def sample_rgb_image():
    """Generates a 128x128 synthetic test image with contrasting colored shapes."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:64, :64] = [220, 50, 50]    # Red patch
    arr[64:, 64:] = [50, 220, 50]    # Green patch
    arr[:64, 64:] = [50, 50, 220]    # Blue patch
    arr[64:, :64] = [220, 220, 50]  # Yellow patch
    return Image.fromarray(arr, "RGB")

def test_ela_valid_image(sample_rgb_image):
    result = generate_ela_analysis(sample_rgb_image, prefix="test_valid")
    assert result["available"] is True
    assert result["error"] is None
    assert result["mean_error"] is not None
    assert result["max_error"] is not None
    assert result["std_error"] is not None
    assert result["high_error_ratio"] is not None
    assert result["image_url"] is not None
    assert result["quality"] == 90

def test_ela_png_input(sample_rgb_image):
    # Save as PNG bytes and reload
    buf = io.BytesIO()
    sample_rgb_image.save(buf, format="PNG")
    buf.seek(0)
    png_img = Image.open(buf)
    
    result = generate_ela_analysis(png_img, prefix="test_png")
    assert result["available"] is True
    assert result["mean_error"] >= 0.0

def test_ela_jpeg_input(sample_rgb_image):
    # Save as JPEG bytes and reload
    buf = io.BytesIO()
    sample_rgb_image.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    jpeg_img = Image.open(buf)

    result = generate_ela_analysis(jpeg_img, prefix="test_jpeg", quality=90)
    assert result["available"] is True
    assert result["mean_error"] >= 0.0

def test_ela_small_image():
    # 1x1 image should fail gracefully
    tiny_img = Image.fromarray(np.array([[[120, 120, 120]]], dtype=np.uint8), "RGB")
    result = generate_ela_analysis(tiny_img, prefix="test_tiny")
    assert result["available"] is False
    assert "too small" in result["error"].lower()
