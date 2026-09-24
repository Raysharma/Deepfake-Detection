import pytest
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import io

from backend.app import app

client = TestClient(app)

@pytest.fixture
def valid_jpeg_bytes():
    arr = np.random.randint(60, 220, (120, 120, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def test_api_forensic_health():
    res = client.get("/api/forensics/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"
    assert "submodules" in data

def test_api_forensic_valid_upload(valid_jpeg_bytes):
    files = {"file": ("test_forensic.jpg", valid_jpeg_bytes, "image/jpeg")}
    res = client.post("/api/forensics/analyze", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "prediction" in data
    assert "ela" in data
    assert "fft" in data
    assert "gradcam" in data
    assert data["ela"]["available"] is True
    assert data["fft"]["available"] is True

def test_api_forensic_empty_file():
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res = client.post("/api/forensics/analyze", files=files)
    assert res.status_code in [400, 422]

def test_api_forensic_invalid_file_type():
    files = {"file": ("bad.txt", b"plain text data", "text/plain")}
    res = client.post("/api/forensics/analyze", files=files)
    assert res.status_code == 400
